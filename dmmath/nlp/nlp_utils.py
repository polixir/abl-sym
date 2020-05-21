import re
from typing import List, Union, Dict, Iterable
from enum import Enum
from collections import OrderedDict
import torch

from allennlp.training.optimizers import Optimizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data import Tokenizer, Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import TextField, MetadataField
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.common import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
import time
import glob
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import random
import torch.distributed as dist


@TokenIndexer.register("math_single_id")
class DMMathSingleIdTokenIndexer(SingleIdTokenIndexer):
    EXPER_SYMBOL = '@@MATH_EXPER@@'

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.is_expr == True:
            new_token = token.replace(text=DMMathSingleIdTokenIndexer.EXPER_SYMBOL)
        else:
            new_token = token
        super(DMMathSingleIdTokenIndexer, self).count_vocab_items(new_token, counter)


class MathToken(Token):
    added_key = 'is_expr'

    def __new__(cls, *args, **kwargs):
        base_kwargs = {k: kwargs[k] for k in Token._fields if k in kwargs}
        self = super(MathToken, cls).__new__(cls, *args, **base_kwargs)
        extra_v = kwargs.get(MathToken.added_key, False)
        setattr(self, MathToken.added_key, extra_v)
        return self

    def replace(self, **kwargs):
        self = self._replace(**kwargs)
        return self


@Tokenizer.register('dm_math')
class DMMathTokenizer(Tokenizer):
    def __init__(self) -> None:
        self.pattern = re.compile("#MATH\[(\w+)]([^#]*)#")
        self.comma_pattern = re.compile('\([^\)]+,[^\(]+\)')
        nths = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
                'square', 'cube']

        self._vocab = {'composite', 'simplify', 'total', 'what', 'rounded', 'common', 'how', 'meters', 'integer',
                       'least', 'do',
                       'greatest', 'are', 'not', 'or', 'difference', 'between', 'rearrange', 'base', 'root', 'for',
                       'prob',
                       'millennia', 'grams', 'collect', 'minutes', 'greater', 'expand', 'bigger', 'be', 'such',
                       'values', 'remainder',
                       'distance', 'of', 'that', 'assuming', 'even', 'tonnes', 'calculate', 'at', 'non-equal', 'power',
                       'have',
                       'decades', 'smaller', 'weeks', 'as', 'digit', 'kilometers', 'evaluate', 'unequal', 'than',
                       'litres', 'when',
                       'list', 'nanograms', 'nanometers', 'add', 'sum', 'smallest', 'suppose', 'factors', 'nearest',
                       'put',
                       'derivative',
                       'nonequal', 'find', 'take', 'micrograms', 'terms', 'form', 'with', 'less', 'centimeters',
                       'millilitres', 'seconds', 'nanoseconds',
                       'number', 'big', 'prime', 'wrt', 'round', 'after', 'given', 'micrometers', 'microseconds', 'so',
                       'positive',
                       'years', 'to',
                       'divisor', 'product', 'express', 'divide', 'in', 'differentiate', 'out', 'times', 'highest',
                       'away', 'plus',
                       'different', 'before', 'solve', 'cb', 'let', 'multiply', 'equal', 'convert', 'many', 'subtract',
                       'days', 'millimeters', 'milligrams', 'kilograms',
                       'months', 'from', 'milliseconds', 'give', 'denominator', 'divided', 'determine', 'minus',
                       'factor', 'is',
                       'work', 'hours', 'respect', 'lowest', 'multiple', 'there', 'by', 'and', 'which', 'the', 'does',
                       'value',
                       'same', 'most', 'together'} - set(nths)

    def _isalpha(self, word):
        if word.lower() in self._vocab:
            return True
        else:
            return False

    def _add_token_to_mathtokens(self, mathtokens, token, is_expr):
        if len(mathtokens) > 0 and mathtokens[-1].is_expr and is_expr is True:
            mathtokens[-1] = MathToken(mathtokens[-1].text + ' ' + token, is_expr=True)
        else:
            mathtokens.append(MathToken(token, is_expr=is_expr))

    def _do_with_paren(self, text):
        def f(matched_group):
            x = matched_group.group('v')
            if len(set([y.lower() for y in x.split()]) & self._vocab) > 0:
                return f'( {x} )'
            else:
                return f'({x})'

        text = re.sub('\((?P<v>[^\)\(]*)\)', f, text)
        return text

    def tokenize(self, text: str) -> List[Token]:
        tokens = []
        text = self._do_with_paren(text)

        sent_lst = re.split('(, |\. |\?|; )', text + ' ')
        for sent in sent_lst:
            part_tokens = sent.strip().split()
            for idx, token in enumerate(part_tokens):
                if not token:
                    continue
                if len(token) == 1:
                    if token == 'a' and idx < len(part_tokens) - 1 and part_tokens[idx + 1] in ['prime',
                                                                                                'composite',
                                                                                                'multiple',
                                                                                                'factor']:
                        self._add_token_to_mathtokens(tokens, token, False)
                    elif token in [',', '.', '?', ';', ':', ')', '(']:
                        self._add_token_to_mathtokens(tokens, token, False)
                    else:
                        self._add_token_to_mathtokens(tokens, token, True)
                elif self._isalpha(token):
                    self._add_token_to_mathtokens(tokens, token, False)
                else:
                    if token[-1] in [',', '.', '?', ';', ':']:
                        if self._isalpha(token[:-1]):
                            self._add_token_to_mathtokens(tokens, token[:-1], False)
                        else:
                            self._add_token_to_mathtokens(tokens, token[:-1], True)
                        self._add_token_to_mathtokens(tokens, token[-1], False)
                    elif token[0] in [',', '.', '?', ';', ':']:
                        self._add_token_to_mathtokens(tokens, token[0], False)
                        if self._isalpha(token[1:]):
                            self._add_token_to_mathtokens(tokens, token[1:], False)
                        else:
                            self._add_token_to_mathtokens(tokens, token[1:], True)
                    else:
                        self._add_token_to_mathtokens(tokens, token, True)
        words = tokens
        words = list(filter(lambda x: x.text.strip(), words))
        return words

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]


@DatasetReader.register('dm_math')
class DMMathDatasetReader(DatasetReader):
    def __init__(self,
                 with_expr_anno: bool = False,
                 with_program_anno: bool = False,
                 target_key: str = 'ans',
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Dict[str, Tokenizer] = None,
                 lazy: bool = False) -> None:

        super(DMMathDatasetReader, self).__init__(lazy)
        self._with_expr_anno = with_expr_anno
        self._with_program_anno = with_program_anno
        self._target_key = target_key
        self._tokenizer = tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, filepath: str):
        paths = glob.glob(filepath)
        if all([path.endswith('.npy') or path.endswith('.npz') for path in paths]):
            Q_list = []
            A_list = []
            P_list = []
            t1 = time.time()
            for path in paths:
                if path.endswith('.npy'):
                    data = np.load(path).item()
                elif path.endswith('.npz'):
                    data = np.load(path)
                Q_list.append(data["Q"])
                A_list.append(data["A"])
                P_list.append(data["P"])
            Q = np.vstack(Q_list)
            A = np.vstack(A_list)
            P = np.vstack(P_list)
            data = {"Q": Q, "A": A, "P": P}
            t2 = time.time()
            print('loading data time: ', t2 - t1)
            yield data
        else:
            for path in paths:
                with open(path, 'r') as f:
                    lines = f.readlines()
                for idx, line in enumerate(lines):
                    if idx % 2 == 0:
                        src = line.strip()
                    else:
                        tgt = line.strip()
                        instance = self.text_to_instance(src, tgt)
                        yield instance

    def text_to_instance(self,
                         source_sentence: str,
                         target_sentence: str = None) -> Instance:
        source_tokenized = self._tokenizer['ques'].tokenize(source_sentence)
        fields = {'source_tokens': TextField(source_tokenized, self._source_token_indexers)}
        if target_sentence:
            target_list = target_sentence.split('###')
            if len(target_list) == 1:
                if target_sentence:
                    if self._target_key == 'ans':
                        target_tokenized = list(map(Token, target_sentence))
                    else:
                        target_tokenized = [Token(x) for x in target_sentence.split()]
                else:
                    target_tokenized = None
                if target_tokenized is not None:
                    fields['target_tokens'] = TextField(target_tokenized,
                                                        {'tokens': self._target_token_indexers['tokens']})
            else:
                assert len(target_list) == 2
                answer, program = [x.strip() for x in target_list]
                answer_tokenized = [Token(x) for x in answer]
                fields["answer"] = TextField(answer_tokenized, {'answer': self._target_token_indexers['answer']})
                program_tokenized = [Token(x) for x in program.split()]
                fields["program"] = TextField(program_tokenized, {'program': self._target_token_indexers['program']})

        if isinstance(self._tokenizer['ques'], DMMathTokenizer):
            avail_pos = [f'position#{idx}' for idx, x in enumerate(source_tokenized) if x.is_expr]
            fields['meta_data'] = MetadataField({"avail_pos": avail_pos})
        return_instance = Instance(fields)
        return return_instance


class DMDataSet():
    def __init__(self, data, split_num=20, batch_size=80, num_gpus=8, shuffle=True, distributed=False,
                 data_slice=False):
        self.data = data
        self.sz = self.data["Q"].shape[0]
        self.split_num = split_num
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.shuffle = shuffle
        self.distributed = distributed

        if self.distributed and dist.is_available():
            self.num_replicas = self.num_gpus
            self.rank = dist.get_rank()
        self.epoch = 0
        self.data_slice = data_slice

    def __len__(self):
        return self.sz

    def shuffle_and_sort(self):
        t1 = time.time()
        I = np.arange(self.sz)
        A = np.hstack((self.data["Q"][:, -1][:, np.newaxis], I[:, np.newaxis]))
        np.random.shuffle(A)
        d = self.sz // self.split_num
        for pidx in range(self.split_num + 1):
            s = pidx * d
            e = (pidx + 1) * d
            if s == e:
                continue
            I = np.argsort(A[s:e, 0])
            A[s:e] = A[s:e][I]
        I = A[:, 1]
        t2 = time.time()
        # print('shuffle and sort data time: ', t2 - t1)
        return I

    def __iter__(self):
        np.random.seed(self.epoch)
        delta = self.batch_size * self.num_gpus
        if self.shuffle:
            I = self.shuffle_and_sort()
        else:
            I = np.arange(self.sz)
        if self.distributed:
            if self.data_slice:
                I = I[np.arange(self.rank, I.shape[0], self.num_replicas)]
            I = np.array_split(I, np.ceil(I.shape[0] / self.batch_size))
        else:
            I = np.array_split(I, np.ceil(I.shape[0] / delta))
        if self.shuffle:
            np.random.shuffle(I)

        for batch_idxes in I:
            if self.distributed:
                src = self.data["Q"][batch_idxes, :max(self.data["Q"][batch_idxes, -1])]
                ans = self.data["A"][batch_idxes, :max(self.data["A"][batch_idxes, -1])]
                prog = self.data["P"][batch_idxes, :max(self.data["P"][batch_idxes, -1])]
                batch_data = {
                    "source_tokens": {
                        "tokens": torch.tensor(src).long()
                    },
                    "target_tokens": {
                        "answer": torch.tensor(ans).long(),
                        "program": torch.tensor(prog).long(),
                    }
                }
                yield batch_data
            else:
                batch_group = []
                gpu_I = np.array_split(batch_idxes, self.num_gpus)
                for gpu_idxes in gpu_I:
                    if gpu_idxes.shape[0] == 0:
                        continue
                    src = self.data["Q"][gpu_idxes, :max(self.data["Q"][gpu_idxes, -1])]
                    ans = self.data["A"][gpu_idxes, :max(self.data["A"][gpu_idxes, -1])]
                    prog = self.data["P"][gpu_idxes, :max(self.data["P"][gpu_idxes, -1])]
                    batch_group.append({
                        "source_tokens": {
                            "tokens": torch.tensor(src).long()
                        },
                        "target_tokens": {
                            "answer": torch.tensor(ans).long(),
                            "program": torch.tensor(prog).long(),
                        }
                    })
                yield batch_group

    def set_epoch(self, epoch):
        self.epoch = epoch


def setup_datasets(params: Params) -> Dict[str, Iterable[Instance]]:
    dataset_reader_params = params.get('dataset_reader')
    validation_dataset_reader_params = params.get('validation_dataset_reader', None)
    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    train_data_path = params.get('train_data_path')
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.get('validation_data_path', None)
    if validation_data_path is not None:
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.get("test_data_path", None)
    if test_data_path is not None:
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data
    return datasets


def setup_vocab(params):
    all_datasets = setup_datasets(params)
    vocab: Vocabulary = Vocabulary.from_params(
        params.get("vocabulary", {}),
        (instance for key, dataset in all_datasets.items() for instance in dataset)
    )
    return vocab


def add_env_tokens_to_vocab(vocab: Vocabulary, actions: List[Union[str, int]] = None, stack_states: Enum = None,
                            exec_states: Enum = None):
    if actions:
        actions = map(lambda x: str(x), actions)
    else:
        actions = []
    if stack_states:
        stack_states = map(lambda x: x.name, stack_states)
    else:
        stack_states = []
    if exec_states:
        exec_states = map(lambda x: x.name, exec_states)
    else:
        exec_states = []
    extra_vob_counter = {
        'stack': OrderedDict({state: 1 for state in stack_states}),
        'exec': OrderedDict({state: 1 for state in exec_states}),
        'action': OrderedDict({action: 1 for action in actions})
    }
    vocab._extend(extra_vob_counter, non_padded_namespaces=['stack', 'exec', 'action'])
    return vocab


def setup_model(params, vocab):
    model = Model.from_params(vocab=vocab, params=params.pop('model'))
    model.extend_embedder_vocab()
    return model


@LearningRateScheduler.register("linear")
class LinearLR(LearningRateScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 last_epoch: int = -1) -> None:
        self.num_epochs = num_epochs
        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_values(self):
        if self.last_epoch == -1:
            return self.base_values

        epoch = self.last_epoch
        lrs = [base_lr - (base_lr * (epoch / float(self.num_epochs))) for base_lr in self.base_values]
        return lrs


def setup_optimizer(params, model):
    parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = Optimizer.from_params(parameters, params)
    return optimizer

