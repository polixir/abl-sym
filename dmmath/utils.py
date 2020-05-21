from enum import Enum
from collections import namedtuple
from allennlp.common import Params
from copy import deepcopy
from tqdm import tqdm

_Problem = namedtuple('Problem', ['text', 'question', 'answer', 'program'])
Rewards = namedtuple('Rewards', ['ans_right', 'run_success', 'run_fail'])
Observation = namedtuple('Observation', ['problem', 'avail_actions', 'stack_state', 'exec_state'])

OP_START_SYMBOL = ("extra", "@start@")
OP_END_SYMBOL = ("extra", "@end@")
OP_PADDING_SYMBOL = ("extra", "padding")


class Problem(_Problem):
    def parse_program(self):
        program = self.program
        pos = lambda pos: [('pos', str(pos))]
        api = lambda argc, fn: [('argc', str(argc)), ('api', str(fn))]
        parse_pos_seq = lambda pos: pos(pos) + api(1, 'parse_expr')
        eval_pos_seq = lambda pos: pos(pos) + api(1, 'eval')
        assert isinstance(program, str)
        if program == 'extra#None':
            return None
        else:
            program_seq = []
            for x in program.split():
                api, name = x.split('#')
                if api == 'pos':
                    api = 'position'
                program_seq.append((api, name))

            return program_seq


class RunStatus(Enum):
    Fail = 0
    Success = 1
    Right = 2


class StackStatus(Enum):
    Len0 = 0
    Len1 = 1
    Len2 = 2
    Len3 = 3


class ExecStatus(Enum):
    Empty = 0
    NotEmpty = 1


class ConstParams(Params):
    @staticmethod
    def from_file(params_file, *args, **kwargs):
        param = Params.from_file(params_file, *args, **kwargs)
        param.__class__ = ConstParams
        return param

    def get(self, key: str, *args, **kwargs):
        obj = super(ConstParams, self).get(key, *args, **kwargs)
        if isinstance(obj, Params):
            obj.__class__ = ConstParams
        return deepcopy(obj)

    def __getattr__(self, item):
        attrs = list(ConstParams.__dict__.keys()) + list(self.__dict__.keys())
        if item in attrs or item.startswith('__'):
            return super(ConstParams, self).__getattr__(item)
        else:
            return self.get(item)


def parse_dataset(filepath):
    from dmmath.nlp.nlp_utils import DMMathTokenizer
    tokenizer = DMMathTokenizer()
    with open(filepath, 'r') as f:
        lines = f.readlines()
    problems = []
    for idx in tqdm(range(len(lines) // 2), 'parsing data...'):
        text = lines[2 * idx].strip()
        if text.startswith('%'):
            continue
        question = tokenizer.tokenize(text)
        anno = lines[2 * idx + 1].strip()
        if '###' in anno:
            answer, program = [x.strip() for x in anno.split('###')]
        else:
            answer, program = anno, None

        problem = Problem(text, question, answer, program)
        problems.append(problem)
    return problems
