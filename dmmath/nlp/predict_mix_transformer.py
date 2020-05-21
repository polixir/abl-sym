from dmmath.utils import ConstParams
from dmmath.nlp.nlp_utils import DMMathDatasetReader
from dmmath.nlp.tp_transformer import TPTransformer
from dmmath.nlp.mix_transformer import MixTransformer
from allennlp.predictors.predictor import Predictor
from allennlp.nn import util
import torch
import sys
import os
import argparse
from tqdm import tqdm
import logging
from dmmath.nlp.nlp_utils import DMDataSet
import numpy as np

torch.manual_seed(1)


@Predictor.register('my_seq2seq')
class MySeq2SeqPredictor(Predictor):
    '''
    predict answers and programs when given a file to be predicted.
    '''

    def _read(self, txt_file):
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        QUES_MAX_LEN = 160 + 2
        ANS_MAX_LEN = 30 + 3
        PROG_MAX_LEN = 30 + 3
        N = len(lines) // 2
        Q = np.zeros((N, QUES_MAX_LEN + 1), dtype=np.ubyte)
        A = np.zeros((N, ANS_MAX_LEN + 1), dtype=np.ubyte)
        P = np.zeros((N, PROG_MAX_LEN + 1), dtype=np.ubyte)
        get_src_idx = lambda x: self._model.vocab.get_token_to_index_vocabulary("source_tokens")[x]
        get_dst_idx = {ns: self._model.vocab.get_token_to_index_vocabulary(ns) for ns in ['answer', 'program']}

        for idx in tqdm(range(len(lines) // 2)):
            ques = lines[2 * idx].strip()
            ans, prog = lines[2 * idx + 1].strip().split('###')
            prog = prog.split()
            assert len(ques) <= QUES_MAX_LEN
            assert len(ans) <= ANS_MAX_LEN
            assert len(prog) <= PROG_MAX_LEN
            Q[idx][:len(ques)] = list(map(get_src_idx, ques))
            Q[idx][-1] = len(ques)
            A[idx][:len(ans) + 2] = [get_dst_idx["answer"]["@start@"]] + [get_dst_idx["answer"][x] for x in ans] + [
                get_dst_idx["answer"]["@end@"]]
            A[idx][-1] = len(ans) + 2
            P[idx][:len(prog) + 2] = [get_dst_idx["program"]["@start@"]] + [get_dst_idx['program'][x] for x in prog] + [
                get_dst_idx["program"]["@end@"]]
            P[idx][-1] = len(prog) + 2

        data = {"Q": Q, "A": A, "P": P}
        return data

    def predict_on_file(self, filepath, args):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        inputs = []
        for idx in range(len(lines) // 2):
            ques = lines[2 * idx].strip()
            lst = lines[2 * idx + 1].strip().split('###')
            if len(lst) == 1:
                ans = lst[0]
                prog = "extra#None"
            elif len(lst) == 2:
                ans, prog = lst
            inputs.append((ques, (ans, prog)))

        data = self._read(filepath)
        dataset = DMDataSet(data, batch_size=args.batch_size, num_gpus=1, shuffle=False, distributed=False,
                            data_slice=False)
        outputs = []
        for batch_data in tqdm(dataset, total=np.ceil(len(dataset) / args.batch_size)):
            batch_data = batch_data[0]
            model_input = util.move_to_device(batch_data, args.cuda)
            batch_outputs = self._model.decode(self._model.forward(**model_input))
            for idx in range(len(batch_outputs["answer"])):
                ans = ''.join(batch_outputs["answer"][idx])
                if 'program' in batch_outputs:
                    prog = ' '.join(batch_outputs["program"][idx])
                else:
                    prog = "extra#None"
                outputs.append([ans, prog])
        return inputs, outputs


parser = argparse.ArgumentParser(description='PyTorch DeepMind mathdataset Training')
parser.add_argument('--model', type=str, help='model(.tar.gz) path')
parser.add_argument('--data', type=str, help='data path for prediction')
parser.add_argument('--result_dir', type=str, help='prediction result directory')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--target_limit_decode_steps', type=bool, default=False,
                    help='whether to set the maximum decoding steps')
parser.add_argument('--cuda', type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='  %(message)s')
    os.makedirs(args.result_dir, exist_ok=True)
    result_path = args.result_dir + f'/{args.data.split("/")[-1]}.predict'

    predictor = MySeq2SeqPredictor.from_path(args.model, predictor_name='my_seq2seq',
                                             cuda_device=args.cuda)
    predictor._model.target_limit_decode_steps = args.target_limit_decode_steps

    inputs, outputs = predictor.predict_on_file(args.data, args)
    assert len(inputs) == len(outputs)
    for i in range(len(outputs)):
        outputs[i][0] = outputs[i][0][:len(inputs[i][0])]
    with open(result_path, 'w') as f:
        for input, output in zip(inputs, outputs):
            f.write(input[0] + '\n')
            f.write('###'.join(input[1]) + '\n')
            f.write('###'.join(output) + '\n')
