from dmmath.utils import ConstParams
from dmmath.nlp.nlp_utils import DMMathDatasetReader
from dmmath.nlp.mix_transformer import MixTransformer
from allennlp.commands.train import train_model
from dmmath.nlp.pt_trainner import PtTrainer
import argparse

import torch
import sys
import os

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True


def count_parameters(model):
    params_cnt = sum(p.numel() for p in model.parameters())
    params_trainable_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_cnt, params_trainable_cnt


parser = argparse.ArgumentParser(description='PyTorch DeepMind math dataset Training')
parser.add_argument('--workdir', type=str,
                    help='directory for serialization, and experiment.config must be in this directory, it is used for training configuration')

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.workdir):
        os.mkdir(args.workdir)

    serialization_dir = f'{args.workdir}/pt'
    param_fil = f'{args.workdir}/experiment.jsonnet'

    params = ConstParams.from_file(param_fil)

    if os.path.exists(serialization_dir):
        recover = True
    else:
        recover = False
    model = train_model(params, serialization_dir, recover=recover)

    params_cnt, params_trainable_cnt = count_parameters(model)
    print("all params cnt: ", params_cnt)
    print("all trainable params cnt: ", params_trainable_cnt)
