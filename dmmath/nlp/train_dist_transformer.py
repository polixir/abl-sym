from allennlp.common.checks import check_for_gpu
from allennlp.common.util import prepare_environment, prepare_global_logging, cleanup_global_logging, dump_metrics
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.training.trainer_base import TrainerBase

import torch
import torch.distributed as dist
import argparse
import logging
import os

from dmmath.utils import ConstParams
from dmmath.nlp.nlp_utils import DMMathDatasetReader
from dmmath.nlp.transformer import MyTransformer
from dmmath.nlp.mix_transformer import MixTransformer
from dmmath.nlp.tp_transformer import TPTransformer
from dmmath.nlp.pt_dist_trainner import PtDistTrainer

torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)

torch.distributed.init_process_group(backend="nccl")


def count_parameters(model):
    params_cnt = sum(p.numel() for p in model.parameters())
    params_trainable_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_cnt, params_trainable_cnt


parser = argparse.ArgumentParser(description='PyTorch DeepMind math dataset Training')
parser.add_argument('--workdir', type=str)
# parser.add_argument('--task_typ', type=str, default='ans', choices=['ans', 'anno'])
parser.add_argument('--local_rank', type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()

    serialization_dir = f'{args.workdir}/pt'
    param_fil = f'{args.workdir}/experiment.jsonnet'

    is_master_rank = (dist.get_rank() == args.local_rank)

    serialize_config_file = os.path.join(serialization_dir, CONFIG_NAME)
    recover = os.path.exists(serialize_config_file)
    if is_master_rank:
        if not os.path.exists(serialization_dir):
            os.makedirs(serialization_dir, exist_ok=True)
            params = ConstParams.from_file(param_fil)
            params.to_file(serialize_config_file)
    dist.barrier()
    params = ConstParams.from_file(serialize_config_file)

    log_dir = os.path.join(serialization_dir, str(dist.get_rank()))
    os.makedirs(log_dir, exist_ok=True)
    stdout_handler = prepare_global_logging(log_dir, file_friendly_logging=False)
    prepare_environment(params)

    cuda_device = params.trainer.get('cuda_device', -1)
    check_for_gpu(cuda_device)

    trainer_type = params.trainer.type

    trainer = TrainerBase.from_params(params, serialization_dir, recover)
    params_cnt, params_trainable_cnt = count_parameters(trainer.model)
    print("all params cnt: ", params_cnt)
    print("all trainable params cnt: ", params_trainable_cnt)

    metrics = trainer.train()

    cleanup_global_logging(stdout_handler)

    if is_master_rank:
        archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)
