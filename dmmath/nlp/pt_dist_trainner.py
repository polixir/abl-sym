import math
import os
import time
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Any

import torch
import torch.optim.lr_scheduler

from allennlp.common.util import dump_metrics
from allennlp.common.checks import parse_cuda_device
from allennlp.common.tqdm import Tqdm
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase
import logging
import os
import re
from typing import Iterable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training import util as training_util
from allennlp.nn import util as nn_util
import time

logger = logging.getLogger(__name__)
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dmmath.nlp.nlp_utils import DMDataSet
from allennlp.common.util import get_frozen_and_tunable_parameter_names


def is_master_rank(master_rank=0):
    return dist.get_rank() == master_rank


@TrainerBase.register("pt_dist")
class PtDistTrainer(TrainerBase):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 batch_size: int = 1,
                 validation_metric: str = "-loss",
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 checkpointer: Checkpointer = None,
                 cuda_device: Union[int, List] = -1,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None) -> None:
        super().__init__(serialization_dir, cuda_device)

        self.local_rank = dist.get_rank()
        self.local_device = torch.device("cuda", self.local_rank)
        self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)

        self.batch_size = batch_size

        self.shuffle = shuffle
        self.optimizer = optimizer
        self.train_data = train_dataset
        self._validation_data = validation_dataset

        self._metric_tracker = MetricTracker(metric_name=validation_metric)
        self._validation_metric = validation_metric[1:]

        self._num_epochs = num_epochs

        if checkpointer is not None:
            self._checkpointer = checkpointer
        else:
            self._checkpointer = Checkpointer(serialization_dir, None, num_serialized_models_to_keep)

        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler

        self._batch_num_total = 0

        self._last_log = 0.0  # time of last logging

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor:
        output_dict = self.model(**batch_group)
        loss = output_dict["loss"]
        return loss

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        train_loss = 0.0
        self.model.train()

        num_gpus = len(self._cuda_devices)

        if getattr(self, "train_dataset", None) is None:
            self.train_dataset = DMDataSet(data=self.train_data[0], batch_size=self.batch_size, num_gpus=num_gpus,
                                           shuffle=True, distributed=True, data_slice=True)
        self.train_dataset.set_epoch(epoch)
        num_training_batches = math.ceil(len(self.train_dataset) / self.batch_size / num_gpus)
        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(self.train_dataset, total=num_training_batches)

        for batch_group in train_generator_tqdm:
            # print('batch_size: ', len(batch_group["source_tokens"]["tokens"]))
            # gpu_data = batch_group
            # src_data = gpu_data["source_tokens"]["tokens"]
            # tgt_data = gpu_data["target_tokens"]["tokens"]
            # for sdata, tdata in zip(src_data, tgt_data):
            #     s = ''.join([self.get_model().vocab.get_token_from_index(x, "source_tokens") if x != 0 else '' for x in
            #                  sdata.numpy()])
            #     t = ''.join([self.get_model().vocab.get_token_from_index(x, "target_tokens") if x != 0 else '' for x in
            #                  tdata.numpy()])
            #     print(s)
            #     print(t)
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            loss = self.batch_loss(batch_group, for_training=True)

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")
            loss.backward()

            train_loss += loss.item()

            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)

            self.optimizer.step()
            metrics = training_util.get_metrics(self.get_model(), train_loss, batches_this_epoch)
            description = self.get_desc_from_metrics(metrics, epoch)
            train_generator_tqdm.set_description(description, refresh=False)

        metrics = training_util.get_metrics(self.get_model(), train_loss, batches_this_epoch, reset=True)
        return metrics

    def get_desc_from_metrics(self, metrics, epoch=None):
        description = training_util.description_from_metrics(metrics)
        if epoch is None:
            description = f'epoch: -- rank: {dist.get_rank()} || {description}'
        else:
            description = f'epoch: {epoch} rank: {dist.get_rank()} || {description}'
        return description

    def get_model(self):
        return self.model.module

    def _validation_loss(self) -> Tuple[float, int]:
        logger.info("Validating")

        self.model.eval()

        num_gpus = len(self._cuda_devices)

        if getattr(self, "val_dataset", None) is None:
            self.val_dataset = DMDataSet(data=self._validation_data[0], batch_size=self.batch_size, num_gpus=num_gpus,
                                         shuffle=False, distributed=True, data_slice=False)
        num_validation_batches = math.ceil(len(self.val_dataset) / self.batch_size / num_gpus)
        val_generator_tqdm = Tqdm.tqdm(self.val_dataset,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch_group in val_generator_tqdm:
            loss = self.batch_loss(batch_group, for_training=False)
            if loss is not None:
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.get_model(), val_loss, batches_this_epoch)
            description = self.get_desc_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        return val_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics['best_epoch'] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value
        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)
            if self._validation_data is not None:
                with torch.no_grad():
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(self.get_model(), val_loss, num_batches, reset=True)
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and is_master_rank():
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
            if is_master_rank():
                self._save_checkpoint(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                                           ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self._batch_num_total
        }

        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()

        self._checkpointer.save_checkpoint(
            model_state=self.model.state_dict(),
            epoch=epoch,
            training_states=training_states,
            is_best_so_far=self._metric_tracker.is_best_so_far())

    def _restore_checkpoint(self) -> int:
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self._learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the ``training_state`` contains a serialized ``MetricTracker``.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked ``val_metric_per_epoch``.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get('batch_num_total')
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return

    # Requires custom from_params.
    @classmethod
    def from_params(cls, params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None) -> 'PtDistTrainer':
        all_datasets = training_util.datasets_from_params(params, cache_directory, cache_prefix)
        vocab = Vocabulary.from_files(params.vocabulary.directory_path)

        model = Model.from_params(vocab=vocab, params=params.pop('model'))
        model.extend_embedder_vocab()
        if is_master_rank():
            vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        train_data = all_datasets['train']
        validation_data = all_datasets.get('validation')

        batch_size = params.iterator.batch_size

        trainer_params = params.pop("trainer")
        keys = [key for key in params]
        for key in keys:
            params.pop(key)
        params = trainer_params
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        pretrain_file = params.pop("pretrain_file", None)

        no_grad_regexes = params.pop("no_grad", ())
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        frozen_parameter_names, tunable_parameter_names = \
            get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are Frozen  (without gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are Tunable (with gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        model = model.cuda(dist.get_rank())
        if pretrain_file:
            model_state = torch.load(pretrain_file, map_location=nn_util.device_mapping(dist.get_rank()))
            model.load_state_dict(model_state)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        # print([n for n, p in model.named_parameters() if p.requires_grad])
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        checkpointer = Checkpointer(
            serialization_dir=serialization_dir,
            num_serialized_models_to_keep=num_serialized_models_to_keep,
            keep_serialized_model_every_num_seconds=None)

        return cls(model, optimizer,
                   train_data, validation_data,
                   batch_size=batch_size,
                   validation_metric=validation_metric,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=lr_scheduler,
                   checkpointer=checkpointer)
