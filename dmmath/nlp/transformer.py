from typing import Dict, List, Tuple, Any

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.nn as nn

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import BLEU, SequenceAccuracy
from torch.nn import Transformer
import math
import time


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


@Model.register("my-transformer")
class MyTransformer(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            source_embedder: TextFieldEmbedder,
            transformer: Dict,
            max_decoding_steps: int,
            target_namespace: str,
            target_embedder: TextFieldEmbedder = None,
            use_bleu: bool = True,
    ) -> None:
        super().__init__(vocab)
        self._target_namespace = target_namespace

        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._pad_index = self.vocab.get_token_index(
            self.vocab._padding_token, self._target_namespace
        )

        if use_bleu:
            self._bleu = BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None
        self._seq_acc = SequenceAccuracy()

        self._max_decoding_steps = max_decoding_steps

        self._source_embedder = source_embedder

        self._ndim = transformer["d_model"]
        self.pos_encoder = PositionalEncoding(self._ndim, transformer["dropout"])

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        self._transformer = Transformer(**transformer)
        self._transformer.apply(inplace_relu)

        if target_embedder is None:
            self._target_embedder = self._source_embedder
        else:
            self._target_embedder = target_embedder

        self._output_projection_layer = Linear(self._ndim, num_classes)

    def _get_mask(self, meta_data):
        mask = torch.zeros(1, len(meta_data), self.vocab.get_vocab_size(self._target_namespace)).float()
        for bidx, md in enumerate(meta_data):
            for k, v in self.vocab._token_to_index[self._target_namespace].items():
                if 'position' in k and k not in md['avail_pos']:
                    mask[:, bidx, v] = float('-inf')
        return mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == False, float('-inf')).masked_fill(mask == True, float(0.0))
        return mask

    @overrides
    def forward(
            self,
            source_tokens: Dict[str, torch.LongTensor],
            target_tokens: Dict[str, torch.LongTensor] = None,
            meta_data: Any = None,
    ) -> Dict[str, torch.Tensor]:
        src, src_key_padding_mask = self._encode(self._source_embedder, source_tokens)
        memory = self._transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)

        if meta_data is not None:
            target_vocab_mask = self._get_mask(meta_data)
            target_vocab_mask = target_vocab_mask.to(memory.device)
        else:
            target_vocab_mask = None
        output_dict = {}
        targets = None
        if target_tokens:
            targets = target_tokens["tokens"][:, 1:]
            target_mask = (util.get_text_field_mask({"tokens": targets}) == 1)
            assert targets.size(1) <= self._max_decoding_steps
        if self.training and target_tokens:
            tgt, tgt_key_padding_mask = self._encode(self._target_embedder, {"tokens": target_tokens["tokens"][:, :-1]})
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(memory.device)
            output = self._transformer.decoder(tgt, memory, tgt_mask=tgt_mask,
                                               tgt_key_padding_mask=tgt_key_padding_mask,
                                               memory_key_padding_mask=src_key_padding_mask)
            logits = self._output_projection_layer(output)
            if target_vocab_mask is not None:
                logits += target_vocab_mask
            class_probabilities = F.softmax(logits.detach(), dim=-1)
            _, predictions = torch.max(class_probabilities, -1)
            logits = logits.transpose(0, 1)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss
        else:
            assert self.training is False
            output_dict["loss"] = torch.tensor(0.0).to(memory.device)
            if targets is not None:
                max_target_len = targets.size(1)
            else:
                max_target_len = None
            predictions, class_probabilities = self._decoder_step_by_step(memory, src_key_padding_mask,
                                                                          target_vocab_mask,
                                                                          max_target_len=max_target_len)
        predictions = predictions.transpose(0, 1)
        output_dict["predictions"] = predictions
        output_dict["class_probabilities"] = class_probabilities.transpose(0, 1)

        if target_tokens:
            with torch.no_grad():
                best_predictions = output_dict["predictions"]
                if self._bleu:
                    self._bleu(best_predictions, targets)
                batch_size = targets.size(0)
                max_sz = max(best_predictions.size(1), targets.size(1), target_mask.size(1))
                best_predictions_ = torch.zeros(batch_size, max_sz).to(memory.device)
                best_predictions_[:, :best_predictions.size(1)] = best_predictions
                targets_ = torch.zeros(batch_size, max_sz).to(memory.device)
                targets_[:, :targets.size(1)] = targets.cpu()
                target_mask_ = torch.zeros(batch_size, max_sz).to(memory.device)
                target_mask_[:, :target_mask.size(1)] = target_mask
                self._seq_acc(best_predictions_.unsqueeze(1), targets_, target_mask_)
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            # shape: (batch_size, num_decoding_steps)
            predicted_indices = predicted_indices.detach().cpu().numpy()
            # class_probabilities = output_dict["class_probabilities"].detach().cpu()
            # sample_predicted_indices = []
            # for cp in class_probabilities:
            #     sample = torch.multinomial(cp, num_samples=1)
            #     sample_predicted_indices.append(sample)
            # # shape: (batch_size, num_decoding_steps, num_samples)
            # sample_predicted_indices = torch.stack(sample_predicted_indices)

        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            predicted_tokens = [
                self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _encode(self, embedder: TextFieldEmbedder, tokens: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        src = embedder(tokens) * math.sqrt(self._ndim)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        mask = util.get_text_field_mask(tokens)
        mask = (mask == 0)
        return src, mask

    def _decoder_step_by_step(self, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor,
                              target_vocab_mask: torch.Tensor = None, max_target_len: int = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        batch_size = memory.size(1)
        if getattr(self, "target_limit_decode_steps", False) and max_target_len is not None:
            num_decoding_steps = min(self._max_decoding_steps, max_target_len)
            print('decoding steps: ', num_decoding_steps)
        else:
            num_decoding_steps = self._max_decoding_steps

        last_predictions = memory.new_full((batch_size,), fill_value=self._start_index).long()

        step_predictions: List[torch.Tensor] = []
        all_predicts = memory.new_full((batch_size, num_decoding_steps), fill_value=0).long()
        for timestep in range(num_decoding_steps):
            all_predicts[:, timestep] = last_predictions
            tgt, tgt_key_padding_mask = self._encode(self._target_embedder, {"tokens": all_predicts[:, :timestep + 1]})
            tgt_mask = self.generate_square_subsequent_mask(timestep + 1).to(memory.device)
            output = self._transformer.decoder(tgt, memory, tgt_mask=tgt_mask,
                                               tgt_key_padding_mask=tgt_key_padding_mask,
                                               memory_key_padding_mask=memory_key_padding_mask)
            output_projections = self._output_projection_layer(output)
            if target_vocab_mask is not None:
                output_projections += target_vocab_mask

            class_probabilities = F.softmax(output_projections, dim=-1)
            _, predicted_classes = torch.max(class_probabilities, -1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes[timestep, :]
            step_predictions.append(last_predictions)
            if ((last_predictions == self._end_index) + (last_predictions == self._pad_index)).all():
                break

        # shape: (num_decoding_steps, batch_size)
        predictions = torch.stack(step_predictions)
        return predictions, class_probabilities

    @staticmethod
    def _get_loss(
            logits: torch.FloatTensor, targets: torch.LongTensor, target_mask: torch.FloatTensor
    ) -> torch.Tensor:
        logits = logits.contiguous()
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets.contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask.contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        all_metrics['seq_acc'] = self._seq_acc.get_metric(reset=reset)
        return all_metrics

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[len('module.'):]] = v
            else:
                new_state_dict[k] = v

        super(MyTransformer, self).load_state_dict(new_state_dict, strict)
