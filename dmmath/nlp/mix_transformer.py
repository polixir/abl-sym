from typing import Dict, List, Tuple, Any

import numpy
from overrides import overrides
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.nn as nn

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import BLEU, SequenceAccuracy
import math
import torch
from torch.nn.modules.transformer import TransformerDecoder, TransformerEncoder, TransformerDecoderLayer, \
    TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
from torch.nn.modules import Module, ModuleDict
from allennlp.training.metrics.metric import Metric


class MultiDecodersTransformer(Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=None, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super(MultiDecodersTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoders = {}
        if num_decoder_layers:
            for k, v in num_decoder_layers.items():
                decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
                decoder_norm = LayerNorm(d_model)
                decoder = TransformerDecoder(decoder_layer, v, decoder_norm)
                decoders[k] = decoder
        self.decoders = ModuleDict(decoders.items())

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


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


@Model.register("mix-transformer")
class MixTransformer(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            source_embedder: TextFieldEmbedder,
            transformer: Dict,
            max_decoding_steps: int,
            target_embedders: Dict[str, TextFieldEmbedder] = None,
            loss_coefs: Dict = None,
    ) -> None:
        super().__init__(vocab)
        self._target_namespaces = list(filter(lambda x: x != 'span', loss_coefs.keys()))
        self._decoder_namespaces = transformer.get("num_decoder_layers", {}).keys()
        self._start_index_dict = {k: self.vocab.get_token_index(START_SYMBOL, k) for k in self._decoder_namespaces}
        self._end_index_dict = {k: self.vocab.get_token_index(END_SYMBOL, k) for k in self._decoder_namespaces}
        self._pad_index_dict = {k: self.vocab.get_token_index(self.vocab._padding_token, k) for k in
                                self._target_namespaces}
        self._loss_coefs = loss_coefs
        self._metrics = {}
        for tn in self._target_namespaces:
            self._metrics[f'{tn}_acc'] = SequenceAccuracy()

        self._max_decoding_steps = max_decoding_steps

        self._source_embedder = source_embedder

        self._ndim = transformer["d_model"]
        self.pos_encoder = PositionalEncoding(self._ndim, transformer["dropout"])

        self._transformer = MultiDecodersTransformer(**transformer)
        self._transformer.apply(inplace_relu)

        self._target_embedders = ModuleDict(target_embedders.items())
        output_projection_layers = {}
        for tn in self._target_namespaces:
            num_classes = self.vocab.get_vocab_size(tn)
            output_projection_layers[tn] = Linear(self._ndim, num_classes)

        self._output_projection_layers = ModuleDict(output_projection_layers.items())

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
        loss = torch.tensor(0.0).to(memory.device)
        output_dict = {}
        for tn in self._target_namespaces:
            if tn in self._decoder_namespaces:
                targets = target_tokens[tn][:, 1:]
                target_mask = (util.get_text_field_mask({tn: targets}) == 1)
                assert targets.size(1) <= self._max_decoding_steps
                if self.training and target_tokens:
                    tgt, tgt_key_padding_mask = self._encode(self._target_embedders[tn],
                                                             {tn: target_tokens[tn][:, :-1]})
                    tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(memory.device)
                    output = self._transformer.decoders[tn](tgt, memory, tgt_mask=tgt_mask,
                                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                                            memory_key_padding_mask=src_key_padding_mask)
                    logits = self._output_projection_layers[tn](output)
                    class_probabilities = F.softmax(logits.detach(), dim=-1)
                    _, predictions = torch.max(class_probabilities, -1)
                    logits = logits.transpose(0, 1)
                    tn_loss = self._get_loss(logits, targets, target_mask)
                else:
                    assert self.training is False
                    tn_loss = torch.tensor(0.0).to(memory.device)
                    if targets is not None:
                        max_target_len = targets.size(1)
                    else:
                        max_target_len = None
                    predictions = self._decoder_step_by_step(memory, src_key_padding_mask, target_vocab_mask=None,
                                                             max_target_len=max_target_len, target_namespace=tn)
            else:
                targets = target_tokens[tn]
                target_mask = (util.get_text_field_mask({tn: targets}) == 1)
                if target_tokens:
                    logits = self._output_projection_layers[tn](memory)
                    class_probabilities = F.softmax(logits.detach(), dim=-1)
                    _, predictions = torch.max(class_probabilities, -1)
                    logits = logits.transpose(0, 1)
                    tn_loss = self._get_loss(logits, targets, target_mask)
                else:
                    tn_loss = torch.tensor(0.0).to(memory.device)
            loss += tn_loss * self._loss_coefs[tn]
            predictions = predictions.transpose(0, 1)
            self._metrics[f"{tn}_loss"] = tn_loss.item()
            output_dict[f"{tn}_predictions"] = predictions

            if target_tokens:
                with torch.no_grad():
                    best_predictions = output_dict[f"{tn}_predictions"]
                    batch_size = targets.size(0)
                    max_sz = max(best_predictions.size(1), targets.size(1), target_mask.size(1))
                    best_predictions_ = torch.zeros(batch_size, max_sz).to(memory.device)
                    best_predictions_[:, :best_predictions.size(1)] = best_predictions
                    targets_ = torch.zeros(batch_size, max_sz).to(memory.device)
                    targets_[:, :targets.size(1)] = targets.cpu()
                    target_mask_ = torch.zeros(batch_size, max_sz).to(memory.device)
                    target_mask_[:, :target_mask.size(1)] = target_mask
                    self._metrics[f'{tn}_acc'](best_predictions_.unsqueeze(1), targets_, target_mask_)
        output_dict["loss"] = loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predict_dict = {}
        for k in output_dict:
            if "predictions" not in k:
                continue
            tn = k.split('_')[0]
            predicted_indices = output_dict[k]
            if not isinstance(predicted_indices, numpy.ndarray):
                predicted_indices = predicted_indices.detach().cpu().numpy()
            all_predicted_tokens = []
            for indices in predicted_indices:
                if len(indices.shape) > 1:
                    indices = indices[0]
                indices = list(indices)
                # Collect indices till the first end_symbol
                if tn in self._end_index_dict and self._end_index_dict[tn] in indices:
                    indices = indices[: indices.index(self._end_index_dict[tn])]
                predicted_tokens = [
                    self.vocab.get_token_from_index(x, namespace=tn)
                    for x in indices
                ]
                all_predicted_tokens.append(predicted_tokens)
            predict_dict[tn] = all_predicted_tokens
        return predict_dict

    def _encode(self, embedder: TextFieldEmbedder, tokens: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        src = embedder(tokens) * math.sqrt(self._ndim)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        mask = util.get_text_field_mask(tokens)
        mask = (mask == 0)
        return src, mask

    def _decoder_step_by_step(self, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor,
                              target_vocab_mask: torch.Tensor = None, max_target_len: int = None,
                              target_namespace: str = None) -> torch.Tensor:
        batch_size = memory.size(1)
        tn = target_namespace
        if getattr(self, "target_limit_decode_steps", False) and max_target_len is not None:
            num_decoding_steps = min(self._max_decoding_steps, max_target_len)
            print('decoding steps: ', num_decoding_steps)
        else:
            num_decoding_steps = self._max_decoding_steps

        last_predictions = memory.new_full((batch_size,), fill_value=self._start_index_dict[tn]).long()

        step_predictions: List[torch.Tensor] = []
        all_predicts = memory.new_full((batch_size, num_decoding_steps), fill_value=0).long()
        for timestep in range(num_decoding_steps):
            all_predicts[:, timestep] = last_predictions
            tgt, tgt_key_padding_mask = self._encode(self._target_embedders[tn],
                                                     {tn: all_predicts[:, :timestep + 1]})
            tgt_mask = self.generate_square_subsequent_mask(timestep + 1).to(memory.device)
            output = self._transformer.decoders[tn](tgt, memory, tgt_mask=tgt_mask,
                                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                                    memory_key_padding_mask=memory_key_padding_mask)
            output_projections = self._output_projection_layers[tn](output)
            if target_vocab_mask is not None:
                output_projections += target_vocab_mask

            class_probabilities = F.softmax(output_projections, dim=-1)
            _, predicted_classes = torch.max(class_probabilities, -1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes[timestep, :]
            step_predictions.append(last_predictions)
            if ((last_predictions == self._end_index_dict[tn]) + (
                    last_predictions == self._pad_index_dict[tn])).all():
                break

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.stack(step_predictions)
        return predictions

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
        for k in self._metrics:
            if isinstance(self._metrics[k], Metric):
                all_metrics[k] = self._metrics[k].get_metric(reset=reset)
            else:
                all_metrics[k] = self._metrics[k]
        return all_metrics

    # def load_state_dict(self, state_dict, strict=True):
    #     new_state_dict = {}
    #     for k, v in state_dict.items():
    #         if k.startswith('module.'):
    #             new_state_dict[k[len('module.'):]] = v
    #         else:
    #             new_state_dict[k] = v
    #
    #     super(MixTransformer, self).load_state_dict(new_state_dict, strict)

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        if any([True for k in state_dict if 'decoders' in k]):
            pretrain = False
        else:
            pretrain = True
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            if pretrain:
                if 'decoder' in k:
                    newk = k.replace('decoder', 'decoders.answer')
                elif 'target_embedder.token_embedder_tokens' in k:
                    newk = k.replace('target_embedder.token_embedder_tokens',
                                     'target_embedders.answer.token_embedder_answer')
                elif 'output_projection_layer' in k:
                    newk = k.replace('output_projection_layer', 'output_projection_layers.answer')
                else:
                    newk = k
            else:
                newk = k
            new_state_dict[newk] = v
        super(MixTransformer, self).load_state_dict(new_state_dict, strict=False)
