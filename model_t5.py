import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import copy

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Model, T5PreTrainedModel, T5Stack
from transformers.models.t5.configuration_t5 import T5Config
from transformers.file_utils import ModelOutput

@dataclass
class CustomizedSeq2SeqLMOutput(Seq2SeqLMOutput):
    main_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    intermediate_logits: torch.FloatTensor = None
    extra_intermediate_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    intermediate_loss: Optional[torch.FloatTensor] = None
    extra_intermediate_loss: Optional[torch.FloatTensor] = None

class CustomizedT5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.model = T5Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # add extra language model head layers for intermediate supervision
        self.num_hybrid_layers = 5
        self.shared_lm = True
        # self.intermediate_lm_layer = nn.Linear(config.d_model, self.model.shared.num_embeddings)
        self.intermediate_weighting = nn.parameter.Parameter(torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2]), requires_grad=True)
        self.extra_intermediate_weighting = nn.parameter.Parameter(torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]), requires_grad=True)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        intermediate_labels=None,
        intermediate_masks=None,
        extra_intermediate_labels=None,
        extra_intermediate_masks=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # add extra language model logits for computing intermediate loss
        intermediate_decoder_weights = nn.functional.softmax(self.intermediate_weighting, dim=0)
        intermediate_decoder_outputs = nn.functional.linear(
            torch.stack(outputs['decoder_hidden_states'][1:self.num_hybrid_layers + 1], dim=-1),
            intermediate_decoder_weights)
        intermediate_lm_logits = self.lm_head(intermediate_decoder_outputs)

        masked_intermediate_lm_loss = None
        masked_overall_lm_loss = None
        if intermediate_labels is not None:
            intermediate_loss_fct = CrossEntropyLoss()

            new_intermediate_labels = intermediate_labels.clone()
            new_intermediate_labels[new_intermediate_labels == -100] = 1
            one_hot_intermediate_labels = F.one_hot(new_intermediate_labels, num_classes=self.config.vocab_size).float()
            intermediate_masks = intermediate_masks.bool().unsqueeze(-1).repeat(1, 1, self.config.vocab_size)
            new_intermediate_lm_logits = torch.where(intermediate_masks, intermediate_lm_logits,
                                                     one_hot_intermediate_labels)
            masked_intermediate_lm_loss = intermediate_loss_fct(
                new_intermediate_lm_logits.view(-1, self.config.vocab_size), new_intermediate_labels.view(-1))
            masked_overall_lm_loss = masked_lm_loss + alpha * masked_intermediate_lm_loss

        if extra_intermediate_labels is not None:
            intermediate_decoder_weights = nn.functional.softmax(self.extra_intermediate_weighting, dim=0)
            intermediate_decoder_outputs = nn.functional.linear(
                torch.stack(outputs['decoder_hidden_states'][1:self.num_hybrid_layers + 1], dim=-1),
                intermediate_decoder_weights)
            extra_intermediate_lm_logits = self.lm_head(intermediate_decoder_outputs)

            new_extra_intermediate_labels = extra_intermediate_labels.clone()
            new_extra_intermediate_labels[new_extra_intermediate_labels == -100] = 1
            one_hot_extra_intermediate_labels = F.one_hot(new_extra_intermediate_labels,
                                                          num_classes=self.config.vocab_size).float()
            extra_intermediate_masks = extra_intermediate_masks.bool().unsqueeze(-1).repeat(1, 1,
                                                                                            self.config.vocab_size)
            new_extra_intermediate_lm_logits = torch.where(extra_intermediate_masks, extra_intermediate_lm_logits,
                                                           one_hot_extra_intermediate_labels)
            masked_extra_intermediate_lm_loss = intermediate_loss_fct(
                new_extra_intermediate_lm_logits.view(-1, self.config.vocab_size),
                new_extra_intermediate_labels.view(-1))
            masked_lm_loss = masked_lm_loss + alpha * masked_extra_intermediate_lm_loss

        if not return_dict:
            output = (lm_logits,) + (intermediate_lm_logits,) + outputs[1:]
            return ((masked_overall_lm_loss,) + output) if masked_overall_lm_loss is not None else output

        return CustomizedSeq2SeqLMOutput(
            loss=masked_overall_lm_loss,
            logits=lm_logits,
            intermediate_logits=intermediate_lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            main_loss=masked_lm_loss,
            intermediate_loss=masked_intermediate_lm_loss,
        )

