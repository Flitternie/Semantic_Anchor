import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import copy

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Model, T5PreTrainedModel, T5Stack, T5ForConditionalGeneration
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
        self.hybrid = False

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

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.num_hybrid_layers = 11
        self.shared_lm = True
        self.intermediate_lm_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.extra_lm_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.intermediate_weighting = nn.parameter.Parameter(torch.ones(11), requires_grad=True)
        self.extra_intermediate_weighting = nn.parameter.Parameter(torch.ones(11), requires_grad=True)
        # Initialize weights and apply final processing
        self.post_init()
        print(config.vocab_size)
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:

        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens

        if hasattr(self, 'intermediate_lm_layer'):
            self.intermediate_lm_layer = nn.Linear(self.config.d_model, self.vocab_size, bias=False)
        if hasattr(self, 'extra_lm_layer'):
            self.extra_lm_layer = nn.Linear(self.config.d_model, self.vocab_size, bias=False)

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

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

    def parallelize(self, device_map=None):
        super(CustomizedT5ForConditionalGeneration, self).parallelize(device_map)
        self.intermediate_weighting = self.intermediate_weighting.to(self.decoder.first_device)
        self.extra_intermediate_weighting = self.extra_intermediate_weighting.to(self.decoder.first_device)
        self.intermediate_lm_layer = self.intermediate_lm_layer.to(self.decoder.first_device)
        self.extra_lm_layer = self.extra_lm_layer.to(self.decoder.first_device)

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

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
            output_hidden_states=True,
            return_dict=None,
            intermediate_labels=None,
            intermediate_masks=None,
            extra_intermediate_labels=None,
            extra_intermediate_masks=None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # computing intermediate loss for the first auxiliary task
        intermediate_lm_logits = None
        if decoder_outputs.hidden_states is not None:
            intermediate_decoder_weights = nn.functional.softmax(self.intermediate_weighting, dim=0)
            intermediate_hidden_states = torch.stack(decoder_outputs.hidden_states[1:self.num_hybrid_layers + 1], dim=-1)
            intermediate_decoder_outputs = \
                (nn.functional.linear(intermediate_hidden_states, intermediate_decoder_weights) + torch.mean(
                    intermediate_hidden_states, dim=-1)) / 2
            intermediate_lm_logits = self.intermediate_lm_layer(intermediate_decoder_outputs)

        masked_intermediate_lm_loss = None
        if intermediate_labels is not None and intermediate_lm_logits is not None:
            new_intermediate_labels = intermediate_labels.clone()
            new_intermediate_labels[new_intermediate_labels == -100] = 1
            one_hot_intermediate_labels = F.one_hot(new_intermediate_labels, num_classes=self.config.vocab_size).float()
            intermediate_masks = intermediate_masks.bool().unsqueeze(-1).repeat(1, 1, self.config.vocab_size)
            new_intermediate_lm_logits = torch.where(intermediate_masks, intermediate_lm_logits, one_hot_intermediate_labels)
            masked_intermediate_lm_loss = loss_fct(new_intermediate_lm_logits.view(-1, self.config.vocab_size), new_intermediate_labels.view(-1))

        extra_intermediate_lm_logits = None
        masked_extra_intermediate_lm_loss = None
        if self.hybrid and decoder_outputs.hidden_states is not None:
            # computing intermediate loss for the second auxiliary task
            extra_intermediate_decoder_weights = nn.functional.softmax(self.extra_intermediate_weighting, dim=0)
            extra_intermediate_hidden_states = torch.stack(decoder_outputs.hidden_states[1:self.num_hybrid_layers + 1], dim=-1)
            extra_intermediate_decoder_outputs = \
                (nn.functional.linear(extra_intermediate_hidden_states, extra_intermediate_decoder_weights) + torch.mean(extra_intermediate_hidden_states, dim=-1)) / 2
            extra_intermediate_lm_logits = self.extra_lm_layer(extra_intermediate_decoder_outputs)

            if extra_intermediate_labels is not None:
                new_extra_intermediate_labels = extra_intermediate_labels.clone()
                new_extra_intermediate_labels[new_extra_intermediate_labels == -100] = 1
                one_hot_extra_intermediate_labels = F.one_hot(new_extra_intermediate_labels, num_classes=self.config.vocab_size).float()
                extra_intermediate_masks = extra_intermediate_masks.bool().unsqueeze(-1).repeat(1, 1, self.config.vocab_size)
                new_extra_intermediate_lm_logits = torch.where(extra_intermediate_masks, extra_intermediate_lm_logits, one_hot_extra_intermediate_labels)
                masked_extra_intermediate_lm_loss = loss_fct(new_extra_intermediate_lm_logits.view(-1, self.config.vocab_size), new_extra_intermediate_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return CustomizedSeq2SeqLMOutput(
            main_loss=masked_lm_loss,
            logits=lm_logits,
            intermediate_logits=intermediate_lm_logits,
            extra_intermediate_logits=extra_intermediate_lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_loss=masked_intermediate_lm_loss,
            extra_intermediate_loss=masked_extra_intermediate_lm_loss,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
