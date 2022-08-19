import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import copy

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel, shift_tokens_right
from transformers.models.bart.configuration_bart import BartConfig
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


class CustomizedBartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.output_mode = "default"
        self.hybrid = False

        # add extra language model head layers for intermediate supervision
        self.num_hybrid_layers = 5

        self.intermediate_lm_layer = nn.Linear(config.d_model, self.model.shared.num_embeddings)
        self.extra_lm_layer = nn.Linear(config.d_model, self.model.shared.num_embeddings)

        self.intermediate_weighting = nn.parameter.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]), requires_grad=True)
        self.extra_intermediate_weighting = nn.parameter.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]), requires_grad=True)
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        # new_embeddings = super().resize_token_embeddings(new_num_tokens)
        new_embeddings = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return new_embeddings

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens
        # Tie weights again if needed
        self.tie_weights()
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            self.set_output_embeddings(new_lm_head)
        # if self.get_intermediate_output_embeddings() is not None and not self.config.tie_word_embeddings:
        #     old_intermediate_lm_head = self.get_intermediate_output_embeddings()
        #     new_intermediate_lm_head = self._get_resized_lm_head(old_intermediate_lm_head, new_num_tokens)
        #     self.set_intermediate_output_embeddings(new_intermediate_lm_head)

        return self.get_input_embeddings()

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
        # intermediate_output_embeddings = self.get_intermediate_output_embeddings()
        # if intermediate_output_embeddings is not None and self.config.tie_word_embeddings:
        #     self._tie_or_clone_weights(intermediate_output_embeddings, self.get_input_embeddings())

        if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def get_output_embeddings(self):
        return self.lm_head
    
    # def get_intermediate_output_embeddings(self):
    #     return self.intermediate_lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    # def set_intermediate_output_embeddings(self, new_embeddings):
    #     self.intermediate_lm_head = new_embeddings

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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        masked_lm_loss = None        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # computing intermediate loss for the first auxiliary task
        intermediate_decoder_weights = nn.functional.softmax(self.intermediate_weighting, dim=0)
        intermediate_hidden_states = torch.stack(outputs['decoder_hidden_states'][1:self.num_hybrid_layers+1], dim=-1)
        intermediate_decoder_outputs = \
            (nn.functional.linear(intermediate_hidden_states, intermediate_decoder_weights) + torch.mean(intermediate_hidden_states, dim=-1))/2
        intermediate_lm_logits = self.intermediate_lm_layer(intermediate_decoder_outputs)

        masked_intermediate_lm_loss = None
        if intermediate_labels is not None:      
            new_intermediate_labels = intermediate_labels.clone()
            new_intermediate_labels[new_intermediate_labels == -100] = 1
            one_hot_intermediate_labels = F.one_hot(new_intermediate_labels, num_classes=self.config.vocab_size).float()
            intermediate_masks = intermediate_masks.bool().unsqueeze(-1).repeat(1, 1, self.config.vocab_size)
            new_intermediate_lm_logits = torch.where(intermediate_masks, intermediate_lm_logits, one_hot_intermediate_labels)
            masked_intermediate_lm_loss = loss_fct(new_intermediate_lm_logits.view(-1, self.config.vocab_size), new_intermediate_labels.view(-1))

        extra_intermediate_lm_logits = None
        masked_extra_intermediate_lm_loss = None
        if self.hybrid or self.output_mode == "extra_intermediate":
            # computing intermediate loss for the second auxiliary task
            extra_intermediate_decoder_weights = nn.functional.softmax(self.extra_intermediate_weighting, dim=0)
            extra_intermediate_hidden_states = torch.stack(outputs['decoder_hidden_states'][1:self.num_hybrid_layers+1], dim=-1)
            extra_intermediate_decoder_outputs = \
                (nn.functional.linear(extra_intermediate_hidden_states, extra_intermediate_decoder_weights) + torch.mean(extra_intermediate_hidden_states, dim=-1))/2
            extra_intermediate_lm_logits = self.extra_lm_layer(extra_intermediate_decoder_outputs)

            if extra_intermediate_labels is not None:
                new_extra_intermediate_labels = extra_intermediate_labels.clone()
                new_extra_intermediate_labels[new_extra_intermediate_labels == -100] = 1
                one_hot_extra_intermediate_labels = F.one_hot(new_extra_intermediate_labels, num_classes=self.config.vocab_size).float()
                extra_intermediate_masks = extra_intermediate_masks.bool().unsqueeze(-1).repeat(1, 1, self.config.vocab_size)
                new_extra_intermediate_lm_logits = torch.where(extra_intermediate_masks, extra_intermediate_lm_logits, one_hot_extra_intermediate_labels)
                masked_extra_intermediate_lm_loss = loss_fct(new_extra_intermediate_lm_logits.view(-1, self.config.vocab_size), new_extra_intermediate_labels.view(-1))

        if self.output_mode == "intermediate":
            lm_logits = intermediate_lm_logits.clone()
        elif self.output_mode == "extra_intermediate":
            lm_logits = extra_intermediate_lm_logits.clone()

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return CustomizedSeq2SeqLMOutput(
            main_loss=masked_lm_loss,
            logits=lm_logits,
            intermediate_logits=intermediate_lm_logits,
            extra_intermediate_logits=extra_intermediate_lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_loss=masked_intermediate_lm_loss,
            extra_intermediate_loss=masked_extra_intermediate_lm_loss,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
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
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past