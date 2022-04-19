import torch
import torch.nn as nn
from typing import Optional, Tuple
import math
import transformers
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import BartPretrainedModel, BartEncoder, BartDecoder, shift_tokens_right

class BartModel(BartPretrainedModel):
    def __init__(self, config: transformers.BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.intermediate_decoder = BartDecoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.intermediate_decoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder
    
    def get_intermediate_decoder(self):
        return self.intermediate_decoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids,
        attention_mask=None,
        intermediate_decoder_input_ids=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        intermediate_decoder_attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        intermediate_decoder_head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        intermediate_decoder_inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # Bart automatically creates decoder_input_ids from input_ids if no decoder_input_ids are provided
        if intermediate_decoder_input_ids is None and intermediate_decoder_inputs_embeds is None:
            intermediate_decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        
        if intermediate_decoder_attention_mask is None:
            intermediate_decoder_attention_mask = torch.ones([intermediate_decoder_input_ids.shape[0], intermediate_decoder_input_ids.shape[1]], device=intermediate_decoder_input_ids.device)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # intermediate decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        intermediate_decoder_outputs = self.intermediate_decoder(
            input_ids=intermediate_decoder_input_ids,
            attention_mask=intermediate_decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=intermediate_decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=intermediate_decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=torch.cat((encoder_outputs[0], intermediate_decoder_outputs[0]), dim=1),
            encoder_attention_mask=torch.cat((attention_mask, intermediate_decoder_attention_mask), dim=1),
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + intermediate_decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=intermediate_decoder_outputs.last_hidden_state,
            past_key_values=intermediate_decoder_outputs.past_key_values,
            decoder_hidden_states=intermediate_decoder_outputs.hidden_states,
            decoder_attentions=intermediate_decoder_outputs.attentions,
            cross_attentions=intermediate_decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=torch.cat((encoder_outputs.last_hidden_state, intermediate_decoder_outputs.last_hidden_state), dim=1) if encoder_outputs.last_hidden_state is not None else None,
            encoder_hidden_states=torch.cat((encoder_outputs.hidden_states, intermediate_decoder_outputs.hidden_states), dim=1) if encoder_outputs.hidden_states is not None else None,
            encoder_attentions=torch.cat((encoder_outputs.attentions, intermediate_decoder_outputs.attentions), dim=1) if encoder_outputs.attentions is not None else None,
        )


class BartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: transformers.BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.intermediate_lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_intermediate_decoder(self):
        return self.model.get_intermediate_decoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
    
    def get_intermediate_embeddings(self):
        return self.intermediate_lm_head
    
    def set_intermediate_embeddings(self, new_embeddings):
        self.intermediate_lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        intermediate_decoder_input_ids=None,
        intermediate_decoder_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        intermediate_decoder_head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        intermediate_decoder_inputs_embeds=None,
        decoder_inputs_embeds=None,
        intermediate_labels=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        intermediate_outputs, outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            intermediate_decoder_input_ids=intermediate_decoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            intermediate_decoder_attention_mask=intermediate_decoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            intermediate_decoder_head_mask=intermediate_decoder_head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            intermediate_decoder_inputs_embeds=intermediate_decoder_inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        intermediate_lm_logits = self.intermediate_lm_head(intermediate_outputs[0])
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        intermediate_masked_lm_loss, masked_lm_loss = None, None
        if intermediate_labels is not None:
            intermediate_loss_fct = nn.CrossEntropyLoss()
            intermediate_masked_lm_loss = intermediate_loss_fct(intermediate_lm_logits.view(-1, self.config.vocab_size), intermediate_labels.view(-1))
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + (intermediate_lm_logits,) + outputs[1:]
            if masked_lm_loss is not None and intermediate_masked_lm_loss is not None:
                return ((masked_lm_loss,) + (intermediate_masked_lm_loss,)  + output) 
            elif masked_lm_loss is not None:
                return ((masked_lm_loss,) + output)
            else:
                return output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
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


if __name__ == '__main__':
    model = BartForConditionalGeneration.from_pretrained('./Unified_IR/bart-base/')