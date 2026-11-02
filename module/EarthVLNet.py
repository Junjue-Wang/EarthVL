"""The two VQA branches of EarthVLNet (paper Sec. 4).

* ``EarthVLNet`` (LLM branch): the stage-1 features (ConvNeXt-L, 1536 channels at stride 32) are concatenated with an
  embedding of the predicted land-cover mask, refined by object-guided attention (OGA) and projected (MMP) into 1024
  visual tokens that replace the ``<image>`` token of a prompt for the Vicuna-7B language model of LLaVA-1.5, which
  answers with free text. As in the released run, the language model stays frozen in bf16 and is fine-tuned with LoRA
  (r=64, alpha=16, dropout 0.05 on every linear layer except ``lm_head``). ``load_in_4bit=True`` loads it in 4-bit NF4
  instead (QLoRA, needs bitsandbytes) to fit smaller GPUs; the released weights were not trained this way.
* ``EarthVLCount`` (counting branch): the same visual encoding projected to 384 channels, together with the embedded
  question, feeds a 3-layer Transformer that classifies the number in the answer (24 classes, see
  ``module.metric.Reverse_Lookup``); trained with ``module.loss.NumericalLoss``.

Module and parameter names match the released checkpoints; do not rename them.

``EarthVLNet._merge_input_ids_with_image_features`` and ``EarthVLNet.llm_forward`` are adapted from the LLaVA /
LLaVA-NeXT models of Hugging Face transformers (v4.40, ``models/llava_next/modeling_llava_next.py``),
Copyright 2024 the HuggingFace Inc. team, licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0); ``get_nb_trainable_parameters`` is adapted from Hugging Face PEFT
(Apache-2.0). See THIRD_PARTY_NOTICES.md.

Note for subclasses: ever-beta's ``ERModule`` turns an ``__init__`` into a keyword-argument constructor unless its
source contains ``super().__init__``; keep that spelling.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import ever as er
from module.base_modules import (DEFAULT_INDEX, OGA, LayerNorm2d, MMProjector, Transformer, device,
                                 find_all_linear_names, rank0_print)
from module.loss import NumericalLoss
from module.metric import Reverse_Lookup, find_pos_index
from module.position_emb import positionalencoding1d

SYSTEM_PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions. ")


@er.registry.MODEL.register()
class EarthVLNet(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, LlamaForCausalLM

        lm_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation=self.config.attn_implementation)
        if self.config.load_in_4bit:  # QLoRA option; the released run used the bf16 base
            from transformers import BitsAndBytesConfig  # needs bitsandbytes at load time
            lm_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.language_model = LlamaForCausalLM.from_pretrained(self.config.pretrained_model_dir, **lm_kwargs)
        if self.config.load_in_4bit:
            self.language_model = prepare_model_for_kbit_training(
                model=self.language_model, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs=dict(use_reentrant=False))
        # LLaVA processor (the tokenizer has <image> = 32000 and <pad> = 32001); a bare tokenizer also works
        self.language_processer = AutoProcessor.from_pretrained(self.config.pretrained_model_dir)
        self.tokenizer = getattr(self.language_processer, 'tokenizer', self.language_processer)
        # Not used in the computation; kept so that the original DeepSpeed checkpoints load with matching keys. Frozen,
        # so that weight files with the trainable parameters only (such as the released one) do not need it. It was
        # trainable in the original runs, so DeepSpeed cannot restore their frozen parameters, and --eval_only reads
        # those checkpoints with the engine's own reader (engine/trainer.py, _frozen_restore_failure).
        self.image_newline = nn.Parameter(torch.zeros(self.config.language_model.hidden_size, dtype=torch.bfloat16),
                                          requires_grad=False)

        self.oga = OGA(self.config.vision_channel)
        self.mm_projector = MMProjector(self.config.vision_channel, self.config.language_model.hidden_size)

        self.language_model.requires_grad_(False)
        # non-reentrant checkpointing for both bases (prepare_model_for_kbit_training only runs for the 4-bit one):
        # the LoRA weights inside the frozen, checkpointed layers get gradients without enable_input_require_grads
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))

        self.mask_emb = nn.Sequential(
            nn.Conv2d(self.config.mask_cls_num, self.config.mask_emb_dim, kernel_size=3, padding=1),
            LayerNorm2d(self.config.mask_emb_dim),
            nn.ReLU(),
            nn.Conv2d(self.config.mask_emb_dim, self.config.mask_emb_dim, kernel_size=3, padding=1),
            LayerNorm2d(self.config.mask_emb_dim),
            nn.ReLU()
        )

        peft_config = LoraConfig(
            bias="none",
            target_modules=find_all_linear_names(self.language_model),
            inference_mode=False, r=self.config.lora.r, lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            task_type="CAUSAL_LM",
        )
        self.language_model = get_peft_model(self.language_model, peft_config)

        self.to(torch.bfloat16)
        self._prompt_mismatch_warned = False
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        trainable_params, all_param = self.get_nb_trainable_parameters()
        rank0_print(f"EarthVLNet: trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
                    f"trainable%: {100 * trainable_params / all_param:.4f}")

    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def pre_vision_embed(self, img_feat, pred_mask):
        pred_mask = pred_mask.unsqueeze(1).to(device)
        pred_mask = F.interpolate(pred_mask.float(), size=img_feat.shape[2:4], mode='nearest').squeeze(1).long()
        pred_mask[pred_mask < 0] = 0
        pred_mask = F.one_hot(pred_mask, num_classes=self.config.mask_cls_num).permute([0, 3, 1, 2]).to(torch.bfloat16)
        img_feat = torch.cat([img_feat, self.mask_emb(pred_mask)], dim=1).to(torch.bfloat16)

        return img_feat, pred_mask

    @staticmethod
    def build_prompt(question):
        """The generation prompt; the training text appends `` <answer> </s>``."""
        return SYSTEM_PROMPT + "USER: <image>\n%s ASSISTANT:" % question

    def build_labels(self, input_ids, attention_mask, prompts):
        """Next-token targets: the answer tokens and ``</s>``; the prompt (up to and including ``ASSISTANT:``) and the
        padding are set to ``ignore_index``.

        The original code located the token ``ANT`` (id 13566) of ``ASSISTANT`` and masked up to the following
        ``:``. Measuring the tokenized prompt gives identical labels (checked with the llava-hf llava-1.5-7b-hf and
        llava-v1.6-vicuna-7b-hf tokenizers on every question / answer of EarthVLSet, left and right padding) without
        depending on a vocabulary id.
        """
        ignore_index = self.config.language_model.ignore_index
        labels = input_ids.clone()
        prompt_ids = self.tokenizer(list(prompts), add_special_tokens=True)['input_ids']
        for i, ids in enumerate(prompt_ids):
            start = int(attention_mask[i].nonzero()[0])  # first non-padding token (left padding)
            n = len(ids)
            if not self._prompt_mismatch_warned and input_ids[i, start:start + n].tolist() != list(ids):
                rank0_print('EarthVLNet: warning: the prompt tokens differ inside the full text; the label mask '
                            'may be off by a few tokens.')
                self._prompt_mismatch_warned = True
            labels[i, :start + n] = ignore_index
        labels[labels == self.tokenizer.pad_token_id] = ignore_index
        return labels

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels=None):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.config.language_model.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.language_model.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.language_model.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.language_model.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not a text position needs filling
        # (transformers PR #29835; the original test "embedding still all zeros" also caught text tokens whose
        # embedding row is zero, e.g. a zero-initialised <pad>, and is otherwise identical)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.config.language_model.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def llm_forward(self,
                    input_ids: torch.LongTensor = None,
                    pixel_values: torch.FloatTensor = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    labels: Optional[torch.LongTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_features = pixel_values
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.language_model.ignore_index).to(torch.long)

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def encode_image(self, x, seg_mask):
        """Visual tokens [bs, h * w, hidden_size] from the stage-1 features and the predicted mask."""
        img_feat, _ = self.pre_vision_embed(x.to(device), seg_mask.to(device))  # bs, c + mask_emb_dim, h, w
        img_feat = self.oga(img_feat)
        bs, channels, h, w = img_feat.shape
        img_feat = img_feat.permute([0, 2, 3, 1]).reshape(bs, -1, channels)  # bs x (h * w) x c
        return self.mm_projector(img_feat)  # bs, h * w, language_hidden_size

    def forward(self, x, y=None):
        ques, ans, seg_mask = y['question'], y['answer'], y['seg_mask']
        prompts = [self.build_prompt(q) for q in ques]
        img_feat = self.encode_image(x, seg_mask)

        if self.training:
            texts = ['%s %s </s>' % (prompt, a) for prompt, a in zip(prompts, ans)]
            inputs = self.language_processer(text=texts, return_tensors="pt", padding='longest')
            labels = self.build_labels(inputs.input_ids, inputs.attention_mask, prompts)
            out = self.llm_forward(input_ids=inputs.input_ids.to(device),
                                   attention_mask=inputs.attention_mask.to(device),
                                   pixel_values=img_feat, labels=labels.to(device))
            return dict(generation_loss=out.loss)

        # greedy decoding from left-padded prompts
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            pred_inputs = self.language_processer(text=prompts, return_tensors="pt", padding='longest')
        finally:
            self.tokenizer.padding_side = old_padding_side
        input_ids = pred_inputs.input_ids.to(device)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, _, _ = self._merge_input_ids_with_image_features(
            img_feat, inputs_embeds, input_ids, pred_inputs.attention_mask.to(device)
        )
        outputs = self.language_model.generate(do_sample=False, max_new_tokens=self.config.max_new_tokens,
                                               use_cache=True, inputs_embeds=inputs_embeds,
                                               attention_mask=attention_mask)
        generated_text = self.language_processer.batch_decode(outputs, skip_special_tokens=True)
        return [text.strip(' ') for text in generated_text]

    def set_default_config(self):
        self.config.update(dict(
            pretrained_model_dir='./pretrain/vicuna-7b',
            vision_channel=1536 + 128,  # ConvNeXt-L stage-4 features + mask embedding
            mask_emb_dim=128,
            mask_cls_num=8,
            load_in_4bit=False,  # bf16 base as in the released run; True: QLoRA (4-bit NF4, needs bitsandbytes)
            attn_implementation='flash_attention_2',  # as in the original run; 'sdpa' or 'eager' without flash-attn
            max_new_tokens=256,
            lora=dict(r=64, lora_alpha=16, lora_dropout=0.05),
            language_model=dict(
                pad_token_id=32001,
                hidden_size=4096,
                image_token_index=32000,
                ignore_index=-100,
            ),
        ))


@er.registry.MODEL.register()
class EarthVLCount(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        from transformers import AutoTokenizer

        self.language_processer = AutoTokenizer.from_pretrained(self.config.pretrained_model_dir)
        self.counting_part = Transformer(**self.config.count_cfg)

        self.mm_projector = MMProjector(self.config.vision_channel, self.config.text_hidden_size)
        self.oga = OGA(self.config.vision_channel)

        self.mask_emb = nn.Sequential(
            nn.Conv2d(8, self.config.mask_emb_dim, kernel_size=3, padding=1, stride=1),
            LayerNorm2d(self.config.mask_emb_dim),
            nn.GELU(),
            nn.Conv2d(self.config.mask_emb_dim, self.config.mask_emb_dim, kernel_size=3, padding=1, stride=1),
            LayerNorm2d(self.config.mask_emb_dim),
            nn.GELU(),
        )
        self.vis_pos_emb = positionalencoding1d(self.config.vision_channel, 32*32)
        self.lang_pos_emb = positionalencoding1d(self.config.text_hidden_size, 30)

        self.counting_input_embedding = nn.Embedding(32128, self.config.text_hidden_size)

        self.nd_loss = NumericalLoss(**self.config.loss)

    def pre_vision_embed(self, img_feat, pred_mask):
        pred_mask = pred_mask.unsqueeze(1).to(device)
        pred_mask = F.interpolate(pred_mask.float(), size=img_feat.shape[2:4], mode='nearest').squeeze(1).long()
        pred_mask[pred_mask < 0] = 0
        pred_mask = F.one_hot(pred_mask, num_classes=8).permute([0, 3, 1, 2]).float()
        pred_mask = self.mask_emb(pred_mask)

        img_feat = torch.cat([img_feat, pred_mask], dim=1)

        return img_feat, pred_mask

    def forward(self, x, y=None):
        ques, ans, seg_mask = y['question'], y['answer'], y['seg_mask']

        # target: the number class of the answer (module.metric.find_pos_index), -1 if the answer has no number
        counting_targets = []
        for ans_i_str in ans:
            pos_list, _ = find_pos_index(ans_i_str, tag='#')
            pos_list += [DEFAULT_INDEX] * (self.config.counting_candidate - len(pos_list))
            counting_targets.append(pos_list[0])

        x = x.to(device)
        inputs = self.language_processer(text=ques, return_tensors="pt", padding='max_length', truncation=True, max_length=30).to(device)

        input_ids = inputs.input_ids
        img_feat, pred_mask = self.pre_vision_embed(x, seg_mask)  # bs, c + mask_emb_dim, h, w
        count_img_feat = self.oga(img_feat)

        bs, hidden_size, h, w = count_img_feat.shape
        count_img_feat = count_img_feat.permute([0, 2, 3, 1]).reshape(bs, -1, hidden_size) + self.vis_pos_emb.to(device)  # bs x 1024 x hidden_size
        count_img_feat = self.mm_projector(count_img_feat)

        count_inputs_embeds = self.counting_input_embedding(input_ids) + self.lang_pos_emb.to(device)

        count_inputs_embeds = torch.cat([count_img_feat, count_inputs_embeds], dim=1)
        count_logits = self.counting_part(count_inputs_embeds)  # bs, num_classes

        if self.training:
            counting_targets = torch.Tensor(counting_targets).long().to(device)
            valid = counting_targets != -1

            # dict of the loss ('numerical_loss', trained) and statistics (logged only)
            return self.nd_loss(count_logits[valid], counting_targets[valid])

        numerical_idx = count_logits.argmax(dim=1)
        return [Reverse_Lookup[num_idx] for num_idx in numerical_idx]

    def set_default_config(self):
        # the configuration of the released counting checkpoint
        self.config.update(dict(
            pretrained_model_dir='google/flan-t5-xl',  # only its tokenizer is used; a local copy also works
            vision_channel=1536 + 128,
            text_hidden_size=384,
            counting_candidate=1,
            mask_emb_dim=128,
            count_cfg=dict(
                dim=384,
                depth=3,
                heads=4,
                mlp_dim=384,
                dim_head=384,
                dropout=0.,
            ),
            loss=dict(
                uper_numer_idx=0,
                alpha=1.,
                keep_ratio=0.8,
                gamma=0.5,
                min_keep=1,
            ),
        ))
