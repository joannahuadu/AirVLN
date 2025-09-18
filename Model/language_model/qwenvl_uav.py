#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from llamavid.model.language_model.llama_uav import CausalLMOutputWithPastUAV, CausalLMOutputWithPastUAVMulLoss
from llamavid.constants import WAYPOINT_LABEL_TOKEN

class QwenConfig(Qwen2_5_VLConfig):
    model_type = "qwenvl_uav"

class QwenVLUAVForCausalLM(Qwen2_5_VLForConditionalGeneration):
    config_class = QwenConfig
    def __init__(self, config, **model_args):
        super(Qwen2_5_VLForConditionalGeneration, self).__init__(config)
        self.use_angle_and_norm_loss = model_args.get('use_angle_and_norm_loss', True)
        self.action_emb = nn.Embedding(1, config.hidden_size)
        self.actions_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 64),
        )
        self.actions_output = nn.Linear(64, 8)
        
        self.history_preprocessor = nn.Sequential(
            nn.Linear(3, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )
        
        self.actions_loss_func = torch.nn.CrossEntropyLoss()
        self.action_loss_scale = 1.0
        self.special_token_dict = None

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_special_token_id(self, special_token_dict):
        self.special_token_dict = special_token_dict
        
    def get_model(self):
        return self.model
    
    def forward_action(self, hidden_states):
        bs, hidden_size = hidden_states.size()
        actions_feature = self.actions_fc(hidden_states.reshape(-1, hidden_size))
        
        predicted_actions = self.actions_output(actions_feature)
        return predicted_actions

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        actions: Optional[torch.FloatTensor] = None,
        orientations: Optional[torch.FloatTensor] = None,
        historys: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        return_actions: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPastUAV]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not self.training:
            if input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)
            if attention_mask.device != self.device:
                attention_mask = attention_mask.to(device=self.device)
            if labels.device != self.device:
                labels = labels.to(device=self.device)
        history_embeds = []
        
        for idx in range(len(historys)):
            history = historys[idx]
            info = history.view(-1, 3)
            history_embed = self.history_preprocessor(info)
            history_embeds.append(history_embed)

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        inputs_embeds = inputs_embeds.to(dtype=self.action_emb.weight.dtype)
        inputs_embeds[labels == WAYPOINT_LABEL_TOKEN] = self.action_emb.weight
        
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        if output_attentions and  "save_attentions" in kwargs:
            torch.save(outputs.attentions, kwargs["save_attentions"])
            
        hidden_states = outputs[0]
        actions_feat = hidden_states[labels == WAYPOINT_LABEL_TOKEN]     
        predicted_actions = self.forward_action(actions_feat)
        
        if actions is None and return_actions:
            return predicted_actions
        
        loss = None
        
        assert len(torch.where(labels == WAYPOINT_LABEL_TOKEN)[0]) == actions.shape[0]
        if actions is not None:
            loss = self.actions_loss_func(predicted_actions, actions) 
        
        if return_actions:
            return loss, predicted_actions
        
        if not return_dict:
            output = (actions_feat,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPastUAVMulLoss(
            loss=loss,
        )

AutoConfig.register("qwenvl_uav", QwenConfig)
AutoModelForCausalLM.register(QwenConfig, QwenVLUAVForCausalLM)
