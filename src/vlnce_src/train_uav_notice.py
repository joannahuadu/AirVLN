# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# ------------------------------------------------------------------------
import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import pickle
import math
import time

import torch

import transformers

# import llamavid.qwen2
from llamavid.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, WAYPOINT_INPUT_TOKEN, WAYPOINT_LABEL_TOKEN, DEFAULT_WP_TOKEN, DEFAULT_HISTORY_TOKEN, WP_TOKEN_INDEX, HIS_TOKEN_INDEX
from torch.utils.data import Dataset
from llamavid.train.llava_trainer import LLaVATrainer

from llamavid import conversation as conversation_lib
# from llamavid.model import *
import sys
sys.path.append(os.getcwd())
from Model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import numpy as np
from decord import VideoReader, cpu


import lmdb
import tqdm
import msgpack_numpy
from transformers.utils import logging
from transformers import AutoProcessor
import torch.nn.functional as F
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def rotation_matrix_from_vector(x, y):
    v_x = np.array([x, y, 0])
    v_x = v_x / np.linalg.norm(v_x)
    v_y = np.array([-v_x[1], v_x[0], 0])
    v_y = v_y / np.linalg.norm(v_y)
    v_z = np.array([0, 0, 1])
    rotation_matrix = np.column_stack((v_x, v_y, v_z))
    return rotation_matrix

def transform_point(point, rotation_matrix):
    return np.dot(point, rotation_matrix)

def waypoint2angle(waypoints):
    angle_and_norm = []
    for waypoint in waypoints:
        norm = np.linalg.norm(waypoint)
        angle = waypoint / (norm + 1e-6)
        angle_and_norm.append([angle[0], angle[1], angle[2], norm])
    return np.array(angle_and_norm)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_waypoint_predictor: bool = field(default=True)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    bert_type: Optional[str] = field(default="qformer_pretrain")
    num_query: Optional[int] = field(default=32)
    pretrain_qformer: Optional[str] = field(default=None)
    compress_type: Optional[str] = field(default=None)
    use_angle_and_norm_loss: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data json."})
    dataset_path: str = field(default=None,
                           metadata={"help": "Path to the raw data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    video_token: Optional[int] = field(default=2)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=True)
    inflection_weight_coef: float = field(default=1.9)
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    lr_multi: Optional[str] = field(default=None)

def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]

class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_args: DataArguments,
        tokenizer: transformers.PreTrainedTokenizer,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=5.0e12,
        batch_size=1,
    ):
        super().__init__()
        list_data_dict = json.load(open(data_args.data_path, "r"))

        self.list_data_dict = list_data_dict
        self.tokenizer = tokenizer
        self.lmdb_features_dir = data_args.dataset_path
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        self.keys = []
        self.seed = 1

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
            readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                self.keys.append(key.decode())

        self.length = len(self.keys)

        self.iter_start = 0
        self.iter_end = self.length
        self.processor = data_args.image_processor
        self.data_args = data_args
        logger.warning("END init Dataset \t start({}) - end({})".format(self.iter_start, self.iter_end))
    
    def __len__(self):
        return sum(len(item['reference_path']) for item in self.list_data_dict['episodes'])
    
    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            sources = []
            source_preload = [] 
            lengths = 0
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=True, 
                readahead=False
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i+1) % 10 == 0:
                        if self.worker_info is not None:
                            logger.info("{} lmdb load: {} / {}".format(self.worker_info.id, i+1, self.preload_size))
                        else:
                            logger.info("{} lmdb load: {} / {}".format(0, i+1, self.preload_size))
                    episode_id = self.keys[self.load_ordering[-1]]
                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.keys[self.load_ordering.pop()]).encode()),
                            raw=False,
                        )
                    )
                    sources.append(next(item for item in self.list_data_dict['episodes'] if item['episode_id'] == episode_id))
                
            for source, new in zip(sources, new_preload):
                for frame in range(len(source['reference_path'])):
                    src = {}
                    src['episode_id'] = source['episode_id']
                    src['rgb'] = new[0]['rgb'][frame]
                    src['depth'] = new[0]['depth'][frame]
                    src['conversations'] = source['instruction']['instruction_text']
                    assert len(source['actions']) == len(source['reference_path'])
                    assert np.all(source['actions'] == new[2])
                    src['length'] = len(source['reference_path'])
                    src['actions'] = source['actions']
                    src['reference_path'] = source['reference_path']
                    src['frame'] = frame
                    
                    source_preload.append(src)
                
                    lengths+=1

            for idx in _block_shuffle(list(range(0, lengths)), self.batch_size):
                self._preload.append(source_preload[idx])

            del sources, new_preload, source_preload

        return self._preload.pop()

    def get_stage(self, trajectory, frame_num):
        def turning_stage(p0,p1,p2):
            prev_vec = p1 - p0
            now_vec = p2 - p1
            delta_angle = np.arccos(np.dot(prev_vec, now_vec) / (np.linalg.norm(prev_vec)+ 1e-6) / (np.linalg.norm(now_vec)+ 1e-6)) * 180 / np.pi
            if delta_angle > 25 and delta_angle < 120:
                if int(np.cross(prev_vec, now_vec)) > 0:
                    return 'right'
                else:
                    return 'left'
            return 'cruise'
        assist = 0
        trajectory = np.asarray(trajectory)
        z_values = trajectory[:, 2]
        now_z = z_values[frame_num - 1]
        future_z = z_values[min(frame_num+2, len(z_values)-1)]
        stage = 'cruise'
        if now_z - future_z > 5:
            stage = 'take off'
        elif now_z - future_z < -5:
            stage = 'landing'
        prev_vec = np.array([0,0,0])
        if frame_num >= 2 and frame_num < len(trajectory):
            prev_vec =  np.array(trajectory[frame_num - 1, :3] - trajectory[frame_num - 2, :3])
            if stage == 'cruise':
                stage = turning_stage(trajectory[frame_num - 2 ,:2], trajectory[frame_num - 1, :2], trajectory[frame_num, :2])
        if frame_num >= 1 and frame_num < len(trajectory) - 1:
            future_p = trajectory[frame_num + 1, :2]
            next_p = trajectory[frame_num, :2]
            next_stage = turning_stage(trajectory[frame_num-1, :2], next_p, future_p)
            future_z = z_values[min(frame_num+3, len(z_values)-1)]
            if trajectory[frame_num, 2] - future_z < -5:
                next_stage = 'landing'
            if next_stage == 'left' or next_stage == 'right' or next_stage == 'landing':
                assist = 1
        return stage, prev_vec, assist

    def __next__(self):
        sources = self._load_next()
        ori_sources = copy.deepcopy(sources)
        frame_num = sources['frame']
        image = sources['rgb']
        ori_image = Image.fromarray(image)
        image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        stage, future_delta, assist = self.get_stage(sources['reference_path'], frame_num)
        cur_pos = sources['reference_path'][frame_num - 1][:3]
        x, y = ori_sources['reference_path'][-1][0], ori_sources['reference_path'][-1][1]
        rotation_matrix = rotation_matrix_from_vector(x, y)
        future_delta =  transform_point(future_delta, rotation_matrix)
        future_delta = future_delta / (np.linalg.norm(future_delta) + 1e-8)
        future_delta_str = ','.join([str(round(x, 1)) for x in future_delta])
        
        cur_pos = transform_point(cur_pos, rotation_matrix)
        cur_pos_str = ','.join([str(round(x, 1)) for x in cur_pos])

        
        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in [sources]]),
            self.data_args, stage=stage, delta = future_delta_str, cur = cur_pos_str)
        
        has_image = (image is not None)
        data_dict = preprocess(
            sources,
            ori_image,
            self.tokenizer,
            has_image=has_image,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt)
        
        if 'prompt' in data_dict:
            prompt = data_dict['prompt']
        else:
            prompt = None

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
            
        data_dict['image'] = image
        trajectory_data = np.array(ori_sources['reference_path'])
        history_waypoint = trajectory_data[0:frame_num, 0:3]
        waypoint = trajectory_data[frame_num:min(ori_sources['length'], frame_num + 7), 0:3]
        if len(waypoint) == 0:
            waypoint = np.array([history_waypoint[-1] for i in range(7)])
        elif len(waypoint) < 7:
            waypoint = np.array([waypoint[i] if i < len(waypoint) else waypoint[-1] for i in range(7)])

        waypoint = waypoint - history_waypoint[-1]
        x, y = ori_sources['reference_path'][-1][0], ori_sources['reference_path'][-1][1]
        rotation_matrix = rotation_matrix_from_vector(x, y)
        history_waypoint = transform_point(history_waypoint, rotation_matrix)
        waypoint = transform_point(waypoint, rotation_matrix)
        
        use_angle = True
        if use_angle:
            waypoint = waypoint2angle(waypoint)
        
        data_dict['history_waypoint'] = torch.tensor(history_waypoint).view(-1)
        data_dict['waypoint'] = torch.tensor(waypoint[0]).view(-1)
        orientation = trajectory_data[frame_num-1, 3:6]
        data_dict['orientation'] = torch.tensor(orientation).view(-1)
        data_dict['is_help'] = torch.tensor(assist).view(-1)
        data_dict['action'] = torch.tensor(ori_sources['actions'][frame_num]).view(-1)
        ## TODO: wmq. depth?
        
        # prompt exist in the data
        if prompt is not None:
            data_dict['prompt'] = prompt
        
        return data_dict

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.worker_info = worker_info
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self
    
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'vlm_att']
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'vlm_att', 'action_emb', 'actions_fc', 'actions_predictor',
                         'actions_output', 'history_predictor', 'history_preprocessor', 'is_help_predictor'] # end_predictor
    
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        # keys_to_match = ['mm_projector']
        keys_to_match = ['mm_projector', 'vision_resampler', 'vlm_att', 'action_emb', 'actions_fc', 'actions_predictor',
                         'actions_output', 'history_predictor', 'history_preprocessor', 'is_help_predictor', 'embed_tokens'] # 'end_predictor',
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
def smarter_tokenizer_and_embedding_resize(
    special_tokens_list: List,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_tokens(special_tokens_list, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        model.get_input_embeddings().weight.requires_grad_(True)
        model.get_output_embeddings().weight.requires_grad_(True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    stage = None,
    delta = None,
    cur = None
) -> Dict:
    """
        process image token's representation
    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    if conversation_lib.default_conversation.version.startswith("imgsp_uav"):
        for i, source in enumerate(sources):
            sources[i] = [{'from': 'human', 'value': '<image>\n'+ source}, {'from': 'gpt', 'value': ''}]
            for sentence in sources[i]:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['prompt'] = copy.deepcopy(sentence['value'])
                    sentence['value'] = '\n\nStage:' + stage + '\n\nPrevious displacement:' + delta  + '\n\nCurrent position:' + cur + '\n\nCurrent image:' + DEFAULT_IMAGE_TOKEN + '\n\nInstruction:' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
                replace_token = DEFAULT_IMAGE_TOKEN
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    elif conversation_lib.default_conversation.version.startswith("imgsp_qwen"):
        for i, sentence in enumerate(sources):
            sources[i] = '\n\nStage:' + stage + '\n\nPrevious displacement:' + delta  + '\n\nCurrent position:' + cur + '\n\nInstruction:' + sentence
    
    return sources


def preprocess_multimodal_movie(
    sources: Sequence[str],
    data_args: DataArguments,
    video_inputs: str
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                prompt = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            replace_token = video_inputs
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources, prompt


def preprocess_imgsp_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    img_token: str = '<image>',
    refine_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    guided_prompt = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        img_in_text = False
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            
            # add guided prompt
            if role==conv.roles[0]:
                guided_sent = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
                if refine_prompt:
                    # only keep the useful part of the prompt
                    if '\n' in guided_sent:
                        for _sent in guided_sent.split('\n'):
                            if '?' in _sent:
                                guided_sent = _sent
                                break
                guided_prompt.append(guided_sent)
            # check if image token in text
            if img_token in sentence["value"]:
                img_in_text = True
            # add image token to all sentence if multimoal input
            if role==conv.roles[0] and img_in_text and img_token not in sentence["value"]:
                # randomly add image token to the beginning or end of the sentence
                if random.randint(0,1)==0:
                    img_conv = img_token + '\n' + sentence["value"]
                else:
                    img_conv = sentence["value"] + '\n' + img_token
                
                conv.append_message(role, img_conv)
            else:
                conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                logger.warning(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        prompt=guided_prompt,
    )


def preprocess_imgsp_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: Image.Image,
    img_token: str = '<image>',
    refine_prompt: bool = False,
) -> Dict:
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    system_message = {
        "role": "system",
        "content": [
            {"type": "text", "text": "The assistant is a navigation model that output the uav waypoints according to the user's instructions."}
        ]
    }
    text = []
    for i, sentence in enumerate(sources):
        messages = [
            system_message, 
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sentence},
                ],
            },
        ]
        text.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    input = processor(text=text, images=has_image)
    input_ids = input.input_ids
    input_ids_pad_wp = torch.zeros(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long)
    input_ids_pad_wp[:, :-2] = input_ids[:, :-1]
    input_ids_pad_wp[:, -2] = WAYPOINT_INPUT_TOKEN
    input_ids_pad_wp[:, -1] = input_ids[:, -1]
    input.input_ids = input_ids_pad_wp
    
    targets = input_ids.clone()
    targets[:, :] = IGNORE_INDEX

    targets_pad_wp = torch.zeros(targets.shape[0], targets.shape[1] + 1, dtype=torch.long)
    targets_pad_wp[:, :-2] = targets[:, :-1]
    targets_pad_wp[:, -2] = WAYPOINT_LABEL_TOKEN
    targets_pad_wp[:, -1] = targets[:, -1]

    return dict(
        **input,
        labels=targets_pad_wp,
        prompt=sources, #list len=1
    )

def preprocess_imgsp_uav(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    img_token: str = '<image>',
    refine_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    guided_prompt = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        img_in_text = False
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            
            # add guided prompt
            if role==conv.roles[0]:
                guided_sent = sentence["prompt"].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
                if refine_prompt:
                    # only keep the useful part of the prompt
                    object_description = guided_sent.split('degrees from you.')[-1].replace('Please control the drone and find the target.', '').strip()
                    guided_sent = 'Please pay attention to the obstacles in images and approach the object described below: ' + object_description

                guided_prompt.append(guided_sent)
            # check if image token in text
            if img_token in sentence["value"]:
                img_in_text = True
            # add image token to all sentence if multimoal input
            if role==conv.roles[0] and img_in_text and img_token not in sentence["value"]:
                # randomly add image token to the beginning or end of the sentence
                img_conv = img_token + '\n' + sentence["value"]
                
                conv.append_message(role, img_conv)
            else:
                conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # add wp embedding, input_ids[-1] is </s>, 
    input_ids_pad_wp = torch.zeros(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long)
    input_ids_pad_wp[:, :-2] = input_ids[:, :-1]
    input_ids_pad_wp[:, -2] = WAYPOINT_INPUT_TOKEN
    input_ids_pad_wp[:, -1] = input_ids[:, -1]
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX

    # add wp embedding, input_ids[-1] is </s>
    targets_pad_wp = torch.zeros(targets.shape[0], targets.shape[1] + 1, dtype=torch.long)
    targets_pad_wp[:, :-2] = targets[:, :-1]
    targets_pad_wp[:, -2] = WAYPOINT_LABEL_TOKEN
    targets_pad_wp[:, -1] = targets[:, -1]
    
    # print(input_ids_pad_wp)
    return dict(
        input_ids=input_ids_pad_wp,
        labels=targets_pad_wp,
        prompt=guided_prompt,
    )
 

def preprocess(
    sources: Sequence[str],
    images: Image.Image,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    prompt: str = None,
    refine_prompt: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith("imgsp_uav"):
        return preprocess_imgsp_uav(sources, tokenizer, has_image=has_image, refine_prompt=refine_prompt)
    elif conversation_lib.default_conversation.version.startswith("imgsp_qwen"):
        return preprocess_imgsp_qwen(sources, tokenizer, has_image=images, refine_prompt=refine_prompt)
    elif conversation_lib.default_conversation.version.startswith("imgsp"):
        return preprocess_imgsp_v1(sources, tokenizer, has_image=has_image, refine_prompt=refine_prompt)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ## TODO: wmq. Qwen?
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # TODO: maybe all list is a good thing. wmq: No! all list is not a good thing for arivln iterable dataset
            if all(x is not None and x.shape == images[0].shape for x in images) and len(images) > 1:
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # if 'prompt' in instances[0]:
            # batch['prompts'] = [instance['prompt'] for instance in instances]
        
        if 'waypoint' in instances[0]:
            batch['waypoints'] = torch.stack([instance['waypoint'] for instance in instances])
            batch['historys'] = [instance['history_waypoint'] for instance in instances]
            batch['history_lengths'] = torch.tensor([len(his) for his in batch['historys']])
            max_length = batch['history_lengths'].max()
            batch['historys'] = torch.stack([
                F.pad(his, (0, max_length - len(his)), value=0) for his in batch['historys']
            ])
        
        if 'orientation' in instances[0]:
            batch['orientations'] = torch.stack([instance['orientation'] for instance in instances])
        
        if 'end' in instances[0]:
            batch['ends'] = torch.stack([instance['end'] for instance in instances]).squeeze()
        
        if 'action' in instances[0]:
            batch['actions'] = torch.stack([instance['action'] for instance in instances]).squeeze()
        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = IWTrajectoryDataset(
                tokenizer=tokenizer,
                data_args=data_args,
                use_iw=True,
                inflection_weight_coef=float(data_args.inflection_weight_coef),
                lmdb_map_size=5.0e12,
                batch_size=training_args.per_device_train_batch_size,
            )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = dict(
        torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
    )
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}

    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    if "llava" in model_args.model_name_or_path:
        ModelClass = LlavaUAVForCausalLM
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        ModelClass = QwenVLUAVForCausalLM
    elif "llama" in model_args.model_name_or_path or 'vicuna' in model_args.model_name_or_path:
        ModelClass = LlavaLlamaAttForCausalLM
    elif "Qwen" in model_args.model_name_or_path:
        ModelClass = LlavaQwenAttForCausalLM
        # config._attn_implementation = 'eager'
    else:
        raise ValueError(f"Unknown model type: {model_args.model_name_or_path}")

    model = ModelClass.from_pretrained(
        model_args.model_name_or_path,
        use_angle_and_norm_loss=model_args.use_angle_and_norm_loss,
        config=config,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            layers_to_transform=[i for i in range(0, config.num_hidden_layers)], 
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        else: #TODO: wmq. NOT SURE!
            tokenizer.unk_token = "<unk>"
            # tokenizer.pad_token = tokenizer.unk_token
            # tokenizer.add_special_tokens({"unk_token": "<unk>"})
            # model.resize_token_embeddings(len(tokenizer))
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        if not ("llava" in model_args.model_name_or_path or "Qwen2.5-VL" in model_args.model_name_or_path):
            model.get_model().initialize_vision_modules(
                model_args=model_args,
                fsdp=training_args.fsdp,
                max_token=training_args.model_max_length
            )
        else:
            model.get_model().initialize_vision_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )
            
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        
    smarter_tokenizer_and_embedding_resize(special_tokens_list=['<wp>', '<his>'], tokenizer=tokenizer, model=model)
    
    model.get_special_token_id({'<wp>': tokenizer.encode('<wp>', add_special_tokens=False)[0], '<his>': tokenizer.encode('<his>', add_special_tokens=False)[0],
                                ',': tokenizer.encode(',', add_special_tokens=False)[0], ';': tokenizer.encode(';', add_special_tokens=False)[0]})

    # all the attention modules require grad
    if not ("llava" in model_args.model_name_or_path or "Qwen2.5-VL" in model_args.model_name_or_path):
        model.get_model().initialize_attention_modules(model_args)
    
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args, training_args=training_args)
    
    if model_args.tune_waypoint_predictor:
        for p in model.action_emb.parameters():
            p.requires_grad = True
        for p in model.actions_fc.parameters():
            p.requires_grad = True
        for p in model.actions_output.parameters():
            p.requires_grad = True
        for p in model.history_preprocessor.parameters():
            p.requires_grad = True

    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        if param.requires_grad:
            print(f"Parameter name: {name}, Parameter shape: {param.shape}")
    
    model.print_trainable_parameters()
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        logger.info("saving model...")
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        logger.info("saved.")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

        
if __name__ == "__main__":
    train()
