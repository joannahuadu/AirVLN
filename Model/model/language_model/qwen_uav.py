from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import ModelOutput
from llamavid.qwen2 import Qwen2Model, Qwen2ForCausalLM
from transformers.utils import add_start_docstrings_to_model_forward, logging

from llamavid.model.llamavid_arch import LLaMAVIDMetaModel, LLaMAVIDMetaForCausalLM

logger = logging.get_logger(__name__)


@dataclass
class CausalLMOutputWithPastUAVMulLoss(ModelOutput):
    ori_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    x_loss: Optional[torch.FloatTensor] = None
    y_loss: Optional[torch.FloatTensor] = None
    z_loss: Optional[torch.FloatTensor] = None
    help_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    is_help_loss: Optional[torch.FloatTensor] = None

@dataclass    
class CausalLMOutputWithPastUAV(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class QwenUAVForCausalLM(Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
