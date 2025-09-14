import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space

from Model.policy import ILPolicy
from Model.encoders.resnet_encoders import TorchVisionResNet50, TorchVisionResNet50Place365, VlnResnetDepthEncoder
from Model.encoders.instruction_encoder import InstructionEncoder, InstructionBertEncoder
from Model.encoders.rnn_state_encoder import build_rnn_state_encoder
from Model.aux_losses import AuxLosses

from src.common.param import args

from .model import *


class LLaVAUAVPolicy(ILPolicy):
    #
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        out_model_config=None,
        device=torch.device("cpu"),
    ):
        super().__init__(
            LlavaLlamaAttForCausalLM(
                observation_space=observation_space,
                num_actions=action_space.n,
                out_model_config=out_model_config,
                device=device,
            ),
            action_space.n,
        )

    #
    @classmethod
    def from_config(
        cls, observation_space: Space, action_space: Space, out_model_config=None,
        device=torch.device("cpu"),
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            out_model_config=out_model_config,
            device=device,
        )

