# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    background_color: Literal["random", "last_sample", "black", "white"] = "black"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 128
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 20
    """Size of the hashmap for the base mlp"""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = True
    """Whether to disable scene contraction or not."""


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = TCNNNerfactoField(
            num_images=23,
            aabb=self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            hidden_dim_color=self.config.hidden_dim_color,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,

            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            use_pred_normals=self.config.predict_normals,
        )
        self.image_data_flag = False


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        # param_groups["density_fields"] = list(self.field.mlp_base.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes,
        ):
        callbacks = []
        
        return callbacks

    def get_outputs(self, ray_samples: RaySamples):
      
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)

        # weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])    
        # rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        if self.image_data_flag:
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])  
            
            outputs = {
            "rgb_pred":field_outputs[FieldHeadNames.RGB],
            "density_pred": field_outputs[FieldHeadNames.DENSITY],
            "weights_pred":weights
            }  
            return outputs

        outputs = {
            "rgb_fields":field_outputs[FieldHeadNames.RGB],
            "density_fields": field_outputs[FieldHeadNames.DENSITY],
        }

        return outputs

    

