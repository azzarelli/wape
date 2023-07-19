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
Implementation of Instant NGP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import nerfacc
import torch
from nerfacc import ContractionType
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.instant_ngp_field import TCNNInstantNGPField

from nerfstudio.models.base_model import Model, ModelConfig


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: NGPModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    max_num_samples_per_ray: int = 24
    """Number of samples in field evaluation."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 20
    """Size of the hashmap for the base mlp"""
    contraction_type: ContractionType = ContractionType.AABB
    """Contraction type used for spatial deformation of the field."""
    cone_angle: float = 0.00
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "black"
    """The color that is given to untrained areas."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 128
    """Dimension of hidden layers for color network"""

class NGPModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: InstantNGPModelConfig
    field: TCNNInstantNGPField

    def __init__(self, config: InstantNGPModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.field = TCNNInstantNGPField(
            num_images=23,
            aabb=self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            hidden_dim_color=self.config.hidden_dim_color,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            use_appearance_embedding=self.config.use_appearance_embedding,
            
            contraction_type=self.config.contraction_type,
        )
        self.image_data_flag = False
        
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_samples: RaySamples):
        assert self.field is not None
        
        field_outputs = self.field(ray_samples)

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