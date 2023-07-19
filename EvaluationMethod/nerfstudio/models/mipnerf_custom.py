"""
MODIFIED Implementation of Mip-NeRF
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss

from nerfstudio.models.base_model import Model, ModelConfig


class MipNerfModel(Model):
    """mip-NeRF model

    Args:
        config: MipNerf configuration to instantiate model
    """

    def __init__(
        self,
        config: ModelConfig,
        **kwargs,
    ) -> None:
        self.field = None
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field = NeRFField(
            position_encoding=position_encoding, direction_encoding=direction_encoding, use_integrated_encoding=True
        )

        self.image_data_flag = False
       
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_samples: RaySamples):

        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # First pass:
        field_outputs_coarse = self.field.forward(ray_samples)
        
        # Second pass:
        field_outputs_fine = self.field.forward(ray_samples)
        
        # weights_coarse = ray_samples.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        
        if self.image_data_flag:
            weights_fine = ray_samples.get_weights(field_outputs_fine[FieldHeadNames.DENSITY]) 

            outputs = {
            "rgb_pred": field_outputs_fine[FieldHeadNames.RGB],
            "density_pred": field_outputs_fine[FieldHeadNames.DENSITY],
            "weights_pred":weights_fine,
            }  
            return outputs
        else:
            outputs = {
                "rgb_first_fields": field_outputs_coarse[FieldHeadNames.RGB],
                "rgb_second_fields": field_outputs_fine[FieldHeadNames.RGB],
                "density_first_fields": field_outputs_coarse[FieldHeadNames.DENSITY],
                "density_second_fields": field_outputs_fine[FieldHeadNames.DENSITY],
            }
        return outputs

