"""
A custom pipeline for handling explicit RF training and eval
"""
from nerfstudio.model_components.renderers import RGBRenderer

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

import torch
from typing_extensions import Literal

from nerfstudio.data.datamanagers.gt_datamanager import GTDataManager
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    Pipeline,
    VanillaPipelineConfig,
)


@dataclass
class GTPipelineConfig(VanillaPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: GTPipeline)
    target_num_samples: int = 262144  # 1 << 18
    """The target number of samples to use for an entire batch of rays."""
    max_num_samples_per_ray: int = 1024  # 1 << 10
    """The maximum number of samples to be placed along a ray."""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""
    enable_collider:bool=False
    background_color: Literal["random", "last_sample", "black", "white"] = "black"
    """Whether to randomize the background color."""

class GTPipeline(Pipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    # pylint: disable=abstract-method

    config: GTPipelineConfig
    datamanager: GTDataManager
    dynamic_num_rays_per_batch: int

    def __init__(
        self,
        config: GTPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__()
        torch.manual_seed(42)

        self.config = config
        self.datamanager: GTDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        scene_box = SceneBox(aabb=torch.tensor([[-5., -5., -5.], [150., 150., 150.]], dtype=torch.float32))

        self._model = config.model.setup(
            scene_box=scene_box,
            num_train_data=self.datamanager.BSL.shape[0]
            # num_train_data=len(self.datamanager.train_dataset),
            # metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        self.renderer_rgb = RGBRenderer(background_color=config.background_color)
    
    def get_train_rays_per_batch(self) -> int:
        return 128

    def get_eval_rays_per_batch(self) -> int:
        return 128


    def get_train_loss_dict(self, step: int):
        """Handle a training step
        """
        data, eval_dict,_ = self.datamanager.next_train(step)
        if data is not None:
            model_outputs = self.model(data)
            # metrics_dict = self.datamanager.get_metrics_dict(model_outputs, eval_dict)
            loss_dict = self.datamanager.get_loss_dict(model_outputs, eval_dict)
        metrics_dict = {}
        return model_outputs, loss_dict, metrics_dict

  
    def get_eval_loss_dict(self, step: int):
        """Handle a training step to get additional metrics
        """
        self.eval()

        data, eval_dict, _ = self.datamanager.next_train(step)
        if data is not None:
            with torch.no_grad():
                model_outputs = self.model(data)
            metrics_dict = self.datamanager.get_metrics_dict(model_outputs, eval_dict)
            loss_dict = self.datamanager.get_loss_dict(model_outputs, eval_dict)

        self.train()
        return model_outputs, loss_dict, metrics_dict

    def get_eval_image_metrics_and_images(self, step: int):
        """Handles the image and WAPE evaluations

            Notes:
                Iterate through the image data loader and process the WAPE, PSNR, SSIM and LPIPS on all image data
        """
        self.eval()
        images_dict = {}
        metrics_dict = {
            'PSNR':0., 'SSIM':0., 'LPIPS':0. ,
            'WAPE_R':[],  'WAPE_D':[],
            # 'WAPE_G':0., 'WAPE_B':0.,
        }
        waper = []
        waped = []

        # Fetch list of data loaders
        #   each data loaders contains rays (and their image indexes) with the same size so they can be batched in processing
        #   therefore each loader has a different size
        nv,nr,nc = self.datamanager.im_labels['metadata']['nv'], self.datamanager.im_labels['metadata']['nr'],  self.datamanager.im_labels['metadata']['nc']
        IMAGES = torch.zeros(nv,nr,nc, 3).to(self.device)
        GTIMAGES = torch.zeros(nv,nr,nc, 3).to(self.device)
   
        loaders = self.datamanager.im_loader
        for loader in loaders:
            for (data, indexes) in loader:
                # Gather all the data
                o = data[..., 0:3]
                d = data[..., 3:6]
                s = data[..., 6].unsqueeze(-1)
                e = data[..., 7].unsqueeze(-1)
                pa = data[..., 13].unsqueeze(-1)
                ci = data[..., 14].unsqueeze(-1)

                # Define the Frustrums for each batch (this is ray-wise)
                frustums = Frustums(
                origins=o,  # same for each sample
                directions=d,  # same for each sample
                starts=s,  # earliest occuring start frustum
                ends=e,  # latest occuring end frustum
                pixel_area=pa,  # same for each sample in ray so pick first
                )

                deltas = (data[..., 7] - data[..., 6]).unsqueeze(-1)
                starts = data[..., 6].unsqueeze(-1)
                ends = data[..., 7].unsqueeze(-1)
                
                # print(deltas.shape, frustums.origins.shape, frustums.starts.shape)

                ray_samples = RaySamples(
                    frustums=frustums,
                    camera_indices=ci.int(),  # [..., 1]
                    deltas=deltas,  # [..., num_samples, 1]
                    spacing_starts=starts,  # [..., num_samples, 1]
                    spacing_ends=ends,  # [..., num_samples, 1]
                )
                
                rgb = data[..., 8:11]
                s = data[..., 11].unsqueeze(-1)
                w = data[..., 12].unsqueeze(-1)
                
                # Set the model into image eval mode 
                self.model.image_data_flag = True
                with torch.no_grad():
                    model_outputs = self.model(ray_samples)
                


                # RGB = self.renderer_rgb(model_outputs['rgb_pred'].unsqueeze(0), model_outputs['weights_pred'].unsqueeze(0)).squeeze(0)
                RGB = self.renderer_rgb(model_outputs['rgb_pred'].unsqueeze(0), w.unsqueeze(0)).squeeze(0)
                RGB_GT = self.renderer_rgb(rgb.unsqueeze(0), w.unsqueeze(0)).squeeze(0)
                
                IMAGES[indexes[...,0], indexes[...,1], indexes[...,2]] = RGB
                GTIMAGES[indexes[...,0], indexes[...,1], indexes[...,2]] = RGB_GT
                

                # To get the WAPE of novel views we can get the indices in our stackpointer (index) which correspond to
                #  the images in our list of desired novel-view images (self.datamanager.ignore_imgae_indexs)
                indexOFindexs_mask = torch.isin(indexes[..., 0], self.datamanager.ignore_image_indexs) 
                indexOFindexes_for_novelview_scene = (indexOFindexs_mask == True).nonzero(as_tuple=True)[0]  
                
                waped += (model_outputs['density_pred'][indexOFindexes_for_novelview_scene].flatten() - s[indexOFindexes_for_novelview_scene].flatten()).abs().tolist()
                waper += (model_outputs['rgb_pred'][indexOFindexes_for_novelview_scene].flatten() - rgb[indexOFindexes_for_novelview_scene].flatten()).abs().tolist()
               
        metrics_dict['WAPE_D'] = torch.tensor(waped).nan_to_num(nan=1.).mean()
        metrics_dict['WAPE_R'] = torch.tensor(waper).nan_to_num(nan=1.).mean()

        # Set data manager mode back to train
        self.model.image_data_flag = False

        num = 0.0
        for idx, (image, gtimage) in enumerate(zip(IMAGES, GTIMAGES)):
            images_dict[str(idx)] = image

            predimage = torch.moveaxis(image, -1, 0)[None, ...].nan_to_num()
            gtimage = torch.moveaxis(gtimage, -1, 0)[None, ...].nan_to_num()
            
            if idx in self.datamanager.ignore_image_indexs:
                metrics_dict['PSNR']  += self.datamanager.psnr(gtimage, predimage)
                metrics_dict['SSIM']  += self.datamanager.ssim(gtimage, predimage)
                metrics_dict['LPIPS'] += self.datamanager.lpips(gtimage, predimage)
                num += 1

        # Divide by number of images tested
        metrics_dict['PSNR'] = metrics_dict['PSNR']/num
        metrics_dict['SSIM'] = metrics_dict['SSIM']/num
        metrics_dict['LPIPS'] = metrics_dict['LPIPS']/num
            
        self.train()
        return metrics_dict, images_dict

    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict = {}
        self.train()
        return metrics_dict

    def get_param_groups(self):
        model_params = self.model.get_param_groups()
        return {**model_params}
