"""
Custom Datamanager for WAPE training and evaluation
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing importType, Union

import torch
import torchvision.transforms as transforms
from PIL import Image
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.model_components.losses import MSELoss, L1Loss
from nerfstudio.model_components.renderers import RGBRenderer

from torcheval.metrics.functional import binary_accuracy
from nerfstudio.utils import misc

@dataclass
class GTDataManagerConfig(DataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: GTDataManager)
    """Target class to instantiate."""
    train_batch_size: int = 256
    """The maximum number of samples to be placed along a ray."""
    test_batch_size: int = 128
    """The maximum number of samples to be placed along a ray."""
    image_batch_size: int = 128
    """The maximum number of samples to be placed along a ray."""
    tt_data_ratio: float = 0.99
    """The train-to-test data split ratio"""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    loss_rgb_scale: float = 1.
    """Scaling the rgb loss function"""
    loss_density_scale: float = 1.
    """Scaling the density loss function"""
    loss_secondary_rgb_scale: float = 1.
    """Scaling secondary loss function if there are multiple conjoined networks (like Mipnerf/Vanilla)"""
    loss_secondary_density_scale: float = 1.
    """Scaling secondary loss function if there are multiple conjoined networks (like Mipnerf/Vanilla)"""
    loss_function_rgb: str = 'MSELoss'
    """The type of loss function for rgb parameters [MSELoss, L1Loss]"""
    loss_function_density: str = 'MSELoss'
    """The type of loss function for density parameters [MSELoss, L1Loss]"""
    accuracy_threshold: float = 0.001
    """Threhold for the binary explicit accuracy"""
    path:str = "data/GT"
    


class GTDataManager(DataManager):  # pylint: disable=abstract-method
    """GT Datamanager class for handling training and evaluation for WAPE framework

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: GTDataManagerConfig
    train_dataset: None
    eval_dataset: None

    def __init__(
        self,
        config: GTDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        print(torch.seed())

        self.camera_optimizer_config = None
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        self.path = Path(self.config.path)

        self.train_data, self.eval_data, self.im_data = self.create_dataset()
        
        self.train_loader = iter(self.train_data)
        self.eval_loader = iter(self.eval_data)
        self.im_loader = self.im_data

        self.gt_images_dict = self.load_images()

        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)

        if self.config.loss_function_rgb ==  'MSELoss':
            self.rgb_loss = MSELoss()
        elif self.config.loss_function_rgb == 'L1Loss':
            self.rgb_loss = L1Loss()
        elif self.config.loss_function_rgb == 'PSNR':
            self.rgb_loss = PeakSignalNoiseRatio(data_range=1.0)

        if self.config.loss_function_density == 'MSELoss':
            self.density_loss = MSELoss()
        elif self.config.loss_function_density == 'L1Loss':
            self.density_loss = L1Loss()
        elif self.config.loss_function_density == 'PSNR':
            self.density_loss = PeakSignalNoiseRatio(data_range=1.0)()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_train_rays_per_batch(self) -> int:
        return int(self.config.train_batch_size)

    def get_eval_rays_per_batch(self) -> int:
        return int(self.config.test_batch_size)

    def load_images(self):
        """Load the ground truth image data which was pre-pregenerated
        """
        path = self.path.__str__()+'/images'
        ims = os.listdir(path)
        
        transform = transforms.ToTensor()
        image_dict = {}
        for idx, im in enumerate(ims): # filelist[:] makes a copy of filelist.
            if im.endswith(".png"): 
                im_ = Image.open(path+'/'+im)

                image_dict[str(idx)] = transform(im_)
        return image_dict
 
    def load_camera_models(self):
        """Load the camera models which were pre-generated
        """
        data =  torch.load(f'{self.path}/camera.pt')
        ray_bundles = []
        o = []
        d = []
        ci = []
        Ap = []

        images = []
        rows = 0
        colums = 0
        for i_idx, idx in enumerate(data.keys()):
            cx, cy = data[idx]['c'] 
            fx,fy = data[idx]['f'] 
            c2w = data[idx]['c2W'] 

            self.camera = Cameras(fx=fx, fy=fy,cx=cx, cy=cy, camera_to_worlds=c2w, camera_type=CameraType.PERSPECTIVE)
            ray_bundle = self.camera.generate_rays(camera_indices=0)
            
            o.append(ray_bundle.origins.unsqueeze(0))
            d.append(ray_bundle.directions.unsqueeze(0))
            Ap.append(ray_bundle.pixel_area.unsqueeze(0))
            ci.append(ray_bundle.camera_indices.unsqueeze(0))
            ray_bundles.append(ray_bundle)

            images = i_idx+1
            rows = ray_bundle.origins.shape[0]
            colums = ray_bundle.origins.shape[1]

        o = torch.cat(o, dim=0)
        d = torch.cat(d, dim=0)
        Ap = torch.cat(Ap, dim=0)
        ci = torch.cat(ci, dim=0)
        
        # Construct a raybundle containing all data
        ray_bundle = RayBundle(origins=o, directions=d, pixel_area=Ap, camera_indices=ci)
                
        self.ray_bundles = ray_bundle
        self.im_metadata = {'nv':int(images), 'nr':int(rows), 'nc':int(colums)}
 
    def load_training_data(self):
        """Load the pre-generated training data
        """
        with open(f'{self.path}/data.json') as fp:
            data = json.load(fp)
        
        # R is the index pointer of sample data relative to the views
        R = torch.tensor(data['R']).to(self.device)
        samples = torch.tensor(data['values']).to(self.device)
        image_labels = {'l':data['labels'], 'metadata':self.im_metadata}
        
        R_indices = torch.tensor(data['Rlabels']).to(self.device)
    
        labels_indices = torch.linspace(0, len(image_labels['l'])-1, len(image_labels['l']), dtype=torch.int).long().to(self.device)

        return R_indices, R, samples.squeeze(0), image_labels, labels_indices

    def load_imaged_eval_data(self):
        """Load the image evaluation data for testing novel views

            Notes:
                We load the eval data relative to the number of intersections along a ray so we can compute matrices with 
                the same number of ray-samples in a higher dimentional matrix (i.e. in one go rather than computing rays one-by-one). This makes it quicker to process.
        """
        with open(f'{self.path}/image_data.json') as fp:
            RBSDict = json.load(fp)
        
        keys = [key for key in RBSDict]

        LOADERS = []

        for key in keys:
            if key != 'ignored img indexs':
                data = torch.tensor(RBSDict[key]['data']).to(self.device)
                indexes = torch.tensor(RBSDict[key]['image_index']).to(self.device)

                assert data.shape[0] == indexes.shape[0], 'Error in image data, [data and indexs dont match size]'
            
                dataset = torch.utils.data.TensorDataset(data, indexes)

                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset, 
                    batch_size=self.config.image_batch_size,
                )
                LOADERS.append(dataloader)
            elif key == 'ignored img indexs':
                self.ignore_image_indexs = torch.tensor(RBSDict['ignored img indexs']).to(self.device)

        return LOADERS

    def create_dataset(self):
        """Define the data loaders for training and testing

            Notes:
                self.BSL is the 'Big Sized List' (tensor) containing the sample data which self.R poins to 
        """
        self.load_camera_models()
        image_dataloaders = self.load_imaged_eval_data()
        self.R_indices, self.R, self.BSL, self.im_labels, labels_indices = self.load_training_data()

        labels_indices_shuffled = labels_indices[torch.randperm(labels_indices.size()[0])]
        train_data = labels_indices_shuffled[:]
        test_data = labels_indices_shuffled[int(labels_indices_shuffled.size(0)*self.config.tt_data_ratio):]

        train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                                batch_size=self.config.train_batch_size, 
                                                shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                                batch_size=self.config.test_batch_size, 
                                                shuffle=True)

        return train_loader, test_loader, image_dataloaders

    def reset_imloader(self):
        """Resetting im loader"""
        self.im_loader = iter(self.im_data)

    def process_step(self,x):
        """Processes the data x being loaded in and returns the samples and eval 

            Args:
                x, Tensor, is the index from R which points to a number of rays in self.BSL
            Return:
                ray_samples, RaySamples, the samples corresponding to the x indices
                eval_dict, dict, the ground truth rgb and weights values for evaluation
        """
        bsl = self.BSL[x]

        if bsl.shape[0] > 0:
            
            frustums = Frustums(
                origins=bsl[:, 0:3].unsqueeze(1),  # [..., 1, 3]
                directions=bsl[:, 3:6].unsqueeze(1),  # [..., 1, 3]
                starts=bsl[:, 6].unsqueeze(1).unsqueeze(1),  # [..., num_samples, 1]
                ends=bsl[:, 7].unsqueeze(1).unsqueeze(1),  # [..., num_samples, 1]
                pixel_area=bsl[:,12].unsqueeze(1).unsqueeze(1),  # [..., 1, 1]
            )

            ray_samples = RaySamples(
                frustums=frustums,
                camera_indices=bsl[:,13].unsqueeze(1).unsqueeze(1).int(),  # [..., 1, 1]
                deltas=bsl[:, 7].unsqueeze(1).unsqueeze(1) - bsl[:, 6].unsqueeze(1).unsqueeze(1),  # [..., num_samples, 1]
                spacing_starts=bsl[:, 6].unsqueeze(1).unsqueeze(1),  # [..., num_samples, 1]
                spacing_ends=bsl[:, 7].unsqueeze(1).unsqueeze(1),  # [..., num_samples, 1]
            )

            eval_dict = {
                "density":bsl[:,11].unsqueeze(1).unsqueeze(1),
                "rgb":bsl[:, 8:11].unsqueeze(1),

            }
            return ray_samples, eval_dict, 1
        else: 
            ray_samples = None
            eval_dict = None
            return None,None, 1
        
    def next_train(self, step: int):
        """Returns the next batch of training data"""
        # self.train_count += 1
        
        try:
            x = next(self.train_loader)
        except StopIteration as e:
            # Reset the data loader
            # TODO: create seperate method to re-randomise the dataloader
            self.train_loader = iter(self.train_data)
            x = next(self.train_loader)

        return self.process_step(x)

    def next_eval(self, step: int):
        """Returns the next batch of data from the eval dataloader."""
        try:
            x = next(self.eval_loader)
        except StopIteration as e:

            self.train_loader = iter(self.eval_data)
            x = next(self.eval_loader)

        return self.process_step(x)

    def get_num_image_evals(self):
        self.is_image_eval = True

        return self.n_iters_for_image_eval
    
    def next_eval_image(self, step: int):
        """Returns a batch of data for building image predictions"""
        try:
            x = next(self.im_loader)
            return self.process_step(x)
        except StopIteration as e:
            # print('Finished Processing Image')
            return None, None, 0

    def get_param_groups(self):  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        return param_groups

    def get_metrics_dict(self, outputs, eval_dict):
        """Get the metrics to validate learning (on training data)

            Args:
                outputs, dict, model outputs
                eval_dict, dict, ground truth values
            
            Return:
                metrics_dict, dict, final metrics
        """
        metrics_dict = {}
        # If there are multiple networks this will return the metrics of the final network
        for key in outputs:
            if 'rgb' in key:
                metrics_dict['r_binary'] = binary_accuracy(
                    eval_dict['rgb'].flatten(), 
                    outputs[key].flatten(), 
                    threshold=self.config.accuracy_threshold
                ) 
                metrics_dict['r_PSNR'] = self.psnr(
                    eval_dict['rgb'].flatten(), 
                    outputs[key].flatten()
                ) 
                metrics_dict['r_Error'] = (eval_dict['rgb'].flatten() - outputs[key].flatten()).mean()

            elif 'density' in key:
                metrics_dict['d_binary'] = binary_accuracy(
                    eval_dict['density'].flatten(), 
                    outputs[key].flatten(), 
                    threshold=self.config.accuracy_threshold
                ) 
                metrics_dict['d_PSNR'] = self.psnr(
                    eval_dict['density'].flatten(),
                    outputs[key].flatten()
                ) 
                metrics_dict['d_Error'] = (eval_dict['density'].flatten() - outputs[key].flatten()).mean()

        return metrics_dict

    def get_loss_dict(self, outputs,eval_dict):
        """Get the loss from the model outputs

            Args:
                outputs, dict, model outputs containing the rgb and desnity predictions
                eval_dict, dict, ground truth values
            
            Return:
                loss_dict, dict, contains the calculated losses
            
            Notes:
                We seperate the loss for INGP, Nerfacto and Mipnerf and Mipnerf requires two losses for both of its networks (which we don't train ent-to end)
        """
        loss_dict = {}

        if 'rgb_fields' in outputs: # For INGP and Nerfacto
            loss_dict['rgb'] = self.config.loss_rgb_scale*self.rgb_loss(eval_dict['rgb'], outputs['rgb_fields'])
            loss_dict['density'] = self.config.loss_density_scale*self.density_loss(eval_dict['density'], outputs['density_fields'])


        elif 'rgb_first_fields' in outputs: # For Mipnerf and vanilla nerf
            loss_dict['rgb_first'] = self.config.loss_rgb_scale*self.rgb_loss(eval_dict['rgb'], outputs['rgb_first_fields'])  
            loss_dict['rgb_second'] = self.config.loss_secondary_rgb_scale*self.rgb_loss(eval_dict['rgb'], outputs['rgb_second_fields']) 
            loss_dict['density_first'] = self.config.loss_density_scale*self.density_loss(eval_dict['density'], outputs['density_first_fields']) 
            loss_dict['density_second'] = self.config.loss_secondary_density_scale*self.density_loss(eval_dict['density'], outputs['density_second_fields']) 

        return loss_dict
