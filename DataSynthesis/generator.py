"""Data Generation Handler

"""
import sys
import json
from tqdm import tqdm as tdqm
import gc
import numpy as np
import tensordict
import torch
from torch.utils.data import DataLoader
from PIL import Image

from mayavi import mlab

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.model_components.renderers import (
    RGBRenderer,
)

from solver import Solver
from utils_ import Surfaces, ExplicitSampler
from scene import Scene


"""Define a class for generating and displaying a scene
"""
class SceneGenerator:

    def load_volumes(self, name):
        """Load the volume meshes using the datafile we made in Blender

            Notes:
                Currently we only read one light-source in
                If you want to use our shaders with MULTIPLE lightsources, it is recomended you run shaders iteratively

            Args:
                name, str, file name (.json)
            Return:
                surfaces, Surface Cls, surface container
                col, opa, Tensor, colour and opacity values
                light_pos, Tensor, position of lightsources
        """
        # Get file
        path = name+'.json'
        with open(path) as json_file:
            data = json.load(json_file)

        # Define lightpositions (the final lightsource will be the default single light)
        light_pos = None # TODO - multiple lightsources
        for l in data['lights']:
            light_pos = torch.tensor(data['lights'][l]).to(self.device)
        
        # Collect the vertices, normals, colours and opacities of each surface
        blocks = []
        normals = []
        col = []
        opa = []
        for volume_id in data['scene']:
            for surface_id in data['scene'][volume_id]: 
                blocks.append(data['scene'][volume_id][surface_id]['vertices'])
                normals.append(data['scene'][volume_id][surface_id]['normals'])
                col.append(data['c_o'][volume_id]['c'])
                opa.append(data['c_o'][volume_id]['o'])
                
        # Define the blocks containing the vertices and normals as Tensors        
        blocks_T = torch.tensor(blocks, dtype=torch.float).to(self.device)
        norms_T = torch.tensor(normals, dtype=torch.float).to(self.device)

        # Set surfaces
        surfaces = Surfaces(bounds=blocks_T, normals=norms_T, colours=col, opacities=opa)
        
        # Define colour and opacity tensors 
        # TODO - Check if this is necessary as have `surfaces.col` and `surfaces.opa`
        col = torch.tensor(col).to(self.device)
        opa = torch.tensor(opa).to(self.device)

        return surfaces, col, opa, light_pos

    def build_views(self, name, camera_size):
        """Build the views (batches of rays in a rectangle shape)

            Args:
                name, str, file name containing view data
                camera_size, list(float), containing x and y half-pixel sizes [100., 100.] is a 200.x200. pixel view

            Return:
                rays_T, Tensor, ray data (origin at [..., 0:3] and directions at [..., 3:-1])
                display_rays, list(Tensor), ray data for visualisation
                camera_save_data, dict, camera data for each view (key==index)
                ray_bundles, Tensor, 
                distribution_difference, torch.float, the difference between novel and training view distributions
        """
        # Fetch data from json file
        path = name+'.json'
        with open(path) as json_file:
            data = json.load(json_file)
        
        # Initialise the lists for collectinf necessary data
        rays = []
        ray_bundles = []
        zr = 10./100. # Zoom control TODO - make controlable at execution
        zoom = int(camera_size[0]*zr)
        camera_save_data = {}
        display_rays = []

        # Intilise list for determining distribution differences
        ORIGINS_ALL = []
        ORIGINS_TRAINING = []

        # Loop through each view
        for idx, camera in enumerate(data['cameras']):
            # Get camera pixel sizes, focal lengths and camera-to-world martrix transform (view projection)
            cx, cy = camera_size 
            fx = data['cameras'][camera]['f'][0]*zoom
            fy = data['cameras'][camera]['f'][1]*zoom
            c2w = torch.tensor([data['cameras'][camera]['world'][:3]])

            # Save camera data
            camera_save_data[str(idx)] = {
                "f":[fx,fy], 
                "c":[cx,cy],
                "c2W":c2w
            }

            # Define Nerfstudio camera model & generate the raybundles from camera placement
            camera = Cameras(fx=fx, fy=fy,cx=cx, cy=cy, camera_to_worlds=c2w, camera_type=CameraType.PERSPECTIVE)
            ray_bundle = camera.generate_rays(camera_indices=0)
            
            # Get origin and direction data
            origins = ray_bundle.origins.view(-1, 3)
            directions = ray_bundle.directions.view(-1, 3)
            ray = torch.cat([origins, directions], dim=1)  # define ray data

            # Append data to lists             
            display_rays.append(ray)
            rays.append(ray_bundle.flatten())
            ray_bundles.append(ray_bundle)

            # Evaluate the distribution of origins for training and evaluation views
            if idx in self.eval_views:
                ORIGINS_ALL += origins.tolist()
            else:
                ORIGINS_TRAINING += origins.tolist()
                ORIGINS_ALL += origins.tolist()
        
        # Calculate normal distribution parameters
        ORIGINS_ALL = torch.tensor(ORIGINS_ALL).to(self.device)
        ORIGINS_TRAINING = torch.tensor(ORIGINS_TRAINING).to(self.device)
        mean_novelview = ORIGINS_ALL.mean(0)
        mean_training = ORIGINS_TRAINING.mean(0)
        std_novelview = ((mean_novelview.unsqueeze(0) - ORIGINS_ALL)**2).sum(-1).mean().sqrt()
        std_training = ((mean_training.unsqueeze(0) - ORIGINS_TRAINING)**2).sum(-1).mean().sqrt()

        # Calculate absolute difference
        distribution_difference = (std_novelview - std_training).abs()
        
        # Get rays from ray-view function
        self.ray_bundles_flat = rays       
        views = torch.cat(display_rays)
        rays = torch.tensor(np.array(views)).squeeze(0)
        
        rays_np = np.array(rays) # convert to numpy
        rays_T = torch.tensor(rays_np, dtype=torch.float) # convert to tensor

        return rays_T, display_rays,camera_save_data, ray_bundles, distribution_difference
    
    def __init__(self, 
        name='', 
        camera_shape = (100.,100.),
        batch_size = (10240, 512),
        max_d = 800.,
        eval_views = [0]

    ) -> None:
        """Initialise
            Args:
                name: str, experiment name
                camera_size: tuple (2,), here cx, cy = camera_size for configuring the number of pixels in the cameras
                batch_szie, type, (2,) where the first index is the ray batch and the second is the sirface batching
        """
        assert name != '', 'Arg Error: We need the name of the experment..haaand it over! (this should also be the name of you Blender json file)'
        assert len(camera_shape) == 2, 'Arg Error: -cxcy or --camera-shape param is needs to be shape (2,)'
        assert type(camera_shape[0]) == float and type(camera_shape[1]) == float, 'Arg Error: -cxcy or --camera-shape param is needs to float'
        assert type(max_d) == float, 'Arg Error: Max distance needs to be a float'
        
        self.eval_views = [int(viewnum) for viewnum in list(eval_views)]

        self.scene = Scene(name=name)

        # Initilise additional params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_d = max_d
        self.camera = None
        self.display_rays = []
        self.camera_save_data = {}
        self.batch_size = batch_size
        
        # Generate the Volume and Viewing Planes in our scene
        self.surf, self.surf_cols, self.surf_opas,self.light_pos = self.load_volumes(name)
        rays, self.display_rays, self.camera_save_data, self.ray_bundles, self.distribution_difference = self.build_views(name, camera_shape)

        print(f'The difference in distribution between training and novel view data is {self.distribution_difference}')
        # Define All origins and directions 
        self.o = rays[:,0:3].to(self.device).contiguous()
        self.d = rays[:,3:6].to(self.device).contiguous()


    def geometry_solver(self):
        """Handle the geometric transformations from mesh to radiance field

            Notes:
                We loop through each view and set of surfaces to solve the geometric position
                We can loop through each surface independantly as intersections are independant. This reduces computation but does not prevent creation of large matrices later on
                E.G. For Nvidia RTX 3090 ~4GB is maximum possible size per matrix - any higher and it will crash
            TODO:
                Apply convex-hull solution to trapezoid created by projected view boundaries to only select 
                Hopefully this will remove unecessary computation
            
        """
        # Batch sizes for rays (a) and views (b)
        abatch = int(self.batch_size[0])
        bbatch = int(self.batch_size[1])
        
        # TODO: replace this with saving data as packets to remove dependency on large matrix generation
        # Initialise the global r, t, tn, tf matrices
        RR = []
        TT = []
        TTn = []
        TTf = []
        # Initialise the index matrix (pointer of intersections to position in global matrices)
        indices = torch.tensor([]).to(self.device)

        # Define the data loaders for each ray data
        o_data_iterator = iter(DataLoader(self.o, batch_size=abatch))
        d_data_iterator = iter(DataLoader(self.d, batch_size=abatch))
        raysbundle_dataset = enumerate(zip(o_data_iterator, d_data_iterator))

        
        # Loop through batches of rays
        raysbundle_dataset = tdqm(raysbundle_dataset)
        for j, (o, d) in raysbundle_dataset:
                
            # Surface dataloaders
            bounds_data_iterator = iter(DataLoader(self.surf.bounds, batch_size=bbatch))
            normals_data_iterator = iter(DataLoader(self.surf.normals, batch_size=bbatch))

            R = []
            T = []
            Tn = []
            Tf = []

            dataset = enumerate(zip(bounds_data_iterator, normals_data_iterator))
            for i,(b,n) in dataset:
                
                # Initialise the solver TODO: initialise outside of loop, this shouldn't impact performance
                solver = Solver(
                    max_render_distance=1000.
                )

                r_, t_, tn_matrix, tf_matrix, on_surface_indices = solver.main(
                    surface_points=b, 
                    surface_normals=n,
                    o=o,
                    d=d,
                )
                
                # Update indices relative to global matrix
                surf_indices_adder = i*bbatch
                ray_indices_adder = j*abatch
                on_surface_indices[:,0] = on_surface_indices[:,0] + surf_indices_adder
                on_surface_indices[:,1] = on_surface_indices[:,1] + ray_indices_adder

                # Collect volume data for ray-batch
                R.append(r_)
                T.append(t_)
                Tn.append(tn_matrix)
                Tf.append(tf_matrix)
         
                if indices.numel() == 0 and on_surface_indices.shape[0] > 0:
                    indices = on_surface_indices
                elif indices.numel() > 0:
                    indices = torch.cat([indices, on_surface_indices], dim=0)

            del  on_surface_indices, r_, t_, tn_matrix, tf_matrix

            RR.append(torch.cat(R, dim=0))
            TT.append(torch.cat(T, dim=0))
            TTn.append(torch.cat(Tn, dim=0))
            TTf.append(torch.cat(Tf, dim=0))

        del o_data_iterator, d_data_iterator, raysbundle_dataset
        
        RR = torch.cat(RR, dim=1).contiguous()
        TT = torch.cat(TT, dim=1).contiguous()
        TTn = torch.cat(TTn, dim=1).contiguous()
        TTf = torch.cat(TTf, dim=1).contiguous()

        torch.cuda.empty_cache()
        
        self.scene.set_world(
            r_matrix=RR, 
            t_matrix=TT, 
            indices_of_intersection=indices,
            tn_matrix=TTn, 
            tf_matrix=TTf,  
            C=self.surf_cols.unsqueeze(1).repeat(1, RR.shape[1], 1),
            S=self.surf_opas.unsqueeze(1).unsqueeze(1).repeat(1, RR.shape[1], 1),
            normals=self.surf.normals
        )
        
        num_elems = RR.numel()
        elem_size = RR.element_size()
        total_mem = num_elems * elem_size
        
        print(f'... Found {indices.size(0)} intersections {total_mem/1000000000.} Gb - Now Aplying Artifacts ...')
        return indices.size(0)

    def get_intersection_data(self,):
        """Build data set of values relative to each intersection

            Notes:
                Firstly, we solve for t-distance along our rays where all intersection occurs.
                Then we transfor t-values to ray-vectors, `r` (which defines a matrix containing all x,y,z values for all plane-ray intersections with shape (plane #, ray#, 3)).
                Then we filter the intersection data to gather the indices in out plane-ray matrix.
                We can then use these to formalise the intersection data
        """
        print('Generate Target Data...')
        
        npts = self.geometry_solver()

        print('~ Shader Info ~')
        print('Diffuse ray-tracer:          Order 1 Range 255 Light 1 Light Samples 25')
        print('Glass Absorption ray-tracer: Order 1 Range 255 Light 1 Light Samples 25')
        print('Glass Reflection ray-tracer: Order 2 Range 128 Light 1 Light Samples 1')
        print('Lighting Complexity: 54.0')
        print(f'Final Complexity: {float(npts)*self.distribution_difference * 54.0}')
        print(f'Without Reflection shader this is  {float(npts)*self.distribution_difference * 50.0}')
        
        self.scene.C = self.scene.get_diffuse_raytracer(
            self.light_pos, self.surf.bounds, 
            batch_size=int(self.batch_size[1])
        )
        self.scene.C = self.scene.get_glass_raytracer(
            self.light_pos, self.surf.bounds, 
            batch_size=int(self.batch_size[1])
        )

        # # Finalised ray-casting for NeRF consistent reflections (includes a lot of visualisation)
        self.scene.C = self.scene.get_reflections_raytracer(
            self.surf.bounds, 
            self.o, self.d, 
            self.light_pos, self.surf, 
            batch_size=int(self.batch_size[1]/2.))

        del self.surf.bounds, self.surf.normals, self.surf.colours, self.surf.opacities, self.surf, self.surf_cols, self.surf_opas
        del self.scene.r, self.scene.normals

        # torch.cuda.empty_cache()
        print(f'... Saving ...')
        self.render_view()

    def generate_data(self):
        """Function call for running the data generation
        """
        self.get_intersection_data()
    
        if self.scene.name != '':
            self.save_data()

    def save_data(self):
        """We are going to save all the data which are essential after generation

            Notes:
                We are saving the contents of
                GT scene propertises for visualisation
        """
        print(f'Saving data for experiment: {self.scene.name} ...')
        # self.scene.save()

        # data_dict = self.render_view()
        
        camera_dict = tensordict.TensorDict(self.camera_save_data, [])
        torch.save(camera_dict, f'save_data/camera.pt')
    
    def render_view(self,):
        """Render all the NeRF data (This can be considered the NERF shader as it uses Nerfstudio render to create NeRF pixel information)
            We also use this to sort and save our global matrices for training and evaluations
        """
        torch.cuda.empty_cache()

        # Initilise the empty images to save (these won't be used but they're useful for validation results) 
        images = torch.zeros((len(self.ray_bundles), self.ray_bundles[0].shape[0], self.ray_bundles[0].shape[1], 3,), dtype=torch.float).to(self.device)
        # Initialise the samples 
        samples = torch.tensor([]).to(images.device)
        #Get the shape of desired raybundles
        sh = [len(self.ray_bundles), *self.ray_bundles[0].shape]
        del self.ray_bundles

        # Initialise the index pointed R (for view # , ray # and sample # in global matrices)
        R = torch.zeros(sh[0], sh[1], sh[2], 2, dtype=torch.int).to(self.device)

        # Define the Samples and renderes
        self.sampler = ExplicitSampler()
        renderer = RGBRenderer(background_color="white")

        # Get the mask to tell us which rays have been intersected
        mask = torch.zeros_like(self.scene.t).to(self.scene.t.device)
        mask[self.scene.ioi[:,0], self.scene.ioi[:,1]]  = 1.
        mask = mask.sum(dim=0).squeeze(0)
    
        del self.scene.ioi

        # Ray Bundle size dicts
        RBSDict = {}

        # For each ray -> get tn tf c and sig, sort by value t_o and render an rgb for this
        mask = tdqm(mask)
        for idx, m in enumerate(mask):
            # Now Generate Ray Samples
            view_idx = int(idx/self.ray_bundles_flat[0].shape[0])
            view = self.ray_bundles_flat[view_idx]

            rb_index = int(view_idx % self.ray_bundles_flat[0].shape[0])
            ray = view[rb_index]

            row = int((idx%self.ray_bundles_flat[0].shape[0])/images.shape[1])
            col = int((idx%self.ray_bundles_flat[0].shape[0]) % images.shape[2])
            
            t, tn, tf, c, s = self.scene.get_volume_parameters_from_ray_id(idx)

            # If the intersection does not exist assign the furthest collision plane
            if m==0:
                tn = torch.tensor([0.5]).unsqueeze(0).unsqueeze(1).to(self.device)
                tf = torch.tensor([200.]).unsqueeze(0).unsqueeze(1).to(self.device)
                c = torch.tensor([0.,0.,0.]).unsqueeze(0).unsqueeze(1).to(self.device)
                s =  torch.tensor([0.]).unsqueeze(0).unsqueeze(1).to(self.device)
                
                # Render ray samples and set the 
                rays = self.sampler.generate_ray_samples(ray_bundle=ray, bin_starts=tn, bin_ends=tf).to(self.device)
                w = rays.get_weights(densities=s)
                rgb = renderer.forward(c, w).to(self.device)
                n_sampled_on_ray = 1

                o = (self.o[idx] + 20.*self.d[idx]).unsqueeze(0).unsqueeze(1).to(self.device)
                d = self.d[idx].unsqueeze(0).unsqueeze(1).to(self.device)
                pa = torch.tensor([0.0001]).unsqueeze(0).unsqueeze(1).to(self.device)
                ci = torch.tensor([0]).unsqueeze(0).unsqueeze(1).to(self.device)
        
            else:
                # Identify the indices of intersections with valid values
                surf_ind = (t < 900.).nonzero(as_tuple=True)[0]
                
                # Determine the order of these intersections (important for rendering)
                t = t[surf_ind, :]
                tsrt = torch.sort(t, dim=0,stable=True)
                t = tsrt.values
                srt = tsrt.indices

                surf_ind = surf_ind[srt].squeeze(1) # resort the surface indices

                tn = tn[surf_ind].unsqueeze(0)
                tf = tf[surf_ind].unsqueeze(0)
                c = c[surf_ind].unsqueeze(0)
                s = 1.* torch.clamp(s[surf_ind].unsqueeze(0), 0., 1.)

                rays = self.sampler.generate_ray_samples(ray_bundle=ray, bin_starts=tn, bin_ends=tf).to(self.device)

                # Get Weights and colours
                w = rays.get_weights(densities=10.*s)
                rgb = renderer.forward(c, w)

                # Set the pixel value
                images[view_idx, row, col] = rgb[0]
            
                o = rays.frustums.origins.squeeze(1)
                d = rays.frustums.directions.squeeze(1)
                pa = rays.frustums.pixel_area.squeeze(1)
                ci = rays.camera_indices.squeeze(1)

                n_sampled_on_ray = tn.shape[1]

            # Training and test data:
            #   We want to remove some of the views from this as we may
            #   want to test novel-view prediction
            if view_idx not in self.eval_views: 
                if samples.numel() == 0:
                    
                    start = 0
                    end = n_sampled_on_ray

                    samples =  torch.cat([o,d, tn, tf, c, s, pa, ci], dim=-1).to(samples.device)
                    
                    R[view_idx, row, col] = torch.tensor([start, end]).to(self.device)
                    labels = [f'{view_idx}-{row}-{col}-{n}' for n in range(n_sampled_on_ray)]
                    Rlabels = [[view_idx, row, col]]

                else:
                    start = samples.shape[1]
                    end = start + n_sampled_on_ray

                    data =  torch.cat([o,d, tn, tf, c, s, pa, ci], dim=-1).to(samples.device)
                    
                    samples = torch.cat([samples, data], dim=1)
                    
                    R[view_idx, row, col] = torch.tensor([start, end]).to(self.device) 

                    labels += [f'{view_idx}-{row}-{col}-{n}' for n in range(n_sampled_on_ray)]
                    Rlabels += [[view_idx, row, col]]
                

            if str(n_sampled_on_ray) not in RBSDict:
                RBSDict[str(n_sampled_on_ray)] = {'data':torch.tensor([]).to(samples.device), 'image_index':[]}

            # Place into raybundle size-depednant dict
            # We pass in s and w to do WAPE eval with s
            # and PNSR SSIM and LPIPs using w
            RBSdata = torch.cat([o,d, tn, tf, c, s, w, pa, ci], dim=-1).to(samples.device)
            RBSDict[str(n_sampled_on_ray)]['data'] = torch.cat([RBSDict[str(n_sampled_on_ray)]['data'], RBSdata], dim=0)
            RBSDict[str(n_sampled_on_ray)]['image_index'].append([view_idx, row, col])
            

        # Set dictionary containing all our training data
        ray_Dict = {"values": samples.cpu().tolist(), "R":R.cpu().tolist(), "labels":labels, "Rlabels":Rlabels}
        # Reformart dictionary containing image data
        for key in RBSDict:
            RBSDict[key]['data'] = RBSDict[key]['data'].cpu().tolist()

        RBSDict['ignored img indexs'] = self.eval_views

        with open(f'save_data/image_data.json', 'w') as f:
            json.dump(RBSDict, f)
        
        with open(f'save_data/data.json', 'w') as f:
            json.dump(ray_Dict, f)

        for idx, im in enumerate(images):
            img = Image.fromarray(np.uint8(255 * im.cpu().numpy()))  # no opencv required
            img.save(f'save_data/images/{idx}.png')


def check_memory():
    i = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                i += 1
                print(i, ' ',type(obj), obj.size(), sys.getsizeof(obj.storage())*(0.000001))
        except:
            pass