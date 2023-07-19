import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tensordict
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import nn
from trainer import Trainer
from utils_ import Accuracy, Loss, Planes, Ray, RayBundle, build_block, get_view_rays


"""Define a class for generating and displaying a scene
"""
class Scene:
    # Template for a block
    unit_cube = torch.tensor([
        [[0.,0., 0.], [1.,0.,0.], [1.,0.,1.], [0.,0.,1.]],
        [[0.,1., 0.], [1.,1.,0.], [1.,1.,1.], [0.,1.,1.]],
        [[0.,0., 0.], [0.,1.,0.], [1.,1.,0.], [1.,0.,0.]],
        [[0.,0., 1.], [0.,1.,1.], [1.,1.,1.], [1.,0.,1.]],
        [[0.,0., 0.], [0.,0.,1.], [0.,1.,1.], [0.,1.,0.]],
        [[1.,0., 0.], [1.,0.,1.], [1.,1.,1.], [1.,1.,0.]],
    ], dtype=torch.double)

    # colour to iterate through when displaying
    display_colours = ['r', 'g', 'b', 'y', 'c']

    # devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_blocks(self):
        """Literally, build the cuboids which make up our scene
        """
        # Generate some blocks (change if you desire)
        cube1 = build_block((5.,5.,5.), (1.,1.,1.), self.unit_cube.clone())
        cube2 = build_block((5.,1.,5.), (6.,1.,1.), self.unit_cube.clone())
        cube3 = build_block((5.,1.,5.), (6.,3.,1.), self.unit_cube.clone())
        blocks = [cube1, cube2, cube3]
        
        # Add colour for visualising the scene
        cols = []
        for i, b in enumerate(blocks):
            local_cols = [self.display_colours[( i % len(self.display_colours) )] for j in range(b.size(0))]
            cols = cols + local_cols

        blocks_T = torch.cat(blocks, dim=0)
        
        # define the planes on our cube
        planes = Planes(bounds=blocks_T, id=1)

        return planes, cols

    def build_views(self):
        """Build the views (batches of rays in a rectangle shape)
        """
        x_sample = 20
        y_sample = 30

        # Define Views: Camera Scale, Shift (relative to x-z), rotation
        r1 = Ray((8.,1.,6.), (0.,12.,1.), (-.2, -0., 0.))
        r2 = Ray((8.,1.,6.), (8.,10.,0.), (0., -.0, +.6))
        r3 = Ray((8.,1.,6.), (3.,10.,-5.), (+0.4, -.0, +.2))
        r_groups = [ r1 , r2, r3] 

        # Get rays from ray-view function
        rays = []
        rays_for_display = []
        cols = []
        for i, r in enumerate(r_groups):
            d,sc,sh = r.dirs, r.scale, r.shift

            ray = get_view_rays(x_sample, y_sample, sc, d, sh)
            rays = rays + ray

            rays_for_display.append(ray)
            cols.append(self.display_colours[ (i % len(self.display_colours)) ])

        rays_np = np.array(rays) # convert to numpy
        rays_T = torch.tensor(rays_np, dtype=torch.double) # convert to tensor

        # Reformat our views for visualisation
        rays_for_display_np = np.array(rays_for_display) # convert tp numpy
        rays_for_display_np[0,:,3:] = rays_for_display_np[0,:,3:]/.5 # make aroows smaller
        return rays_T, rays_for_display_np, cols
    
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate the Volume and Viewing Planes in our scene
        self.blocks, self.blocks_cols = self.build_blocks()
        self.rays, self.display_rays, self.rays_cols = self.build_views()

        # Define All origins and directions 
        self.o = self.rays[:,0:3].to(self.device)
        self.d = self.rays[:,3:6].to(self.device)

        # Init the Trainer veriable
        self.trainer = None

    def set_trainer(self, tClass):
        """ Set the trainer class
        """
        self.trainer = tClass
        
    def initialiser_new_trainer(self, params):
        """Initialise a new Trainer using new params
        """
        assert self.trainer is not None, 'Scene Error: No Trainer initialised'
        self.trainer.initialise_trainer_network(*params)

    def save_trainer(self):
        """Save the important trainer info
        """
        self.trainer.save_state()

    def load_trainer(self, title):
        """Load the trainer
        """
        assert self.rays is not None, 'Error: Need to load/generate scene before loading trainer '
        self.trainer.load_state(title, self.rays)


    def get_ray_plane_intersections_t(self,):
        """Find t-distance scalar for the points where rays intersect wth plane
                Warning: we are dealing with batches of planes and batches of rays all at once! If your scene is huge, this will surely kill.

            Notes:
                The planes are refered to by their a, b, c and k components, as part of the planar equation: `p(.)= ax+by+cz+k=0`.
                Additionally, ray-points are referd to by their o, d and t components, as part of the ray equation: r = o + td, where r is vector in R^3 (x,y,z)
                
                Our objective is to solve for t, where r lies along a p(.). 
                
                To do this,we first need to solve for `k`, to define the whole planar reference. We do this by finding the unit-norm (normalised cross-product, n) of two surface axis' and multiplying the unit-norm by another point on the surface to find k (diagonal function is used for matrix computation). This represents the function `k = -(x, y, z).n`

                Next: Consider, x.n+k = 0 defines a point on a plane given x and k. As we now know k, we just need to find x which satisfies this equation. "Simulataneously", we need to solve r = x, thus, we are given the equaiton (o+td).n+k = 0, thuss `o.n + t(d.n) + k = 0`. Finally, `t = - (o.n + k) / (d.n)`. Which is solved in batched form.
        """
        # Fetch the boundaries of the surface-blocks
        P = self.blocks.bounds.to(self.device)

        # Get two arbitrary vectors along the plane (to solve planar definition)
        v1 = P[:, 0] - P[:, 1]
        v2 = P[:, 0] - P[:, 3]  
        
        # Solve plane's normal vector
        v = torch.cross(v1,v2, dim=1)
        # Reduce planes normal to unit-norm (divide vector by HCF)
        mag_v = torch.linalg.norm(v, dim=1).unsqueeze(1)
        v = torch.div(v, mag_v) # this should give us a,b,c in planar equation

        k = - torch.diagonal(torch.mm(v, P[:,2].T))

        vd = torch.mm(v, self.d.T)
        vo = torch.mm(v, self.o.T)
        k1 = k.unsqueeze(1).repeat(1,vo.size(1))
        kvo = -torch.add(k1, vo)
        t = torch.div(kvo, vd)

        return t
 
    def tansform_t_to_xyz(self, t):
        """Transform t-values to r_xyz values using ray equation
        """
        ts = t.unsqueeze(2)
        d = self.d.unsqueeze(0).repeat(ts.size(0), 1,1)
        o = self.o.unsqueeze(0).repeat(ts.size(0), 1,1)
        td = ts*d
        r = o+td
        return r, o, td

    def on_surface_filter(self, r):
        """Reduce all planar intersections to the existing surface

            Args:
                r: Tensor, shape(# of planes, # of rays, 3, 1) for xyz intersections given a plane and ray crossing
            Notes:
                We first get the min and max x,y,z values from the bounds of each surface. Then we evaluate the ray-plane xyz interception, to determine if its is between the max and min bounds of our rectangular 3D surface.
                We create two matrix masks for min and max values raltive to r and combine the two to give us a mask for r which defines the on-surface intersections. 
                We extract the indices of these points in the r-matrix and return these.
        """
        # Fetch the boundaries of the surface-blocks
        P = self.blocks.bounds.to(self.device)
        # Get the min and max xyz values for the bounds of the square surface
        surf_min = torch.min(torch.transpose(P, 1, 2), 2).values.unsqueeze(1).repeat(1,self.o.size(0),1)
        surf_max = torch.max(torch.transpose(P, 1, 2), 2).values.unsqueeze(1).repeat(1,self.o.size(0),1)
        # Reshape to evaluate ray-plane matrix
        surf_min = P[:, 0].unsqueeze(1).repeat(1,self.o.size(0),1)
        surf_max = P[:, 2].unsqueeze(1).repeat(1,self.o.size(0),1)

        # Evaluate the xyz intercepts with the min and max positions and combine binary masks to hide intercepts not on surface
        err = 0.001 # we have to force an error value because some the precision of the equality is smaller than the precision of a double so values which should be equal will be tiny bit off.
        on_surf_mask_min = torch.prod(torch.le(surf_min, r+err), dim=2)
        on_surf_mask_max = torch.prod(torch.le(r-err, surf_max), dim=2)
        on_surf_mask = (on_surf_mask_min * on_surf_mask_max)
        on_surf_indices = (on_surf_mask > 0.).nonzero(as_tuple=True)
        
        on_surf_indices = torch.cat([on_surf_indices[0].unsqueeze(1), on_surf_indices[1].unsqueeze(1)], dim=1)
        
        # torch.set_printoptions(threshold=10_000)
        # print(torch.cat([surf_min[2], r[2], on_surf_mask_min[2].unsqueeze(1).repeat(1,1)], dim=1))

        # Select planer
        # indices = []
        # for o in on_surf_indices:
        #     if o[0] == 2: # slect plane number (2 is bottom red box)
        #         indices.append(o.cpu().tolist())
        # on_surf_indices = torch.tensor(indices)

        return on_surf_indices

    def first_intersection_filter(self, on_surf_indices, o, td):
        """Refine all on-surface intersections to the earliest occuring intersections relative to each ray
        """
        # Initialise on-surface t-matrix as inf distance away
        on_surf_td = torch.full_like(td, torch.inf).to(self.device)
        on_surf_td[on_surf_indices[:,0], on_surf_indices[:, 1]]  = td[on_surf_indices[:,0], on_surf_indices[:,1]]

        # Get magnitude of td to get distance scalar
        inplane_ts = torch.linalg.norm(on_surf_td, dim=2)
        # Get min distance and define the indices of the tmins for the r-matrix
        t_mins = torch.argmin(torch.transpose(inplane_ts, 0, 1), 1)
        tmins_0 = t_mins.unsqueeze(1)
        tmins_1 = torch.arange(0, tmins_0.size(0)).to(tmins_0.device).unsqueeze(1)
        tmins = torch.cat([tmins_0, tmins_1], dim=1)
        
        indices = []
        for tm in tmins:
            if inplane_ts[tm[0],tm[1]] != torch.inf:
                    indices.append(tm.tolist())
        indices = torch.tensor(indices)

        return indices
    
    def get_intersection(self, indices, r, t):
        """Get intersection of xyz, t and indices (ref to plane-ray matrix) form
        """
        # Set max distance as 1000. - arbitrary far-far distance
        max_d = 1000.
        # tranfer intersections where our desired indices point
        ray_indices = torch.full_like(r, max_d).to(self.device)
        ray_indices[indices[:,0], indices[:,1]]  = r[indices[:,0], indices[:,1]]
        
        # transfer t-intersections where our desired indices point
        ts = t.unsqueeze(2)
        dist_indices = torch.full_like(ts, torch.inf).to(self.device)
        dist_indices[indices[:,0], indices[:,1]]  = ts[indices[:,0], indices[:,1]]

        self.display_intersections = ray_indices.flatten(start_dim=0, end_dim=1).cpu().numpy()
        return ray_indices, dist_indices, indices

    def get_intersection_data(self):
        """Method to call for getting intersection data (xyz indices, )

            Notes:
                Firstly, we solve for t-distance along our rays where all intersection occurs.
                Then we transfor t-values to ray-vectors, `r` (which defines a matrix containing all x,y,z values for all plane-ray intersections with shape (plane #, ray#, 3)).
                Then we filter the intersection data to gather the indices in out plane-ray matrix.
                We can then use these to formalise the intersection data
        """
        t = self.get_ray_plane_intersections_t()
        r, o ,td = self.tansform_t_to_xyz(t)
        on_surface_indices = self.on_surface_filter(r)
        first_on_surf_intersections = self.first_intersection_filter(on_surface_indices, o, td)
        r_, t_, indices = self.get_intersection(first_on_surf_intersections, r, t)

        return r_, t_, indices

    def generate_target_data(self):
        """Get the target data as Dict
            TODO:
                See if any faster ref structures exist
        """
        xyz_intersections, t_intersections, indices = self.get_intersection_data()
        
        # Get the coordinates of non-inf t-values (i.e. those in our local scene)
        t_in_scene = torch.lt(t_intersections.float(), torch.inf)
        coords = ((t_in_scene == True).nonzero(as_tuple=True)) # fetch indices
        coords = torch.cat([coords[0].unsqueeze(1), coords[1].unsqueeze(1)],dim =1)

        target_half_delta = 0.05 # from a known surface point we deviate collision bounds by 0.05 (near and far target have a .1 separation) TODO: Might need to reduce the separation
        ray_targ_dict = {}

        # Initialise all target data with some arbitrary far-distant value
        #  (We assume a NN will not produce nan or +-inf values, thus we need a Real target)
        max_d = 10.
        for os,ds in zip(self.o, self.d):
            od_id = "[%.2f %.2f %.2f %.2f %.2f %.2f]" % (os[0].float(), os[1].float(), os[2].float(), ds[0].float(), ds[1].float(), ds[2].float())
            n_target = torch.tensor(max_d)
            f_target = torch.tensor(max_d+target_half_delta)
            ray_targ_dict[od_id] = {'targets':{'n':n_target,'f':f_target},
                                'planes-ray indices':None} 
        # Insert targets for known intersections
        for c in coords:
            od_id = "[%.2f %.2f %.2f %.2f %.2f %.2f]" % (self.o[c[1]][0].float(), self.o[c[1]][1].float(), self.o[c[1]][2].float(), self.d[c[1]][0].float(), self.d[c[1]][1].float(), self.d[c[1]][2].float())
            # To filter out any intersections which could plausibly be behind our view we clamp distances to 0
            n_target = torch.clamp(t_intersections[c[0], c[1]], min = 0.)
            f_target = t_intersections[c[0], c[1]] + target_half_delta
            ray_targ_dict[od_id] = {'targets':{'n':n_target.float(),'f':f_target.float()},
                                'planes-ray indices':c}

        return ray_targ_dict

    def get_targets(self, oris, dirs, targs):
        """Given the origins and directions and target dictionary, return the near and far targets
        """
        targets_ = []
        for o,d in zip(oris, dirs):
            # targ_id = str(o.tolist()+d.tolist())
            targ_id = "[%.2f %.2f %.2f %.2f %.2f %.2f]" % (o[0].float(), o[1].float(), o[2].float(),
                                                    d[0].float(), d[1].float(), d[2].float())
            n = targs[targ_id]['targets']['n']
            f = targs[targ_id]['targets']['f']
            targets_.append([n,f])
        
        targets_= torch.tensor(targets_).to(dirs.device)
        return targets_[:,0], targets_[:,1]

    def save_data(self, name=''):
        """We are going to save all the data which are essential after generation

            Notes:
                We are saving, ray inputs and target dictionary for training. And 
                GT scene properties for visualisation
        """
        # print(f'Saving data for experiment: {name} ...')

        assert self.rays is not None
        torch.save(self.rays, f'save_data/rays_{name}.pt')
        
        assert self.target_data is not None, self.target_data
        targs = tensordict.TensorDict(self.target_data, [])
        torch.save(targs, f'save_data/targs_{name}.pt')

        scene_data = {
            "blocks":{
                "data":self.blocks.bounds
            },
            "rays":{
                "data": self.display_rays
            },
            "intersections":{
                "data":self.display_intersections
            }
        }

        scene = tensordict.TensorDict(scene_data, [])

        torch.save(scene, f'save_data/scene_data_{name}.pt')


    def load_data(self, name=''):
        """Load Scene data using name
        """
        assert name != '', 'Load Data Error: No experiment name provided'
        print(f'Loading data for experiment: {name} ...')

        self.rays = torch.load(f'save_data/rays_{name}.pt')
        self.target_data = torch.load(f'save_data/targs_{name}.pt').to_dict()
        scene = torch.load(f'save_data/scene_data_{name}.pt').to_dict()

        self.blocks.bounds = scene["blocks"]["data"]
        self.display_rays = scene["rays"]["data"]
        self.display_intersections = scene["intersections"]["data"]

        self.blocks_cols = []
        for i in range(scene["blocks"]["data"].size(0)):
            self.blocks_cols.append(self.display_colours[ (i % len(self.display_colours)) ])
        self.rays_cols = []
        for i in range(self.rays.size(0)):
            self.rays_cols.append(self.display_colours[ (i % len(self.display_colours)) ])


    def generate_data(self, save=''):
        """Function call for running the data generation
        """
        print('Generate Target Data...')
        self.target_data = self.generate_target_data()
        if save != '':
            self.save_data(save)

    def run(self, title='Test', load_data=False):
        """Run the training cycle
        """
        assert self.trainer is not None, 'Run: No Trainer initialised - run Scene.set_trainer(cls) '

        # Reload data at beginning of each training epoch - for my personal sanity
        if load_data:
            self.load_data(title)
        else:
            self.generate_data(title)
        
        traking = self.trainer.run(self.rays, 
                    targs=self.target_data, 
                    get_targs_fn=self.get_targets)
        
        return traking
        # self.trainer.save_training_results_png(title)
    
    
    """Now, the FUN part: Visualisation functions
    """
    
    def display_pred_scene(self, 
        title = 'Test',
        view_scale = (6,6,6),
        show_cubes = True,
        show_rays = True,
        show_GT_intersections = True,
        alpha = .5
    ):
        """Display the Predicted Scene Using predicted values

        Args:
            title: experiment title
            view_scale: Matplotlib 3D graph scale
            show_cubes,show_rays,show_GT_intersections: If true show these components in the plot
            alpha: opacity of cubes
        """
        bounds, blocks_cols = None, None
        display_rays, rays_cols = None, None
        display_intersections = None
        if show_cubes:
            bounds = self.blocks.bounds
            blocks_cols = self.blocks_cols
        if show_rays:
            display_rays = self.display_rays
            rays_cols = self.rays_cols
        if show_GT_intersections:
            display_intersections = self.display_intersections

        self.trainer.disp_approximations(title, view_scale, bounds,
                blocks_cols, display_rays, rays_cols, 
                display_intersections, alpha = alpha)
    
    def display_scene(self, 
        title = 'Test',
        view_scale = (20,20,20),
        show_cubes = True,
        show_rays = True,
        show_GT_intersections = True
    ):
        """Visualise the scene without predictions
        """
        bounds, blocks_cols = None, None
        display_rays, rays_cols = None, None
        display_intersections = None

        if show_cubes:
            bounds = self.blocks.bounds
            blocks_cols = self.blocks_cols
        if show_rays:
            display_rays = self.display_rays
            rays_cols = self.rays_cols
        if show_GT_intersections:
            display_intersections = self.display_intersections

        self.trainer.disp_scene(
            title, view_scale, bounds,
            blocks_cols, display_rays, rays_cols, 
            display_intersections
        )

    def final_metric(self, 
        title = 'Test',
        view_scale = (6,6,6),
        show_cubes = True,
        show_rays = True,
        show_GT_intersections = True,
        alpha = .5
    ):
        """Determine the final accuracy metric (in Trainer we can manually select the function for accuracy)
        """
        bounds, blocks_cols = None, None
        display_rays, rays_cols = None, None
        display_intersections = None
        if show_cubes:
            bounds = self.blocks.bounds
            blocks_cols = self.blocks_cols
        if show_rays:
            display_rays = self.display_rays
            rays_cols = self.rays_cols
        if show_GT_intersections:
            display_intersections = self.display_intersections

        self.trainer.metric(targs=self.target_data, 
                    get_targs_fn=self.get_targets)

    def disp_heatmap(self, 
        title = 'Test'
        ):
        """Return the rendered View depth-maps of our scene 
        """
        
        return self.trainer.disp_heatmap(self.target_data, 
                    self.get_targets)
