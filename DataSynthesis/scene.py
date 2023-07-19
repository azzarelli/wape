"""Handles the scene contstruction and shading functions

    Notes:
        Shading functions are found under 
"""

import numpy as np
import torch
import tensordict
from tqdm import tqdm
from torch.utils.data import DataLoader
from mayavi import mlab

from shaders.reflection_shader import ReflectionRayTracer
from shaders.glass_shader import GlassShader
from shaders.diffuse_shader import DiffuseShader

class Scene:
    def __init__(self,
        name=''
    ) -> None:
        self.name = name # perhaps a bit arbitrary

    def set_world(self, 
        r_matrix=None,
        t_matrix=None,
        indices_of_intersection=None,
        tn_matrix=None,
        tf_matrix=None, 
        C=None, S=None,
        normals = None
        ):
        """Set the NeRF volumetric representation properies as intersection matrices"""

        self.r = r_matrix # Global positional matrix
        self.t = t_matrix   # Local position matrix (scalar distances for each  rayintersection)
        self.ioi = indices_of_intersection # List containing all indices in our intersection matrics which correspond to valid intersections

        self.tn = tn_matrix # Volumetric parameter matrices tn and tf which describe the near and far collision planes of each intersection
        self.tf = tf_matrix

        self.C = C # Colour and Density Matrices defining the un-rendered surface colours
        S = torch.where(S >= 1., 100., S) # Sometimes we will transform densities = 1. to be >>1. This reverts the operation and ensure opacities are 0-1
        self.S = S

        self.normals = normals # Contains intersection matrix of surface normals at each intersection
    
    def save(self):
        assert self.name != ''

        scene_data = {
            "r_matrix":{
                "data":self.r
            },
            "t_matrix":{
                "data": self.t
            },
            "tn_matrix":{
                "data":self.tn
            },
            "tf_matrix":{
                "data":self.tf
            },
            "ioi":{
                "data":self.ioi
            },

        }


        scene_TDict = tensordict.TensorDict(scene_data, [])

        torch.save(scene_TDict, f'save_data/scene_{self.name}.pt')

    def get_normals(self):
        return self.normals #.unsqueeze(1)# .repeat(1,self.r.shape[1],1)

    def get_volume_parameters_from_ray_id(self,rayid):
        """Get tn and tf from rayid (return tn tf for ray)
            
            Args:
                rayid: Tensor, list tensor of rays which we can the NeRF volumetricparameters for
            
            Return
                t, tn, tf: Tensors, These are the NeRF volumetric spatial properties
                c, s: Tensors, These are the NeRF volumetric visual properties
        """
        t = self.t[:, rayid]
        tn = self.tn[:, rayid]
        tf = self.tf[:, rayid]
        c = self.C[:, rayid]
        s = self.S[:, rayid]
        
        return t, tn, tf, c, s

    def get_tntf_from_ray_surf_id(self, surfid, rayid):
        """Get tn and tf from surfid and rayid
            
            Args:
                surfid, rayid: Tensors, List indices relating to tn tf and t parameters
            Return:
                t, tn, tf: Tensors
        """
        rayid = torch.full_like(surfid, rayid)
        tn = self.tn[surfid, rayid]
        tf = self.tf[surfid, rayid]
        
        return tn, tf, tn

    
    def get_display_intersections(self):
        """Get xyz of in-bound ray-surface point of intersection

            Return:
                x,y,z: Tensors, The x,y,z componts of the gloval position of all intersections (Used for visualisation)
        """
        self.r = self.r.flatten(0,1)
        sum_r = self.r.sum(1).unsqueeze(1)
        indices = (sum_r < 3000.).nonzero(as_tuple=True)[0]

        r = self.r[indices].cpu().numpy()
        x,y,z = r[:,0], r[:,1], r[:,2]
        return x,y,z
        
    def get_reflections_raytracer(self, V, o, d, l, surf, batch_size=5000):
        """Map the reflections of othe objects in a scene onto matrials with density < 1.
                ~ This current simulates glass-like reflection for a singular lightsoruce
            
            Args:
                V: Tensor, Matrix containing all vertices packaged relative to each face
                o,d: Tensors, Vectors containing the global origin and directions of every ray
                l: Tensor, Global position of a singular light
                surf: Surface Class, contains all surface informaion which we have used just to visuall check internals are working
            
            Returns:
                COLOR: Tensor, A matrix containing the resulting colour from every intersection in a scene.
        """
        torch.cuda.empty_cache() # We have been using a lot of tensors so lets ease up memory

        indices = self.ioi  # Fetch intersections over the scene
        assert indices is not None, 'No indices for point light transform'

        COLOR = self.C
        # Create Dataloader to batch shading opertations (i.e. we iterate through batches of intersection)
        points = self.r[indices[:,0], indices[:,1]]
        points_data_iterator = iter(DataLoader(points, batch_size=batch_size))

        # Define Tracer prior to trace operations
        reflections = ReflectionRayTracer(o,d).to(self.C.device)

        data = tqdm(points_data_iterator)
        for pi, p in enumerate(data):
            # Get the index of our data in the ray-surface data table
            start = pi*batch_size
            end = start + p.shape[0]-1
            pointindex = torch.tensor(np.linspace(start,end, p.shape[0])).to(self.C.device).int()

            """ Uncomment the following to see the batch of the mesh mesh
            for i in range(surf.num_surfaces):
                x, y, z, tris, col, opa, tx, ty, tz = surf.get_surf_data(id=i)
                mlab.triangular_mesh(x,y,z, tris, color=col,opacity=.5, representation='surface')
            """

            # Precalculate the target origins and direction
            res = reflections(
                V, indices, pointindex, self.C, self.S, 
                p, 
                o[indices[pointindex.squeeze(0).long(),1]], 
                d[indices[pointindex.squeeze(0).long(),1]], # pass in only the necessary origins and directions 
                normals=self.get_normals(), it=pi, light=l, surfobj=surf)
            
            # Set the colour Render
            COLOR[indices[pointindex.squeeze(0).long(),0], indices[pointindex.squeeze(0).long(),1]] = res

        return COLOR

    def get_glass_raytracer(self, l, V, batch_size=1024):
        """Render Glass Matrerial
                ~ Currently this looks at shadow casting ~
            
            Args:
                V: Tensor, Matrix containing all vertices packaged relative to each face
                l: Tensor, Global position of a singular light
            
            Returns:
                COLOR: Tensor, A matrix containing the resulting colour from every intersection in a scene.
        """
        torch.cuda.empty_cache()

        indices = self.ioi
        assert indices is not None, 'No indices for point light transform'

        # Initialise Tracer
        light = GlassShader(dispersion=.5).to(self.C.device)

        # Set target data and data loader
        points = self.r[indices[:,0], indices[:,1]]
        points_data_iterator = iter(DataLoader(points, batch_size=batch_size))
        
        COLOR = self.C
        
        data = tqdm(points_data_iterator)
        for pi, p in enumerate(data):
            # Determine our local index in the dataloader
            start = pi*batch_size
            end = start + p.shape[0]-1
            pointindex = torch.tensor(np.linspace(start,end, p.shape[0])).to(self.C.device).int()

            # Run Shader
            COLOR[indices[pointindex.squeeze(0).long(),0], indices[pointindex.squeeze(0).long(),1]] = light(V, indices,pointindex, self.C,self.S, p, l, self.get_normals(), it=pi)
       
        return COLOR
    
    def get_diffuse_raytracer(self, l, V, batch_size=1024):
        """Render Diffuse Matrerial
                ~ Currently this looks at shadow casting ~
            
            Args:
                V: Tensor, Matrix containing all vertices packaged relative to each face
                l: Tensor, Global position of a singular light
            
            Returns:
                COLOR: Tensor, A matrix containing the resulting colour from every intersection in a scene.
        """
        torch.cuda.empty_cache()

        indices = self.ioi
        assert indices is not None, 'No indices for point light transform'

        # Set Data and Dataloader
        points = self.r[indices[:,0], indices[:,1]]
        points_data_iterator = iter(DataLoader(points, batch_size=batch_size))

        # Set Tracer
        light = DiffuseShader(dispersion=.5).to(self.C.device)

        COLOR = self.C

        data = tqdm(points_data_iterator)
        for pi, p in enumerate(data):
            # Get local index in data loader
            start = pi*batch_size
            end = start + p.shape[0]-1
            pointindex = torch.tensor(np.linspace(start,end, p.shape[0])).to(self.C.device).int()

            # Run ray trace
            COLOR[indices[pointindex.squeeze(0).long(),0], indices[pointindex.squeeze(0).long(),1]] = light(V, indices,pointindex, self.C,self.S, p, l, self.get_normals(), it=pi)
       
        return COLOR
