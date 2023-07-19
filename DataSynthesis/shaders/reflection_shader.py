import numpy as np
from mayavi import mlab
import torch

from utils.normals import *
from utils_ import *

def generate_hemisphere(r, samples):
    """Generate Hemispheres for caluclating Solid angles given a sample size"""
    u, v = np.mgrid[0:np.pi:samples[0], 0:np.pi:samples[1]]
    x = (r*np.cos(u) * np.sin(v)).flatten()
    y = (r*np.sin(u) * np.sin(v)).flatten()
    z = (r*np.cos(v)).flatten()
    vec = np.c_[x,y, z]
    return torch.tensor(vec).float()

"""Define the Global Veriables for a single pointlight"""
DISPERSION = 4. # Size of light if it were a ball
SAMPLE = (5j, 5j) # Sample size along hemisphere (long and latitude)
VECS = generate_hemisphere(DISPERSION, SAMPLE) # Vector positions of samples in local space (are precomputed and transformed with each reflections world matrix)

"""Class function for Reflection tracer"""
class ReflectionRayTracer(torch.nn.Module):
    def __init__(self, o, d):
        """Constructor

            Args:
                o,d : Tensor, Origin and Direction list containign all ray information
        """
        super().__init__()
        self.o = o
        self.d = d

        # Choose a vanishing distance for the power of reflection (fased with distance)
        self.power = 1. # 5.
        self.reflection_intensity = 1. # Define how 'intense' a reflection will be
        
    def forward(self, V, indices, pointindex, COL, OPA, p, o, d, it, normals, light, surfobj):
        """Processes a trace given a set of NeRF data

            Args:
                V: Tensor, global vertex position for each surface
                indices: Tensor, indices indicating the ray and surface positions in NeRF world matrices
                pointindex: Tensor, Local position of our data relative to the shader's dataloader
                COL, OPA: Tensors, Matrices containing colour and opacity for the entire scne
                p, o, d: Tensors, The global position of points and the ray origins and directions they belong to
                it: int, iteration of dataloader
                normals: Tensor, Contains all the normals of every surface for our target scene (precomputed in blender)
                surfobj: Surface Object Class, Contains the surface properties for a whole scene, only being used for visualisation

            Returns:
                render: Tensor, Contains the rendered colours
        """
        pointindex = pointindex.squeeze(0).long()

        # Fetch the normals for the batch
        n = normals[indices[pointindex,0]] #, indices[pointindex,1]]

        # Get the glass related factors:
        opacities = OPA[indices[pointindex,0], indices[pointindex,1]]
        glass_indices = (opacities < .999).nonzero(as_tuple=True)[0]
        
        # If theres no glass to process skip...
        if glass_indices.shape[0] <= 0:
            col = COL[indices[pointindex,0], indices[pointindex,1]]
            return col.float()
        
    
        o_glass = p[glass_indices] # set new vectors holding the point and
        d_glass = d[glass_indices] # direction vectors of each glass point
        n_glass = n[glass_indices]
        
        d_reflected = get_reflection_direction(dir=d_glass, norm=n_glass)
        
        # Find where costhera < 0. (this is where an exit intersect point in our batch exists and needs its reflection flipped as if it were glass.)
        # This is important for simulating a part of less dense materials
        cos_theta = torch.diagonal(torch.mm(n_glass, d_reflected.T)).unsqueeze(1)
        cos_theta_violater_indicies = (cos_theta < 0.).nonzero(as_tuple=True)[0]
        d_reflected[cos_theta_violater_indicies,:] = d_glass[cos_theta_violater_indicies, :]
        
        """ View the glass only reflections vectors
        n_np =  n_glass.cpu().numpy() # normals[closest_valid_t_surface_indices[valid_reflected_ray_indices], 0,:].cpu().numpy()
        o_np = (o_glass).cpu().numpy()
        d_np = (d_reflected).cpu().numpy()

        normalsss = mlab.quiver3d(o_np[:,0], o_np[:,1],o_np[:,2], n_np[:,0], n_np[:,1],n_np[:,2], opacity=.1, scale_factor=.5, line_width=.05, color=(0.,1.,0.), mode='arrow')
        reflected_ray = mlab.quiver3d(o_np[:,0], o_np[:,1],o_np[:,2],d_np[:,0], d_np[:,1],d_np[:,2], opacity=.1,   line_width=.5, scale_factor=1., color=(1.,0.,0.), mode='arrow')

        # intersections = mlab.points3d(valid_r[:,0], valid_r[:,1], valid_r[:,2],  opacity=1.,  scale_factor=.05, color=(0.,0.,1.))
        
        for i in range(surfobj.num_surfaces):
            x_, y_, z_, tris, colz, _, _,_,_ = surfobj.get_surf_data(id=i)
            mlab.triangular_mesh(x_,y_,z_, tris, color=colz,opacity=.2, representation='surface')
        
        mlab.show()
        exit()
        """

        # Find the intersections of rays with position o and direction d
        t_reflected = get_intersections(o_glass, d_reflected, V)
        # Fetch a mask of t > 0 : i.e. we are interested in intersections which lie infront of the 
        #   reflected (glass) ray (shape: #Glass, #surface, 1) 500, 252, 1
        infront_mask =  torch.gt(t_reflected, 0.0)

        # Fetch a mask for t where we don't want to include the surfaces of each glass index
        surface_indices = indices[pointindex,0][glass_indices].long()
        ordered_reflected_indices = torch.linspace(0, surface_indices.shape[0]-1,  surface_indices.shape[0]).to(surface_indices.device).long()
        # Set the infront mask to invalid (=0) where a t values is calucalted for its own surface
        infront_mask[ordered_reflected_indices, surface_indices] = 0

        # Get the indicies in matrix r_reflected (shape: #GlassIndices, #Surfaces, 3)

        r_reflected = o_glass.unsqueeze(1).repeat(1, t_reflected.shape[1],1) +\
                             t_reflected*d_reflected.unsqueeze(1).repeat(1,t_reflected.shape[1],1)
        on_surface_mask = get_onsurface_surface_intersections(r_reflected, V).unsqueeze(-1) # (shape: #GlassIndices, #Surfaces, 1)
        
        # Gather valid (onsurface) reflections infront of reflect ray
        infront_onsurface_mask = on_surface_mask*infront_mask
        t_valid_reflected = t_reflected * infront_onsurface_mask # (t= 0 is invalid t>0 is only valid)
        t_valid_reflected = ((~infront_onsurface_mask)*1000.) + t_valid_reflected # where invalid t = 1mil (extra large)
        
        # Get the indices (in the valid t matrix) of the smallest t for each ray
        #   and also create ordered batch indices to call cells
        closest_valid_t_surface_indices = torch.argmin(t_valid_reflected, dim=1).squeeze(-1)
        closest_valid_t_batch_indices = torch.linspace(0, closest_valid_t_surface_indices.shape[0]-1,  closest_valid_t_surface_indices.shape[0]).to(closest_valid_t_surface_indices.device).long()

        # Get the t_values at these points
        t_closest_to_reflection = t_valid_reflected[closest_valid_t_batch_indices, closest_valid_t_surface_indices]

        
        # Filter our which reflected rays with valid intersections
        valid_reflected_ray_indices =  (t_closest_to_reflection < 999.).nonzero(as_tuple=True)[0]
        t_valid = t_closest_to_reflection[valid_reflected_ray_indices]

        # If there are not surface intersections, leave alone
        if t_valid.shape[0] <= 0 :
            col = COL[indices[pointindex,0], indices[pointindex,1]]
            return col.float()
        
        """ Now we should have all the valid intersections from the glass reflections into the scene
        
        """
        ##### Now Onto Getting light from these points #####
        o_valid = o_glass[closest_valid_t_batch_indices[valid_reflected_ray_indices]]
        d_valid = d_reflected[closest_valid_t_batch_indices[valid_reflected_ray_indices]]
        
        p = o_valid+ t_valid*d_valid
        
        n_p =  normals[closest_valid_t_surface_indices[valid_reflected_ray_indices]]

        
        """ View Intersections coming from glass 
        o_np = (o_valid).cpu().numpy()
        d_np = (d_valid).cpu().numpy()
        valid_r = p.cpu().numpy()
        
        show_initrays = mlab.quiver3d(o_np[:,0], o_np[:,1],o_np[:,2],d_np[:,0], d_np[:,1],d_np[:,2], opacity=.1,   line_width=.5, scale_factor=1., color=(1.,0.,0.), mode='arrow')
        show_intersections = mlab.points3d(valid_r[:,0], valid_r[:,1], valid_r[:,2],  opacity=1.,  scale_factor=.1, color=(0.,0.,1.))
        
        for i in range(surfobj.num_surfaces):
            x_, y_, z_, tris, colz, _, _,_,_ = surfobj.get_surf_data(id=i)
            mlab.triangular_mesh(x_,y_,z_, tris, color=colz,opacity=.2, representation='surface')
        mlab.show()
        exit()
        """
        
        # Get Light Samples
        lh_samples = project_hemisphere_projection(p, light)
        o_ = p.unsqueeze(1).repeat(1,lh_samples.shape[1],1)
        d_ = (lh_samples-o_)

        """ Visually check the size of light-balls
        o = o_.cpu().numpy()
        d= d_.cpu().numpy()
        lh_samples = lh_samples.cpu().numpy()
        p = p.unsqueeze(1).repeat(1,lh_samples.shape[1],1).cpu().numpy()
        l = light.unsqueeze(0).repeat(1,p.shape[-2]).cpu().numpy()

        lightball = mlab.quiver3d(o[:,:,0], o[:,:,1],o[:,:,2],d[:,:,0], d[:,:,1],d[:,:,2],  line_width=.05, scale_factor=.05, color=(1.,0.,0.), mode='arrow')
        obj = mlab.points3d(lh_samples[:,:,0],lh_samples[:,:,1],lh_samples[:,:,2], scale_factor=1.2, color=(0.,1.,0.))
        obj2 = mlab.points3d(l[:,0],l[:,1],l[:,2], scale_factor=2., color=(0.,0.,1.))

        for i in range(surfobj.num_surfaces):
            x_, y_, z_, tris, colz, _, _,_,_ = surfobj.get_surf_data(id=i)
            mlab.triangular_mesh(x_,y_,z_, tris, color=colz,opacity=.2, representation='surface')
        print(o.shape, d.shape)
        mlab.show()
        exit()
        """
        
        # Get intersections for points
        v = torch.cross((V[:, 1] - V[:, 0]), (V[:, 2] - V[:, 0]), dim=1)
        v = torch.div(v, torch.linalg.norm(v, dim=1).unsqueeze(1)) # this should give us a,b,c in planar equation
        k = - torch.diagonal(torch.mm(v, V[:,0].T))
        v = v.unsqueeze(0).repeat(o_.shape[0],1,1)
        vo = torch.bmm(v, o_.transpose(1,2))
        vd = torch.bmm(v, d_.transpose(1,2))
        k1 = k.unsqueeze(0).unsqueeze(2).repeat(vo.shape[0],1,vo.shape[2])
        kvo = -torch.add(k1, vo)
        t_ = torch.div(kvo, vd).unsqueeze(-1)

        # Get masks for a surface being between lightsource and t
        between_mask =  torch.gt(t_, 0.0) * torch.lt(t_, 1.00)

        # Get mask for a point on its own surface
        t_surf_indices = closest_valid_t_surface_indices[valid_reflected_ray_indices]
        ordered_t_batch_indices = torch.linspace(0, t_surf_indices.shape[0]-1,  t_surf_indices.shape[0]).to(t_surf_indices.device).long()
        
        between_mask[:, t_surf_indices] = 0.

        # Calculate the global positions of intersections (only to check validity of surface)
        r = o_.unsqueeze(1).repeat(1,t_.shape[1],1,1) + \
            t_* d_.unsqueeze(1).repeat(1,t_.shape[1],1,1)
            
        # CHeck if valid intersections exist and applie on mask
        mask = get_off_surface_intersections_mask(r, V, between_mask).squeeze(-1)
        
        """ View Intersections coming from samples
        o_np = (o_valid).cpu().numpy()
        d_np = (d_valid).cpu().numpy()
        o_np2 = (o_).cpu().numpy()
        d_np2 = (d_).cpu().numpy()

        r_mirrored = o_.cpu().numpy()
        indices0, indices1, _ = (mask > 0).nonzero(as_tuple=True)

        r_samples = r[indices0, indices1].cpu().numpy()

        show_initrays = mlab.quiver3d(
            o_np[:,0], o_np[:,1],o_np[:,2],d_np[:,0], d_np[:,1],d_np[:,2], 
            opacity=.1,   line_width=.5, scale_factor=1., color=(1.,0.,0.), mode='arrow')
        
        # show_initrays2 = mlab.quiver3d(
        #     o_np2[...,0], o_np2[...,1], o_np2[...,2], d_np2[...,0], d_np2[...,1], d_np2[...,2], 
        #     opacity=.1,   line_width=.5, scale_factor=.05, color=(1.,0.,0.), mode='arrow')
        
        points_being_sampled = mlab.points3d(
            r_samples[...,0], r_samples[...,1], r_samples[...,2],  
            opacity=1.,  scale_factor=.1, color=(0.,1.,0.))

        points_being_mirrored = mlab.points3d(
            r_mirrored[...,0], r_mirrored[...,1], r_mirrored[...,2],  
            opacity=1.,  scale_factor=.1, color=(0.,0.,1.))
        
        for i in range(surfobj.num_surfaces):
            x_, y_, z_, tris, colz, _, _,_,_ = surfobj.get_surf_data(id=i)
            mlab.triangular_mesh(x_,y_,z_, tris, color=colz,opacity=.2, representation='surface')
        mlab.show()
        exit()
        """
        
        count_inbetween_surfaces = torch.gt(mask.sum(-2), 0.) # any obstruction = 1
        light_sample_result = ((count_inbetween_surfaces).sum(-1)/ count_inbetween_surfaces.shape[-1]).unsqueeze(-1).float()
        dark_mask = light_sample_result
        light_mask =  ((~count_inbetween_surfaces).sum(-1)/ count_inbetween_surfaces.shape[-1]).unsqueeze(-1).float()
        maskcombine = dark_mask+light_mask

        """ View Intersections (size dependant on count)
        o_np = (o_valid).cpu().numpy()
        d_np = (d_valid).cpu().numpy()
        r_mirrored = p.cpu().numpy()
        
        indices0, indices1, _ = (count_inbetween_surfaces > 0).nonzero(as_tuple=True)
        r_samples = r[indices0, indices1].cpu().numpy()

        show_initrays = mlab.quiver3d(
            o_np[:,0], o_np[:,1],o_np[:,2],d_np[:,0], d_np[:,1],d_np[:,2], 
            opacity=.1,   line_width=.5, scale_factor=1., color=(1.,0.,0.), mode='arrow')

        points_being_sampled = mlab.points3d(
            r_samples[...,0], r_samples[...,1], r_samples[...,2],  
            opacity=1.,  scale_factor=.1, color=(0.,1.,0.))

        points_being_mirrored = mlab.points3d(
            r_mirrored[...,0], r_mirrored[...,1], r_mirrored[...,2],  
            opacity=.5,  scale_factor=.1, color=(0.,0.,1.))
        
        for i in range(surfobj.num_surfaces):
            x_, y_, z_, tris, colz, _, _,_,_ = surfobj.get_surf_data(id=i)
            mlab.triangular_mesh(x_,y_,z_, tris, color=colz,opacity=.2, representation='surface')
        mlab.show()
        exit()
        """

        # Get the intensity by combining samples
        light_col = torch.tensor([1., 1., 1.], dtype=torch.float).to(valid_reflected_ray_indices.device).unsqueeze(0).repeat(valid_reflected_ray_indices.shape[0],1)        

        # Get the lambertian cosine term (and normalise it, you really need to normalise it or it never works..........)
        n_p = normals[closest_valid_t_surface_indices[valid_reflected_ray_indices]]
        p_norm = normalise_normals(light-p)
        costheta = (torch.diagonal(torch.mm(n_p, p_norm.T))/(torch.linalg.norm(n_p, dim=1)* torch.linalg.norm(p_norm, dim=1))).unsqueeze(-1).float()

        col = COL[indices[pointindex,0], indices[pointindex,1]].float()
        ray_colours = COL[closest_valid_t_surface_indices[valid_reflected_ray_indices]][:,0]

        brdf = costheta*light_col
        
        col[glass_indices[valid_reflected_ray_indices]] = .5*brdf + ray_colours.float()

        """ View Intersections (size dependant on count)
        n_np = (n_p).cpu().numpy()
        p_np = (p_norm).cpu().numpy()

        r_mirrored = p.cpu().numpy()
        
        points_being_mirrored = mlab.quiver3d(
            r_mirrored[...,0], r_mirrored[...,1], r_mirrored[...,2], p_np[...,0], p_np[...,1], p_np[...,2],
            opacity=.5,  scale_factor=1., color=(0.,0.,1.))

        points_being_mirrored = mlab.quiver3d(
            r_mirrored[...,0], r_mirrored[...,1], r_mirrored[...,2], n_np[...,0], n_np[...,1], n_np[...,2],
            opacity=.5,  scale_factor=1., color=(1.,0.,0.))
        
        for i in range(surfobj.num_surfaces):
            x_, y_, z_, tris, colz, _, _,_,_ = surfobj.get_surf_data(id=i)
            mlab.triangular_mesh(x_,y_,z_, tris, color=colz,opacity=.2, representation='surface')
        mlab.show()
        # exit()
        """

        return torch.clamp(col, 0., 1.).float()
        
"""Functions for tracing rays"""
def get_intersections(o,d,V):
    """Calculates  potential intersections

        Args:
            o,d: Tensor, contains the spatial componnts of rays shape (N, 3)
            V: Tensor, contains the global positions of each surface intersection shape(M,4,3)

        Returns
            t: Tensor, contains scalar for all rays and surfaces (M, N, 1)
    """
    v = torch.cross((V[:, 1] - V[:, 0]), (V[:, 2] - V[:, 0]), dim=1)
    v = torch.div(v, torch.linalg.norm(v, dim=1).unsqueeze(1)) # this should give us a,b,c in planar equation

    k = - torch.diagonal(torch.mm(v, V[:,3].T)).unsqueeze(0)

    O = o.view(o.shape[0], 3, 1)
    D = d.view(o.shape[0], 3, 1)
    B = v.view(1, v.shape[0], 3)
    vo = torch.matmul(B, O)
    vo = vo.permute(0, 1, 2).contiguous().view(o.shape[0], v.shape[0], 1)
    vd = torch.matmul(B, D)
    vd = vd.permute(0, 1, 2).contiguous().view(o.shape[0], v.shape[0], 1)

    k = k.unsqueeze(-1).repeat(vo.shape[0],1,1)

    k = -torch.add(k, vo)
    t = torch.div(k, vd)
    return t

def get_off_surface_intersections_mask(r, V, mask):
    """Calculates valid intersection and modifies a mask with previously determined validity

        Args:
            r: Tensor, contains the spatial positions of intersections
            V: Tensor, contains the global positions of each surface intersection
            mask: Tensor, contains a mask defining viable and unviable intersections

        Returns
            mask: Tensor, contains a mask of all valid intersections
    """
    a = V[:, 0].unsqueeze(0).unsqueeze(2).repeat(r.shape[0],1, r.shape[2],1)
    b = V[:, 1].unsqueeze(0).unsqueeze(2).repeat(r.shape[0],1, r.shape[2],1)
    c = V[:, 2].unsqueeze(0).unsqueeze(2).repeat(r.shape[0],1, r.shape[2],1)
    e = .000000000000000001

    # Solve Gam Nomincator and denominator
    A = ((a[...,0] * r[...,2])-(a[...,2] * r[...,0]))
    B = ((a[...,0]*b[...,2]) - (a[...,2] * b[...,0]))
    C = ((a[...,0] * r[...,1]) - (a[...,1] * r[...,0]))
    D = ((a[...,0] * b[...,1]) - (a[...,1] * b[...,0]))

    E = ((a[...,0]*c[...,2]) - (a[...,2]*c[...,0]))
    F = (((a[...,0]*b[...,2]) - (a[...,2]*b[...,0])) * ((a[...,1]*c[...,0]) - (a[...,0]*c[...,1])))
    G = ((a[...,0]*b[...,1]) - (a[...,1]*b[...,0]))

    # zeros = (D == 0.).nonzero(as_tuple=True)
    # D[zeros[0], zeros[1]] = e
    # zeros = (G == 0.).nonzero(as_tuple=True)
    # G[zeros[0], zeros[1]] = e

    g_nom = G*((A*D) - ((B * C)))
    # g_nom = A - ((B * C) / D)
    g_denom = D*( (E*G) + F )
    zeros = (g_denom == 0.).nonzero(as_tuple=True)
    g_denom[zeros[0], zeros[1]] = e
    gam = g_nom / g_denom
    # print(A, ((B * C) / D))
    # print(gam, r.shape)
    # exit()

    del A,B,C,D,E,F,G, g_nom, g_denom

    # Solve Beta parameter
    A = (a[...,0]*r[...,1]) - (a[...,1]*r[...,0])
    B = (gam*a[...,1]*c[...,0]) - (gam * a[...,0] * c[...,1])
    b_nom =  A+B
    b_denom = (a[...,0]*b[...,1]) - (a[...,1]*b[...,0])
    zeros = (b_denom == 0.).nonzero(as_tuple=True)
    b_denom[zeros[0], zeros[1]] = e
    beta = b_nom/b_denom

    del A,B,b_nom,b_denom

    a_nom = r[...,0]-((beta*b[...,0]) + (gam*c[...,0]))
    a_denom = a[...,0]
    zeros = (a_denom == 0.).nonzero(as_tuple=True)
    a_denom[zeros[0], zeros[1]] = e
    alpha = a_nom/ a_denom

    del a_nom,a_denom, zeros, a, b, c
    beta_mask = torch.gt(beta, 0.)
    gam_mask = torch.gt(gam, 0.)
    alpha_mask = torch.gt(alpha, 0.)
    
    conditionmasks = beta_mask*alpha_mask*gam_mask # (gam_mask*beta_mask* alpha_mask)#* beta_gam_mask)
    off_surf_indices = (conditionmasks == False).nonzero(as_tuple=True)

    mask[off_surf_indices[0], off_surf_indices[1], off_surf_indices[2]] = 0.

    return mask

def get_onsurface_surface_intersections(r, V):
    """Calculates valid intersection and returns a valid mask
        Args:
            r: Tensor, contains the spatial positions of intersections
            V: Tensor, contains the global positions of each surface intersection

        Returns
            mask: Tensor, contains a mask of all valid intersections
    """
    a = V[:, 0].unsqueeze(0).repeat(r.shape[0],1,1)
    b = V[:, 1].unsqueeze(0).repeat(r.shape[0],1,1)
    c = V[:, 2].unsqueeze(0).repeat(r.shape[0],1,1)
    e = .000000000000000001
    # Solve Gam Nomincator and denominator
    A = ((a[...,0] * r[...,2])-(a[...,2] * r[...,0]))
    B = ((a[...,0]*b[...,2]) - (a[...,2] * b[...,0]))
    C = ((a[...,0] * r[...,1]) - (a[...,1] * r[...,0]))
    D = ((a[...,0] * b[...,1]) - (a[...,1] * b[...,0]))

    E = ((a[...,0]*c[...,2]) - (a[...,2]*c[...,0]))
    F = (((a[...,0]*b[...,2]) - (a[...,2]*b[...,0])) * ((a[...,1]*c[...,0]) - (a[...,0]*c[...,1])))
    G = ((a[...,0]*b[...,1]) - (a[...,1]*b[...,0]))


    g_nom = G*((A*D) - ((B * C)))
    # g_nom = A - ((B * C) / D)
    g_denom = D*( (E*G) + F )
    zeros = (g_denom == 0.).nonzero(as_tuple=True)
    g_denom[zeros[0], zeros[1]] = e
    gam = g_nom / g_denom
    # print(A, ((B * C) / D))
    # print(gam, r.shape)
    # exit()

    del A,B,C,D,E,F,G, g_nom, g_denom

    # Solve Beta parameter
    A = (a[...,0]*r[...,1]) - (a[...,1]*r[...,0])
    B = (gam*a[...,1]*c[...,0]) - (gam * a[...,0] * c[...,1])
    b_nom =  A+B
    b_denom = (a[...,0]*b[...,1]) - (a[...,1]*b[...,0])
    zeros = (b_denom == 0.).nonzero(as_tuple=True)
    b_denom[zeros[0], zeros[1]] = e
    beta = b_nom/b_denom

    del A,B,b_nom,b_denom

    a_nom = r[...,0]-((beta*b[...,0]) + (gam*c[...,0]))
    a_denom = a[...,0]
    zeros = (a_denom == 0.).nonzero(as_tuple=True)
    a_denom[zeros[0], zeros[1]] = e
    alpha = a_nom/ a_denom

    del a_nom,a_denom, zeros, a, b, c
    beta_mask = torch.gt(beta, 0.0)
    gam_mask = torch.gt(gam, 0.0)
    alpha_mask = torch.gt(alpha, 0.0)
    
    # Execute condition on new Mask
    conditionmasks = beta_mask*alpha_mask*gam_mask # (gam_mask*beta_mask* alpha_mask)#* beta_gam_mask)

    on_surf_indices = (conditionmasks == True).nonzero(as_tuple=True)
    mask = torch.zeros_like(conditionmasks)

    mask[on_surf_indices[0], on_surf_indices[1]] = True

    return mask

"""Functions for accessing light samples"""
def project_hemisphere_projection(p, l):
    """Project Hemisphere onto local reflection vectors to get vectors of light samples

        Args:
            p: Tensor, global position of intersections
            l: Tensor, global position of light source

        Return:
            vecs: Tensor, contains all the component of sample rays from intersection p
    """
    assert p is not None, 'need point to project hemisphere'

    # Get global vector
    u = l-p
    
    # Set-up Local projection (along y-axis)
    u_norm = torch.linalg.norm(u,dim=1)
    p_local = torch.zeros_like(u)
    p_local[:,1] = u_norm
    # Get local vector
    v = torch.tensor([0.,0.,0.]).to(p_local.device) - p_local
    # Get Rotational Transformation
    u = u/u_norm.unsqueeze(1)

    v = v/torch.linalg.norm(v, dim=1).unsqueeze(1)
    c = torch.diagonal(torch.mm(v, u.T)).unsqueeze(1)
    w = torch.cross(v, u)

    vmat = torch.zeros(w.shape[0], 3, 3).to(w.device)
    vmat[:,0,1] = -w[:,2]
    vmat[:,0,2] = w[:,1]
    vmat[:,1,0] = w[:,2]
    vmat[:,1,2] = -w[:,0]
    vmat[:,2,1] = w[:,0]
    vmat[:,2,0] = -w[:,1]

    FFF =  (1. + c).unsqueeze(1)
    R = torch.eye(3).unsqueeze(0).repeat(vmat.shape[0],1,1).to(vmat.device) + vmat + (torch.bmm(vmat, vmat) /FFF)

    vecs = VECS.to(R.device).unsqueeze(0).repeat(R.shape[0],1,1).transpose(1,2)
    
    RV = torch.bmm(R, vecs)
    T = l.unsqueeze(0).unsqueeze(1).repeat(R.shape[0],vecs.shape[2], 1).transpose(1,2)

    vecs = (RV + T).transpose(1,2)

    return vecs