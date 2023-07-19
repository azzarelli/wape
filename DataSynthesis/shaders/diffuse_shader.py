"""Diffuse Shader for Explicit Field Shading
"""
import numpy as np
from mayavi import mlab
import torch

from utils.normals import normalise_normals
from utils.normals import *

class DiffuseShader(torch.nn.Module):
    def __init__(self, dispersion=1., sample=(5j, 5j)):
        super().__init__()
        self.sample = sample

        self.vecs = self.generate_hemisphere(dispersion, sample)

    def generate_hemisphere(self, r, samples):
        """Generate the hemispheric surface samples for caluclating solid angles for sampling light intensity
            
            Args:
                r, Tensor, position of a point to sample light intensity (in local space, this is fixed)
                samples, list(int), which contain the size of samples along the longitude and latitude of a hemisphere
            
            Return:
                vec, tensor, contains the x,y,z position of each sample 
        """
        u, v = np.mgrid[0:np.pi:samples[0], 0:np.pi:samples[1]]
        x = (r*np.cos(u) * np.sin(v)).flatten()
        y = (r*np.sin(u) * np.sin(v)).flatten()
        z = (r*np.cos(v)).flatten()
        vec = np.c_[x,y, z]
        return torch.tensor(vec).float()

    def project_hemisphere_projection(self, p, l):
        """Proejct the hemispheric samples into global space

            Args:
                p, Tensor, position of point in global space
                l, Tensor, hemisphere in global space
            Return:
                vecs, Tensor, position of projected light samples in global space
        """
        assert p is not None, 'need point to project hemisphere'

        # Get global vector
        u = l-p
        
        # Set-up Local projection (along y-axis)
        u_norm = torch.linalg.norm(u,dim=1)
        p_local = torch.zeros_like(u, dtype=torch.float)
        p_local[:,1] = u_norm
        # Project the light-to-surface vector in the local space
        v = torch.tensor([0.,0.,0.], dtype=torch.float).to(p_local.device) - p_local
        # Get Rotational Transformation in local space
        u = u/u_norm.unsqueeze(1)
        v = v/torch.linalg.norm(v, dim=1).unsqueeze(1)
        c = torch.diagonal(torch.mm(v, u.T)).unsqueeze(1)
        w = torch.cross(v, u)
        vmat = torch.zeros(w.shape[0], 3, 3).to(w.device) # this defines our rotational matrix transform for projection
        vmat[:,0,1] = -w[:,2] # assign the indices of vector w to the indices of the matrix vmat
        vmat[:,0,2] = w[:,1]
        vmat[:,1,0] = w[:,2]
        vmat[:,1,2] = -w[:,0]
        vmat[:,2,1] = w[:,0]
        vmat[:,2,0] = -w[:,1]
        FFF =  (1 + c).unsqueeze(1) 
        # The following defines the transformation matrix for global project of rotation
        R = torch.eye(3).unsqueeze(0).repeat(vmat.shape[0],1,1).to(vmat.device) + vmat + (torch.bmm(vmat, vmat) /FFF)
        
        # Apply the rotation and shift matrices for projecting hemispheric samples
        vecs = self.vecs.to(R.device).unsqueeze(0).repeat(R.shape[0],1,1).transpose(1,2)
        RV = torch.bmm(R, vecs) # Rotation and scale
        T = l.unsqueeze(0).unsqueeze(1).repeat(R.shape[0],vecs.shape[2], 1).transpose(1,2) # shift
        vecs = (RV + T).transpose(1,2)

        return vecs
    
    def forward(self, V, indices, pointindex, COL, OPA, p, l, normals, it):
        """Step the tracer through a batch of points in a scene

            Args:
                V, Tensor, surface vertices in current batch
                indices, Tensor, valid indices for surface intersection
                pointindex, Tensor, pointer to position of our batch in the global stack (of valid intersections)
                COL,OPA, Tensor, global colour and opacity matrices
                p, Tensor, the position of intersections in global space
                l, Tensor, the position of light source in global space,
                normals, Tensor, surface normals
                it, int, dataloader index
            
            Return:
                render, Tensor, resulting global colour matrix which has been shaded
        """
        # Define the local position of our data in the data loader stack
        pointindex = pointindex.squeeze(0).long()
        local_pointindex = pointindex % pointindex.shape[0]

        # Define the light-samples and parameters for sampling light intensity
        lh_samples = self.project_hemisphere_projection(p, l)
        o = p.unsqueeze(1).repeat(1,lh_samples.shape[1],1) # origin and direction vectors from target intersections to lightsource
        d = lh_samples-o

        """ Visually check the size of light-balls
        o = o.cpu().numpy()
        d= d.cpu().numpy()
        lh_samples = lh_samples.cpu().numpy()
        p = p.unsqueeze(1).repeat(1,lh_samples.shape[1],1).cpu().numpy()
        l = l.unsqueeze(0).repeat(1,p.shape[-2]).cpu().numpy()
        lightball = mlab.quiver3d(o[0,:,0], o[0,:,1],o[0,:,2],d[0,:,0], d[0,:,1],d[0,:,2],  line_width=.05, scale_factor=.05, color=(1.,0.,0.), mode='arrow')
        obj = mlab.points3d(lh_samples[0,:,0],lh_samples[0,:,1],lh_samples[0,:,2], scale_factor=1.2, color=(0.,1.,0.))
        obj2 = mlab.points3d(l[:,0],l[:,1],l[:,2], scale_factor=2., color=(0.,0.,1.))
        print(o.shape, d.shape)
        mlab.show()
        exit()
        """

        # Get surface intersections
        t = get_intersection(o.flatten(0,1), d.flatten(0,1), V)
        t = t.view(o.shape[0], V.shape[0], o.shape[1])

        # Get near and far mask conditions for light-obsertuction between surface point and light
        infront_mask =  torch.gt(t, -0.0001) # bound this between 0 and 1 (with a tiny little precision errorr)
        inbetween_mask =  torch.lt(t, 1.00) # Our light projection ensures that t = 1 is the position of the light hemisphere relative to the pos of the target point
        
        # Remove the inclusion of the plane which our target points lie on
        empty = torch.zeros_like(inbetween_mask).to(inbetween_mask.device)
        empty[local_pointindex, indices[pointindex,0], :] = 1.
        full = ~empty

        # Combine the masks for ignoring invalid intersections
        viable_mask = infront_mask * inbetween_mask * full 

        # Find the final mask for all non-valid intersection by apply barycentric surface conditions
        r = o.unsqueeze(1).repeat(1,t.shape[1],1,1) + t.unsqueeze(-1).repeat(1,1,1,3)* d.unsqueeze(1).repeat(1,t.shape[1],1,1)
        mask = get_point_light_surface_intersections(r, V, viable_mask)

        """ Visually check the intersections between surfaces and rays
        mask3 = mask.unsqueeze(-1).repeat(1,1,1,3)
        r = r*mask3
        point_in_batch = 0
        sample_point = -1
        r_select = r[point_in_batch, :, sample_point] #.cpu().numpy()
        r_select_indices = (r_select.sum(-1) > 0.0001).nonzero(as_tuple=True)[0]
        r_select = r_select[r_select_indices, :].cpu().numpy()

        obj2 = mlab.points3d(r_select[:,0],r_select[:,1],r_select[:,2], scale_factor=1., color=(.0,1.,0.))

        o = o.cpu().numpy()
        d = d.cpu().numpy()
        pointray = mlab.quiver3d(
            o[point_in_batch,sample_point,0], o[point_in_batch,sample_point,1],o[point_in_batch,sample_point,2],
            d[point_in_batch,sample_point,0], d[point_in_batch,sample_point,1],d[point_in_batch,sample_point,2],  
            line_width=.05, scale_factor=.05, color=(1.,0.,0.), mode='arrow'
        )        

        l = l.unsqueeze(0).cpu().numpy()
        obj1 = mlab.points3d(l[:,0],l[:,1],l[:,2], scale_factor=1., color=(0.,0.,1.))

        mlab.show()
        exit()
        """

        # Assign local colour and opacity matrices
        col = COL[indices[pointindex,0], indices[pointindex,1]]
        opa = OPA[indices[pointindex,0], indices[pointindex,1]]
        opacity_of_casted_ray = torch.clamp(opa, 0., 1)

        # Get the indices of glass materials
        diffuse_indices = (opacity_of_casted_ray > .999).nonzero(as_tuple=True)[0]
        
        # Define the power of reflector the higher the power the more reflection
        power = .8 # TODO: control this externally - we found this best suited for the level of light in our scene but can be increased
                    # TODO: see reflection shader...
        light_col = power*torch.tensor([1., 1., 1.]).to(col.device).unsqueeze(0).repeat(col.shape[0],1)        


        # Calculate costheta angular reflection intensity
        n =  normals[indices[pointindex,0]]
        reflected_ray = (l-p)
        # print(n.shape, reflected_ray.shape)
        # exit()
        p_norm = normalise_normals(reflected_ray)
        costheta = torch.diagonal(torch.mm(n, p_norm.T))/(torch.linalg.norm(n, dim=1)* torch.linalg.norm(p_norm, dim=1))
        
        # Get the highlights and shadow masks
        sun_mask = torch.clamp(((~torch.gt((mask.sum(-2)), 0.)).sum(-1)/ 25.).unsqueeze(1), 0.,1.)
        shadow_mask = torch.clamp(((torch.gt((mask.sum(-2)), 0.)).sum(-1)/ 25.).unsqueeze(1), 0.,1.)
        chosenmask = sun_mask + shadow_mask #(sun_mask+shadow_mask)/2.
        
        # Create additive colour (colour 0,0,0 initially absent of additive col)
        add_colour = torch.zeros_like(col).to(col.device).float()
        all_colours_transformed = (light_col  * costheta.unsqueeze(1) *  chosenmask).squeeze(1) # /torch.pi
        add_colour[diffuse_indices] = all_colours_transformed[diffuse_indices]
        render = col + add_colour
        
        """ Visualise the angle between light and surface
        if it > 20:

            o_np = p.cpu().numpy()
            d_np = reflected_ray.cpu().numpy()
            n_np = n.cpu().numpy()
            n_o_np = p.cpu().numpy()
            print(n_np)

            # light_vectors = mlab.quiver3d(o_np[...,0], o_np[...,1],o_np[...,2], d_np[...,0], d_np[...,1],d_np[...,2], opacity=.1, line_width=.05, color=(0.,1.,0.), mode='arrow')
            surface_normals = mlab.quiver3d(n_o_np[...,0], n_o_np[...,1],n_o_np[...,2], n_np[...,0], n_np[...,1],n_np[...,2], opacity=.1, line_width=.05, color=(0.,0.,1.), mode='arrow', scale_factor=1.,)

            if it == 55:
                mlab.show()
                exit()
        """  

        return torch.clamp(render, 0.,1.) .float()


def get_intersection(o, d, V):
    """Get the intersection between rays and surfaces

        Notes:
            Similar function has been used in Solve.py
        
        Args:
            o,d , Tensor, origin and directios of rays
            V, Tensor, vertices of surface 
        
        Returns:
            t, Tensor, scalar distance of point along vector
    """
    v = torch.cross((V[:, 1] - V[:, 0]), (V[:, 2] - V[:, 0]), dim=1)
    v = torch.div(v, torch.linalg.norm(v, dim=1).unsqueeze(1))

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

def get_abcr(r, V):
    """Get a,b,c, parameters for barycentric function and return a mask of indices whre index=0 means barycentric condition violation

        Args:
            r, Tensor, position of intersection in global space
            
        Returns:
            V, Tensor, vertices of surface
    """
    a = V[:, 0].unsqueeze(0).unsqueeze(2).repeat(r.shape[0],1, r.shape[2],1).flatten()
    b = V[:, 1].unsqueeze(0).unsqueeze(2).repeat(r.shape[0],1, r.shape[2],1).flatten()
    c = V[:, 2].unsqueeze(0).unsqueeze(2).repeat(r.shape[0],1, r.shape[2],1).flatten()
    r = r.flatten()
    return a, b, c, r

def get_point_light_surface_intersections(r, V, mask):
    """Perform barycentric surface conditions

        Args:
            r, Tensor, position of intersection in global space
            V, Tesnor, vertices of surface
            mask, Tensor, mask of intersections which was previsouly defined
        Returns:
            mask

    """
    SHAPE = mask.shape
    a,b,c,r = get_abcr(r,V)
    r = r.flatten()
    
    # Solve Gam Nomincator and denominator
    A = ((a[...,0] * r[...,2])-(a[...,2] * r[...,0]))
    B = ((a[...,0]*b[...,2]) - (a[...,2] * b[...,0]))
    C = ((a[...,0] * r[...,1]) - (a[...,1] * r[...,0]))
    D = ((a[...,0] * b[...,1]) - (a[...,1] * b[...,0]))

    E = ((a[...,0]*c[...,2]) - (a[...,2]*c[...,0]))
    F = (((a[...,0]*b[...,2]) - (a[...,2]*b[...,0])) * ((a[...,1]*c[...,0]) - (a[...,0]*c[...,1])))
    G = ((a[...,0]*b[...,1]) - (a[...,1]*b[...,0]))

    g_nom = G*((A*D) - ((B * C)))
    g_denom = D*( (E*G) + F )
    gam = g_nom / g_denom

    del A,B,C,D,E,F,G, g_nom, g_denom

    # Solve Beta parameter
    A = (a[...,0]*r[...,1]) - (a[...,1]*r[...,0])
    B = (gam*a[...,1]*c[...,0]) - (gam * a[...,0] * c[...,1])
    b_nom =  A+B
    b_denom = (a[...,0]*b[...,1]) - (a[...,1]*b[...,0])
    beta = b_nom/b_denom

    del A,B,b_nom,b_denom

    a_nom = r[...,0]-((beta*b[...,0]) + (gam*c[...,0]))
    a_denom = a[...,0]
    alpha = a_nom/ a_denom

    del a_nom,a_denom, a, b, c
    beta_mask = torch.gt(beta, 0.)
    gam_mask = torch.gt(gam, 0.)
    alpha_mask = torch.gt(alpha, 0.)
    
    # Apply barycentric conditional masks
    conditionmasks = beta_mask*alpha_mask*gam_mask 
    off_surf_indices = (conditionmasks == False).nonzero(as_tuple=True)

    mask = mask.flatten(0,2)
    mask[off_surf_indices[0]] = 0.
    mask = mask.view(SHAPE)
    return mask