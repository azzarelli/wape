"""Extract mesh data for DataSynthesis
    
    Scene Collection Requirements:
        Use 3 folders: 'scene'    - for containing all objects
                       'cameras'   - contains all cameras (including novel & training views)
                       'lighting' - contains all lights
        Adjust camera 'focal length' (we use 50mm) - for pin-hole camera model in data generation
        Traingular all meshes before loading
        ALL materials must use 'Glass BSDF' node where 'color' is the colour and 'roughness'isrepresents the density of a NeRF
        You must be in 'Object mode' before running the script
        
    Notes:
        Run file in `scripting` tab of Blender
        Both novel and training views to be defined during data generation
         
"""
import bpy
import json
import numpy as np

save_dir = 'C:/to/your/folder/'

def select_name(name = "", extend = True ):
    """Select object given a name
    
        Args:
            name: str, object name
            extend: bool, choice to deselect selected objected before re-selecting 'name' obj
        Return:
            obj: Bpy.object, mesh object
    """
    # Reset current selection
    if extend == False:
        bpy.ops.object.select_all(action='DESELECT')
    
    # Fetch object by name
    ob = bpy.data.objects.get(name)
    
    # Check if object exists
    if ob == None:
        return None
    else: # Set the object to the active element
        bpy.context.view_layer.objects.active = ob
        return ob

def get_material_with_object_name(name = "", extend = True ):
    """Get the main material given an objects name
    
        Args:
            name, str, object name
            extend, bool, deselect objects prior to object selection
        Return:
            material_name, str, matrial name
    """
    # Deselect current selection
    if extend == False:
        bpy.ops.object.select_all(action='DESELECT')
        
    # Fetch object by name
    ob = bpy.data.objects.get(name)
    
    # Check if obj exists
    if ob == None:
        return None
    else:
        # Make object active selection
        bpy.context.view_layer.objects.active = ob
        
        # Check if material exists with object -> otherwise it will have to be added
        assert bpy.context.active_object.active_material.name != None, f'Need Material for object {name}'
        material_name = bpy.context.active_object.active_material.name
        
        return material_name

def equation_plane(triangles):
    """Get the equation of a plane to return the vertices of the plane

        Args:
            triangles, np.array, shape (3,3) contains the vertices of triangle
        Return:
            r[0].tolist(), list, the x-y-z coefficients for vector plane equation
            
        Note:
            This is no longer useful for the in the final implementation. Consider deleting function & use of returned vars.
    """
    # Get unit norm and d (i.e. abc and k params)
    n =  np.cross(triangles[:,1] - triangles[:,0], triangles[:,2] - triangles[:,0], axis=1)
    u = n / np.linalg.norm(n, axis=1, keepdims=True)
    k = -np.einsum('ij,ij->i', triangles[:, 0], u)
    
    # Get the paramete
    r1 = (triangles[:,1]+triangles[:,0])/2
    perr1 = r1[0].dot(u[0]) +k
    r2 = (triangles[:,1]+triangles[:,2])/2
    perr2 = r2[0].dot(u[0]) +k
    r3 = (triangles[:,2]+triangles[:,0])/2
    perr3 = r3[0].dot(u[0]) +k
    
    if perr1 < perr2 and perr1 < perr3:
        r = r1
    elif perr2 < perr1 and perr2 < perr3:
        r = r2
    else:
        r = r3

    return r[0].tolist()
        
"""
"""
     
# Deselect all Objects & select our 'scene' mesh
bpy.ops.object.select_all(action='DESELECT')

# Collect names & data of objects, camera and light relative to a collection 
objs = []
lights = []
cam_dict = {}

# For each collection
for b in bpy.data.collections:
    # Collect objects in our scene
    if b.name == 'scene':
        for o in b.objects:
            objs.append(o.name)
    # Collect light global position
    if b.name == 'lighting':
        for o in b.objects:
            lights.append([o.matrix_world.to_translation().x, o.matrix_world.to_translation().y, o.matrix_world.to_translation().z])
    # Collect camera matrices and focal points
    if b.name == 'cameras':
        for o in b.objects:
            foc = bpy.data.cameras[o.name].lens
            sx,sy = 1.,1.
            fx,fy = foc*sx, foc*sy
        
            cam_dict[o.name]={
                'f':[fx,fy],
                'o': [o.matrix_world.to_translation().x, o.matrix_world.to_translation().y, o.matrix_world.to_translation().z],
                'd': [o.matrix_world.to_euler().x, o.matrix_world.to_euler().y, o.matrix_world.to_euler().z],
                'world':[list(row) for row in o.matrix_world]
                }

# Fetch all light objects
light_dict = {}
for id, l in enumerate(lights):
    light_dict[str(id)] = l

# Initialise the Mesh data
data_dict = {
    "novel_view_index":[] # TODO - find a way to define novel views at this stage rather than later
}
co_dict = {} # colour and opacity dictionary

# For each object in
for id, o in enumerate(objs):
    print('Working on ...', o)
    scene_obj = select_name(o)
    
    # Get object matrial
    material = get_material_with_object_name(o)
    assert bpy.data.materials[material].node_tree.nodes["Glass BSDF"], f'Material {material} needs to be Glass BSDF'
    rgb = bpy.data.materials[material].node_tree.nodes["Glass BSDF"].inputs["Color"].default_value
    
    # Fetch colour & density values from object material properties
    c = (rgb[0], rgb[1], rgb[2])
    opa = bpy.data.materials[material].node_tree.nodes["Glass BSDF"].inputs["Roughness"].default_value
    
    # Flatten all surfaces
    bpy.ops.object.editmode_toggle() # go into edit from object mode
    bpy.ops.mesh.face_make_planar(repeat=1000) # for potentially curved triangles
    
    # Sanity-check if object exists
    if scene_obj is not None:
        data_dict[str(id)] = {}
        mat = scene_obj.matrix_world
        
        co_dict[id] = {'c':c, 'o':opa}
        
        # For each triangular surface in a mesh
        for id_s, face in enumerate(scene_obj.data.polygons):
            # Ensure triangle
            assert len(face.vertices) == 3, 'Some faces are not triangles! We only want traingles !'
            verts = []
            
            # Get global coordinates of mesh
            for v in face.vertices:
                v = scene_obj.data.vertices[v].co
                loc = mat @ v
                verts.append([loc.x, loc.y, loc.z])
            
            # Fetch the vertices, xyz cooefficients of plane equation and normals of plane
            verts = np.array([verts])
            xyz  = equation_plane(verts)
            verts = [verts[0,0].tolist(), verts[0,1].tolist(), verts[0,2].tolist(), xyz]
            nrm_fixed = (face.normal)
            norms = [nrm_fixed[0], nrm_fixed[1], nrm_fixed[2]]
            
            # Add to our data dict
            data_dict[str(id)][str(id_s)] = {"vertices":verts, "normals":norms}
    # Untoggle edit mode
    bpy.ops.object.editmode_toggle() # return to edit mode

# Construct save-file and save
scene_dict = {
    "scene":data_dict,
    "lights":light_dict,
    "c_o":co_dict,
    "cameras":cam_dict
}
with open(save_dir+"test.json", "w") as outfile:
    json.dump(scene_dict, outfile)