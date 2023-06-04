import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
import random
import time

EPSILON = 0.001
objects_lst = []
materials_lst = []
surfaces_lst = [] 
lights_lst = []

def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    global objects_lst, materials_lst, surfaces_lst, lights_lst
    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    objects_lst = objects
    image_array = np.zeros((args.height, args.width, 3))
    #rays_array = np.zeros((args.height, args.width),dtype=object)
    #associated_surfaces = np.zeros((args.height, args.width),dtype=object)
    #need to use np.vectorize on the lambda expression in order to insert it into the rays_array
    init_lists()
    #measure time
    start_time = time.time()

    #extract parameters and name them in the same way as in the slides
    p_0_cam = np.array(camera.position)
    w = camera.screen_width
    d = camera.screen_distance
    v_to = (np.array(camera.look_at)-np.array(camera.position))/np.linalg.norm(np.array(camera.look_at)-np.array(camera.position))
    #print("v_to: " +str(v_to))
    v_up = np.array(camera.up_vector)/np.linalg.norm(np.array(camera.up_vector)) #TODO: find a perpendicular vector
    #print("v_up: " +str(v_up))
    #TODO: check if we should keep it that way or do it according to v_to
    rx, ry = args.width, args.height
    #image center
    p_c = p_0_cam+d*v_to
    # do the cross product and normalize
    v_right = np.cross(v_to,v_up)/np.linalg.norm(np.cross(v_to,v_up))
    v_up_tilda = np.cross(v_to,v_right)/np.linalg.norm(np.cross(v_to,v_right))
    R = w/rx
    #TODO: ADDITION
    screen_height = (float(args.height)/float(args.width))*w
    sinx = -1*v_to[1]
    cosx = np.sqrt(1-sinx*sinx)
    siny = -1*v_to[0]/cosx
    cosy = v_to[2]/cosx
    M = np.array([[cosy,0,siny],[-1*sinx*siny,cosx,sinx*cosy],[-1*cosx*siny,-1*sinx,cosx*cosy]])
    v_x = (np.array([1,0,0]) @ M)/np.linalg.norm(np.array([1,0,0]) @ M)
    v_y = (np.array([0,-1,0]) @ M)/np.linalg.norm(np.array([0,-1,0]) @ M)
    p_0 = p_c-0.5*w*v_x-0.5*screen_height*v_y
    #TODO: ADDITION END
    count = 0
    #iterate over all the pixels
    for i in range(args.height):
        #if count==1:
         #       break
        #TODO: ADDITION
        p = p_0+(i*(screen_height/args.height))*v_y
        #TODO: ADDITION END
        for j in range(args.width):
            #for pixel (i,j)
            #if count==1:
             #   break
            if (count%1000==0):
                print("count: "+str(count))
            #Shoot a ray through each pixel in the image
            #Discover the location of the pixel on the camera’s screen (using camera parameters).
            #TODO: ADDITION
            #p = p_c+(j-np.floor(rx/2))*R*v_right-\
             #   (i-np.floor(ry/2))*R*v_up_tilda

            #Construct a ray from the camera through that pixel.
            V = (p-p_0_cam)/np.linalg.norm(p-p_0_cam)
            #TODO: ADDITION END
            ray = lambda t: p_0_cam + t*V
            #rays_array[i,j] = np.vectorize(ray)

            closest_surface, min_t = find_intersection(p_0_cam,V)
            #print("min_t = "+str(min_t) +" closest obj: " +str(closest_surface))
            #here we've got the closest surface saved together with the t_value
            #might not need this line
            #associated_surfaces[i,j] = closest_surface
            #Compute the color of the surface:
            #TODO: ADDITION
            intersect_point = ray(min_t)
            color = find_color(p_0_cam,intersect_point,V,closest_surface,scene_settings,1,camera)
            #TODO: ADDITION END


            image_array[i,j,0] = 255*color[0]
            image_array[i,j,1] = 255*color[1]
            image_array[i,j,2] = 255*color[2]
            #print("imcolor: " +str(image_array[i,j]))
            count+=1
            #TODO: ADDITION
            p+=v_x*(w/args.width)
            #TODO: ADDITION END


            


    #print("image array: "+str(image_array))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    # Save the output image
    save_image(image_array)

def find_intersection(p_0,V):
    global surfaces_lst
    closest_surface = 0
    min_t = float('inf')
    for surf in surfaces_lst:
        #Check the intersection of the ray with all surfaces in the scene (you can add \
        # optimizations such as BSP trees if you wish but they are not mandatory).
        #Find the nearest intersection of the ray. This is the surface that will be
        # seen in the image.
        if isinstance(surf,Cube):
            #extract parameters
            cube_mid = np.array(surf.position)
            edge_len = surf.scale
            min_x, min_y, min_z = min(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
            min(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), min(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
            max_x, max_y, max_z = max(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
            max(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), max(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
            t_min_x, t_max_x = min_x-p_0[0]/V[0], max_x-p_0[0]/V[0]
            t_min_y, t_max_y = min_y-p_0[1]/V[1], max_y-p_0[1]/V[1]
            t_min_z, t_max_z = min_z-p_0[2]/V[2], max_z-p_0[2]/V[2]
            t_enter = max(t_min_x,t_min_y,t_min_z)
            t_exit = min(t_max_x,t_max_y,t_max_z)
            if (t_enter>t_exit) or (t_max_x<0) or (t_max_y<0) or (t_max_z<0) or (t_enter<=0):
                #no intersection
                continue
            #there is an intersection, t=t_enter
            if (t_enter<=min_t):
                min_t = t_enter
                closest_surface = surf
        elif isinstance(surf,Sphere):
            #geometric approach
            O = np.array(surf.position)
            r = surf.radius
            L = O-p_0
            t_ca = np.dot(L,V)/(np.linalg.norm(L)*np.linalg.norm(V))
            if t_ca<0:
                continue
            d_sq = np.sum(np.square(L))-np.square(t_ca)
            if d_sq > r**2:
                continue
            t_hc = np.sqrt(r**2-d_sq)
            sphere_min_t = t_ca-t_hc
            if (sphere_min_t<=0):
                continue
            #print("sphere instersect")
            if (sphere_min_t<=min_t):
                #need to update closest surface
                min_t =sphere_min_t
                closest_surface = surf




            """
            O = np.array(obj.position)
            r = obj.radius
            #need to solve a quadratic equation
            a = 1
            b = 2*np.dot(V,p_0-O)
            c = ((p_0[0]-O[0])**2+(p_0[1]-O[1])**2+(p_0[2]-O[2])**2)-r**2
            delta = b**2-4*a*c
            if (delta<0): #a complex number - ray does not intersect with the sphere
                continue
            t1 = (-b+np.sqrt(delta))/(2*a)
            t2 = (-b-np.sqrt(delta))/(2*a)
            if (t1<0):
                sphere_min_t = t2
            if (t2<0):
                sphere_min_t = t1
            else:
                sphere_min_t = min(t1,t2)
            if sphere_min_t<=0:
                continue
            if (sphere_min_t<=min_t):
                #need to update closest surface
                min_t =sphere_min_t
                closest_surface = obj
        """
        elif isinstance(surf, InfinitePlane):
            #extract parameters
            N = np.array(surf.normal)
            d_plane = surf.offset
            p0_dot_N = np.dot(p_0,N)
            V_dot_N = np.dot(V,N)
            t_plane = (d_plane-p0_dot_N)/V_dot_N
            if t_plane<=0:
                continue
            if t_plane<=min_t:
                min_t = t_plane
                closest_surface = surf

            """
            #extract parameters
            N = np.array(obj.normal)
            d_plane = obj.offset
            #need to find a point on the plane
            if N[2]!=0:
                plane_point = np.array((0.0,0.0,-d_plane/N[2]))
            elif N[1]!=0:
                plane_point = np.array((0.0,-d_plane/N[1],0.0))
            elif N[0]!=0:
                plane_point = np.array((-d_plane/N[0],0.0,0.0))
            dot_prod = np.dot(V,N)
            if (np.abs(dot_prod)<EPSILON):
                #ray is parallel or nearly parallel to the plane - no intersection
                continue
            #there could still be an intersection
            #TODO: check if valid
            t_plane = np.dot(N,plane_point-p_0)/dot_prod
            if t_plane<=0: #no intersection
                continue
            if t_plane<=min_t:
                min_t = t_plane
                closest_surface = obj
                """
    #if isinstance(closest_surface,Sphere):
        #print("Sphere chosen")
    return closest_surface, min_t

def find_color(p_0_cam,intersect_point,V,surface,scene,depth,camera):
    #Add the value it induces on the surface.
    #Find out whether the light hits the surface or not:
    #Shoot rays from the light towards the surface
    #Find whether the ray intersects any other surfaces before the required
    # surface - if so, the surface is occluded from the light and the light does
    # not affect it (or partially affects it because of the shadow intensity parameter).
    #output color =
    #(background color) · transparency
    #+(diffuse + specular) · (1 − transparency)
    #+(reflection color)
    #print("depth = " +str(depth))
    if surface==0:
        #return background color if the original ray does not intersect any surface
        #print("background color: "+str(scene.background_color))
        return np.array(scene.background_color)
    global lights_lst, materials_lst
    mat_id = surface.material_index
    transparency = materials_lst[mat_id-1].transparency
    diffuse_color = np.array(materials_lst[mat_id-1].diffuse_color)
    specular_color = np.array(materials_lst[mat_id-1].specular_color)
    reflection_color = np.array(materials_lst[mat_id-1].reflection_color)
    #print("reflection_color: "+str(reflection_color))
    phong_spec_efficiency = materials_lst[mat_id-1].shininess
    #p is the ray's intersection with the surface/pixel's location in the real world
    N = find_surface_normal(surface,p_0_cam,intersect_point)
    diff_color = np.array((0.0,0.0,0.0))
    spec_color = np.array((0.0,0.0,0.0))
    color = np.array((0.0,0.0,0.0))
    for light in lights_lst:
        #extract light data
        light_pos = np.array(light.position)
        light_color = np.array(light.color)
        light_specular_intensity = light.specular_intensity
        #it's a light
        light_intensity = find_light_intensity(light,intersect_point,scene,surface)
        #print("light_intensity: " +str(light_intensity))
        I_L = light_intensity*light_color
        #print("I_L: " +str(I_L))
        #create a ray from the light to the object
        L = (light_pos-intersect_point)/np.linalg.norm(light_pos-intersect_point)
        N_dot_L = max(0,np.dot(L,N)/(np.linalg.norm(N)*np.linalg.norm(L)))
        #print(np.dot(N,L)/(np.linalg.norm(N)*np.linalg.norm(L)))
        #print("N_dot_L: " +str(N_dot_L))

        R = (2*N_dot_L*N-L)/np.linalg.norm(2*N_dot_L*N-L)
        R_dot_V = max(0,np.dot(R,-1*V)/(np.linalg.norm(V)*np.linalg.norm(R)))
        #TODO: check if R_V or V_R

        #calculate diffuse_color
        diff_color += I_L*diffuse_color*N_dot_L
        #print("I_L: "+str(I_L))
        #print("N_dot_L: "+str(N_dot_L))
        #print("diff_color: " +str(diff_color))

        #calculate specular color
        spec_color += I_L*specular_color*light_specular_intensity*\
                            R_dot_V**phong_spec_efficiency
        #print("spec_color: " +str(spec_color))
        color += (1-transparency)*(diff_color+spec_color)
        #print("color: " +str(color))

    #print("max recursion:" +str(scene.max_recursions))
    #check whether we've already reached our depth maximum. If so - return the color
    #If not - need to further check the reflections
    if (depth<scene.max_recursions):
        #need to handle reflection - every time a ray hits the surface, it reflects back the light
        #let's consruct the R vector
        V_dot_N = np.dot(V,N)/(np.linalg.norm(N)*np.linalg.norm(V))
        R_reflect = (V-2*V_dot_N*N)/np.linalg.norm(V-2*V_dot_N*N)
        closest_next_surface, min_t_to_next_surface = find_intersection(intersect_point,R_reflect)
        next_p = intersect_point+min_t_to_next_surface*R_reflect
        reflect_color = find_color(p_0_cam,next_p,R_reflect,closest_next_surface,scene,depth+1,camera)
        color += reflect_color*reflection_color
        #print("color: " +str(color) + ", depth: " +str(depth))

        #print("transparency: " +str(transparency))
        if transparency!=0: #the material is somewhat transparent and not opaque
            #need to calculate reflections
            exit_point = find_ray_exit_point(intersect_point,V,surface,camera)
            closest_next_surface, min_t_to_next_surface = find_intersection(exit_point,V)
            next_p = exit_point+min_t_to_next_surface*V
            transp_color = find_color(p_0_cam,next_p,V,closest_next_surface,scene,depth,camera)
            
            color += transparency*transp_color
            #print("color: " +str(color) + ", depth: " +str(depth))



    #print("final color: " +str(color) + ", depth = " +str(depth))
    return np.array((min(color[0],1.0),min(color[1],1.0),min(color[2],1.0)))

def find_surface_normal(surface,p_0_cam,intersect_point):
    if isinstance(surface,InfinitePlane):
        return np.array(surface.normal)/np.linalg.norm(np.array(surface.normal))
    if isinstance(surface,Sphere):
        sphere_center = np.array(surface.position)
        return (intersect_point-sphere_center)/np.linalg.norm(intersect_point-sphere_center)
    if isinstance(surface,Cube):
        """PREV_IMP"""
        cube_center = np.array(surface.position)
        edge_len = surface.scale
        #need to find which plane of the six edges includes the intersection point
        #divide to cases
        #plane's equation is x = d
        d = cube_center[0]-edge_len/2
        if (intersect_point[0]-d<EPSILON):
            return np.array((1.0,0.0,0.0))
        #plane's equation is x = -d
        if (intersect_point[0]+d<EPSILON):
            return np.array((1.0,0.0,0.0))
        #plane's equation is y = d
        d = cube_center[1]-edge_len/2
        if (intersect_point[1]-d<EPSILON):
            return np.array((0.0,1.0,0.0))
        #plane's equation is y = -d
        if (intersect_point[1]+d<EPSILON):
            return np.array((0.0,1.0,0.0))
        #plane's equation is z = d
        d = cube_center[2]-edge_len/2
        if (intersect_point[2]*1-d<EPSILON):
            return np.array((0.0,0.0,1.0))
        #plane's equation is z = -d
        if (intersect_point[2]*1+d<EPSILON):
            return np.array((0.0,0.0,1.0))
    
def find_ray_exit_point(p,V,surface,camera):
    p_0 = np.array(camera.position)
    if isinstance(surface, InfinitePlane):
        return p
    if isinstance(surface, Sphere):
        O = np.array(surface.position)
        r = surface.radius
        #need to solve a quadratic equation
        a = 1
        b = 2*np.dot(V,p_0-O)
        c = ((p_0[0]-O[0])**2+(p_0[1]-O[1])**2+(p_0[2]-O[2])**2)-r**2
        delta = b**2-4*a*c
        if (delta<0): #a complex number - ray does not intersect with the sphere
            #shouldn't happen here
            return p
        t1 = (-b+np.sqrt(delta))/(2*a)
        t2 = (-b-np.sqrt(delta))/(2*a)
        sphere_max_t = max(t1,t2)
        return p_0+sphere_max_t*V
    
    if isinstance(surface, Cube):
        #extract parameters
        cube_mid = np.array(surface.position)
        edge_len = surface.scale
        min_x, min_y, min_z = min(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
        min(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), min(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
        max_x, max_y, max_z = max(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
        max(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), max(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
        t_min_x, t_max_x = min_x-p_0[0]/V[0], max_x-p_0[0]/V[0]
        t_min_y, t_max_y = min_y-p_0[1]/V[1], max_y-p_0[1]/V[1]
        t_min_z, t_max_z = min_z-p_0[2]/V[2], max_z-p_0[2]/V[2]
        t_exit = min(t_max_x,t_max_y,t_max_z)
        return p_0+t_exit*V


def find_light_intensity(light,p,scene,surface):
    #Produce soft shadows, as explained below:
        #Shoot several rays from the proximity of the light to the surface.
        #Find out how many of them hit the required surface
    #extract light parameters
    light_pos = np.array(light.position)
    light_shadow_intensity = light.shadow_intensity
    light_radius = light.radius
    #shoot a ray from the light position to the surface point
    ray_vector = (p-light_pos)/np.linalg.norm(p-light_pos)
    #find a plane perpendicular to the ray that includes the light position
    #the normal is the same as the ray_vector
    d = light_pos[0]*ray_vector[0]+light_pos[1]*ray_vector[1]+light_pos[2]*ray_vector[2]
    plane_point = np.array((0.0,0.0,0.0))
    #TODO: check if we can just take zeros instead of randoms
    if (ray_vector[2]!=0):
        plane_point_x = 10*random.random()
        plane_point_y = 20*random.random()
        plane_point = np.array((plane_point_x,plane_point_y,\
                                (d-plane_point_x*ray_vector[0]-plane_point_y*ray_vector[1])/ray_vector[2]))
        #plane_point = np.array((0.0,0.0,-d/ray_vector[2]))
    elif (ray_vector[1]!=0):
        plane_point_x = 10*random.random()
        plane_point_z = 20*random.random()
        plane_point = np.array((plane_point_x,(d-plane_point_x*ray_vector[0]-plane_point_z*ray_vector[2])/ray_vector[1]),\
                                plane_point_z)
        #plane_point = np.array((0.0,-d/ray_vector[1],0.0))
    elif (ray_vector[0]!=0):
        plane_point_y = 10*random.random()
        plane_point_z = 20*random.random()
        plane_point = np.array(((d-plane_point_y*ray_vector[1]-plane_point_z*ray_vector[2])/ray_vector[0]),\
                                plane_point_y,plane_point_z)
        #plane_point = np.array((-d/ray_vector[0],0.0,0.0))
    #now we have a point, let's find parametric representation of the plane
    first_direc = (plane_point-light_pos)/np.linalg.norm(light_pos-plane_point)
    second_direc = np.cross(first_direc,ray_vector)/np.linalg.norm(np.cross(first_direc,ray_vector))
    first_rect_point = light_pos-0.5*light_radius*(first_direc+second_direc)
    #first_rect_point = np.array((light_pos[0]-0.5*light_radius*(first_direc[0]+second_direc[0]),\
    #                    light_pos[1]-0.5*light_radius*(first_direc[1]+second_direc[1]),\
    #                    light_pos[2]-0.5*light_radius*(first_direc[2]+second_direc[2])))
    shadow_rays_num = int(scene.root_number_shadow_rays)
    hit_count = 0
    for i in range(shadow_rays_num):
        sample_point = np.copy(first_rect_point)
        sample_point = sample_point+(i*light_radius/shadow_rays_num)*first_direc
        for j in range(shadow_rays_num):
            sample_point_in_loop = sample_point+(random.random()*light_radius/shadow_rays_num)*first_direc
            sample_point_in_loop = sample_point+(random.random()*light_radius/shadow_rays_num)*second_direc
            shadow_ray = (p-sample_point_in_loop)/np.linalg.norm(p-sample_point_in_loop)
            hit_count+=calculate_shadow_transparency(sample_point_in_loop,shadow_ray,surface)

    hit_rate = hit_count/(shadow_rays_num**2)
    return (1-light_shadow_intensity)+light_shadow_intensity*hit_rate

def calculate_shadow_transparency(sample_point_in_loop,shadow_ray,surface):
    global surfaces_lst, materials_lst
    shadow_transp = 1
    t_list = np.array([0 for i in range(len(surfaces_lst))])
    t_thresh = 0
    obj_ind = 0
    for i in range(len(surfaces_lst)):
        obj = surfaces_lst[i]
        if isinstance(obj,Cube):
            #extract parameters
            cube_mid = np.array(obj.position)
            edge_len = obj.scale
            min_x, min_y, min_z = min(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
            min(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), min(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
            max_x, max_y, max_z = max(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
            max(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), max(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
            t_min_x, t_max_x = min_x-sample_point_in_loop[0]/shadow_ray[0],\
                  max_x-sample_point_in_loop[0]/shadow_ray[0]
            t_min_y, t_max_y = min_y-sample_point_in_loop[1]/shadow_ray[1], \
                max_y-sample_point_in_loop[1]/shadow_ray[1]
            t_min_z, t_max_z = min_z-sample_point_in_loop[2]/shadow_ray[2], \
                max_z-sample_point_in_loop[2]/shadow_ray[2]
            t_enter = max(t_min_x,t_min_y,t_min_z)
            t_exit = min(t_max_x,t_max_y,t_max_z)
            if (t_enter>t_exit) or (t_max_x<0) or (t_max_y<0) or (t_max_z<0):
                #no intersection
                continue
            #there is an intersection, t=t_enter
            t_list[i] = t_enter
            if (isinstance(surface, Cube) and \
                obj.position==surface.position and obj.scale==surface.scale and \
                obj.material_index==surface.material_index):
                obj_ind = i

        elif isinstance(obj,Sphere):
            O = np.array(obj.position)
            r = obj.radius
            #need to solve a quadratic equation
            a = 1
            b = 2*np.dot(shadow_ray,sample_point_in_loop-O)
            c = ((sample_point_in_loop[0]-O[0])**2+(sample_point_in_loop[1]-O[1])**2+(sample_point_in_loop[2]-O[2])**2)-r**2
            delta = b**2-4*a*c
            if (delta<0): #a complex number - ray does not intersect with the sphere
                continue
            t1 = (-b+np.sqrt(delta))/(2*a)
            t2 = (-b-np.sqrt(delta))/(2*a)
            sphere_min_t = min(t1,t2)
            t_list[i] = sphere_min_t
            if (isinstance(surface, Sphere) and \
                obj.position==surface.position and obj.radius==surface.radius and \
                obj.material_index==surface.material_index):
                obj_ind = i

        elif isinstance(obj, InfinitePlane):
            #extract parameters
            N = np.array(obj.normal)
            d_plane = obj.offset
            #need to find a point on the plane
            if N[2]!=0:
                plane_point = np.array((0.0,0.0,-d_plane/N[2]))
            elif N[1]!=0:
                plane_point = np.array((0.0,-d_plane/N[1],0.0))
            elif N[0]!=0:
                plane_point = np.array((-d_plane/N[0],0.0,0.0))
            dot_prod = np.dot(shadow_ray,N)
            if (np.abs(dot_prod)<EPSILON):
                #ray is parallel or nearly parallel to the plane - no intersection
                continue
            #there could still be an intersection
            t_plane = np.dot(N,plane_point-sample_point_in_loop)/dot_prod
            if t_plane<0: #no intersection
                continue
            t_list[i] = t_plane
            if (isinstance(surface, InfinitePlane) and \
                obj.normal==surface.normal and obj.offset==surface.offset and \
                obj.material_index==surface.material_index):
                obj_ind = i
    t_thresh = t_list[obj_ind]
    #erase every objects that are beyond our desired object to check
    t_list_with_thresh = t_list*(t_list<t_thresh).astype(int)
    for i in range(len(t_list_with_thresh)):
        if t_list_with_thresh[i]!=0:
            mat_ind = surfaces_lst[i].material_index
            shadow_transp *= materials_lst[mat_ind-1].transparency
    return shadow_transp


def init_lists():
    global objects_lst, materials_lst, surfaces_lst, lights_lst
    #preassumption: the order of the objects_lst is materials, surfaces, lights
    mat_lst = []
    surf_lst = []
    light_lst = []
    for i in range(len(objects_lst)):
        obj = objects_lst[i]
        if isinstance(obj,Material):
            mat_lst.append(obj)
        if isinstance(obj,Light):
            light_lst.append(obj)
        else:
            surf_lst.append(obj)
    
    mat_lst = np.array(mat_lst, dtype=object)
    surf_lst = np.array(surf_lst, dtype=object)
    light_lst = np.array(light_lst, dtype=object)
    materials_lst = mat_lst
    surfaces_lst = surf_lst
    lights_lst = light_lst
    return


if __name__ == '__main__':
    main()
