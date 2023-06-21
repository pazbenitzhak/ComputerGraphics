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

EPSILON = 0.00001
objects_lst = []
materials_lst = []
surfaces_lst = [] 
lights_lst = []
camera = 0
scene_settings = 0
shadow = 0
out_img = ""

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
    global out_img
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(out_img)


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    parser.add_argument('--shadow', type=str, default='y', help='Run with Shadow Transparency Check')
    #by default we run the function with the shadow transparency check. To run it without the check
    #one must use --shadow n
    args = parser.parse_args()

    global objects_lst, materials_lst, surfaces_lst, lights_lst, camera, scene_settings, shadow, out_img
    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    objects_lst = objects
    image_array = np.zeros((args.height, args.width, 3),dtype=np.uint8)
    shadow = args.shadow
    out_img = args.output_image
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
    v_to_unnorm = np.array(camera.look_at)-np.array(camera.position)
    v_to = v_to_unnorm/find_norm(v_to_unnorm)
    #print("v_to: " +str(v_to))
    v_up = np.array(camera.up_vector)/find_norm(np.array(camera.up_vector)) #TODO: find a perpendicular vector
    #print("v_up: " +str(v_up))
    #TODO: check if we should keep it that way or do it according to v_to
    rx, ry = args.width, args.height
    #image center
    p_c = p_0_cam+d*v_to
    # do the cross product and normalize
    v_right_unnorm = np.cross(v_to,v_up)
    v_right = v_right_unnorm/find_norm(v_right_unnorm)
    v_up_tilda_unnorm = np.cross(v_to,v_right)
    v_up_tilda = v_up_tilda_unnorm/find_norm(v_up_tilda_unnorm)
    R = w/rx
    #TODO: ADDITION
    screen_height = (float(args.height)/float(args.width))*w
    sinx = -1*v_to[1]
    cosx = np.sqrt(1-sinx*sinx)
    siny = -1*v_to[0]/cosx
    cosy = v_to[2]/cosx
    M = np.array([[cosy,0,siny],[-1*sinx*siny,cosx,sinx*cosy],[-1*cosx*siny,-1*sinx,cosx*cosy]])
    v_x_unnorm = np.array([1,0,0]) @ M
    v_x = v_x_unnorm/find_norm(v_x_unnorm)
    v_y_unnorm = np.array([0,-1,0]) @ M
    v_y = v_y_unnorm/find_norm(v_y_unnorm)
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
            V_unnorm = (p-p_0_cam)
            V = V_unnorm/find_norm(V_unnorm)
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
            color = find_color(intersect_point,V,closest_surface,1)
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
            #TODO: check compatibility
            #extract parameters

            cube_mid = np.array(surf.position)
            edge_len = surf.scale
            planes_N = [np.array((1,0,0)),np.array((1,0,0)),np.array((0,1,0)),np.array((0,1,0)),\
                      np.array((0,0,1)),np.array((0,0,1))]
            planes_d = [cube_mid[0]+0.5*edge_len,cube_mid[0]-0.5*edge_len,cube_mid[1]+0.5*edge_len,cube_mid[1]-0.5*edge_len,\
                        cube_mid[2]+0.5*edge_len,cube_mid[2]-0.5*edge_len]
            planes_t = []
            for i in range(6): #number of edges in a cube
                N = planes_N[i]
                d_plane = planes_d[i]
                p0_dot_N = np.dot(p_0,N)
                V_dot_N = np.dot(V,N)
                t_plane = (d_plane-p0_dot_N)/V_dot_N
                #if t_plane<=0:
                #    t_plane = float("inf")
                planes_t.append(t_plane)
            t_x_min = min(planes_t[0],planes_t[1])
            t_x_max = max(planes_t[0],planes_t[1])
            t_y_min = min(planes_t[2],planes_t[3])
            t_y_max = max(planes_t[2],planes_t[3])
            if t_x_min>t_y_max or t_y_min>t_x_max:
                #no intersection
                continue

            t_min = max(t_x_min,t_y_min)
            t_max = min(t_x_max,t_y_max)
            t_z_min = min(planes_t[4],planes_t[5])
            t_z_max = max(planes_t[4],planes_t[5])
            if t_min>t_z_max or t_z_min>t_max:
                #no_intersction
                continue
            t_enter = max(t_min,t_z_min)
            if (t_enter<min_t):
                min_t = t_enter
                closest_surface = surf
            # min_x, min_y, min_z = min(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
            # min(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), min(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
            # max_x, max_y, max_z = max(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
            # max(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), max(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
            # t_min_x, t_max_x = min_x-p_0[0]/V[0], max_x-p_0[0]/V[0]
            # t_min_y, t_max_y = min_y-p_0[1]/V[1], max_y-p_0[1]/V[1]
            # t_min_z, t_max_z = min_z-p_0[2]/V[2], max_z-p_0[2]/V[2]
            # t_enter = max(t_min_x,t_min_y,t_min_z)
            # t_exit = min(t_max_x,t_max_y,t_max_z)
            # if (t_enter>t_exit) or (t_max_x<0) or (t_max_y<0) or (t_max_z<0) or (t_enter<=0):
            #     #no intersection
                #continue
            #there is an intersection, t=t_enter
            #if (t_enter<min_t):
             #   min_t = t_enter
              #  closest_surface = surf
        elif isinstance(surf,Sphere):
            #geometric approach
            O = np.array(surf.position)
            r = surf.radius
            L = O-p_0
            t_ca = np.dot(L,V)
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
            if (sphere_min_t<min_t):
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
            if t_plane<min_t:
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

def find_color(intersect_point,V,surface,depth):
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
    global camera, scene_settings
    scene = scene_settings
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
    N = find_surface_normal(surface,intersect_point)
    if N is None:
        print("intersection point: " +str(intersect_point))
        print("depth: " +str(depth))
    diff_color = np.array((0.0,0.0,0.0))
    spec_color = np.array((0.0,0.0,0.0))
    color = np.array((0.0,0.0,0.0))
    for light in lights_lst:
        #extract light data
        light_pos = np.array(light.position)
        light_color = np.array(light.color)
        light_specular_intensity = light.specular_intensity
        #it's a light
        light_intensity = find_light_intensity(light,intersect_point,surface)
        #TODO: STOP
        #print("light_intensity: " +str(light_intensity))
        I_L = light_intensity*light_color
        #print("I_L: " +str(I_L))
        #create a ray from the light to the object
        L_unnorm = light_pos-intersect_point
        L = L_unnorm/find_norm(L_unnorm)
        N_dot_L = max(0,np.dot(L,N))
        #print(np.dot(N,L)/(np.linalg.norm(N)*np.linalg.norm(L)))
        #print("N_dot_L: " +str(N_dot_L))
        R_unnorm = 2*N_dot_L*N-L
        R = R_unnorm/find_norm(R_unnorm)
        R_dot_V = max(0,np.dot(R,-1*V))
        #TODO: check if R_V or V_R

        #calculate diffuse_color
        diff_color += I_L*diffuse_color*N_dot_L
        #print("I_L: "+str(I_L))
        #print("N_dot_L: "+str(N_dot_L))
        #print("diff_color: " +str(diff_color))

        #calculate specular color
        spec_color += I_L*specular_color*light_specular_intensity*\
                            (R_dot_V**phong_spec_efficiency)
        #print("spec_color: " +str(spec_color))
    color += (1-transparency)*(diff_color+spec_color)
    #print("color: " +str(color))

    #print("max recursion:" +str(scene.max_recursions))
    #check whether we've already reached our depth maximum. If so - return the color
    #If not - need to further check the reflections
    if (depth<scene.max_recursions):
        #need to handle reflection - every time a ray hits the surface, it reflects back the light
        #let's consruct the R vector
        V_dot_N = np.dot(V,N)
        R_reflect_unnorm = V-2*V_dot_N*N
        R_reflect = R_reflect_unnorm/find_norm(R_reflect_unnorm)
        closest_next_surface, min_t_to_next_surface = find_intersection(intersect_point,R_reflect)
        next_p = intersect_point+min_t_to_next_surface*R_reflect
        reflect_color = find_color(next_p,R_reflect,closest_next_surface,depth+1)
        color += reflect_color*reflection_color
        #print("color: " +str(color) + ", depth: " +str(depth))

        #print("transparency: " +str(transparency))
        if transparency!=0: #the material is somewhat transparent and not opaque
            #need to calculate reflections
            exit_point = find_ray_exit_point(intersect_point,V,surface)
            closest_next_surface, min_t_to_next_surface = find_intersection(exit_point,V)
            next_p = exit_point+min_t_to_next_surface*V
            transp_color = find_color(next_p,V,closest_next_surface,depth)
            
            color += transparency*transp_color
            #print("color: " +str(color) + ", depth: " +str(depth))



    #print("final color: " +str(color) + ", depth = " +str(depth))
    return np.array((min(color[0],1.0),min(color[1],1.0),min(color[2],1.0)))

def find_surface_normal(surface,intersect_point):
    if isinstance(surface,InfinitePlane):
        return np.array(surface.normal)/find_norm(np.array(surface.normal))
    
    if isinstance(surface,Sphere):
        sphere_center = np.array(surface.position)
        norm_unnorm = intersect_point-sphere_center
        return norm_unnorm/find_norm(norm_unnorm)
    
    if isinstance(surface,Cube):
        """PREV_IMP"""
        cube_center = np.array(surface.position)
        edge_len = surface.scale
        #need to find which plane of the six edges includes the intersection point
        #divide to cases
        #plane's equation is x = d or x = -d
        d1 = cube_center[0]-0.5*edge_len
        if (abs(intersect_point[0]-d1)<EPSILON):
            return np.array((1.0,0.0,0.0))
        d2 = cube_center[0]+0.5*edge_len
        if (abs(intersect_point[0]-d2)<EPSILON):
            return np.array((1.0,0.0,0.0))
        #plane's equation is y = d or y = -d
        d3 = cube_center[1]-0.5*edge_len
        if (abs(intersect_point[1]-d3)<EPSILON):
            return np.array((0.0,1.0,0.0))
        d4 = cube_center[1]+0.5*edge_len
        if (abs(intersect_point[1]-d4)<EPSILON):
            return np.array((0.0,1.0,0.0))
        #plane's equation is z = d or z = -d
        d5 = cube_center[2]-0.5*edge_len
        if (abs(intersect_point[2]*1-d5)<EPSILON):
            return np.array((0.0,0.0,1.0))
        d6 = cube_center[2]+0.5*edge_len
        if (abs(intersect_point[2]*1-d6)<EPSILON):
            return np.array((0.0,0.0,1.0))
        #print(abs(intersect_point[0]-d1))
        #print(abs(intersect_point[0]+d1))
        #print(abs(intersect_point[1]-d2))
        #print(abs(intersect_point[1]+d2))
        #print(abs(intersect_point[2]-d))
        #print(abs(intersect_point[2]+d))
        print("WHYYYYYYYY")
    
def find_ray_exit_point(p,V,surface):
    global camera
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
        # cube_mid = np.array(surface.position)
        # edge_len = surface.scale
        # min_x, min_y, min_z = min(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
        # min(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), min(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
        # max_x, max_y, max_z = max(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
        # max(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), max(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
        # t_min_x, t_max_x = min_x-p_0[0]/V[0], max_x-p_0[0]/V[0]
        # t_min_y, t_max_y = min_y-p_0[1]/V[1], max_y-p_0[1]/V[1]
        # t_min_z, t_max_z = min_z-p_0[2]/V[2], max_z-p_0[2]/V[2]
        # t_exit = min(t_max_x,t_max_y,t_max_z)


        cube_mid = np.array(surface.position)
        edge_len = surface.scale
        planes_N = [np.array((1,0,0)),np.array((1,0,0)),np.array((0,1,0)),np.array((0,1,0)),\
                    np.array((0,0,1)),np.array((0,0,1))]
        planes_d = [cube_mid[0]+0.5*edge_len,cube_mid[0]-0.5*edge_len,cube_mid[1]+0.5*edge_len,cube_mid[1]-0.5*edge_len,\
                    cube_mid[2]+0.5*edge_len,cube_mid[2]-0.5*edge_len]
        planes_t = []
        for i in range(6): #number of edges in a cube
            N = planes_N[i]
            d_plane = planes_d[i]
            p0_dot_N = np.dot(p,N)
            V_dot_N = np.dot(V,N)
            t_plane = (d_plane-p0_dot_N)/V_dot_N
            #if t_plane<=0:
            #    t_plane = float("inf")
            planes_t.append(t_plane)
        t_x_min = min(planes_t[0],planes_t[1])
        t_x_max = max(planes_t[0],planes_t[1])
        t_y_min = min(planes_t[2],planes_t[3])
        t_y_max = max(planes_t[2],planes_t[3])
        if t_x_min>t_y_max or t_y_min>t_x_max:
            #no intersection - doesn't make sense
            return p

        t_min = max(t_x_min,t_y_min)
        t_max = min(t_x_max,t_y_max)
        t_z_min = min(planes_t[4],planes_t[5])
        t_z_max = max(planes_t[4],planes_t[5])
        if t_min>t_z_max or t_z_min>t_max:
            #no_intersction
            return p
        t_exit = max(t_min,t_z_min)
        return p+t_exit*V


def find_light_intensity(light,p,surface):
    #Produce soft shadows, as explained below:
        #Shoot several rays from the proximity of the light to the surface.
        #Find out how many of them hit the required surface
    #extract light parameters
    global scene_settings, shadow
    scene = scene_settings
    light_pos = np.array(light.position)
    light_shadow_intensity = light.shadow_intensity
    light_radius = light.radius
    #shoot a ray from the light position to the surface point
    ray_vector_unnorm = p-light_pos
    ray_vector = ray_vector_unnorm/find_norm(ray_vector_unnorm)
    #find a plane perpendicular to the ray that includes the light position
    #the normal is the same as the ray_vector
    d = light_pos[0]*ray_vector[0]+light_pos[1]*ray_vector[1]+light_pos[2]*ray_vector[2]
    plane_point = np.array((0.0,0.0,0.0))
    if (ray_vector[2]!=0):
        plane_point_x = 10*random.random()
        plane_point_y = 20*random.random()
        plane_point = np.array((plane_point_x,plane_point_y,\
                                (d-plane_point_x*ray_vector[0]-plane_point_y*ray_vector[1])/ray_vector[2]))
        #plane_point = np.array((0.0,0.0,-d/ray_vector[2]))
    elif (ray_vector[1]!=0):
        plane_point_x = 10*random.random()
        plane_point_z = 20*random.random()
        plane_point = np.array((plane_point_x,(d-plane_point_x*ray_vector[0])/ray_vector[1],\
                                plane_point_z))
        #plane_point = np.array((0.0,-d/ray_vector[1],0.0))
    elif (ray_vector[0]!=0):
        plane_point_y = 10*random.random()
        plane_point_z = 20*random.random()
        plane_point = np.array(d/ray_vector[0],\
                                plane_point_y,plane_point_z)
        #plane_point = np.array((-d/ray_vector[0],0.0,0.0))
    #now we have a point, let's find parametric representation of the plane
    first_direc_unnorm = plane_point-light_pos
    first_direc = first_direc_unnorm/find_norm(first_direc_unnorm)
    second_direc_unnorm = np.cross(first_direc,ray_vector)
    second_direc = second_direc_unnorm/find_norm(second_direc_unnorm)
    first_rect_point = light_pos-0.5*light_radius*(first_direc+second_direc)
    #first_rect_point = np.array((light_pos[0]-0.5*light_radius*(first_direc[0]+second_direc[0]),\
    #                    light_pos[1]-0.5*light_radius*(first_direc[1]+second_direc[1]),\
    #                    light_pos[2]-0.5*light_radius*(first_direc[2]+second_direc[2])))
    shadow_rays_num = int(scene.root_number_shadow_rays)
    hit_count = 0
    for i in range(shadow_rays_num):
        sample_point = np.copy(first_rect_point)
        sample_point = sample_point+(i*(light_radius/shadow_rays_num))*second_direc
        for j in range(shadow_rays_num):
            sample_point_in_loop = np.copy(sample_point)
            sample_point_in_loop += (random.random()*(light_radius/shadow_rays_num))*first_direc
            sample_point_in_loop += (random.random()*(light_radius/shadow_rays_num))*second_direc
            shadow_ray_unnorm = p-sample_point_in_loop
            shadow_ray = shadow_ray_unnorm/find_norm(shadow_ray_unnorm)
            if shadow=='n':
                hit_count+=1
            else:
                hit_count+=calculate_shadow_transparency(sample_point_in_loop,shadow_ray,surface)

    hit_rate = hit_count/(float(shadow_rays_num)*shadow_rays_num)
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
            # cube_mid = np.array(obj.position)
            # edge_len = obj.scale
            # min_x, min_y, min_z = min(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
            # min(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), min(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
            # max_x, max_y, max_z = max(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
            # max(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), max(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
            # t_min_x, t_max_x = min_x-sample_point_in_loop[0]/shadow_ray[0],\
            #       max_x-sample_point_in_loop[0]/shadow_ray[0]
            # t_min_y, t_max_y = min_y-sample_point_in_loop[1]/shadow_ray[1], \
            #     max_y-sample_point_in_loop[1]/shadow_ray[1]
            # t_min_z, t_max_z = min_z-sample_point_in_loop[2]/shadow_ray[2], \
            #     max_z-sample_point_in_loop[2]/shadow_ray[2]
            # t_enter = max(t_min_x,t_min_y,t_min_z)
            # t_exit = min(t_max_x,t_max_y,t_max_z)
            # if (t_enter>t_exit) or (t_max_x<0) or (t_max_y<0) or (t_max_z<0) or (t_enter<=0):
            #     #no intersection
            #     continue
            cube_mid = np.array(obj.position)
            edge_len = obj.scale
            planes_N = [np.array((1,0,0)),np.array((1,0,0)),np.array((0,1,0)),np.array((0,1,0)),\
                      np.array((0,0,1)),np.array((0,0,1))]
            planes_d = [cube_mid[0]+0.5*edge_len,cube_mid[0]-0.5*edge_len,cube_mid[1]+0.5*edge_len,cube_mid[1]-0.5*edge_len,\
                        cube_mid[2]+0.5*edge_len,cube_mid[2]-0.5*edge_len]
            planes_t = []
            for i in range(6): #number of edges in a cube
                N = planes_N[i]
                d_plane = planes_d[i]
                p0_dot_N = np.dot(sample_point_in_loop,N)
                V_dot_N = np.dot(shadow_ray,N)
                t_plane = (d_plane-p0_dot_N)/V_dot_N
                #if t_plane<=0:
                #    t_plane = float("inf")
                planes_t.append(t_plane)
            t_x_min = min(planes_t[0],planes_t[1])
            t_x_max = max(planes_t[0],planes_t[1])
            t_y_min = min(planes_t[2],planes_t[3])
            t_y_max = max(planes_t[2],planes_t[3])
            if t_x_min>t_y_max or t_y_min>t_x_max:
                #no intersection
                continue

            t_min = max(t_x_min,t_y_min)
            t_max = min(t_x_max,t_y_max)
            t_z_min = min(planes_t[4],planes_t[5])
            t_z_max = max(planes_t[4],planes_t[5])
            if t_min>t_z_max or t_z_min>t_max:
                #no_intersction
                continue
            t_enter = max(t_min,t_z_min)
            #there is an intersection, t=t_enter
            t_list[i] = t_enter
            if (isinstance(surface, Cube) and \
                obj.position==surface.position and obj.scale==surface.scale and \
                obj.material_index==surface.material_index):
                obj_ind = i

        elif isinstance(obj,Sphere):
            #geometric approach
            O = np.array(obj.position)
            r = obj.radius
            L = O-sample_point_in_loop
            t_ca = np.dot(L,shadow_ray)
            if t_ca<0:
                continue
            d_sq = np.sum(np.square(L))-np.square(t_ca)
            if d_sq > r**2:
                continue
            t_hc = np.sqrt(r**2-d_sq)
            sphere_min_t = t_ca-t_hc
            if (sphere_min_t<=0):
                continue
            
            """prev_imp
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
            """
            t_list[i] = sphere_min_t
            if (isinstance(surface, Sphere) and \
                obj.position==surface.position and obj.radius==surface.radius and \
                obj.material_index==surface.material_index):
                obj_ind = i

        elif isinstance(obj, InfinitePlane):
            N = np.array(obj.normal)
            d_plane = obj.offset
            p0_dot_N = np.dot(sample_point_in_loop,N)
            V_dot_N = np.dot(shadow_ray,N)
            t_plane = (d_plane-p0_dot_N)/V_dot_N
            if t_plane<=0:
                continue

            """prev_imp
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
            if t_plane<=0: #no intersection
                continue
                """
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

def find_norm(vec):
    return np.sqrt(np.square(vec[0])+np.square(vec[1])+np.square(vec[2]))

if __name__ == '__main__':
    main()
