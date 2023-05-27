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

EPSILON = 0.001

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

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    print(objects)
    image_array = np.zeros((args.height, args.width, 3))
    rays_array = np.zeros((args.height, args.width),dtype=object)
    #need to use np.vectorize on the lambda expression in order to insert it into the rays_array

    #extract parameters and name them in the same way as in the slides
    p_0 = camera.position
    w = camera.screen_width
    d = camera.screen_distance
    v_to = camera.look_at
    v_up = camera.up_vector
    rx, ry = args.width, args.height
    #image center
    p_c = tuple(p_0+d*np.array(v_to))
    # do the cross product and normalize
    v_right = tuple(np.cross(np.array(v_to),np.array(v_up))/np.linalg.norm(np.cross(np.array(v_to),np.array(v_up))))
    v_up_tilda = tuple(np.cross(np.array(v_right),np.array(v_to))/np.linalg.norm(np.cross(np.array(v_right),np.array(v_to))))
    R = w/rx
    for i in range(args.height):
        for j in range(args.width):
            #for pixel (i,j)
            #Shoot a ray through each pixel in the image
            #Discover the location of the pixel on the camera’s screen (using camera parameters).
            p = p_c+tuple((j-np.floor(rx/2))*R*np.array(v_right)-\
                (i-np.floor(ry/2))*R*np.array(v_up_tilda))
            #Construct a ray from the camera through that pixel.
            ray = lambda t: p_0 + tuple(t*(np.array(p)-np.array(p_0)))
            V = tuple(np.array(p)-np.array(p_0))
            rays_array[i][j] = np.vectorize(ray)
            closest_surface = 0
            min_t = float('inf')
            for obj in objects:
                #Check the intersection of the ray with all surfaces in the scene (you can add \
                # optimizations such as BSP trees if you wish but they are not mandatory).
                #Find the nearest intersection of the ray. This is the surface that will be
                # seen in the image.
                if isinstance(obj,Cube):
                    #extract parameters
                    cube_mid = obj.position
                    edge_len = obj.scale
                    min_x, min_y, min_z = min(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
                    min(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), min(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
                    max_x, max_y, max_z = max(cube_mid[0]+edge_len/2,cube_mid[0]-edge_len/2),\
                    max(cube_mid[1]+edge_len/2,cube_mid[1]-edge_len/2), max(cube_mid[2]+edge_len/2,cube_mid[2]-edge_len/2)
                    t_min_x, t_max_x = min_x-p_0[0]/V[0], max_x-p_0[0]/V[0]
                    t_min_y, t_max_y = min_y-p_0[1]/V[1], max_y-p_0[1]/V[1]
                    t_min_z, t_max_z = min_z-p_0[2]/V[2], max_z-p_0[2]/V[2]
                    t_enter = max(t_min_x,t_min_y,t_min_z)
                    t_exit = min(t_max_x,t_max_y,t_max_z)
                    if (t_enter>t_exit) or (t_max_x<0) or (t_max_y<0) or (t_max_z<0):
                        #no intersection
                        continue
                    #there is an intersection, t=t_enter
                    if (t_enter<=min_t):
                        min_t = t_enter
                        closest_surface = obj
                elif isinstance(obj,Sphere):
                    O = obj.position
                    r = obj.radius
                    #need to solve a quadratic equation
                    a = 1
                    b = 2*np.dot(np.array(V),(np.array(p_0)-np.array(O)))
                    c = np.sqrt((p_0[0]-O[0])**2+(p_0[1]-O[1])**2+(p_0[2]-O[2])**2)-r**2
                    delta = b**2-4*a*c
                    if (delta<0): #a complex number - ray does not intersect with the sphere
                        continue
                    t1 = (-b+np.sqrt(delta))/(2*a)
                    t2 = (-b-np.sqrt(delta))/(2*a)
                    sphere_min_t = min(t1,t2)
                    if (sphere_min_t<=min_t):
                        #need to update closest surface
                        min_t =sphere_min_t
                        closest_surface = obj
                elif isinstance(obj, InfinitePlane):
                    #extract parameters
                    N = np.array(obj.normal)
                    d_plane = obj.offset
                    #need to find a point on the plane
                    if N[2]!=0:
                        plane_point = (0,0,-d_plane/N[2])
                    elif N[1]!=0:
                        plane_point = (0,-d_plane/N[1],0)
                    elif N[0]!=0:
                        plane_point = (-d_plane/N[0],0,0)
                    dot_prod = np.dot(np.array(V),np.array(N))
                    if (np.abs(dot_prod)<EPSILON):
                        #ray is parallel or nearly parallel to the plane - no intersection
                        continue
                    #there could still be an intersection
                    t_plane = np.dot(N,plane_point-p_0)/dot_prod
                    if t_plane<0: #no intersection
                        continue
                    if t_plane<min_t:
                        min_t = t_plane
                        closest_surface = obj
            #here we've got the closest surface saved together with the t_value

    #Compute the color of the surface:
        #Go over each light in the scene.
        #Add the value it induces on the surface.
    #Find out whether the light hits the surface or not:
        #Shoot rays from the light towards the surface
        #Find whether the ray intersects any other surfaces before the required
        # surface - if so, the surface is occluded from the light and the light does
        # not affect it (or partially affects it because of the shadow intensity parameter).
    #Produce soft shadows, as explained below:
        #Shoot several rays from the proximity of the light to the surface.
        #Find out how many of them hit the required surface


    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
