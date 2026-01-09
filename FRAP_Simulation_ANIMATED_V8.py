# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 19:24:26 2024
ANIMATED VERSION
Feature notes: 
    Runtime optimizations were made with NumPy vectorization:
    All shapes (sphere, rod, helical cell) are defined by midpoints
    All particles immediately spawn within the shape
    Diffusion happens in one step
    Particle distances to the midpoints are now calculated in one go
    Particles are kept inside by implementing membrane collisions
    Photobleaching with Boolean masks for probabilities and positions
    Particle diffusion types (fast, slow, immobile) handled with 2D array
    Visualization is enhanced
@author: Shariful Sakib
"""
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import curve_fit
import tkinter
from tkinter import filedialog
from decimal import Decimal #needed to avoid floating point errors
#program ignores this syntax error. %matplotlib auto allows for plotting in a separate window for interactive graphs 
# %matplotlib auto
plt.ion()
#NOTE: can also comment out %matplotlib auto and then go to Tools>Preferences>IPython console>Graphics>Backend>Automatic
colorMap = LinearSegmentedColormap.from_list('Black', ['red','aqua','black']) 
plt.close("all")
# time0 = time.time()

# fill in any parameters of interest.
# Don't worry about leaving other parameters in.
# Program is robust and will use what it needs to make the necessary shape.
# For spheres, set shape = 0 and you can vary internal radius.
# For rods, set shape = 1 and you can vary internal radius and length.
# For spirilla, set shape >= 2 and you can vary everything.
# Finally, change any other setting for the FRAP experiment.

shape = 2# 0 = sphere, 1 = rod, 2 = helical cell
shapeDict = {0: "sphere", 1:"rod", 2:"helical cell"}
shapeName = shapeDict[shape]
internal_radius = 0.27 #um
length = 0.6 #um
pitch = 2.5#um
amplitude = 2 #um
contourLength = (length - internal_radius*2) * math.sqrt(1 + (2*math.pi*amplitude/pitch)**2) + internal_radius*2
# temp = 5
scale_factor = max(3.5 * length / 5, 1)   #for graphing purposes
length_correction = internal_radius     #ensures the shape produced is of the correct length
max_segment_error = 0.0042              #um ~4.2 nm maximum error between true and calculated distances (approximate length of an FP molecule)
segment_resolution = 2*math.sqrt((max_segment_error + internal_radius)**2 - internal_radius**2)
pixel_resolution = 0.085                #um 85 nm resolution used for creating plot profiles

numParticles = 1000
fast_diffusion_constant = 0.1 # um^2/s
immobile_diffusion_constant = 0 #um^2/s
slow_diffusion_constant = 1 # um^2/s
fast_particle_proportion = 1
immobile_particle_proportion = 0
APPLY_COMPARTMENT_COLLISIONS = True # Can toggle collisions (required for FRAP)
DRAW_DISTANCES_TO_MIDPOINTS = False # Can toggle drawing distances (slower and not reccommended)
SHOW_MIDPOINTS = True              # Can toggle plotting midpoints on the 3D view (optional)

numFastParticles = int(numParticles * fast_particle_proportion)
numImmobileParticles = int(numParticles * immobile_particle_proportion)
numSlowParticles = numParticles - numFastParticles - numImmobileParticles
diffusion_constant = np.repeat([np.full(3, fast_diffusion_constant), np.full(3, slow_diffusion_constant), np.full(3, immobile_diffusion_constant)],
                               [numFastParticles, numSlowParticles, numImmobileParticles], axis=0) #d_c um^2/s #mask of diffusion constants
np.random.shuffle(diffusion_constant)

simulated_seconds = 1#math.ceil(contourLength**2*((3/4) if length <= 5 else (1/3))/(2*fast_diffusion_constant) + 1)  #sec Simulated total duration of the experiment
frame_interval = 0.030 #sec
simulated_interval = 0.001 #sec
simulated_range = int((simulated_seconds / frame_interval) * (frame_interval / simulated_interval)) #iterations
# animated_frames = simulated_range + 100

bleach_duration = frame_interval #s how long should the bleach be
bleach_start_time = frame_interval * 2 #s what timepoint should the bleach begin
bleach_end_time = bleach_start_time + bleach_duration
gamma = 5 #related to laser power
fluorophore_lifetime = frame_interval / gamma
bleach_probability = simulated_interval / fluorophore_lifetime #probability of bleaching
bleach_region_fraction = 0.5 #proportion of compartment to bleach (right hand edge of bleach region)
bleach_box_scan_size = None #um or None, Can be set to None if you want to bleach from endcap to a midpoint within the cell (bleach_region_fraction) 
                            # or can be set to a numerical value to determine the size of a bleach region with the location of the right-hand boundary of the region determined by bleach_region_fraction
bleach_region = length * bleach_region_fraction #x values less than and equal to this value will be bleached for rods and helical cell
pixel_intervals = np.reshape(np.arange(0 + internal_radius, length - internal_radius - pixel_resolution, pixel_resolution), (-1, 1)) #used for plot profile segmentation, also avoids end caps
# print(pixel_intervals.squeeze())
if bleach_box_scan_size:
    cosineFittingMask = pixel_intervals >= bleach_region - 0.5 * bleach_box_scan_size if bleach_region_fraction <= 0.5 else pixel_intervals <= bleach_region + 0.5 * bleach_box_scan_size
else:
    cosineFittingMask = pixel_intervals >= 0
    # print(cosineFittingMask)
graphTransformation = 1 # 0 = stationary + interactive, 1 = triangle function wobble, 2 = swimming, 3 = swimming + turning
SAVE_ANIMATION = False #Users can change this to True if they want to save a GIF of a simulation

if SAVE_ANIMATION:
    tkinter.Tk().withdraw();
    saveFolder = str(filedialog.askdirectory()) + "\\"; #Asks user for save folder
    print(saveFolder)
    if saveFolder == "\\": #doesn't save gif if user cancels   
        SAVE_ANIMATION = False
if shape <= 1:
    amplitude = 0
    pitch = np.inf
    contourLength = length
if shape == 0:
    length = internal_radius * 2 # corrects length for spheres
    bleach_region = length * bleach_region_fraction - internal_radius # corrects bleach_region for spheres since they are centered at (0,0,0)
    segment_resolution = length # corrects segment_resolution for spheres since they are defined by one point (0,0,0)
    scale_factor = 1       
    length_correction = 0

time_list = ([])
FRAP_values = ([],[],[]) # 1st element holds simple FRAP value list, 2nd element holds simple FLAP value list, 3rd element hold cosine amplitude list
data_list = ([],[],[]) #1st and 2nd element contains lists of data from simple analysis (FRAP and FLAP) and 3rd element contains lists of data from plot profile
                        # each element in each list includes in order: Alpha, Diffusion coefficient, Tau(half-time), Length 
time0 = time.time()
def FRAP_Model(t, A, C, 𝜏):
   return (A * (1 - C * np.exp(-t / 𝜏)))

def FLAP_Model(t, A, B, 𝜏2):
   return (A * np.exp(-t / 𝜏2) + B)

def COSINE_FIT_Model(x, I0, I_amplitude):
    return (I0 + (I_amplitude * np.cos(np.pi * x / length)))

def generate_shape_midPoints(num_points):
    midPoints_vals = np.linspace(0 + length_correction, length - length_correction, num_points)  # created spaced out points on the shape mid points
    z_mid = amplitude * np.sin((2 * np.pi * midPoints_vals) / pitch) #parametric equations
    y_mid = amplitude * np.cos((2 * np.pi * midPoints_vals) / pitch)
    x_mid = midPoints_vals
    shape_points = np.column_stack((x_mid, y_mid, z_mid))
    return shape_points[np.newaxis, :, :]

def generate_random_particles(num_particles, shape_spawn_points):
    random_particles_indices = np.random.choice(len(shape_spawn_points[0,:]), size= num_particles, replace= True) #get random shape midPoint indices for spawning
    random_particles = shape_spawn_points[0,:][random_particles_indices] #get random shape midPoints for spawning
    theta_random, phi_random, distance_random = (np.random.rand(num_particles) * 2 * np.pi,
                                                np.arccos(2 * np.random.rand(num_particles) - 1),
                                                np.cbrt(np.random.rand(num_particles) * (internal_radius**3 - 0.001))) # cube root and cube for even spawning and subtract 1nm to prevent immediate collision
    x_spawn, y_spawn, z_spawn = (np.cos(theta_random) * np.sin(phi_random) * distance_random,
                                 np.sin(theta_random) * np.sin(phi_random) * distance_random,
                                 np.cos(phi_random) * distance_random)
    spawn_points = np.column_stack((x_spawn, y_spawn, z_spawn))
    random_particles += spawn_points #arrange particles at a random point in a sphere around random midpoints
    return random_particles

def diffuse_particles(particles_expanded, diffusion_constant):
    displacement = np.random.normal(loc=0, scale=np.sqrt(2 * diffusion_constant * simulated_interval))
    return particles_expanded + displacement, displacement

def distance_to_shape_midPoints(shape_expanded, particles_expanded):
    squared_distances = np.sum((particles_expanded[:, np.newaxis, :] - shape_expanded) ** 2, axis=-1)
    min_indices = np.argmin(squared_distances, axis=1)
    closest_points = shape_expanded[0][min_indices]
    min_distances = np.sqrt(np.min(squared_distances, axis=1))
    return closest_points, min_distances

def photobleach_particles(fluorescent_particles, diffusion_constant, bleach_region_particles_mask):
    bleach_random_chance = np.random.rand(len(fluorescent_particles)) #array of probabilities for bleaching
    bleach_outcome_mask = bleach_random_chance < bleach_probability #mask of bleachable particles that would be successful
    bleached_particles_mask = np.logical_and(bleach_outcome_mask, bleach_region_particles_mask)
    unbleached_particles_mask = ~bleached_particles_mask #mask of unbleachable particles (that would result in unsuccessful bleaching)
    return fluorescent_particles[unbleached_particles_mask], diffusion_constant[unbleached_particles_mask]

def detect_fluorescence(fluorescent_particles):
    inside_compartment_mask = min_distances <= internal_radius
    outside_compartment_mask = ~inside_compartment_mask
    if not bleach_box_scan_size:
        if shape == 2 and length <= pitch:
            bleach_region_mask = ((length-2*internal_radius)*fluorescent_particles[:,0] +
                                 amplitude*(math.cos(2*math.pi*(length-internal_radius)/pitch) - math.cos(2*math.pi*(internal_radius)/pitch))*fluorescent_particles[:,1] +
                                 amplitude*(math.sin(2*math.pi*(length-internal_radius)/pitch) - math.sin(2*math.pi*(internal_radius)/pitch))*fluorescent_particles[:,2]
                                 < length/2*(length-2*internal_radius))
        else:
            bleach_region_mask = fluorescent_particles[:,0] <= bleach_region  #Boolean mask of particles on the same side as the bleach region (based on the x-coordinate)
    else:
        bleach_region_mask_Left = fluorescent_particles[:,0] >= bleach_region - 0.5 * bleach_box_scan_size 
        bleach_region_mask_Right = fluorescent_particles[:,0] <= bleach_region + 0.5 * bleach_box_scan_size
        bleach_region_mask = np.logical_and(bleach_region_mask_Left, bleach_region_mask_Right) 
    Non_bleach_region_mask = ~bleach_region_mask
    bleach_region_particles_mask = np.logical_and(inside_compartment_mask,bleach_region_mask)  #Boolean mask of particles within the bleach region inside the shape compartment
    Non_bleach_region_particles_mask = np.logical_and(inside_compartment_mask,Non_bleach_region_mask)    
    fast_mask = np.logical_and(inside_compartment_mask, diffusion_constant[:,0] == fast_diffusion_constant)
    slow_mask = np.logical_and(inside_compartment_mask, diffusion_constant[:,0] == slow_diffusion_constant)
    immobile_mask = np.logical_and(inside_compartment_mask, diffusion_constant[:,0] == immobile_diffusion_constant)
    return (bleach_region_particles_mask, Non_bleach_region_particles_mask,
            inside_compartment_mask, outside_compartment_mask, fast_mask, slow_mask, immobile_mask)   

SKIP = False # Keep as true, do not change
DEFINITION = True
def update(frame): # Update the animation
    current_simulation_time = round(frame * simulated_interval, 3)
    global particles_expanded, min_distances, diffusion_constant, SKIP, DEFINITION
    if SKIP == False:     
        ax_3D.cla()  # Clear the current axes
        
        if current_simulation_time > round(bleach_start_time,3) and current_simulation_time < round(bleach_end_time, 3): #handles calls to photobleaching
            BLOCK_DETECTOR = True
            bleach_region_particles_mask = detect_fluorescence(particles_expanded)[0]
            particles_expanded, diffusion_constant = photobleach_particles(particles_expanded, diffusion_constant, bleach_region_particles_mask)
            closest_points, min_distances = distance_to_shape_midPoints(shape_expanded, particles_expanded)
            ax_3D.set_title(f"Particle diffusion inside a {shapeDict[shape]} (t = {current_simulation_time:.3f} s)\n" + r"$\bf{PHOTOBLEACHING}$", color='dodgerblue', x=0.5, y=1.05)
            
            vertices = [
                [(0 if not bleach_box_scan_size else bleach_region - 0.5 * bleach_box_scan_size, math.floor(np.min(shape_expanded, axis=1)[0][1] - internal_radius*2), math.floor(np.min(shape_expanded, axis=1)[0][2] - internal_radius*2)),
                 (bleach_region if not bleach_box_scan_size else bleach_region + 0.5 * bleach_box_scan_size , math.floor(np.min(shape_expanded, axis=1)[0][1] - internal_radius*2), math.floor(np.min(shape_expanded, axis=1)[0][2] - internal_radius*2)),
                 (bleach_region if not bleach_box_scan_size else bleach_region + 0.5 * bleach_box_scan_size , math.ceil(np.max(shape_expanded, axis=1)[0][1] + internal_radius*2), math.floor(np.min(shape_expanded, axis=1)[0][2] - internal_radius*2)),
                 (0 if not bleach_box_scan_size else bleach_region - 0.5 * bleach_box_scan_size , math.ceil(np.max(shape_expanded, axis=1)[0][1] + internal_radius*2), math.floor(np.min(shape_expanded, axis=1)[0][2] - internal_radius*2))],
                
                [(0 if not bleach_box_scan_size else bleach_region - 0.5 * bleach_box_scan_size, math.ceil(np.max(shape_expanded, axis=1)[0][1] + internal_radius*2), math.floor(np.min(shape_expanded, axis=1)[0][2] - internal_radius*2)),
                 (bleach_region if not bleach_box_scan_size else bleach_region + 0.5 * bleach_box_scan_size , math.ceil(np.max(shape_expanded, axis=1)[0][1] + internal_radius*2), math.floor(np.min(shape_expanded, axis=1)[0][2] - internal_radius*2)),
                 (bleach_region if not bleach_box_scan_size else bleach_region + 0.5 * bleach_box_scan_size , math.ceil(np.max(shape_expanded, axis=1)[0][1] + internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][2] + internal_radius*2)),
                 (0 if not bleach_box_scan_size else bleach_region - 0.5 * bleach_box_scan_size , math.ceil(np.max(shape_expanded, axis=1)[0][1] + internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][2] + internal_radius*2))]
            ]
            ax_3D.add_collection3d(Poly3DCollection(vertices, color='dodgerblue', alpha=0.1))
        else:
            BLOCK_DETECTOR = False
            ax_3D.set_title(f"Particle diffusion inside a {shapeDict[shape]} (t = {current_simulation_time:.3f} s)", color='white', x=0.5, y=1.05)
            
        (bleach_region_particles_mask, Non_bleach_region_particles_mask,inside_compartment_mask,
         outside_compartment_mask, fast_mask, slow_mask, immobile_mask) = detect_fluorescence(particles_expanded)
        particles_expanded, displacement = diffuse_particles(particles_expanded, diffusion_constant)
        closest_points, min_distances = distance_to_shape_midPoints(shape_expanded, particles_expanded)
        
        if APPLY_COMPARTMENT_COLLISIONS: #Avoid turning this into a function to prevent particles slipping through the cracks
            collision_mask = min_distances > internal_radius  # Particles that have collided
            while np.any(collision_mask):
                collision_vectors = particles_expanded[collision_mask] - closest_points[collision_mask]# get vectors from closest shape midPoints to particle positions
                norms = np.linalg.norm(collision_vectors, axis=1).reshape(-1, 1)# Normalize the collision vectors
                norms[norms == 0] = 1e-8 # Avoid division by zero (0.00000001 nm wiggle room)
                normal_vectors = collision_vectors / norms
                particles_expanded[collision_mask] -= 2 * (norms - internal_radius) * normal_vectors # get reflected displacements: v' = v - 2(v · n)n
                closest_points, min_distances = distance_to_shape_midPoints(shape_expanded, particles_expanded)
                collision_mask = min_distances > internal_radius
    
        if SHOW_MIDPOINTS: ax_3D.scatter(shape_expanded[0, :, 0], shape_expanded[0, :, 1], shape_expanded[0, :, 2], color='white', label='Discretized helical curve', lw=0.1, marker='.')
        if numFastParticles != 0: ax_3D.scatter(particles_expanded[fast_mask, 0], particles_expanded[fast_mask, 1], particles_expanded[fast_mask, 2],
                   color='lime', alpha=0.75, s=20, label=f"Inside {shapeDict[shape]} [FAST] N= {fast_mask.sum()}", edgecolors='none')
        if numSlowParticles != 0: ax_3D.scatter(particles_expanded[slow_mask, 0], particles_expanded[slow_mask, 1], particles_expanded[slow_mask, 2],
                   color='cyan', alpha=0.75, s=20, label=f"Inside {shapeDict[shape]} [SLOW] N= {slow_mask.sum()}", edgecolors='none')
        if numImmobileParticles != 0: ax_3D.scatter(particles_expanded[immobile_mask, 0], particles_expanded[immobile_mask, 1], particles_expanded[immobile_mask, 2],
                   color='fuchsia', alpha=0.75, s=20, label=f"Inside {shapeDict[shape]} [IMMOBILE] N= {immobile_mask.sum()}", edgecolors='none')
        if outside_compartment_mask.sum() != 0: ax_3D.scatter(particles_expanded[outside_compartment_mask, 0], particles_expanded[outside_compartment_mask, 1], particles_expanded[outside_compartment_mask, 2],
                   color='r', s=20, alpha=0.75, label=f"Outside {shapeDict[shape]} N= {outside_compartment_mask.sum()}", edgecolors='none')
        if DRAW_DISTANCES_TO_MIDPOINTS:  
            ax_3D.scatter(closest_points[:, 0], closest_points[:, 1], closest_points[:, 2], color='k', s=20, label='Closest Points') #optional: can comment this out #plots closest points
            for point, closest_point in zip(particles_expanded, closest_points): #optional: can comment this loop out, it plots distances
                ax_3D.plot([point[0], closest_point[0]], [point[1], closest_point[1]], [point[2], closest_point[2]], color='orange', linestyle='--', linewidth=0.5)
          
        ax_3D.set_xlabel('(µm)', color='white' if graphTransformation < 2 else 'black')
        ax_3D.xaxis.labelpad = 20
        ax_3D.tick_params(axis='x', colors='white' if graphTransformation < 2 else 'black')
        ax_3D.set_ylabel('', color='white')
        ax_3D.yaxis.labelpad = 5
        ax_3D.set_zlabel('', color='white')
        ax_3D.zaxis.labelpad = 5
        ax_3D.set_zlim(math.floor(np.min(shape_expanded, axis=1)[0][2] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][2] + internal_radius*2))
        ax_3D.set_ylim(math.floor(np.min(shape_expanded, axis=1)[0][1] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][1] + internal_radius*2))
        ax_3D.set_xlim(math.floor(np.min(shape_expanded, axis=1)[0][0] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][0] + internal_radius*2))
        ax_3D.zaxis.set_ticks(np.arange(math.floor(np.min(shape_expanded, axis=1)[0][2] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][2] + internal_radius*2), 1))
        ax_3D.yaxis.set_ticks(np.arange(math.floor(np.min(shape_expanded, axis=1)[0][1] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][1] + internal_radius*2), 1))
        ax_3D.xaxis.set_ticks(np.arange(math.floor(np.min(shape_expanded, axis=1)[0][0] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][0] + internal_radius*2), 1))
        ax_3D.set_box_aspect([scale_factor, 1, 1])  # Equal scaling for all axes
        # ax_3D.set_box_aspect([length , length * internal_radius, internal_radius])  # Equal scaling for all axes
        # ax_3D.set_box_aspect([length, (amplitude*temp) if amplitude != 0 else internal_radius*2, (amplitude*temp) if amplitude != 0 else internal_radius*2])  # Equal scaling for all axes
        
        ax_3D.legend(bbox_to_anchor=(1.98, 1.05), facecolor= 'black', labelcolor='white', framealpha=1, draggable=True)
        if graphTransformation >= 0: ax_3D.view_init(30 + 45 * frame if graphTransformation >= 2 else 30,
                                                  -150 * abs(current_simulation_time / (simulated_seconds * 2/3) - math.floor(current_simulation_time / (simulated_seconds * 2/3) + 0.5)) - 90 if graphTransformation == 1 else -90,
                                                  0 + frame if graphTransformation == 3 else 0)
    
        ax_FRAP.set_xlabel('Time (s)')
        ax_FRAP.set_ylabel('Fluorescence proportion (Arbitrary units)')
        ax_FRAP.set_ylim(0, 1)
        ax_FRAP.set_xlim(0, simulated_seconds)
        ax_INTENSITY_PROFILE.set_xlabel(f'Position along {shapeDict[shape]} without endcaps (µm)\nPixel resolution: {pixel_resolution * 1000:.2f} nm')
        ax_INTENSITY_PROFILE.set_ylabel('Fluorescence proportion (Arbitrary units)')
        ax_INTENSITY_PROFILE.set_xlim(0, length)
        ax_COSINE_AMPLITUDES.set_xlabel('Time (s)')
        ax_COSINE_AMPLITUDES.set_ylabel('Fluorescence amplitudes of cosine fits (Arbitrary units)')
        ax_COSINE_AMPLITUDES.set_xlim(0, simulated_seconds)
    
        if not BLOCK_DETECTOR and Decimal(str(current_simulation_time)) % Decimal(str(frame_interval)) == 0 and SKIP == False:
            time_list.append(current_simulation_time)
            FRAP_values[0].append(bleach_region_particles_mask.sum() / numParticles) #FRAP_values[0] is Fluorescence recovery, FRAP_values[1] is fluorescence loss
            FRAP_values[1].append(Non_bleach_region_particles_mask.sum() / numParticles)
            ax_FRAP.scatter(current_simulation_time, FRAP_values[0][-1], color= (255/255, 97/255, 1/255), linewidth=1,
                            s=30, alpha=0.8, label= "Bleach region fluorescence" if frame == 0 else None, edgecolors='black')
            ax_FRAP.scatter(current_simulation_time, FRAP_values[1][-1], color= (255/255, 0/255, 255/255), linewidth=1,
                            s=30, alpha=0.8, label= "Non-Bleach region fluorescence" if frame == 0 else None, edgecolors='black')
            if shape != 0 and Decimal(str(length)) >= Decimal(str(internal_radius*2 + pixel_resolution)): #plot profiles not supported for spheres
                if current_simulation_time == round(bleach_end_time, 3): [plot_lines.remove() for plot_lines in ax_INTENSITY_PROFILE.lines]
                fluorescence_profile = (np.count_nonzero((particles_expanded[inside_compartment_mask, 0] >= pixel_intervals)
                                                        & (particles_expanded[inside_compartment_mask, 0] < pixel_intervals + pixel_resolution), axis= 1)) / numParticles  
    
                ax_INTENSITY_PROFILE.plot(pixel_intervals, fluorescence_profile, alpha=0.75, color= (colorMap(frame/simulated_range)),
                                linestyle='-', linewidth=1)
                ax_INTENSITY_PROFILE.relim()
                ax_INTENSITY_PROFILE.autoscale_view()
                optimalValues, covarianceMatrix = curve_fit(COSINE_FIT_Model, pixel_intervals[cosineFittingMask], fluorescence_profile[cosineFittingMask.squeeze()], p0= [0.0,0.0])
                
                c1_opt, I_amplitude_opt = optimalValues
                FRAP_values[2].append(I_amplitude_opt)
                x_fit = np.linspace(pixel_intervals[cosineFittingMask], fluorescence_profile[cosineFittingMask.squeeze()], 100)
                y_fit = COSINE_FIT_Model(x_fit, c1_opt, I_amplitude_opt)
                ax_INTENSITY_PROFILE.plot(x_fit, y_fit, color=(colorMap(frame/simulated_range)))
                ax_COSINE_AMPLITUDES.scatter(current_simulation_time, FRAP_values[2][-1], color= (colorMap(frame/simulated_range)), marker='o',
                                s=30, alpha=0.5, edgecolors='black')
                
        if frame == 0 and DEFINITION == True:
            DEFINITION = False
            ax_FRAP.plot([0,1e-10],[0,0], color='red', linestyle='-', linewidth=1, label='Red line fit')
            ax_FRAP.legend(loc='best', draggable=True)
            ax_COSINE_AMPLITUDES.legend(loc='best', draggable=True)
            
        if frame == simulated_range and SKIP == False:
            SKIP = True
            frame_list = np.array(time_list)
            for count in range(len(FRAP_values)):
                optimalValues, covarianceMatrix = curve_fit(modelDict[count], frame_list[frame_list >= round(bleach_end_time, 3)], np.array(FRAP_values[count][np.argmax(frame_list >= round(bleach_end_time, 3)):]), p0=[0,0.5,5])
                c1_opt, c2_opt, 𝜏_opt = optimalValues
                data_list[count].append([(fast_diffusion_constant * 𝜏_opt )/ length**2, fast_diffusion_constant, 𝜏_opt, length])
                x_fit = np.linspace(frame_list[frame_list >= round(bleach_end_time, 3)][0], frame_list[-1], 100)
                y_fit = modelDict[count](x_fit, c1_opt, c2_opt, 𝜏_opt)    
                graphDict[count].plot(x_fit, y_fit, color='r', label= 'red line fit')            
                
            fig.text(label_pos_x1, label_pos_y1, f'τ = {data_list[0][0][2]:.4f} s', fontsize=16, ha='center', va='center', bbox=dict(facecolor='lightblue', alpha=0.5))
            fig.text(label_pos_x2, label_pos_y2, f'τ = {data_list[2][0][2]:.4f} s', fontsize=16, ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5))

            print(f"Bleach Region Analysis: alpha | diffusion constant | tau | cell length \n{data_list[0][0]}"
                  f"\nNon-Bleach Region Analysis: alpha | diffusion constant | tau | cell length \n{data_list[1][0]}"
                  f"\nProfile Analysis: alpha | diffusion constant | tau | cell length \n{data_list[2][0]}")
            print(f"this took {time.time() - time0:.2f} sec to be animated with {numParticles} particles " 
                  f"for {simulated_seconds} simulated second(s) ({simulated_range} iterations)")

            if not SAVE_ANIMATION:
                plt.show()
            
    else:
        ax_3D.set_title(f"Particle diffusion inside a {shapeDict[shape]} (t = {simulated_seconds:.3f} s)\n" + r"$\bf{END-OF-SIMULATION}$", color='red', x=0.5, y=1.05)


shape_expanded = generate_shape_midPoints(int(Decimal(str(contourLength)) / Decimal(str(segment_resolution))))
particles_expanded = generate_random_particles(numParticles, shape_expanded)
closest_points, min_distances = distance_to_shape_midPoints(shape_expanded, particles_expanded)

fig = plt.figure(figsize=(16, 9)) # Set up the figure and 3D axis
ax_3D = fig.add_subplot(2,1,1, projection='3d')
ax_3D.xaxis.pane.fill = False
ax_3D.yaxis.pane.fill = False
ax_3D.zaxis.pane.fill = False
# # ax_3D.style.use('default')
ax_3D.set_facecolor('black')
if graphTransformation >= 2:
    ax_3D.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3D.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3D.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3D.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax_3D.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax_3D.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
# ax_black_background = fig.add_axes([0, 0.5, 1, 0.5], facecolor='black', zorder=-1)
ax_black_background = fig.add_axes([0, 0.5, 1, 0.5], facecolor='black', zorder=-1)

ax_black_background.set_xticks([])
ax_black_background.set_yticks([])
ax_black_background.set_xticklabels([])
ax_black_background.set_yticklabels([])
ax_FRAP = fig.add_subplot(2, 3, 4)
ax_FRAP.set_title(f"FRAP of a {shapeDict[shape]}")
ax_INTENSITY_PROFILE = fig.add_subplot(2, 3, 5)
ax_INTENSITY_PROFILE.set_title("Intensity Profiles")
ax_COSINE_AMPLITUDES = fig.add_subplot(2, 3, 6)
ax_COSINE_AMPLITUDES.set_title("Cosine Amplitudes vs Time")
# ax_COSINE_AMPLITUDES.plot([0,1e-10],[0,0], color='red', linestyle='-', linewidth=1, label='Red line fit')
fig.tight_layout(pad=5)
plt.subplots_adjust(bottom=0.07, left=0.04, right=0.98, top=0.95, hspace=0.19)
graphDict = {0: ax_FRAP, 1: ax_FRAP, 2: ax_COSINE_AMPLITUDES, 3: ax_INTENSITY_PROFILE}
modelDict = {0: FRAP_Model, 1: FLAP_Model, 2: FLAP_Model, 3: COSINE_FIT_Model}

label_pos_x1, label_pos_x2, label_pos_y1, label_pos_y2 = 0.25 * sum(ax_FRAP.get_xlim()), 0.926 * sum(ax_COSINE_AMPLITUDES.get_xlim()), 0.3 * sum(ax_FRAP.get_ylim()), 0.3 * sum(ax_COSINE_AMPLITUDES.get_ylim())

animate = animation.FuncAnimation(fig, update, frames=simulated_range + 200, interval=frame_interval*1000, repeat= False) #animate through time
if SAVE_ANIMATION:
    writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    animate.save(saveFolder + f"Animation of Simulated AMB-1, L_T= 4.62.gif", writer=writer)
