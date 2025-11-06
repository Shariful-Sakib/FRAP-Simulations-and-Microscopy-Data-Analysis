# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 19:24:26 2024
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit;
import sys
from decimal import Decimal

COLLECT_DATA = False # Toggle to 
colorMap = LinearSegmentedColormap.from_list('Black', ['red','aqua','black'])
# option = ctypes.windll.user32.MessageBoxW(0, 'Please select the folder with the FRAP data in the next pop-up window', 'FRAP Analyzer - User Input Required', 0x01|0x40|0x00001000);
# if option == 2:
#     sys.exit(); #Stops the operation if user cancels
    
# tkinter.Tk().withdraw();
# directory = str(filedialog.askdirectory()) + "/"; #Sends the directory to the macro
# if directory == '/':
#     sys.exit(); #Stops the operation if user cancels 

#ignore this syntax error
# %matplotlib auto
#NOTE: can also comment out %matplotlib auto and then go to Tools>Preferences>IPython console>Graphics>Backend>Automatic

# fill in any parameters of interest.
# Don't worry about leaving other parameters in.
# Program is robust and will use what it needs to make the necessary shape.
# For spheres, set shape = 0 and you can vary internal radius.
# For rods, set shape = 1 and you can vary internal radius and length.
# For spirilla, set shape = 2 and you can vary everything.
# Finally, change any other setting for the FRAP experiment.

shape = 2 # 0 = sphere, 1 = rod, 2 = helical cell
shapeDict = {0: "sphere", 1:"rod", 2:"helical cell"}

internal_radius = 0.27 #um
length = 4.62 #um
pitch = 2.5#um
amplitude = 0.2 #um
contourLength = (length - internal_radius*2) * math.sqrt(1 + (2*math.pi*amplitude/pitch)**2) + internal_radius*2

scale_factor = 3#max(2.64 * length / 5, 1)    #graphing purposes
length_correction = internal_radius         #ensures the shape produced is of the correct length
max_segment_error = 0.0042              #um ~4.2 nm maximum error between true and calculated distances (approximate length of an FP molecule)
segment_resolution = 2*math.sqrt((max_segment_error + internal_radius)**2 - internal_radius**2)
pixel_resolution = 0.085                    #um 85 nm resolution used for creating plot profiles

numParticles = 1000
fast_diffusion_constant = 5 # um^2/s
immobile_diffusion_constant = 0 #um^2/s
slow_diffusion_constant = 5 # um^2/s
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

SHOW_HIGHLIGHTS = True # Can toggle to get a play-by-play of all the steps: initialization, pre-bleach, mid-bleach, post bleach. (optional)
SHOW_FINAL_VISUAL = True # Can toggle to show the end result visually
frame_interval = 0.030 #sec
time_step = 0.001 #sec

bleach_duration = frame_interval                             #s how long should the bleach be
bleach_start_time = frame_interval * 4                          #s what timepoint should the bleach begin
bleach_end_time = bleach_start_time + bleach_duration
gamma = 5 
fluorophore_lifetime = frame_interval / gamma 
bleach_probability = time_step / fluorophore_lifetime  #probability of bleaching
bleach_region_fraction = 0.5                                 #proportion of compartment to bleach
bleach_box_scan_size = None
bleach_region = length * bleach_region_fraction                 #x values less than and  equal to this value will be bleached
pixel_intervals = np.reshape(np.arange(0 + internal_radius, length - internal_radius - pixel_resolution, pixel_resolution), (-1, 1)) #used for plot profile segmentation, also avoids end caps

if bleach_box_scan_size:
    cosineFittingMask = pixel_intervals >= bleach_region - 0.5 * bleach_box_scan_size if bleach_region_fraction <= 0.5 else pixel_intervals <= bleach_region + 0.5 * bleach_box_scan_size
else:
    cosineFittingMask = pixel_intervals >= 0

if shape <= 1:
    amplitude = 0
    contourLength = length
if shape == 0:
    length = internal_radius * 2        # corrects length for spheres
    bleach_region = length * bleach_region_fraction - internal_radius    # corrects bleach_region for spheres since they are centered at (0,0,0)
    segment_resolution = length         # corrects segment_resolution for spheres since they are defined by one point (0,0,0)
    scale_factor = 1
    length_correction = 0

simulated_seconds = math.ceil(contourLength**2*((3/4) if length <= 5 else (2/3))/(2*fast_diffusion_constant) + 1)  #sec Simulated total duration of the experiment

simulated_range = int((simulated_seconds / frame_interval) * (frame_interval / time_step)) #iterations
    
time_list = ([])            # collects list of time data
FRAP_values = ([],[],[])    # 1st element holds simple FRAP value list, 2nd element holds simple FLAP value list, 3rd element hold cosine amplitude list
profile_list = ([])         # collects list of plot profile data
data_list = ([],[],[])      # 1st and 2nd element contains lists of data from simple analysis (FRAP and FLAP) and 3rd element contains lists of data from plot profile
                            # each element in each list includes in order: Alpha, Diffusion coefficient, Tau(half-time), Length 
time0 = time.time()
def FRAP_Model(t, A, C, 𝜏):
   return (A * (1 - C * np.exp(-t / 𝜏)))

def FLAP_Model(t, A, B, 𝜏2):
   return (A * np.exp(-t / 𝜏2) + B)

def COSINE_FIT_Model(x, I0, I_amplitude):
    return (I0 + (I_amplitude * np.cos(np.pi * x / length)))

def generate_shape_midPoints(num_points):
    midPoints_vals = np.linspace(0 + length_correction, length - length_correction, num_points)  # created spaced out points on the shape midPoints
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

def diffuse_particles(particles_expanded):
    displacement = np.random.normal(loc=0,scale=np.sqrt(2*diffusion_constant*time_step))
    return particles_expanded + displacement, displacement

def distance_to_shape_midPoints(shape_expanded, particles_expanded):
    squared_distances = np.sum((particles_expanded[:, np.newaxis, :] - shape_expanded) ** 2, axis=-1) 
    min_indices = np.argmin(squared_distances, axis=1) # Find index of the closest shape point for each particle
    closest_points = shape_expanded[0][min_indices]  # Get closest points from the shape
    min_distances = np.sqrt(np.min(squared_distances, axis=1))  # Calculate actual distances
    return closest_points, min_distances  

def photobleach_particles(fluorescent_particles, diffusion_constant, bleach_region_particles_mask):
    bleach_random_chance = np.random.rand(len(fluorescent_particles)) #array of probabilities for bleaching
    bleach_outcome_mask = bleach_random_chance < bleach_probability #mask of bleachable particles that would be successful
    bleached_particles_mask = np.logical_and(bleach_outcome_mask, bleach_region_particles_mask)
    unbleached_particles_mask = ~bleached_particles_mask #mask of unbleachable particles (that would result in unsuccessful bleaching)
    return fluorescent_particles[unbleached_particles_mask], diffusion_constant[unbleached_particles_mask]

def detect_fluorescence(fluorescent_particles, min_distances):
    inside_compartment_mask = min_distances <= internal_radius
    outside_compartment_mask = ~inside_compartment_mask
    if not bleach_box_scan_size:
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
     
def visualize_environment(midPoints, particles, closest_points, min_distances, PLOT_GRAPHS, sim_time_now, title): 
    fig = plt.figure(figsize=(16, 9)) # Set up the figure and 3D axis
    ax_3D = fig.add_subplot(2, 1, 1, projection='3d')
    # ax_3D.xaxis.pane.fill = False
    # ax_3D.yaxis.pane.fill = False
    # ax_3D.zaxis.pane.fill = False
    # # ax_3D.style.use('default')
    ax_3D.set_facecolor('black')
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
    ax_COSINE_AMPLITUDES.plot([0,1e-10],[0,0], color='red', linestyle='-', linewidth=1, label='Red line fit')
    fig.tight_layout()
    graphDict = {0: ax_FRAP, 1: ax_FRAP, 3: ax_COSINE_AMPLITUDES, 2: ax_INTENSITY_PROFILE}  #note 3 and 2 are swapped in the non-animated version
    modelDict = {0: FRAP_Model, 1: FLAP_Model, 3: FLAP_Model, 2: COSINE_FIT_Model}          #note 3 and 2 are swapped in the non-animated version
    
    (bleach_region_particles_mask, Non_bleach_region_particles_mask,
     inside_compartment_mask, outside_compartment_mask, fast_mask, slow_mask, immobile_mask) = detect_fluorescence(particles_expanded, min_distances)
    
    if SHOW_MIDPOINTS: ax_3D.scatter(midPoints[0, :, 0], midPoints[0, :, 1], midPoints[0, :, 2], color='blue', label='Discretized helical curve', lw=0.5) # Plot the entire midPoints
    if numFastParticles != 0: ax_3D.scatter(particles[fast_mask, 0], particles[fast_mask, 1], particles[fast_mask, 2],
               color= 'lime', alpha=0.75, s=20, label=f"Fluorophores Remaining N= {fast_mask.sum()}", edgecolors='none')
    if numSlowParticles != 0: ax_3D.scatter(particles[slow_mask, 0], particles[slow_mask, 1], particles[slow_mask, 2],
               color='cyan', alpha=0.75, s=20, label=f"Inside {shapeDict[shape]} [SLOW] N= {slow_mask.sum()}", edgecolors='none')
    if numImmobileParticles != 0:ax_3D.scatter(particles[immobile_mask, 0], particles[immobile_mask, 1], particles[immobile_mask, 2],
               color='fuchsia', alpha=0.75, s=20, label=f"Inside {shapeDict[shape]} [IMMOBILE] N= {immobile_mask.sum()}", edgecolors='none')
    if outside_compartment_mask.sum() != 0: ax_3D.scatter(particles[outside_compartment_mask, 0], particles[outside_compartment_mask, 1], particles[outside_compartment_mask, 2],
               color='r', s=20, alpha=0.75, label=f"Outside {shapeDict[shape]} N= {outside_compartment_mask.sum()}")
    if DRAW_DISTANCES_TO_MIDPOINTS:      
        ax_3D.scatter(closest_points[:, 0], closest_points[:, 1], closest_points[:, 2], color='k', s=20, label='Closest Points') #optional: can comment this out #plots closest points
        for point, closest_point in zip(particles, closest_points): #optional: can comment this loop out # Draw distance lines
            ax_3D.plot([point[0], closest_point[0]], [point[1], closest_point[1]], [point[2], closest_point[2]], color='orange', linestyle='--', linewidth=0.5)

    if PLOT_GRAPHS:
        ax_FRAP.scatter(time_list, FRAP_values[0], color='orange',
                        s=50, label=f"Fluorescence within {shapeDict[shape]} inside bleach region", edgecolors='black')
        ax_FRAP.scatter(time_list, FRAP_values[1], color='fuchsia',
                        s=50, label=f"Fluorescence within {shapeDict[shape]} outside bleach region", edgecolors='black')
        # ax_FRAP.scatter(time_list, np.array(FRAP_values[1]) - np.array(FRAP_values[0]), color='deepskyblue',
        #                 s=20, label=f"Fluorescence within {shapeDict[shape]} outside bleach region", edgecolors='black')
        
        for plotCount in range(4 if shape != 0 else 2):
            if plotCount == 2:
                FRAP_values[plotCount].clear()
                for profile_count, profile_plots in enumerate(profile_list):
                    try:
                        ax_INTENSITY_PROFILE.plot(pixel_intervals, profile_plots, alpha=0.75, color=colorMap(profile_count/len(profile_list)),
                                        linestyle='-', linewidth=0.5, label=f"Fluorescence within {shapeDict[shape]}")
                        optimalValues, covarianceMatrix = curve_fit(modelDict[2], pixel_intervals[cosineFittingMask], profile_plots[cosineFittingMask.squeeze()], p0=[0.0, 0.0], maxfev=10000)
                        c1_opt, I_amplitude_opt = optimalValues
                        FRAP_values[plotCount].append(I_amplitude_opt)
                        x_fit = np.linspace(pixel_intervals[cosineFittingMask], profile_plots[cosineFittingMask.squeeze()], 100)
                        y_fit = modelDict[plotCount](x_fit, c1_opt, I_amplitude_opt)
                        graphDict[plotCount].plot(x_fit, y_fit, color=(colorMap(profile_count/len(profile_list))))
                        graphDict[plotCount + 1].scatter(time_list[profile_count], FRAP_values[plotCount][-1], color= colorMap(profile_count/len(profile_list)), marker='o',s=30, alpha=0.5, edgecolors='black')
                    except:
                        pass
            else:
                if sim_time_now * 1000 >= simulated_range:
                    frame_list = np.array(time_list)
                    ax_FRAP.axvline(x = bleach_start_time, color = 'dodgerblue', label = f'Photobleaching at {round(bleach_start_time, 4)} s', linestyle= 'dashed')
                    ax_COSINE_AMPLITUDES.axvline(x = bleach_start_time, color = 'dodgerblue', label = f'Photobleaching at {round(bleach_start_time, 4)} s', linestyle= 'dashed')
                    # print(np.argmax(frame_list >= round(bleach_end_time, 3)))
                    # print(frame_list[frame_list >= round(bleach_end_time, 3)])
                    try:  
                        optimalValues, covarianceMatrix = curve_fit(modelDict[plotCount], frame_list[frame_list >= round(bleach_end_time, 3)],
                                                                    np.array(FRAP_values[2 if plotCount == 3 else plotCount][np.argmax(frame_list >= round(bleach_end_time, 3)):]), p0=[(1.0 - bleach_region_fraction), 1.0, 0.5] if modelDict[plotCount] == FRAP_Model else [10.0, (1.0- bleach_region_fraction), 0.5], maxfev=10000)
                        c1_opt, c2_opt, 𝜏_opt = optimalValues
                        data_list[2 if plotCount == 3 else plotCount].append([f"Alpha: {(fast_diffusion_constant * 𝜏_opt )/ length**2}",
                                                                              f"Diffusion Constant: {fast_diffusion_constant}",
                                                                              f"Tau: {𝜏_opt}"])
                        x_fit = np.linspace(frame_list[frame_list >= round(bleach_end_time, 3)], frame_list[-1], 100)
                        y_fit = modelDict[plotCount](x_fit, c1_opt, c2_opt, 𝜏_opt)
                        graphDict[plotCount].plot(x_fit, y_fit, color='r', linewidth = 0.75)
                    except:
                        pass

    ax_3D.set_title(title, color='white', x=0.5, y=1.0)
    ax_3D.set_xlabel('(µm)', color='white')
    ax_3D.xaxis.labelpad = 20
    ax_3D.set_ylabel('(µm)', color='white')
    ax_3D.yaxis.labelpad = 5
    ax_3D.set_zlabel('(µm)', color='white')
    ax_3D.zaxis.labelpad = 5
    
    ax_3D.set_title(title, color='white', x=0.5, y=1.0)
    ax_3D.set_xlabel(None)
    ax_3D.xaxis.labelpad = 20
    ax_3D.set_ylabel(None)
    ax_3D.yaxis.labelpad = 5
    ax_3D.set_zlabel(None)
    ax_3D.zaxis.labelpad = 5
    
    ax_3D.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3D.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3D.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax_3D.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax_3D.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax_3D.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax_3D.view_init(60,-90,0)
# ax_black_background = fig.add_axes([0, 0.5, 1, 0.5], facecolor='black', zorder=-1)
    
    ax_3D.set_zlim(math.floor(np.min(shape_expanded, axis=1)[0][2] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][2] + internal_radius*2))
    ax_3D.set_ylim(math.floor(np.min(shape_expanded, axis=1)[0][1] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][1] + internal_radius*2))
    ax_3D.set_xlim(math.floor(np.min(shape_expanded, axis=1)[0][0] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][0] + internal_radius*2))
    ax_3D.zaxis.set_ticks(np.arange(math.floor(np.min(shape_expanded, axis=1)[0][2] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][2] + internal_radius*2), 1))
    ax_3D.yaxis.set_ticks(np.arange(math.floor(np.min(shape_expanded, axis=1)[0][1] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][1] + internal_radius*2), 1))
    ax_3D.xaxis.set_ticks(np.arange(math.floor(np.min(shape_expanded, axis=1)[0][0] - internal_radius*2), math.ceil(np.max(shape_expanded, axis=1)[0][0] + internal_radius*2), 1))
    ax_3D.tick_params(colors='white')
    ax_3D.set_box_aspect([scale_factor, 1, 1])  # Equal scaling for all axes
    ax_3D.legend(bbox_to_anchor=(1.98, 1.05), facecolor= 'white', framealpha=1, draggable=True)
    
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
    plt.subplots_adjust(bottom=0.07, left=0.04, right=0.98, top=0.95, hspace=0.19, wspace= 0.23)
    # fig.tight_layout()
    plt.show()
plt.close("all")
shape_expanded = generate_shape_midPoints(int(Decimal(str(contourLength)) / Decimal(str(segment_resolution)))) #section the shape path
particles_expanded = generate_random_particles(numParticles, shape_expanded)  # Generate random point particles
closest_points, min_distances = distance_to_shape_midPoints(shape_expanded, particles_expanded)
if SHOW_HIGHLIGHTS: visualize_environment(shape_expanded, particles_expanded, closest_points, min_distances, False, 0.00, f"Particle diffusion inside a {shapeDict[shape]} (INITIALIZATION) \n Elapsed time: {0:.3f} s")

for count in range(simulated_range): #iterate through time
    current_simulation_time = round(count * time_step, 3)
    print("\r", f"iteration {count} out of {simulated_range}", end="")
    sys.stdout.flush()

    if current_simulation_time > round(bleach_start_time,3) and current_simulation_time < round(bleach_end_time, 3): #handles calls to photobleaching
        bleach_region_particles_mask = detect_fluorescence(particles_expanded, min_distances)[0]   
        particles_expanded, diffusion_constant = photobleach_particles(particles_expanded, diffusion_constant, bleach_region_particles_mask) #Photobleach particles, redimension array shapes

        if (Decimal(str(current_simulation_time)) - Decimal(str(bleach_start_time))) % (Decimal(str(bleach_duration)) / 2) == 0 and SHOW_HIGHLIGHTS: #show frames pre, mid, and post bleach
            closest_points, min_distances = distance_to_shape_midPoints(shape_expanded, particles_expanded)
            visualize_environment(shape_expanded, particles_expanded, closest_points, min_distances, True, current_simulation_time,
                                    f"Particle diffusion inside a {shapeDict[shape]} \n BLEACHING: {current_simulation_time - bleach_start_time:.3f} s \n Elapsed time: {count * time_step:.3f} s") 
    else:

        if Decimal(str(current_simulation_time)) % Decimal(str(frame_interval)) == 0: #this block of code collects data
            (bleach_region_particles_mask, Non_bleach_region_particles_mask,
             inside_compartment_mask, outside_compartment_mask, fast_mask, slow_mask, immobile_mask) = detect_fluorescence(particles_expanded, min_distances)
            time_list.append(current_simulation_time)
            FRAP_values[0].append(bleach_region_particles_mask.sum() / numParticles)
            FRAP_values[1].append(Non_bleach_region_particles_mask.sum() / numParticles)
            profile_list.append(np.count_nonzero((particles_expanded[inside_compartment_mask, 0] >= pixel_intervals)
                                                    & (particles_expanded[inside_compartment_mask, 0] < pixel_intervals + pixel_resolution), axis= 1) / numParticles)
            
    particles_expanded, displacement = diffuse_particles(particles_expanded) #diffusion
    closest_points, min_distances = distance_to_shape_midPoints(shape_expanded, particles_expanded) #distance search to see where particles are in reference to shape compartment
    
    if APPLY_COMPARTMENT_COLLISIONS: #Avoid turning this into a function to prevent particles slipping through the cracks (determined by testing)
        collision_mask = min_distances > internal_radius  # Particles that have collided (went outside the compartment)
        while np.any(collision_mask):    
            collision_vectors = particles_expanded[collision_mask] - closest_points[collision_mask]# get vectors from closest shape midPoints to particle positions
            norms = np.linalg.norm(collision_vectors, axis=1).reshape(-1, 1)# Normalize the collision vectors
            norms[norms == 0] = 1e-8 # Avoid division by zero (0.00000001 nm wiggle room)
            normal_vectors = collision_vectors / norms 
            particles_expanded[collision_mask] -= 2 * (norms - internal_radius) * normal_vectors # get reflected displacements: v' = v - 2(v · n)n
            closest_points, min_distances = distance_to_shape_midPoints(shape_expanded, particles_expanded)
            collision_mask = min_distances > internal_radius

print(f"this took {time.time() - time0:.2f} sec with {numParticles} particles for {simulated_seconds} simulated second(s) ({simulated_range} iterations)")
visualize_environment(shape_expanded, particles_expanded, closest_points, min_distances, True, (current_simulation_time + time_step), 
                                            f"Particle diffusion inside a {shapeDict[shape]} \n Elapsed time: {current_simulation_time + time_step:.3f} s")
# plt.close("all")
print(data_list) 
