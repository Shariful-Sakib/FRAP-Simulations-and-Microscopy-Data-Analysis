# FRAP-Simulations-and-Data-Analysis
In this repository are Python programs that simulate Fluorescence Recovery After Photobleaching (FRAP) in spherical, cylindrical, and helical compartments with visuals and data analysis. 
  - Three versions of the simulation are given: standard, batch, and animated
    - The standard version performs one simulation per run. The batch version runs replicates of simulations for data collection. The animated version gives a visual version of the standard version.
    - The user can set the values for the different parameters related to shape creation, FRAP, and data acquisition.

- The compartment is defined by discretizing the shape into an array of 3D mid points. 
  - If the **shape** variable is set to 0 then a sphere will be produced, shape= 1 is a cylinder/rod, and shape= 2 is a helical cell. Shape 1 defaults are preset for _E. coli_ and shape 2 defaults are preset for _(Para)Magnetospirillum magneticum_, also known as AMB-1. The program will dynamically use the relevant variables to make the desired shape.
  - The **length** of the compartment, **L**, can be any value greater than or equal to 0
  - The helical **amplitude** and helical **pitch** can be any value greater than or equal to 0
    - If the **shape** is set to 1 for rods, then the **amplitude** is set to 0 and **pitch** is set to infinity 
  - The spacing, S, between each mid point is calculated by determining the maximum tolerance on the error for any point particle such that the **max_segment_error**, **E** = 4.2 nm (approximate length of a GFP molecule)
    - S = 2*sqrt((E+r)^2 - r^2)
      - This works well for very small S
  - The parametric equations for a helix is used to generate all the shapes where X(t) = t, Y(t) = A cos(2πt/λ), and Z(t) = A sin(2πt/λ). The number of values for t (**contour_length** / S) are equally spaced from 0 to **L**

- Point particles representing fluorophores are spawned randomly within the compartment within the **internal_radius** around the mid points 
  - The particles diffuse every timestep with the mean displacement centered at 0 microns and the standard deviation equal to sqrt(2Dt), where D is the **diffusion_constant** (typically 15 um^2/s) and t is the **time_step** (1 ms)
    - There can be a hetergenous mix of immobile, slow, and fast particles if the user wishes
    - The particles are allowed to diffuse for 5 **frame_interval**s or 150 **time_step**s
  - The user can set when the photobleaching starts and ends. The intensity of the bleach can tweaked by modifying the **gamma** value. This influences the probability of a fluorphore photobleaching (being deleted from the array) within the **bleach_region** for every **time_step**. Whether half the cell is bleached or if a smaller bleach region within the cell is bleached can be specified as well.

- During each **time_step**, the particles' distance from the compartment is calculated by finding the closest mid point and if the distance from the closest mid point exceeds the internal_radius, then a collision is detected
  - A collision is handled by calculating the following d′ = d − 2(n − r)(n_unit/n) such that the new position is calculated by taking the current uncorrected collision and subtracting 2 times the distance from the membrane (n-r) multiplied by the unit vector normal to the membrane at the collision point (n_unit/n).

- Microscopy data acquisition is simulated by analyzing the fluorescence (number of point particles) periodically with **frame_interval**s (typically 30 ms). Two methods are used to analyze the fluorescence:
  - Bleach region analysis where the count of fluorophores in the bleached and unbleached regions are tracked every frame interval. At the end of the simulation, the characteristic recovery time is calculated by fitting the fluorescence over time data with an exponential decay function
  - Fourier profile analysis where the count of fluorophores along columns of the compartment (along the x-axis for convenience in this simulation) sectioned by a resolution of 85 nm is observed every frame interval. The value is then fit with a cosine function i(x, t) = a_1(t) cos(πx/L), then at the end of the simulation, the amplitudes of that first Fourier mode, a_1(t), is then fit to an exponential function, a_{1,0}e^{−t/τ} + a_{1,i}  

# Automated Zeiss microscopy image processing and data analysis using PyImageJ 
Also included in the repository is a script that uses PyImageJ for automating Zeiss microscopy image processing and automated data analysis using PyImageJ. The functions and analysis used in the script can be adapted for any microscope or software
- The image is opened in Fiji, the metadata is extracted to retrieve the information on the photobleaching time index, photobleaching duration, and frame_interval. This metadata is sent to python.
- Any photobleaching region of interest (ROI 1) that was defined by the user using a compatible software during the microscopy is accessed by the bioformats plugin in ImageJ.
- To correctly define the regions of interest for the analysis, the time series is duplicated, then flattened by taking the average intensity of the whole stack and applying a gaussian blur (STD = 2 pixels). The image is then auto thresholded using the preset Yen algorithm (in a few cases the Triangle algorithm is more optimal) to define the whole cell body (ROI 2). The Fit Rectangle funciton is used on ROI 2 to get a rectangle that encapsulates the entire cell body (ROI 3). Logical operators are used to get the bleach region ROI 4 (ROI 1 && ROI 2) and the non-bleach region ROI 5 (ROI 4 XOR ROI 2). The plot profile function is used per time slice along the long axis of ROI 3 to get the data needed for the Fourier profile analysis. The fluorescence intensity data in ROI 4 and ROI 5 are collected using the Plot Z-Axis Profile function to collect the data for the bleach region analysis. 
- The fluorescence data is then saved as a .csv for access by Python for data visualization and analysis as described above for the simulated FRAP.


Author: Shariful Sakib
Reviewer: Cécile Fradin
