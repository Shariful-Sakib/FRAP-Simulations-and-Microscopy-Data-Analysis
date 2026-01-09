# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:22:10 2024
This program analyzes videos from photobleaching experiments.
Python operates with ImageJ to automate FRAP microscopy data processing
and data analysis
Last revised: Oct 6 2025
@author: Shariful Sakib
"""

import imagej # remember to run the following in a command prompt before starting: conda activate pyimagej
import sys
import easygui
from collections import defaultdict
import glob
import os.path
import re
import pandas as pd
import numpy as np
import math
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
import gc

def FRAP_Model(t, A, C, 𝜏):
   return (A * (1 - C * np.exp(-t / 𝜏)))

def FLAP_Model(t, B, A, 𝜏2):
   return (B * np.exp(-t / 𝜏2) + A)

def COSINE_COLLAPSE_Model(x, I0, I_amplitude):
    return (I0 + (I_amplitude * np.cos(np.pi * x / cell_Length)))

def MASTER_CURVE(cellType, analysisType, cell_Length, internal_radius, 𝜏):
    a_Dict = {0: 2.221, 2: 5.264}###Dictionary keys correspond to analysis type. 0 = bleach region analysis, 2 = Fourier Transform
    key = int(analysisType)
    aspect_Ratio = cell_Length / (internal_radius * 2)

    if cellType == 1:
        cell_amplitude = 0
        cell_pitch = np.inf
    else:
        cell_amplitude = 0.2
        cell_pitch = 2.5

    D =  (0.0575 + ((1/np.pi**2)*(1 + (2 * math.pi * cell_amplitude / cell_pitch)**2) - 0.0575) * (1 - np.exp((1 - aspect_Ratio) / a_Dict[key]))) * (cell_Length**2 / 𝜏)
    return D
# B1(lc/L)^B2
def extract_float(fileName):
    match = re.search(r' mNG-(\d+)', fileName)
    return float(match.group(1))   

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
BLEACH_REGION_ANALYSIS = True #User can toggle if they want to analyze the fluorescence in the bleach and non-bleach region
PROFILE_ANALYSIS = True #User can can toggle if they want to analyze the fluorescence profile 
matplotlib.use('Agg')  

option = easygui.ynbox(msg='Please select the folder with the FRAP data in the next pop-up window', title='FRAP Analyzer - User Input Required', choices=('Yes', 'No'), image=None, default_choice='Yes', cancel_choice='No')
if option == False:
    sys.exit() #Stops the operation if user cancels

directory = easygui.diropenbox("Please select the folder containing your FRAP data", "Select FRAP Data Directory")

if not directory:
    easygui.msgbox("User did not select a folder...Exiting", "Cancelled")
    sys.exit()
    
# if directory == '/':
#     sys.exit(); #Stops the operation if user cancels 
    
ij = imagej.init('sc.fiji:fiji:2.14.0' , mode=imagej.Mode.INTERACTIVE) #Initializes a specifc version of imageJ for consistent data analysis
ij.log().setLevel(0)
ij.ui().showUI()

ImageJ_FRAP_Macro = """
//This macro is sent to ImageJ for execution
//This program searches for and opens files from photobleaching experiments.
//Depending on the user input, it extracts the avg fluorescence vs time data for the bleach and non-bleach regions, as well as the fluorescence profile data.
//Figures of the timepoint and regions of interest are always created.
//Relevant variables are passed to Python and any data is outputted as .csv files for further analysis in Python.
//Programmer: Shariful Sakib
//Date created: April 3, 2024
//Last revised: March 19, 2025

//Following output variables will be sent back to python for processing

#@output Integer bleachTimeIndexOutput
#@output Float bleachDurationOutput
#@output Float frameTimeOutput
#@output Float cellLengthOutput
#@output String saveLocationOutput

#@ String directory //Receives the files variable from python 
#@ Integer ID //Receives the fileCount variable from python to act as an ID variable for data organization purposes
#@ Boolean BLEACH_REGION_ANALYSIS //Receives the boolean variable from python indicating whether the user wants this data to be extracted
#@ Boolean PROFILE_ANALYSIS //Receives the boolean variable from python indicating whether the user wants this data to be extracted

run("Input/Output...", "jpeg=100 gif=-1 file=.csv use_file copy_column save_column"); //edit output window
run("Set Measurements...", "area mean centroid feret's integrated redirect=None decimal=4");
//fileType = ".czi"; //processes .czi (Zeiss) by default. Can be modified to process other file types

outputExperimentData = FRAP_DATA_EXTRACTION(directory);

bleachTimeIndexOutput = outputExperimentData[1];
bleachDurationOutput = outputExperimentData[2];
frameTimeOutput = outputExperimentData[3];
cellLengthOutput = outputExperimentData[11];
saveLocationOutput = outputExperimentData[12];

//////Closing procedures	
selectWindow("ROI Manager");
run("Close");
close("*");
close("CP");
close("Threshold");
selectWindow("Results");
run("Close");
setBatchMode(false);
//////^^^Closing procedures

function FRAP_DATA_EXTRACTION(directory)
{
    saveLocation = File.getParent(directory) + "/" + File.getNameWithoutExtension(directory) + "_MacroGeneratedData";
    File.makeDirectory(saveLocation);
    run("Bio-Formats Importer", "open=["+directory+"] autoscale color_mode=Colorized display_rois rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
    run("Maximize");
    roiManager("Show None");
    
    //////Extract relevant experiment data
    metaData = split(getImageInfo(), "\\n");
    bleachBoxIndex = Array.filter(metaData, "Information|TimelineTrack|TimelineElement|EventInformation|Bleaching|BleachRegion|Name"); //subtract 1 from bleach region to get ROI index
    bleachTimeIndex = Array.filter(metaData, "Experiment|AcquisitionBlock|BleachingSetup|StartIndex"); //gives time index of when bleach occured
    bleachDuration = Array.filter(metaData, "Information|TimelineTrack|TimelineElement|Duration"); //gives the length of bleaching in seconds
    frameTime = Array.filter(metaData, "Information|Image|T|Interval|Increment"); //gives the time between frames in seconds
    laserWavelength = Array.filter(metaData, "Information|Image|Channel|Wavelength"); //gives laser wavelength
    bleachLaserPower = Array.filter(metaData, "Experiment|AcquisitionBlock|BleachingSetup|IntensityThreshold"); //retrieves in percentage of laser power
    acquisitionLaserPower = Array.filter(metaData, "Experiment|AcquisitionBlock|TimeSeriesSetup|MultiTrackSetup|Track|DetectionModeSetup|Zeiss.Micro.LSM.Acquisition.Lsm880ChannelTrackDetectionMode|ParameterCollection|Intensity #03"); //retrieves acquisition laser power
    detectorGain = Array.filter(metaData, "HardwareSetting|ParameterCollection|DetectorGain #7"); //retrieves gain for detector
    experimentTime = Array.filter(metaData, "Information|Image|AcquisitionDateAndTime");  //time image was taken
    bleachBoxRotation = Array.filter(metaData, "Experiment|AcquisitionBlock|ExperimentRegionsSetup|RegionItem|Rotation #1"); //The original rotation of the bleach box during the experiment
    width = Array.filter(metaData, "Width:");
    height = Array.filter(metaData, "Height:");
    imageDimensions =  "W x H:" + substring(width[0], indexOf(width[0], " ") + 1) + " x" + substring(height[0], indexOf(height[0], " ") + 1);
    imageDimensions = replace(replace(imageDimensions, "microns", "µm"), ")", " px)");
    
    experimentData = processMetaData(Array.concat(bleachBoxIndex,bleachTimeIndex,bleachDuration,frameTime,laserWavelength,bleachLaserPower,acquisitionLaserPower,detectorGain,experimentTime,bleachBoxRotation)); //relevant metadata retrieved and processed in this order
    experimentData = Array.concat(experimentData, imageDimensions);
    
	function processMetaData(data) 
	{ 
       	for (dataCount = 0; dataCount < data.length; dataCount++) 
	    {
   			tempString = split(data[dataCount], "= "); //ImageJ macros can't handle 2D arrays so a temporary string is used
			data[dataCount] = tempString[tempString.length - 1];
		}
		return data;
	}
	//////^^^Extract relevant experiment data
               
    //////Threshold and segment into regions
	bleachBoxROIindex = parseInt(experimentData[0]) - 1;
	bleachIndex = parseInt(experimentData[1]);
	bleachSec = parseFloat(d2s(experimentData[2], 4));
	frameSec = parseFloat(d2s(experimentData[3], 4));			
	
	run("Select All");
	run("Plot Z-axis Profile"); //extracts time data from the Z-axis profile
	Plot.getValues(timePoints, placeHolder); //extracts time data from the Z-axis profile
	close;
	roiManager("Select", bleachBoxROIindex);
	RoiManager.rotate(parseFloat(experimentData[9]));
	//roiManager("Deselect");
    //run("Select None");
    
    run("Duplicate...", "duplicate");
    
	if (matches(directory, ".*coli.*")) 
	{setMinAndMax(1, 150);}
	else if (matches(directory, ".*AMB.*"))
	{setMinAndMax(1, 6);}    
    
    run("Gaussian Blur...", "sigma=2 stack");
	run("Z Project...", "stop="+bleachIndex+" projection=[Average Intensity]");
	
	run("Threshold...");
	setAutoThreshold("Yen dark no-reset");
	run("Convert to Mask");
 	run("Create Selection");
	roiManager("Add");
	roiManager("Select", newArray(bleachBoxROIindex, roiManager("count") - 1));
	roiManager("AND");
	roiManager("Add");
	run("Measure");
   
	centroid_X = Table.get("X", 0);
	centroid_Y = Table.get("Y", 0);
	toUnscaled(centroid_X, centroid_Y);
	roiManager("Deselect");
	roiManager("Select", roiManager("count") - 2);
	roiManager("Delete");
	doWand(centroid_X, centroid_Y);
	roiManager("Add");
	run("Make Inverse");
	run("Color Picker...");
	setForegroundColor(0, 0, 0);
	run("Fill", "slice");
	run("Make Inverse");
	roiManager("Select", newArray(roiManager("count") - 2, roiManager("count") - 1));
	roiManager("XOR");
	roiManager("Add");				
	roiManager("Select", roiManager("count") - 2);
	run("Fit Rectangle");
	roiManager("Add");
	run("Clear Results");
	run("Measure");
	cellLength = Table.get("RRLength", 0); //Gets the length of the cell from endcap to endcap from the length of the rectangle fit
	experimentData = Array.concat(experimentData, cellLength);
	labels = newArray("Bleach box index:", "Bleach time-point index:", "Bleach duration (s):", "Frame time (s):",
					"Laser wavelength (nm):", "Laser power during bleach (%):", "Laser power during acquisition (%):",
					"Detector gain (V):", "Date and time:", "Rotation of bleach box (deg):", "Image dimensions:", "Cell length (um):");
	run("Clear Results");
	Table.create("Results");
	Table.setColumn("Parameters", labels);
	Table.setColumn("Experiment Data", experimentData);
	saveAs("Results", saveLocation + "/" + File.getNameWithoutExtension(directory) + "_ExperimentData ID = "+ID+".csv"); //Save Experiment data immediately after thresholding
 	//run("Select None");
	//////^^^Threshold and segment into regions
	
	//////Capturing Images
	ImageList = getList("image.titles");
	selectWindow(ImageList[0]);								
		
    if (matches(directory, ".*coli.*")) 
	{setMinAndMax(1, 150);}
	else if (matches(directory, ".*AMB.*"))
	{setMinAndMax(1, 6);} 
	
	run("Scale Bar...", "width=2 height=3 thickness=2 font=8 bold overlay label");
	setSlice(bleachIndex);
	roiManager("Select", bleachBoxROIindex);
	run("Add Selection...");
	run("Capture Image");
	saveAs("PNG", saveLocation + "/" + File.getNameWithoutExtension(directory) + "_1.ImmediatelyBeforeBleach.png");
	run("Close");
	run("Remove Overlay");
	run("Scale Bar...", "width=2 height=3 thickness=2 font=8 bold overlay label");
	roiManager("Select", roiManager("count") - 1);
	Roi.setStrokeColor(148,203,236)
	roiManager("Set Line Width", 1);
	run("Add Selection...");
	run("Capture Image");
	saveAs("PNG", saveLocation + "/" + File.getNameWithoutExtension(directory) + "_5.ProfileBox.png");
	run("Close");
	setSlice(bleachIndex + 1);
	roiManager("Select", bleachBoxROIindex);
	run("Add Selection...");
	run("Capture Image");
	saveAs("PNG", saveLocation + "/" + File.getNameWithoutExtension(directory) + "_2.ImmediatelyAfterBleachDuration of ("+ bleachSec * 1000 +" ms).png");
	run("Close");
	run("Remove Overlay");
	run("Scale Bar...", "width=2 height=3 thickness=2 font=8 bold overlay label");
	roiManager("Select", roiManager("count") - 2);
	roiManager("Set Color", "#ff00ff");
	roiManager("Set Line Width", 2);
	run("Add Selection...");
	roiManager("Select", roiManager("count") - 4);
	Roi.setStrokeColor("#FB9364")
	roiManager("Set Line Width", 2);
	run("Add Selection...");
	run("Capture Image");
	saveAs("PNG", saveLocation + "/" + File.getNameWithoutExtension(directory) + "_4.BleachRegionsImmediatelyAfterBleachDuration of ("+ bleachSec * 1000 +" ms).png");
	run("Close");
	setSlice(bleachIndex + 11);
	roiManager("Select", bleachBoxROIindex);
	run("Add Selection...");
	run("Capture Image");
	saveAs("PNG", saveLocation + "/" + File.getNameWithoutExtension(directory) + "_3.LongAfterBleach ("+frameSec * 1000 * 10+" ms).png");
	run("Close");
	run("Remove Overlay");
	resetMinAndMax();
	//////^^^Capturing Images
	
	setBatchMode(true);
	//////Bleach and Non-Bleach Region Fluorescence Data Extraction
	if (BLEACH_REGION_ANALYSIS == true) 
	{
		run("Clear Results");
		for (roiCount = 4; roiCount >= 2; roiCount-= 2) 
	   	{
	   		roiManager("Select", roiManager("count") - roiCount);
	   		run("Plot Z-axis Profile");
	   		Plot.getValues(time, avgIntensity);
	   		close;
	   		run("Clear Results");
	   		Table.create("Results");
	   		Table.setColumn("Time [s]", time);
	   		Table.setColumn("Average Intensity [Arbitrary Units]", avgIntensity);
	   		if (roiCount == 4)
	   		{saveAs("Results", saveLocation + "/" + File.getNameWithoutExtension(directory) + "_BleachRegionMacroData.csv");}
	   		else
	   		{saveAs("Results", saveLocation + "/" + File.getNameWithoutExtension(directory) + "_Non-BleachRegionMacroData.csv");}
	   	}
	}
	//////^^^Bleach and Non-Bleach Region Fluorescence Data Extraction
	
	//////Fluorescence Profile Data Extraction
	if (PROFILE_ANALYSIS == true) 
	{
		run("Clear Results");
		for (sliceCount = 1; sliceCount <= nSlices(); sliceCount++)
		{
		   	setSlice(sliceCount);
		   	roiManager("Select", roiManager("count") - 1);
			run("Plot Profile");
			Plot.getValues(positionAlongCell, avgIntensity);
			close;
			if (sliceCount == 1) 
			{
				Table.create("Results");
				Table.setColumn("Distance [microns]", positionAlongCell);
			}
			Table.setColumn("Average Intensity ["+timePoints[sliceCount - 1]+" s]", avgIntensity);
		}
		saveAs("Results", saveLocation + "/" + File.getNameWithoutExtension(directory) + "_ProfileMacroData.csv");
	}
	//////^^^Fluorescence Profile Data Extraction
    experimentData = Array.concat(experimentData, saveLocation);               
    return experimentData;
}

"""

extensionList = ["czi"] #Zeiss files are accepted by default, users can add more when prompted
colorMap = LinearSegmentedColormap.from_list('Black', ['red','aqua','black'])
cell_Length = 0
fileCount = 0
for folders in glob.iglob(directory, recursive= True):
    for extensions in extensionList:
        sorted_files = sorted(sorted(glob.iglob(folders + f'/**/*.{extensions}', recursive= True), key=extract_float))
        #fileCount = 98 #for skipping
        for files in (sorted_files):
            if len(os.listdir(files)) == False:
                fileCount += 1
                continue #if a folder is empty, skip it
            print(files)
            dataDict = defaultdict(list); #makes a dictionary for metadata values
            macro_arguments = {"directory": files, "ID": fileCount, "BLEACH_REGION_ANALYSIS": BLEACH_REGION_ANALYSIS, "PROFILE_ANALYSIS": PROFILE_ANALYSIS} #dictionary of arguments sent to ImageJ macro.
            FRAPMacro = ij.py.run_macro(ImageJ_FRAP_Macro, macro_arguments)
            dataDict["bleachTimeIndex"].append(int(FRAPMacro.getOutput('bleachTimeIndexOutput')))
            dataDict["bleachDuration"].append(float(FRAPMacro.getOutput('bleachDurationOutput')))
            dataDict["frameTime"].append(float(FRAPMacro.getOutput('frameTimeOutput')))
            dataDict["cellLength"].append(float(FRAPMacro.getOutput('cellLengthOutput')))
            dataDict["saveLocation"].append(str(FRAPMacro.getOutput('saveLocationOutput')))
            
            if "coli" in os.path.basename(dataDict['saveLocation'][0]):
                cellType = 1
                internal_radius = 0.3
            elif "AMB" in os.path.basename(dataDict['saveLocation'][0]):
                cellType = 2
                internal_radius = 0.27
            cell_Length = dataDict['cellLength'][0] # Not to be confused with contour length
            
            bleachRegionData = pd.read_csv(glob.glob(dataDict["saveLocation"][0] + '/*BleachRegionMacroData.csv', recursive= False)[0])
            nonbleachRegionData = pd.read_csv(glob.glob(dataDict["saveLocation"][0] + '/*Non-BleachRegionMacroData.csv', recursive= False)[0])
            profileData = pd.read_csv(glob.glob(dataDict["saveLocation"][0] + '/*ProfileMacroData.csv', recursive= False)[0])
            preBleachImage = np.asarray(Image.open(glob.glob(dataDict["saveLocation"][0] + '/*ImmediatelyBefore*.png', recursive= False)[0]))
            postBleachImage = np.asarray(Image.open(glob.glob(dataDict["saveLocation"][0] + '/*ImmediatelyAfter*.png', recursive= False)[0]))
            longAfterBleachImage = np.asarray(Image.open(glob.glob(dataDict["saveLocation"][0] + '/*LongAfter*.png', recursive= False)[0]))
            bleachRegionsImage = np.asarray(Image.open(glob.glob(dataDict["saveLocation"][0] + '/*BleachRegions*.png', recursive= False)[0]))
            profileBoxImage = np.asarray(Image.open(glob.glob(dataDict["saveLocation"][0] + '/*ProfileBox*.png', recursive= False)[0]))
            
            processedBleachRegionData = pd.DataFrame()
            processedBleachRegionData["Time [s]"] = bleachRegionData.iloc[:,0]
            processedBleachRegionData["Average Intensity Difference [Arbitrary Units]"] = (nonbleachRegionData.iloc[:, 1] - bleachRegionData.iloc[:,1])
            processedProfileData = profileData.iloc[np.argmax(profileData.iloc[:,0] >= internal_radius) : np.argmax(profileData.iloc[:,0] >= dataDict["cellLength"][0] - internal_radius) + 1,:]
            
            fig = plt.figure(figsize=(16, 9)) # Set up the figure and 3D axis
            ax_2D_pre = fig.add_subplot(2, 3, 1)
            ax_2D_post = fig.add_subplot(2, 3, 2)
            ax_2D_long = fig.add_subplot(2, 3, 3)
                        
            ax_FRAP = fig.add_subplot(2, 3, 4)
            ax_FRAP.set_title("FRAP/FLIP in bleach and non-bleach region")
            ax_INTENSITY_PROFILE = fig.add_subplot(2, 3, 5)
            ax_INTENSITY_PROFILE.set_title("Intensity Profiles")
            ax_COSINE_AMPLITUDES = fig.add_subplot(2, 3, 6)
            ax_COSINE_AMPLITUDES.set_title("Cosine Amplitudes vs Time")
            ax_COSINE_AMPLITUDES.plot([0,1e-10],[0,0], color='red', linestyle='-', linewidth=1, label='Red line fit')
            ax_2D_regions = fig.add_axes([0.2185, 0.351, 0.12, 0.12])
            ax_2D_box = fig.add_axes([0.8865, 0.351, 0.12, 0.12])
            
            plt.subplots_adjust(bottom=0.07, left=0.04, right=0.98, top=0.95, hspace=0.19, wspace= 0.23)
            ax_black_background = fig.add_axes([0, 0.5, 1, 0.5], facecolor='black', zorder=-1)
            ax_black_background.set_xticks([])
            ax_black_background.set_yticks([])
            ax_black_background.set_xticklabels([])
            ax_black_background.set_yticklabels([])
            ax_2D_pre.set_axis_off()
            ax_2D_post.set_axis_off()
            ax_2D_long.set_axis_off()
            ax_2D_regions.set_axis_off()
            ax_2D_box.set_axis_off()
            ax_2D_pre.imshow(preBleachImage)
            ax_2D_post.imshow(postBleachImage)
            ax_2D_long.imshow(longAfterBleachImage)
            ax_2D_regions.imshow(bleachRegionsImage)
            ax_2D_box.imshow(profileBoxImage)
            
            ax_FRAP.scatter(bleachRegionData.iloc[:,0], bleachRegionData.iloc[:,1], color='#FB9364', alpha = 0.9,
                            s=40, label="Fluorescence within bleach region", edgecolors='black')
            ax_FRAP.scatter(nonbleachRegionData.iloc[:, 0], nonbleachRegionData.iloc[:, 1], color='#ff00ff', alpha = 0.9,
                            s=40, label="Fluorescence within non-bleach region", edgecolors='black')
            ax_FRAP.scatter(processedBleachRegionData.iloc[:, 0], processedBleachRegionData.iloc[:, 1], color='lime', alpha = 0.9,
                            s=40, label="Fluorescence within non-bleach region", edgecolors='black')
            ax_FRAP.axvline(x = bleachRegionData.iloc[dataDict["bleachTimeIndex"],0].item(), color = 'dodgerblue', label = f'Photobleaching at {round(bleachRegionData.iloc[dataDict["bleachTimeIndex"],0], 4)} s', linestyle= 'dashed')
            ax_COSINE_AMPLITUDES.axvline(x = bleachRegionData.iloc[dataDict["bleachTimeIndex"],0].item(), color = 'dodgerblue', label = f'Photobleaching at {round(bleachRegionData.iloc[dataDict["bleachTimeIndex"],0], 4)} s', linestyle= 'dashed')
            
            processedProfileData.plot(x = 0, y = np.arange(1, processedProfileData.shape[1]),
                                      alpha=0.75, cmap = colorMap, linestyle='-', linewidth=0.5, ax= ax_INTENSITY_PROFILE)
            # print(processedProfileData.mean(axis=1))
            ax_INTENSITY_PROFILE.get_legend().remove()

            graphDict = {0: ax_FRAP, 1: ax_INTENSITY_PROFILE, 2: ax_COSINE_AMPLITUDES}  #note 3 and 2 are swapped in the non-animated version
            modelDict = {0: FLAP_Model, 1: COSINE_COLLAPSE_Model, 2: FRAP_Model}          #note 3 and 2 are swapped in the non-animated version
            varDict = {0: processedBleachRegionData}
            
            amplitude_fits = []
            amplitudeData = pd.DataFrame()
            data_list = ([],[])
            for plotCount in range(3):
                graphData = pd.DataFrame()
                if plotCount == 1:                   
                    try:

                        for profileCount in range(1, processedProfileData.shape[1]):
                            optimalValues, covarianceMatrix = curve_fit(modelDict[plotCount], processedProfileData.iloc[:,0], processedProfileData.iloc[:, profileCount], p0=[0.0, 0.0], maxfev=10000)
                            c1_opt, I_amplitude_opt = optimalValues
                            amplitude_fits.append(I_amplitude_opt)
                            x_fit = np.linspace(processedProfileData.iloc[0,0], processedProfileData.iloc[-1,0] , 100)
                            y_fit = modelDict[plotCount](x_fit, c1_opt, I_amplitude_opt)
                            graphDict[plotCount].plot(x_fit, y_fit, color=(colorMap(profileCount/processedProfileData.shape[1])))
                            ax_COSINE_AMPLITUDES.scatter(bleachRegionData.iloc[:, 0][profileCount - 1], amplitude_fits[-1], color= colorMap(profileCount/processedProfileData.shape[1]),
                                            alpha=0.50, s=40, label="Cosine amplitudes of fluorescence", edgecolors='black')
                        
                        amplitudeData['Time (s)'] = bleachRegionData.iloc[:, 0]
                        amplitudeData['Amplitudes'] = amplitude_fits                        
                        varDict[2] = amplitudeData

                    except:
                        pass
                else:

                    try:
                        for testTau in (np.arange(0.1, 0.51, 0.1)):
                            optimalValues, covarianceMatrix = curve_fit(modelDict[plotCount], varDict[plotCount].iloc[dataDict['bleachTimeIndex'][0]:,0],
                                                                        varDict[plotCount].iloc[dataDict['bleachTimeIndex'][0]:,1],
                                                                        p0=[0.50, 0.50, testTau], maxfev=10000)
                            c1_opt, c2_opt, 𝜏_opt = optimalValues
                            if 𝜏_opt >= dataDict['frameTime'][0]:
                                break
                                
                        diffusion_Coefficient = MASTER_CURVE(cellType, plotCount, cell_Length, internal_radius, 𝜏_opt)
                        data_list[0].append(𝜏_opt)
                        data_list[1].append(diffusion_Coefficient)
                        x_fit = np.linspace(varDict[plotCount].iloc[dataDict['bleachTimeIndex'][0], 0], varDict[plotCount].iloc[-1, 0], 100)
                        y_fit = modelDict[plotCount](x_fit, c1_opt, c2_opt, 𝜏_opt)                        
                        graphData["x"] = x_fit
                        graphData["y"] = y_fit
                        graphData.plot(x = 0, y = 1, color = 'r', ax= graphDict[plotCount])

                    except:
                        pass
            
            ax_FRAP.set_xlabel('Time (s)')
            ax_FRAP.set_ylabel('Avg Fluorescence Intensity (Arbitrary units)')
            ax_INTENSITY_PROFILE.set_xlabel('Position along cell without endcaps (µm)')
            ax_INTENSITY_PROFILE.set_ylabel('Avg Fluorescence Intensity (Arbitrary units)')
            ax_COSINE_AMPLITUDES.set_xlabel('Time (s)')
            ax_COSINE_AMPLITUDES.set_ylabel('Fluorescence amplitudes of cosine fits (Arbitrary units)')
            ax_COSINE_AMPLITUDES.set_ylim(abs(amplitudeData.iloc[dataDict['bleachTimeIndex'][0], 1]) * -1, abs(amplitudeData.iloc[dataDict['bleachTimeIndex'][0], 1]))
          
            fig.savefig(dataDict["saveLocation"][0] + "\\" + os.path.basename(dataDict['saveLocation'][0]).replace("MacroGeneratedData", f"MontageFigure ID = {fileCount}.png"), dpi = 600, bbox_inches = "tight") #saves as png
            fig.savefig(dataDict["saveLocation"][0] + "\\" + os.path.basename(dataDict['saveLocation'][0]).replace("MacroGeneratedData", f"MontageFigure ID = {fileCount}.svg"), dpi = 600, bbox_inches = "tight") #saves as svg
            plt.close(fig)
            processedBleachRegionData.to_csv(dataDict['saveLocation'][0] + "\\" + os.path.basename(dataDict['saveLocation'][0]).replace("MacroGeneratedData", f"ProcessedBleachRegionData(I2-I1) ID = {fileCount}.csv") , index= False)
            amplitudeData.to_csv(dataDict['saveLocation'][0] + "\\" + os.path.basename(dataDict['saveLocation'][0]).replace("MacroGeneratedData", f"ProcessedIntensityAmplitudeData ID = {fileCount}.csv") , index= False)
            processedProfileData.to_csv(dataDict['saveLocation'][0] + "\\" + os.path.basename(dataDict['saveLocation'][0]).replace("MacroGeneratedData", f"ProcessedProfileData ID = {fileCount}.csv") , index= False)
            experimentData = pd.read_csv(glob.glob(dataDict["saveLocation"][0] + f'/*_ExperimentData ID = {fileCount}.csv', recursive= False)[0], encoding='unicode_escape')
            newEntry = pd.DataFrame([
                                    ['Aspect Ratio', cell_Length / (internal_radius * 2)],
                                    ['Tau_BleachRegion[I2-I1] (s)', data_list[0][0]],
                                    ['D_BleachRegion[I2-I1] (µm^2/s)', data_list[1][0]],
                                    ['Tau_Profile (s)', data_list[0][1]],
                                    ['D_Profile (µm^2/s)', data_list[1][1]],
                                    ],
                                    columns= ['Parameters', 'Experiment Data'])
            experimentData = pd.concat([experimentData, newEntry], ignore_index= True)
            experimentData.to_csv(glob.glob(dataDict["saveLocation"][0] + f'/*_ExperimentData ID = {fileCount}.csv', recursive= False)[0], index= False)
              
            fileCount += 1
            gc.collect()
            # sys.exit()

ij.dispose()
sys.exit()


