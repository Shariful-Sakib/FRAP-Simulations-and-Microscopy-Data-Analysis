# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 13:22:21 2026

@author: sakib
"""

import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit;
import tkinter
from tkinter import filedialog
import ctypes
import sys
import os
from decimal import Decimal
import re
import statistics as stat
import gc
import psutil 



r = 0.27
# fractional_sections = np.linspace(0, 1, 10 + 1)

p1 = np.array([r,-1,0])
p2 = np.array([1,-1,0])

particles = np.random.uniform((p1[0],-5,-5),(p2[0],5,5),[10000,3])
print(particles)
L_P = np.sqrt(np.sum((p2 - p1) ** 2)) 

L_T = L_P + 2*r
resolution = 0.085
segments = Decimal(str(L_P)) // Decimal(str(resolution))
f= np.reshape(np.arange(0, L_P, resolution) / L_P,(-1,1))
intervals = (L_T-2*r) * (f*L_T+r-2*r*f)
# j = float(Decimal(str(intervals[1][0]))-Decimal(str(intervals[0][0])))
# q= intervals + j
calcs = (p2[0] - p1[0]) * particles[:, 0] + (p2[1] - p1[1]) * particles[:, 1] + (p2[2] - p1[2]) * particles[:, 2]
g = (p2[0] - p1[0]) * particles[:, 0] + (p2[1] - p1[1]) * particles[:, 1] + (p2[2] - p1[2]) * particles[:, 2] >= (L_T-2*r) * (f*L_T+r-2*r*f)
h = (p2[0] - p1[0]) * particles[:, 0] + (p2[1] - p1[1]) * particles[:, 1] + (p2[2] - p1[2]) * particles[:, 2] < (L_T-2*r) * (f*L_T+r-2*r*f) + float(Decimal(str(intervals[1][0]))-Decimal(str(intervals[0][0])))

profiles = np.count_nonzero(((p2[0] - p1[0]) * particles[:, 0] + (p2[1] - p1[1]) * particles[:, 1] + (p2[2] - p1[2]) * particles[:, 2] >= intervals)
                            & ((p2[0] - p1[0]) * particles[:, 0] + (p2[1] - p1[1]) * particles[:, 1] + (p2[2] - p1[2]) * particles[:, 2] < intervals + float(Decimal(str(intervals[1][0]))-Decimal(str(intervals[0][0])))), axis=1)
b = np.logical_and(((L_T- 2*r) * particles[:, 0] + (p2[1] - p1[1]) * particles[:, 1] + (p2[2] - p1[2]) * particles[:, 2] >= intervals),
    ((L_T- 2*r) * particles[:, 0] + (p2[1] - p1[1]) * particles[:, 1] + (p2[2] - p1[2]) * particles[:, 2] < intervals + float(Decimal(str(intervals[1][0]))-Decimal(str(intervals[0][0])))))

y = np.arange(r, L_P + r, resolution)
print(np.arange(0.27, 0.26))
print(b.shape[0])
colors = plt.cm.tab10.colors  # tuple of distinct colors

fig = plt.figure(figsize=(16, 9)) # Set up the figure and 3D axis
ax_3D = fig.add_subplot(1, 1, 1, projection='3d')
for region in range(b.shape[0]):
    # idx = b[region,:]
    ax_3D.scatter(particles[b[region,:], 0], particles[b[region,:], 1], particles[b[region,:], 2], color=colors[region % len(colors)], label=f'region{region}', s=5) 

# print(np.reshape(b[7,:], (-1,1)))
# print(particles[b[17,:], 0])
# print(b.shape[0])
# print(particles)
# print(L_P)
# sectioned_line = p1+(p2-p1)*fractional_sections[:,None]

# ax_3D.scatter(sectioned_line[:,0], sectioned_line[:,1], sectioned_line[:,2], color='blue', label='Discretized line', lw=0.5) 
###########
# point  = np.array([1, 2, 3])
# normal = np.array([1, 1, 2])

# point2 = np.array([10, 50, 50])

# # a plane is a*x+b*y+c*z+d=0
# # [a,b,c] is the normal. Thus, we have to calculate
# # d and we're set
# d = -point.dot(normal)

# # create x,y
# xx, yy = np.meshgrid(range(5), range(5))

# # calculate corresponding z
# z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

# # plot the surface
# fig = plt.figure(figsize=(16, 9)) # Set up the figure and 3D axis
# ax_3D = fig.add_subplot(1, 1, 1, projection='3d')
# ax_3D.plot_surface(xx, yy, z, alpha=0.2)



plt.show()