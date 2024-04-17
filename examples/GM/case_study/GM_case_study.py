#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:13:49 2023

@author: bossema
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bossema
"""

import markers as M
import numpy as np
from collections import namedtuple
from flexdata import data
from flexdata import display
import trackpy as tp
import matplotlib.pyplot as plt
import matplotlib
import random
from pandas import read_csv
import pandas as pd
from pathlib import Path
from itertools import combinations

#%% Set directories and user input
#Data
path = '/export/scratch2/bossema/datasets/markers/GM/'
name = 'case_study/DICOMs/' 
folder_name = 'data'
save_path = '/export/scratch2/bossema/results/markers/GM/publication/case_study/'
file_name = 'IMG'
M.make_directory_structure(save_path)

bb = 1 #Binning the detector pixels
skip = 1 #Reducing number of projections

#Marker detection
minR = 5
maxR = 10
sigma = 4
invert = False

#Calibration
#n_markers = 20
n_markers_selected_1 = 15
n_markers_selected_2 = 8
distance = 9 #max distance between two markers
det_pixel_size = bb*0.2 #Detector pixels are 200 micron
SOD = 2500
ODD = 23.5

n_iterations_step1 = 100
n_iterations_step2 = 100
n_iterations_step3 = 100

n_rounds = 2
n_angles = 718//skip

#Inpainting
mag = 1.5

#Reconstruction
fdk = 1
sirt = 0
sirt_iter = 100

plot = True
server_run = False
y_cut = 10
x_cut = 0

upper_y = 600
lower_y = 1600
#%% Reading dark and flat data


dark, flat = M.Getty_flatdark(path, name, file_name, bb, skip, form = 'DICOM')
dark = dark[y_cut//bb:-(y_cut)//bb,:]
flat = flat[y_cut//bb:-(y_cut)//bb,:]

#%%
# Read data
frames = M.Getty_load(path+name, folder_name, file_name, n_angles, bb, skip, counterclockwise = False, invert = False)[:,y_cut//bb:-(y_cut)//bb,:]
print(frames.shape)
#%%
display.pyqt_graph(frames)


#%% Use this block to find the relevant parameters for marker detection
if server_run is True:
    matplotlib.use('Agg')
    
test_run = True
if test_run is True:
    #Marker detection

    minR = 5
    maxR = 10
    sigma = 4
    invert = False
    image1 = frames[175][:upper_y,]
    drange = image1.max() - image1.min()
    low_threshold = 0.1*drange
    high_threshold = 0.2*drange
    print(low_threshold, high_threshold)
    
    image2 = frames[175][lower_y:,]
    test_frames = [image1,image2]
    plt.imshow(image1)
    plt.imshow(test_frames[1])
    marker_blobdetect = M.markers_blobdetector(test_frames,  minR, maxR, low_threshold, high_threshold, sigma = sigma, invert = invert, save_path = None, test = True)

#%% Find all marker positions
test_run = False

drange = frames[0].max() - frames[0].min()
low_threshold = 0.1*drange
high_threshold = 0.2*drange
print(low_threshold, high_threshold)

frames_upper = frames[:,:upper_y,:]
frames_lower = frames[:,lower_y:,:]
del(frames)
#%%
markerpositions_upper, marker_projected_size_upper = M.markers_blobdetector(frames_upper, minR, maxR,low_threshold, high_threshold, sigma = sigma, invert = invert, test = test_run) 


markerpositions_lower, marker_projected_size_lower = M.markers_blobdetector(frames_lower, minR, maxR,low_threshold, high_threshold, sigma = sigma, invert = invert, test = False) 
markerpositions_lower['y'] = markerpositions_lower['y'] + lower_y

markerpositions = pd.concat([markerpositions_upper,markerpositions_lower])
markerpositions.to_csv(str(save_path+'markerpositions'))  

#%% Start here if markerpositions already found
markerpositions = read_csv(str(save_path + 'markerpositions'))

    
#%%
markerpositions_upper = markerpositions[markerpositions['y']<600]
markerpositions_lower = markerpositions[markerpositions['y']>1600]

   

#%% Find right parameters to get best trajectories upper part
x_lim = [0,dark.shape[1]]
y_lim = [0,upper_y]

allowed_shift = 150*skip//bb
memory = 100
min_traj = 50

trajectories_upper = M.find_trajectories(save_path, bb, skip, markerpositions_upper, allowed_shift = allowed_shift, memory = memory, min_traj = min_traj, y_lim = y_lim, x_lim = x_lim, save = False)

trajectories_upper = trajectories_upper[trajectories_upper['frame']<n_angles]
print('number of trajectories found:',len(set(trajectories_upper['particle'])))
trajectories_upper = M.rearrange_labels(trajectories_upper)
plot = tp.plot_traj(trajectories_upper,label = True)



#%% Find right parameters to get best trajectories lower part
x_lim = [0,dark.shape[1]]
y_lim = [lower_y,dark.shape[0]]

memory = 100
allowed_shift = 50*skip//bb #150
min_traj = 50
trajectories_lower = M.find_trajectories(save_path, bb, skip, markerpositions_lower, allowed_shift = allowed_shift, memory = memory, min_traj = min_traj, y_lim = y_lim, x_lim = x_lim, save = False)
trajectories_lower = trajectories_lower[trajectories_lower['frame']<n_angles]
trajectories_lower = M.rearrange_labels(trajectories_lower)
tp.plot_traj(trajectories_lower, label = True)



#%%
for i in np.arange(len(set(trajectories_lower['particle']))):
    plot = tp.plot_traj(trajectories_lower[trajectories_lower['particle'].isin([i])],label = True)

#%%

print('number of trajectories found:',len(set(trajectories_lower['particle'])))
trajectories_lower = M.rearrange_labels(trajectories_lower)
plot = tp.plot_traj(trajectories_lower,label = True)

#renumber the lower particles before merging
trajectories_lower['particle'] = trajectories_lower['particle'] + len(set(trajectories_upper['particle']))
#%%


#%%Merge the two trajectory frames and save
trajectories = pd.concat([trajectories_upper,trajectories_lower])
trajectories = M.rearrange_labels(trajectories)
plot = tp.plot_traj(trajectories,label = True)
n_markers = len(set(trajectories['particle']))
print('number of trajectories found:',len(set(trajectories['particle'])))
trajectories.to_csv(str(save_path+'trajectories'))
  
    
#%%
trajectories, n_markers = M.read_trajectories(str(save_path+'trajectories'), plot = True) 

#%% Calibration
Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi','det_eta', 'angles', 'markers' ])

# np.random.seed(1)
# rest = n_angles-(n_angles//n_rounds)*int(np.floor(n_rounds))
# full_rounds = np.tile(np.linspace(0,2*np.pi,n_angles//n_rounds, endpoint = False),int(np.floor(n_rounds)))
# partial_rounds = np.linspace(0,rest/n_angles*2*np.pi,rest, endpoint = False)

# angles_estimated = np.concatenate((full_rounds,partial_rounds))

# marker_locations =  (-2* np.random.rand(n_markers,3)+1)*150
# marker_locations = marker_locations[marker_locations[:, 2].argsort()][::-1]

# geom_parameters_estimated = Geom_param(source = np.array([0,-SOD, 0]), det = np.array([-0,ODD,0]), det_theta = np.radians(0.), det_phi = np.radians(0.),
#                                                det_eta = np.radians(0.), angles = angles_estimated,
#                                                markers = marker_locations)


# bounds = (np.concatenate((np.array([-50,ODD-200,-50]),np.array([-1/8*np.pi,-1/8*np.pi,-1/8*np.pi]),
#                            np.repeat(0,n_angles-1),np.repeat(-1000,n_markers*3))), 
#            np.concatenate((np.array([50,ODD+200,50]),np.array([1/8*np.pi,1/8*np.pi,1/8*np.pi]),
#                            np.repeat(n_rounds*2*np.pi+1,n_angles-1),np.repeat(1000,n_markers*3))))



det_size = np.array([dark.shape[1], dark.shape[0]])
found_parameters, trajectories_after, abs_error = M.calibration_parameters(trajectories, SOD, ODD, n_angles,
                                                            det_pixel_size, 
                                                            det_size, n_markers_selected_1, n_markers_selected_2,
                                                            distance, n_rounds, n_seeds = 10, radius = 150, plot = 1, 
                                                            show_all = 2, 
                                                            bounds_given = None,
                                                            geom_parameters_estimated = None,
                                                            n_iterations_step1 = n_iterations_step1,
                                                            n_iterations_step2 = n_iterations_step2)
                                              
save_path = '/export/scratch2/bossema/results/markers/Getty/publication/case_study_july/'

M.write_geom(save_path + 'geometry_%d_selected%d_selected2'%(n_markers_selected_1,n_markers_selected_2), found_parameters)




#%%
x_lim = [0,dark.shape[1]]
y_lim = [0,dark.shape[0]]
#calculate forward projected marker locations
trajectories_forward = M.plot_result_trajectories(save_path, found_parameters, trajectories, det_pixel_size, det_size, y_lim = y_lim, x_lim = x_lim)



    #markerpositions measured and projected
angle_index = 5
measured_values, n_markers = M.trajectories_to_values(trajectories, det_size)
M.plot_forward_measured(angle_index, found_parameters, det_pixel_size, trajectories, det_size, save_path)
    #histogram of used markers

flat_list = [item for sublist in M.indices_used for item in sublist]
fig, ax = plt.subplots(figsize=(16, 16))
counts, bins, bars = ax.hist(flat_list, density=False, bins = (np.linspace(0, n_markers, n_markers, endpoint = True).astype(int)).tolist())
    
for rect in ax.patches:
    height = rect.get_height()
    ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

plt.savefig(str(save_path + 'marker_hist.png'), bbox_inches='tight')

tp.plot_traj(trajectories_forward,label = True)

# #%%Trajectories cleanup
# indexs_badmarkers = np.where(counts == 0)[0]

# for i in indexs_badmarkers:
#     trajectories_forward = M.delete_trajectory(i, trajectories_forward)

# trajectories_forward = M.rearrange_labels(trajectories_forward)
save_path_traj = str(save_path+'trajectories_forward_bin%d_skip%d_angles%d'%(bb,skip,n_angles))
trajectories_forward.to_csv(save_path_traj)
#%% show the distance between markers
# from itertools import combinations
# locations = found_parameters.markers
# n_markers = len(locations)
# pairs = combinations(np.arange(n_markers), 2)
# same_marker = []
# #variables_array = pack_variables(variables)
# norm_tot = []
                         
# for pair in pairs:
#         norm = np.linalg.norm(locations[pair[0]] - locations[pair[1]])

#         if  norm <= 20:
#             same_marker.append(pair)
#         norm_tot.append((pair, norm))
    
# norm_tot.sort(key = lambda i: i[1])
# print(norm_tot)

#plot = tp.plot_traj(trajectories[trajectories['particle'].isin([12,13])],label = True)
#%% Inpainting
save_path_traj = str(save_path+'trajectories_forward_bin%d_skip%d_angles%d'%(bb,skip,n_angles))
trajectories, n_markers = M.read_trajectories(save_path_traj) 
   

                                                                                                     #%%
frames = M.Getty_load(path+name, folder_name, file_name, n_angles, bb, skip, counterclockwise = False, invert = False)[:,y_cut//bb:-(y_cut)//bb,:]
frames = M.Getty_preprocess(frames, flat, dark)
#%%

marker_size = trajectories['radius'].max()
test_run = True
mag = 1.5
frame_nr = 0
if test_run is True:
    M.inpaint(frames[frame_nr], trajectories_forward[trajectories_forward['frame']==frame_nr], marker_size, mag = mag, show = True, save = False)
#%% 
M.inpaint_all(frames, save_path, trajectories_forward, n_angles, bb, skip, marker_size, mag, show = True)
del(frames)
#%% Reconstruction 
save_path = '/export/scratch2/bossema/results/markers/Getty/publication/case_study_july/'
bb_load = 2
frames_inpainted = M.read_inpainted(save_path, bb, skip, bb_load = bb_load)
#%%
found_parameters = M.read_geom(save_path + 'geometry_%d_selected%d_selected2'%(n_markers_selected_1,n_markers_selected_2))
sirt_iterations = 50
sirt_bool = 1

fdk_vol_markers, sirt_vol_markers = M.recon_markers(found_parameters, frames_inpainted, det_pixel_size*bb_load, rotate = 0., sirt_iter = sirt_iterations, fdk = 1, sirt = sirt_bool, img_pixel = 0.2*bb_load, vol_size = [2000//bb_load,1200//bb_load,2400//bb_load],vol_tra = None, beam_hardening = True, compound = 'Ca', padding = 50//bb_load)
#data.write_stack(save_path + 'fdk_markercalibration_inp_bin%s_padded'%bb, 'slice', fdk_vol_markers[::-1], dim = 1)

#display.pyqt_graph(fdk_vol_markers[::-1], dim = 0)
#sirt_vol_markers = M.recon_markers_sirt(found_parameters, frames_inpainted, det_pixel_size, rotate = 0., sirt_iter = sirt_iterations, img_pixel = 0.2, vol_size = [2000,1200,2400],vol_tra = None, beam_hardening = False, padding = 50)


data.write_stack(save_path + 'sirt%d_markercalibration_inp_bin%s_beam_hardening'%(sirt_iterations, bb_load), 'slice', sirt_vol_markers[::-1], dim = 1)
display.pyqt_graph(sirt_vol_markers[::-1], dim = 0)
#%% Cast data to 8bit and scale

vol = data.read_stack(save_path +'fdk_markercalibration_inp_bin1_padded','slice', sample = 1, skip = 1, updown=False, transpose=[1,0,2])
#display.pyqt_graph(vol, dim = 1)

vol = data.cast2type(vol, 'uint8', bounds = [0, 0.06])

data.write_stack(save_path + 'fdk_markercalibration_inp_bin1_padded_8bit', 'slice', vol, dim = 1)
#, format = 'jpeg')
#%% Recon with marker calibration without inpainting
# found_parameters = M.read_geom(save_path + 'geometry_%d_selected%d_selected2'%(n_markers_selected_1,n_markers_selected_2))
# bb = 1 #does not work properly with bb = 2!!
# dark, flat = M.Getty_flatdark(path, name, file_name, bb, skip, form = 'DICOM')
# dark = dark[y_cut//bb:-(y_cut)//bb,:]
# flat = flat[y_cut//bb:-(y_cut)//bb,:]
# frames = M.Getty_load(path+name, folder_name, file_name, n_angles, bb, skip, counterclockwise = False, invert = False)[:,y_cut//bb:-(y_cut)//bb,:]
# frames = M.Getty_preprocess(frames, flat, dark)
#%%
# sirt_iterations = 200
# sirt_bool = 0
# fdk_vol_markers, sirt_vol_markers = M.recon_markers(found_parameters, frames, det_pixel_size, rotate = 0., sirt_iter = sirt_iterations, fdk = 1, sirt = sirt_bool, img_pixel = 0.2, vol_size = [2000,1200,2400],vol_tra = None, beam_hardening = False)
# display.pyqt_graph(fdk_vol_markers[::-1], dim = 1)

# data.write_stack(save_path + 'fdk_markercalibration_inp_bin%s'%bb, 'slice', fdk_vol_markers[::-1], dim = 0)




