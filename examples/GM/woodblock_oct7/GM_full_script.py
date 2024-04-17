
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
from pathlib import Path
import pandas as pd
#%% Set directories and user input
#Data
path = '/export/scratch2/bossema/datasets/markers/GM/'
name = 'woodblock_7oct/' 
folder_name = 'data1/'
folder_name2 = 'data2/'
save_path = '/export/scratch2/bossema/results/markers/GM/publication/'+name+'all_angles/'
file_name = 'IMG'
M.make_directory_structure(save_path)

bb = 1 #Binning the detector pixels
skip = 1 #Reducing number of projections

#Marker detection
minR = 12
maxR = 1
sigma = 4
invert = False

#Calibration

n_markers_selected = 8
distance = 9 #max distance between two markers
det_pixel_size = bb*0.2 #Detector pixels are 200 micron
SOD = 881
ODD = 1362
n_rounds = 3.05
n_iterations_step1 = 100
n_iterations_step2 = 100


#Inpainting
mag = 1.5

#Reconstruction
fdk = 1
sirt = 0
sirt_iter = 100
counter = False
plot = False 
server_run = False
y_cut = 500 #TODO check of dit effect heeft omdat het assymetrisch is!
x_cut = 200
#%% Reading dark and flat data


dark, flat = M.Getty_flatdark(path, name, file_name, bb, skip, form = 'DICOM')
dark = dark[y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb]
flat = flat[y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb]
# Read data
frames = M.Getty_load(path+name, folder_name, file_name,691, bb, skip, counterclockwise = counter, invert = False)[:,y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb]
frames2 = M.Getty_load(path+name, folder_name2, file_name, 778, bb, skip, counterclockwise = counter, invert = False)[:,y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb]
frames = np.concatenate((frames,frames2))
n_angles = frames.shape[0]

del(frames2)
#%% Use this block to find the relevant parameters for marker detection
if server_run is True:
    matplotlib.use('Agg')
    
test_run = True
if test_run is True:
    #Marker detection

    minR = 12
    maxR = 16
    sigma = 4
    invert = False
    image1 = frames[0]
    drange = image1.max() - image1.min()
    low_threshold = 0.1*drange
    high_threshold = 0.2*drange
    print(low_threshold, high_threshold)
    
    image2 = frames[3]
    test_frames = [image1,image2]
    plt.imshow(image1)
    plt.imshow(test_frames[1])
    marker_blobdetect = M.markers_blobdetector(test_frames,  minR, maxR, low_threshold, high_threshold, sigma = sigma, invert = invert, save_path = None, test = True)

#%% Find all marker positions
drange = frames[0].max() - frames[0].min()
low_threshold = 0.1*drange
high_threshold = 0.2*drange
print(low_threshold, high_threshold)
    
markerpositions, marker_projected_size = M.markers_blobdetector(frames, minR, maxR,low_threshold, high_threshold, sigma = sigma, invert = invert, test = False) 
markerpositions.to_csv(str(save_path+'markerpositions_all')) 

# #%% Add second part
# markerpositions2, marker_projected_size2 = M.markers_blobdetector(frames2, minR, maxR,low_threshold, high_threshold, sigma = sigma, invert = invert, test = False) 

# markerpositions_all = pd.concat(markerpositions,markerpositions2)
# markerpositions_all.to_csv(str(save_path+'markerpositions_all'))  
#markerpositions.to_csv(str(save_path+'markerpositions')) 
#%%
markerpositions = read_csv(str(save_path + 'markerpositions_all'))
allowed_shift = 100*skip//bb
memory = 100
min_traj = 10
    
x_lim = [0,dark.shape[1]]
y_lim = [0,dark.shape[0]]
skip = 1
trajectories = M.find_trajectories(save_path, bb, skip, markerpositions, allowed_shift = allowed_shift, memory = memory, min_traj = min_traj, y_lim = y_lim, x_lim = x_lim, save = True)
trajectories.to_csv(str(save_path+'trajectories'))

#%% Calibration
Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi','det_eta', 'angles', 'markers' ])
trajectories, n_markers = M.read_trajectories(str(save_path+'trajectories')) 

# For subset of angles tests
# n_angles_selected = 800
# trajectories = trajectories[trajectories['frame']<n_angles_selected]
# trajectories = M.rearrange_labels(trajectories)
# tp.plot_traj(trajectories, label = True)
# n_markers = len(set(trajectories['particle']))
# n_angles = n_angles_selected
# n_rounds = 1.6

# For subset of angles tests

# trajectories = trajectories[trajectories['frame']>0]
# trajectories = M.rearrange_labels(trajectories)
# n_angles -= 1





np.random.seed(1)
# rest = n_angles-(n_angles//n_rounds)*int(np.floor(n_rounds))
# full_rounds = np.tile(np.linspace(0,2*np.pi,n_angles//n_rounds, endpoint = False),int(np.floor(n_rounds)))
# partial_rounds = np.linspace(0,rest/n_angles*2*np.pi,rest, endpoint = False)

# angles_estimated = np.concatenate((full_rounds,partial_rounds))

angles_estimated = np.linspace(0,n_rounds*2*np.pi,n_angles, endpoint = False) #doorlopende hoeken


marker_locations =  (-2* np.random.rand(n_markers,3)+1)*30
marker_locations = marker_locations[marker_locations[:, 2].argsort()][::-1]

geom_parameters_estimated = Geom_param(source = np.array([0,-SOD, 0]), det = np.array([-30,ODD,0]), det_theta = np.radians(0.), det_phi = np.radians(0.),
                                               det_eta = np.radians(0.), angles = angles_estimated,
                                               markers = marker_locations)


# bounds = (np.concatenate((np.array([-200,ODD-200,-200]),np.array([-1/8*np.pi,-1/8*np.pi,-1/8*np.pi]),
#                            np.repeat(-0.5,n_angles-1),np.repeat(-1000,n_markers*3))), 
#            np.concatenate((np.array([200,ODD+200,200]),np.array([1/8*np.pi,1/8*np.pi,1/8*np.pi]),
#                            np.repeat(n_rounds*2*np.pi+1,n_angles-1),np.repeat(1000,n_markers*3))))

bounds = (np.concatenate((np.array([-200,ODD-200,-200]),np.array([-1/8*np.pi,-1/8*np.pi,-1/8*np.pi]),
                           np.repeat(0,n_angles-1),np.repeat(-1000,n_markers*3))), 
           np.concatenate((np.array([200,ODD+200,200]),np.array([1/8*np.pi,1/8*np.pi,1/8*np.pi]),
                           np.repeat(n_rounds*2*np.pi,n_angles-1),np.repeat(1000,n_markers*3))))
#bounds tot max angles


det_size = np.array([dark.shape[1], dark.shape[0]]) #width, height

found_parameters, trajectories_after, abs_error = M.calibration_parameters(trajectories, SOD, ODD, n_angles,
                                                            det_pixel_size, 
                                                            det_size, n_markers_selected,
                                                            distance, n_rounds, plot = 1, 
                                                            show_all = 2, 
                                                            bounds_given = bounds,
                                                            geom_parameters_estimated = geom_parameters_estimated,
                                                            markers_loc_given = None, n_seeds = 10,
                                                            n_iterations_step1 = n_iterations_step1,
                                                            n_iterations_step2 = n_iterations_step2)
#%%
M.write_geom(save_path + 'geometry_%d_selected'%n_markers_selected, found_parameters)
x_lim = [0,dark.shape[1]]
y_lim = [0,dark.shape[0]]
#calculate forward projected marker locations
trajectories_forward = M.plot_result_trajectories(save_path, found_parameters, trajectories, det_pixel_size, det_size,y_lim = y_lim, x_lim = x_lim)
save_path_traj = str(save_path+'trajectories_forward_bin%d_skip%d'%(bb,skip))
trajectories_forward.to_csv(save_path_traj)
plot = True
if plot is True:
    #markerpositions measured and projected
    angle_index = 0
    measured_values, n_markers = M.trajectories_to_values(trajectories, det_size)
    M.plot_forward_measured(angle_index, found_parameters, det_pixel_size, trajectories, det_size, save_path)
    #histogram of used markers
    plt.figure()
    flat_list = [item for sublist in M.indices_used for item in sublist]
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.hist(flat_list, density=False, bins = [0,1,2,3,4,5,6,7,8,9])

    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    plt.savefig(str(save_path + 'marker_hist.png'), bbox_inches='tight')
plt.figure()
plt.scatter(np.linspace(0,n_angles,n_angles, endpoint = False), found_parameters.angles)
#%% Inpainting
save_path_traj = str(save_path+'trajectories_forward_bin%d_skip%d'%(bb,skip))
trajectories_forward, n_markers = M.read_trajectories(save_path_traj) 
#%%
frames = M.Getty_preprocess(frames, flat, dark)
#%%
marker_size = np.asarray(trajectories['radius']).mean()
test_run = True
mag = 1.5
frame_nr = 0
if test_run is True:
    M.inpaint(frames[frame_nr], trajectories_forward[trajectories_forward['frame']==frame_nr], marker_size, mag = mag, show = True)
#%%
M.inpaint_all(frames, save_path, trajectories_forward, n_angles, bb, skip, marker_size, mag, show = True)
#del(frames)
#%% Reconstruction 
frames_inpainted = M.read_inpainted(save_path, bb, skip)

#%% Recon with marker calibration

found_parameters = M.read_geom(save_path + 'geometry_%d_selected'%n_markers_selected)

sirt_iterations = 200
sirt_bool = 1 #rotate 94
fdk_vol_markers, sirt_vol_markers = M.recon_markers(found_parameters, frames_inpainted, det_pixel_size, rotate = 95., sirt_iter = sirt_iterations,fdk = 1, sirt = sirt_bool, vol_tra = None, beam_hardening = False)
#%%
dim0_low, dim0_high = 0,900
dim1_low, dim1_high = 200,1400
dim2_low, dim2_high = 400,1200
plot = False

if plot is True:
    display.pyqt_graph(fdk_vol_markers[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)

#%%
data.write_stack(save_path + 'fdk_markercalibration_inp_bin%s'%bb, 'slice', fdk_vol_markers[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
data.write_stack(save_path + 'sirt_%s_markercalibration_inp_bin%s'%(sirt_iterations, bb), 'slice', sirt_vol_markers[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)

#%% Recon with marker calibration no inpainting
frames = M.Getty_preprocess(frames, flat, dark)

found_parameters = M.read_geom(save_path + 'geometry_%d_selected'%n_markers_selected)

sirt_iterations = 200
sirt_bool = 1 
fdk_vol_markers, sirt_vol_markers = M.recon_markers(found_parameters, frames, det_pixel_size, rotate = 95., sirt_iter = sirt_iterations,fdk = 1, sirt = sirt_bool, vol_tra = None, beam_hardening = False)

dim0_low, dim0_high = 0,900
dim1_low, dim1_high = 200,1400
dim2_low, dim2_high = 400,1200
plot = False

if plot is True:
    display.pyqt_graph(fdk_vol_markers[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)

data.write_stack(save_path + 'fdk_markercalibration_noinp_bin%s'%bb, 'slice', fdk_vol_markers[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
data.write_stack(save_path + 'sirt_%s_markercalibration_noinp_bin%s'%(sirt_iterations, bb), 'slice', sirt_vol_markers[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)

#%% FDK with equidistant angles
print('angles for first round', found_parameters.angles[found_parameters.angles < 2*np.pi].shape)

frames_inpainted = M.read_inpainted(save_path, bb, skip)

#%%
subset = 515
#display.pyqt_graph(frames_inpainted[:subset])
Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi','det_eta', 'angles', 'markers' ])

n_angles = frames_inpainted[:subset].shape[0]
n_rounds = 1
n_markers = 10
angles_estimated = np.linspace(0,n_rounds*2*np.pi,n_angles, endpoint = False) #doorlopende hoeken
marker_locations =  (-2* np.random.rand(n_markers,3)+1)*30
marker_locations = marker_locations[marker_locations[:, 2].argsort()][::-1]

standard_parameters = Geom_param(source = np.array([0,-SOD, 0]), det = np.array([-33,ODD,0]), det_theta = np.radians(0.), det_phi = np.radians(0.),
                                               det_eta = np.radians(0.), angles = angles_estimated,
                                               markers = marker_locations)
fdk_vol_markers, sirt_vol_markers = M.recon_markers(standard_parameters, frames_inpainted[:subset], det_pixel_size, rotate = 95., sirt_iter = 200,fdk = 1, sirt = 0, vol_tra = None, beam_hardening = False)

dim0_low, dim0_high = 0,900
dim1_low, dim1_high = 200,1400
dim2_low, dim2_high = 400,1200
data.write_stack(save_path + 'fdk_equidistant_inp_bin%s_500angles_1round'%bb, 'slice', fdk_vol_markers[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)

