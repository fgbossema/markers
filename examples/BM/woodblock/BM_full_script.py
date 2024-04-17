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

#%% Set directories and user input
#Data
path = '/export/scratch2/bossema/datasets/markers/BM/'
name = 'CWI-TestObject-0mmBrass-16May2022'
save_path = '/export/scratch2/bossema/results/markers/BM/publication_sod881/'
dark_flat_name = '16May2022_darkflat/'
M.make_directory_structure(save_path)

bb = 1 #Binning the detector pixels
skip = 1 #Reducing number of projections

#Marker detection
minR = 10
maxR = 16

invert = True

#Calibration
n_markers_selected_2 = 8
distance = 9 #max distance between two markers
det_pixel_size = bb*0.2 #Detector pixels are 200 micron
SOD = 881
ODD = 1362
n_rounds = 1
n_iterations_step1 = 100
n_iterations_step2 =100
n_angles = 1800//skip

#Inpainting
mag = 1.2

#Reconstruction
fdk = 1
sirt = 1
sirt_iter = 200

plot = False
server_run = False
test_run = True

y_cut = 500
x_cut = 150
#%% Reading dark and flat data
dark = data.read_image(path+dark_flat_name +'I500F_DarkField_16b.tif', sample = bb)[60//bb:-60//bb, 60//bb:-60//bb][y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb]
flat = data.read_image(path+dark_flat_name +'I500F_FlatField_60kV_3p0mA_foc_0mmBrass_FDD-2334p7_ODD-1362p1_16b.tif', sample = bb)[60//bb:-60//bb, 60//bb:-60//bb][y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb] 
frames1 = M.BM_load(path, name, bb, skip, counterclockwise = False)[:900//bb][:,y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb]
frames1 = M.BM_preprocess(frames1, flat, dark)
#%%
frames2 = M.BM_load(path, name, bb, skip, counterclockwise = False)[900//bb:1350//bb][:,y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb]
frames2 = M.BM_preprocess(frames2, flat, dark)
frames3 = M.BM_load(path, name, bb, skip, counterclockwise = False)[1350//bb:][:,y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb]
frames3 = M.BM_preprocess(frames3, flat, dark)
frames = np.concatenate((frames1,frames2, frames3))[::-1] #counterclockwise correction
display.pyqt_graph(frames)
del(frames1, frames2, frames3)
#%%
offset = 31.46
sirt_iterations = 200
sirt_bool = 1
bb_load = 1
(geom_BM, fdk_vol_system, sirt_vol_system) = M.BM_recon(frames, path, name, bb_load, rotate = -35., fdk = fdk, sirt = sirt_bool, sirt_iter = sirt_iterations, real_sod = 881, offset_corr = offset)

dim0_low, dim0_high = 100//bb, 1000//bb
dim1_low, dim1_high = 300//bb, 1500//bb
dim2_low, dim2_high = 300//bb, 1100//bb

data.write_stack(save_path + 'recon_system_fdk_bin%s_noinp'%bb_load, 'slice', fdk_vol_system[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
data.write_stack(save_path + 'recon_system_sirt_%s_inp_bin%s_noinp'%(sirt_iterations, bb), 'slice', sirt_vol_system[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)

#%% Use this block to find the relevant parameters for marker detection
if server_run is True:
    matplotlib.use('Agg')

if test_run is True:
    #Marker detection
    minR = 10
    maxR = 16
    invert = True
    image1 = frames[600]
    image2 = frames[1]
    test_frames = [image1]
    plt.imshow(image1)
    plt.imshow(image2)
    marker_blobdetect = M.markers_blobdetector(test_frames, minR, maxR, invert = invert, save_path = save_path, test = True)

#%% Find all marker positions
n_angles = frames.shape[0]
markerpositions, marker_projected_size = M.markers_blobdetector(frames, minR, maxR,invert = invert) 
markerpositions.to_csv(str(save_path+'markerpositions')) 
#%%
markerpositions = read_csv(str(save_path + 'markerpositions'))
allowed_shift = 25*skip//bb
memory = 300
min_traj = 10
    
x_lim = [0,frames[0].shape[1]]
y_lim = [0,frames[0].shape[0]]
trajectories = M.find_trajectories(save_path, bb, skip, markerpositions, allowed_shift = allowed_shift, memory = memory, min_traj = min_traj, y_lim = y_lim, x_lim = x_lim)
trajectories.to_csv(str(save_path+'trajectories'))
    
#%% Calibration
Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi','det_eta', 'angles', 'markers' ])
trajectories, n_markers = M.read_trajectories(str(save_path+'trajectories')) 
marker_size = np.asarray(trajectories['radius']).mean()
x_lim = [0,dark.shape[1]]
y_lim = [0,dark.shape[0]]
np.random.seed(1)
rest = n_angles-(n_angles//n_rounds)*int(np.floor(n_rounds))
full_rounds = np.tile(np.linspace(0,2*np.pi,n_angles//n_rounds, endpoint = False),int(np.floor(n_rounds)))
partial_rounds = np.linspace(0,rest/n_angles*2*np.pi,rest, endpoint = False)

angles_estimated = np.concatenate((full_rounds,partial_rounds))

marker_locations =  (-2* np.random.rand(n_markers,3)+1)*50
marker_locations = marker_locations[marker_locations[:, 2].argsort()][::-1]

geom_parameters_estimated = Geom_param(source = np.array([0,-SOD, 0]), det = np.array([-30,ODD,0]), det_theta = np.radians(0.), det_phi = np.radians(0.),
                                               det_eta = np.radians(0.), angles = angles_estimated,
                                               markers = marker_locations)


bounds = (np.concatenate((np.array([-50,ODD-200,-50]),np.array([-1/8*np.pi,-1/8*np.pi,-1/8*np.pi]),
                           np.repeat(0,n_angles-1),np.repeat(-1000,n_markers*3))), 
           np.concatenate((np.array([50,ODD+200,50]),np.array([1/8*np.pi,1/8*np.pi,1/8*np.pi]),
                           np.repeat(n_rounds*2*np.pi+1,n_angles-1),np.repeat(1000,n_markers*3))))



det_size = np.array([dark.shape[1], dark.shape[0]])
found_parameters, trajectories_after, abs_error = M.calibration_parameters(trajectories, SOD, ODD, n_angles,
                                                            det_pixel_size, 
                                                            det_size, n_markers_selected_1 = n_markers, n_markers_selected_2 = n_markers_selected_2, 
                                                            distance = distance, n_rounds = n_rounds, plot = 1, 
                                                            show_all = 2, n_seeds = 10,
                                                            bounds_given = bounds,
                                                            geom_parameters_estimated = geom_parameters_estimated,
                                                            n_iterations_step1 = n_iterations_step1,
                                                            n_iterations_step2 = n_iterations_step2)

M.write_geom(save_path + 'geometry_%d_selected'%n_markers_selected_2, found_parameters)

#%%#calculate forward projected marker locations
trajectories_forward = M.plot_result_trajectories(save_path, found_parameters, trajectories, det_pixel_size, det_size,y_lim, x_lim)
save_path_traj = str(save_path+'trajectories_forward_bin%d_skip%d'%(bb,skip))
trajectories_forward.to_csv(save_path_traj)

if plot is True:
    #markerpositions measured and projected
    angle_index = 5
    measured_values, n_markers = M.trajectories_to_values(trajectories, det_size)
    M.plot_forward_measured(angle_index, found_parameters, det_pixel_size, trajectories, det_size, save_path)
    #histogram of used markers
    flat_list = [item for sublist in M.indices_used for item in sublist]
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.hist(flat_list, density=False, bins = [0,1,2,3,4,5,6,7,8,9])

    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    plt.savefig(str(save_path + 'marker_hist.png'), bbox_inches='tight')
    
#%% Inpainting
save_path_traj = str(save_path+'trajectories_forward_bin%d_skip%d'%(bb,skip))
trajectories_forward, n_markers = M.read_trajectories(save_path_traj) 
#%%
test_run = True

frame_nr = 600
if test_run is True:
    M.inpaint(frames[frame_nr], trajectories_forward[trajectories_forward['frame']==frame_nr], marker_size, mag = mag, show = True, save_path = save_path)
#%%
M.inpaint_all(frames, save_path, trajectories_forward, n_angles, bb, skip,  marker_size, mag, show = False)
#del(frames)
#%% Reconstruction 
bb_load = 1

frames_inpainted = M.read_inpainted(save_path, bb, skip, bb_load)
#frames_inpainted[:,y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb]
#%% Recon with system parameters
offset = 31.46
sirt_iterations = 200
sirt_bool = 1

(geom_BM, fdk_vol_system, sirt_vol_system) = M.BM_recon(frames_inpainted, path, name, bb_load, rotate = -35., fdk = fdk, sirt = sirt_bool, sirt_iter = sirt_iterations, real_sod = 881, offset_corr = offset)

#%%  

dim0_low, dim0_high = 100//bb, 1000//bb
dim1_low, dim1_high = 300//bb, 1500//bb
dim2_low, dim2_high = 300//bb, 1100//bb

plot = False
if plot is True:
    display.pyqt_graph(fdk_vol_system[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
#%%
data.write_stack(save_path + 'recon_system_fdk_bin%s'%bb_load, 'slice', fdk_vol_system[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
data.write_stack(save_path + 'recon_system_sirt_%s_inp_bin%s'%(sirt_iterations, bb), 'slice', sirt_vol_system[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)

#%%
del(fdk_vol_system)
del(sirt_vol_system)
#%% Recon with marker calibration
found_parameters = M.read_geom(save_path + 'geometry_%d_selected'%n_markers_selected_2)
sirt_iterations = 200
sirt_bool = 1
fdk_vol_markers, sirt_vol_markers = M.recon_markers(found_parameters, frames_inpainted, det_pixel_size, rotate = -35., sirt_iter = sirt_iterations,fdk = 1, sirt = sirt_bool, vol_tra = None, beam_hardening = False)

dim0_low, dim0_high = 100//bb, 1000//bb
dim1_low, dim1_high = 300//bb, 1500//bb
dim2_low, dim2_high = 300//bb, 1100//bb

plot = False


if plot is True:
    display.pyqt_graph(fdk_vol_markers[::-1][dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
#%%
#To Do check reconstruction upside down by markers
data.write_stack(save_path + 'fdk_markercalibration_inp_bin%s'%bb, 'slice', fdk_vol_markers[::-1][dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
data.write_stack(save_path + 'sirt_%s_markercalibration_inp_bin%s'%(sirt_iterations, bb), 'slice', sirt_vol_markers[::-1][dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
#%%
#%% Recon with marker calibration no inpainting
found_parameters = M.read_geom(save_path + 'geometry_%d_selected'%n_markers_selected_2)
sirt_iterations = 200
sirt_bool = 1
fdk_vol_markers, sirt_vol_markers = M.recon_markers(found_parameters, frames, det_pixel_size, rotate = -35., sirt_iter = sirt_iterations,fdk = 1, sirt = sirt_bool, vol_tra = None, beam_hardening = False)

dim0_low, dim0_high = 100//bb, 1000//bb
dim1_low, dim1_high = 300//bb, 1500//bb
dim2_low, dim2_high = 300//bb, 1100//bb

plot = False


if plot is True:
    display.pyqt_graph(fdk_vol_markers[::-1][dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
#%%
#To Do check reconstruction upside down by markers
data.write_stack(save_path + 'fdk_markercalibration_noinp_bin%s'%bb, 'slice', fdk_vol_markers[::-1][dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
data.write_stack(save_path + 'sirt_%s_markercalibration_noinp_bin%s'%(sirt_iterations, bb), 'slice', sirt_vol_markers[::-1][dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)

