#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bossema

This is a script to test the estimation of system parameters using markers. 
In the FleX-ray lab we scanned a marker holder of foam containing 10 small 
metal balls.
The system parameters are estimated by minimizing the difference between 
the forward projection of the estimated marker positions.
Then a reconstruction is made with the FleX-ray system parameters and 
the computed parameters for comparison. 
 
"""

import markers as M
import numpy as np
from collections import namedtuple
from flexdata import data
from flexdata import display
import trackpy as tp

import matplotlib.pyplot as plt
import random
from pandas import read_csv
from pathlib import Path


#%% Set directories:
path = '/export/scratch2/bossema/datasets/markers/flexray/Markers12June2020/'
name = 'markertest_2020_highresdet/'
save_path = '/export/scratch2/bossema/results/markers/flexray/Markers12June2020/'+name
#%% User input
bb = 1 #Binning the detector pixels
skip = 1 #Reducing number of projections
n_markers = 10
n_markers_selected = 8
det_pixel_size = bb*0.0748

SOD = 658
ODD = 1089-658
n_rounds = 1
n_iterations_step1 = 100
n_iterations_step2 =100
#%% Read data and preprocess
dark = data.read_stack(path+name, 'di000000', sample = bb, updown=False, 
                       transpose=[0,1,2])
flat = data.read_stack(path+name, 'io000000', sample = bb, updown=False, 
                       transpose=[0,1,2])    
proj = data.read_stack(path+name, 'scan_', sample = bb, skip = skip, updown=False, 
                       transpose=[0,1,2])

proj1 = proj[:700]
#preprocess
flat = (flat - dark).mean(0)
proj1 = (proj1 - dark) / flat[None,:, :]
proj1 = -np.log(proj1).astype('float32')

#%%
proj2 = proj[700:]

#preprocess
proj2 = (proj2 - dark) / flat[None,:, :]
proj2 = -np.log(proj2).astype('float32')

proj = np.concatenate((proj1,proj2))
n_angles = proj.shape[0]
#display.pyqt_graph(proj, dim = 0)
del(proj1, proj2)
#%%
test_run = False
if test_run is True:
    #Marker detection
    minR = 21
    maxR = 27

    invert = True
    #import tifffile

    #im_name1 = path + name +'/dx/CWI-TestObject-0mmBrass-16May2022_00001'+'.tif'
    image1 = proj[0]
    #tifffile.imread(im_name1)[60//bb:-60//bb, 60//bb:-60//bb]

    #im_name2 = path + name +'/dx/CWI-TestObject-0mmBrass-16May2022_00451'+'.tif'
    image2 = proj[n_angles//4]#tifffile.imread(im_name2)[60//bb:-60//bb, 60//bb:-60//bb]
    test_frames = [image1, image2]
    plt.imshow(image1)
    plt.imshow(image2)
    marker_blobdetect = M.markers_blobdetector(test_frames, minR, maxR, invert = invert, save_path = None, test = True)

#%% Find the marker positions on the detector
minR = 21
maxR = 27
invert = True
markerpositions, marker_projected_size = M.markers_blobdetector(proj, minR, maxR, invert = invert, test = False)
markerpositions.to_csv(str(save_path+'markerpositions'))
#%%

markerpositions = read_csv(str(save_path + 'markerpositions'))
trajectories = M.find_trajectories(save_path, bb, skip, markerpositions, allowed_shift = 15*skip, memory = 100, min_traj = 10)
#%% Save!
trajectories.to_csv(str(save_path+'trajectories_frame_bin%d_skip%d'%(bb,skip)))


#%% Read the marker list and transform to values measured from center of detector.
trajectories, n_markers = M.read_trajectories(str(save_path+'trajectories_frame_bin%d_skip%d'%(bb,skip))) 

#trajectories = trajectories[trajectories['frame'].isin(np.arange(0,n_angles))]
#tp.plot_traj(trajectories[trajectories['frame'].isin(np.arange(0,40))],label = True)
#%%
Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi','det_eta', 'angles', 'markers' ])

np.random.seed(1)
rest = n_angles-(n_angles//n_rounds)*int(np.floor(n_rounds))
full_rounds = np.tile(np.linspace(0,2*np.pi,n_angles//n_rounds, endpoint = False),int(np.floor(n_rounds)))
partial_rounds = np.linspace(0,rest/n_angles*2*np.pi,rest, endpoint = False)

angles_estimated = np.concatenate((full_rounds,partial_rounds))

marker_locations =  (-2* np.random.rand(n_markers,3)+1)*100
marker_locations = marker_locations[marker_locations[:, 2].argsort()][::-1]
#source = np.array([0,-SOD,0]),
print(marker_locations)

geom_parameters_estimated = Geom_param(source = np.array([0,-SOD, 0]), det = np.array([0,ODD,0]), det_theta = np.radians(0.), det_phi = np.radians(0.),
                                           det_eta = np.radians(0.), angles = angles_estimated,
                                           markers = marker_locations)


bounds = (np.concatenate((np.array([-50,ODD-200,-50]),np.array([-1/8*np.pi,-1/8*np.pi,-1/8*np.pi]),
                       np.repeat(0,n_angles-1),np.repeat(-1000,n_markers*3))), 
       np.concatenate((np.array([50,ODD+200,50]),np.array([1/8*np.pi,1/8*np.pi,1/8*np.pi]),
                       np.repeat(n_rounds*2*np.pi+1,n_angles-1),np.repeat(1000,n_markers*3))))

#%% Calibration
distance = 9 #max distance between two markers

det_size = np.array([proj[0].shape[1], proj[0].shape[0]])
#Fun the calibration function that uses leastsq to find parameters
found_parameters, trajectories, abs_error = M.calibration_parameters(trajectories, SOD, ODD, n_angles,
                                                            det_pixel_size, 
                                                            det_size, n_markers_selected, 
                                                            distance, n_rounds, plot = 1, 
                                                            show_all = 2, 
                                                            bounds_given = bounds,
                                                            geom_parameters_estimated = geom_parameters_estimated,
                                                            n_iterations_step1 = n_iterations_step1,
                                                            n_iterations_step2 = n_iterations_step2)

angle_index = 1
M.write_geom(save_path + 'fdk_markercalibration/geometry_%d_selected'%n_markers_selected, found_parameters)
#M.plot_forward_measured(angle_index, found_parameters, det_pixel_size, measured_values, save_path)
#%% Inpainting
x_lim = [0,dark[0].shape[1]]
y_lim = [0,dark[0].shape[0]]
found_parameters = M.read_geom(save_path+'fdk_markercalibration/geometry_%d_selected'%n_markers_selected)
#calculate forward projected marker locations
trajectories_forward = M.plot_result_trajectories(save_path, found_parameters, trajectories, det_pixel_size, det_size,y_lim = y_lim, x_lim = x_lim)
save_path_traj = str(save_path+'trajectories_forward_bin%d_skip%d'%(bb,skip))
trajectories_forward.to_csv(save_path_traj)
#%%
trajectories_forward, n_markers = M.read_trajectories(save_path_traj) 

#%%
marker_size = np.asarray(trajectories['radius']).mean()
test_run = True
mag = 1.1
frame_nr = 350
if test_run is True:
    M.inpaint(proj[frame_nr], trajectories_forward[trajectories_forward['frame']==frame_nr], marker_size, mag = mag, show = True)
#%%
M.inpaint_all(proj, save_path, trajectories_forward, n_angles, bb, skip, marker_size, mag, show = True)

#%% Reconstruction 
proj_inpainted = M.read_inpainted(save_path, bb, skip)#%% Reconstruction with the parameters found by the optimisation
#found_parameters = M.read_geom(save_path + 'marker_geometry')


#%% Reconstruction
import markers as M
found_parameters = M.read_geom(save_path +  'fdk_markercalibration/geometry_%d_selected'%n_markers_selected)
ODD = found_parameters.det[1]
mag = (SOD+ODD)/SOD 
img_pixel = det_pixel_size/mag

fdk_vol, sirt_vol = M.recon_markers(found_parameters, proj, bb, det_pixel_size, sirt_iter = 5,fdk = 1, sirt = 1, vol_tra = None, beam_hardening = False)

data.write_stack(save_path + 'fdk_markercalibration', 'slice', fdk_vol, dim = 0)
display.pyqt_graph(fdk_vol, dim = 0)

#%%
det_size = np.array([proj[0].shape[1], proj[0].shape[0]])
measured_values, n_markers = M.trajectories_to_values(trajectories, det_size)

angle_index = 10
M.plot_forward_measured(angle_index, found_parameters, det_pixel_size, measured_values, save_path)

#%% Reconstruction with original Flexray metadata not inpainted
(geom, fdk_vol_flex, sirt_vol_flex) = M.recon_flexray(path+name, proj, binn = bb, skip = skip, sirt_iter = 10, fdk = 1, sirt = 0, parameters_date = 'June_2020')
data.write_stack(save_path + 'fdk_flexraycalibration', 'slice', fdk_vol_flex, dim = 0)

#%% Reconstruction with original Flexray metadata
proj_inpainted = data.read_stack(save_path+'data_inpainted_binn1_skip1', 'scan', sample = bb, skip = skip, updown=False, transpose=[0,1,2])

n_angles = proj_inpainted.shape[0]

(geom, fdk_vol_flex_ipt, sirt_vol_flex) = M.recon_flexray(path+name, proj_inpainted, binn = bb, skip = skip, sirt_iter = 10, fdk = 1, sirt = 0, parameters_date = 'June_2020')
#display.pyqt_graph(fdk_vol_flex_ipt, dim = 0)

data.write_stack(save_path + 'fdk_flexraycalibration_inpainted', 'slice', fdk_vol_flex_ipt, dim = 0)





