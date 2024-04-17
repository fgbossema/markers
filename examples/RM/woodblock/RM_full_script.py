

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
import matplotlib.pyplot as plt
import matplotlib
from pandas import read_csv
from pathlib import Path
'''
First run in terminal (with right in and out path): 

ffmpeg -i /ufs/bossema/datasets/markers/rijkxray/Blokje_april2021_markers/Blokje6_60kv_3ma.mp4  -pix_fmt gray16be -vcodec tiff /ufs/bossema/datasets/markers/rijkxray/Blokje_april2021_markers/Blokje6/data/frame%06d.tif
ffmpeg -i /ufs/bossema/datasets/markers/rijkxray/Blokje_april2021_markers/Darkfield_blokje.mp4  -pix_fmt gray16be -vcodec tiff /ufs/bossema/datasets/markers/rijkxray/Blokje_april2021_markers/Blokje6/darks_flats/dark_data/frame%06d.tif
ffmpeg -i /ufs/bossema/datasets/markers/rijkxray/Blokje_april2021_markers/Flatfield_60kv_3ma_blokje.mp4  -pix_fmt gray16be -vcodec tiff /ufs/bossema/datasets/markers/rijkxray/Blokje_april2021_markers/Blokje6/darks_flats/flat_data/frame%06d.tif
'''
#%% Set directories and user input
#Data
path = str(Path('~/datasets/markers/RM/Woodblock_april2021_markers/Block6').expanduser())
name = ''
save_path = '/export/scratch2/bossema/results/markers/RM/Woodblock_april2021_markers/Block6/publication/'
 
M.make_directory_structure(save_path)

#%%

bb = 1 #Binning the detector pixels
skip = 2 #Reducing number of projections

#Marker detection
minR = 16
maxR = 21
sigma = 4
invert = False

#Calibration
n_markers = 10
n_markers_selected_2 = 8
distance = 9 #max distance between two markers
det_pixel_size = bb*0.127
n_markers_selected = 8
SOD = 500
ODD = 580
n_rounds = 1.5
n_iterations_step1 = 100
n_iterations_step2 =100
n_angles = 2700//skip
#Inpainting
mag = 1.5

#Reconstruction
fdk = 1
sirt = 0
sirt_iter = 100

plot = False 
server_run = False

#%% Run only once
#M.rijxray_datacorrection(save_path, name, n_angles, n_flat = 65, n_dark = 60, invert = False) 
#%% Reading dark and flat data
y_cut = 150
x_cut = 5

# Read data
frames = M.rijxray_load(path,name, n_angles, bb,skip, counterclockwise = False)[:,y_cut//bb:-(y_cut)//bb,x_cut//bb:-(x_cut)//bb]
n_angles = frames.shape[0]
#%% Use this block to find the relevant parameters for marker detection
if server_run is True:
    matplotlib.use('Agg')
    
test_run = True
if test_run is True:
    #Marker detection
    minR = 16
    maxR = 21

    sigma = 4
    invert = True
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
markerpositions, marker_projected_size = M.markers_blobdetector(frames, minR, maxR, sigma = sigma, invert = invert, test = False) 
markerpositions.to_csv(str(save_path+'markerpositions_skip%s'%skip))  
markerpositions = read_csv(str('/export/scratch2/bossema/results/markers/rijkxray/Blokje_april2021_markers/Blokje6/publication/'+'markerpositions_skip%s'%skip))
#%% Labelling markers
allowed_shift = 10*skip//bb
memory = 100
min_traj = 10
    
x_lim = [0,frames[0].shape[1]]
y_lim = [0,frames[0].shape[0]]
trajectories = M.find_trajectories(save_path, bb, skip, markerpositions, allowed_shift = allowed_shift, memory = memory, min_traj = min_traj, y_lim = y_lim, x_lim = x_lim, save = True)

#%% Calibration
Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi','det_eta', 'angles', 'markers' ])

det_size = np.array([frames[0].shape[1], frames[0].shape[0]])
found_parameters, trajectories_after, abs_error = M.calibration_parameters(trajectories, SOD, ODD, n_angles,
                                                            det_pixel_size, 
                                                            det_size, n_markers_selected_1 = n_markers, n_markers_selected_2 = n_markers_selected, 
                                                            distance = distance, n_rounds = n_rounds, plot = 1, 
                                                            show_all = 2, n_seeds = 10,
                                                            n_iterations_step1 = n_iterations_step1,
                                                            n_iterations_step2 = n_iterations_step2)

M.write_geom(save_path + 'geometry_%d_selected'%n_markers_selected, found_parameters)
x_lim = [0,frames[0].shape[1]]
y_lim = [0,frames[0].shape[0]]
#calculate forward projected marker locations
trajectories_forward = M.plot_result_trajectories(save_path, found_parameters, trajectories, det_pixel_size, det_size,y_lim = y_lim, x_lim = x_lim)
save_path_traj = str(save_path+'trajectories_forward_bin%d_skip%d'%(bb,skip))
trajectories_forward.to_csv(save_path_traj)
#%% Plotting results
plot = True
if plot is True:
    #markerpositions measured and projected
    angle_index = 5
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
    
#%% Inpainting
save_path_traj = str(save_path+'trajectories_forward_bin%d_skip%d'%(bb,skip))
trajectories, n_markers = M.read_trajectories(save_path_traj) 

test_run = True
mag = 1.5
frame_nr = 0
if test_run is True:
    M.inpaint(frames[frame_nr], trajectories[trajectories['frame']==frame_nr], mag = mag, marker_size = round(marker_projected_size), show = True, save = False)
#%%
M.inpaint_all(frames, save_path, trajectories_forward, n_angles, bb, skip, marker_projected_size, mag, show = True)
del(frames)
#%% Reconstruction 
bb_load = 1
frames_inpainted = M.read_inpainted(save_path, bb, skip, bb_load)
found_parameters = M.read_geom(save_path + 'geometry_%d_selected'%n_markers_selected)
#%% Recon with marker calibration
sirt_iterations = 200
sirt_bool = 1
fdk_vol_markers, sirt_vol_markers = M.recon_markers(found_parameters, frames_inpainted, det_pixel_size, rotate = 27., sirt_iter = sirt_iterations,fdk = 1, sirt = sirt_bool, vol_tra = None, beam_hardening = False)

dim0_low, dim0_high = 0,1100
dim1_low, dim1_high = 300,1600 
dim2_low, dim2_high = 400,1400 

if plot is True:
    display.pyqt_graph(fdk_vol_markers[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)

data.write_stack(save_path + 'fdk_markercalibration_inp_bin%s_%sangles'%(bb, n_angles), 'slice', fdk_vol_markers[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)
data.write_stack(save_path + 'sirt_%s_markercalibration_inp_bin%s_%sangles'%(sirt_iterations, bb, n_angles), 'slice', sirt_vol_markers[dim0_low:dim0_high,dim1_low:dim1_high,dim2_low:dim2_high], dim = 0)

#%% FDK with equidistant angles


frames_inpainted = M.read_inpainted(save_path, bb, skip)
#%%
Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi','det_eta', 'angles', 'markers' ])

n_angles = frames_inpainted.shape[0]
n_rounds = 1.5
n_markers = 10
angles_estimated = np.linspace(0,n_rounds*2*np.pi,n_angles, endpoint = False) #doorlopende hoeken
marker_locations =  (-2* np.random.rand(n_markers,3)+1)*30
marker_locations = marker_locations[marker_locations[:, 2].argsort()][::-1]

standard_parameters = Geom_param(source = np.array([0,-SOD, 0]), det = np.array([0,ODD,0]), det_theta = np.radians(0.), det_phi = np.radians(0.),
                                               det_eta = np.radians(0.), angles = angles_estimated,
                                               markers = marker_locations)
fdk_vol_markers, sirt_vol_markers = M.recon_markers(standard_parameters, frames_inpainted, det_pixel_size, rotate = 0., sirt_iter = 200,fdk = 1, sirt = 0, vol_tra = None, beam_hardening = False)

dim0_low, dim0_high = 0,1100
dim1_low, dim1_high = 300,1600 
dim2_low, dim2_high = 400,1400 
data.write_stack(save_path + 'fdk_equidistant_inp_bin%s'%bb, 'slice', fdk_vol_markers, dim = 0)