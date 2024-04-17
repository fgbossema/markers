#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:58:21 2021

@author: bossema
"""
import markers as M
import autograd.numpy as np
from collections import namedtuple
from flexdata import data
from flexdata import display
import trackpy as tp
import matplotlib.pyplot as plt
import random
from pandas import read_csv
from pathlib import Path

 #%%
np.random.seed(0)

SOD = 948.
ODD = 1089-948
n_rounds = 1
n_angles = 1440
n_markers = 10
n_markers_selected = 10
n_iter_optimisation_1 = 50
n_iter_optimisation_2 = n_iter_optimisation_1
locations = np.array([[238, 342,580],[318,340,521],[462,358,502],[550,314,479],[635,308,440], 
                     [206,643,391],[308,654,367],[465,650,340],[564,651,306],[669,718,285]], dtype = 'float64')[::-1]

locations_corrected = np.empty((10,3))
locations_corrected[:,0] = locations[:,0] - 968/2
locations_corrected[:,1] = locations[:,1] - 968/2 
locations_corrected[:,2] = 764/2 - locations[:,2] 

#locations_corrected *= geom['img_pixel']
locations_corrected *= 0.130232566
marker_locations = locations_corrected
pixel_size = 0.1496
det_size = [968, 764]

voxel_size = pixel_size * SOD/(SOD+ODD)

#%%
save_path = '/export/scratch2/bossema/results/markers/simulations/simulation_figures/feb_24_location_deviation/'
 
max_markers = 10
cost_list = []
cost_list_original = []
max_dist = []
av_dist = []



mu = 0
sigmas = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

seeds = [0,1,2,3,4,5,6,7,8,9]
for sigma in sigmas:
    for seed in seeds:
        np.random.seed(seed)
        sim_dataframe, geom_parameters = M.simulate_data(SOD, ODD, n_rounds, n_angles, n_markers, marker_locations, det_size, pixel_size)
        binn = 1
        skip = 1                               
        trajectories = M.find_trajectories(save_path, binn, skip, sim_dataframe, allowed_shift = 100, memory = 10, min_traj = 10)
        trajectories['x'] = trajectories['x'] + np.random.normal(mu, sigma, len(trajectories['x']) )
        trajectories['y'] = trajectories['y'] + np.random.normal(mu, sigma, len(trajectories['y']) )

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_ylim([0,764])
        ax.set_xlim([0,968])
        plot = tp.plot_traj(trajectories,label = True)

        fig.savefig(str(save_path +  'trajectories/'+ 'seed%s_sigma%s.png'%(seed,sigma)), bbox_inches='tight')

        s = np.array([0, -SOD, 0])
        measured_values, n_markers = M.trajectories_to_values(trajectories, det_size)
        total_cost = M.res_all(M.pack_variables(geom_parameters), n_angles, s, measured_values, pixel_size, n_markers)
    
        print('starting with optimisation for seed', seed)
        print('total cost with original geometry and disturbed trajectories', 0.5*sum(total_cost**2))

#%Calibration
        Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi','det_eta', 'angles', 'markers' ])




        distance = 9 #max distance between two markers


#Fun the calibration function that uses leastsq to find parameters
        found_parameters, trajectories, abs_error = M.calibration_parameters(trajectories, SOD, ODD, n_angles,
                                                                         pixel_size, 
                                                                         det_size, n_markers_selected, 
                                                                         distance, n_rounds, n_seeds = 10, radius = 30,
                                                                         plot = 0, 
                                                                         show_all = 2, 
                                                                         n_iterations_step1 = n_iter_optimisation_1,
                                                                         n_iterations_step2 = n_iter_optimisation_2)
    
        distance_to_real = found_parameters.markers - marker_locations
    
        squared_dist = np.sum((distance_to_real)**2, axis=1)
        dist = np.sqrt(squared_dist)
    
        max_dist.append(dist.max())
        av_dist.append((dist.mean()))
    
        cost_list.append(0.5*sum(abs_error**2))
        print('original parameters', geom_parameters)
        print('found parameters', found_parameters)
        M.write_geom(save_path+'geom_param_%s_seed%s'%(sigma,seed), geom_parameters)
        M.write_geom(save_path+'geom_param_found_%s_seed%s'%(sigma,seed), found_parameters)
    
        s = np.array([0, -SOD, 0])
        measured_values, n_markers = M.trajectories_to_values(trajectories, det_size)
        total_cost = M.res_all(M.pack_variables(geom_parameters), n_angles, s, measured_values, pixel_size, n_markers)
        print('total cost with original geometry and disturbed trajectories', 0.5*sum(total_cost**2))
        print('total cost with found geometry', sum(abs_error**2)*0.5)
        cost_list_original.append(0.5*sum(total_cost**2))

#%%
sigmas = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]#
seeds = [0,1,2,3,4,5,6,7,8,9]

av_loc_corr = np.zeros((len(sigmas),len(seeds)))

for seed in seeds:
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        orig_geom = M.read_geom(save_path+'geom_param_%s_seed%s'%(sigma,seed))
        found_geom =M.read_geom(save_path+'geom_param_found_%s_seed%s'%(sigma,seed))
        angle_bias = (found_geom.angles-orig_geom.angles).mean()
        R = M.rotation_matrix_z(-angle_bias)

        mag_orig = (SOD+ODD)/SOD
        mag_found = (abs(found_geom.source[1])+found_geom.det[1])/abs(found_geom.source[1])

        found_markers_corrected = (R.dot(found_geom.markers.T).T)
        found_markers_corrected[:,2] = found_markers_corrected[:,2]-found_geom.det[2]*(1/mag_found)
        found_markers_corrected *= (mag_found/mag_orig)
        squared_dist = np.sum((found_markers_corrected-orig_geom.markers)**2, axis=1)
        #print(found_markers_corrected-orig_geom.markers)
        error = np.sqrt(squared_dist)
        av_loc_corr[i,seed] = error.mean()


#%%
sigma = 2
fig, ax = plt.subplots(figsize=(8,5))

plt.boxplot(av_loc_corr.T, whis=(0.,100.), patch_artist=True, boxprops=dict(facecolor="lightblue"), medianprops = dict(color = "blue", linewidth = 1))
plt.xticks([1,2,3,4,5,6,7,8,9,10,11], sigmas)
#plt.xticks(np.arange(10, len(percentages_missing), 10))
plt.ylim((0,0.2))
plt.xlabel('standard deviation of added noise (pixels)')
plt.ylabel('average error in marker locations (mm)')
plt.savefig(save_path+'6feb_boxplot.pdf', dpi = 300)

