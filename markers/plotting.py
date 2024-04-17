#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:30:48 2019

@author: bossema
"""

import markers as M
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import trackpy as tp
import pandas as pd
#%% Plotting functions
#TODO gaat er nu nog vanuit dat de measured values een lange array is, terwijl dat nu een lijst van lijsten met markerpunten is
def to_world(detector_values,u,v,det):
    """Turns measured values on the detector into real world coordinates, 
    only for forward projected by equations. 
    (not measured values as given by trackpy)
    """
    
    real_coord = []
    for frame_nr in np.arange(len(detector_values)):
        for blob in detector_values[frame_nr]:
            #if blob.marker_id != None:
            real_coord.append(det + u*blob[0]+v*blob[1])
        
    return np.asarray(real_coord)


def plot_setup(geom_parameters,pixel_size):
    """
    Plots the setup, s, d, markers and u, v. 
    """   
    
    (s,d,u_angle,v_angle,eta_angle,theta,P)= geom_parameters
    u,v = M.detector_vectors(u_angle,v_angle,eta_angle, pixel_size)
    
    measured_values = M.markers_projection(0,geom_parameters, pixel_size)
    measured_values_world = to_world([measured_values],u,v,d)
    
    #Start plotting
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111, projection='3d')
    
    #set axes
    left = s[1]
    right = d[1]
    ax.set_xlim(left, right)
   
    ax.set_ylim(left,right)

    ax.set_zlim(-200,200)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #plot the markers
    ax.scatter(xs = P[:,0], ys=P[:,1], zs =P[:,2], zdir='z', s=20, c=None, depthshade=True)
    #plot source and detector points
    ax.scatter(s[0],s[1],s[2],c = 'r')
    ax.scatter(d[0],d[1],d[2])
    #plot the X-axis
    ax.plot([s[0],d[0]],[s[1],d[1]],[s[2],d[2]],c = 'k')
    #plot the u and v vector
    ax.plot(np.array([d[0],(d+u)[0]]),np.array([d[1],(d+u)[1]]),np.array([d[2],(d+u)[2]]), c = 'y')
    ax.plot(np.array([d[0],(d+v)[0]]),np.array([d[1],(d+v)[1]]),np.array([d[2],(d+v)[2]]), c = 'g')
    #calculate the forward projection and plot lines

    ax.scatter(measured_values_world[:,0],measured_values_world[:,1],measured_values_world[:,2])
    plt.close()
    
    #%%
    
def plot_forward_measured(angle_index, found_parameters, pixel_size, trajectories, det_size, path):
    
    measured_values = M.trajectories_to_values(trajectories, det_size)
    plt.figure()
    angle = found_parameters.angles[angle_index]
    forward = M.markers_projection(angle,found_parameters,pixel_size)  

    measured = measured_values[0][angle_index]

    for values in forward:
        plt.scatter(values[0],values[1], color = 'green')
    for values in measured:
        plt.scatter(values.pos[0],values.pos[1], color = 'red')
    #for i in range(len(forward)):
        #values_m = measured[i]
        #values_f = forward[i]
        #plt.scatter((values_m.pos[0]+values_f[0])/2,(values_m.pos[1]+values_f[1])/2, color = 'yellow')
        
    plt.savefig(str(path + 'forward_error_%d.png'%angle_index), bbox_inches='tight')
    #plt.close()
    

    
def plot_result_trajectories(save_path, found_parameters, trajectories, det_pixel_size, det_size, y_lim, x_lim):
    
    t1 = trajectories
    t2 = M.create_trajectories(found_parameters, det_size, det_pixel_size) 
   
    if det_size[0] == det_size[1]:
        figsize=(16, 16)
    else: 
        figsize = (16,10)
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim)
    plt.scatter(t1['x'], t1['y'], s = 0.5, c = 'red', label = 'original trajectories')
    plt.scatter(t2['x'], t2['y'], s = 0.5, c = 'green', label = 'resulting trajectories')
    plt.legend()
    
    fig.savefig(str(save_path + 'trajectories_after_scatter.png'), bbox_inches='tight')
    
    return t2

def forward_trajectories(save_path, found_parameters, det_pixel_size, det_size, real_axes = True):
    forward_proj = np.asarray(M.markers_all_projections(found_parameters, det_pixel_size))
    
    x = float(det_size[0]/2)
    y = float(det_size[1]/2)
    
    n_angles = found_parameters.angles.shape[0]
    n_markers = found_parameters.markers.shape[0]
    
    t2 = pd.DataFrame()
    t2['frame'] = np.repeat(np.arange(0,n_angles),n_markers)
    t2['x'] = np.around((forward_proj[:,:,0]+x).flatten(),2)
    t2['y'] = np.around((y-forward_proj[:,:,1]).flatten(),2)
    t2['particle'] = np.tile(np.arange(n_markers)+n_markers,n_angles)
    
    x_lim = [0,det_size[0]]
    y_lim = [0,det_size[1]]
    
    if real_axes == True:
        if x_lim == y_lim:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = plt.subplots(figsize=(16, 10))
            
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        
    plot = tp.plot_traj(t2,label = True)
    
    fig = plot.get_figure()
    fig.savefig(str(save_path + 'trajectories_after.png'), bbox_inches='tight')
    plt.close()
    return t2
    
