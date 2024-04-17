# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:09:42 2019

@author: Francien
"""
#%% import necessary packages
import string
import matplotlib as mpl
import matplotlib.pyplot as plt
# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')

import trackpy as tp
import os
import markers as M

from collections import namedtuple
import pandas as pd
import warnings
import autograd.numpy as np
from pandas import read_csv

def find_markerpositions(data, save_path, n_proj, binn, skip, feature_size,  min_mass, separation, invert = True, figures = True, n_markers = 10):
    """Uses the trackpy toolbox to find the markers in each frame and link them
     together to form the trajectories every marker follows. Output is a dataframe
     with the x,y coordinates and the markernumber (called 'particle'). 
    
     """
    
    path_check = os.path.dirname(save_path)
    if not os.path.exists(path_check): 
       os.makedirs(path_check)
       
    all_features = []
    for i in np.arange(n_proj, step = skip):
        image = data[i]
        features = tp.locate(-image, feature_size, separation = separation, minmass = min_mass, invert = invert)
        if figures:
            plt.figure()
            tp.annotate(features, -image)
            plt.title('frame %s'%i)
            plt.close()
        features['frame'] = i

        if len(features) < n_markers:
            print('Less than %s markers found in frame  '%n_markers,i)
        if len(features) == 0:
            print('No features found in frame ',i)
            continue
        all_features.append(features)
        
        if i%100 == 0:
            print('Ẃorking on frame %s'%i)
        
        
    if len(all_features) > 0:
        f = pd.concat(all_features).reset_index(drop=True)
    else:  # return empty DataFrame
        warnings.warn("No maxima found in any frame.")
        return pd.DataFrame(columns=list(features.columns) + ['frame'])
    
    f.to_csv(str(save_path+'markerpositions_frame_bin%d_skip%d'%(binn,skip)))  

    
    return f
    
    
    
def find_trajectories(save_path,binn, skip, positions_frame, allowed_shift, memory, y_lim = [0,764], x_lim = [0,968], min_traj = 10, show = True, save = True):
    path_check = os.path.dirname(save_path)
    if not os.path.exists(path_check): 
       os.makedirs(path_check)
    #link the blobs
    t = tp.link_df(positions_frame, allowed_shift, memory = memory, pos_columns = ['y', 'x']) #Second entry is number of pixels a particle can move, memory keeps a particle in mind if it is not found in a few frames
# print_info = False
    t1 = tp.filter_stubs(t, min_traj) #Filters out found trajectories that last for less than min_traj frames

    #For more filtering check: http://soft-matter.github.io/trackpy/v0.3.0/tutorial/walkthrough.html
    if show:
        if x_lim == y_lim:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        plot = tp.plot_traj(t1, label = True)
        fig = plot.get_figure()
        fig.savefig(str(save_path + 'trajectories_binn%s_skip%s.png'%(binn,skip)), bbox_inches='tight', dpi = 300)

    
    #Make sure trajectories range starting with 1 in integers without gaps.
    t1 = rearrange_labels(t1)
    if save:    
        t1.to_csv(str(save_path+'trajectories.csv'))
    
    return t1

def rearrange_labels(t1):
    labels = np.sort(np.array((list(set(t1['particle'])))))
    n_markers = len(labels)
    for i in range(n_markers):
        if i not in labels:
            #print('missing label', i)
            y_array = np.zeros(shape =(len(set(t1[t1['particle']>i]['particle'])), 2))
            larger_labels = list(set(t1[t1['particle']>i]['particle']))
            for j, label in enumerate(larger_labels): 
                if label > i:
                    y_av = np.average(t1[t1['particle'] == label]['y'])
                    y_array[j,:] = [y_av, label]
            label_to_merge = int(y_array[np.argmin(y_array[:,0]),1])
            t1 = merge_trajectories(label_to_merge, i, t1)
    labels = np.sort(np.array(list(set(t1['particle']))))
            
    #sort labels: higher label for lower y
    y_array = np.zeros(shape =(len(labels), 2))

    for i, label in enumerate(labels): 
        y_av = np.average(t1[t1['particle'] == label]['y'])
        y_array[i,:] = [y_av, label]
        
    labels_sorted = y_array[np.argsort(y_array[:,0]).astype('int'),1]
   
    list_dummy = ['a%s'%i for i in range(n_markers)]
    for i in np.arange(len(labels)): 
        t1 = t1.replace( { 'particle': int(labels_sorted[i]) }, list_dummy[i])
        
    for i in np.arange(len(labels)):
        t1 = t1.replace( { 'particle': list_dummy[i] }, i)

    return t1


def merge_trajectories(t1, t2, trajectories):
    """Merges trajectories t1 and t2, by redefining 
    the marker_id associated with t1 to be t2, can also be used to rename one 
    of the trajectories to a non-existing (lower) one"""

    trajectories = trajectories.replace( { 'particle': t1 }, t2 )
  
    return trajectories

def clean_trajectory(trajectories, n_angles):
    trajectories = trajectories[trajectories['frame'].isin(np.arange(0,n_angles))]
    labels = np.sort(np.array((list(set(trajectories['particle'])))))

    for i in np.arange(len(labels)):
        if i not in labels:
            trajectories = merge_trajectories(labels[-1], i, trajectories)
    return trajectories
    
def delete_trajectory(t, trajectories, save_path = None):
    """Deletes trajectory t from the dataframe"""

    trajectories = trajectories[trajectories.particle != t]
    
    if save_path:
        plot = tp.plot_traj(trajectories,label = True)
        fig = plot.get_figure()
        fig.savefig(str(save_path + 'trajectories.png'), bbox_inches='tight')
        plt.close('all')
        
    return trajectories

def split_trajectory(t, trajectories, frame_nr):
    tot_nr = len(set(trajectories['particle']))
    trajectories.loc[(trajectories['particle']==t) & (trajectories['frame'] >= frame_nr), 'particle'] = tot_nr + 1
    return trajectories
    

def save_trajectory(trajectories, save_path):
    trajectories.to_csv(save_path)
    
    
def break_trajectories_radius(trajectories, max_radius_change, min_traj):
    
    trajectories.index.name = None
    radius_info = []
    for marker in set(trajectories['particle']):
        traj_marker = trajectories[trajectories['particle']==marker]
        traj_marker['dr'] = traj_marker['radius'].diff()
        overlap_frames = traj_marker[(abs(traj_marker['dr'])>max_radius_change)]['frame']
        if len(overlap_frames) != 0:
            radius_info.append((marker, overlap_frames))

    for split_marker in radius_info:
        marker = split_marker[0]
        for frame_nr in split_marker[1]:
            trajectories = M.split_trajectory(marker, trajectories, frame_nr)

    trajectories = tp.filter_stubs(trajectories, min_traj)
    trajectories = M.rearrange_labels(trajectories)
    
    return trajectories

    
def trajectories_to_values(trajectories, det_size, noise = 0):
    """Takes the trajectories from the dataframe and puts them in the format
    needed for the functions as written in the marker3.py project. 
    For simulation purposes it is possible to add noise here.
    """
    Blob = namedtuple('Blob', [ 'pos', 'marker_id' ])
    
    measured_values = []
    nr_frames = len(set(trajectories.frame))
    n_markers = len(set(trajectories['particle']))

    #numbers to translate the measured value with respect to the middle
    #instead of the top left corner
    x = float(det_size[0]/2)
    y = float(det_size[1]/2)


    for frame_nr in np.arange(nr_frames):
        blobs_in_frame = []
        frame_info = trajectories[trajectories['frame'] == frame_nr]

        
        frame_info = frame_info.sort_values('particle')
        ids = np.asarray(frame_info['particle'])

        #add the blobs to the list of blobs in this frame
        for blob_nr in ids:
            blob_x = np.asarray(frame_info[frame_info['particle'] == blob_nr]['x'])
            blob_y = np.asarray(frame_info[frame_info['particle'] == blob_nr]['y'])
            if noise == 1:
                blob_x += (np.random.rand(noise)-1/2)*2
                blob_y += (np.random.rand(noise)-1/2)*2
            #translate the measured value to be measured from the middle
            blob = Blob(pos = np.array([(blob_x[0]-x),(y-blob_y[0])]), marker_id = int(blob_nr))
            
            blobs_in_frame.append(blob)

        
        #add the list of blobs in this frame to the measured_values
        measured_values.append(blobs_in_frame)
        
    return measured_values, n_markers

def trajectories_to_values_simulation(trajectories, noise = 0):
    """Takes the trajectories from the dataframe and puts them in the format
    needed for the functions as written in the marker3.py project. 
    For simulation purposes it is possible to add noise here.
    """
    Blob = namedtuple('Blob', [ 'pos', 'marker_id' ])
    
    measured_values = []
    nr_frames = len(set(trajectories.frame))
    n_markers = len(set(trajectories['particle']))

    #numbers to translate the measured value with respect to the middle
    #instead of the top left corner


    for frame_nr in np.arange(nr_frames):
        blobs_in_frame = []
        frame_info = trajectories[trajectories['frame'] == frame_nr]

        
        frame_info = frame_info.sort_values('particle')
        ids = np.asarray(frame_info['particle'])

        #add the blobs to the list of blobs in this frame
        for blob_nr in ids:
            blob_x = np.asarray(frame_info[frame_info['particle'] == blob_nr]['x'])
            blob_y = np.asarray(frame_info[frame_info['particle'] == blob_nr]['y'])
            if noise == 1:
                blob_x += (np.random.rand(noise)-1/2)*2
                blob_y += (np.random.rand(noise)-1/2)*2
            #translate the measured value to be measured from the middle
            blob = Blob(pos = np.array([blob_x[0],blob_y[0]]), marker_id = int(blob_nr))
            
            blobs_in_frame.append(blob)

        
        #add the list of blobs in this frame to the measured_values
        measured_values.append(blobs_in_frame)
        
    return measured_values, n_markers

def read_trajectories(save_path_traj, plot = True):
    '''Reads in a trajectories csv from file, displays the trajectories and calculates the number of markers in these trajectories.'''
    
    trajectories = read_csv(str(save_path_traj))
    if plot:
        plot = tp.plot_traj(trajectories,label = True)
        plot.get_figure()
        plt.close('all')
    
    n_markers = len(set(trajectories['particle']))
    
    return trajectories, n_markers
    
def create_trajectories(found_parameters, det_size, det_pixel_size):
    ODD = found_parameters.det[1]
    SOD = np.abs(found_parameters.source[1])
    mag = (SOD+ODD)/SOD 


    forward_proj = np.asarray(M.markers_all_projections(found_parameters, det_pixel_size))
    n_angles = found_parameters.angles.shape[0]
    n_markers = found_parameters.markers.shape[0]

    x = float(det_size[0]/2)
    y = float(det_size[1]/2)
    t2 = pd.DataFrame()
    t2['frame'] = np.repeat(np.arange(0,n_angles),n_markers)
    t2['x'] = np.around((forward_proj[:,:,0]+x).flatten(),2)
    t2['y'] = np.around((y-forward_proj[:,:,1]).flatten(),2)
    t2['particle'] = np.tile(np.arange(n_markers),n_angles)

    return t2



def remove_crossovers(trajectories, max_distance_x, max_distance_y):
    "Removes the points in the markerpositions frame for which the x_difference is below max_distance_x and the y difference below max_distance_y."
    n_frames = len(set(trajectories['frame']))
    markers_overlap = []
    new_df = pd.DataFrame(columns = trajectories.columns)
    new_df["dx"] = ""
    new_df["dy"] = ""
    new_df.index.name = None 

    for marker in set(trajectories['particle']):
        traj_marker = trajectories[trajectories['particle']==marker]
        traj_marker['dx'] = traj_marker['x'].diff()
        traj_marker['dy'] = traj_marker['y'].diff()
        overlap_frames = traj_marker[(abs(traj_marker['dx'])<max_distance_x) & (abs(traj_marker['dy'])<max_distance_y)]
        if not overlap_frames.empty:
            markers_overlap.append(marker)
            for frame in overlap_frames['frame']:
                new_df = new_df.append(traj_marker[(traj_marker['dx']<max_distance_x) & (traj_marker['dy']<max_distance_y) & ((traj_marker['frame']<frame-n_frames) | (traj_marker['frame']>frame+n_frames))])
        else:
            new_df = new_df.append(traj_marker)
    new_df = new_df.drop(['particle', 'dx', 'dy'], axis = 1)
    print('These markers had overlap and pieces of their trajectory have been removed:', markers_overlap)
    return new_df.sort_values('frame'), markers_overlap