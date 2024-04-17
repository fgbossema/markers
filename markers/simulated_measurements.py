#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:32:37 2021

@author: bossema
"""
import markers as M
import autograd.numpy as np
import pandas as pd


def simulate_data(SOD, ODD, n_rounds, n_angles, n_markers, marker_locations, det_size, pixel_size, geom_parameters = None):
    if geom_parameters is None:
        geom_parameters = M.standard_initial(SOD, ODD, n_rounds, n_angles, n_markers, radius = 0, marker_locations = marker_locations)

   
    simulated_measurement = np.asarray(M.markers_all_projections(geom_parameters, pixel_size))
    
    x = float(det_size[0]/2)
    y = float(det_size[1]/2)
  
    positions_frame = pd.DataFrame()
    positions_frame['frame'] = np.repeat(np.arange(0,n_angles),n_markers)
    positions_frame['x'] = np.around((simulated_measurement[:,:,0]+x).flatten(),2)
    positions_frame['y'] = np.around((y-simulated_measurement[:,:,1]).flatten(),2)
    positions_frame['particle'] = np.tile(np.arange(n_markers),n_angles)
    
    positions_frame = positions_frame.drop(positions_frame[positions_frame['x']>det_size[0]].index)
    positions_frame = positions_frame.drop(positions_frame[positions_frame['x']<0].index)
    
    return positions_frame, geom_parameters

#%%

