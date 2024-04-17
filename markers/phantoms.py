#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:17:07 2019

@author: bossema
"""

from flexdata import geometry
from flextomo import phantom
import numpy as np

def helical_markers(radius, phantom_size, n_markers, thetas = None): 
    # Create points on a circle:
    
    if thetas == None:
        thetas = np.linspace(0,360, n_markers, endpoint = False)+np.random.rand(n_markers)*10
    vect, src_tan, src_rad, orth = geometry.circular_orbit(radius, thetas)
        
    # Add axial motion:
    vrt = np.linspace(phantom_size[0], phantom_size[1], n_markers)
    vect = vect - orth * (vrt[:, None])
    
    return vect


    
def marker_volume(marker_positions, geom, shape, r= 5):
    #Use the positions given by the function helical_markers to generate a volume containing the markers. 
    n_markers = marker_positions.shape[0]

    
    marker_positions = marker_positions[:,[2,1,0]]

    vol = phantom.sphere(shape, geom, r, offset = marker_positions[0,:])
    for i in np.arange(n_markers-1):
        vol += phantom.sphere(shape, geom, r, offset = marker_positions[i+1,:])
        
    return vol