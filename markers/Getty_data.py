#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:38:10 2022

@author: bossema
"""
import autograd.numpy as np
#import imageio
import skimage
import tifffile
from flexdata import data
import os 
from flexdata import geometry
import toml 

from flextomo import projector
import flexdata as fd

#%%
def Getty_load(path, name, file_name, n_proj, binn, skip, invert = True, counterclockwise = True, form = 'DICOM'):
    
    # proj = []
    # for i in np.arange(n_proj, step = skip):
    #     name_tif = path + name + file_name+ str(i+1).zfill(5) + '.tif'
    #     image = data.read_image(name_tif, sample = binn)
        
    #     if invert is True:
    #         image = skimage.util.invert(image)
    #     proj.append(image)

    # proj = np.asarray(proj)
    

        
    proj = data.read_stack(path + name, file_name, sample = binn, skip = skip, updown=False, 
                       transpose=[0,1,2], format = form, dtype = 'int16')
    if counterclockwise is True:
        print('Performing counterclockwise correction')
        proj = proj[::-1]
        
    proj = proj[:n_proj, 20//binn:-20//binn, 20//binn:-20//binn]
    
    return proj

def Getty_preprocess(projs, flat, dark):
    flat = (flat - dark)
    
    projs_processed = np.zeros(((int(len(projs)),int(projs[0].shape[0]), int(projs[0].shape[1]))), dtype = 'float32')
    
    for i, proj in enumerate(projs):
        #.mean(0)
        proj = (proj - dark) / flat
        #proj = (proj ) / flat
        proj = -np.log(proj).astype('float32')
        proj[np.where(np.isinf(proj)==True)] = 0

        projs_processed[i] = proj
    
    return projs_processed

def Getty_flatdark(path, name, file_name, bb, skip, form = 'DICOM', invert = False):
    """This function performs the darkfield, flatfield and logarithm correction 
    on all the data and saves the data in a new folder called data_corrected."""


    #Get the flatfields, and invert if necessesary

    flats = data.read_stack(path + name + 'flats/', file_name, sample = bb, skip = skip, updown=False, 
                       transpose=[0,1,2], format = form, dtype = 'int16')
        
    
        
    flats = flats[:,20//bb:-20//bb, 20//bb:-20//bb]
    flats = np.asarray(flats).astype('float32')
    
    #Average over the flatfields to get one flatfield and save
    flatfield = np.mean(flats, axis = 0)
    
    #tifffile.imsave(path +'/darks_flats/flatfield.tif', flatfield)
    
    ##Get the darkfields, and invert if necessesary

    darks = data.read_stack(path + name + 'darks/', file_name, sample = bb, skip = skip, updown=False, 
                       transpose=[0,1,2], format = form, dtype = 'int16')
        
    
        
    darks = darks[:,20//bb:-20//bb, 20//bb:-20//bb]
    darks = np.asarray(darks).astype('float32')
    
    #Average over the flatfields to get one flatfield and save
    darkfield = np.mean(darks, axis = 0)
    
    return darkfield, flatfield