#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:36:46 2019

@author: bossema
"""
#from flexdata import data

import autograd.numpy as np
#import imageio
import skimage
import tifffile
from flexdata import data


def rijxray_datacorrection(path, name, n_proj,n_flat, n_dark, invert = True):
    """This function performs the darkfield, flatfield and logarithm correction 
    on all the data and saves the data in a new folder called data_corrected."""
    
    flats = np.empty([n_flat,1536,1920])
    darks = np.empty([n_dark,1536,1920])  

    #Get the flatfields, and invert if necessesary
    for i in np.arange(n_flat):
        im_name = path + '/darks_flats/flat_data/frame'+str(i+1).zfill(6)+'.tif'
        image = tifffile.imread(im_name)
        if invert is True:
            image = skimage.util.invert(image)
        image = skimage.img_as_float(image)
        flats[i,:,:] = image
    
    #Average over the flatfields to get one flatfield and save
    flatfield = np.mean(flats, axis = 0)
    tifffile.imsave(path +'/darks_flats/flatfield.tif', flatfield)
    
    ##Get the darkfields, and invert if necessesary
    for i in np.arange(n_dark):
        im_name = path + '/darks_flats/dark_data/frame'+str(i+1).zfill(6)+'.tif'
        image = tifffile.imread(im_name)
        if invert is True:
            image = skimage.util.invert(image)
        image = skimage.img_as_float(image)
        darks[i,:,:] = image

    #Average over the darkfields to get one darkfield and save
    darkfield = np.mean(darks, axis = 0)
    tifffile.imsave(path +'/darks_flats/darkfield.tif', darkfield)


    for i in np.arange(n_proj):
        if i%100 == 0:
            print('Working on Frame ',i)
        im_name = path + name +'/data/frame'+str(i+1).zfill(6)+'.tif'
        image = tifffile.imread(im_name)
        #if the data is inverted, invert it back so that it is consistent with 
        #CT data
        if invert is True:
            image = skimage.util.invert(image)
            
        image = skimage.img_as_float(image)
        #dark- and flatfield correction
        image = -np.log((image-darkfield)/flatfield).astype('float32')[10:-10,10:-10]
        
        save_name = path + name +'/data_corrected/frame'+str(i+1).zfill(6)+'.tif'
        tifffile.imsave(save_name, image)



def rijxray_load(path, name, n_angles, binn, skip, counterclockwise = False):
    
 
    proj = data.read_stack(path + name +'data_corrected/','frame0', sample = binn, skip = skip, updown=False, 
                       transpose=[0,1,2])
    
    if counterclockwise is True:
        print('Performing counterclockwise correction')
        proj = proj[::-1]
    
    return proj.astype('float32')[:n_angles,:,:]

def rijkxray_load_inpainted(path, name, n_angles, binn, skip, counterclockwise = False):

        
    proj = data.read_stack(path + name +'/data_inpainted/','frame0', sample = binn, skip = skip, updown=False, 
                       transpose=[0,1,2])
    
    
    
    if counterclockwise is True:
        print('Performing counterclockwise correction')
        proj = proj[::-1]
            
    return proj.astype('float32')[:n_angles,:,:]



