#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:06:57 2022

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
def BM_load(path, name, binn, skip, counterclockwise = True):
    
    proj = data.read_stack(path+name + '/' + 'dx', name + '_', sample = binn, skip = skip, updown=False, 
                       transpose=[0,1,2])
    
    if counterclockwise is True:
        print('Performing counterclockwise correction')
        proj = proj[::-1]
    
    proj = proj[:, 60//binn:-60//binn, 60//binn:-60//binn]
    
    return proj.astype('float32')

def BM_read_metadata(path, name, sample, real_sod = None, offset_corr = None):
    """
    Read the log file of BM CT scanner and return dictionaries with parameters of the scan.

    Args:
        path   (str): path to the files location
        sample (int): subsampling of the input data

    Returns:
        geometry    : circular geometry class
    """
   
    
    file1 = open(path + name +'/'  +'ScanData.txt', 'r')
    lines = file1.readlines()
    
    odd = float([i for i in lines if '#1008/2' in i][0].split()[1])
    sod = float([i for i in lines if '#1009/2' in i][0].split()[1])
    
    
    
    # Initialize geometry:
    geom = geometry.circular()
    if real_sod:
        geom['src2obj'] = real_sod
    else:
        geom['src2obj'] = sod
    geom['det2obj' ] = odd
    
    mag = (geom['src2obj']+geom['det2obj' ])/geom['src2obj']
    
    geom['det_pixel'] = 0.2*sample
    img_pixel = geom['det_pixel']/mag
    geom['img_pixel'] = img_pixel
    
    #horizontal offset correction
    #geom['axs_tan'] = 31.0049/mag
    if offset_corr:
        # geom['axs_tan'] = offset_corr/mag
        # print('correcting rot axis offset by:', geom['axs_tan'])
        geom['det_tan'] = -offset_corr
        print('correcting detector offset by:', geom['det_tan'])
    else:
        geom['axs_tan'] = 12.9187
    return geom

def BM_preprocess(proj, flat, dark):
    print('Applying dark- and flatfield, logarithm and pixel correction.')
    flat = (flat - dark)
    proj = (proj - dark) / flat

    proj = -np.log(proj).astype('float32')
    proj[np.isnan(proj)] = 0
    
    return proj
    

def BM_recon(proj, path, name, binn, rotate = 0.0, fdk = 1, sirt = 0, sirt_iter = 10, real_sod = None, offset_corr = None):
    proj = proj.transpose([1,0,2]) #for flexbox the second dimension is the projection angles.
    #read metadata
    geom = BM_read_metadata(path, name, binn, real_sod, offset_corr)
    
    geom['vol_rot'] = [rotate, 0.0, 0.0]
    
    #transpose = [1,0,2]
    #updown = True
    #proj = data.flipdim(proj, transpose, updown)

    if fdk == 1:
        projector.settings.subsets = 1
        fdk_vol = projector.init_volume(proj)
        projector.FDK(proj, fdk_vol, geom)

        fd.display.slice(fdk_vol, bounds = [], title = 'FDK')

    #sirt
    if sirt == 1:
        vol = projector.init_volume(proj)

        projector.settings.bounds = [0, 10]
        projector.settings.subsets = 10
        projector.settings.sorting = 'equidistant'

        projector.SIRT(proj, vol, geom, iterations = sirt_iter)

        fd.display.slice(vol, title = 'SIRT')
        
    if sirt == 0:
        vol = None
    
    return (geom, fdk_vol, vol)
   

    