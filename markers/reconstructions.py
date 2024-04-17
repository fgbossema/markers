#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:54:58 2019

@author: bossema
"""

import markers as M
import numpy as np
from flextomo import projector
import flexdata as fd
from flexdata import data
import os
import toml    
from collections import namedtuple
from flexcalc import analyze, process

#For inpainting section
import skimage 
import skimage.io
from skimage.restoration.inpaint import inpaint_biharmonic
from skimage.draw import disk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#%%
Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi','det_eta', 'angles', 'markers' ])



#%% This function calculates the vectors that discribe the geometry
# from the geometry parameters. These can be used by Astra for a reconstruction.
def proj_geom_vectors(geom_parameters, pixel_size, rotate = 0.0):
    """
    This function turns the system parameters vector V into a matrix with the setup parameters (source position, detector position etc) 
    for each projection angle. The matrix size is n_angles by 12 system parameters. This is the format of projection geometry
    required by the astra 'cone_vec' geometry. 
    """

    (u,v) = M.detector_vectors(geom_parameters.det_theta,geom_parameters.det_phi,geom_parameters.det_eta, pixel_size)
    n_angles = geom_parameters.angles.shape[0]
    proj_geom_vectors = np.zeros([n_angles,12])
    
    for i in np.arange(n_angles):
        angle = geom_parameters.angles[i]
        proj_geom_vectors[i,:] = np.concatenate(M.rotate_setup(angle - np.radians(rotate),geom_parameters.source,geom_parameters.det,u,v))
        
    return proj_geom_vectors


def make_geom(geom_parameters, pixel_size, rotate):
    ODD = geom_parameters.det[1]
    SOD = np.abs(geom_parameters.source[1])
    mag = (SOD+ODD)/SOD 
    img_pixel = pixel_size/mag
    
    #for flexbox the second dimension is the projection angles.
    astra_vectors = proj_geom_vectors(geom_parameters, pixel_size, rotate)
    geom = fd.geometry.circular(src2obj = 0, det2obj = 0, det_pixel = pixel_size, img_pixel = img_pixel, ang_range = (0, 360), unit = 'mm')
    
    geom.from_astra_cone_vec(astra_vectors)
    
    return geom

def recon_markers(geom_parameters, proj, pixel_size, rotate = 0.0, sirt_iter = 5,fdk = 1, sirt = 1, img_pixel = None, vol_size = None, vol_tra = None, beam_hardening = False, compound = 'AlSi', padding = None, save_path = None):
    
    """Makes a reconstruction based on the parameters found using minimisation of the residual."""

    geom = make_geom(geom_parameters, pixel_size, rotate)
    #proj = proj.transpose([1,0,2]) #for flexbox the second dimension is the projection angles.
    transpose = [1,0,2]
    updown = True
    proj = data.flipdim(proj, transpose, updown)
    
    if img_pixel == None:
        ODD = geom_parameters.det[1]
        SOD = np.abs(geom_parameters.source[1])
        mag = (SOD+ODD)/SOD 
        img_pixel = pixel_size/mag
        
    print('Reconstruction resolution', img_pixel)
    
    geom.get_vectors(proj.shape[1])
    
    if padding != None:
        proj = data.pad(proj, dim = 2, width = [padding,padding], mode = 'linear')
        #proj = data.pad(proj, dim = 1, width = [padding,padding], mode = 'linear')
    
   #FDK
    if fdk == 1:
        if vol_size == None:
            fdk_vol = projector.init_volume(proj)
        else:
            fdk_vol = np.zeros((vol_size[0], vol_size[1], vol_size[2]), dtype = 'float32')  
        #fdk_vol = np.zeros((proj.shape[0], proj.shape[2]//2, proj.shape[2]//2), dtype = 'float32')  
        print(fdk_vol.shape)
        projector.settings.bounds = [0, 10]
        projector.settings.subsets = 1
        projector.settings.sorting = 'equidistant'
        
        projector.FDK(proj, fdk_vol, geom)
        fd.display.slice(fdk_vol, bounds = [], title = 'FDK')

        if beam_hardening: 
            energy, spec = analyze.calibrate_spectrum(proj, fdk_vol, geom, compound = compound, density = 1)   
            del fdk_vol
# Beam Hardening correction based on the estimated spectrum:
# Correct data:
            print('applying beam hardening correction')
            proj_corr = process.equivalent_density(proj,  geom, energy, spec, compound = compound, density = 1)
            
            plt.imshow(proj_corr[:,0,:])
            
            print('reconstuction after correction')
            #fdk_vol_corr = projector.init_volume(proj)
            if vol_size == None:
                fdk_vol_corr = projector.init_volume(proj)
            else:
                fdk_vol_corr = np.zeros((vol_size[0], vol_size[1], vol_size[2]), dtype = 'float32')  
        
            #fdk_vol_corr = np.zeros((proj.shape[0], proj.shape[2]//2, proj.shape[2]//2), dtype = 'float32')  
            

            projector.settings.bounds = [0, 10]
            projector.settings.subsets = 1
            projector.settings.sorting = 'equidistant'
            projector.FDK(proj_corr, fdk_vol_corr, geom)
        
            fd.display.slice(fdk_vol_corr, bounds = [], title = 'FDK')
            fdk_vol = fdk_vol_corr
    # if save_path:
    #     data.write_stack(save_path + 'fdk_markercalibration_inp_bin%s_padded'%bb, 'slice', fdk_vol_markers[::-1], dim = 1)
    #     del fdk_vol
    #     fdk_vol = None
    #sirt
    if sirt == 1:
        if vol_size == None:
            vol = projector.init_volume(proj)
        else:
            vol = np.zeros((vol_size[0], vol_size[1], vol_size[2]), dtype = 'float32')  
        projector.settings.bounds = [0, 10]
        projector.settings.subsets = 10
        projector.settings.sorting = 'equidistant'
        projector.settings.update_residual = True

        projector.SIRT(proj, vol, geom, iterations = sirt_iter)

        fd.display.slice(vol, title = 'SIRT')
        
    if sirt ==0:
        vol = None
    
    if fdk ==0:
        fdk_vol = None
        
    return (fdk_vol, vol)


def recon_markers_fdk(geom_parameters, proj, pixel_size, rotate = 0.0, img_pixel = None, vol_size = None, vol_tra = None, beam_hardening = False, padding = None, save_path = None):
    
    """Makes a reconstruction based on the parameters found using minimisation of the residual."""

    geom = make_geom(geom_parameters, pixel_size, rotate)
    #proj = proj.transpose([1,0,2]) #for flexbox the second dimension is the projection angles.
    transpose = [1,0,2]
    updown = True
    proj = data.flipdim(proj, transpose, updown)
    
    if img_pixel == None:
        ODD = geom_parameters.det[1]
        SOD = np.abs(geom_parameters.source[1])
        mag = (SOD+ODD)/SOD 
        img_pixel = pixel_size/mag
        
    print('Reconstruction resolution', img_pixel)
    
    geom.get_vectors(proj.shape[1])
    
    if padding != None:
        proj = data.pad(proj, dim = 2, width = [padding,padding], mode = 'linear')
        #proj = data.pad(proj, dim = 1, width = [padding,padding], mode = 'linear')
    
   #FDK
    fdk = 1
    if fdk == 1:
        if vol_size == None:
            fdk_vol = projector.init_volume(proj)
        else:
            fdk_vol = np.zeros((vol_size[0], vol_size[1], vol_size[2]), dtype = 'float32')  
        #fdk_vol = np.zeros((proj.shape[0], proj.shape[2]//2, proj.shape[2]//2), dtype = 'float32')  
        print(fdk_vol.shape)
        projector.settings.bounds = [0, 10]
        projector.settings.subsets = 1
        projector.settings.sorting = 'equidistant'
        
        projector.FDK(proj, fdk_vol, geom)
        fd.display.slice(fdk_vol, bounds = [], title = 'FDK')
    return fdk_vol

def recon_markers_sirt(geom_parameters, proj, pixel_size, rotate = 0.0, sirt_iter = 5, img_pixel = None, vol_size = None, vol_tra = None, beam_hardening = False, padding = None, save_path = None):
    
    """Makes a reconstruction based on the parameters found using minimisation of the residual."""

    geom = make_geom(geom_parameters, pixel_size, rotate)
    #proj = proj.transpose([1,0,2]) #for flexbox the second dimension is the projection angles.
    transpose = [1,0,2]
    updown = True
    proj = data.flipdim(proj, transpose, updown)
        
    if img_pixel == None:
        ODD = geom_parameters.det[1]
        SOD = np.abs(geom_parameters.source[1])
        mag = (SOD+ODD)/SOD 
        img_pixel = pixel_size/mag
        
    print('Reconstruction resolution', img_pixel)
    
    geom.get_vectors(proj.shape[1])
    
    if padding != None:
        proj = data.pad(proj, dim = 2, width = [padding,padding], mode = 'linear')
        #proj = data.pad(proj, dim = 1, width = [padding,padding], mode = 'linear')
    
    sirt = 1
    
    if sirt == 1:
        if vol_size == None:
            vol = projector.init_volume(proj)
        else:
            vol = np.zeros((vol_size[0], vol_size[1], vol_size[2]), dtype = 'float32')  
        projector.settings.bounds = [0, 10]
        projector.settings.subsets = 10
        projector.settings.sorting = 'equidistant'
        projector.settings.update_residual = True

        projector.SIRT(proj, vol, geom, iterations = sirt_iter)

        fd.display.slice(vol, title = 'SIRT')
    return vol

def write_geom(filename, parameters):
    record = parameters._asdict()
    
    #make path if not existent
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
        
    # Save TOML to a file:
    with open(filename, 'w') as f:
        d = toml.dumps(record, encoder=toml.TomlNumpyEncoder())
        f.write(d)
        
def read_geom(path):
    var = toml.load(path)
    
    Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi'
                                       ,'det_eta', 'angles', 'markers' ])
    
    geom_parameters = Geom_param(source = np.asarray(var['source'], dtype = 'float32'), 
                                 det = np.asarray(var['det'], dtype = 'float32'), 
                                 det_theta = np.asarray(var['det_theta'], dtype = 'float32'), 
                                 det_phi = np.asarray(var['det_phi'], dtype = 'float32'), 
                                 det_eta = np.asarray(var['det_eta'], dtype = 'float32'),
                                 angles = np.asarray(var['angles'], dtype = 'float32'),
                                 markers = np.asarray(var['markers'], dtype = 'float32'))
    
    
    return geom_parameters
    
#%% Inpainting code
""" Several functions to inpaint the markers and save the projections prior to reconstruction, to avoid streaks from the metal artefacts due to the metal markers."""
def inpaint_projection(image, mask):
    return inpaint_biharmonic(image, mask)

def mark_blobs(mask, blobs, mag, radius):
    for (x,y), rad in blobs:
        rr, cc = disk ((y,x),mag*radius) 
        indices = np.where((rr> 0 )& (rr < mask.shape[0]-1) & (cc> 0) & (cc < mask.shape[1]-1))
        rr = rr[indices] #remove those indexes that lie outside the mask image
        cc = cc[indices]

        mask[rr, cc] = 1
    return mask

def inpaint(image, trajectories_frame, marker_size = 10, mag = 1.5, show = False, save_path = None):
    
    x_lim = [0,image.shape[1]]
    y_lim = [0,image.shape[0]]
    
    radius = round(marker_size*2)
    trajectories_frame = trajectories_frame[(trajectories_frame['x']>x_lim[0]) & (trajectories_frame['x']<x_lim[1]) & (trajectories_frame['y']>y_lim[0]) & (trajectories_frame['y']<y_lim[1])]
    blobs = [((trajectories_frame['x'].iloc[index], trajectories_frame['y'].iloc[index]), radius) for index in range(trajectories_frame.shape[0])]

    # mark blobs
    mask = np.zeros(image.shape, image.dtype)
    mask = mark_blobs(mask, blobs, mag, radius)
    
    imageP = inpaint_projection(image, mask).astype('float32')
    
    # if save_path:
    #     outfile = 'inpainted_image'
    #     skimage.io.imsave(save_path + outfile, imageP)
          
    if show:
        fig, axes = plt.subplots(1, 3)
        im_min, im_max = np.min(image), np.max(image)
        ax = axes.ravel()

        shape = image.shape
        ax[0].set_title('image' +"  ("+(str)(shape[1])+'x'+(str)(shape[0])+")")
        ax[0].imshow(image, interpolation='nearest', cmap=cm.gray, vmin = im_min, vmax = im_max)
        my_red_cmap = cm.Reds
        my_red_cmap.set_under(color="white", alpha=0)
        
        ax[1].set_title("mask")
        ax[1].imshow(image, interpolation='nearest', cmap=cm.gray, vmin = im_min, vmax = im_max)
        ax[1].imshow(mask.astype(int), interpolation='nearest', cmap=my_red_cmap, vmin=0.5, alpha = 0.6)

        ax[2].set_title("painted")
        ax[2].imshow(imageP, interpolation='nearest', cmap=cm.gray, vmin = im_min, vmax = im_max)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path+'inpaint.png', dpi = 300)
        plt.show()
    
    return imageP
        
     
def inpaint_all(proj, save_path, trajectories, n_proj, binn, skip, marker_size, mag, show = False):
    print('Starting inpainting of projections, this can take some time.')
    for i in np.arange(n_proj):
        
        if i%100 == 0:
            print('Inpainting projection:', i)
        image = proj[i]
        
        #Inpaint image
        imageP = inpaint(image, trajectories[trajectories['frame'] == i], marker_size, mag, show = show)
        save_directory = save_path + '/data_inpainted_binn%d_skip%d'%(binn,skip)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory) 
        save_name = save_directory + '/scan_'+str(i).zfill(6)+'.tif'
        skimage.io.imsave(save_name, imageP)
        
        
def read_inpainted(save_path, bb, skip, bb_load = 1, name = 'scan_'):
   
    proj = data.read_stack(save_path + 'data_inpainted_binn%s_skip%s'%(bb,skip), name, sample = bb_load, skip = 1, updown=False, 
                       transpose=[0,1,2])
    
    return proj

#%% FleXray recon functions
def recon_flexray(path, proj, binn = 1, skip = 1, sirt_iter = 5, fdk = 1, sirt = 1, parameters_date = 'May_2019'):
    """ Reconstructs with FleX-ray system parameters."""
    proj = proj.transpose([1,0,2]) #for flexbox the second dimension is the projection angles.
    #read flexray metadata
    geom = fd.data.read_flexraylog(path, sample = binn)
    
    if parameters_date == 'Sep_2019':
        geom['det_tan'] += 0.4
        
        geom['axs_tan'] -= 0.09
        
    if parameters_date == 'June_2020':
        print('parameters are off')
        #geom['axs_tan'] =- 0.4
        geom['det_tan'] += 0.4
        #geom['det_tan'] = -0.3
        geom['src_ort'] += 0.7
        #geom.parameters['src_ort'] += 1.1
        geom['det_roll'] -= 0.25 #added october 2020
        
        
    geom['vol_tra'] = [geom['det_ort'], geom['vol_tra'][1],geom['vol_tra'][2]]
    
    print('parameters_date:', parameters_date)
    print('geometry given by flexray', geom)
    #fd.display.slice(proj, dim = 0, title = 'Projection')
    
    #geom.parameters['det_tan'] = 1
    #FDK
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

def inpaint_all_flexray(path, name, trajectories, n_proj, binn, skip, show = False):
    print('Starting inpainting of projections, this can take some time.')
    for i in np.arange(n_proj):
        if i%100 == 0:
            print('Inpainting projection:', i)
        im_name = path + name +'/scan_'+str(i).zfill(6)+'.tif'
        image = skimage.io.imread(im_name)
        
        #Inpaint image
        imageP = inpaint(image, trajectories[trajectories['frame'] == i], show = show, save = False)
        save_path = path + name +'/data_inpainted_binn%d_skip%d'%(binn,skip)
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        
        save_name = path + name +'/data_inpainted_binn%d_skip%d'%(binn,skip) + '/scan_'+str(i).zfill(6)+'.tif'
        skimage.io.imsave(save_name, imageP)