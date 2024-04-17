# -*- coding: utf-8 -*-

"""


This is the main 3D module. It provides functions that are needed to use the 
leastsq from scipy to estimate the parameters of an X-ray setup 
from the projection data, based on measured markers in the radiograph.
 
The parameters that define the setup are: 
    * source: the source postion (sx, sy, sz)
    * det: the detector position (dx, dy, dz)
    * det_theta: the tilt of the detector, first out of plane rotation. 
    * det_phi: tilt in the other direction, second out of plane rotation. 
    * det_eta: inplane detector rotation. 
    
The parameters that define the projection angles and marker positions:
    * angles: an array of projection angles (the first is fixed to be 0)
    * markers: the matrix containing 
        x,y,z positions of all the markers (size: (n_markers, 3))
    
The module contains the following functions:
    For easy use of the variables in different formats:
    * pack_variables: turns the parameters into one array of variables 
        as this is the format scipy's leastsq needs as input
    * full_parameters: the inverse of pack_variables, turns a list of 
    variables into the tuple with separate system parameters
    
    For rotation of the setup:
    * rotate_setup: Rotates s,d and u around the center of rotation (0,0)
    * rotate_axis: Rotates a vector around a given axis
    * detector_vectors: Calculates the actual detector vectors u and v 
        from the given angles and detector pixel size 
        (norm of u and v is detector pixel size, 
        u and v are perpendicular and span the detector plane)
    
    Forward projecting and residual:
    * markers_projection: Calculates the detector values 
        (on which pixel the marker is measured) for one projection angle
    * markers_all projections: Uses markers_projection to create a list of 
        detector values for all projection angles
        
    Residuals and jacobian:
    * residuals_single: Calculates the distance from the current forward 
        projection of one marker to the measured value on the detector. 
    * marker_selection: selects the n_selected markers that whose residual 
        is lowest
    * res_all: for every measured marker in the selected markerset, 
        the corresponding residual is calculated
    * jacobian_all: using autograd, the jacobian is calculated 
        for the same points as in residuals_all
    
    Minimalisation:
    * standard_initial: initializes a set of standard initial parameters
    * choose_seed: test a given number of random seeds to generate the 
        standard initial and outputs the one with lowest cost
    * calibration_parameters: Minimizes the residual using scipy's leastsq
 
"""

import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import jacobian
from scipy.optimize import least_squares
import markers as M
from collections import namedtuple
from itertools import combinations
import trackpy as tp
import os   

def make_directory_structure(save_path):
    # Make path if does not exist:
    if not os.path.exists(save_path):
        os.makedirs(save_path)  

indices_used = []


#%% Packing and unpacking the variables

Geom_param = namedtuple('Geom_param', [ 'source', 'det','det_theta','det_phi','det_eta', 'angles', 'markers' ])


def pack_variables(geom_parameters):
    """
    This functions packs the variables into one array, 
    turning it into the format of the variable input of scipy's leastsq function.
    """
    variables = np.hstack(((geom_parameters.det,
                            geom_parameters.det_theta,
                            geom_parameters.det_phi,
                            geom_parameters.det_eta,
                            geom_parameters.angles[1:],
                            geom_parameters.markers.flatten())))

    return variables


def full_parameters(variables,n_angles,s):
    """
    This functions unpacks the variables from one array to the 
    Tuple containing the separate parameters of the setup. 
    """
  
    geom_parameters = Geom_param(source = np.array([s[0],s[1],s[2]]), 
                                 det = variables[0:3], 
                                 det_theta = variables[3], 
                                 det_phi = variables[4], 
                                 det_eta = variables[5],
                                 angles = np.concatenate((np.array([0]),variables[int(6):int(n_angles-1+6)])),
                                 markers = np.reshape(variables[int(n_angles-1+6):],(-1,3)))

    return geom_parameters
#%% In this block all the rotation functions for rotating the whole setup 
#and the detector (inplane and out of plane) are defined. 
    
def rotation_matrix_z(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    #counter clockwise, corresponds to clockwise rotation table rotation
    R = np.array([[cos,-sin,0], [sin, cos,0],[0,0,1]])
    return R

def rotate_setup(angle,source,det,u,v):
    """
    Rotates source and detector around (0,0) by angle. Output are the rotated (s,d,u,v).
    """
    R = rotation_matrix_z(angle)
    return(R@source,R@det,R@u,R@v)

def rotate_axis(x,axis,angle): 
    """
    Calculates the vector that results from rotating x around axis by angle, by righthand rule. 
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """

    axis = axis/(np.linalg.norm(axis))

    x_new = x*np.cos(angle)+np.cross(axis,x)*np.sin(angle)+axis*(axis@x*(1-np.cos(angle)))
    return(x_new)

def detector_vectors(theta, phi, eta, pixel_size):
    '''
    This function calculates the detector vectors u and v. 
    u and v (without tilt or rotation) are vectors along z and y axis.
    input:
        *det_theta: determines detector tilt around z-axis
        *det_phi: determines detector tilt around u-axis
        *det_eta: determines in-plane rotation of the detector plane
    '''
    u = np.array([1,0,0]) #norm of u,v is pixel size
    v = np.array([0,0,1])
    
    #rotate u around the z axis by angle theta
    #should be -theta for clockwise rotation?
    Rz = rotation_matrix_z(theta)
    #print(Rz)
    u = np.dot(Rz,u)
    #print(u)
    #rotate v around axis u
    v = rotate_axis(v,u, phi)
    
    axis = np.cross(u,v) #this is the normal of the plane
    u = rotate_axis(u,axis,eta)
    v = rotate_axis(v,axis,eta)
    #print('u',u,'v',v)
    
    u = u*pixel_size
    v = v*pixel_size

    return (u,v)
#%% For functionality, not used further anymore.
        
def markers_projection(angle, geom_parameters, pixel_size):
    """
    For one projection angle calculate the forward projection. 
    Output is an array of the values as measured on the detector in pixels. 
    
    The equivalent function in Mathematica markertest3D.nb is:
    3D:
    ((s + ((Cross[u, v].(d - s))/(Cross[u, v].(p - s))) (p - s)) - d).Transpose[{u, v}]
    (http://geomalgorithms.com/a05-_intersect-1.html)
    """
    
    #Calculate detector vectors from angles
    (u,v) = detector_vectors(geom_parameters.det_theta, geom_parameters.det_phi, geom_parameters.det_eta, pixel_size)
    #Rotate the setup
    (s,d,u,v) = rotate_setup(angle, geom_parameters.source, geom_parameters.det,u,v)
    
    normal_uv = np.cross(u,v)

    detector_values = np.empty((geom_parameters.markers.shape[0],2))
    
    for i, marker_pos in enumerate(geom_parameters.markers):
    #calculate the detector pixel the marker is measured on:
        value = np.array(((s + ((normal_uv@(d - s))/(normal_uv@(marker_pos - s)))*
                           (marker_pos - s)) - d)@np.transpose(np.array([u,v])))/(pixel_size**2)

        detector_values[i] = value
    
    return detector_values

 
def markers_all_projections(geom_parameters, pixel_size):
    """
    This function calculates the forward projection for all projection angles 
    and returns one array with all the measured values on the detector. 
    """
      
    return [markers_projection(a, geom_parameters, pixel_size) for a in geom_parameters.angles]

#%%
def res_single(variables, measured_marker, n_angles, s, pixel_size, frame_nr, marker_nr):
    """Calculates the distance from the current forward 
        projection of one marker to the measured value on the detector. """
        
    source, detector, det_theta, det_phi, det_eta, angles, markers = M.full_parameters(variables,n_angles,s)

    (u0,v0) = M.detector_vectors(det_theta, det_phi, det_eta, pixel_size)
    #Rotate the setup
    (s,d,u,v) = M.rotate_setup(angles[frame_nr], source, detector,u0,v0)
    normal_uv = np.cross(u,v)

    marker_pos = markers[marker_nr]
    blob_expected = np.array(((s + ((normal_uv@(d - s))/(normal_uv@(marker_pos - s)))*
                           (marker_pos - s)) - d)@np.transpose(np.array([u,v])))/(pixel_size**2)

    diff_vec = blob_expected - measured_marker
    return diff_vec[0]**2+diff_vec[1]**2

def marker_selection_frame(variables, n_angles, s, measured_values, pixel_size, n_markers_selected, frame_nr):    
    """Find the indices of the n_selected markers that whose residual is lowest in a single frame."""
    if len(measured_values[frame_nr]) <= n_markers_selected:
        indices_selected = np.arange(len(measured_values[frame_nr]))
            
    else:   
        blobs_res = []
        for blob in measured_values[frame_nr]: 
            if blob.marker_id is not None:

                blobs_res.append(res_single(variables, blob.pos, n_angles, s, pixel_size, frame_nr, blob.marker_id))
                    
        indices_selected = np.argsort(np.asarray(blobs_res))[0:n_markers_selected]
        
    return (np.sort(indices_selected)).astype(int)

def res_all(variables, n_angles, s, measured_values, pixel_size, n_markers_selected):
    """For every measured marker in the selected markerset, the corresponding
        residual is calculated"""
        
    residuals = []
    global indices_used
    
    param = full_parameters(variables,n_angles,s)
    n_markers = param.markers.shape[0]
   
    for frame_nr in np.arange(len(measured_values)):  
        if n_markers_selected < n_markers: 
            indices_selected = marker_selection_frame(variables, n_angles, s, measured_values, pixel_size, n_markers_selected, frame_nr)
            indices_used.append(indices_selected)
            
            for blob in [measured_values[frame_nr][i] for i in indices_selected]: 
                if blob.marker_id is not None:
                    residuals.append(res_single(variables, blob.pos, n_angles, s, pixel_size, frame_nr, blob.marker_id))
        else:
            for blob in measured_values[frame_nr]: 
                if blob.marker_id is not None:
                    residuals.append(res_single(variables, blob.pos, n_angles, s, pixel_size, frame_nr, blob.marker_id))
            
    return np.asarray(residuals).flatten()

#Note that res_all and jacobian_all need to have the exact same input because of the least_squares function. 
def jacobian_all(variables, n_angles, s, measured_values, pixel_size, n_markers_selected):
    """Using autograd, the jacobian is calculated for the same points as in residuals_all"""
    jacobian_all= []
    jacob = jacobian(res_single)

    
    param = full_parameters(variables,n_angles,s)
    n_markers = param.markers.shape[0]
    
    for frame_nr in np.arange(len(measured_values)):  

        if n_markers_selected < n_markers: #? waarom 10
            indices_selected = marker_selection_frame(variables, n_angles, s, measured_values, pixel_size, n_markers_selected, frame_nr)
            indices_used.append(indices_selected)
            
            for blob in [measured_values[frame_nr][i] for i in indices_selected]:
                if blob.marker_id is not None:        
                    jacobian_all.append(jacob(variables, blob.pos, n_angles,s, pixel_size, frame_nr, blob.marker_id))
        else:
            for blob in measured_values[frame_nr]:
                if blob.marker_id is not None:        
                    jacobian_all.append(jacob(variables, blob.pos, n_angles,s, pixel_size, frame_nr, blob.marker_id))

    return np.asarray(jacobian_all)
#%%
def merge_markers(variables, trajectories, distance = 10, max_overlap = 10, plot = False):
    locations = variables.markers
    n_markers = len(locations)
    pairs = combinations(np.arange(n_markers), 2)
    same_marker = []
    norm_tot = []
    variables_array = pack_variables(variables)
                         
    for pair in pairs:
        norm = np.linalg.norm(locations[pair[0]] - locations[pair[1]])
        if  norm <= distance:
            same_marker.append(pair)
        norm_tot.append((pair, norm))
    
    #norm_tot.sort(key = lambda i: i[1])
    #print(norm_tot)

    print('these markers are the same', same_marker)
    if len(same_marker) ==0:
        return trajectories, variables_array
    else:

                        
        same_marker_copy = same_marker.copy()
                    #If the markers are found in too many frames. The set of frames should ideally be disjunct if the particles are the same. 
        for pair in same_marker:

            intersect_frames = set(trajectories[trajectories['particle'] == pair[0]]['frame']).intersection(set(trajectories[trajectories['particle'] == pair[1]]['frame']))
            print('number of overlapping frames for pair', pair,':', len(intersect_frames))
            if len(intersect_frames) > max_overlap:
                same_marker_copy.remove(pair) #we do nothing, they overlap too much so cannot be the same marker
                print('No merge due to too many overlapping frames.')
            elif len(intersect_frames) != 0:
            #delete the second measured particle if there are frames in which both are found, so as to have just one measurement point per marker per frame
                trajectories = trajectories.drop(trajectories[(trajectories['particle'] == pair[1]) & (trajectories['frame'].isin(intersect_frames))].index)
    
    #Merge the markers for each pair in same_marker.  
        removed_labels = []
        for pair in same_marker_copy:
            if plot:
                tp.plot_traj(trajectories[trajectories['particle'].isin([pair[0],pair[1]])],label = True)
            if (pair[0] not in removed_labels) and (pair[1] not in removed_labels):
                trajectories = M.merge_trajectories(pair[1], pair[0], trajectories)
                removed_labels.append(pair[1])
                print('Merging markers %d and %d.'%(pair[0], pair[1]))
        
        removed_labels.sort()
        print(removed_labels)

        trajectories = M.rearrange_labels(trajectories)
        
        indexs = []
        
        for label in removed_labels:

            indexs.append(len(variables_array)+(label-n_markers)*3)
            indexs.append(len(variables_array)+(label-n_markers)*3+1)
            indexs.append(len(variables_array)+(label-n_markers)*3+2)
            
        variables_array = np.delete(variables_array, indexs)
    
        return trajectories, variables_array

    
#%%
#SOD
def standard_initial(SOD, ODD, n_rounds, n_angles, n_markers, radius, seed = 1, marker_locations = None):
    """Gives an initial set of parameters that can be used as initial guess in the calibration_parameters optimization."""
    angles_estimated = np.linspace(0,n_rounds*2*np.pi,n_angles, endpoint = False)
    
    if marker_locations is None:
        np.random.seed(seed)
        marker_locations =  (2*(np.random.rand(n_markers,3)-1/2))*radius
        marker_locations = marker_locations[marker_locations[:, 2].argsort()][::-1]

    geom_parameters_estimated = Geom_param(source = np.array([0,-SOD, 0]), det = np.array([0,ODD,0]), det_theta = np.radians(0.), det_phi = np.radians(0.),
                                               det_eta = np.radians(0.), angles = angles_estimated,
                                               markers = marker_locations)

    
    
    return geom_parameters_estimated

def choose_seed(n_seeds, radius, measured_values, pixel_size, SOD, ODD, n_rounds, n_angles, n_markers):
    angles_estimated = np.linspace(0,n_rounds*2*np.pi,n_angles, endpoint = False)
    
    cost_list = []
    for i in range(n_seeds):
        np.random.seed(i)
        marker_locations = (2*(np.random.rand(n_markers,3)-1/2))*radius
        marker_locations = marker_locations[marker_locations[:, 2].argsort()][::-1]

        geom_parameters_estimated = Geom_param(source = np.array([0,-SOD, 0]), det = np.array([0,ODD,0]), det_theta = np.radians(0.), det_phi = np.radians(0.),
                                               det_eta = np.radians(0.), angles = angles_estimated,
                                               markers = marker_locations)
        
        estimated_variables = pack_variables(geom_parameters_estimated)
        source_fixed = np.array([0, -SOD, 0])
        abs_error = np.abs(res_all(estimated_variables, n_angles, 
                                  source_fixed, measured_values, pixel_size, n_markers))
        cost = sum(abs_error**2)*0.5
        cost_list.append(cost)
    
    cost_list = np.asarray(cost_list)
    seed = np.where(cost_list == cost_list.min())[0][0]

    print('optimal seed:', seed)
    print('initial cost with seed:', "{:.2e}".format(cost_list[seed]))
    return seed

def standard_bounds(SOD, ODD, n_rounds, n_angles, n_markers):
    
    bounds = (np.concatenate((np.array([-50,ODD-200,-50]),np.array([-1/8*np.pi,-1/8*np.pi,-1/8*np.pi]),
                           np.repeat(0,n_angles-1),np.repeat(-1000,n_markers*3))), 
           np.concatenate((np.array([50,ODD+200,50]),np.array([1/8*np.pi,1/8*np.pi,1/8*np.pi]),
                           np.repeat(n_rounds*2*np.pi,n_angles-1),np.repeat(1000,n_markers*3))))
    
    return bounds
    
# def calibration_parameters_0(trajectories, SOD, ODD, n_angles, pixel_size, det_size, n_markers_selected, distance = 10, n_rounds = 1, bounds_given = None,  geom_parameters_estimated = None, markers_loc_given = None, n_seeds = 10, radius = 50, plot = False, show_all = 0, n_iterations_step1 = 50, n_iterations_step2 = 50):
#     """
#     Uses the least_squares function of scipy to find an estimate for the system parameters. 
#     """
   
#     measured_values, n_markers = M.trajectories_to_values(trajectories, det_size)

#     if geom_parameters_estimated is None: 
#         if markers_loc_given is None:
#             seed = choose_seed(n_seeds, radius, measured_values, pixel_size, SOD, ODD, n_rounds, n_angles, n_markers)
#             geom_parameters_estimated = M.standard_initial(SOD,ODD,n_rounds, n_angles, n_markers, radius, seed = seed)
#         else: 
#             geom_parameters_estimated = M.standard_initial(SOD,ODD,n_rounds, n_angles, n_markers,radius, seed = None, marker_locations = markers_loc_given)

#     print(geom_parameters_estimated.markers)
#     if bounds_given is None:
#         bounds = standard_bounds(SOD, ODD, n_rounds, n_angles, n_markers)
#     else: 
#         bounds = bounds_given
   
#     estimated_variables = pack_variables(geom_parameters_estimated)
#     source_fixed = np.array([0, -SOD, 0])


#     #Option in least_squares to make the scale of all parameters similar.
#     scale =np.concatenate((np.array([1,1,1]),np.array([0.1,0.1,0.1]),
#                           np.repeat(0.001,n_angles-1),np.repeat(0.1,n_markers*3)))

#     #Performs the actual minimization
#     #First perform the optimalisation with all markers for 10 iterations. 
#     print('Running first optimalisation with all markers.')
#     lsq_0 = least_squares(res_all, estimated_variables, bounds = bounds, jac = jacobian_all,
#                         args=(n_angles, source_fixed, measured_values, pixel_size, n_markers),
#                               verbose = show_all, x_scale = scale, 
#                               max_nfev = n_iterations_step1, xtol= 1e-6, gtol = 1e-6, 
#                               ftol = 1e-6)


#     found_variables_0 = full_parameters(lsq_0['x'],n_angles,source_fixed)


#     #Merging markers that get too close in 3D space, update measured values, scale and bounds
#     trajectories, estimated_variables_0  = merge_markers(found_variables_0, trajectories, distance, plot = False)
#     if plot:
#         tp.plot_traj(trajectories,label = True)


#     measured_values, n_markers_2 = M.trajectories_to_values(trajectories, det_size)
#     print(n_markers_2)

#     scale = np.concatenate((np.array([1,1,1]),np.array([0.1,0.1,0.1]),
#                            np.repeat(0.001,n_angles-1),np.repeat(0.1,n_markers_2*3)))
    

#     if bounds_given is None:
#         bounds_0 = standard_bounds(SOD, found_variables_0.det[1], n_rounds, n_angles, n_markers_2)
#     else: 
#         diff = n_markers-n_markers_2
#         if diff != 0:
#             indexs = bounds_given[0].shape-1*(np.arange(diff*3)+1)
#             bounds_0 = (np.delete(bounds_given[0],indexs), np.delete(bounds_given[1],indexs))
#         else:
#             bounds_0 = bounds_given

#     #Then perform the rest of the iterations on n_selected number of markers per frame
#     print('Running second optimalisation with reduced number of markers per frame.')
    
#     global indices_used
#     indices_used.clear()
    
   
#     lsq = least_squares(res_all, estimated_variables_0, bounds = bounds_0, jac = jacobian_all,
#                         args=(n_angles, source_fixed, measured_values, pixel_size, n_markers_selected),
#                               verbose = show_all, x_scale = scale, 
#                               max_nfev = n_iterations_step2, xtol= 1e-6, gtol = 1e-6, 
#                               ftol = 1e-6)
   
#     found_variables = lsq['x'] 

#     found_geometry = full_parameters(found_variables,n_angles,source_fixed)
    
#     abs_error = np.abs(res_all(found_variables, n_angles, 
#                                   source_fixed, measured_values, pixel_size, n_markers_selected))
#     print('sum of the error between input measured values and forward projection using found parameters', abs_error.sum())
#     print('average error',abs_error.sum()/abs_error.shape[0])
#     print('min/max error', abs_error.min(), abs_error.max())
#     print('max in frame nr', np.floor(np.where(abs_error == abs_error.max())[0][0]/10))
#     print('cost is',sum(abs_error**2)*0.5)
    
#     #plot the found setup
#     if plot:
#         M.plot_setup(found_geometry, pixel_size)

#     return found_geometry, trajectories, abs_error

def calibration_parameters(trajectories, SOD, ODD, n_angles, pixel_size, det_size, n_markers_selected_1, n_markers_selected_2, distance = 10, n_rounds = 1, bounds_given = None,  geom_parameters_estimated = None, markers_loc_given = None, n_seeds = 1, radius = 50, plot = False, show_all = 0, n_iterations_step1 = 50, n_iterations_step2 = 50):
    """
    Uses the least_squares function of scipy to find an estimate for the system parameters. 
    """
   
    measured_values, n_markers = M.trajectories_to_values(trajectories, det_size)

    if geom_parameters_estimated is None: 
        if markers_loc_given is None:
            seed = choose_seed(n_seeds, radius, measured_values, pixel_size, SOD, ODD, n_rounds, n_angles, n_markers)
            geom_parameters_estimated = M.standard_initial(SOD,ODD,n_rounds, n_angles, n_markers, radius, seed = seed)
        else: 
            geom_parameters_estimated = M.standard_initial(SOD,ODD,n_rounds, n_angles, n_markers,radius, seed = None, marker_locations = markers_loc_given)

    print(geom_parameters_estimated.markers)
    if bounds_given is None:
        bounds = standard_bounds(SOD, ODD, n_rounds, n_angles, n_markers)
    else: 
        bounds = bounds_given
   
    estimated_variables = pack_variables(geom_parameters_estimated)
    source_fixed = np.array([0, -SOD, 0])


    #Option in least_squares to make the scale of all parameters similar.
    scale =np.concatenate((np.array([1,1,1]),np.array([0.1,0.1,0.1]),
                          np.repeat(0.001,n_angles-1),np.repeat(0.1,n_markers*3)))

    #Performs the actual minimization
    #First perform the optimalisation with all markers for 10 iterations. 
    print('Running first optimalisation with %s markers.'%n_markers_selected_1)
    lsq_0 = least_squares(res_all, estimated_variables, bounds = bounds, jac = jacobian_all,
                        args=(n_angles, source_fixed, measured_values, pixel_size, n_markers_selected_1),
                              verbose = show_all, x_scale = scale, 
                              max_nfev = n_iterations_step1, xtol= 1e-6, gtol = 1e-6, 
                              ftol = 1e-6)


    found_variables_0 = full_parameters(lsq_0['x'],n_angles,source_fixed)


    #Merging markers that get too close in 3D space, update measured values, scale and bounds
    trajectories, estimated_variables_0  = merge_markers(found_variables_0, trajectories, distance, plot = False)
    if plot:
        tp.plot_traj(trajectories,label = True)


    measured_values, n_markers_2 = M.trajectories_to_values(trajectories, det_size)
    print(n_markers_2)

    scale = np.concatenate((np.array([1,1,1]),np.array([0.1,0.1,0.1]),
                           np.repeat(0.001,n_angles-1),np.repeat(0.1,n_markers_2*3)))
    

    if bounds_given is None:
        bounds_0 = standard_bounds(SOD, found_variables_0.det[1], n_rounds, n_angles, n_markers_2)
    else: 
        diff = n_markers-n_markers_2
        if diff != 0:
            indexs = bounds_given[0].shape-1*(np.arange(diff*3)+1)
            bounds_0 = (np.delete(bounds_given[0],indexs), np.delete(bounds_given[1],indexs))
        else:
            bounds_0 = bounds_given

    #Then perform the rest of the iterations on n_selected number of markers per frame
    print('Running second optimalisation with reduced number of markers per frame.')
    
    global indices_used
    indices_used.clear()
    
   
    lsq = least_squares(res_all, estimated_variables_0, bounds = bounds_0, jac = jacobian_all,
                        args=(n_angles, source_fixed, measured_values, pixel_size, n_markers_selected_2),
                              verbose = show_all, x_scale = scale, 
                              max_nfev = n_iterations_step2, xtol= 1e-6, gtol = 1e-6, 
                              ftol = 1e-6)
   
    found_variables = lsq['x'] 

    found_geometry = full_parameters(found_variables,n_angles,source_fixed)
    
    abs_error = np.abs(res_all(found_variables, n_angles, 
                                  source_fixed, measured_values, pixel_size, n_markers_selected_2))
    print('sum of the error between input measured values and forward projection using found parameters', abs_error.sum())
    print('average error',abs_error.sum()/abs_error.shape[0])
    print('min/max error', abs_error.min(), abs_error.max())
    print('max in frame nr', np.floor(np.where(abs_error == abs_error.max())[0][0]/10))
    print('cost is',sum(abs_error**2)*0.5)
    
    #plot the found setup
    if plot:
        M.plot_setup(found_geometry, pixel_size)

    return found_geometry, trajectories, abs_error

