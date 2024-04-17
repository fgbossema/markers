#import time
import sys
#math, os
import numpy as np
#from math import sqrt
#import scipy
from scipy.spatial import distance
#import pickle
import skimage 
import skimage.io as io
#from skimage import util
#from skimage import img_as_ubyte
from skimage.morphology import dilation #opening, closing, 
from skimage.feature import canny, peak_local_max
from skimage.transform import hough_circle #, hough_circle_peaks
from skimage.filters import unsharp_mask
import pandas as pd
np.warnings.filterwarnings('ignore')

############## bgn show; for display only....
import matplotlib.pyplot as plt
#from matplotlib.patches import Circle
import matplotlib.cm as cm
#import matplotlib

def press(event):
    print (event.key)
    if event.key == 'escape': sys.exit(0)

def show1(title, image, pntL=None, save_path = None):
            fig, ax = plt.subplots(1, 1)
            fig.canvas.mpl_connect('key_press_event', press)

            #shape = image.shape

            #ax.set_title(title +"  ("+(str)(shape[1])+'x'+(str)(shape[0])+")")
            ax.imshow(image, interpolation='nearest', cmap=cm.gray)
            if pntL is not None:
                for blob in pntL:
                    #print (blob)
                    (x, y), r, a = blob
                    c = plt.Circle((x, y), 2, color="red", linewidth=1, fill=True)
                    ax.add_patch(c)
                    c = plt.Circle((x, y), r, color="yellow", linewidth=2, fill=False)
                    ax.add_patch(c)
            plt.tight_layout()
            if save_path:
               plt.savefig(save_path+'detect.png', dpi = 300)

            plt.show()
################### end show
############## bgn Hough space compute + list administration
def inBlobList (blob, blobsL): 
    (xx,yy),rr,aa = blob
    for (x,y),r,a in blobsL:
        dist = distance.euclidean (np.array ((xx, yy)), np.array ((x, y)))
        if dist <= rr or dist <= r : 
            if a < aa:
                print ("REMOVE", blob,  (x,y),r,a)
                blobsL.remove(((x,y),r,a))
                return False
            else:
                return True
    return False

def getHoughPnts (centers, radii, accums, alo, ahi):
    # Select the most prominent circles
    #for idx in np.argsort(accums)[::-1][:num_peaks]:
    blobs = []
    for idx in np.argsort(accums)[::-1][:]:
        y, x = centers[idx]
        r = radii[idx]
        a = accums[idx]
        if a <= alo: continue
        if a >  ahi: continue
        if inBlobList (((x,y),r,a), blobs): continue
        blobs.append (((x,y), r,a))

    blobsL = []
    for blob in blobs:
        if inBlobList (blob, blobsL): continue
        blobsL.append (blob)
    return sorted (blobsL, key=lambda r:r[0][1]) # sort on Y

def getHoughPnts2 (centers, radii, accums, n_peaks):
    # Select the most prominent circles
    #for idx in np.argsort(accums)[::-1][:num_peaks]:
    blobs = []
    for idx in np.argpartition(accums, -n_peaks)[-n_peaks:]:
        y, x = centers[idx]

        r = radii[idx]
        a = accums[idx]
        if inBlobList (((x,y),r,a), blobs): continue
        blobs.append (((x,y), r,a))

    # Prune overlapping points wrt accum-val
    blobsL = []
    for blob in blobs:
        if inBlobList (blob, blobsL): continue
        blobsL.append (blob)
    return sorted (blobsL, key=lambda r:r[0][1]) # sort on Y

def getHoughSpaces (image, num_peaks, sigma, minR, maxR, low_threshold= None, high_threshold= None, test= False) :
    #imageS = unsharp_mask(image, radius=minR, amount=1)
    #edges = canny(imageS, sigma=1)
    if low_threshold is None:
            drange = image.max() - image.min()
            low_threshold = 0.1*drange
            high_threshold = 0.2*drange

    edges = canny(image, sigma=sigma,low_threshold=low_threshold, high_threshold=high_threshold) 
    
    #original: 0.6-0.9, 0.2-0.5
    if test:
        plt.imshow(edges)
    #edges = closing(edges)
    edges = dilation(edges)
    #edges = canny(imageS, sigma=3,low_threshold=.1, high_threshold=.3,use_quantiles=False) 

    hough_radii = np.arange(minR, maxR) 
    hough_res = hough_circle(edges, hough_radii)
    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract num_peaks circles
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)
        #print('radius')

    return centers , radii , accums 
############## end Hough space compute + list administration


def markers_blobdetector(infiles, minR, maxR, low_threshold = None, high_threshold = None, sigma = 3, invert = False, save_path = None, test = False):
    total_list = []
    
    if low_threshold is None:
        drange = infiles[0].max() - infiles[0].min()
        low_threshold = 0.1*drange
    if high_threshold is None:
        drange = infiles[0].max() - infiles[0].min()
        high_threshold = 0.2*drange
        
    for i, image in enumerate(infiles):
        #io.imread(infile, False)
        # for 
        if invert is True: image = skimage.util.invert(image)

        
        num_peaks = 32 

        centers, radii, accums = getHoughSpaces (image, num_peaks, sigma, minR, maxR, low_threshold, high_threshold, test)
        accums = np.around (accums, 4)

        blobsL = []


        # find sufficient number of points to search for lines
        blobsL = getHoughPnts (centers, radii, accums, 0.9, 1.)
            
        #blobsL = getHoughPnts2 (centers, radii, accums, n_markers+5)
        #print(blobsL)
        if test:
            print('circle radii: \n', np.asarray(blobsL)[:,1])
        #print (infile, len(blobsL), " circles")
        for blob in blobsL: total_list.append((blob[0][0], blob[0][1], blob[1], i)) 
        
        if len(blobsL) ==0:
            print('No blobs found in frame', i)
        if test:
            show1 (image, skimage.util.invert(image), blobsL, save_path)
        if i%100 == 0:
            print('working on frame ', i)
            show1 (image, skimage.util.invert(image), blobsL)
            
    dataframe = pd.DataFrame(total_list, columns = ['x', 'y', 'radius','frame'])
    if save_path is not None:
        dataframe.to_csv(str(save_path+'markerpositions_blobdetector'))
        
    marker_projected_size = np.asarray(dataframe['radius']).mean()
    print('marker projected size', marker_projected_size)#TODO maak deze kloppend
    return dataframe, marker_projected_size
    
