#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:48:42 2019

@author: bossema
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Register them and use in a multi-axis reconstruction.
"""
#%% Imports

from flexdata import display
from flexdata import data

from flexcalc import process
import numpy as np
import matplotlib.pyplot as plt
import markers as M
#%%
binn = 1
data_path_BM_system = '/export/scratch2/bossema/results/markers/BM/publication_sod881/recon_system_rescaled/'
data_path_BM_markers = '/export/scratch2/bossema/results/markers/BM/publication_sod881/sirt_200_markercalibration_inp_bin1_rescaled'
data_path_Getty_markers = '/export/scratch2/bossema/results/markers/GM/publication/woodblock_7oct/all_angles/sirt_200_markercalibration_inp_bin1_rescaled'
data_path_RM_markers = '/export/scratch2/bossema/results/markers/RM/Woodblock_april2021_markers/Block6/publication/sirt_200_markercalibration_inp_bin1_1350angles_rescaled'
data_path_flexray = '/ufs/bossema/results/markers/flexray/Markers12June2020/markertest_2020_highresdet/fdk_flexraycalibration_inpainted_rescaled'

save_path = '/export/scratch2/bossema/results/markers/registration/'
#%%
# First volume:
vol_BM_system = data.read_stack(data_path_BM_system,'tiff_', sample = binn, skip = binn, updown=True, transpose=[0,1,2])[100:-100, 350:1100, 350:1150]
display.slice(vol_BM_system, dim = 1,  title = 'BM system')
#display.pyqt_graph(vol_BM_system, dim = 0)
print(vol_BM_system.shape)
#%%
angle = np.radians(35)
shift = -15
R_BM = np.array([[1,0,0], [0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
T_BM = np.array([-15,-15,30])

vol_BM_system = process.affine(vol_BM_system, R_BM, T_BM)[:,215:565,:-50]

print(vol_BM_system.shape)
vol_BM_system = data.cast2type(vol_BM_system, 'float32', bounds = [0,1])

vol_BM_system = vol_BM_system[:,:, ::-1]/(vol_BM_system.max()*10)
#display.pyqt_graph(vol_BM_system, dim = 0)
#%%
# Second volume:
vol_BM_markers = data.read_stack(data_path_BM_markers,'sirt_', sample = binn, skip = binn, updown=False, transpose=[0,2,1])[:,:,:]
display.projection(vol_BM_markers, dim = 1,  title = 'BM markers')
vol_BM_markers = vol_BM_markers[3:-29,155:-135,80:-130]

#vol_BM_markers = vol_BM_markers[28:-29,167:-147,98:-148]
print(vol_BM_markers.shape)
#vol_BM_markers = np.pad(vol_BM_markers, ((0,0), (0,0),(50,50)), mode = 'constant')

#vol_BM_markers = vol_BM_markers
#display.pyqt_graph(vol_BM_markers, dim = 0)
#%%Third volume:
vol_Getty_markers = data.read_stack(data_path_Getty_markers, 'sirt', sample = binn, skip = binn, updown=True, transpose=[0,2,1])[40:-41, 141:-141, 74:-124]

display.projection(vol_Getty_markers, dim = 1,  title = 'Getty markers')
#display.pyqt_graph(vol_Getty_markers, dim = 0)
print(vol_Getty_markers.shape)
#%%
# Fourth volume:
vol_RM_markers = data.read_stack(data_path_RM_markers,'sirt', sample = binn, skip = binn, updown=True, transpose=[0,2,1])[:,::-1,::-1][4:-4,115:-115,27:-77]
display.projection(vol_RM_markers, dim = 1,  title = 'RM markers')
#display.pyqt_graph(vol_RM_markers, dim =1)

vol_RM_markers = np.pad(vol_RM_markers, ((0,0), (0,0),(50,50)), mode = 'constant')
print(vol_RM_markers.shape)
#%%Fifth volume
vol_flexray = data.read_stack(data_path_flexray,'flexray', sample = binn, skip = binn, updown=True, transpose=[0,1,2])[::-1,::-1,:]
vol_flexray = vol_flexray[88:722,270:-290,80:-80]
display.projection(vol_flexray, dim = 1,  title = 'flexray')
#display.pyqt_graph(vol_flexray, dim =1)
#[250:850,475:810,240:950]
#vol_flexray = np.pad(vol_flexray, ((30,0), (0,15),(20,20)), mode = 'constant')
print(vol_flexray.shape)

def intensity_ratio(volume):
    perc = np.percentile(volume, 99.99)
    print(perc)
    volume /= perc
    return volume
    

#%%
R_BM, T_BM = process.register_volumes(vol_BM_system, vol_BM_markers, subsamp = 1, use_moments = True, use_CG = True)
vol_BM_markers_reg = process.affine(vol_BM_markers, R_BM, T_BM)
#vol_BM_markers_reg_eq = process.equalize_intensity(vol_BM_system, vol_BM_markers_reg)
vol_BM_markers_reg_eq = intensity_ratio(vol_BM_markers_reg)

#%% Register Getty to BM:
R_Getty, T_Getty = process.register_volumes(vol_BM_system, vol_Getty_markers, subsamp = 1, use_moments = True, use_CG = True)
vol_Getty_markers_reg = process.affine(vol_Getty_markers, R_Getty, T_Getty)
#vol_Getty_markers_reg_eq = process.equalize_intensity(vol_BM_system, vol_Getty_markers_reg)
vol_Getty_markers_reg_eq = intensity_ratio(vol_Getty_markers_reg)
#%% Register RM to BM
R_RM, T_RM = process.register_volumes(vol_BM_system, vol_RM_markers, subsamp = 1, use_moments = True, use_CG = True)
vol_RM_markers_reg = process.affine(vol_RM_markers, R_RM, T_RM)
#vol_RM_markers_reg_eq = process.equalize_intensity(vol_BM_system, vol_RM_markers_reg)
vol_RM_markers_reg_eq = intensity_ratio(vol_RM_markers_reg)
#%% Register flex to BM
R_FR, T_FR = process.register_volumes(vol_BM_system, vol_flexray, subsamp = 1, use_moments = True, use_CG = True)
vol_flexray_reg = process.affine(vol_flexray, R_FR, T_FR)
#vol_flexray_reg_eq = process.equalize_intensity(vol_BM_system, vol_flexray_reg)
vol_flexray_reg_eq = intensity_ratio(vol_flexray_reg)
#%%
vol_BM_system_eq = intensity_ratio(vol_BM_system)
#R_Getty, T_Getty = process.register_volumes(vol_RM_markers, vol_Getty_markers, subsamp = 1, use_moments = True, use_CG = True)
#%%
# display.projection(vol_Getty_markers, dim = 0,  title = 'Getty Original')
# display.projection(vol_Getty_markers_reg , dim = 0,  title = 'Getty transform')
# display.projection(vol_BM_system, dim = 0, title = 'BM')
# display.projection(vol_BM_system-vol_Getty_markers_reg , dim = 0, title = 'diff')

# display.projection(vol_Getty_markers, dim = 1,  title = 'Getty Original')
# display.projection(vol_Getty_markers_reg , dim = 1,  title = 'Getty transform')
# display.projection(vol_BM_system, dim = 1, title = 'BM')
# display.projection(vol_BM_system-vol_Getty_markers_reg , dim = 1, title = 'diff')

# display.projection(vol_Getty_markers, dim = 2,  title = 'Getty Original')
# display.projection(vol_Getty_markers_reg , dim = 2,  title = 'Getty transform')
# display.projection(vol_BM_system, dim = 2, title = 'BM')
# display.projection(vol_BM_system-vol_Getty_markers_reg , dim = 2, title = 'diff')

#%%
slice_nr = 200
index = 200//binn


fig, axes = plt.subplots(3,5, figsize=(18,6), gridspec_kw={'width_ratios': [1,1 ,1,1, 1],'height_ratios': [1,1,1]})

ax = axes.ravel()
start, end = 100//binn, 250//binn

cut = 50

clim = [0, 1]
        
ax[0].set_title("BM system")
plot = ax[0].imshow(vol_BM_system[slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[0].plot(x, y, color="red", linewidth=2//binn)
ax[0].set_ylabel('pixel index')
ax[0].set_xlabel('pixel index')

ax[1].set_title('BM markers')
plot = ax[1].imshow(vol_BM_markers_reg_eq[slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[1].plot(x, y, color="red", linewidth=2//binn)
ax[1].set_ylabel('pixel index')
ax[1].set_xlabel('pixel index')

ax[2].set_title('GM markers')
plot = ax[2].imshow(vol_Getty_markers_reg_eq[slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[2].plot(x, y, color="red", linewidth=2//binn)
ax[2].set_ylabel('pixel index')
ax[2].set_xlabel('pixel index')

ax[3].set_title('RM markers')
plot = ax[3].imshow(vol_RM_markers_reg_eq[slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[3].plot(x, y, color="red", linewidth=2//binn)
ax[3].set_ylabel('pixel index')
ax[3].set_xlabel('pixel index')

ax[4].set_title('FleX-ray system')
plot = ax[4].imshow(vol_flexray_reg_eq[slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[4].plot(x, y, color="red", linewidth=2//binn)
ax[4].set_ylabel('pixel index')
ax[4].set_xlabel('pixel index')

ax[5].set_ylim((index+cut, index - cut))
ax[5].set_xlim((start-cut, end + cut))
plot = ax[5].imshow(vol_BM_system[slice_nr],cmap="gray")
ax[5].set_xticks([100,200])
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[5].plot(x, y, color="red", linewidth=2//binn)
ax[5].set_ylabel('pixel index')
ax[5].set_xlabel('pixel index')

ax[6].set_ylim((index+cut, index - cut))
ax[6].set_xlim((start-cut, end + cut))
plot = ax[6].imshow(vol_BM_markers_reg_eq[slice_nr],cmap="gray")
plot.set_clim(clim)
ax[6].set_xticks([100,200])
y = [index,index]
x = [start,end]
ax[6].plot(x, y, color="red", linewidth=2//binn)
ax[6].set_ylabel('pixel index')
ax[6].set_xlabel('pixel index')

ax[7].set_ylim((index+cut, index - cut))
ax[7].set_xlim((start-cut, end + cut))
ax[7].set_xticks([100,200])
plot = ax[7].imshow(vol_Getty_markers_reg_eq[slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[7].plot(x, y, color="red", linewidth=2//binn)
ax[7].set_ylabel('pixel index')
ax[7].set_xlabel('pixel index')

ax[8].set_ylim((index+cut, index - cut))
ax[8].set_xlim((start-cut, end + cut))
plot = ax[8].imshow(vol_RM_markers_reg_eq[slice_nr],cmap="gray")
plot.set_clim(clim)
ax[8].set_xticks([100,200])
y = [index,index]
x = [start,end]
ax[8].plot(x, y, color="red", linewidth=2//binn)
ax[8].set_ylabel('pixel index')
ax[8].set_xlabel('pixel index')

ax[9].set_ylim((index+cut, index - cut))
ax[9].set_xlim((start-cut, end + cut))
plot = ax[9].imshow(vol_flexray_reg_eq[slice_nr],cmap="gray")
plot.set_clim(clim)
ax[9].set_xticks([100,200])
y = [index,index]
x = [start,end]
ax[9].plot(x, y, color="red", linewidth=2//binn)
ax[9].set_ylabel('pixel index')
ax[9].set_xlabel('pixel index')

ylim = [0,1]
y_ticks = [0,0.4, 0.8]
ax[10].plot(np.linspace(start, end, end-start, endpoint = False), vol_BM_system[slice_nr, index, start:end],color="red")
ax[10].set_ylim(ylim)
ax[10].set_yticks(y_ticks)
ax[10].set_xlabel('pixel index')
ax[11].plot(np.linspace(start, end, end-start, endpoint = False), vol_BM_markers_reg_eq[slice_nr, index, start:end],color="red")
ax[11].set_ylim(ylim)
ax[11].set_yticks(y_ticks)

ax[11].set_xlabel('pixel index')
ax[12].plot(np.linspace(start, end, end-start, endpoint = False), vol_Getty_markers_reg_eq[slice_nr, index, start:end],color="red")
ax[12].set_ylim(ylim)
ax[12].set_yticks(y_ticks)

ax[12].set_xlabel('pixel index')
ax[13].plot(np.linspace(start, end, end-start, endpoint = False), vol_RM_markers_reg_eq[slice_nr, index, start:end],color="red")
ax[13].set_ylim(ylim)
ax[13].set_yticks(y_ticks)

ax[13].set_xlabel('pixel index')
ax[14].plot(np.linspace(start, end, end-start, endpoint = False), vol_flexray_reg_eq[slice_nr, index, start:end],color="red")
ax[14].set_ylim(ylim)
ax[14].set_yticks(y_ticks)

ax[14].set_xlabel('pixel index')

plt.tight_layout()
plt.savefig(save_path+'19mar2_slice%s_index%s_start%s_end%s.pdf'%(slice_nr, index, start, end), dpi = 300, bbox_inches='tight')
plt.show()

#%%
slice_nr =300
index = 200//binn


fig, axes = plt.subplots(3,5, figsize=(18,8), gridspec_kw={'height_ratios': [1.5,1,0.7]})

ax = axes.ravel()
start, end = 70//binn, 220//binn

cut = 50

clim = [0, 0.025]
        
ax[0].set_title("BM system")
plot = ax[0].imshow(vol_BM_system[:,slice_nr,:],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[0].plot(x, y, color="red", linewidth=2//binn)

ax[1].set_title('BM markers')
plot = ax[1].imshow(vol_BM_markers_reg_eq[:,slice_nr,:],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[1].plot(x, y, color="red", linewidth=2//binn)

ax[2].set_title('GM markers')
plot = ax[2].imshow(vol_Getty_markers_reg_eq[:,slice_nr,:],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[2].plot(x, y, color="red", linewidth=2//binn)

ax[3].set_title('RM markers')
plot = ax[3].imshow(vol_RM_markers_reg_eq[:,slice_nr,:],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[3].plot(x, y, color="red", linewidth=2//binn)

ax[4].set_title('FleX-ray system')
plot = ax[4].imshow(vol_flexray_reg_eq[:,slice_nr,:],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[4].plot(x, y, color="red", linewidth=2//binn)

ax[5].set_ylim((index+cut, index - cut))
ax[5].set_xlim((start-cut, end + cut))
plot = ax[5].imshow(vol_BM_system[:,slice_nr,:],cmap="gray")
ax[5].set_xticks([100,200])
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[5].plot(x, y, color="red", linewidth=2//binn)

ax[6].set_ylim((index+cut, index - cut))
ax[6].set_xlim((start-cut, end + cut))
plot = ax[6].imshow(vol_BM_markers_reg_eq[:,slice_nr,:],cmap="gray")
plot.set_clim(clim)
ax[6].set_xticks([100,200])
y = [index,index]
x = [start,end]
ax[6].plot(x, y, color="red", linewidth=2//binn)

ax[7].set_ylim((index+cut, index - cut))
ax[7].set_xlim((start-cut, end + cut))
ax[7].set_xticks([100,200])
plot = ax[7].imshow(vol_Getty_markers_reg_eq[:,slice_nr,:],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[7].plot(x, y, color="red", linewidth=2//binn)

ax[8].set_ylim((index+cut, index - cut))
ax[8].set_xlim((start-cut, end + cut))
plot = ax[8].imshow(vol_RM_markers_reg_eq[:,slice_nr,:],cmap="gray")
plot.set_clim(clim)
ax[8].set_xticks([100,200])
y = [index,index]
x = [start,end]
ax[8].plot(x, y, color="red", linewidth=2//binn)

ax[9].set_ylim((index+cut, index - cut))
ax[9].set_xlim((start-cut, end + cut))
plot = ax[9].imshow(vol_flexray_reg_eq[:,slice_nr,:],cmap="gray")
plot.set_clim(clim)
ax[9].set_xticks([100,200])
y = [index,index]
x = [start,end]
ax[9].plot(x, y, color="red", linewidth=2//binn)


ax[10].plot(np.linspace(start, end, end-start, endpoint = False), vol_BM_system[ index,slice_nr,start:end],color="red")
ax[10].set_ylim([0, 0.025])
ax[10].set_yticks([0,0.01,0.02])

ax[11].plot(np.linspace(start, end, end-start, endpoint = False), vol_BM_markers_reg_eq[index,slice_nr, start:end],color="red")
ax[11].set_ylim([0, 0.025])
ax[11].set_yticks([0,0.01,0.02])

ax[12].plot(np.linspace(start, end, end-start, endpoint = False), vol_Getty_markers_reg_eq[ index,slice_nr, start:end],color="red")
ax[12].set_ylim([0, 0.025])
ax[12].set_yticks([0,0.01,0.02])

ax[13].plot(np.linspace(start, end, end-start, endpoint = False), vol_RM_markers_reg_eq[ index,slice_nr, start:end],color="red")
ax[13].set_ylim([0, 0.025])
ax[13].set_yticks([0,0.01,0.02])

ax[14].plot(np.linspace(start, end, end-start, endpoint = False), vol_flexray_reg_eq[index,slice_nr, start:end],color="red")
ax[14].set_ylim([0, 0.025])
ax[14].set_yticks([0,0.01,0.02])


plt.tight_layout()
plt.savefig(save_path+'4feb_dim1_slice%s_index%s_start%s_end%s.pdf'%(slice_nr, index, start, end), dpi = 300, bbox_inches='tight')
plt.show()


#%%dim 2
slice_nr =600
index = 200//binn


fig, axes = plt.subplots(3,5, figsize=(18,8), gridspec_kw={'height_ratios': [1.5,1,0.7]})

ax = axes.ravel()
start, end = 130//binn, 270//binn

cut = 50

clim = [0, 0.025]
        
ax[0].set_title("BM system")
plot = ax[0].imshow(vol_BM_system[:,:, slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[0].plot(x, y, color="red", linewidth=2//binn)


ax[1].set_title('BM markers')
plot = ax[1].imshow(vol_BM_markers_reg_eq[:,:, slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[1].plot(x, y, color="red", linewidth=2//binn)

ax[2].set_title('GM markers')
plot = ax[2].imshow(vol_Getty_markers_reg_eq[:,:, slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[2].plot(x, y, color="red", linewidth=2//binn)

ax[3].set_title('RM markers')
plot = ax[3].imshow(vol_RM_markers_reg_eq[:,:, slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[3].plot(x, y, color="red", linewidth=2//binn)

ax[4].set_title('FleX-ray system')
plot = ax[4].imshow(vol_flexray_reg_eq[:,:, slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[4].plot(x, y, color="red", linewidth=2//binn)

ax[5].set_ylim((index+cut, index - cut))
ax[5].set_xlim((start-cut, end + cut))
plot = ax[5].imshow(vol_BM_system[:,:, slice_nr],cmap="gray")
ax[5].set_xticks([100,200])
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[5].plot(x, y, color="red", linewidth=2//binn)

ax[6].set_ylim((index+cut, index - cut))
ax[6].set_xlim((start-cut, end + cut))
plot = ax[6].imshow(vol_BM_markers_reg_eq[:,:, slice_nr],cmap="gray")
plot.set_clim(clim)
ax[6].set_xticks([100,200])
y = [index,index]
x = [start,end]
ax[6].plot(x, y, color="red", linewidth=2//binn)

ax[7].set_ylim((index+cut, index - cut))
ax[7].set_xlim((start-cut, end + cut))
ax[7].set_xticks([100,200])
plot = ax[7].imshow(vol_Getty_markers_reg_eq[:,:, slice_nr],cmap="gray")
plot.set_clim(clim)
y = [index,index]
x = [start,end]
ax[7].plot(x, y, color="red", linewidth=2//binn)

ax[8].set_ylim((index+cut, index - cut))
ax[8].set_xlim((start-cut, end + cut))
plot = ax[8].imshow(vol_RM_markers_reg_eq[:,:, slice_nr],cmap="gray")
plot.set_clim(clim)
ax[8].set_xticks([100,200])
y = [index,index]
x = [start,end]
ax[8].plot(x, y, color="red", linewidth=2//binn)

ax[9].set_ylim((index+cut, index - cut))
ax[9].set_xlim((start-cut, end + cut))
plot = ax[9].imshow(vol_flexray_reg_eq[:,:, slice_nr],cmap="gray")
plot.set_clim(clim)
ax[9].set_xticks([100,200])
y = [index,index]
x = [start,end]
ax[9].plot(x, y, color="red", linewidth=2//binn)


ax[10].plot(np.linspace(start, end, end-start, endpoint = False), vol_BM_system[ index,start:end, slice_nr],color="red")
ax[10].set_ylim([0, 0.025])
ax[10].set_yticks([0,0.01,0.02])

ax[11].plot(np.linspace(start, end, end-start, endpoint = False), vol_BM_markers_reg_eq[index,start:end, slice_nr],color="red")
ax[11].set_ylim([0, 0.025])
ax[11].set_yticks([0,0.01,0.02])

ax[12].plot(np.linspace(start, end, end-start, endpoint = False), vol_Getty_markers_reg_eq[index,start:end, slice_nr],color="red")
ax[12].set_ylim([0, 0.025])
ax[12].set_yticks([0,0.01,0.02])

ax[13].plot(np.linspace(start, end, end-start, endpoint = False), vol_RM_markers_reg_eq[ index,start:end, slice_nr],color="red")
ax[13].set_ylim([0, 0.025])
ax[13].set_yticks([0,0.01,0.02])

ax[14].plot(np.linspace(start, end, end-start, endpoint = False), vol_flexray_reg_eq[index,start:end, slice_nr],color="red")
ax[14].set_ylim([0, 0.025])
ax[14].set_yticks([0,0.01,0.02])


plt.tight_layout()
plt.savefig(save_path+'4feb_dim2_slice%s_index%s_start%s_end%s.pdf'%(slice_nr, index, start, end), dpi = 300, bbox_inches='tight')
plt.show()

#%%compare our method to 'standard' fdk
vol_Getty_markers = data.read_stack(data_path_Getty_markers, 'sirt', sample = binn, skip = binn, updown=True, transpose=[0,2,1])[40:-41, 41:-41,74:-124]
vol_Getty_fdk = data.read_stack('/export/scratch2/bossema/results/markers/GM/publication/woodblock_7oct/all_angles/fdk_equidistant_inp_bin1_500angles_1round_rescaled', 'fdk', sample = binn, skip = binn, updown=True, transpose=[0,2,1])[40:-41, 41:-41,74:-124]
vol_Getty_markers_reg = process.affine(vol_Getty_markers, R_Getty, T_Getty)
vol_Getty_markers_reg_eq = process.equalize_intensity(vol_BM_system, vol_Getty_markers_reg)

vol_Getty_fdk_reg = process.affine(vol_Getty_fdk, R_Getty, T_Getty)
vol_Getty_fdk_reg_eq = process.equalize_intensity(vol_BM_system, vol_Getty_fdk_reg)

#%%
R_Getty2, T_Getty2 = process.register_volumes(vol_Getty_markers_reg_eq , vol_Getty_fdk_reg_eq, subsamp = 1, use_moments = True, use_CG = True)
vol_Getty_fdk_reg_eq2 = process.affine(vol_Getty_fdk_reg_eq, R_Getty2, T_Getty2)

#%%
slice_nr = 200
index = 200//binn


fig, axes = plt.subplots(1,2, figsize=(7.2,7.2))

ax = axes.ravel()
start, end = 100//binn, 250//binn

cut = 50

clim = [0, 0.025]
ax[0].set_title('GM FDK')
plot = ax[0].imshow(vol_Getty_fdk_reg_eq2[slice_nr],cmap="gray")
plot.set_clim(clim)
        
ax[1].set_title("GM markers")
plot = ax[1].imshow(vol_Getty_markers_reg_eq[slice_nr],cmap="gray")
plot.set_clim(clim)

plt.savefig(save_path+'30jan3_FDKcomparison_slice%s_index%s_start%s_end%s.pdf'%(slice_nr, index, start, end), dpi = 300, bbox_inches='tight')
plt.show()

#%%
data_path_flexray = '/ufs/bossema/results/markers/flexray/Markers12June2020/markertest_2020_highresdet/fdk_flexraycalibration_inpainted_rescaled'
data_path_flexray2 = '/ufs/bossema/results/markers/flexray/Markers12June2020/markertest_2020_highresdet/fdk_flexraycalibration_rescaled'

vol_flexray = data.read_stack(data_path_flexray2,'flexray', sample = binn, skip = binn, updown=True, transpose=[0,1,2])[::-1,::-1,:][88:722,170:-190,80:-80]
vol_flexray_inp = data.read_stack(data_path_flexray,'flexray', sample = binn, skip = binn, updown=True, transpose=[0,1,2])[::-1,::-1,:][88:722,170:-190,80:-80]

vol_flexray_reg = process.affine(vol_flexray, R_FR, T_FR)
#vol_flexray_reg_eq = process.equalize_intensity(vol_BM_system, vol_flexray_reg)
vol_flexray_inp_reg = process.affine(vol_flexray_inp, R_FR, T_FR)
#vol_flexray_inp_reg_eq = process.equalize_intensity(vol_BM_system, vol_flexray_inp_reg)
display.pyqt_graph(vol_flexray_reg, dim = 0)
#%%
slice_nr = 245

fig, axes = plt.subplots(1,2, figsize=(7.2,7.2))

ax = axes.ravel()
start, end = 100//binn, 250//binn

clim = [0, 0.05]
        
ax[0].set_title("FleX-ray original")
plot = ax[0].imshow(vol_flexray_reg[slice_nr],cmap="gray")
plot.set_clim(clim)

ax[1].set_title('FleX-ray inpainted')
plot = ax[1].imshow(vol_flexray_inp_reg[slice_nr],cmap="gray")
plot.set_clim(clim)

plt.tight_layout()
plt.savefig(save_path+'30jan_inp_comparison_slice%s.pdf'%(slice_nr), dpi = 300, bbox_inches='tight')
plt.show()

#%%
data_path_GM_markers = '/export/scratch2/bossema/results/markers/GM/publication/woodblock_7oct/all_angles/sirt_200_markercalibration_inp_bin1_rescaled'
data_path_GM_markers_noinp = '/export/scratch2/bossema/results/markers/GM/publication/woodblock_7oct/all_angles/sirt_200_markercalibration_noinp_bin1_rescaled'


vol_GM_markers = data.read_stack(data_path_GM_markers,'sirt_', sample = binn, skip = binn, updown=False, transpose=[0,2,1])[40:-41, 41:-41, 74:-124]
vol_GM_markers_noinp = data.read_stack(data_path_GM_markers_noinp,'sirt_', sample = binn, skip = binn, updown=False, transpose=[0,2,1])[40:-41, 41:-41,74:-124]

vol_GM_markers_reg = process.affine(vol_GM_markers, R_Getty, T_Getty)

vol_GM_markers_noinp_reg = process.affine(vol_GM_markers_noinp, R_Getty, T_Getty)
#%%
slice_nr = 410

fig, axes = plt.subplots(1,2, figsize=(7.2,7.2))

ax = axes.ravel()
start, end = 100//binn, 250//binn

clim = [0, 0.03]
        
ax[0].set_title("GM original")
plot = ax[0].imshow(vol_GM_markers_noinp_reg[slice_nr],cmap="gray")
plot.set_clim(clim)

ax[1].set_title('GM inpainted')
plot = ax[1].imshow(vol_GM_markers_reg[slice_nr],cmap="gray")
plot.set_clim(clim)

plt.tight_layout()
plt.savefig(save_path+'30jan2_GM_inp_comparison_slice%s.pdf'%(slice_nr), dpi = 300, bbox_inches='tight')
plt.show()

#%%
#%% increasinglyslow rotation
data_path_flexray0 = '/ufs/bossema/results/markers/flexray/Markers12June2020/markertest_2020_highresdet/angle_selection/fdk_flexraycalibration_10_wedge_rescaled/'
data_path_flexray1 = '/ufs/bossema/results/markers/flexray/Markers12June2020/markertest_2020_highresdet/angle_selection/fdk_flexraycalibration_20_wedge_rescaled/'
data_path_flexray2 = '/ufs/bossema/results/markers/flexray/Markers12June2020/markertest_2020_highresdet/angle_selection/fdk_flexraycalibration_30_wedge_rescaled/'
data_path_flexray3 = '/ufs/bossema/results/markers/flexray/Markers12June2020/markertest_2020_highresdet/angle_selection/fdk_flexraycalibration_40_wedge_rescaled/'


vol_flexray0 = data.read_stack(data_path_flexray0,'flex', sample = binn, skip = binn, updown=True, transpose=[0,1,2])[::-1,::-1,:][88:722,170:-190,80:-80]

#display.pyqt_graph(vol_flexray0)

vol_flexray_reg0 = process.affine(vol_flexray0, R_FR, T_FR)
display.pyqt_graph(vol_flexray_reg0)
#del(vol_flexray0)

#%%

vol_flexray1 = data.read_stack(data_path_flexray1,'flex', sample = binn, skip = binn, updown=True, transpose=[0,1,2])[::-1,::-1,:][88:722,170:-190,80:-80]
vol_flexray_reg1 = process.affine(vol_flexray1, R_FR, T_FR)
del(vol_flexray1)


vol_flexray2 = data.read_stack(data_path_flexray2,'flex', sample = binn, skip = binn, updown=True, transpose=[0,1,2])[::-1,::-1,:][88:722,170:-190,80:-80]
vol_flexray_reg2 = process.affine(vol_flexray2, R_FR, T_FR)
del(vol_flexray2)


vol_flexray3 = data.read_stack(data_path_flexray3,'flex', sample = binn, skip = binn, updown=True, transpose=[0,1,2])[::-1,::-1,:][88:722,170:-190,80:-80]
vol_flexray_reg3 = process.affine(vol_flexray3, R_FR, T_FR)
del(vol_flexray3)

#%%
slice_nr = 300

fig, axes = plt.subplots(1,4, figsize=(15,7.2))

ax = axes.ravel()


clim = [0, 0.05]
        
ax[0].set_title("FDK 357.5 degrees")
plot = ax[0].imshow(vol_flexray_reg0[slice_nr,:,:],cmap="gray")
plot.set_clim(clim)

ax[1].set_title('FDK 355 degrees')
plot = ax[1].imshow(vol_flexray_reg1[slice_nr,:,:],cmap="gray")
plot.set_clim(clim)

ax[2].set_title('FDK 352.5 degrees')
plot = ax[2].imshow(vol_flexray_reg2[slice_nr,:,:],cmap="gray")
plot.set_clim(clim)

ax[3].set_title('FDK 350 degrees')
plot = ax[3].imshow(vol_flexray_reg3[slice_nr,:,:],cmap="gray")
plot.set_clim(clim)

plt.tight_layout()
plt.savefig(save_path+'30jan_wedge_comparison_slice%s.pdf'%(slice_nr), dpi = 300, bbox_inches='tight')
plt.show()

