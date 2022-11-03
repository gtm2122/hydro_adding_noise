import multiprocessing as mp
import numpy as np
import os
import scipy.interpolate
import numpy as np
import cv2
import argparse
import xarray as xr
import torch
import time

# currently the parallelism is in the sequence level meaning I apply the spin function to more than one sequence simultaneously,
# but a better parallelism can be acheived if applied to spin_image function in the inner loop (TODO)

def get_rho(filename,clampval=50,crop_pixel=350):
    
    Height=crop_pixel*2; Width=Height; nFrames=41;
    
    rho=np.zeros((nFrames,Height,Width,)) # Initializing the density array
    sim = xr.open_dataarray(filename)  # Reading .nc xarray file
    
#     print(sim)
    
    for i in range(nFrames):
        a=sim.isel(t=i);
        a=a[:crop_pixel,:crop_pixel] # Cropping the image to 350x350
        ar=np.concatenate((np.flipud(a),a), axis=0) # Flipping array to get the right part
        al=np.concatenate((np.flipud(np.fliplr(a)),np.fliplr(a)), axis=0) # Flipping array to get the left part
        # Combining to form a full circle from a quarter image
        rho[i]=np.concatenate((al,ar), axis=1)
            
    rho=torch.tensor(rho,dtype=torch.float64)
    rho=torch.clamp(rho, min=None, max=clampval) # Clamping value at rho

    return rho.numpy().astype(np.float32)

def spin_image(args):
    
    # input is a single image from a sequence produced by get_rho()
    if None not in args[0]:
        rho_clean = args[0]
        index = args[1]
        save_path = args[2]
            
        ### initialize empty volume
        
        vol2 = np.zeros((rho_clean.shape[1],rho_clean.shape[1],rho_clean.shape[1]),dtype=rho_clean.dtype)
        vol2[rho_clean.shape[1]//2,:,:] = rho_clean 
        # middle 1-2 plane (sometimes known as x-y plane) gets rho_clean image
        vol2[:,rho_clean.shape[1]//2,:]= rho_clean 
        # middle 0-2 plane gets rho_clean image

        spun_img = np.zeros((700,700,700))

        for cc in range(0,700):
            #interpolate concentric circles seen in the top view ie plane 0-1
            
            img = vol2[:,:,cc].copy()
            value = np.sqrt(((img.shape[0]/2)**2.0)+((img.shape[1]/2)**2.0))

            polar_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS) 
            # conversion to polar coordinates and then interpolating between the straight lines 

            new_image = polar_image.copy()

            for i in range(0,polar_image.shape[1]): 
                # interpolate column wise, it may be better to apply parallelism here rather than at the sequence level.
                locs = np.where(polar_image[:,i]>0)[0]

                if len(locs)>1:
                    int1 = scipy.interpolate.interp1d(np.where(polar_image[:,i]>0)[0],
                                                           polar_image[:,i][np.where(polar_image[:,i]>0)],
                                                        kind='linear', 
                                                        axis= -1, 
                                                        copy=True, 
                                                         bounds_error=None,
                                                        fill_value=0, 
                                                        assume_sorted=False)
                    for k in range(0,len(locs)-1):
                        start_pos = locs[k]
                        end_pos = locs[k+1]
                        new_image[start_pos:end_pos,i] = int1(np.arange(locs[k],locs[k+1]))

                    if end_pos< new_image.shape[0]-1:
                        new_image[end_pos+1:,i] = new_image[end_pos,i]

            spun_img[:,:,cc] = cv2.linearPolar(new_image,
                                              (img.shape[0]//2, img.shape[1]//2), 
                                              value, 
                                              cv2.WARP_INVERSE_MAP)
        np.save(save_path+'/'+str(index)+'.npy',spun_img)


def rotate_img(path,save_path,cpu_count=4):
    rho_seq = get_rho(filename=path)
    rho_idx = np.arange(0,len(rho_seq))
    
    batch_idx = [rho_idx[i:i+cpu_count] for i in range(0,len(rho_idx),cpu_count)]
    
    os.makedirs(save_path,exist_ok=True)
    
    result = []
    t=time.time()
#     print(batch_idx)
    for batch in batch_idx[:]:
        print(batch)
        pool = mp.Pool(cpu_count)
        batch_rho = [(rho_seq[k],k,save_path) for k in batch]
        pool.map(spin_image,batch_rho) 
        pool.close()
        pool.join()
        
        
        
if __name__=="__main__":
    
    cpu_count = 4 # choose how many cpus to use to parallelize the spinning
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,help='path of a single sequence',default='/mnt/Data/HydroSim_googledrive/data_ta_2d_profile0.vel0.mgrg00.s10.cs2.cv1.ptwg00.nc')
    parser.add_argument('--save_path',type=str,help='path of a single sequence',default='/mnt/Data/HydroSim_spin/')
    # make batch of inputs
    args = parser.parse_args()
    t = time.time()
    rotate_img(path=args.path,save_path=args.save_path,cpu_count=cpu_count)
    print(time.time()-t)
    # about 120-140 seconds to do this

    
    