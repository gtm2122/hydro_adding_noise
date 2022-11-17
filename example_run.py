from add_noise import make_noisy_rad
from par_rotate import rotate_img
import os
import time
import numpy as np
if __name__=="__main__":
    nc_loc = '/mnt/Data/HydroSim_googledrive/data_ta_2d_profile0.vel0.mgrg00.s10.cs2.cv1.ptwg00.nc' # location of .nc file
    spun_img_save_dir = '/mnt/Data/HydroSim_spin/data_ta_2d_profile0.vel0.mgrg00.s10.cs2.cv1.ptwg00/'
    noisy_rad_save_dir = '/mnt/Data/HydroSim_spin_noisy_radiographs/data_ta_2d_profile0.vel0.mgrg00.s10.cs2.cv1.ptwg00/'
    
    delete_spun_images = True
    
    t = time.time()
    
    
    res_imgs = rotate_img(path=nc_loc,save_path=spun_img_save_dir) 
    # this will load the .nc file and start making spun images for ALL time points, 
    # if you want to spin images for certain time points in your sequence rather than ALL time points then,
    # please modify the get_rho function so that it outputs a sequence of desriable time points.
    
    t1 = time.time()-t    
    print('time taken to produce spun images - ',t1)
    
    t = time.time()
    
    os.makedirs(noisy_rad_save_dir,exist_ok=True)

    for k in os.listdir(spun_img_save_dir):
        
        make_noisy_rad(img_dir = spun_img_save_dir+'/'+k, save_dir=noisy_rad_save_dir+'/'+k,
                        dso=1592,dod=488,det_cols = 672,
                        det_rows=672,
                        angles=np.linspace(0,np.pi,8,endpoint=False),
                        detector_pixel_size=1,mac=0.025*4e-2,proj=0)
        if delete_spun_images:
            os.remove(spun_img_save_dir+'/'+k)
    t2 = time.time()-t
    print('time taken to produce noisy radiographs for all the spun images - ',t2)
    print('total time taken - ', t1+t2)