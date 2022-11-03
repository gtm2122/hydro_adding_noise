from add_noise import make_noisy_rad
from par_rotate import rotate_img
import os
import time

if __name__=="__main__":
    nc_loc = '/mnt/Data/HydroSim_googledrive/data_ta_2d_profile0.vel0.mgrg00.s10.cs2.cv1.ptwg00.nc' # directory of .nc file
    spun_img_save_dir = '/mnt/Data/HydroSim_spin/data_ta_2d_profile0.vel0.mgrg00.s10.cs2.cv1.ptwg00/'
    noisy_rad_save_dir = '/mnt/Data/HydroSim_spin_noisy_radiographs/data_ta_2d_profile0.vel0.mgrg00.s10.cs2.cv1.ptwg00/'
    t = time.time()
    rotate_img(path=nc_loc,save_path=spun_img_save_dir) 
    # this will load the .nc file and start making spun images for ALL time points, 
    # if you want to spun images for certain elements then please modify the get_rho function
    
    print('time taken to produce spun images - ',time.time()-t)
    
    t = time.time()
    for k in os.listdir(spun_img_save_dir):
        make_noisy_rad(img_dir = spun_img_save_dir+'/'+k, save_dir=noisy_rad_save_dir+'/'+k)
    
    print('time taken to produce noisy radiographs for all the spun images - ',time.time()-t)
    
    print('total time taken to produce noisy radiographs for all the spun images in all the frames of sequence')