import os
import numpy as np
from scipy import signal
from astropy.convolution import Gaussian2DKernel
import math
import matplotlib.pyplot as plt
import astra
import multiprocessing as mp
import scipy.ndimage as ndi
import time
import argparse
def get_direct(img,dso,dod,det_cols,det_rows,angles,detector_pixel_size=1,mac=4e-4):

    img[img<0] = 0
    vol_geom = astra.create_vol_geom(img.shape[0],img.shape[1],img.shape[2])

    proj_geom = astra.create_proj_geom('cone', 1,1,det_cols,det_rows, angles,(dso+dod)/detector_pixel_size,0); 
    # transformation so that detector pixel spacing is 1 mm
    # based on https://tomroelandts.com/articles/astra-toolbox-tutorial-reconstruction-from-projection-images-part-2
    proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    W = astra.OpTomo(proj_id)
    spun = img.transpose((2,1,0)) 
    # transposing so that detector and source are perpendicular to plane of interest
#     print(spun.dtype)
    fp = (W*(spun.reshape(-1))).reshape((det_rows,len(angles),det_cols)) # multiply by mass attenuation coeff
    fp = fp*mac
#     print(fp)
    return np.exp(-fp)
    
from numpy.random import MT19937

from numpy.random import RandomState, SeedSequence

def add_noise(args):
    # adds noise to a single direct radiographic slice
    direct=args[0]
    idx = args[1]
#     print(idx)
    rs = RandomState(MT19937(SeedSequence(123456789*idx)))

    fact = 2560/direct.shape[0]
    ### Jennie's code works for 2560x2560 sized radiographic slice, 
    ### so the direct is blown up to 2560x2560 first, after that it is downsampled
    direct = ndi.zoom(direct,(fact,fact))
    
    
    f = open('gamma_kernel.dat', 'r')
    gamma = np.genfromtxt(f)
    gamma = np.reshape(gamma, (301,301))

    # gamma = zoom(gamma, fact)

    # Load Detector Blur Kernel
    f = open('photon_kernel.dat', 'r')
    photon = np.genfromtxt(f)

    photon = np.reshape(photon, (81,81))

    # photon = zoom(photon, fact)


    # Load AZ Detector Blur Kernel
    f = open('detector_blur_az.dat', 'r')
    detector = np.genfromtxt(f)
    detector = np.reshape(detector, (201,201))
    
    gaussian_xstep = np.arange(1.,3.1,0.1)
    gaussian_ystep = np.arange(1.,3.1,0.1)
    gaussian_angle = np.arange(5.,26.,1.) * math.pi / 180.0
    xstep = (np.random.choice(gaussian_xstep,1))
    loc = np.where(gaussian_ystep == xstep)
    ystep = (np.random.choice(gaussian_ystep[loc[0][0]:],1))
    angle = (np.random.choice(gaussian_angle,1))

    gaussian_2D_kernel = Gaussian2DKernel(xstep, ystep, angle, x_size=141, y_size=141)


    imgconlv1 = signal.fftconvolve(direct, gaussian_2D_kernel, mode='same')

    imgconlv1 = imgconlv1 - imgconlv1.min()

    imgconlv = signal.fftconvolve(imgconlv1, detector, mode='same')
    scatterkernel = np.arange(10,31,1)

    kernelsize = np.random.choice(scatterkernel)
    #         print(kernelsize)
    scatter = Gaussian2DKernel(kernelsize)

    scatterlevel = np.arange(0.1,0.31,0.01)
    scatterlevelstep = np.random.choice(scatterlevel)
    #plt.imshow(scatter)
    #plt.show()
    #         print(scatter.shape)
    scatterimage = signal.fftconvolve(direct, scatter, mode='same') * scatterlevelstep

    maxsig = np.mean(np.mean(imgconlv[880:1680,880:1680]))
    level = np.arange(0.5, 1.6, 0.1)
    levelstep = np.random.choice(level)
    avsiglevel = maxsig * levelstep
    avsig2D = np.ones(2560) * avsiglevel
    #plt.imshow(imgconlv[1080:1480,1080:1480])
    #plt.show()

    m = np.shape(imgconlv)[0]
    n = np.shape(imgconlv)[1]

    x1, x2 = np.mgrid[:m, :n]
    tiltstep = np.arange(-0.000039,0.00004,0.0000039)
    a = np.random.choice(tiltstep)
    b = np.random.choice(tiltstep)
    tilt = (a*x1 + b*x2)
    tilt = np.ones((2560,2560)) + tilt - np.mean(np.mean(tilt))

    tilt = tilt * avsiglevel

    signalblurscatter = imgconlv + tilt + scatterimage
    maxtotalsig = np.max(np.max(signalblurscatter))
    normsignal = signalblurscatter/maxtotalsig

    gammastep = np.arange(39000, 50000,1000)
    gammalevel = np.random.choice(gammastep)

    gammasignal = normsignal * gammalevel
    gammanoise = (np.random.poisson(gammasignal) - gammasignal) / gammalevel
    correlatedgamma = signal.fftconvolve(gammanoise, gamma, mode='same')


    photonstep = np.arange(350,450,10)
    photonlevel = np.random.choice(photonstep)

    photonsignal = normsignal * photonlevel
    photonnoise = (np.random.poisson(photonsignal) - photonsignal) / photonlevel
    correlatedphoton = signal.fftconvolve(photonnoise, photon, mode='same')

    signalblurscatternoise = (normsignal + correlatedgamma + correlatedphoton) * maxtotalsig

    noisy_direct = signalblurscatternoise

    noisy_direct = ndi.zoom(noisy_direct,[1/fact,1/fact])
    
    return noisy_direct


def make_noisy_rad(img_dir,dso=1592,dod=488,det_cols = 672,
           det_rows=672,
         angles=np.linspace(0,np.pi,8,endpoint=False),
         detector_pixel_size=1,mac=4e-4,
         save_dir='/mnt/Data/noisy_radiographs/'):
    
    
    # outputs a single numpy file with noisy radiographic slices in the shape det_rows x num_angles x det_cols
    """    
    # img_dir = directory of 3D volume
    # dso = source to object distance in mm
    # dod = object to detector distance in mm
    # angles = array of angles in radians from which projections are being taken
    # detector_pixel_size = width of detector pixel in mm
    # mac = in mm^2/gram
    # save_dir = location to save radiograph. example -  /dir/to/your/file.npy
    """
    img = np.load(img_dir) # loads numpy file of spun image
    direct = get_direct(img,dso,dod,det_cols,det_rows,angles,detector_pixel_size,mac=4e-4) 
    # this will be of size det_cols x num_angles x det_rows    
    # apply noise to each slice parallely.
    cpu_count = 4
    index = np.arange(0,len(angles))
    batch_idx = [index[k:k+cpu_count] for k in range(0,len(angles),cpu_count)]
    
    noisy_radiograph = np.zeros_like(direct)
#     print(direct.shape)
    
    for idx_b in batch_idx:
#         print(idx_b)
        pool = mp.Pool(cpu_count)
        res = pool.map(add_noise,[(direct[:,kk,:],kk) for kk in idx_b])
        pool.close()
        pool.join()
        print(idx_b)
        for k in idx_b:
            print(k)
            noisy_radiograph[:,k,:] = res[k%cpu_count]
        
    np.save(save_dir,noisy_radiograph)        
            
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_dir',type=str,help='path of a 3D volume',default='/mnt/Data/HydroSim_spin/1.npy')
    parser.add_argument('--save_dir',type=str,help='path of a 3D volume',default='/mnt/Data/HydroSim_spin_noisy_radiographs/1.npy')
    args = parser.parse_args()
    
    t = time.time()
    
    make_noisy_rad(img_dir = args.img_dir,
         save_dir=args.save_dir)
    
    print(time.time()-t)
    
    # about 10-11 seconds to do this