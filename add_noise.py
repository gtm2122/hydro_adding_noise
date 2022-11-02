import os
import numpy as np
from scipy import signal
from astropy.convolution import Gaussian2DKernel
import math
import matplotlib.pyplot as plt
import astra
import multiprocessing as mp

def get_direct(img,dso,dod,det_cols,det_rows):
    
    # img = 3D volume
    # dso = source to object distance in mm
    # dod = object to detector distance in mm
    
    proj_geom = astra.create_proj_geom('cone', 1,1,det_cols,det_rows, angles,dso,dod);
    proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    W = astra.OpTomo(proj_id)

    spun = img.transpose((2,1,0)) # transposing so that detector and source are perpendicular to plane of interest
    fp = W*(spun.reshape(-1))
    fp = fp/fp.max()
    
    return np.exp(-fp.reshape(det_rows,len(angles),det_cols))
    

def add_noise(direct):
    # adds noise to a single direct radiograph
    
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
    #     gaussian_2D_kernel = Gaussian2DKernel(xstep, ystep, angle, x_size=37, y_size=37)


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

    noisy_direct[c,:,:] = signalblurscatternoise

    c+=1
    
    noisy_direct = zoom(noisy_direct,[1,fact,fact])
    
    return noisy_direct

if __name__=="__main__":
    cpu_count = 2 # choose how many cpus to use to parallelize the spinning
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,help='path of a 3D volume',default='/mnt/Data/HydroSim_spin/1.npy')
    
    
    