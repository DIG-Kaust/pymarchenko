#!/usr/bin/env python
# coding: utf-8

# # Rayleigh-Marchenko imaging with angle gathers

# In[1]:

import os
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from multiprocessing import set_start_method
from multiprocessing import get_context
from scipy.sparse import csr_matrix, vstack
from scipy.linalg import lstsq, solve
from scipy.sparse.linalg import cg, lsqr
from scipy.signal import convolve, filtfilt
from scipy.io import loadmat

from pylops                            import LinearOperator
from pylops.utils                      import dottest
from pylops.utils.wavelets             import *
from pylops.utils.seismicevents        import *
from pylops.utils.tapers               import *
from pylops.basicoperators             import *
from pylops.signalprocessing           import *
from pylops.waveeqprocessing.mdd       import *
from pylops.waveeqprocessing.marchenko import *
from pylops.optimization.leastsquares  import *

from pymarchenko.raymarchenko import RayleighMarchenko
from pymarchenko.anglegather import AngleGather

from rayimaging_depth_level import rayimaging_depth_level


def main():
	print('OMP_NUM_THREADS', os.environ["OMP_NUM_THREADS"])
	print('MKL_NUM_THREADS', os.environ["MKL_NUM_THREADS"])
	
	inputfile = '../data/raymarchenko/input_dualsensor.npz' # choose file in testdata folder of repo

	vel = 2400.0        # velocity
	tsoff = 0.06        # direct arrival time shift source side
	troff = 0.06        # direct arrival time shift receiver side
	nsmooth = 10        # time window smoothing 
	nfmax = 300         # max frequency for MDC (#samples)
	nstaper = 11        # source/receiver taper lenght
	n_iter = 30         # iterations
	kt = 120            # portion of time axis to remove
	convolvedata = False # Apply convolution to data

	jr = 1              # subsampling in r
	js = 1              # subsampling in s
	jt = 1              # subsampling for MDD

	nalpha = 41         # number of angles in Angle gathers
	plotflag = False


	# Load input

	# In[4]:


	inputdata = np.load(inputfile)


	# Read and visualize geometry

	# In[5]:


	# Receivers
	r = inputdata['r'][:,::jr]
	nr = r.shape[1]
	dr = r[0,1]-r[0,0]

	# Sources
	s = inputdata['s'][:,::js]
	ns = s.shape[1]
	ds = s[0,1]-s[0,0]

	# Density model
	rho = inputdata['rho']
	z, x = inputdata['z'], inputdata['x']


	# Read data

	# In[6]:


	# time axis
	t = inputdata['t']
	ot, dt, nt = t[0], t[1]-t[0], len(t)

	# data
	#p = inputdata['p'][::js, :, ::jr]
	#vz = inputdata['vz'][::js, :, ::jr]

	# separated data
	d = loadmat('../data/raymarchenko/separated_data.mat')
	Vzu = d['VUP'][:,:,::js]
	Vzd = d['VDOWN'][:,:,::js]


	# In[7]:


	# remove early time
	Vzu = Vzu[kt:]
	Vzd = Vzd[kt:]
	t = t[:-kt]
	nt = len(t)


	# In[8]:


	wav = inputdata['wav']
	wav = wav / np.max(np.abs(np.fft.fft(wav))*dt)
	wav_c = np.argmax(wav)


	# Convolve data with wavelet (optional)

	# In[9]:


	if convolvedata:
		Vzu = dt * np.apply_along_axis(convolve, 0, Vzu, wav, mode='full')
		Vzu = Vzu[wav_c:][:nt]
		Vzd = dt * np.apply_along_axis(convolve, 0, Vzd, wav, mode='full')
		Vzd = Vzd[wav_c:][:nt]


	# In[10]:


	# move receivers to integration axis
	Vzu = Vzu.transpose(2, 1, 0) # R[s, r, t]
	Vzd = Vzd.transpose(2, 1, 0) # R[s, r, t]


	# In[11]:


	fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 9))
	axs[0].imshow(Vzu[ns//2].T+Vzd[ns//2].T, cmap='gray', vmin=-1e-1, vmax=1e-1, extent=(r[0,0], r[0,-1], t[-1], t[0]))
	axs[0].set_title(r'$Vz$'), axs[0].set_xlabel(r'$x_R$'), axs[0].set_ylabel(r'$t$')
	axs[0].axis('tight')
	axs[0].set_ylim(1.5, 0)
	axs[1].imshow(Vzu[ns//2].T, cmap='gray', vmin=-1e-1, vmax=1e-1, extent=(r[0,0], r[0,-1], t[-1], t[0]))
	axs[1].set_title(r'$Vz_{up}$'), axs[1].set_xlabel(r'$x_R$')
	axs[1].axis('tight')
	axs[1].set_ylim(1.5, 0)
	axs[2].imshow(Vzd[ns//2].T, cmap='gray', vmin=-1e-1, vmax=1e-1, extent=(r[0,0], r[0,-1], t[-1], t[0]))
	axs[2].set_title(r'$Vz_{dn}$'), axs[2].set_xlabel(r'$x_R$')
	axs[2].axis('tight')
	axs[2].set_ylim(1.5, 0);


	# Define imaging domain

	# In[12]:


	#nvsx, nvsz = 81, 1
	#dvsx, dvsz = 20, 10
	#vsx = np.arange(nvsx)*dvsx + 700
	#vsz = np.arange(nvsz)*dvsz + 850

	#nvsx, nvsz = 81, 6
	#dvsx, dvsz = 20, 10
	#vsx = np.arange(nvsx)*dvsx + 700
	#vsz = np.arange(nvsz)*dvsz + 850

	#nvsx, nvsz = 11, 8
	#dvsx, dvsz = 20, 10
	#vsx = np.arange(nvsx)*dvsx + 800
	#vsz = np.arange(nvsz)*dvsz + 1040
	#igaths = [1, 8, ]       # indeces of Angle gathers

	nvsx, nvsz = 61, 40  # 60
	dvsx, dvsz = 20, 10
	vsx = np.arange(nvsx)*dvsx + 900
	vsz = np.arange(nvsz)*dvsz + 920
	igaths = [1, 10, 20, 30, 40]       # indeces of Angle gathers

	ngath = len(igaths)
	VSX, VSZ = np.meshgrid(vsx, vsz, indexing='ij')

	plt.figure(figsize=(18,9))
	plt.imshow(rho, cmap='gray', extent = (x[0], x[-1], z[-1], z[0]))
	plt.scatter(s[0, 5::10], s[1, 5::10], marker='*', s=150, c='r', edgecolors='k')
	plt.scatter(r[0, ::10],  r[1, ::10], marker='v', s=150, c='b', edgecolors='k')
	plt.scatter(VSX.ravel(), VSZ.ravel(), marker='.', s=250, c='m', edgecolors='k')
	plt.axis('tight')
	plt.xlabel('x [m]'),plt.ylabel('y [m]'),plt.title('Model and Geometry')
	plt.xlim(x[0], x[-1]);
	plt.show()


	# Perform imaging for the different depth levels
	#rayimaging_depth_level(4, nvsz, vsx, vsz, r, s, nr, dr, dvsx, nt, dt, vel, troff, tsoff, nsmooth, 
	#	                   wav[wav_c-60:wav_c+60], nfmax, igaths, nalpha, Vzd, Vzu, 1, 1, 1, True);t0 = time.time()
	#results = [rayimaging_depth_level(ivsz, nvsz, vsx, vsz, r, s, nr, dr, dvsx, nt, dt, 
	#	                              vel, troff, tsoff, nsmooth, wav[wav_c-60:wav_c+60], nfmax, 
	#	                              igaths, nalpha, Vzd, Vzu, 1, 1, 1, False) for ivsz in range(nvsz)]
	#print('Elapsed time (mins): ', (time.time()- t0) / 60.)
	# In[13]:
	

	nprocs = 4 #mp.cpu_count()
	print("%d workers available" %nprocs)
	set_start_method("spawn")

	#with mp.Pool(processes=nprocs) as pool:
	with get_context("spawn").Pool(processes=nprocs) as pool:
		t0 = time.time()
		results = pool.starmap(rayimaging_depth_level, [(ivsz, nvsz, vsx, vsz, r, s, nr, dr, dvsx, nt, dt, 
				                                         vel, troff, tsoff, nsmooth, wav[wav_c-60:wav_c+60], nfmax, 
				                                         igaths, nalpha, Vzd, Vzu, 1, n_iter, 10, False) for ivsz in range(nvsz)])
		print('Elapsed time (mins): ', (time.time()- t0) / 60.)

	irtm = np.vstack([results[ivsz][0] for ivsz in range(nvsz)])
	imck = np.vstack([results[ivsz][1] for ivsz in range(nvsz)])
	artm = np.concatenate([results[ivsz][2][:, np.newaxis, :] for ivsz in range(nvsz)], axis=1)
	amck = np.concatenate([results[ivsz][3][:, np.newaxis, :] for ivsz in range(nvsz)], axis=1)


	# Visualize the stardard single-scattering (eg RTM) image and the Marchenko image
	fig, axs = plt.subplots(1, 3, figsize=(17, 6))
	axs[0].imshow(rho, cmap='gray_r', interpolation='sinc', extent=(x[0], x[-1], z[-1], z[0]))
	axs[0].axis('tight')
	axs[0].set_xlim(vsx[0], vsx[-1])
	axs[0].set_ylim(vsz[-1], vsz[0])
	axs[1].imshow(irtm, cmap='gray_r', vmin=-1e6, vmax=1e6, interpolation='sinc', 
		          extent=(vsx[0], vsx[-1], vsz[-1], vsz[0]))
	axs[1].axis('tight')
	axs[2].imshow(imck, cmap='gray_r', vmin=-5e-1, vmax=5e-1, interpolation='sinc', 
		          extent=(vsx[0], vsx[-1], vsz[-1], vsz[0]))
	axs[2].axis('tight');


	# And the same for angle gathers
	fig, axs = plt.subplots(1, 2, figsize=(20, 4))
	axs[0].imshow(artm.transpose(0, 2, 1).reshape(ngath*nalpha, nvsz).T, cmap='gray_r', vmin=-5e9, vmax=5e9,
		          interpolation='sinc')
	axs[0].axis('tight')
	axs[0].set_title('Gathers RTM')
	axs[1].imshow(amck.transpose(0, 2, 1).reshape(ngath*nalpha, nvsz).T, cmap='gray_r', vmin=-5e2, vmax=5e2,
		          interpolation='sinc')
	axs[1].set_title('Gathers Mck')
	axs[1].axis('tight');

	plt.show()


if __name__ == '__main__':
    main()


