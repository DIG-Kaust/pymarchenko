"""
7. Imaging with Rayleigh-Marchenko fields
==========================================
This example shows how to perform target-oriented imaging with
Rayleigh-Marchenko wavefields using the
:py:class:`pymarchenko.imaging.MarchenkoImaging` routine.

"""
# sphinx_gallery_thumbnail_number = 3
# pylint: disable=C0103
import warnings
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve
from pymarchenko.imaging import MarchenkoImaging

warnings.filterwarnings('ignore')
plt.close('all')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

###############################################################################
# Let's start by defining some input parameters and loading the test data

# Input parameters
inputfile = '../testdata/raymarchenko/input.npz'

vel = 2400.0         # velocity
tsoff = 0.06         # direct arrival time shift source side
troff = 0.06         # direct arrival time shift receiver side
nsmooth = 10         # time window smoothing
nfmax = 500          # max frequency for MDC (#samples)
niter = 30           # iterations
convolvedata = True  # apply convolution to data
igaths = [21, 31,
          41, 51,
          61]        # indeces of angle gathers
nalpha = 41          # number of angles in Angle gathers

inputdata = np.load(inputfile)

# Receivers
r = inputdata['r']
nr = r.shape[1]
dr = r[0, 1]-r[0, 0]

# Sources
s = inputdata['s']
ns = s.shape[1]
ds = s[0, 1]-s[0, 0]

# Imaging domain
nvsx, nvsz = 81, 40
dvsx, dvsz = 20, 10
vsx = np.arange(nvsx)*dvsx + 700
vsz = np.arange(nvsz)*dvsz + 650
VSX, VSZ = np.meshgrid(vsx, vsz, indexing='ij')

# Density model
rho = inputdata['rho']
z, x = inputdata['z'], inputdata['x']

# Time axis
t = inputdata['t']
ot, dt, nt = t[0], t[1]-t[0], len(t)

# Wavelet
wav = inputdata['wav']
wav_c = np.argmax(wav)

# Reflection data (R[s, r, t]) and subsurface fields
Vzu = inputdata['Vzu']
Vzd = inputdata['Vzd']

# Convolve data with wavelet
if convolvedata:
    Vzu = dt * np.apply_along_axis(convolve, -1, Vzu, wav, mode='full')
    Vzu = Vzu[..., wav_c:][..., :nt]
    Vzd = dt * np.apply_along_axis(convolve, -1, Vzd, wav, mode='full')
    Vzd = Vzd[..., wav_c:][..., :nt]

plt.figure(figsize=(18,9))
plt.imshow(rho, cmap='gray', extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(s[0, 5::10], s[1, 5::10], marker='*', s=150, c='r', edgecolors='k')
plt.scatter(r[0, ::10],  r[1, ::10], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(VSX.ravel(), VSZ.ravel(), marker='.', s=250, c='m', edgecolors='k')
plt.axis('tight')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Model and Geometry')
plt.xlim(x[0], x[-1])
plt.tight_layout()


##############################################################################
# Let's perform imaging now
iss, imck, ass, amck = \
    MarchenkoImaging(vsx, vsz, r, s, dr, dt, nt, vel,
                     tsoff, nsmooth, wav, wav_c, nfmax, igaths, nalpha, 1,
                     dict(Vzu=Vzu, Vzd=Vzd), kind='rmck', niter=niter, nproc=4)

##############################################################################
# Finally let's display the images

fig, axs = plt.subplots(1, 2, figsize=(17, 6))
axs[0].imshow(iss, cmap='gray', vmin=-1e6, vmax=1e6, interpolation='sinc')
axs[0].axis('tight')
axs[1].imshow(imck, cmap='gray', vmin=-5e-1, vmax=5e-1, interpolation='sinc')
axs[1].axis('tight')
fig.tight_layout()

##############################################################################
# And the angle gathers

ngath = len(igaths)
fig, axs = plt.subplots(1, 2, figsize=(20, 4))
axs[0].imshow(ass.transpose(0, 2, 1).reshape(ngath * nalpha, nvsz).T,
              cmap='gray', vmin=-5e9, vmax=5e9,
              interpolation='sinc')
axs[0].axis('tight')
axs[0].set_title('Gathers RTM')
axs[1].imshow(amck.transpose(0, 2, 1).reshape(ngath * nalpha, nvsz).T,
              cmap='gray', vmin=-5e2, vmax=5e2,
              interpolation='sinc')
axs[1].set_title('Gathers Mck')
axs[1].axis('tight')
fig.tight_layout()