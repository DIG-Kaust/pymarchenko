"""
4. Rayleigh-Marchenko redatuming
================================
This example shows how to set-up and run the
:py:class:`pymarchenko.raymarchenko.RayleighMarchenko` algorithm using
synthetic data.

"""
# sphinx_gallery_thumbnail_number = 5
# pylint: disable=C0103
import warnings
import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import convolve
from pymarchenko.raymarchenko import RayleighMarchenko

warnings.filterwarnings('ignore')
plt.close('all')

###############################################################################
# Let's start by defining some input parameters and loading the geometry

# Input parameters
inputfile = '../testdata/raymarchenko/input.npz'

vel = 2400.0         # velocity
tsoff = 0.06         # direct arrival time shift source side
troff = 0.06         # direct arrival time shift receiver side
nsmooth = 10         # time window smoothing
nfmax = 550          # max frequency for MDC (#samples)
niter = 30           # iterations
convolvedata = True  # Apply convolution to data

inputdata = np.load(inputfile)

# Receivers
r = inputdata['r']
nr = r.shape[1]
dr = r[0, 1]-r[0, 0]

# Sources
s = inputdata['s']
ns = s.shape[1]
ds = s[0, 1]-s[0, 0]

# Virtual points
vs = inputdata['vs']

# Density model
rho = inputdata['rho']
z, x = inputdata['z'], inputdata['x']

plt.figure(figsize=(10, 5))
plt.imshow(rho, cmap='gray', extent=(x[0], x[-1], z[-1], z[0]))
plt.scatter(s[0, 5::10], s[1, 5::10], marker='*', s=150, c='r', edgecolors='k')
plt.scatter(r[0, ::10], r[1, ::10], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(vs[0], vs[1], marker='.', s=250, c='m', edgecolors='k')
plt.axis('tight')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Model and Geometry')
plt.xlim(x[0], x[-1])
plt.tight_layout()

###############################################################################
# Let's now load and display the up and downgoing particle velocity data and
# subsurface fields

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

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 9))
axs[0].imshow(Vzu[ns//2].T+Vzd[ns//2].T, cmap='gray', vmin=-1e-1, vmax=1e-1,
              extent=(r[0,0], r[0,-1], t[-1], t[0]))
axs[0].set_title(r'$Vz$')
axs[0].set_xlabel(r'$x_R$')
axs[0].set_ylabel(r'$t$')
axs[0].axis('tight')
axs[0].set_ylim(1.5, 0)
axs[1].imshow(Vzu[ns//2].T, cmap='gray', vmin=-1e-1, vmax=1e-1,
              extent=(r[0,0], r[0,-1], t[-1], t[0]))
axs[1].set_title(r'$Vz_{up}$')
axs[1].set_xlabel(r'$x_R$')
axs[1].axis('tight')
axs[1].set_ylim(1.5, 0)
axs[2].imshow(Vzd[ns//2].T, cmap='gray', vmin=-1e-1, vmax=1e-1,
              extent=(r[0,0], r[0,-1], t[-1], t[0]))
axs[2].set_title(r'$Vz_{dn}$')
axs[2].set_xlabel(r'$x_R$')
axs[2].axis('tight')
axs[2].set_ylim(1.5, 0)
fig.tight_layout()

###############################################################################
# And subsurface fields

Gsub = inputdata['Gsub']
G0sub = inputdata['G0sub']

Gsub = np.apply_along_axis(convolve, 0, Gsub, wav, mode='full')
Gsub = Gsub[wav_c:][:nt]
G0sub = np.apply_along_axis(convolve, 0, G0sub, wav, mode='full')
G0sub = G0sub[wav_c:][:nt]

# Convolve reference Green's function with wavelet
if convolvedata:
    Gsub = dt * np.apply_along_axis(convolve, 0, Gsub, wav, mode='full')
    Gsub = Gsub[wav_c:][:nt]

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 6))
axs[0].imshow(Gsub, cmap='gray', vmin=-1e7, vmax=1e7,
              extent=(s[0, 0], s[0, -1], t[-1], t[0]))
axs[0].set_title('G')
axs[0].set_xlabel(r'$x_R$')
axs[0].set_ylabel(r'$t$')
axs[0].axis('tight')
axs[0].set_ylim(1.5, 0)
axs[1].imshow(G0sub, cmap='gray', vmin=-1e7, vmax=1e7,
              extent=(r[0, 0], r[0, -1], t[-1], t[0]))
axs[1].set_title('G0')
axs[1].set_xlabel(r'$x_R$')
axs[1].set_ylabel(r'$t$')
axs[1].axis('tight')
axs[1].set_ylim(1.5, 0)
fig.tight_layout()

##############################################################################
# Let's now create an object of the
# :py:class:`pymarchenko.raymarchenko.RayleighMarchenko` class and apply
# redatuming for a single subsurface point ``vs``.

# Direct arrival traveltimes
travs = np.sqrt((vs[0]-s[0])**2+(vs[1]-s[1])**2)/vel
travr = np.sqrt((vs[0]-r[0])**2+(vs[1]-r[1])**2)/vel

rm = RayleighMarchenko(Vzd, Vzu, dt=dt, dr=dr,
                       nfmax=nfmax, wav=wav, toff=troff, nsmooth=nsmooth,
                       dtype='float64', saveVt=True, prescaled=False)

f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus = \
    rm.apply_onepoint(travs, travr, G0=G0sub.T, rtm=True, greens=True,
                      dottest=True, **dict(iter_lim=niter, show=True))
g_inv_tot = -(g_inv_minus + g_inv_plus)

##############################################################################
# We can now compare the result of Rayleigh-Marchenko redatuming
# with standard redatuming
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 7))
axs[0].imshow(p0_minus.T, cmap='gray', vmin=-1e7, vmax=1e7,
              extent=(r[0, 0], r[0, -1], t[-1], -t[-1]))
axs[0].set_title(r'$p_0^-$')
axs[0].set_xlabel(r'$x_R$')
axs[0].set_ylabel(r'$t$')
axs[0].axis('tight')
axs[0].set_ylim(2, 0)
axs[1].imshow(g_inv_minus.T, cmap='gray', vmin=-1e7, vmax=1e7,
              extent=(r[0, 0], r[0, -1], t[-1], -t[-1]))
axs[1].set_title(r'$g^-$')
axs[1].set_xlabel(r'$x_R$')
axs[1].set_ylabel(r'$t$')
axs[1].axis('tight')
axs[1].set_ylim(2, 0)
axs[2].imshow(g_inv_plus.T, cmap='gray', vmin=-1e7, vmax=1e7,
              extent=(r[0, 0], r[0, -1], t[-1], -t[-1]))
axs[2].set_title(r'$g^+$')
axs[2].set_xlabel(r'$x_R$')
axs[2].set_ylabel(r'$t$')
axs[2].axis('tight')
axs[2].set_ylim(2, 0)
fig.tight_layout()

##############################################################################
# And compare the total Green's function with the directly modelled one

fig = plt.figure(figsize=(12, 7))
ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=2)
ax3 = plt.subplot2grid((1, 5), (0, 4))
ax1.imshow(Gsub / Gsub.max(), cmap='gray', vmin=-3e-1, vmax=3e-1,
           extent=(r[0, 0], r[0, -1], t[-1], t[0]))
ax1.set_title(r'$G_{true}$')
axs[0].set_xlabel(r'$x_R$')
axs[0].set_ylabel(r'$t$')
ax1.axis('tight')
ax1.set_ylim(2, 0)
ax2.imshow(g_inv_tot.T / g_inv_tot.max(), cmap='gray', vmin=-3e-1, vmax=3e-1,
           extent=(r[0, 0], r[0, -1], t[-1], -t[-1]))
ax2.set_title(r'$G_{est}$')
axs[1].set_xlabel(r'$x_R$')
axs[1].set_ylabel(r'$t$')
ax2.axis('tight')
ax2.set_ylim(2, 0)
ax3.plot(Gsub[:, ns//2] / Gsub.max() * (t ** 1.5), t, 'r', lw=5)
ax3.plot(g_inv_tot[ns//2, nt-1:] / g_inv_tot.max() * (t ** 1.5), t, 'k', lw=3)
ax3.set_ylim(1.6, 0)
fig.tight_layout()
