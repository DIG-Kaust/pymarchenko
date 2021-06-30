"""
2. Marchenko redatuming by inversion
====================================
This example is an extended version of the original tutorial from the PyLops
documentation and shows how to set-up and run the
:py:class:`pylops.waveeqprocessing.Marchenko` inversion using synthetic data
for both single and multiple virtual points.

"""
# sphinx_gallery_thumbnail_number = 5
# pylint: disable=C0103
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve
from pylops.waveeqprocessing import MDD
from pylops.waveeqprocessing.marchenko import directwave
from pylops.utils.tapers import taper3d
from pymarchenko.marchenko import Marchenko

warnings.filterwarnings('ignore')
plt.close('all')

###############################################################################
# Let's start by defining some input parameters and loading the test data

# Input parameters
inputfile = '../testdata/marchenko/input.npz'

vel = 2400.0         # velocity
toff = 0.045         # direct arrival time shift
nsmooth = 10         # time window smoothing
nfmax = 500          # max frequency for MDC (#samples)
nstaper = 11         # source/receiver taper lenght
niter = 10           # iterations

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

# Time axis
t = inputdata['t'][:-100]
ot, dt, nt = t[0], t[1]-t[0], len(t)

# Reflection data (R[s, r, t]) and subsurface fields
R = inputdata['R'][:, :, :-100]
R = np.swapaxes(R, 0, 1) # just because of how the data was saved
taper = taper3d(nt, [ns, nr], [nstaper, nstaper], tapertype='hanning')
R = R * taper

# Subsurface fields
Gsub = inputdata['Gsub'][:-100]
G0sub = inputdata['G0sub'][:-100]
wav = inputdata['wav']
wav_c = np.argmax(wav)

Gsub = np.apply_along_axis(convolve, 0, Gsub, wav, mode='full')
Gsub = Gsub[wav_c:][:nt]
G0sub = np.apply_along_axis(convolve, 0, G0sub, wav, mode='full')
G0sub = G0sub[wav_c:][:nt]

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

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 7))
axs[0].imshow(R[0].T, cmap='gray', vmin=-1e-2, vmax=1e-2,
              extent=(r[0, 0], r[0, -1], t[-1], t[0]))
axs[0].set_title('R shot=0')
axs[0].set_xlabel(r'$x_R$')
axs[0].set_ylabel(r'$t$')
axs[0].axis('tight')
axs[0].set_ylim(1.5, 0)
axs[1].imshow(R[ns//2].T, cmap='gray', vmin=-1e-2, vmax=1e-2,
              extent=(r[0, 0], r[0, -1], t[-1], t[0]))
axs[1].set_title('R shot=%d' %(ns//2))
axs[1].set_xlabel(r'$x_R$')
axs[1].set_ylabel(r'$t$')
axs[1].axis('tight')
axs[1].set_ylim(1.5, 0)
axs[2].imshow(R[-1].T, cmap='gray', vmin=-1e-2, vmax=1e-2,
              extent=(r[0, 0], r[0, -1], t[-1], t[0]))
axs[2].set_title('R shot=%d' %ns)
axs[2].set_xlabel(r'$x_R$')
axs[2].axis('tight')
axs[2].set_ylim(1.5, 0)
fig.tight_layout()

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 6))
axs[0].imshow(Gsub, cmap='gray', vmin=-1e6, vmax=1e6,
              extent=(r[0, 0], r[0, -1], t[-1], t[0]))
axs[0].set_title('G')
axs[0].set_xlabel(r'$x_R$')
axs[0].set_ylabel(r'$t$')
axs[0].axis('tight')
axs[0].set_ylim(1.5, 0)
axs[1].imshow(G0sub, cmap='gray', vmin=-1e6, vmax=1e6,
              extent=(r[0, 0], r[0, -1], t[-1], t[0]))
axs[1].set_title('G0')
axs[1].set_xlabel(r'$x_R$')
axs[1].set_ylabel(r'$t$')
axs[1].axis('tight')
axs[1].set_ylim(1.5, 0)
fig.tight_layout()

##############################################################################
# Let's now create an object of the
# :py:class:`pylops.waveeqprocessing.Marchenko` class and apply redatuming
# for a single subsurface point ``vs``.

# Direct arrival traveltime
trav = np.sqrt((vs[0]-r[0])**2+(vs[1]-r[1])**2)/vel

MarchenkoWM = Marchenko(R, dt=dt, dr=dr, nfmax=nfmax, wav=wav,
                        toff=toff, nsmooth=nsmooth)

t0 = time.time()
f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus = \
    MarchenkoWM.apply_onepoint(trav, G0=G0sub.T, rtm=True, greens=True,
                               dottest=True, **dict(iter_lim=niter, show=True))
g_inv_tot = g_inv_minus + g_inv_plus
tone = time.time() - t0
print('Elapsed time (s): %.2f' % tone)

##############################################################################
# We can now compare the result of Marchenko redatuming via LSQR
# with standard redatuming
fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 7))
axs[0].imshow(p0_minus.T, cmap='gray', vmin=-5e5, vmax=5e5,
              extent=(r[0, 0], r[0, -1], t[-1], -t[-1]))
axs[0].set_title(r'$p_0^-$')
axs[0].set_xlabel(r'$x_R$')
axs[0].set_ylabel(r'$t$')
axs[0].axis('tight')
axs[0].set_ylim(1.2, 0)
axs[1].imshow(g_inv_minus.T, cmap='gray', vmin=-5e5, vmax=5e5,
              extent=(r[0, 0], r[0, -1], t[-1], -t[-1]))
axs[1].set_title(r'$g^-$')
axs[1].set_xlabel(r'$x_R$')
axs[1].set_ylabel(r'$t$')
axs[1].axis('tight')
axs[1].set_ylim(1.2, 0)
axs[2].imshow(g_inv_plus.T, cmap='gray', vmin=-5e5, vmax=5e5,
              extent=(r[0, 0], r[0, -1], t[-1], -t[-1]))
axs[2].set_title(r'$g^+$')
axs[2].set_xlabel(r'$x_R$')
axs[2].set_ylabel(r'$t$')
axs[2].axis('tight')
axs[2].set_ylim(1.2, 0)
fig.tight_layout()

fig = plt.figure(figsize=(12, 7))
ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=2)
ax3 = plt.subplot2grid((1, 5), (0, 4))
ax1.imshow(Gsub, cmap='gray', vmin=-5e5, vmax=5e5,
           extent=(r[0, 0], r[0, -1], t[-1], t[0]))
ax1.set_title(r'$G_{true}$')
axs[0].set_xlabel(r'$x_R$')
axs[0].set_ylabel(r'$t$')
ax1.axis('tight')
ax1.set_ylim(1.2, 0)
ax2.imshow(g_inv_tot.T, cmap='gray', vmin=-5e5, vmax=5e5,
           extent=(r[0, 0], r[0, -1], t[-1], -t[-1]))
ax2.set_title(r'$G_{est}$')
axs[1].set_xlabel(r'$x_R$')
axs[1].set_ylabel(r'$t$')
ax2.axis('tight')
ax2.set_ylim(1.2, 0)
ax3.plot(Gsub[:, nr//2]/Gsub.max(), t, 'r', lw=5)
ax3.plot(g_inv_tot[nr//2, nt-1:]/g_inv_tot.max(), t, 'k', lw=3)
ax3.set_ylim(1.2, 0)
fig.tight_layout()

##############################################################################
# Finally, we show that when interested in creating subsurface wavefields
# for a group of subsurface points the
# :py:func:`pylops.waveeqprocessing.Marchenko.apply_multiplepoints` should be
# used instead of
# :py:func:`pylops.waveeqprocessing.Marchenko.apply_onepoint`.
nvs = 51
dvsx = 20
vs = [np.arange(nvs)*dvsx + 1000, np.ones(nvs)*1060]

plt.figure(figsize=(18, 9))
plt.imshow(rho, cmap='gray', extent = (x[0], x[-1], z[-1], z[0]))
plt.scatter(s[0, 5::10], s[1, 5::10], marker='*', s=150, c='r', edgecolors='k')
plt.scatter(r[0, ::10],  r[1, ::10], marker='v', s=150, c='b', edgecolors='k')
plt.scatter(vs[0], vs[1], marker='.', s=250, c='m', edgecolors='k')
plt.axis('tight')
plt.xlabel('x [m]'),plt.ylabel('y [m]'),plt.title('Model and Geometry')
plt.xlim(x[0], x[-1])

# Direct arrival traveltime
directVS = np.sqrt((vs[0]-r[0][:, np.newaxis])**2+(vs[1]-r[1][:, np.newaxis])**2)/vel
directVS_off = directVS - toff

plt.figure()
im = plt.imshow(directVS, cmap='gist_rainbow')
plt.axis('tight')
plt.xlabel('#VS'),plt.ylabel('#Rec'),plt.title('Direct arrival')
plt.colorbar(im)

# Inversion
MarchenkoWM = Marchenko(R, dt=dt, dr=dr, nfmax=nfmax, wav=wav,
                        toff=toff, nsmooth=nsmooth)

t0 = time.time()
f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus = \
    MarchenkoWM.apply_multiplepoints(directVS, nfft=2**11, rtm=True,
                                     greens=True, dottest=False,
                                     **dict(iter_lim=niter, show=True))
g_inv_tot = g_inv_minus + g_inv_plus
tmulti = time.time() - t0
print('Elapsed time (s): %.2f' % tmulti)

fig, axs = plt.subplots(5, 1, figsize=(16, 22))
axs[0].imshow(np.swapaxes(p0_minus, 0, 1).reshape(nr*nvs, 2*nt-1).T, cmap='gray',
              vmin=-5e-1, vmax=5e-1, extent=(0, nr*nvs, t[-1], -t[-1]))
axs[0].set_title(r'$p_0^-$')
axs[0].set_xlabel(r'$x_R$')
axs[0].set_ylabel(r'$t$')
axs[0].axis('tight')
axs[0].set_ylim(1, -1)
axs[1].imshow(np.swapaxes(f1_inv_minus, 0, 1).reshape(nr*nvs,2*nt-1).T,
              cmap='gray', vmin=-5e-1, vmax=5e-1,
              extent=(0, nr*nvs, t[-1], -t[-1]))
axs[1].set_title(r'$f^-$')
axs[1].set_xlabel(r'$x_R$')
axs[1].axis('tight')
axs[1].set_ylim(1, -1)
axs[2].imshow(np.swapaxes(f1_inv_plus, 0, 1).reshape(nr*nvs,2*nt-1).T,
              cmap='gray', vmin=-5e-1, vmax=5e-1,
              extent=(0, nr*nvs, t[-1], -t[-1]))
axs[2].set_title(r'$f^+$')
axs[2].set_xlabel(r'$x_R$')
axs[2].axis('tight')
axs[2].set_ylim(1, -1)
axs[3].imshow(np.swapaxes(g_inv_minus, 0, 1).reshape(nr*nvs,2*nt-1).T,
              cmap='gray', vmin=-5e-1, vmax=5e-1,
              extent=(0, nr*nvs, t[-1], -t[-1]))
axs[3].set_title(r'$g^-$')
axs[3].set_xlabel(r'$x_R$')
axs[3].axis('tight')
axs[3].set_ylim(1.5, 0)
axs[4].imshow(np.swapaxes(g_inv_plus, 0, 1).reshape(nr*nvs,2*nt-1).T,
              cmap='gray', vmin=-5e-1, vmax=5e-1,
              extent=(0, nr*nvs, t[-1], -t[-1]))
axs[4].set_title(r'$g^+$')
axs[4].set_xlabel(r'$x_R$')
axs[4].axis('tight')
axs[4].set_ylim(1.5, 0)
fig.tight_layout()

##############################################################################
# Let's evaluate how faster is to actually use
# :py:func:`pylops.waveeqprocessing.Marchenko.apply_multiplepoints`
# instead of repeatedly applying
# :py:func:`pylops.waveeqprocessing.Marchenko.apply_onepoint`.
print('Speedup between single and multi: %.2f' % ((tone * nvs) / tmulti))

##############################################################################
# Finally we can take this example one step further and try to recover the
# local reflectivity at the depth level of the virtual sources using
# :py:func:`pylops.waveeqprocessing.mdd.MDD`.

# Taper gplus
tap = taper3d(2*nt-1, (nr, nvs), (1, 5))
g_inv_plus *= tap

# Direct wave
G0sub = np.zeros((nr, nvs, nt))
for ivs in range(nvs):
    G0sub[:, ivs] = directwave(wav, directVS[:,ivs], nt, dt,
                               nfft=int(2**(np.ceil(np.log2(nt))))).T

# MDD
_, Rrtm = MDD(G0sub, p0_minus[:, :, nt-1:],
              dt=dt, dr=dvsx, twosided=True, adjoint=True,
              psf=False, wav=wav[wav_c-60:wav_c+60],
              nfmax=nfmax, dtype='complex64', dottest=False,
              **dict(iter_lim=0, show=0))

Rmck = MDD(g_inv_plus[:, :, nt-1:], g_inv_minus[:, :, nt-1:],
           dt=dt, dr=dvsx, twosided=True, adjoint=False, psf=False,
           nfmax=nfmax, dtype='complex64', dottest=False,
           **dict(iter_lim=10, show=0))

fig, axs = plt.subplots(1, 2,  sharey=True, figsize=(10, 8))
im = axs[0].imshow(Rrtm[nvs//2, :, nt:].T, cmap='gray',
                   vmin=-0.4*np.max(np.abs(Rrtm[nvs//2, :, nt:])),
                   vmax=0.4*np.max(np.abs(Rrtm[nvs//2, :, nt:])),
                   extent=(vs[0][0], vs[0][-1], t[-1], t[0]))
axs[0].set_title('R single-scattering')
axs[0].set_xlabel(r'$x_{VS}$')
axs[0].set_ylabel(r'$t$')
axs[0].axis('tight')
axs[1].imshow(Rmck[nvs//2, :, nt:].T, cmap='gray',
              vmin=-0.7*np.max(np.abs(Rmck[nvs//2, :, nt:])),
              vmax=0.7*np.max(np.abs(Rmck[nvs//2, :, nt:])),
              extent=(vs[0][0], vs[0][-1], t[-1], t[0]))
axs[1].set_title('R Mck')
axs[1].set_xlabel(r'$x_{VS}$')
axs[1].axis('tight')
axs[1].set_ylim(0.7, 0.)
fig.tight_layout()
