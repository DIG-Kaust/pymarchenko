"""
3. Marchenko redatuming with missing sources
============================================
This example shows how the :py:class:`pymarchenko.marchenko.Marchenko`
routine can handle acquisition geometries with missing sources. We will first
see that using least-squares inversion leads to retrieving focusing functions
that present gaps due to the missing sources. We further leverage sparsity-
promoting inversion and show that focusing functions can be retrieved that are
almost of the same quality as those constructed with the full acquisition
geometry.

"""
# sphinx_gallery_thumbnail_number = 5
# pylint: disable=C0103
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve
from pylops.basicoperators import Transpose
from pylops.signalprocessing import Radon2D, Sliding2D
from pylops.waveeqprocessing import MDD
from pylops.waveeqprocessing.marchenko import directwave
from pylops.utils.tapers import taper3d
from pymarchenko.marchenko import Marchenko

warnings.filterwarnings('ignore')
plt.close('all')
np.random.seed(10)

###############################################################################
# Let's start by defining some input parameters and loading the geometry

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
# Let's now load and display the reflection response

# Time axis
t = inputdata['t'][:-100]
ot, dt, nt = t[0], t[1]-t[0], len(t)
t2 = np.concatenate([-t[::-1], t[1:]])
nt2 = 2 * nt - 1

# Reflection data (R[s, r, t]) and subsurface fields
R = inputdata['R'][:, :, :-100]
R = np.swapaxes(R, 0, 1) # just because of how the data was saved
taper = taper3d(nt, [ns, nr], [nstaper, nstaper], tapertype='hanning')
R = R * taper

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 7))
axs[0].imshow(R[20].T, cmap='gray', vmin=-1e-2, vmax=1e-2,
              extent=(r[0, 0], r[0, -1], t[-1], t[0]))
axs[0].set_title('R shot=20')
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
axs[2].imshow(R[ns-20].T, cmap='gray', vmin=-1e-2, vmax=1e-2,
              extent=(r[0, 0], r[0, -1], t[-1], t[0]))
axs[2].set_title('R shot=%d' %(ns-20))
axs[2].set_xlabel(r'$x_R$')
axs[2].axis('tight')
axs[2].set_ylim(1.5, 0)
fig.tight_layout()

###############################################################################
# and the true and background subsurface fields

# Subsurface fields
Gsub = inputdata['Gsub'][:-100]
G0sub = inputdata['G0sub'][:-100]
wav = inputdata['wav']
wav_c = np.argmax(wav)

Gsub = np.apply_along_axis(convolve, 0, Gsub, wav, mode='full')
Gsub = Gsub[wav_c:][:nt]
G0sub = np.apply_along_axis(convolve, 0, G0sub, wav, mode='full')
G0sub = G0sub[wav_c:][:nt]

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
# First, we use the entire data and create our benchmark solution by means of
# least-squares inversion

# Direct arrival traveltime
trav = np.sqrt((vs[0]-r[0])**2+(vs[1]-r[1])**2)/vel

MarchenkoWM = Marchenko(R, dt=dt, dr=dr, nfmax=nfmax, wav=wav,
                        toff=toff, nsmooth=nsmooth)

f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus = \
    MarchenkoWM.apply_onepoint(trav, G0=G0sub.T, rtm=True, greens=True,
                               dottest=True, **dict(iter_lim=niter, show=True))
g_inv_tot = g_inv_minus + g_inv_plus

##############################################################################
# Second, we define the available sources (60% of the original array randomly
# selected) and perform least-squares inversion

# Subsampling
perc_subsampling=0.6
nsava = int(np.round(ns*perc_subsampling))
ishuffle = np.random.permutation(np.arange(ns))
iava = np.sort(ishuffle[:nsava])
inotava = np.sort(ishuffle[nsava:])

MarchenkoWM = Marchenko(R[iava], dt=dt, dr=dr, nfmax=nfmax, wav=wav,
                        toff=toff, nsmooth=nsmooth, isava=iava)

f1_inv_minus_ls, f1_inv_plus_ls, p0_minus_ls, g_inv_minus_ls, g_inv_plus_ls = \
    MarchenkoWM.apply_onepoint(trav, G0=G0sub.T, rtm=True,
                               greens=True, dottest=False,
                               **dict(iter_lim=niter, show=True))
g_inv_tot_ls = g_inv_minus_ls + g_inv_plus_ls

##############################################################################
# Finally, we define a sparsifying transform and set up the inversion using
# a sparsity promoting solver like :func:`pylops.optimization.sparsity.FISTA`

# Sliding Radon as sparsifying transform
nwin = 25
nwins = 6
nover = 10
npx = 101
pxmax = 1e-3
px = np.linspace(-pxmax, pxmax, npx)
dimsd = (nr, nt2)
dimss = (nwins*npx, dimsd[1])

Top = Transpose((nt2, nr), axes=(1, 0), dtype=np.float64)
RadOp = Radon2D(t2, np.linspace(-dr*nwin//2, dr*nwin//2, nwin),
                px, centeredh=True, kind='linear', engine='numba')
Slidop = Sliding2D(RadOp, dimss, dimsd, nwin, nover, tapertype='cosine')

MarchenkoWM = Marchenko(R[iava], dt=dt, dr=dr, nfmax=nfmax, wav=wav,
                        toff=toff, nsmooth=nsmooth, isava=iava,
                        S=Top.H*Slidop)

f1_inv_minus_l1, f1_inv_plus_l1, p0_minus_l1, g_inv_minus_l1, g_inv_plus_l1 = \
    MarchenkoWM.apply_onepoint(trav, G0=G0sub.T, rtm=True,
                               greens=True, dottest=False,
                               **dict(eps=1e4, niter=400,
                                      alpha=1.05e-3,
                                      show=True))
g_inv_tot_l1 = g_inv_minus_l1 + g_inv_plus_l1

##############################################################################
# Let's now compare the three solutions starting from the focusing functions

fig, axs = plt.subplots(2, 3, sharey=True, figsize=(14, 12))
axs[0][0].imshow(f1_inv_minus.T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[0][0].set_title(r'$f^-$')
axs[0][0].axis('tight')
axs[0][0].set_ylim(1, -1)
axs[0][1].imshow(f1_inv_minus_ls.T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[0][1].set_title(r'$f^- L2$')
axs[0][1].axis('tight')
axs[0][1].set_ylim(1, -1)
axs[0][2].imshow(f1_inv_minus_l1.T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[0][2].set_title(r'$f^- L1$')
axs[0][2].axis('tight')
axs[0][2].set_ylim(1, -1)
axs[1][0].imshow(f1_inv_plus.T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[1][0].set_title(r'$f^+$')
axs[1][0].axis('tight')
axs[1][0].set_ylim(1, -1)
axs[1][1].imshow(f1_inv_plus_ls.T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[1][1].set_title(r'$f^+ L2$')
axs[1][1].axis('tight')
axs[1][1].set_ylim(1, -1)
axs[1][2].imshow(f1_inv_plus_l1.T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[1][2].set_title(r'$f^- L1$')
axs[1][2].axis('tight')
axs[1][2].set_ylim(1, -1)
fig.tight_layout()

##############################################################################
# and the up- and down- Green's functions

fig, axs = plt.subplots(2, 3, sharey=True, figsize=(14, 12))
axs[0][0].imshow(g_inv_minus[iava, :].T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[0][0].set_title(r'$g^-$')
axs[0][0].axis('tight')
axs[0][0].set_ylim(1.2, 0)
axs[0][1].imshow(g_inv_minus_ls.T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[0][1].set_title(r'$g^- L2$')
axs[0][1].axis('tight')
axs[0][1].set_ylim(1.2, 0)
axs[0][2].imshow(g_inv_minus_l1.T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[0][2].set_title(r'$g^- L1$')
axs[0][2].axis('tight')
axs[0][2].set_ylim(1.2, 0)
axs[1][0].imshow(g_inv_plus[iava, :].T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[1][0].set_title(r'$g^+$')
axs[1][0].axis('tight')
axs[1][0].set_ylim(1.2, 0)
axs[1][1].imshow(g_inv_plus_ls.T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[1][1].set_title(r'$g^+ L2$')
axs[1][1].axis('tight')
axs[1][1].set_ylim(1.2, 0)
axs[1][2].imshow(g_inv_plus_l1.T, cmap='gray', vmin=-5e5, vmax=5e5,
                 extent=(r[0, 0], r[0, -1], t2[-1], t2[0]))
axs[1][2].set_title(r'$g^- L1$')
axs[1][2].axis('tight')
axs[1][2].set_ylim(1.2, 0)
fig.tight_layout()

##############################################################################
# and finally the total Green's functions

fig = plt.figure(figsize=(18,9))
ax1 = plt.subplot2grid((1, 7), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 7), (0, 2), colspan=2)
ax3 = plt.subplot2grid((1, 7), (0, 4), colspan=2)
ax4 = plt.subplot2grid((1, 7), (0, 6))

ax1.imshow(Gsub[:, iava], cmap='gray', vmin=-5e5, vmax=5e5,
           extent=(r[0,0], r[0,-1], t[-1], t[0]))
ax1.set_title(r'$G_{true}$')
ax1.set_xlabel(r'$x_R$')
ax1.set_ylabel(r'$t$')
ax1.axis('tight')
ax1.set_ylim(1.2, 0)
ax2.imshow(g_inv_tot_ls.T, cmap='gray', vmin=-5e5, vmax=5e5,
           extent=(r[0,0], r[0,-1], t2[-1], t2[0]))
ax2.set_title(r'$G_{est} L2$')
ax2.set_xlabel(r'$x_R$')
ax2.axis('tight')
ax2.set_ylim(1.2, 0)
ax3.imshow(g_inv_tot_l1.T, cmap='gray', vmin=-5e5, vmax=5e5,
           extent=(r[0,0], r[0,-1], t2[-1], t2[0]))
ax3.set_title(r'$G_{est}$ L1 radon')
ax3.set_xlabel(r'$x_R$')
ax3.axis('tight')
ax3.set_ylim(1.2, 0)
ax4.plot(t**2*Gsub[:, iava][:, nr//4]/Gsub.max(), t, 'k', lw=7, label='True')
ax4.plot(t**2*g_inv_tot[iava][nr//4, nt-1:]/g_inv_tot.max(), t, 'r', lw=5, label='Full')
ax4.plot(t**2*g_inv_tot_ls[nr//4, nt-1:]/g_inv_tot.max(), t, 'b', lw=3, label='L2')
ax4.plot(t**2*g_inv_tot_l1[nr//4, nt-1:]/g_inv_tot.max(), t, '--g', lw=3, label='L1')
ax4.set_ylim(1.2, 0)
ax4.legend()
fig.tight_layout()

