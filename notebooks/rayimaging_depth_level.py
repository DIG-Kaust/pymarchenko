import os
import warnings
warnings.filterwarnings('ignore')

import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

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


def rayimaging_depth_level(ivsz, nvsz, vsx, vsz, r, s, nr, dr, dvsx, nt, dt, vel, troff, tsoff, nsmooth, 
                           wav, nfmax, igaths, nalpha, Vzd, Vzu, jt=1, iter_mck=30, iter_mdd=10, verb=False):
    print('Working on %d/%d' % (ivsz, nvsz))
    nvsx = len(vsx)
    ns = s.shape[1]
    ngath = len(igaths)
    
    # direct arrival window - traveltime
    directVSr = np.sqrt((vsx-r[0][:, np.newaxis])**2+(vsz[ivsz]-r[1][:, np.newaxis])**2)/vel
    directVSr_off = directVSr - troff
    directVSs = np.sqrt((vsx-s[0][:, np.newaxis])**2+(vsz[ivsz]-s[1][:, np.newaxis])**2)/vel
    directVSs_off = directVSs - tsoff

    # window
    idirectVSr_off = np.round(directVSr_off/dt).astype(np.int)
    idirectVSs_off = np.round(directVSs_off/dt).astype(np.int)
    
    wr = np.zeros((nr, nvsx, nt))
    ws = np.zeros((ns, nvsx, nt))
    for ir in range(nr):
        for ivs in range(nvsx):
            wr[ir, ivs, :idirectVSr_off[ir, ivs]]=1
    for ir in range(ns):    
        for ivs in range(nvsx):
            ws[ir, ivs, :idirectVSs_off[ir, ivs]]=1
    wr = np.hstack((np.fliplr(wr), wr[:, 1:]))
    ws = np.hstack((np.fliplr(ws), ws[:, 1:]))

    if nsmooth>0:
        smooth=np.ones(nsmooth)/nsmooth
        wr  = filtfilt(smooth, 1, wr)
        ws  = filtfilt(smooth, 1, ws)
       
    G0sub_rec = np.zeros((nr, nvsx, nt))
    for ivs in range(nvsx):
        G0sub_rec[:, ivs] = directwave(wav, directVSr[:,ivs], nt, dt, nfft=int(2**(np.ceil(np.log2(nt))))).T
    
    G0sub_src = np.zeros((ns, nvsx, nt))
    for ivs in range(nvsx):
        G0sub_src[:, ivs] = directwave(wav, directVSs[:,ivs], nt, dt, nfft=int(2**(np.ceil(np.log2(nt))))).T
    
    irtm, imck, artm, amck = 0, 0, 0, 0
    
    #print('Mck %d/%d' % (ivsz, nvsz))
    rm = RayleighMarchenko(Vzd, Vzu, dt=dt, dr=dr,
                           nfmax=nfmax, wav=wav, toff=troff, nsmooth=nsmooth,
                           saveVt=False, prescaled=False)
    f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus = \
        rm.apply_multiplepoints(directVSs, directVSr, G0=G0sub_rec, rtm=True, greens=True, 
                                dottest=False, **dict(iter_lim=iter_mck, show=verb))
    #print('Done Mck %d/%d' % (ivsz, nvsz))

    
    # MDD
    _, Rrtm = MDD(G0sub_src[:, :, ::jt], p0_minus[:, :, nt-1:][:, :, ::jt], 
                  dt=jt*dt, dr=dvsx, twosided=True, adjoint=True, psf=False, wav=wav,
                  nfmax=nfmax, dtype='complex64', dottest=False, **dict(iter_lim=0, show=verb))
 
    Rmck = MDD(g_inv_plus[:, :, nt-1:][:, :, ::jt], g_inv_minus[:, :, nt-1:][:, :, ::jt], 
               dt=jt*dt, dr=dvsx, twosided=True, adjoint=False, psf=False, wav=wav,
               nfmax=nfmax, dtype='complex64', dottest=False, **dict(iter_lim=iter_mdd, show=verb))
    #print('Done MDD %d/%d' % (ivsz, nvsz))
    
    # Images
    irtm = np.diag(Rrtm[:, :, nt-1])
    imck = np.diag(Rmck[:, :, nt-1])
    
    # Angle gathers
    artm = np.zeros((ngath, nalpha))
    amck = np.zeros((ngath, nalpha))
    for i, igath in enumerate(igaths):
        artm[i], angle, Ra = AngleGather(Rrtm.transpose(2, 0, 1), nvsx, nalpha, 
                                         dt*jt, dvsx, igath, vel, plotflag=False)
        amck[i], angle, Ra = AngleGather(Rmck.transpose(2, 0, 1), nvsx, nalpha, 
                                         dt*jt, dvsx, igath, vel, plotflag=False)
    
    print('Done %d/%d' % (ivsz, nvsz))
    return irtm, imck, artm, amck