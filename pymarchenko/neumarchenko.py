import logging
import warnings
import numpy as np

from scipy.signal import filtfilt
from scipy.sparse.linalg import lsqr
from scipy.special import hankel2
from pylops.utils import dottest as Dottest
from pylops import Diagonal, Identity, Block, BlockDiag
from pylops.waveeqprocessing.mdd import MDC
from pylops.waveeqprocessing.marchenko import directwave
from pylops.optimization.basic import cgls
from pylops.utils.backend import get_array_module, get_module_name, \
    to_cupy_conditional

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class NeumannMarchenko():
    r"""Iterative Marchenko redatuming

    Solve multi-dimensional Marchenko redatuming problem using
    Neumann iterative substitution.

    Parameters
    ----------
    R : :obj:`numpy.ndarray`
        Multi-dimensional reflection response in time or frequency
        domain of size :math:`[n_s \times n_r \times n_t/n_{fmax}]`. If
        provided in time, ``R`` should not be of complex type. If
        provided in frequency, ``R`` should contain the positive time axis
        followed by the negative one. Note that the reflection response
        should have already been multiplied by 2.
    dt : :obj:`float`, optional
        Sampling of time integration axis
    nt : :obj:`float`, optional
        Number of samples in time (not required if ``R`` is in time)
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    nfmax : :obj:`int`, optional
        Index of max frequency to include in deconvolution process
    wav : :obj:`numpy.ndarray`, optional
        Wavelet to apply to direct arrival when created using ``trav``
    toff : :obj:`float`, optional
        Time-offset to apply to traveltime
    nsmooth : :obj:`int`, optional
        Number of samples of smoothing operator to apply to window
    dtype : :obj:`bool`, optional
        Type of elements in input array.
    saveRt : :obj:`bool`, optional
        Save ``R`` and ``R^H`` to speed up the computation of adjoint of
        :class:`pylops.signalprocessing.Fredholm1` (``True``) or create
        ``R^H`` on-the-fly (``False``) Note that ``saveRt=True`` will be
        faster but double the amount of required memory
    prescaled : :obj:`bool`, optional
        Apply scaling to ``R`` (``False``) or not (``False``)
        when performing spatial and temporal summations within the
        :class:`pylops.waveeqprocessing.MDC` operator. In case
        ``prescaled=True``, the ``R`` is assumed to have been pre-scaled by
        the user.

    Attributes
    ----------
    ns : :obj:`int`
        Number of samples along source axis
    nr : :obj:`int`
        Number of samples along receiver axis
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (True) or not (False)

    Raises
    ------
    TypeError
        If ``t`` is not :obj:`numpy.ndarray`.

    Notes
    -----
    Marchenko redatuming is a method that allows to produce correct
    subsurface-to-surface responses given the availability of a
    reflection data and a macro-velocity model [1]_.

    The Marchenko equations can be solved via Neumann iterative substitution:

    .. math::
        \mathbf{f_m^+} =  \Theta \mathbf{R^*} (\Theta \mathbf{R}
        \mathbf{f_d^+} + \Theta \mathbf{R} \mathbf{f_m^+})

    and isolating :math:`\mathbf{f_m^+}`:

    .. math::
        (\mathbf{I} - \Theta \mathbf{R^*}\Theta \mathbf{R}) \mathbf{f_m^+} =
        \Theta \mathbf{R^*} \Theta \mathbf{R} \mathbf{f_d^+}

    We can then expand the term within parenthesis as a Neumann series and write:

    .. math::
        \mathbf{f^+} = \sum_{k=0}^\inf (\Theta \mathbf{R^*}\Theta
        \mathbf{R})^k \mathbf{f_d^+}

    Finally the subsurface Green's functions can be obtained applying the
    following operator to the retrieved focusing functions

    .. math::
        \begin{bmatrix}
           -\mathbf{g^-}  \\
           \mathbf{g^{+ *}}
        \end{bmatrix} =
        \mathbf{I} -
        \begin{bmatrix}
           \mathbf{0}  &    \mathbf{R}   \\
           \mathbf{R^*} & \mathbf{0}
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{f^-}  \\
           \mathbf{f^+}
        \end{bmatrix}

    Here :math:`\mathbf{R}` is the monopole-to-particle velocity seismic
    response (already multiplied by 2).

    .. [1] Wapenaar, K., Thorbecke, J., Van der Neut, J., Broggini, F.,
        Slob, E., and Snieder, R., "Marchenko imaging", Geophysics, vol. 79,
        pp. WA39-WA57. 2014.

    """
    def __init__(self, R, dt=0.004, nt=None, dr=1.,
                 nfmax=None, wav=None, toff=0.0, nsmooth=10,
                 dtype='float64', saveRt=True, prescaled=False):
        # Save inputs into class
        self.dt = dt
        self.dr = dr
        self.wav = wav
        self.toff = toff
        self.nsmooth = nsmooth
        self.saveRt = saveRt
        self.prescaled = prescaled
        self.dtype = dtype
        self.explicit = False
        self.ncp = get_array_module(R)

        # Infer dimensions of R
        if not np.iscomplexobj(R):
            self.ns, self.nr, self.nt = R.shape
            self.nfmax = nfmax
        else:
            self.ns, self.nr, self.nfmax = R.shape
            self.nt = nt
            if nt is None:
                logging.error('nt must be provided as R is in frequency')
        self.nt2 = int(2*self.nt-1)
        self.t = np.arange(self.nt)*self.dt

        # Fix nfmax to be at maximum equal to half of the size of fft samples
        if self.nfmax is None or self.nfmax > np.ceil((self.nt2 + 1) / 2):
            self.nfmax = int(np.ceil((self.nt2+1)/2))
            logging.warning('nfmax set equal to (nt+1)/2=%d', self.nfmax)

        # Add negative time to reflection data and convert to frequency
        if not np.iscomplexobj(R):
            Rtwosided = np.concatenate((R, self.ncp.zeros((self.ns, self.nr,
                                                           self.nt - 1),
                                                          dtype=R.dtype)),
                                       axis=-1)
            Rtwosided_fft = np.fft.rfft(Rtwosided, self.nt2,
                                        axis=-1) / np.sqrt(self.nt2)
            self.Rtwosided_fft = Rtwosided_fft[..., :nfmax]
        else:
            self.Rtwosided_fft = R
        # bring frequency to first dimension
        self.Rtwosided_fft = self.Rtwosided_fft.transpose(2, 0, 1)

    def apply_onepoint(self, trav, G0=None, nfft=None, rtm=False, greens=False,
                       usematmul=False, n_iter=10):
        r"""Marchenko redatuming for one point

        Solve the Marchenko redatuming iterative substitution for a single point
        given its direct arrival traveltime curve (``trav``)
        and waveform (``G0``).

        Parameters
        ----------
        trav : :obj:`numpy.ndarray`
            Traveltime of first arrival from subsurface point to
            surface receivers of size :math:`[n_r \times 1]`
        G0 : :obj:`numpy.ndarray`, optional
            Direct arrival in time domain of size :math:`[n_r \times n_t]`
            (if None, create arrival using ``trav``)
        nfft : :obj:`int`, optional
            Number of samples in fft when creating the analytical direct wave
        rtm : :obj:`bool`, optional
            Compute and return rtm redatuming
        greens : :obj:`bool`, optional
            Compute and return Green's functions
        usematmul : :obj:`bool`, optional
            Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
            (``False``) in :py:class:`pylops.signalprocessing.Fredholm1` operator.
            Refer to Fredholm1 documentation for details.
        n_iter : :obj:`int`, optional
            Number of iterations of Neumann series

        Returns
        ----------
        f1_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing focusing function of size :math:`[n_r \times n_t]`
        f1_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing focusing function
            of size :math:`[n_r \times n_t]`
        p0_minus : :obj:`numpy.ndarray`
            Single-scattering standard redatuming upgoing Green's function of
            size :math:`[n_r \times n_t]`
        g_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing Green's function of size :math:`[n_r \times n_t]`
        g_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing Green's function
            of size :math:`[n_r \times n_t]`

        """
        # Create window
        trav_off = trav - self.toff
        trav_off = np.round(trav_off / self.dt).astype(np.int32)

        w = np.zeros((self.nr, self.nt), dtype=self.dtype)
        for ir in range(self.nr):
            w[ir, :trav_off[ir]] = 1
        w = np.hstack((w[:, 1:], np.fliplr(w)))
        if self.nsmooth > 0:
            smooth = np.ones(self.nsmooth, dtype=self.dtype) / self.nsmooth
            w = filtfilt(smooth, 1, w)
        w = to_cupy_conditional(self.Rtwosided_fft, w)

        # Create operators
        Rop = MDC(self.Rtwosided_fft, self.nt2, nv=1, dt=self.dt, dr=self.dr,
                  twosided=False, conj=False,
                  saveGt=self.saveRt, prescaled=self.prescaled,
                  usematmul=usematmul)
        R1op = MDC(self.Rtwosided_fft, self.nt2, nv=1, dt=self.dt, dr=self.dr,
                   twosided=False, conj=True,
                   saveGt=self.saveRt, prescaled=self.prescaled,
                   usematmul=usematmul)
        Wop = Diagonal(w.T.flatten())

        # Create input focusing function
        if G0 is None:
            if self.wav is not None and nfft is not None:
                G0 = (directwave(self.wav, trav, self.nt,
                                 self.dt, nfft=nfft, derivative=True)).T
                G0 = to_cupy_conditional(self.Rtwosided_fft, G0)
            else:
                logging.error('wav and/or nfft are not provided. '
                              'Provide either G0 or wav and nfft...')
                raise ValueError('wav and/or nfft are not provided. '
                                 'Provide either G0 or wav and nfft...')
        fd_plus = np.concatenate((self.ncp.zeros((self.nt - 1, self.nr),
                                                 dtype=self.dtype),
                                  np.fliplr(G0).T))

        # Run standard redatuming as benchmark
        if rtm:
            p0_minus = Rop * fd_plus.flatten()
            p0_minus = p0_minus.reshape(self.nt2, self.ns).T

        # Run iterative scheme
        f1_sub_plus = fd_plus.copy().ravel()
        df1_sub_plus = fd_plus.copy().ravel()
        for _ in range(n_iter - 1):
            df1_sub_plus = (Wop * R1op * Wop * Rop) * df1_sub_plus
            f1_sub_plus += df1_sub_plus
        f1_sub_minus = Wop * Rop * f1_sub_plus.ravel()
        g_sub_minus = - (f1_sub_minus.ravel() - Rop * f1_sub_plus.ravel())
        g_sub_plus = f1_sub_plus.ravel() - R1op * f1_sub_minus.ravel()

        f1_sub_plus = f1_sub_plus.reshape(self.nt2, self.nr).T
        f1_sub_minus = f1_sub_minus.reshape(self.nt2, self.nr).T
        g_sub_minus = g_sub_minus.reshape(self.nt2, self.nr).T
        g_sub_plus = np.flipud(g_sub_plus.reshape(self.nt2, self.nr)).T

        # Bring back to time axis with negative part
        f1_sub_minus = np.fft.ifftshift(f1_sub_minus, axes=1)
        f1_sub_plus = np.fft.fftshift(f1_sub_plus, axes=1)
        if rtm:
            p0_minus = np.fft.ifftshift(p0_minus, axes=1)
        if greens:
            g_sub_minus = np.fft.ifftshift(g_sub_minus, axes=1)
            g_sub_plus = np.fft.fftshift(g_sub_plus, axes=1)

        if rtm and greens:
            return f1_sub_minus, f1_sub_plus, p0_minus, g_sub_minus, g_sub_plus
        elif rtm:
            return f1_sub_minus, f1_sub_plus, p0_minus
        elif greens:
            return f1_sub_minus, f1_sub_plus, g_sub_minus, g_sub_plus
        else:
            return f1_sub_minus, f1_sub_plus
