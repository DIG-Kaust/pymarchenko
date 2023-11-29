import logging
import warnings
import numpy as np

from scipy.signal import convolve, filtfilt
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


class MME():
    r"""Marchenko Multiple Elimination

    Solve multi-dimensional Marchenko Multiple Elimination problem using
    Neumann iterative substitution.

    Parameters
    ----------
    R : :obj:`numpy.ndarray`
        Multi-dimensional reflection response in time or frequency
        domain of size :math:`[n_s \times n_r \times n_t/n_{fmax}]`. If
        provided in time, `R` should not be of complex type. If
        provided in frequency, `R` should contain the positive time axis
        followed by the negative one. Note that the reflection response
        should have already been multiplied by 2.
    wav : :obj:`numpy.ndarray`
        Wavelet to apply to the reflection response shot gather used as
        initial guess
    wav_c : :obj:`int`, optional
        Index of center of wavelet. If ``None`` the middle sample is used.
    dt : :obj:`float`, optional
        Sampling of time integration axis
    nt : :obj:`float`, optional
        Number of samples in time (not required if ``R`` is in time)
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    nfmax : :obj:`int`, optional
        Index of max frequency to include in deconvolution process
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
    Marchenko Multiple Elimination is a method that allows to produce a
    primary-only reflection data by repeated filtering of the data with
    itself and windowing [1]_.

    The projected Marchenko equations can be written in compact matrix-vector
    notation as:

    .. math::
        \mathbf{v^-}  = \Theta \mathbf{R} + \Theta \mathbf{R} \mathbf{v^+} \\
        \mathbf{v^+} = \Theta \mathbf{R^*} \mathbf{v^-}

    and solved for :math:`\mathbf{v^+}` via Neumann iterative substitution:

    .. math::
        \mathbf{v^+} = \sum_{k=0}^\inf (\Theta \mathbf{R^*}\Theta \mathbf{R})^k
        \Theta \mathbf{R^*} \Theta R = \sum_{k=1}^\inf (\Theta \mathbf{R^*}
        \Theta \mathbf{R})^k \delta

    where :math:`R` is the reflection response (or :math:`\delta` is a
    spatio-temporal delta) and :math:`\Theta` is the window. The MME algorithm
    requires a small time offset ``toff`` (on the order of the wavelet
    lenght) to be subtracted to the window time, whilst the TMME requires such
    time offset to be added.

    At this point the projected demultipled data :math:`\mathbf{U^-}` is
    computed and its values at time sample :math:`t=2t_d` is extracted:

    .. math::
        \mathbf{U^-} = \mathbf{R} + \mathbf{R} \mathbf{v^+} = \mathbf{R} +
        \mathbf{R} \sum_{k=1}^\inf (\Theta \mathbf{R^*}\Theta \mathbf{R})^k
        \delta

    If we repeat the same procedure for all possible $t=2t_d$, the retrived
    dataset is deprived of all internal multiples.

    .. [1] Zhang, L., Thorbecke, J., Wapenaar, K., and Slob, E., "Data-driven
        internal multiple elimination and its consequences for imaging:
        A comparison of strategies", Geophysics, vol. 84, pp. S365–S372. 2019.

    .. [2] Zhang, L., Thorbecke, J., Wapenaar, K., and Slob, E., "Transmission
        compensated primary reflection retrieval in the data domain and
        consequences for imaging", Geophysics, vol. 84, pp. Q27–Q36. 2019.

    """
    def __init__(self, R, wav, wav_c=None, dt=0.004, nt=None, dr=1.,
                 nfmax=None, toff=0.0, nsmooth=10,
                 dtype='float64', saveRt=True, prescaled=False):
        # Save inputs into class
        self.dt = dt
        self.dr = dr
        self.wav = wav
        self.wav_c = wav_c
        if wav is not None and wav_c is None:
            self.wav_c = len(wav) // 2
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
        self.nt2 = int(2 * self.nt - 1)
        self.t = np.arange(self.nt) * self.dt

        # Fix nfmax to be at maximum equal to half of the size of fft samples
        if self.nfmax is None or self.nfmax > np.ceil((self.nt2 + 1) / 2):
            self.nfmax = int(np.ceil((self.nt2 + 1) / 2))
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

    def _apply_onetime_onesrc(self, t0, Rop, R1op, Rsrc, n_iter=10):
        """Marchenko redatuming for one time step and one source
        """
        # Create window
        w = np.zeros((self.nr, 2 * self.nt - 1), dtype=self.dtype)
        w[:, int(self.toff / self.dt):int(t0 / self.dt)] = 1
        if self.nsmooth > 0:
            smooth = np.ones(self.nsmooth, dtype=self.dtype) / self.nsmooth
            w = filtfilt(smooth, 1, w)
        w = to_cupy_conditional(self.Rtwosided_fft, w)
        Wop = Diagonal(w.T.ravel())

        # Create initial guess
        if self.wav is not None:
            Rsrc = np.apply_along_axis(convolve, -1, Rsrc, self.wav,
                                       mode='full')
            Rsrc = Rsrc[:, self.wav_c:][:, :self.nt]
        v_sub_plus = np.concatenate((Rsrc.T,
                                     np.zeros((self.nt - 1, self.nr),
                                              dtype=self.dtype)))

        # First step
        v_sub_plus = Wop * R1op * Wop * v_sub_plus.ravel()

        # Run iterative scheme
        dv_sub_plus = v_sub_plus.copy().ravel()
        for _ in range(n_iter - 1):
            dv_sub_plus = (Wop * R1op * Wop * Rop) * dv_sub_plus
            v_sub_plus += dv_sub_plus

        v_sub_minus = Rop * v_sub_plus.ravel()
        U_sub_minus = Rsrc.T + v_sub_minus.reshape(2*self.nt-1, self.nr)[:self.nt]
        return U_sub_minus

    def apply_onesrc(self, Rsrc, usematmul=False, trcomp=False, ntmax=None,
                     n_iter=10):
        r"""Marchenko Multiple elimination for one shot gather

        Solve the Marchenko Multiple elimination problem via
        iterative substitution for a set of sources
        Parameters
        ----------
        t0 : :obj:`float`
            Time level
        Rsrc : :obj:`np.ndarray`
            Reflection response in time domain for single source of
            size :math:`[n_r \times n_t]`.
        usematmul : :obj:`bool`, optional
            Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
            (``False``) in :py:class:`pylops.signalprocessing.Fredholm1` operator.
            Refer to Fredholm1 documentation for details.
        trcomp : :obj:`bool`, optional
            Transmission compensation
        ntmax : :obj:`int`, optional
            Index of maximum time to run demultiple for
        n_iter : :obj:`int`, optional
            Number of iterations of Neumann series

        Returns
        ----------
        U_inv_minus : :obj:`numpy.ndarray`
            Upgoing projected focusing function of size
            :math:`[n_r \times n_t]`

        """
        # Choose how to add offset to window (positive or negative)
        itmin = int(self.toff / self.dt)
        if trcomp:
            trcomp = -1.
            itmin = 1
        else:
            trcomp = 1.

        # Create operators
        Rop = MDC(self.Rtwosided_fft, self.nt2, nv=1, dt=self.dt, dr=self.dr,
                  twosided=False, conj=False,
                  saveGt=self.saveRt, prescaled=self.prescaled,
                  usematmul=usematmul)
        R1op = MDC(self.Rtwosided_fft, self.nt2, nv=1, dt=self.dt, dr=self.dr,
                   twosided=False, conj=True,
                   saveGt=self.saveRt, prescaled=self.prescaled,
                   usematmul=usematmul)

        # Run estimate over time steps
        U_sub_minus = np.zeros((self.nr, self.nt), dtype=self.dtype)
        for it in range(itmin, ntmax if ntmax is not None else self.nt):
            U_sub_minus[:, it] = \
                self._apply_onetime_onesrc(it * self.dt - trcomp * self.toff, Rop,
                                           R1op, Rsrc, n_iter)[it]
        return U_sub_minus

    def _apply_onetime_multisrc(self, t0, Rop, R1op, Rsrcs, n_iter=10):
        """Marchenko redatuming for one time step and multiple sources
        """
        nsrc = Rsrcs.shape[0]

        # Create window
        w = np.zeros((self.nr, nsrc, 2 * self.nt - 1), dtype=self.dtype)
        w[:, :, int(self.toff / self.dt):int(t0 / self.dt)] = 1
        if self.nsmooth > 0:
            smooth = np.ones(self.nsmooth, dtype=self.dtype) / self.nsmooth
            w = filtfilt(smooth, 1, w)
        w = to_cupy_conditional(self.Rtwosided_fft, w)
        Wop = Diagonal(w.transpose(2, 0, 1).ravel())

        # Create initial guess
        if self.wav is not None:
            Rsrcs = np.apply_along_axis(convolve, -1, Rsrcs, self.wav,
                                        mode='full')
            Rsrcs = Rsrcs[:, :, self.wav_c:][:, :, :self.nt]
        v_sub_plus = np.concatenate((Rsrcs.transpose(2, 1, 0),
                                     np.zeros((self.nt - 1, self.nr, nsrc),
                                              dtype=self.dtype)))

        # First step
        v_sub_plus = Wop * R1op * Wop * v_sub_plus.ravel()

        # Run iterative scheme
        dv_sub_plus = v_sub_plus.copy().ravel()
        for _ in range(n_iter - 1):
            dv_sub_plus = (Wop * R1op * Wop * Rop) * dv_sub_plus
            v_sub_plus += dv_sub_plus

        v_sub_minus = Rop * v_sub_plus.ravel()
        U_sub_minus = Rsrcs.transpose(2, 1, 0) + \
                      v_sub_minus.reshape(2*self.nt-1, self.nr, nsrc)[:self.nt]
        return U_sub_minus

    def apply_multisrc(self, Rsrcs, usematmul=False, trcomp=False, ntmax=None,
                       n_iter=10):
        r"""Marchenko Multiple elimination for multiple shot gathers

        Solve the Marchenko Multiple elimination problem via
        iterative substitution for a set of sources

        Parameters
        ----------
        t0 : :obj:`float`
            Time level
        Rsrcs : :obj:`np.ndarray`
            Reflection response in time domain for multiple sources of
            size :math:`[n_s \times n_r \times n_t]`.
        usematmul : :obj:`bool`, optional
            Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
            (``False``) in :py:class:`pylops.signalprocessing.Fredholm1` operator.
            Refer to Fredholm1 documentation for details.
        trcomp : :obj:`bool`, optional
            Transmission compensation
        ntmax : :obj:`int`, optional
            Index of maximum time to run demultiple for
        n_iter : :obj:`int`, optional
            Number of iterations of Neumann series

        Returns
        ----------
        U_inv_minus : :obj:`numpy.ndarray`
            Upgoing projected focusing function of size
            :math:`[n_s \times n_r \times n_t]`

        """
        # Choose how to add offset to window (positive or negative)
        itmin = int(self.toff / self.dt)
        if trcomp:
            trcomp = -1.
            itmin = 1
        else:
            trcomp = 1.

        # Create operators
        nsrc = Rsrcs.shape[0]
        Rop = MDC(self.Rtwosided_fft, self.nt2, nv=nsrc, dt=self.dt, dr=self.dr,
                  twosided=False, conj=False,
                  saveGt=self.saveRt, prescaled=self.prescaled,
                  usematmul=usematmul)
        R1op = MDC(self.Rtwosided_fft, self.nt2, nv=nsrc, dt=self.dt, dr=self.dr,
                   twosided=False, conj=True,
                   saveGt=self.saveRt, prescaled=self.prescaled,
                   usematmul=usematmul)

        # Run estimate over time steps
        U_sub_minus = np.zeros((nsrc, self.nr, self.nt), dtype=self.dtype)
        for it in range(itmin, ntmax if ntmax is not None else self.nt):
            U_sub_minus[:, :, it] = \
                self._apply_onetime_multisrc(it * self.dt - trcomp * self.toff, Rop,
                                             R1op, Rsrcs, n_iter)[it].T
        return U_sub_minus