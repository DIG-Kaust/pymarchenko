import logging
import warnings
import numpy as np

from scipy.signal import filtfilt
from scipy.sparse.linalg import lsqr
from scipy.special import hankel2
from pylops.utils import dottest as Dottest
from pylops import Diagonal, Identity, Block, BlockDiag, Restriction
from pylops.waveeqprocessing.mdd import MDC
from pylops.waveeqprocessing.marchenko import directwave
from pylops.optimization.basic import cgls
from pylops.optimization.sparsity import fista
from pylops.utils.backend import get_array_module, get_module_name, \
    to_cupy_conditional

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class Marchenko():
    r"""Marchenko redatuming

    Solve multi-dimensional Marchenko redatuming problem using
    :py:func:`scipy.sparse.linalg.lsqr` iterative solver.

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
    nt : :obj:`int`, optional
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
    isava : :obj:`list`, optional
        Indices of available sources. If not ``None``, a
        :class:`pylops.Restriction` operator is used instead of
        the :class:`pylops.Identity` operator along the main
        diagonal of the Marchenko operator
    S : :obj:`pylops.LinearOperator`, optional
        Sparsifying transform to be provided to solve the Marchenko equations
        via the :func:`pylops.optimization.sparsity.FISTA` solver in the case
        of missing sources. If ``S=None``, least-squares inversion is used
        instead.

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

    See Also
    --------
    MDC : Multi-dimensional convolution
    MDD : Multi-dimensional deconvolution

    Notes
    -----
    Marchenko redatuming is a method that allows to produce correct
    subsurface-to-surface responses given the availability of a
    reflection data and a macro-velocity model [1]_.

    The Marchenko equations can be written in a compact matrix
    form [2]_ and solved by means of iterative solvers such as LSQR:

    .. math::
        \begin{bmatrix}
           \Theta \mathbf{R} \mathbf{f_d^+}  \\
           \mathbf{0}
        \end{bmatrix} =
        \mathbf{I} -
        \begin{bmatrix}
           \mathbf{0}  &   \Theta \mathbf{R}   \\
           \Theta \mathbf{R^*} & \mathbf{0}
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{f^-}  \\
           \mathbf{f_m^+}
        \end{bmatrix}

    Subsequently the subsurface Green's functions can be obtained applying the
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

    Finally this routine can also be used to solve the Marchenko equations in
    the case of missing sources (provided that the available sources are
    co-located with receivers at indices ``isava``):

    .. math::
        \begin{bmatrix}
           \Theta \mathbf{R} \mathbf{f_d^+}  \\
           \mathbf{0}
        \end{bmatrix} =
        \begin{bmatrix}
           \mathbf{S}  &   \Theta \mathbf{R}   \\
           \Theta \mathbf{R^*} & \mathbf{S}
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{f^-}  \\
           \mathbf{f_m^+}
        \end{bmatrix}

    where :math:`\mathbf{S}` is a :class:`pylops.Restriction`
    operator. Note that in order to succesfully reconstruct focusing functions
    that do not present gaps at the location of missing sources, additional
    prior information must be provided in the form of sparsifying transforms
    and the equation must be solved via sparsity-promoting inversion. This
    is achieived by providing an appropriate ``Sop`` operator.


    .. [1] Wapenaar, K., Thorbecke, J., Van der Neut, J., Broggini, F.,
        Slob, E., and Snieder, R., "Marchenko imaging", Geophysics, vol. 79,
        pp. WA39-WA57. 2014.

    .. [2] van der Neut, J., Vasconcelos, I., and Wapenaar, K. "On Green's
       function retrieval by iterative substitution of the coupled
       Marchenko equations", Geophysical Journal International, vol. 203,
       pp. 792-813. 2015.

    .. [3] Haindl, C., Ravasi, M., and Broggini, F., K. "Handling gaps in
       acquisition geometries — Improving Marchenko-based imaging using
       sparsity-promoting inversion and joint inversion of time-lapse data",
       Geophysics, vol. 86, pp. S143-S154. 2021.

    """
    def __init__(self, R, dt=0.004, nt=None, dr=1.,
                 nfmax=None, wav=None, toff=0.0, nsmooth=10,
                 dtype='float64', saveRt=True, prescaled=False,
                 isava=None, S=None):
        # Save inputs into class
        self.dt = dt
        self.dr = dr
        self.wav = wav
        self.toff = toff
        self.nsmooth = nsmooth
        self.saveRt = saveRt
        self.prescaled = prescaled
        self.isava = isava
        self.S = S
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

        if self.isava is None:
            self.Iop = Identity(self.nr * self.nt2)
        else:
            self.Iop = Restriction((self.nt2, self.nr), isava, axis=1)
        if S is not None:
            self.Sop = BlockDiag([S, S])

    def apply_onepoint(self, trav, G0=None, nfft=None, rtm=False, greens=False,
                       dottest=False, usematmul=False, **kwargs_solver):
        r"""Marchenko redatuming for one point

        Solve the Marchenko redatuming inverse problem for a single point
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
        dottest : :obj:`bool`, optional
            Apply dot-test
        fast : :obj:`bool`
            *Deprecated*, will be removed in v2.0.0
        usematmul : :obj:`bool`, optional
            Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
            (``False``) in :py:class:`pylops.signalprocessing.Fredholm1` operator.
            Refer to Fredholm1 documentation for details.
        **kwargs_solver
            Arbitrary keyword arguments for chosen solver
            (:py:func:`scipy.sparse.linalg.lsqr` and
            :py:func:`pylops.optimization.basic.cgls` are used as default
            for numpy and cupy `data`, respectively.
            :py:func:`pylops.optimization.sparsity.fista` is used when a
            sparsifying transform ``S`` is provided).

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
        Wop = Diagonal(w.T.ravel())
        if self.isava is not None:
            Wsop = Diagonal(w[self.isava].T.ravel())
        else:
            Wsop = Wop
        Mop = Block([[self.Iop, -1 * Wsop * Rop],
                     [-1 * Wsop * R1op, self.Iop]]) * BlockDiag([Wop, Wop])
        Gop = Block([[self.Iop, -1 * Rop],
                     [-1 * R1op, self.Iop]])

        if dottest:
            Dottest(Gop, 2 * self.ns * self.nt2,
                    2 * self.nr * self.nt2,
                    raiseerror=True, verb=True,
                    backend=get_module_name(self.ncp))
        if dottest:
            Dottest(Mop, 2 * self.ns * self.nt2,
                    2 * self.nr * self.nt2,
                    raiseerror=True, verb=True,
                    backend=get_module_name(self.ncp))

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
            p0_minus = Rop * fd_plus.ravel()
            p0_minus = p0_minus.reshape(self.nt2, self.ns).T

        # Create data and inverse focusing functions
        d = Wsop * Rop * fd_plus.ravel()
        d = np.concatenate((d.reshape(self.nt2, self.ns),
                            self.ncp.zeros((self.nt2, self.ns), self.dtype)))

        # Invert for focusing functions
        if self.S is None:
            if self.ncp == np:
                f1_inv = lsqr(Mop, d.ravel(), **kwargs_solver)[0]
            else:
                f1_inv = cgls(Mop, d.ravel(),
                              x0=self.ncp.zeros(2*(2*self.nt-1)*self.nr, dtype=self.dtype),
                              **kwargs_solver)[0]
        else:
            f1_inv = fista(Mop * self.Sop, d.ravel(), **kwargs_solver)[0]
            f1_inv = self.Sop * f1_inv

        f1_inv = f1_inv.reshape(2 * self.nt2, self.nr)
        f1_inv_tot = f1_inv + np.concatenate((self.ncp.zeros((self.nt2,
                                                              self.nr),
                                                             dtype=self.dtype),
                                              fd_plus))
        f1_inv_minus = f1_inv_tot[:self.nt2].T
        f1_inv_plus = f1_inv_tot[self.nt2:].T
        if greens:
            # Create Green's functions
            g_inv = Gop * f1_inv_tot.ravel()
            g_inv = g_inv.reshape(2 * self.nt2, self.ns)
            g_inv_minus, g_inv_plus = -g_inv[:self.nt2].T, \
                                      np.fliplr(g_inv[self.nt2:].T)
        # Bring back to time axis with negative part
        f1_inv_minus = np.fft.ifftshift(f1_inv_minus, axes=1)
        f1_inv_plus = np.fft.fftshift(f1_inv_plus, axes=1)
        if rtm:
            p0_minus = np.fft.ifftshift(p0_minus, axes=1)
        if greens:
            g_inv_minus = np.fft.ifftshift(g_inv_minus, axes=1)
            g_inv_plus = np.fft.fftshift(g_inv_plus, axes=1)

        if rtm and greens:
            return f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus
        elif rtm:
            return f1_inv_minus, f1_inv_plus, p0_minus
        elif greens:
            return f1_inv_minus, f1_inv_plus, g_inv_minus, g_inv_plus
        else:
            return f1_inv_minus, f1_inv_plus

    def apply_multiplepoints(self, trav, G0=None, nfft=None,
                             rtm=False, greens=False, dottest=False,
                             usematmul=False, **kwargs_solver):
        r"""Marchenko redatuming for multiple points

        Solve the Marchenko redatuming inverse problem for multiple
        points given their direct arrival traveltime curves (``trav``)
        and waveforms (``G0``).

        Parameters
        ----------
        trav : :obj:`numpy.ndarray`
            Traveltime of first arrival from subsurface points to
            surface receivers of size :math:`[n_r \times n_{vs}]`
        G0 : :obj:`numpy.ndarray`, optional
            Direct arrival in time domain of size
            :math:`[n_r \times n_{vs} \times n_t]` (if None, create arrival
            using ``trav``)
        nfft : :obj:`int`, optional
            Number of samples in fft when creating the analytical direct wave
        rtm : :obj:`bool`, optional
            Compute and return rtm redatuming
        greens : :obj:`bool`, optional
            Compute and return Green's functions
        dottest : :obj:`bool`, optional
            Apply dot-test
        usematmul : :obj:`bool`, optional
            Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
            (``False``) in :py:class:`pylops.signalprocessing.Fredholm1` operator.
            Refer to Fredholm1 documentation for details.
        **kwargs_solver
            Arbitrary keyword arguments for chosen solver
            (:py:func:`scipy.sparse.linalg.lsqr` and
            :py:func:`pylops.optimization.solver.cgls` are used as default
            for numpy and cupy `data`, respectively)

        Returns
        ----------
        f1_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing focusing function of size
            :math:`[n_r \times n_{vs} \times n_t]`
        f1_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing focusing functionof size
            :math:`[n_r \times n_{vs} \times n_t]`
        p0_minus : :obj:`numpy.ndarray`
            Single-scattering standard redatuming upgoing Green's function
            of size :math:`[n_r \times n_{vs} \times n_t]`
        g_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing Green's function of size
            :math:`[n_r \times n_{vs} \times n_t]`
        g_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing Green's function of size
            :math:`[n_r \times n_{vs} \times n_t]`

        """
        nvs = trav.shape[1]

        # Create window
        trav_off = trav - self.toff
        trav_off = np.round(trav_off / self.dt).astype(np.int32)

        w = np.zeros((self.nr, nvs, self.nt), dtype=self.dtype)
        for ir in range(self.nr):
            for ivs in range(nvs):
                w[ir, ivs, :trav_off[ir, ivs]] = 1
        w = np.concatenate((w[:, :, 1:], np.flip(w, axis=-1)), axis=-1)
        if self.nsmooth > 0:
            smooth = np.ones(self.nsmooth, dtype=self.dtype) / self.nsmooth
            w = filtfilt(smooth, 1, w)
        w = to_cupy_conditional(self.Rtwosided_fft, w)

        # Create operators
        Rop = MDC(self.Rtwosided_fft, self.nt2, nv=nvs,
                  dt=self.dt, dr=self.dr, twosided=False,
                  conj=False, prescaled=self.prescaled,
                  usematmul=usematmul)
        R1op = MDC(self.Rtwosided_fft, self.nt2, nv=nvs,
                   dt=self.dt, dr=self.dr, twosided=False,
                   conj=True, prescaled=self.prescaled,
                   usematmul=usematmul)
        Wop = Diagonal(w.transpose(2, 0, 1).ravel())
        Iop = Identity(self.nr * nvs * self.nt2)
        Mop = Block([[Iop, -1 * Wop * Rop],
                     [-1 * Wop * R1op, Iop]]) * BlockDiag([Wop, Wop])
        Gop = Block([[Iop, -1 * Rop],
                     [-1 * R1op, Iop]])

        if dottest:
            Dottest(Gop, 2 * self.nr * nvs * self.nt2,
                    2 * self.nr * nvs * self.nt2,
                    raiseerror=True, verb=True,
                    backend=get_module_name(self.ncp))
        if dottest:
            Dottest(Mop, 2 * self.ns * nvs * self.nt2,
                    2 * self.nr * nvs * self.nt2,
                    raiseerror=True, verb=True,
                    backend=get_module_name(self.ncp))

        # Create input focusing function
        if G0 is None:
            if self.wav is not None and nfft is not None:
                G0 = np.zeros((self.nr, nvs, self.nt), dtype=self.dtype)
                for ivs in range(nvs):
                    G0[:, ivs] = (directwave(self.wav, trav[:, ivs],
                                             self.nt, self.dt,
                                             nfft=nfft, derivative=True)).T
                G0 = to_cupy_conditional(self.Rtwosided_fft, G0)
            else:
                logging.error('wav and/or nfft are not provided. '
                              'Provide either G0 or wav and nfft...')
                raise ValueError('wav and/or nfft are not provided. '
                                 'Provide either G0 or wav and nfft...')
        fd_plus = np.concatenate((self.ncp.zeros((self.nt - 1, self.nr, nvs),
                                                 dtype = self.dtype),
                                  np.flip(G0, axis=-1).transpose(2, 0, 1)))

        # Run standard redatuming as benchmark
        if rtm:
            p0_minus = Rop * fd_plus.ravel()
            p0_minus = p0_minus.reshape(self.nt2, self.ns,
                                        nvs).transpose(1, 2, 0)

        # Create data and inverse focusing functions
        d = Wop * Rop * fd_plus.ravel()
        d = np.concatenate((d.reshape(self.nt2, self.ns, nvs),
                            self.ncp.zeros((self.nt2, self.ns, nvs),
                                           dtype=self.dtype)))

        # Invert for focusing functions
        if self.ncp == np:
            f1_inv = lsqr(Mop, d.ravel(), **kwargs_solver)[0]
        else:
            f1_inv = cgls(Mop, d.ravel(),
                          x0=self.ncp.zeros(2 * (2 * self.nt - 1) *
                                            self.nr * nvs,
                                            dtype=self.dtype),
                          **kwargs_solver)[0]

        f1_inv = f1_inv.reshape(2 * self.nt2, self.nr, nvs)
        f1_inv_tot = \
            f1_inv + np.concatenate((self.ncp.zeros((self.nt2, self.nr, nvs),
                                                    dtype=self.dtype), fd_plus))
        f1_inv_minus = f1_inv_tot[:self.nt2].transpose(1, 2, 0)
        f1_inv_plus = f1_inv_tot[self.nt2:].transpose(1, 2, 0)

        if greens:
            # Create Green's functions
            g_inv = Gop * f1_inv_tot.ravel()
            g_inv = g_inv.reshape(2 * self.nt2, self.ns, nvs)
            g_inv_minus = -g_inv[:self.nt2].transpose(1, 2, 0)
            g_inv_plus = np.flip(g_inv[self.nt2:], axis=0).transpose(1, 2, 0)

        # Bring back to time axis with negative part
        f1_inv_minus = np.fft.ifftshift(f1_inv_minus, axes=-1)
        f1_inv_plus = np.fft.fftshift(f1_inv_plus, axes=-1)
        if rtm:
            p0_minus = np.fft.ifftshift(p0_minus, axes=-1)
        if greens:
            g_inv_minus = np.fft.ifftshift(g_inv_minus, axes=-1)
            g_inv_plus = np.fft.fftshift(g_inv_plus, axes=-1)

        if rtm and greens:
            return f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus
        elif rtm:
            return f1_inv_minus, f1_inv_plus, p0_minus
        elif greens:
            return f1_inv_minus, f1_inv_plus, g_inv_minus, g_inv_plus
        else:
            return f1_inv_minus, f1_inv_plus