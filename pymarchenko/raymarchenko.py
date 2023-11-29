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


class RayleighMarchenko():
    r"""Rayleigh-Marchenko redatuming

    Solve multi-dimensional Rayleigh-Marchenko redatuming problem using
    :py:func:`scipy.sparse.linalg.lsqr` iterative solver.

    Parameters
    ----------
    VZplus : :obj:`numpy.ndarray`
        Multi-dimensional downgoing particle velocity data in time or frequency
        domain of size :math:`[n_s \times n_r \times n_t/n_{fmax}]`. If
        provided in time, ``VZplus`` should not be of complex type. If
        provided in frequency, ``VZpl`` should contain the positive time axis
        followed by the negative one.
    VZminus : :obj:`numpy.ndarray`
        Multi-dimensional upgoing particle velocity data in time or frequency
        domain of size :math:`[n_s \times n_r \times n_t/n_{fmax}]`. If
        provided in time, ``VZminus`` should not be of complex type. If
        provided in frequency, ``VZminus`` should contain the positive time axis
        followed by the negative one.
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
    saveVt : :obj:`bool`, optional
        Save ``VZplus`` and ``VZplus^H`` (and ``VZminus`` and ``VZminus^H``)
        to speed up the computation of adjoint of
        :class:`pylops.signalprocessing.Fredholm1` (``True``) or create
        ``VZplus^H`` and ``VZminus^H`` on-the-fly (``False``)
        Note that ``saveVt=True`` will be faster but double the amount of
        required memory
    prescaled : :obj:`bool`, optional
        Apply scaling to ``Vzplus`` and ``VZminus`` (``False``) or
        not (``False``) when performing spatial and temporal summations within
        the :class:`pylops.waveeqprocessing.MDC` operator.

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
    Rayleigh-Marchenko redatuming is a method that allows to produce correct
    subsurface-to-surface responses given the availability of up- and down-
    separated particle velocity data and a macro-velocity model [1]_.

    The Rayleigh-Marchenko equations can be written in a compact matrix
    form and solved by means of iterative solvers such as LSQR:

    .. math::
        \begin{bmatrix}
           -\Theta \mathbf{V}_z^- \mathbf{f_d^+}  \\
           -\Theta \mathbf{V}_z^{+*} \mathbf{f_d^+}
        \end{bmatrix} =
        \begin{bmatrix}
           \Theta \mathbf{V}_z^+     &   \Theta \mathbf{V}_z^-  \\
           \Theta \mathbf{V}_z^{-*} & \Theta \mathbf{V}_z^{+*}
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{f^-}  \\
           \mathbf{f_m^+}
        \end{bmatrix}

    Finally the subsurface Green's functions can be obtained applying the
    following operator to the retrieved focusing functions

    .. math::
        \begin{bmatrix}
           -\mathbf{p}^-  \\
           \mathbf{p}^{+*}
        \end{bmatrix} =
        \begin{bmatrix}
           \mathbf{V}_z^+    &   \mathbf{V}_z^-  \\
           \mathbf{V}_z^{-*} & \mathbf{V}_z^{+*}
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{f^-}  \\
           \mathbf{f^+}
        \end{bmatrix}

    .. [1] Ravasi, M., "Rayleigh-Marchenko redatuming for target-oriented,
        true-amplitude imaging", Geophysics, vol. 82, pp. S439-S452. 2017.

    """
    def __init__(self, VZplus, VZminus, dt=0.004, nt=None, dr=1.,
                 nfmax=None, wav=None, toff=0.0, nsmooth=10,
                 dtype='float64', saveVt=True, prescaled=False):

        # Save inputs into class
        self.dt = dt
        self.dr = dr
        self.wav = wav
        self.toff = toff
        self.nsmooth = nsmooth
        self.saveVt = saveVt
        self.prescaled = prescaled
        self.dtype = dtype
        self.explicit = False
        self.ncp = get_array_module(VZplus)

        # Infer dimensions of R
        if not np.iscomplexobj(VZplus):
            self.ns, self.nr, self.nt = VZplus.shape
            self.nfmax = nfmax
        else:
            self.ns, self.nr, self.nfmax = VZplus.shape
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
        if not np.iscomplexobj(VZplus):
            VZplus = np.concatenate((VZplus,
                                     self.ncp.zeros((self.ns, self.nr,
                                                     self.nt - 1),
                                                    dtype=VZplus.dtype)),
                                    axis=-1)
            VZplus_fft = np.fft.rfft(VZplus, self.nt2,
                                     axis=-1) / np.sqrt(self.nt2)
            self.VZplus_fft = VZplus_fft[..., :nfmax]
        else:
            self.VZplus_fft = VZplus
        if not np.iscomplexobj(VZminus):
            VZminus = np.concatenate((VZminus,
                                      self.ncp.zeros((self.ns, self.nr,
                                                      self.nt - 1),
                                                     dtype=VZminus.dtype)),
                                     axis=-1)
            VZminus_fft = np.fft.rfft(VZminus, self.nt2,
                                      axis=-1) / np.sqrt(self.nt2)
            self.VZminus_fft = VZminus_fft[..., :nfmax]
        else:
            self.VZminus_fft = VZminus

        # bring frequency to first dimension
        self.VZplus_fft = self.VZplus_fft.transpose(2, 0, 1)
        self.VZminus_fft = self.VZminus_fft.transpose(2, 0, 1)

    def apply_onepoint(self, travsrc, travrec, G0=None, nfft=None, rtm=False, greens=False,
                       dottest=False, usematmul=False, **kwargs_solver):
        r"""Rayleigh-Marchenko redatuming for one point

        Solve the Rayleigh-Marchenko redatuming inverse problem for a single point
        given its direct arrival traveltime curves (``travsrc`` and
        ``travrec``) and waveform (``G0``).

        Parameters
        ----------
        travsrc : :obj:`numpy.ndarray`
            Traveltime of first arrival from subsurface point to
            surface sources of size :math:`[n_s \times 1]`
        travsrc : :obj:`numpy.ndarray`
            Traveltime of first arrival from subsurface point to
            surface receivers of size :math:`[n_r \times 1]`
        G0 : :obj:`numpy.ndarray`, optional
            Direct arrival in time domain of size :math:`[n_r \times n_t]`
            (if None, create arrival using ``travrec``)
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
            Inverted upgoing focusing function of size :math:`[n_r \times n_t]`
        f1_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing focusing function
            of size :math:`[n_r \times n_t]`
        p0_minus : :obj:`numpy.ndarray`
            Single-scattering standard redatuming upgoing Green's function of
            size :math:`[n_s \times n_t]`
        g_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing Green's function of size :math:`[n_s \times n_t]`
        g_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing Green's function
            of size :math:`[n_s \times n_t]`

        """
        # Create windows
        travsrc_off = travsrc - self.toff
        travsrc_off = np.round(travsrc_off / self.dt).astype(np.int32)
        travrec_off = travrec - self.toff
        travrec_off = np.round(travrec_off / self.dt).astype(np.int32)

        ws = np.zeros((self.ns, self.nt), dtype=self.dtype)
        wr = np.zeros((self.nr, self.nt), dtype=self.dtype)
        for ir in range(self.ns):
            ws[ir, :travsrc_off[ir]] = 1
        for ir in range(self.nr):
            wr[ir, :travrec_off[ir]] = 1
        ws = np.hstack((ws[:, 1:], np.fliplr(ws)))
        wr = np.hstack((wr[:, 1:], np.fliplr(wr)))

        if self.nsmooth > 0:
            smooth = np.ones(self.nsmooth, dtype=self.dtype) / self.nsmooth
            ws = filtfilt(smooth, 1, ws)
            wr = filtfilt(smooth, 1, wr)
        ws = to_cupy_conditional(self.VZminus_fft, ws)
        wr = to_cupy_conditional(self.VZminus_fft, wr)

        # Create operators
        Vzuop = MDC(self.VZminus_fft, self.nt2, nv=1, dt=self.dt, dr=self.dr,
                    twosided=False, conj=False,
                    saveGt=self.saveVt, prescaled=self.prescaled,
                    usematmul=usematmul)
        Vzu1op = MDC(self.VZminus_fft, self.nt2, nv=1, dt=self.dt, dr=self.dr,
                     twosided=False, conj=True,
                     saveGt=self.saveVt, prescaled=self.prescaled,
                     usematmul=usematmul)
        Vzdop = MDC(self.VZplus_fft, self.nt2, nv=1, dt=self.dt, dr=self.dr,
                    twosided=False, conj=False,
                    saveGt=self.saveVt, prescaled=self.prescaled,
                    usematmul=usematmul)
        Vzd1op = MDC(self.VZplus_fft, self.nt2, nv=1, dt=self.dt, dr=self.dr,
                     twosided=False, conj=True,
                     saveGt=self.saveVt, prescaled=self.prescaled,
                     usematmul=usematmul)

        Wfop = Diagonal(wr.T.flatten())
        Wgop = Diagonal(ws.T.flatten())

        Dop = Block([[Wgop * Vzdop, Wgop * Vzuop],
                     [Wgop * Vzu1op, Wgop * Vzd1op]])
        Mop = Dop * BlockDiag([Wfop, Wfop])
        Gop = Block([[Vzdop, Vzuop],
                     [Vzu1op, Vzd1op]])

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
                G0 = (directwave(self.wav, travrec, self.nt,
                                 self.dt, nfft=nfft, derivative=True)).T
                G0 = to_cupy_conditional(self.VZminus_fft, G0)
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
            p0_minus = Vzuop * fd_plus.flatten()
            p0_minus = p0_minus.reshape(self.nt2, self.ns).T

        # Create data and inverse focusing functions
        dinp = np.concatenate((self.ncp.zeros((self.nt2, self.nr), self.dtype),
                               fd_plus), axis=0)
        d = -Dop * dinp.ravel()

        # Invert for focusing functions
        if self.ncp == np:
            f1_inv = lsqr(Mop, d.flatten(), **kwargs_solver)[0]
        else:
            f1_inv = cgls(Mop, d.flatten(),
                          x0=self.ncp.zeros(2*(2*self.nt-1)*self.nr, dtype=self.dtype),
                          **kwargs_solver)[0]

        f1_inv = f1_inv.reshape(2 * self.nt2, self.nr)
        f1_inv_tot = f1_inv + np.concatenate((self.ncp.zeros((self.nt2,
                                                              self.nr),
                                                             dtype=self.dtype),
                                              fd_plus))
        f1_inv_minus = f1_inv_tot[:self.nt2].T
        f1_inv_plus = f1_inv_tot[self.nt2:].T
        if greens:
            # Create Green's functions
            g_inv = Gop * f1_inv_tot.flatten()
            g_inv = g_inv.reshape(2 * self.nt2, self.ns)
            g_inv_minus, g_inv_plus = -g_inv[:self.nt2].T, \
                                      np.fliplr(g_inv[self.nt2:].T)
        # Bring back to time axis with negative part
        f1_inv_minus = np.fft.ifftshift(f1_inv_minus, axes=1)
        f1_inv_plus = np.fft.ifftshift(f1_inv_plus, axes=1)
        if rtm:
            p0_minus = np.fft.ifftshift(p0_minus, axes=1)
        if greens:
            g_inv_minus = np.fft.ifftshift(g_inv_minus, axes=1)
            g_inv_plus = np.fft.ifftshift(g_inv_plus, axes=1)

        if rtm and greens:
            return f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus
        elif rtm:
            return f1_inv_minus, f1_inv_plus, p0_minus
        elif greens:
            return f1_inv_minus, f1_inv_plus, g_inv_minus, g_inv_plus
        else:
            return f1_inv_minus, f1_inv_plus

    def apply_multiplepoints(self, travsrc, travrec, G0=None, nfft=None,
                             rtm=False, greens=False,
                             dottest=False, usematmul=False, **kwargs_solver):
        r"""Rayleigh-Marchenko redatuming for multiple points

        Solve the Rayleigh-Marchenko redatuming inverse problem for multiple
        points given their direct arrival traveltime curves (``travsrc`` and
        ``travrec``) and waveforms (``G0``).

        Parameters
        ----------
        travsrc : :obj:`numpy.ndarray`
            Traveltime of first arrival from subsurface point to
            surface sources of size :math:`[n_s \times 1]`
        travsrc : :obj:`numpy.ndarray`
            Traveltime of first arrival from subsurface point to
            surface receivers of size :math:`[n_r \times 1]`
        G0 : :obj:`numpy.ndarray`, optional
            Direct arrival in time domain of size
            :math:`[n_r \times n_{vs} \times n_t]` (if None, create arrival
            using ``travrec``)
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
        nvs = travsrc.shape[1]

        # Create window
        travsrc_off = travsrc - self.toff
        travsrc_off = np.round(travsrc_off / self.dt).astype(np.int32)
        travrec_off = travrec - self.toff
        travrec_off = np.round(travrec_off / self.dt).astype(np.int32)

        ws = np.zeros((self.ns, nvs, self.nt), dtype=self.dtype)
        wr = np.zeros((self.nr, nvs, self.nt), dtype=self.dtype)
        for ir in range(self.ns):
            for ivs in range(nvs):
                ws[ir, ivs, :travsrc_off[ir, ivs]] = 1
        for ir in range(self.nr):
            for ivs in range(nvs):
                wr[ir, ivs, :travrec_off[ir, ivs]] = 1
        ws = np.concatenate((ws[:, :, 1:], np.flip(ws, axis=-1)), axis=-1)
        wr = np.concatenate((wr[:, :, 1:], np.flip(wr, axis=-1)), axis=-1)
        if self.nsmooth > 0:
            smooth = np.ones(self.nsmooth, dtype=self.dtype) / self.nsmooth
            ws = filtfilt(smooth, 1, ws)
            wr = filtfilt(smooth, 1, wr)
        ws = to_cupy_conditional(self.VZminus_fft, ws)
        wr = to_cupy_conditional(self.VZminus_fft, wr)

        # Create operators
        Vzuop = MDC(self.VZminus_fft, self.nt2, nv=nvs, dt=self.dt, dr=self.dr,
                    twosided=False, conj=False,
                    saveGt=self.saveVt, prescaled=self.prescaled,
                    usematmul=usematmul)
        Vzu1op = MDC(self.VZminus_fft, self.nt2, nv=nvs, dt=self.dt, dr=self.dr,
                     twosided=False, conj=True,
                     saveGt=self.saveVt, prescaled=self.prescaled,
                     usematmul=usematmul)
        Vzdop = MDC(self.VZplus_fft, self.nt2, nv=nvs, dt=self.dt, dr=self.dr,
                    twosided=False, conj=False,
                    saveGt=self.saveVt, prescaled=self.prescaled,
                    usematmul=usematmul)
        Vzd1op = MDC(self.VZplus_fft, self.nt2, nv=nvs, dt=self.dt, dr=self.dr,
                     twosided=False, conj=True,
                     saveGt=self.saveVt, prescaled=self.prescaled,
                     usematmul=usematmul)
        Wfop = Diagonal(wr.transpose(2, 0, 1).flatten())
        Wgop = Diagonal(ws.transpose(2, 0, 1).flatten())

        Dop = Block([[Wgop * Vzdop, Wgop * Vzuop],
                     [Wgop * Vzu1op, Wgop * Vzd1op]])
        Mop = Dop * BlockDiag([Wfop, Wfop])
        Gop = Block([[Vzdop, Vzuop],
                     [Vzu1op, Vzd1op]])

        if dottest:
            Dottest(Gop, 2 * self.ns * nvs * self.nt2,
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
                    G0[:, ivs] = (directwave(self.wav, travrec[:, ivs],
                                             self.nt, self.dt,
                                             nfft=nfft, derivative=True)).T
                G0 = to_cupy_conditional(self.VZminus_fft, G0)
            else:
                logging.error('wav and/or nfft are not provided. '
                              'Provide either G0 or wav and nfft...')
                raise ValueError('wav and/or nfft are not provided. '
                                 'Provide either G0 or wav and nfft...')
        fd_plus = np.concatenate((self.ncp.zeros((self.nt - 1, self.nr, nvs),
                                                 dtype=self.dtype),
                                  np.flip(G0, axis=-1).transpose(2, 0, 1)))

        # Run standard redatuming as benchmark
        if rtm:
            p0_minus = Vzuop * fd_plus.flatten()
            p0_minus = p0_minus.reshape(self.nt2, self.ns,
                                        nvs).transpose(1, 2, 0)

        # Create data and inverse focusing functions
        dinp = np.concatenate((self.ncp.zeros((self.nt2, self.nr, nvs),
                                              self.dtype),
                               fd_plus), axis=0)
        d = -Dop * dinp.ravel()

        # Invert for focusing functions
        if self.ncp == np:
            f1_inv = lsqr(Mop, d.flatten(), **kwargs_solver)[0]
        else:
            f1_inv = cgls(Mop, d.flatten(),
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
            g_inv = Gop * f1_inv_tot.flatten()
            g_inv = g_inv.reshape(2 * self.nt2, self.ns, nvs)
            g_inv_minus = -g_inv[:self.nt2].transpose(1, 2, 0)
            g_inv_plus = np.flip(g_inv[self.nt2:], axis=0).transpose(1, 2, 0)

        # Bring back to time axis with negative part
        f1_inv_minus = np.fft.ifftshift(f1_inv_minus, axes=-1)
        f1_inv_plus = np.fft.ifftshift(f1_inv_plus, axes=-1)
        if rtm:
            p0_minus = np.fft.ifftshift(p0_minus, axes=-1)
        if greens:
            g_inv_minus = np.fft.ifftshift(g_inv_minus, axes=-1)
            g_inv_plus = np.fft.ifftshift(g_inv_plus, axes=-1)

        if rtm and greens:
            return f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus
        elif rtm:
            return f1_inv_minus, f1_inv_plus, p0_minus
        elif greens:
            return f1_inv_minus, f1_inv_plus, g_inv_minus, g_inv_plus
        else:
            return f1_inv_minus, f1_inv_plus