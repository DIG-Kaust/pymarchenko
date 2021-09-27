import numpy as np
import matplotlib.pyplot as plt
from pylops.waveeqprocessing.wavedecomposition import _obliquity2D


def wavefield_separation(p, vz, dt, dr, rho, vel, nffts,
                         critical, ntaper, verb=False, plotflag=False):
    r"""Up/down wavefield separation

    Separate multi-component seismic data in their up- and down-going particle
    velocity components

    Parameters
    ----------
    p : :obj:`numpy.ndarray`
        Pressure data of size :math:`[n_s \times n_r \times n_t]`
    vz : :obj:`numpy.ndarray`
        Vertical particle velocity data of size
        :math:`[n_s \times n_r \times n_t]`
    dt : :obj:`float`
        Time sampling
    dr : :obj:`float`
        Receiver sampling
    rho : :obj:`float`
        Density along the receiver array (must be constant)
    vel : :obj:`float`
        Velocity along the receiver array (must be constant)
    nffts : :obj:`tuple`, optional
        Number of samples along the wavenumber and frequency axes
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor. For example, if
        ``critical=100`` only angles below the critical angle
        :math:`|k_x| < \frac{f(k_x)}{vel}` will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    verb : :obj:`bool`, optional
        Verbosity
    plotflag : :obj:`bool`, optional
        Plotting flag, if ``True`` plot results for the middle shot

    Returns
    -------
    vzup : :obj:`numpy.ndarray`
        Upgoing particle velocity data of size
        :math:`[n_s \times n_r \times n_t]`
    vzdown : :obj:`numpy.ndarray`
        Downgoing particle velocity data of size
        :math:`[n_s \times n_r \times n_t]`

    """
    ns, nr, nt = p.shape

    FFTop, OBLop = \
        _obliquity2D(nt, nr, dt, dr, rho, vel,
                     nffts=nffts, critical=critical,
                     ntaper=ntaper, composition=True)

    vzup, vzdown = np.zeros_like(vz), np.zeros_like(vz)
    for isrc in range(ns):
        if verb:
            print('Working with source %d' % isrc)
        # FK transform
        P = FFTop * p[isrc].ravel()

        # Scale Vz
        P_obl = OBLop * P.ravel()
        p_obl = FFTop.H * P_obl
        p_obl = np.real(p_obl.reshape(nr, nt))

        # Separation
        vzup[isrc] = (p_obl - vz[isrc]) / 2
        vzdown[isrc] = -(p_obl + vz[isrc]) / 2

        if plotflag and isrc == ns // 2:
            fig, axs = plt.subplots(1, 2, figsize=(9, 6))
            axs[0].imshow(p_obl.T, cmap='gray',
                          vmin=-0.1 * np.abs(vz).max(),
                          vmax=0.1 * np.abs(vz).max(),
                          extent=(0, nr, 0, nt))
            axs[0].set_title(r'$p$')
            axs[0].axis('tight')
            axs[1].imshow(vz[isrc].T, cmap='gray',
                          vmin=-0.1 * np.abs(vz).max(),
                          vmax=0.1 * np.abs(vz).max(),
                          extent=(0, nr, 0, nt))
            axs[1].set_title(r'$vzobl$')
            axs[1].axis('tight')

            fig, axs = plt.subplots(1, 2, figsize=(9, 6))
            axs[0].imshow(vzup[isrc].T, cmap='gray',
                          vmin=-0.1 * np.abs(vz).max(),
                          vmax=0.1 * np.abs(vz).max(),
                          extent=(0, nr, 0, nt))
            axs[0].set_title(r'$vzup$')
            axs[0].axis('tight')
            axs[1].imshow(vzdown[isrc].T, cmap='gray',
                          vmin=-0.1 * np.abs(vz).max(),
                          vmax=0.1 * np.abs(vz).max(),
                          extent=(0, nr, 0, nt))
            axs[1].set_title(r'$vzdown$')
            axs[1].axis('tight')

            plt.figure(figsize=(14, 3))
            plt.plot(p_obl[nr // 2], 'r', lw=2, label=r'$p$')
            plt.plot(vz[isrc, nr // 2], '--b', lw=2, label=r'$v_z$')
            plt.xlim(0, nt // 4)
            plt.legend()
            plt.figure(figsize=(14, 3))
            plt.plot(vzup[isrc, nr // 2], 'r', lw=2, label=r'$p^-$')
            plt.plot(vzdown[isrc, nr // 2], 'b', lw=2, label=r'$p^+$')
            plt.xlim(0, nt // 4)
            plt.legend()

    return vzup, vzdown
