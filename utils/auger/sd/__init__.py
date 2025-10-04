__all__ = ['moni', 'histo']

from ...binaries import np

def muLDF(R, n_mu_ref, zenith,
          beta=None, 
          energy=None):

    r0, r450 = 320, 450
    alpha, gamma = 0.75, 3
    umd_active_area = 10.496
    rho_mu = n_mu_ref / (umd_active_area * np.cos(zenith))

    if beta is None:
        assert energy is not None, "Need energy if beta is None!"

        # from Joaquins thesis
        m, b0, b1 = -1.21, 2.71, 0.2
        b = b1 * (np.log10(energy) - 17.8) + b0
        beta = m * (1/np.cos(zenith) - 1.2) + b

    f_alpha = R / r450
    f_beta = (1 + R/r0) / (1 + r450/r0)
    f_gamma = (1 + (R/(10*r0))**2) / (1 + (r450/(10*r0))**2)

    return rho_mu * f_alpha**(-alpha) * f_beta**(-beta) * f_gamma**(-gamma)