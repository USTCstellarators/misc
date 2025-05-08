import numpy as np
from simsopt.mhd import Vmec, vmec_splines
from simsopt._core.util import Struct
import matplotlib.pyplot as plt

def triple_product_spline(vs, s, theta, phi):
    """
    Stand‑alone computation of
        f_T = (∇ψ × ∇|B|) · ∇(B·∇|B|)
    together with its normalised form and flux‑surface averages.

    Parameters
    ----------
    vs : simsopt.mhd.Vmec | simsopt.mhd.vmec_splines
        Loaded VMEC equilibrium or its spline wrapper.
    s : float | (ns,) ndarray
        Normalised toroidal flux surfaces (0‒1).  Scalar or 1‑d array.
    theta_vmec : float | (ntheta,) ndarray
        VMEC poloidal angle grid (0‒2π).
    phi : float | (nphi,) ndarray
        Standard toroidal angle grid (0‒2π/Nfp).

    Returns
    -------
    A Struct with the following attributes:
        f_T         : ndarray, shape (ns, ntheta, nphi)
            Dimensional quantity; units T⁴ m⁻².
        f_T_norm    : ndarray, same shape
            Dimensionless; normalised by L_ref² / B_ref⁴.
        f_T_fsa_dim : ndarray, shape (ns,)
            Flux‑surface average of f_T_dim.
        f_T_fsa_norm : ndarray, shape (ns,)
            Flux‑surface average of f_T_norm.
        f_T_hat     : ndarray, shape (ns,)
        f_T_hat_desc : ndarray, shape (ns,)
    """
    # If given a Vmec object, convert it to vmec_splines:
    R0 = vs.wout.Rmajor_p
    if isinstance(vs, Vmec):
        vs = vmec_splines(vs)

    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)

    # Handle theta
    try:
        ntheta = len(theta)
    except:
        theta = [theta]
    theta_vmec = np.array(theta)
    if theta_vmec.ndim == 1:
        ntheta = len(theta_vmec)
    elif theta_vmec.ndim == 3:
        ntheta = theta_vmec.shape[1]
    else:
        raise ValueError("theta argument must be a float, 1d array, or 3d array.")

    # Handle phi
    try:
        nphi = len(phi)
    except:
        phi = [phi]
    phi = np.array(phi)
    if phi.ndim == 1:
        nphi = len(phi)
    elif phi.ndim == 3:
        nphi = phi.shape[2]
    else:
        raise ValueError("phi argument must be a float, 1d array, or 3d array.")

    # If theta and phi are not already 3D, make them 3D:
    if theta_vmec.ndim == 1:
        theta_vmec = np.kron(np.ones((ns, 1, nphi)), theta_vmec.reshape(1, ntheta, 1))
    if phi.ndim == 1:
        phi = np.kron(np.ones((ns, ntheta, 1)), phi.reshape(1, 1, nphi))

    dtheta_vmec = theta[1] - theta[0]
    dphi_vmec   = phi[0, 0, 1] - phi[0, 0, 0]

    # Shorthand:
    mnmax_nyq = vs.mnmax_nyq
    xm_nyq = vs.xm_nyq
    xn_nyq = vs.xn_nyq

    gmnc = np.zeros((ns, mnmax_nyq))
    bmnc = np.zeros((ns, mnmax_nyq))
    bsupumnc = np.zeros((ns, mnmax_nyq))
    bsupvmnc = np.zeros((ns, mnmax_nyq))
    for jmn in range(mnmax_nyq):
        gmnc[:, jmn] = vs.gmnc[jmn](s)
        bmnc[:, jmn] = vs.bmnc[jmn](s)
        bsupumnc[:, jmn] = vs.bsupumnc[jmn](s)
        bsupvmnc[:, jmn] = vs.bsupvmnc[jmn](s)

    angle = xm_nyq[:, None, None, None] * theta_vmec[None, :, :, :] \
            - xn_nyq[:, None, None, None] * phi[None, :, :, :]
    cosangle    = np.cos(angle)
    sinangle    = np.sin(angle)
    msinangle   = xm_nyq[:, None, None, None] * sinangle
    nsinangle   = xn_nyq[:, None, None, None] * sinangle

    m2cosangle  = xm_nyq[:, None, None, None]**2 * cosangle
    n2cosangle  = xn_nyq[:, None, None, None]**2 * cosangle
    mncosangle  = xm_nyq[:, None, None, None] * xn_nyq[:, None, None, None] * cosangle

    sqrt_g_vmec             = np.einsum('ij,jikl->ikl', gmnc, cosangle)
    d_B_d_theta_vmec        = np.einsum('ij,jikl->ikl', -bmnc, msinangle)
    d_B_d_phi               = np.einsum('ij,jikl->ikl', bmnc, nsinangle)
    d2_B_d_theta_vmec2      = np.einsum('ij,jikl->ikl', -bmnc, m2cosangle)
    d2_B_d_phi2             = np.einsum('ij,jikl->ikl', -bmnc, n2cosangle)
    d2_B_d_theta_vmec_d_phi = np.einsum('ij,jikl->ikl',  bmnc, mncosangle)

    B_sup_theta_vmec            = np.einsum('ij,jikl->ikl', bsupumnc, cosangle)
    B_sup_phi                   = np.einsum('ij,jikl->ikl', bsupvmnc, cosangle)
    d_B_sup_phi_d_theta_vmec    = np.einsum('ij,jikl->ikl', -bsupvmnc, msinangle)
    d_B_sup_phi_d_phi           = np.einsum('ij,jikl->ikl', bsupvmnc, nsinangle)

    d_B_sup_theta_vmec_d_theta_vmec = np.einsum('ij,jikl->ikl', -bsupumnc, msinangle)
    d_B_sup_theta_vmec_d_phi        = np.einsum('ij,jikl->ikl', bsupumnc, nsinangle)

    '''
    IF NUMBERICLE
    '''
    # BdotgradB = (
    #         B_sup_theta_vmec * d_B_d_theta_vmec +
    #         B_sup_phi         * d_B_d_phi
    # )
    # g_theta = np.gradient(BdotgradB,dtheta_vmec,axis = 1)
    # g_phi   = np.gradient(BdotgradB,dphi_vmec,axis = 2)
    '''
    Analytic
    '''
    g_theta = (
            d_B_sup_theta_vmec_d_theta_vmec * d_B_d_theta_vmec
            + B_sup_theta_vmec               * d2_B_d_theta_vmec2
            + d_B_sup_phi_d_theta_vmec       * d_B_d_phi
            + B_sup_phi                      * d2_B_d_theta_vmec_d_phi
        )
    
    g_phi = (
        d_B_sup_theta_vmec_d_phi * d_B_d_theta_vmec
        + B_sup_theta_vmec         * d2_B_d_theta_vmec_d_phi   
        + d_B_sup_phi_d_phi        * d_B_d_phi
        + B_sup_phi                * d2_B_d_phi2
    )

    edge_toroidal_flux_over_2pi = -vs.phiedge / (2 * np.pi)
    psi_r_over_sqrtg = edge_toroidal_flux_over_2pi/sqrt_g_vmec

    L_reference = vs.Aminor_p
    B_reference = 2 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)

    '''
    f_T: T^4/m^2,multi-dimensional
    '''
    f_T = psi_r_over_sqrtg * (d_B_d_theta_vmec * g_phi - d_B_d_phi * g_theta) # T^4/m^2

    '''
    f_T_norm: normalized by L_ref^2 / B_ref^4, multi-dimensional
    '''
    f_T_norm = f_T * (L_reference**2 / B_reference**4) 

    '''
    Surface_averaged f_T, f_T_norm
    '''
    weight          = sqrt_g_vmec
    num_f_T         = np.sum(np.abs(f_T) * weight, axis=(1,2)) * dtheta_vmec * dphi_vmec
    num_f_T_norm    = np.sum(np.abs(f_T_norm) * weight, axis=(1,2)) * dtheta_vmec * dphi_vmec
    den             = np.sum(weight, axis=(1,2)) * dtheta_vmec * dphi_vmec
    f_T_fsa         = num_f_T / den
    f_T_norm_fsa    = num_f_T_norm / den

    '''
    hat 
    '''
    f_T_hat = np.mean(np.abs(f_T)*weight, axis=(1,2)) / np.mean(weight, axis=(1,2)) * L_reference**2 / B_reference**4

    '''
    hat desc,R0=Major radius, DESC using it for target
    '''
    modB = np.einsum('ij,jikl->ikl', bmnc, cosangle)
    B0 = np.mean(modB * sqrt_g_vmec)/np.mean(sqrt_g_vmec)
    f_T_hat_desc = np.mean(np.abs(f_T) * sqrt_g_vmec, axis=(1,2)) / np.mean(sqrt_g_vmec, axis=(1,2)) * R0**2 / B0**4


    results = Struct()
    variables = ['f_T','f_T_norm','f_T_fsa','f_T_norm_fsa','f_T_hat','f_T_hat_desc']
    for v in variables:
        results.__setattr__(v, eval(v))

    return results

from simsopt.mhd import Vmec
filename = './wouts/wout_qa_desc.nc'
v = Vmec(filename)

s = np.linspace(0.1,1,10)       
theta = np.linspace(0, 2*np.pi, 64)    
phi   = np.linspace(0, 2*np.pi/2,64)  

comp = triple_product_spline(v, s, theta, phi)
f_T = comp.f_T
f_T_fsa = comp.f_T_hat_desc

plt.figure(figsize=(8, 6))
plt.plot(s, f_T_fsa, marker='o', markersize=5)
plt.xlabel(r'$s$', fontsize=14)
plt.ylabel(r'$f_T$', fontsize=14)
plt.yscale('log')
plt.title(r'$f_T$ on each flux surface', fontsize=16)
plt.grid()
plt.savefig('ft1d.png')


plt.figure(figsize=(8, 6))
plt.contourf(phi, theta, f_T[-1, :, :])
plt.colorbar()
plt.savefig('ft2d.png')
