"""
Created by Vitor Monteiro, 23/09/2025

Global model with full PCs

For more details please see: 
Vitor A. Monteiro, Savvinos Aristeidou and Gerard J. O'Reilly (2026), 
'Spatial cross-correlation models for next-generation amplitude and cumulative intensity measures'
(under review)

"""

import re
import numpy as np
from scipy.interpolate import interp1d

T_list = np.array(['Sa(0.1)', 'Sa(0.5)', 'Sa(1.0)', 'Sa(2.0)', 'Sa(3.0)',
                   'Saavg2(0.1)', 'Saavg2(0.5)', 'Saavg2(1.0)', 'Saavg2(2.0)', 'Saavg2(3.0)',
                   'Saavg3(0.1)', 'Saavg3(0.5)', 'Saavg3(1.0)', 'Saavg3(2.0)', 'Saavg3(3.0)',
                   'FIV3(0.1)', 'FIV3(0.5)', 'FIV3(1.0)', 'FIV3(2.0)', 'FIV3(3.0)', 'PGA', 'PGV'])


pcs = np.array([
[0.13592564,	0.425895839,	-0.304606915],
[0.183322706,	0.186191781,	0.297506932],
[0.211835488,	-0.016367706,	0.397640891],
[0.22182256,	-0.174642377,	-0.001307827], 
[0.216286415,	-0.21174461,	-0.318130387],
[0.147704017,	0.419328798,	-0.230014625],
[0.205560782,	0.211019175,	0.224509564],
[0.232906094,	0.028446169,	0.227469062],
[0.242847633,	-0.122874441,	-0.045336653],
[0.235088181,	-0.174112632,	-0.208723859],
[0.154169146,	0.403140216,	-0.157418772],
[0.222262342,	0.118286784,	0.244865139],
[0.242951743,	-0.046412545,	0.066748683],
[0.237248768,	-0.154234894,	-0.205108842],
[0.219848464,	-0.176862315,	-0.286890567],
[0.228923987,	0.010300995,	0.232469904],
[0.231549529,	-0.079028972,	0.21415625],
[0.234055192,	-0.143068708,	0.044134728],
[0.230112173,	-0.157553595,	-0.113490261],
[0.223534065,	-0.149771535,	-0.119896544],
[0.168501568,	0.362160761,	-0.121224871],
[0.213470506,	0.052739442,	-0.036445378],
])

nested_para = np.array([
    [8.14, 13.20, 94.15, 0.00, 216.86],
    [0.95, 2.26, 61.67, 1.43, 226.49],
    [0.27, 0.41, 46.65, 0.64, 227.73],
])


pcs_dict = {label: pcs[i, :] for i, label in enumerate(T_list)}

def get_pc(label):
    if label in pcs_dict:
        return pcs_dict[label]

    # Extract type and period
    m = re.match(r"([A-Za-z0-9]+)\(([\d.]+)\)", label)
    if m:
        prefix, period = m.groups()
        period = float(period)
        # Collect all matching labels
        matching = [(float(re.match(rf"{prefix}\(([\d.]+)\)", l).group(1)), pcs[i,:])
                    for i,l in enumerate(T_list) if l.startswith(prefix+'(')]
        periods = np.array([p for p,_ in matching])
        pcs_vals = np.array([v for _,v in matching])

        log_periods = np.log(periods)
        log_target = np.log(period)
        interpolated_pc = np.zeros(pcs.shape[1])
        for i in range(pcs.shape[1]):
            f = interp1d(log_periods, pcs_vals[:,i], kind='linear', fill_value='extrapolate')
            interpolated_pc[i] = f(log_target)
        return interpolated_pc

    raise ValueError(f"Label '{label}' not found")


def GlobalMAO26(IM1, IM2, h):
    pc1 = get_pc(IM1)
    pc2 = get_pc(IM2)
    
    # Convert h to array if it's scalar, so calculations work element-wise
    h_array = np.atleast_1d(h)
    
    # Handle nugget effect element-wise
    Inugget = np.zeros_like(h_array)
    Inugget[h_array == 0] = 1

    C_h = np.zeros_like(h_array, dtype=float)
    Cii_0 = 0.0
    Cjj_0 = 0.0
    gamma = 0.0
    
    for i in range(pcs.shape[1]):  # Loop over number of PCs
        c0i, c1i, a1i, c2i, a2i = nested_para[i]
        # Element-wise operations
        gammai_h = (c0i*(1-Inugget) + c1i*(1-np.exp(-3*h_array/a1i)) + c2i*(1-np.exp(-3*h_array/a2i)))
        Cij_h = (c0i*Inugget + c1i*(np.exp(-3*h_array/a1i)) + c2i*(np.exp(-3*h_array/a2i)))
        silli = c0i + c1i + c2i
        
        
        gamma += pc1[i]*pc2[i]*gammai_h 
        C_h += pc1[i]*pc2[i]*Cij_h/0.95
        Cii_0 += pc1[i]*pc1[i]*silli/0.95
        Cjj_0 += pc2[i]*pc2[i]*silli/0.95
    
    
    C_h_normalized = C_h / np.sqrt(Cii_0*Cjj_0)
    C_h_normalized[C_h_normalized <= 0.001] = 0

    corr = C_h_normalized
    # If input was scalar, return scalar
    if np.isscalar(h):
        return float(corr[0])
    else:
        return corr





