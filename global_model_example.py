"""
Author: Vitor Monteiro, IUSS-Pavia, 2025 PhD Candidate, 18/09/2025.


Compute cross-spatial correlation coefficients using Principal Component Analysis
(PCA) for Sa(T*), PGA, PGV, Saavg2(T), Saavg3(T), and FIV3(T) for NGA-West2
and ESM databases for T = [0.1-3]s

For more details please see: 
Monteiro, V.A, Aristeidou, S. and O’Reilly, G.J. (2026) ‘Spatial cross-correlation
models for next-generation amplitude and cumulative intensity measures’ 
(under review)


---------------  MAO2026 function  ---------------

Arguments:
    IM1 (float): First intensity measure (e.g., ground motion parameter).
    IM2 (float): Second intensity measure.
    h (float): Depth or scaling parameter.
Returns:
    corr (float): Correlation value retrieved for the given IM1, IM2, and distance h.
                    
                    
IMs available are 'Sa', 'Saavg2', 'Saavg3', 'FIV3', 'PGA', and 'PGV'.
For Sa, Saavg2, Saavg3 and FIV3 the range of periods is [0.1, 3.0]s
"""


from GlobalModelMAO26 import GlobalMAO26

# Example
IM1 = "FIV3(0.6)"
IM2 = "Saavg2(1.0)"
import numpy as np
h_distance = np.linspace(0,150,151) #km
for h in h_distance:
    corr = GlobalMAO26(IM1, IM2, h)
    print(corr)

print(f'Spatial Correlation between {IM1} and {IM2} at {h_distance} km is: {corr:.4f}')


