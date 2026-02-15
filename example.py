"""
Author: Vitor Monteiro, IUSS-Pavia, 2025 PhD Candidate, 18/09/2025.


Compute cross-spatial correlation coefficients using Principal Component Analysis
(PCA) for Sa(T*), PGA, PGV, Saavg2(T**), Saavg3(T**), and FIV3(T**) for NGA-West2
and ESM databases for T* = [0.01-5.0]s and T** = [0.1-4.0]s.

For more details please see: 
Monteiro, V.A, Aristeidou, S. and O’Reilly, G.J. (2025) ‘Modelling Cross-Spatial
Correlation for Amplitude and Next-Generation Intensity Measures using principal
component analysis and geostatistics for active shallow crustal zones’ 
(under review)


---------------  MAO2025 function  ---------------

Arguments:
    IM1 (float): First intensity measure (e.g., ground motion parameter).
    IM2 (float): Second intensity measure.
    h (float): Depth or scaling parameter.
    cluster (int, optional): Clustering flag. Defaults to 0.
        - 0 → use "non_cluster" database (ignores vs30).
        - 1 → use "cluster" database (requires vs30).
    vs30 (int, optional): Site condition index. Used only if cluster=1.
        - 1 → low Vs30 (soft soil).
        - 2 → high Vs30 (stiff soil).
        If cluster=0, this is ignored.

Returns:
    corr (float): Correlation value retrieved from the chosen database,
                    for the given IM1, IM2, and distance h.
                    
                    
IMs available are 'Sa', 'Saavg2', 'Saavg3', 'FIV3', 'PGA', and 'PGV'.
For Sa the range of periods is [0.01, 5.0]s
For Saavg2, Saavg3, and FIV3 the range of periods is [0.1, 4.0]s

"""


from GetSpatialCorrelation import MAO2025

# Example
IM1 = "FIV3(0.6)"
IM2 = "FIV3(0.6)"
import numpy as np
h_distance = np.linspace(0,150,1500) #km
for h in h_distance:
    corr = MAO2025(IM1, IM2, h, cluster=0, vs30=2)
    print(corr)

#print(f'Spatial Correlation between {IM1} and {IM2} at {h_distance} km is: {corr:.4f}')
