# Cross-spatial correlation model
Inter-IM and intra-IM spatial correlation models for several IMs (PGA, PGV, Sa(T) at periods between 0.01–5.0 s, Saavg2(T), Saavg3(T), and FIV3(T) at periods between 0.1–2.0, 0.1-3.0, and 0.1-4.0s, respectively), using principal component analysis (PCA) and geostatistical tools.
Additional figures, beyond those presented in the paper, are provided in the folder Data_Resources.

# Reference
Monteiro, V.A, Aristeidou, S. and O’Reilly, G.J. (2026) ‘Spatial cross-correlation models for next-generation amplitude and cumulative intensity measures’, (under review)

# How to use
### Define the IM inter-distance. See example.py

```python
---------------  MAO2026 function  ---------------
Arguments:
    IM1 (float): First intensity measure.
    IM2 (float): Second intensity measure.
    h (float): 
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
For Saavg2 the range of periods is [0.1, 3.0]s
For Saavg3 the range of periods is [0.1, 2.0]s
For FIV3 the range of periods is [0.1, 4.0]s

from GetSpatialCorrelation import MAO2026

# Example
IM1 = "FIV3(1.5)"
IM2 = "Sa(0.1)"
h_distance = 20 #km
corr = MAO2026(IM1, IM2, h_distance, cluster=1, vs30=2)

print(f'Spatial Correlation between {IM1} and {IM2} at {h_distance} km is: {corr:.4f}')

```
