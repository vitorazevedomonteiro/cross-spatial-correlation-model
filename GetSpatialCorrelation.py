import numpy as np
import pandas as pd
import os
from pathlib import Path

my_path = Path(__file__).parent

sa_periods = np.array([0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
saavg2_periods = np.array([0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
saavg3_periods = np.array([0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0])
fiv3_periods = np.array([0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])

ims_priority = ['Sa', 'Saavg2', 'Saavg3', 'FIV3', 'PGA', 'PGV']


# Function that extract the prefix and the period. E.g., Sa(0.01), "Sa" and "0.01" 
def extract_prefix_period(im):
    if '(' in im and ')' in im:
        prefix = im.split('(')[0]
        period = float(im.split('(')[1].replace(')', ''))
        return prefix, period
    return im, None

def format_im_name(prefix, period):
    if period is None:
        return prefix
    else:
        return f"{prefix}({period})"

def interpolate_first_row_and_copy_second(low_df, high_df, ratio):
    interp_df = low_df.copy()

    # Interpolate first row (PC1) columns 1 to 5
    interp_df.iloc[0, 1:6] = (
        low_df.iloc[0, 1:6] + (high_df.iloc[0, 1:6] - low_df.iloc[0, 1:6]) * ratio
    )

    # Handle second row (PC2)
    if len(low_df) > 1 and len(high_df) > 1:
        # Both have PC2: interpolate as normal
        interp_df.iloc[1, 1:6] = (
            low_df.iloc[1, 1:6] + (high_df.iloc[1, 1:6] - low_df.iloc[1, 1:6]) * ratio
        )
    elif len(low_df) > 1:
        # Only low_df has PC2: copy it directly
        interp_df = pd.concat([
            interp_df.iloc[[0]],         # keep interpolated row 0
            low_df.iloc[[1]]             # copy row 1 from low_df
        ], ignore_index=True)
    elif len(high_df) > 1:
        # Only high_df has PC2: copy it directly
        interp_df = pd.concat([
            interp_df.iloc[[0]],         # keep interpolated row 0
            high_df.iloc[[1]]            # copy row 1 from high_df
        ], ignore_index=True)
    else:
        # Neither has PC2 → return only row 0
        interp_df = interp_df.iloc[[0]]

    return interp_df.select_dtypes(include=[np.number]).to_numpy()



# Function that interpolate the model parameters in case an im is selected and is the period is not found
def interpolate_parameters(im1, im2, database):
    
    # Extract prefix and period
    p1, t1 = extract_prefix_period(im1)
    p2, t2 = extract_prefix_period(im2)
    
    # Period maps for known IMs
    period_map = {
        'Sa': sa_periods,
        'Saavg2': saavg2_periods,
        'Saavg3': saavg3_periods,
        'FIV3': fiv3_periods
    }
    
    if t1 is None and t2 is None:
        # Case: both IMs have no periods (e.g., PGA_PGV)
        folder_name = f"{p1}_{p2}"
        file_name = f"{p1}_{p2}.csv"   # e.g., PGA_PGV.csv
        file_path = my_path / "model_parameters" / database / folder_name / file_name
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"File not found for non-period IMs: {file_path}")

    elif t1 is None or t2 is None:
        # Case: one IM has a period, the other doesn’t
        folder_name = f"{p1}_{p2}"
        if t1 is None:
            file_name = f"{t2}_{p1}.csv"   # e.g. 0.2_PGA.csv
        else:
            file_name = f"{t1}_{p2}.csv"   # e.g. 0.2_PGV.csv

        file_path = my_path / "model_parameters" / database / folder_name / file_name
        if file_path.exists():
            return pd.read_csv(file_path)


    def get_surrounding_periods(p, t):
        if p not in period_map:
            return None, None, None
        valid = sorted(period_map[p])
        if t in valid:
            return t, t, True  # no interpolation needed
        if t < valid[0] or t > valid[-1]:
            raise ValueError(f"Period {t} for {p} is out of bounds.")
        below = max(pv for pv in valid if pv < t)
        above = min(pv for pv in valid if pv > t)
        return below, above, False
    
    def load_df(t1_val, t2_val, database):
        folder = f"{p1}_{p2}"

        if t1_val is None and t2_val is None:
            fname = f"{p1}_{p2}.csv"   # e.g. PGA_PGV.csv
        elif t1_val is None:
            fname = f"{t2_val}_{p1}.csv"  # e.g. 0.2_PGA.csv
        elif t2_val is None:
            fname = f"{t1_val}_{p2}.csv"  # e.g. 0.2_PGV.csv
        else:
            fname = f"{t1_val}_{t2_val}.csv"  # both have periods

        path = my_path / "model_parameters" / database / folder / fname
        return pd.read_csv(path)


    # Check if full file already exists
    folder_name = f"{p1}_{p2}"
    file_name = f"{t1}_{t2}.csv"
    file_path = my_path / f"model_parameters"/ database / folder_name / file_name
    if file_path.exists():
        return pd.read_csv(file_path)

    # Get surrounding periods
    t1_below, t1_above, t1_exact = get_surrounding_periods(p1, t1)
    t2_below, t2_above, t2_exact = get_surrounding_periods(p2, t2)

    if t1_exact and not t2_exact:
        # Interpolate in t2 only
        df_low = load_df(t1, t2_below, database)
        df_high = load_df(t1, t2_above, database)
        r = (t2 - t2_below) / (t2_above - t2_below)
        return interpolate_first_row_and_copy_second(df_low, df_high, r)


    elif t2_exact and not t1_exact:
        # Interpolate in t1 only
        df_low = load_df(t1_below, t2, database)
        df_high = load_df(t1_above, t2, database)
        r = (t1 - t1_below) / (t1_above - t1_below)
        return interpolate_first_row_and_copy_second(df_low, df_high, r)


    elif not t1_exact and not t2_exact:
        # Only 2 files needed — same-period lower and same-period upper
        df_low = load_df(t1_below, t2_below, database)
        df_high = load_df(t1_above, t2_above, database)
        r = (t1 - t1_below) / (t1_above - t1_below)
        return interpolate_first_row_and_copy_second(df_low, df_high, r)





# Function that interpolate the pc values in case an im is selected and is the period is not found
def interpolate_pc(im1, im2, value, database):

    # Assuming these are defined elsewhere or passed in scope
    # sa_periods, other_periods, my_path, extract_prefix_period

    # Extract prefix and period
    p1, t1 = extract_prefix_period(im1)
    p2, t2 = extract_prefix_period(im2)

    period_map = {
        'Sa': sa_periods,
        'Saavg2': saavg2_periods,
        'Saavg3': saavg3_periods,
        'FIV3': fiv3_periods
    }
        
    if t1 is None and t2 is None:
        # Case: both IMs have no periods (e.g., PGA_PGV)
        folder_name = f"{p1}_{p2}"
        file_name = f"{p1}_{p2}.csv"   # e.g., PGA_PGV.csv
        file_path = my_path / "PCA_coeff" / database / folder_name / file_name
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"File not found for non-period IMs: {file_path}")
        
    elif t1 is None or t2 is None:
        # Case: one IM has a period, the other doesn’t
        folder_name = f"{p1}_{p2}"
        if t1 is None:
            file_name = f"{t2}_{p1}.csv"   # e.g. 0.2_PGA.csv
        else:
            file_name = f"{t1}_{p2}.csv"   # e.g. 0.2_PGV.csv

        file_path = my_path / "PCA_coeff"/ database / folder_name / file_name
        if file_path.exists():
            return pd.read_csv(file_path)



    def get_surrounding_periods(p, t):
        if p not in period_map:
            return None, None, None
        valid = sorted(period_map[p])
        if t in valid:
            return t, t, True
        if t < valid[0] or t > valid[-1]:
            raise ValueError(f"Period {t} for {p} is out of bounds.")
        below = max(pv for pv in valid if pv < t)
        above = min(pv for pv in valid if pv > t)
        return below, above, False

    def load_df(t1_val, t2_val, database):
        folder = f"{p1}_{p2}"

        if t1_val is None and t2_val is None:
            fname = f"{p1}_{p2}.csv"   # e.g. PGA_PGV.csv
        elif t1_val is None:
            fname = f"{t2_val}_{p1}.csv"  # e.g. 0.2_PGA.csv
        elif t2_val is None:
            fname = f"{t1_val}_{p2}.csv"  # e.g. 0.2_PGV.csv
        else:
            fname = f"{t1_val}_{t2_val}.csv"  # both have periods

        path = my_path / "PCA_coeff" / database / folder / fname
        return pd.read_csv(path)

    folder = "PCA_coeff"
    folder_name = f"{p1}_{p2}"
    file_name = f"{t1}_{t2}.csv"
    full_path = my_path / folder / database /folder_name / file_name
    if full_path.exists():
        return pd.read_csv(full_path)

    t1_below, t1_above, t1_exact = get_surrounding_periods(p1, t1)
    t2_below, t2_above, t2_exact = get_surrounding_periods(p2, t2)

    # Helper for interpolation between two dfs or scalar baseline
    def interpolate_between_dfs(low_df, high_df, ratio):
        # Handle edge case: one df has 1x1 and is a repeated IM
        if low_df.shape == (1, 1) and high_df.shape != (1, 1):
            # Assume the value from the more complete df (high_df) is representative
            return high_df.to_numpy(dtype=float)
        elif high_df.shape == (1, 1) and low_df.shape != (1, 1):
            return low_df.to_numpy(dtype=float)
        elif low_df.shape == (1, 1) and high_df.shape == (1, 1):
            # Both are single values — simple scalar interpolation
            val = low_df.iloc[0, 0] + (high_df.iloc[0, 0] - low_df.iloc[0, 0]) * ratio
            return np.array([[val]], dtype=float)

        # Proceed with normal interpolation
        interp_df = low_df.copy()

        # Check if either low_df or high_df has only one column (single IM case)
        low_single_im = (low_df.shape[1] == 1)
        high_single_im = (high_df.shape[1] == 1)

        # Vector baseline to interpolate against if single IM
        baseline = np.array([1, 1], dtype=float)

        if low_single_im and not high_single_im:
            for col in [0, 1]:
                low_val = baseline[col]
                high_val = high_df.iloc[:, col].astype(float).values
                interp_vals = low_val + (high_val - low_val) * ratio
                interp_df.iloc[:, col] = interp_vals
        elif high_single_im and not low_single_im:
            for col in [0, 1]:
                low_val = low_df.iloc[:, col].astype(float).values
                high_val = baseline[col]
                interp_vals = low_val + (high_val - low_val) * ratio
                interp_df.iloc[:, col] = interp_vals
        else:
            cols_to_interp = [0, 1] if low_df.shape[1] > 1 else [0]
            for col in cols_to_interp:
                low_val = low_df.iloc[:, col].astype(float).values
                high_val = high_df.iloc[:, col].astype(float).values
                interp_vals = low_val + (high_val - low_val) * ratio
                interp_df.iloc[:, col] = interp_vals

        return interp_df.to_numpy(dtype=float)


    # CASE: Only t2 needs interpolation
    if t1_exact and not t2_exact:
        df_low = load_df(t1, t2_below, database)
        df_high = load_df(t1, t2_above, database)
        r = (t2 - t2_below) / (t2_above - t2_below)
        return interpolate_between_dfs(df_low, df_high, r)

    # CASE: Only t1 needs interpolation
    elif t2_exact and not t1_exact:
        df_low = load_df(t1_below, t2, database)
        df_high = load_df(t1_above, t2, database)
        r = (t1 - t1_below) / (t1_above - t1_below)
        return interpolate_between_dfs(df_low, df_high, r)

    elif not t1_exact and not t2_exact:
        df_low = load_df(t1_below, t2_below, database)
        df_high = load_df(t1_above, t2_above, database)

        x = t1
        x1 = t1_below
        x2 = t1_above

        interp_df = df_low.copy()

        # Detect if single IM (same IM on both axes)
        if (t1_below == t2_below) and (t1_above == t2_above):
            # Single IM: interpolate all relevant numeric columns directly
            for col in df_low.select_dtypes(include=[np.number]).columns:
                f_low = df_low[col]
                f_high = df_high[col]
                interp_df[col] = f_low + (f_high - f_low) * ((x - x1) / (x2 - x1))
        else:
            # Different IMs: maybe only interpolate a subset of columns (1 to 5 here)
            for col in df_low.columns[1:6]:
                f_low = df_low[col]
                f_high = df_high[col]
                interp_df[col] = f_low + (f_high - f_low) * ((x - x1) / (x2 - x1))

        return interp_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    # If both exact (shouldn't reach here)
    raise RuntimeError("Interpolation not needed; file should already exist.")




# Function to organise the order of how the ims should be called im1-im2 or im2-im1
def sort_im_pair(im1, im2):
    p1, t1 = extract_prefix_period(im1)
    p2, t2 = extract_prefix_period(im2)

    i1 = ims_priority.index(p1) if p1 in ims_priority else len(ims_priority)
    i2 = ims_priority.index(p2) if p2 in ims_priority else len(ims_priority)

    if i1 > i2 or (i1 == i2 and t1 is not None and t2 is not None and t1 > t2):
        return im2, im1
    return im1, im2


def read_params_file(im1, im2, database):
    p1, t1 = extract_prefix_period(im1)
    p2, t2 = extract_prefix_period(im2)

    def format_im_name(prefix, period):
        if period is None:
            return prefix
        else:
            return f"{prefix}({period})"

    im1_new = format_im_name(p1, t1)
    im2_new = format_im_name(p2, t2)

    folder_name = f"{p1}_{p2}"
    filename = f"{t1}_{t2}.csv"
    file_path = my_path / "model_parameters" / database / folder_name / filename
    
    if file_path.exists():
        df = pd.read_csv(file_path)
        df = df.select_dtypes(include=[np.number])
        return df.to_numpy(dtype=float)
    return None



def read_pc_file(im1, im2, database):
    p1, t1 = extract_prefix_period(im1)
    p2, t2 = extract_prefix_period(im2)
    folder_name = f"{p1}_{p2}"
    filename = f"{t1}_{t2}.csv"
    path = my_path / f" PCA_coeff" / database / folder_name / filename
    if path.exists():
        return pd.read_csv(path).to_numpy(dtype=float)  # <-- Convert to NumPy here
    return None


def get_pc(im1, im2, value, database):
    im1, im2 = sort_im_pair(im1, im2)
    data = read_pc_file(im1, im2, database)
    if data is not None:
        return data  # already a NumPy array
    interpolated = interpolate_pc(im1, im2, value, database)
    return interpolated.to_numpy(dtype=float) if isinstance(interpolated, pd.DataFrame) else interpolated



def get_nested_parameters(im1, im2, database):
    im1, im2 = sort_im_pair(im1, im2)
    data = read_params_file(im1, im2, database)
    if data is not None:
        return data
    return interpolate_parameters(im1, im2, database)


def MAO2025(IM1, IM2, h, cluster=0, vs30=None):
    """
    MAO2025 function
    
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
    """
    
    if cluster == 0:
        database = "non_cluster"
    
    elif cluster == 1:
        if vs30 == 1:
            database = "cluster_vs30_low"
        elif vs30 == 2:
            database = "cluster_vs30_high"
        else:
            raise ValueError("When cluster=1, vs30 must be 1 (low) or 2 (high).")
    
    else:
        raise ValueError("cluster must be 0 (non_cluster) or 1 (cluster).")
        
        
        
    IM1, IM2 = sort_im_pair(IM1, IM2)

    
    nested_para = np.asarray(get_nested_parameters(IM1, IM2, database))
    
    pc_all = get_pc(IM1, IM2, None, database)  # Get full array
    pc1 = pc_all[:, 0]
    if IM1 != IM2 and pc_all.shape[1] > 1:
        pc2 = pc_all[:, 1]
    else:
        pc2 = pc1
        
    
    Inugget = 1 if h == 0 else 0
    sill = C_h = Cii_0 = Cjj_0 =0.0
    for i in range(len(nested_para)):
        _, c0i, c1i, a1i, c2i, a2i = nested_para[i]
        Cij_h = (c0i * Inugget + c1i * np.exp(-3 * h / a1i) +
                 c2i * np.exp(-3 * h / a2i))
        # semiv_ij_h = (c0i * (1-Inugget) + c1i * (1-np.exp(-3 * h / a1i)) +
        #          c2i *(1-np.exp(-3 * h / a2i)))
        silli = c0i + c1i + c2i

        sill += pc1[i] * pc2[i] * silli
        C_h += pc1[i] * pc2[i] * Cij_h
        
        
        
    # For the IM1 individually
    nested_para1 = np.asarray(get_nested_parameters(IM1, IM1, database))
    pc_all1 = get_pc(IM1, IM1, None, database)  # Get full array
    pc1_1 = pc_all1[:, 0]  # First and only principal component =1
    
    C_zero1 = 0.0
    for i in range(len(nested_para1)):
            _, c0i, c1i, a1i, c2i, a2i = nested_para1[i]
            Ci1_h_1 = (c0i * Inugget + c1i * np.exp(-3 * h / a1i) +
                    c2i * np.exp(-3 * h / a2i))
            silli_1 = c0i + c1i + c2i
            C_zero1 += pc1_1 ** 2 * silli_1

            
    # For the IM2 individually
    nested_para2 = np.asarray(get_nested_parameters(IM2, IM2, database))
    pc_all2 = get_pc(IM2, IM2, None, database)  # Get full array
    pc1_2 = pc_all2[:, 0]  # First and only principal component =1

    C_zero2 = 0.0
    for i in range(len(nested_para2)):
            _, c0i, c1i, a1i, c2i, a2i = nested_para2[i]
            Ci1_h_2 = (c0i * Inugget + c1i * np.exp(-3 * h / a1i) +
                    c2i * np.exp(-3 * h / a2i))
            silli_2 = c0i + c1i + c2i
            C_zero2 += pc1_2 ** 2 * silli_2


    corr = C_h / np.sqrt(C_zero1 * C_zero2)
    
    return float(np.asarray(corr).item())

