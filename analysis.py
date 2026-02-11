import pandas as pd
import numpy as np
import os

def compute_log_linear_tempco(df, col, group_col=None): # for exponential
    """
    Computes log-linear temperature coefficient.
    
    ln(Y) = slope * T + intercept
    returns fractional change per degree C (0.005 = 0.5% / degree C)
    """
    if col not in df.columns or 'temperature_C' not in df.columns:
        return np.nan, np.nan
    
    df_clean = df[['temperature_C', col] + 
                  ([group_col] if group_col and group_col in df.columns else [])].dropna()
    
    # Remove non-positive values (for log calc)
    df_clean = df_clean[df_clean[col] > 0]
    
    if len(df_clean) < 3: #should have more than 2 points, only 2 is useless, increasing will be safer/explain more, but will lose a lot of data in not enough measurements across temperature range
        return np.nan, np.nan # return 2 to preserve output tuple dim
    
    slopes = []
    r_squared_values = []
    
    if group_col and group_col in df_clean.columns:
        grouped = df_clean.groupby(group_col)
    else:
        grouped = [(None, df_clean)]
    
    for _, g in grouped: # group_name, group_df
        if len(g) < 3:
            continue
        
        Temp = g['temperature_C'].values
        Y = g[col].values


        
        try:
            # Fit ln(Y) = slope * Temp + intercept

            # making more robust by centering:
            Temp_centered = Temp - np.mean(Temp)
            slope, intercept = np.polyfit(Temp_centered, np.log(Y), 1) # polynomial ls regression
            
            # Simple r^2 for fit quality
            y_pred = slope * Temp_centered + intercept
            y_actual = np.log(Y)
            r_squared = 1 - (np.sum((y_actual - y_pred)**2) / 
                           np.sum((y_actual - np.mean(y_actual))**2))
            
            
            slopes.append(slope)
            r_squared_values.append(r_squared)
        except:
            continue
    
    if len(slopes) == 0:
        return np.nan, np.nan
    
    # Return average slope and average fit quality
    return np.median(slopes), np.median(r_squared_values) # median probably more robust for curve idk 


def compute_linear_tempco(df, col, group_col=None): # for (exactly) linear (V_th??)
    """
    Compute linear temperature coefficient.
    
    Y = slope * Temp + intercept
    returns absolute change per degree C
    """

    # beginning is largely the same
    if col not in df.columns or 'temperature_C' not in df.columns:
        return np.nan, np.nan
    
    df_clean = df[['temperature_C', col] + 
                  ([group_col] if group_col and group_col in df.columns else [])].dropna()
    
    if len(df_clean) < 3:
        return np.nan, np.nan
    
    slopes = []
    r_squared_values = []
    
    if group_col and group_col in df_clean.columns:
        grouped = df_clean.groupby(group_col)
    else:
        grouped = [(None, df_clean)]
    
    for _, g in grouped:
        if len(g) < 3:
            continue
        
        Temp = g['temperature_C'].values
        Y = g[col].values
        
        try:
            slope, intercept = np.polyfit(Temp, Y, 1) # not log(Y)
            
            # r^2
            y_pred = slope * Temp + intercept
            r_squared = 1 - (np.sum((Y - y_pred)**2) / 
                           np.sum((Y - np.mean(Y))**2))
            
            slopes.append(slope)
            r_squared_values.append(r_squared)
        except:
            continue
    
    if len(slopes) == 0:
        return np.nan, np.nan
    
    return np.mean(slopes), np.mean(r_squared_values)


def compute_percent_change(df, col):
    """
    % change for min to max temps, sanity check. If big difference somethings wrong
    """
    if col not in df.columns or 'temperature_C' not in df.columns:
        return np.nan
    
    df_clean = df[['temperature_C', col]].dropna()
    if len(df_clean) < 2:
        return np.nan
    
    # Group by temperature and take mean
    temp_means = df_clean.groupby('temperature_C')[col].mean()
    
    if len(temp_means) < 2:
        return np.nan
    
    min_temp = temp_means.index.min()
    max_temp = temp_means.index.max()
    
    val_at_min = temp_means[min_temp]
    val_at_max = temp_means[max_temp]
    
    if abs(val_at_min) < 1e-12: # super small
        return np.nan
    
    # Total percent change over temperature range
    return ((val_at_max - val_at_min) / val_at_min) * 100


# ============================================================================
# Analysis loop for each transistor csv  folder in /outputs


outputs_dir = './outputs'

transistor_folders = [f for f in os.listdir(outputs_dir) 
                     if os.path.isdir(os.path.join(outputs_dir, f))]

master_results = []

for folder in transistor_folders:
    folder_path = os.path.join(outputs_dir, folder)
    print(f"\nProcessing {folder}...")

    # Load CSV files safely
    def load_csv_safe(filename):
        path = os.path.join(folder_path, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        else:
            print(f"error: {filename} missing")
            return pd.DataFrame()

    iv_switch = load_csv_safe('1_iv_characteristics_switch.csv')
    iv_diode = load_csv_safe('2_iv_characteristics_diode.csv')
    cap_volt = load_csv_safe('3_capacitance_voltage.csv')
    derived = load_csv_safe('4_derived_parameters.csv')
    thermal_v = load_csv_safe('5_thermal_voltage.csv')

    # print(iv_switch.columns)
    # print(iv_diode.columns)
    # print(cap_volt.columns)
    # print(derived.columns)
    # print(thermal_v.columns)

    print("iv_switch head:")
    print(iv_switch.head())
    print("Columns:", iv_switch.columns.tolist())
    

    # parameters map to make function calling easier: (dataframe, column, group_column, method)
    # method: 'log' = exponential, 'linear' = linear


    # below is from claude - double check - DONE
    
    parameters = {
        'Id_switch': (iv_switch, 'id_A', 'vgs_V', 'log'),
        'Rds_on_switch': (iv_switch, 'rds_on_Ohm', 'vgs_V', 'log'),
        'Power_switch': (iv_switch, 'power_W', 'vgs_V', 'log'),
        
        'Id_diode': (iv_diode, 'id_A', None, 'log'),
        'Rds_on_diode': (iv_diode, 'rds_on_Ohm', None, 'log'),
        'Power_diode': (iv_diode, 'power_W', None, 'log'),
        
        'Ciss': (cap_volt, 'c_iss_F', 'voltage_V', 'linear'),
        'Coss': (cap_volt, 'c_oss_F', 'voltage_V', 'linear'),
        'Crss': (cap_volt, 'c_rss_F', 'voltage_V', 'linear'),
        
        # Derived (take with a grain of salt in analysis)
        'Vth': (derived, 'vth_estimate_V', None, 'linear'),  # Linear
        'gm': (derived, 'gm_estimate_S', None, 'log'),       # Exponential
        'gds': (derived, 'gds_S', None, 'log'),              # Exponential
        'Id_sat': (derived, 'id_sat_A', None, 'log'),        # Exp
        'lambda': (derived, 'lambda_per_V', None, 'linear'), # Linear
        
        # Thermal voltage (V_T = kT/q) (Should be exactly linear)
        'Vt_thermal': (thermal_v, 'thermal_voltage_V', None, 'linear')
    }

    # Compute
    result = {'transistor_name': folder}
    
    for param, (df, col, group, method) in parameters.items():
        if df.empty or col not in df.columns:
            result[f'{param}_tempco'] = np.nan
            result[f'{param}_r2'] = np.nan
            result[f'{param}_pct_change'] = np.nan
            continue
        
        # Compute temperature coefficient
        if method == 'log':
            tempco, r2 = compute_log_linear_tempco(df, col, group)
            result[f'{param}_tempco'] = tempco  # fractional/C
            result[f'{param}_r2'] = r2
        else:  # linear
            tempco, r2 = compute_linear_tempco(df, col, group)
            result[f'{param}_tempco'] = tempco  # absolute/C
            result[f'{param}_r2'] = r2
        
        # Also compute simple percent change for reference
        pct = compute_percent_change(df, col)
        result[f'{param}_pct_change'] = pct
    
    master_results.append(result)
    
    # Print summary for this device
    print(f"Analyzed {len([k for k in result.keys() if 'tempco' in k])} parameters")

# Convert to df
df_sensitivity = pd.DataFrame(master_results)
df_sensitivity = df_sensitivity.sort_values('transistor_name').reset_index(drop=True)

# Reorder columns for clarity
first_cols = ['transistor_name']
other_cols = [c for c in df_sensitivity.columns if c != 'transistor_name']
df_sensitivity = df_sensitivity[first_cols + sorted(other_cols)]

# Save results
output_file = 'all_transistors_sensitivity.csv'
df_sensitivity.to_csv(output_file, index=False)
