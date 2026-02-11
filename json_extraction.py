"""
Complete IGBT/MOSFET Parameter Extraction Suite
================================================

Extracts all available data from transistor JSON files:
1. I-V Characteristics (temperature dependent)
2. Capacitance-Voltage (C-V) characteristics
3. Derived device parameters (V_th, μ, λ, etc.)

Saves organized CSV files with diagnostic warnings.




Claude AI developed the majority of this code, soem extractions or derivations may be incorrect. EDA in analyze.ipynb will avoid using derived/computed values 
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


def json_to_iv_dataframe(json_data: Dict, component_type: str = 'switch') -> pd.DataFrame:
    """
    Extract I-V characteristics from JSON.
    
    Returns DataFrame with: temperature_C, vgs_V, vds_V, id_A, rds_on_Ohm, power_W
    """
    rows = []
    
    component = json_data.get(component_type, {})
    channel_data = component.get('channel', [])
    
    if not channel_data:
        print(f"⚠️  WARNING: No channel data found for {component_type}")
        return pd.DataFrame()
    
    for temp_curve in channel_data:
        temp_c = temp_curve.get('t_j')
        vgs_v = temp_curve.get('v_g')
        graph_v_i = temp_curve.get('graph_v_i', [])
        
        if len(graph_v_i) >= 2:
            voltage_list = graph_v_i[0]
            current_list = graph_v_i[1]
            
            for v, i in zip(voltage_list, current_list):
                row = {
                    "temperature_C": temp_c,
                    "vgs_V": vgs_v,
                    "vds_V": v,
                    "id_A": i
                }
                
                if i > 0.1:
                    row["rds_on_Ohm"] = v / i
                else:
                    row["rds_on_Ohm"] = None
                
                row["power_W"] = v * i
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        sort_cols = ['temperature_C']
        if df['vgs_V'].notna().any():
            sort_cols.append('vgs_V')
        sort_cols.append('vds_V')
        df = df.sort_values(sort_cols).reset_index(drop=True)
    
    return df


def json_to_capacitance_dataframe(json_data: Dict) -> pd.DataFrame:
    """
    Extract C_iss, C_oss, C_rss capacitance-voltage curves from JSON.
    
    Returns DataFrame with: temperature_C, voltage_V, c_iss_F, c_oss_F, c_rss_F
    """
    rows = []
    
    # Track which capacitances are available and at what temperatures
    cap_types = ['c_iss', 'c_oss', 'c_rss']
    available_caps = []
    temp_ranges = {}
    
    for cap_type in cap_types:
        cap_data = json_data.get(cap_type, [])
        
        if cap_data:
            available_caps.append(cap_type)
            temps = [entry.get('t_j') for entry in cap_data]
            temp_ranges[cap_type] = temps
    
    if not available_caps:
        print("⚠️  WARNING: No capacitance data (c_iss, c_oss, c_rss) found in JSON")
        return pd.DataFrame()
    
    # Print diagnostic info
    print(f"✓ Found capacitance data for: {', '.join(available_caps)}")
    for cap_type in available_caps:
        temps = temp_ranges[cap_type]
        if len(temps) == 1:
            print(f"⚠️  WARNING: {cap_type.upper()} only available at ONE temperature: {temps[0]}°C")
            print(f"   → Cannot analyze {cap_type.upper()} vs temperature!")
        else:
            print(f"✓ {cap_type.upper()} available at {len(temps)} temperatures: {temps}")
    
    # Create unified voltage grid by collecting all unique voltages
    all_voltages = set()
    
    for cap_type in cap_types:
        cap_data = json_data.get(cap_type, [])
        for entry in cap_data:
            graph_v_c = entry.get('graph_v_c', [])
            if len(graph_v_c) >= 2:
                voltages = graph_v_c[0]
                all_voltages.update(voltages)
    
    # Convert to sorted list
    voltage_grid = sorted(list(all_voltages))
    
    # Now extract data for each temperature
    temperatures = set()
    for cap_type in cap_types:
        cap_data = json_data.get(cap_type, [])
        for entry in cap_data:
            temperatures.add(entry.get('t_j'))
    
    for temp in sorted(temperatures):
        # Get data for each capacitance type at this temperature
        cap_values = {cap_type: {} for cap_type in cap_types}
        
        for cap_type in cap_types:
            cap_data = json_data.get(cap_type, [])
            for entry in cap_data:
                if entry.get('t_j') == temp:
                    graph_v_c = entry.get('graph_v_c', [])
                    if len(graph_v_c) >= 2:
                        voltages = graph_v_c[0]
                        capacitances = graph_v_c[1]
                        # Create voltage -> capacitance mapping
                        for v, c in zip(voltages, capacitances):
                            cap_values[cap_type][v] = c
        
        # Create rows for each voltage point
        for voltage in voltage_grid:
            row = {
                "temperature_C": temp,
                "voltage_V": voltage,
            }
            
            for cap_type in cap_types:
                col_name = f"{cap_type}_F"
                row[col_name] = cap_values[cap_type].get(voltage, None)
            
            # Only add row if at least one capacitance value exists
            if any(row[f"{cap_type}_F"] is not None for cap_type in cap_types):
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df = df.sort_values(['temperature_C', 'voltage_V']).reset_index(drop=True)
    
    return df


def extract_device_parameters(df_iv: pd.DataFrame) -> pd.DataFrame:
    """
    Extract derived device parameters from I-V curves.
    
    Extracts:
    - V_th: Threshold voltage (approximated from linear region)
    - λ: Channel length modulation parameter (from saturation region)
    - g_ds: Output conductance (∂I_D/∂V_DS in saturation)
    - g_m: Transconductance (approximation from available data)
    
    Returns DataFrame with: temperature_C, vgs_V, vth_estimate_V, lambda_per_V, 
                           gds_S, gm_estimate_S, id_sat_A
    """
    if df_iv.empty:
        print("⚠️  WARNING: Cannot extract device parameters - I-V DataFrame is empty")
        return pd.DataFrame()
    
    # Check if we have multiple VGS values
    vgs_values = df_iv['vgs_V'].dropna().unique()
    if len(vgs_values) <= 1:
        print("⚠️  WARNING: Only ONE V_GS value available - cannot extract full transfer characteristics")
        print(f"   → V_GS = {vgs_values[0] if len(vgs_values) > 0 else 'N/A'} V")
        print("   → V_th extraction will be APPROXIMATE (from output curves only)")
        print("   → g_m (transconductance) cannot be accurately computed")
    
    results = []
    
    # Group by temperature and VGS
    for (temp, vgs), group in df_iv.groupby(['temperature_C', 'vgs_V']):
        if group.empty or len(group) < 5:
            continue
        
        # Sort by VDS
        group = group.sort_values('vds_V')
        vds = group['vds_V'].values
        id_vals = group['id_A'].values
        
        # 1. Estimate threshold voltage from linear region
        # In linear region: I_D ≈ K * (V_GS - V_th) * V_DS
        # Find where I_D becomes significant (> 1A)
        linear_mask = (vds < 0.5) & (id_vals > 1.0)
        
        if linear_mask.sum() >= 2:
            vds_lin = vds[linear_mask]
            id_lin = id_vals[linear_mask]
            
            # Linear fit: I_D = slope * V_DS + intercept
            # slope = K * (V_GS - V_th)
            coeffs = np.polyfit(vds_lin, id_lin, 1)
            slope = coeffs[0]  # dI_D/dV_DS in linear region
            
            # Rough V_th estimate (assumes standard parameters)
            # This is very approximate without knowing K = μC_ox(W/L)
            vth_estimate = None  # Cannot accurately determine without transfer curve
        else:
            vth_estimate = None
        
        # 2. Channel length modulation (λ) from saturation region
        # In saturation: I_D = I_D0 * (1 + λ*V_DS)
        # Look at high VDS region where curve should be nearly flat
        sat_mask = (vds > 1.5) & (id_vals > 50)
        
        lambda_param = None
        gds = None
        
        if sat_mask.sum() >= 3:
            vds_sat = vds[sat_mask]
            id_sat = id_vals[sat_mask]
            
            # Fit linear trend: I_D = a + b*V_DS
            # Then λ ≈ b / I_D0
            coeffs_sat = np.polyfit(vds_sat, id_sat, 1)
            slope_sat = coeffs_sat[1]  # dI_D/dV_DS
            intercept_sat = coeffs_sat[0]
            
            gds = slope_sat  # Output conductance [S]
            
            if intercept_sat != 0:
                lambda_param = slope_sat / intercept_sat  # [1/V]
            
            # Saturation current (extrapolated to V_DS = 0)
            id_sat_extrap = intercept_sat
        else:
            id_sat_extrap = id_vals[-1] if len(id_vals) > 0 else None
        
        # 3. Transconductance (g_m) - needs dI_D/dV_GS, which we don't have
        # Can only approximate if we had multiple VGS curves
        gm_estimate = None
        
        results.append({
            'temperature_C': temp,
            'vgs_V': vgs,
            'vth_estimate_V': vth_estimate,
            'lambda_per_V': lambda_param,
            'gds_S': gds,
            'gm_estimate_S': gm_estimate,
            'id_sat_A': id_sat_extrap
        })
    
    df_params = pd.DataFrame(results)
    
    if not df_params.empty:
        df_params = df_params.sort_values('temperature_C').reset_index(drop=True)
    
    # Print diagnostic summary
    print("\n" + "="*80)
    print("Device Parameter Extraction Summary:")
    print("="*80)
    if not df_params.empty:
        print(f"✓ Extracted parameters for {len(df_params)} temperature points")
        print(f"✓ λ (channel length modulation): Available for {df_params['lambda_per_V'].notna().sum()} points")
        print(f"✓ g_ds (output conductance): Available for {df_params['gds_S'].notna().sum()} points")
        print(f"⚠️  V_th (threshold voltage): NOT accurately extracted (needs transfer curves)")
        print(f"⚠️  g_m (transconductance): NOT available (needs multiple V_GS)")
        print(f"⚠️  μ (mobility): NOT extracted (needs device geometry W, L)")
    
    return df_params


def compute_thermal_voltage(df_iv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute thermal voltage V_T = kT/q for each temperature.
    
    Returns DataFrame with: temperature_C, temperature_K, thermal_voltage_V
    """
    if df_iv.empty:
        return pd.DataFrame()
    
    # Boltzmann constant and elementary charge
    k = 1.380649e-23  # J/K
    q = 1.602176634e-19  # C
    
    temps_c = df_iv['temperature_C'].unique()
    
    results = []
    for temp_c in temps_c:
        temp_k = temp_c + 273.15
        v_t = (k * temp_k) / q
        
        results.append({
            'temperature_C': temp_c,
            'temperature_K': temp_k,
            'thermal_voltage_V': v_t
        })
    
    df_vt = pd.DataFrame(results).sort_values('temperature_C').reset_index(drop=True)
    
    return df_vt


def save_device_data(device_name: str, 
                     df_iv_switch: pd.DataFrame,
                     df_iv_diode: pd.DataFrame,
                     df_capacitance: pd.DataFrame,
                     df_parameters: pd.DataFrame,
                     df_thermal_voltage: pd.DataFrame,
                     base_path: str = '/mnt/user-data/outputs') -> None:
    """
    Save all dataframes to organized folder structure.
    
    Creates: base_path/device_name/*.csv
    """
    # Create device folder
    device_folder = Path(base_path) / device_name
    device_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Saving data to: {device_folder}")
    print(f"{'='*80}")
    
    saved_files = []
    
    # Save each dataframe if not empty
    if not df_iv_switch.empty:
        filepath = device_folder / "1_iv_characteristics_switch.csv"
        df_iv_switch.to_csv(filepath, index=False)
        saved_files.append(f"✓ {filepath.name} ({len(df_iv_switch)} rows)")
    else:
        print("⚠️  Skipped: Switch I-V data (empty)")
    
    if not df_iv_diode.empty:
        filepath = device_folder / "2_iv_characteristics_diode.csv"
        df_iv_diode.to_csv(filepath, index=False)
        saved_files.append(f"✓ {filepath.name} ({len(df_iv_diode)} rows)")
    else:
        print("⚠️  Skipped: Diode I-V data (empty)")
    
    if not df_capacitance.empty:
        filepath = device_folder / "3_capacitance_voltage.csv"
        df_capacitance.to_csv(filepath, index=False)
        saved_files.append(f"✓ {filepath.name} ({len(df_capacitance)} rows)")
    else:
        print("⚠️  Skipped: Capacitance data (empty)")
    
    if not df_parameters.empty:
        filepath = device_folder / "4_derived_parameters.csv"
        df_parameters.to_csv(filepath, index=False)
        saved_files.append(f"✓ {filepath.name} ({len(df_parameters)} rows)")
    else:
        print("⚠️  Skipped: Derived parameters (empty)")
    
    if not df_thermal_voltage.empty:
        filepath = device_folder / "5_thermal_voltage.csv"
        df_thermal_voltage.to_csv(filepath, index=False)
        saved_files.append(f"✓ {filepath.name} ({len(df_thermal_voltage)} rows)")
    else:
        print("⚠️  Skipped: Thermal voltage (empty)")
    
    # Print summary
    print("\nSaved files:")
    for file_info in saved_files:
        print(f"  {file_info}")
    
    print(f"\nTotal files saved: {len(saved_files)}")
    print(f"{'='*80}\n")


def analyze_data_completeness(json_data: Dict) -> Dict:
    """
    Analyze what data is available and what's missing.
    Returns diagnostic report.
    """
    report = {
        'device_name': json_data.get('name', 'Unknown'),
        'device_type': json_data.get('type', 'Unknown'),
        'manufacturer': json_data.get('manufacturer', 'Unknown'),
        'issues': [],
        'available': []
    }
    
    # Check switch data
    switch_data = json_data.get('switch', {})
    switch_channels = switch_data.get('channel', [])
    
    if switch_channels:
        temps = [ch.get('t_j') for ch in switch_channels]
        vgs_vals = [ch.get('v_g') for ch in switch_channels]
        report['available'].append(f"Switch I-V: {len(switch_channels)} temperature points ({temps})")
        
        # Check VGS diversity
        unique_vgs = set([v for v in vgs_vals if v is not None])
        if len(unique_vgs) <= 1:
            report['issues'].append(
                f"Only ONE V_GS value ({list(unique_vgs)[0] if unique_vgs else 'N/A'}V) - "
                "cannot extract transfer characteristics or V_th accurately"
            )
        
        # Check if we have both linear and saturation regions
        for ch in switch_channels:
            graph_v_i = ch.get('graph_v_i', [])
            if len(graph_v_i) >= 2:
                max_vds = max(graph_v_i[0]) if graph_v_i[0] else 0
                if max_vds < 2.0:
                    report['issues'].append(
                        f"V_DS range may be insufficient (max {max_vds:.2f}V) - "
                        "need higher voltages for saturation region analysis"
                    )
                    break
    else:
        report['issues'].append("No switch I-V data available")
    
    # Check diode data
    diode_data = json_data.get('diode', {})
    diode_channels = diode_data.get('channel', [])
    if diode_channels:
        temps = [ch.get('t_j') for ch in diode_channels]
        report['available'].append(f"Diode I-V: {len(diode_channels)} temperature points ({temps})")
    else:
        report['issues'].append("No diode I-V data available")
    
    # Check capacitance data
    cap_temps = {}
    for cap_type in ['c_iss', 'c_oss', 'c_rss']:
        cap_data = json_data.get(cap_type, [])
        if cap_data:
            temps = [entry.get('t_j') for entry in cap_data]
            cap_temps[cap_type] = temps
            report['available'].append(f"{cap_type.upper()}: {len(temps)} temperature point(s) ({temps})")
            
            if len(temps) == 1:
                report['issues'].append(
                    f"{cap_type.upper()} only at {temps[0]}°C - "
                    "cannot analyze temperature dependence"
                )
    
    if not cap_temps:
        report['issues'].append("No capacitance (C_iss, C_oss, C_rss) data available")
    
    # Check for switching energy data (useful but not critical)
    switch_e_on = switch_data.get('e_on', [])
    switch_e_off = switch_data.get('e_off', [])
    
    if switch_e_on or switch_e_off:
        report['available'].append("Switching energy data (E_on/E_off) available")
    
    return report


def print_diagnostic_report(report: Dict) -> None:
    """
    Print formatted diagnostic report.
    """
    print("\n" + "="*80)
    print("DATA COMPLETENESS DIAGNOSTIC REPORT")
    print("="*80)
    print(f"Device: {report['device_name']}")
    print(f"Type: {report['device_type']}")
    print(f"Manufacturer: {report['manufacturer']}")
    print("="*80)
    
    print("\n✓ AVAILABLE DATA:")
    if report['available']:
        for item in report['available']:
            print(f"  • {item}")
    else:
        print("  (none)")
    
    print("\n⚠️  LIMITATIONS & MISSING DATA:")
    if report['issues']:
        for i, issue in enumerate(report['issues'], 1):
            print(f"  {i}. {issue}")
    else:
        print("  (none - complete dataset)")
    
    print("\n" + "="*80)
    print("PARAMETER EXTRACTION CAPABILITIES:")
    print("="*80)
    
    # Determine what can be extracted
    has_multi_temp_iv = any("temperature points" in item and 
                            int(item.split()[2]) > 1 
                            for item in report['available'] 
                            if "I-V" in item)
    
    has_multi_vgs = not any("ONE V_GS" in issue for issue in report['issues'])
    
    has_multi_temp_cap = any("C_" in item and 
                             int(item.split()[1]) > 1 
                             for item in report['available'] 
                             if "C_" in item)
    
    print(f"{'Parameter':<25} {'Status':<15} {'Notes'}")
    print("-" * 80)
    print(f"{'V_GS':<25} {'✓ Direct':<15} From I-V curves")
    print(f"{'V_DS':<25} {'✓ Direct':<15} From I-V curves")
    print(f"{'I_D':<25} {'✓ Direct':<15} From I-V curves")
    print(f"{'T_j':<25} {'✓ Direct':<15} From I-V curves")
    print(f"{'C_iss, C_oss, C_rss':<25} {'✓ Direct':<15} From capacitance data")
    
    print(f"{'V_th':<25} {'⚠️  Approximate':<15} {'Need transfer curves' if not has_multi_vgs else 'Can estimate'}")
    print(f"{'λ (lambda)':<25} {'✓ Extractable':<15} From saturation region")
    print(f"{'g_ds':<25} {'✓ Extractable':<15} From output curves")
    print(f"{'g_m':<25} {'❌ Cannot extract':<15} Need multiple V_GS")
    print(f"{'μ (mobility)':<25} {'❌ Cannot extract':<15} Need W, L geometry")
    print(f"{'V_T (thermal)':<25} {'✓ Computable':<15} V_T = kT/q")
    print(f"{'V_SB':<25} {'N/A':<15} IGBTs don't have body effect")
    print(f"{'I_G':<25} {'Not measured':<15} Assumed negligible")
    
    print("\n" + "="*80 + "\n")


# Main execution
if __name__ == "__main__":
    
    # Load JSON data
    json_file_path = '/mnt/user-data/uploads/transistor_data.json'  # Adjust path as needed
    
    # For demo, use embedded sample
    print("Loading transistor data...")
    
    # You would normally load from file:
    # with open(json_file_path, 'r') as f:
    #     transistor_data = json.load(f)
    
    # For this demo, we'll create sample data
    print("⚠️  Note: Using sample data for demonstration")
    print("   Replace with: json.load(open('your_file.json'))\n")
    
    # This would be your actual JSON data
    # For now, we'll just show the structure
    
    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    print("""
To use this script with your JSON file:

1. Load your JSON:
   with open('Fuji_2MBI300XBE120-50.json', 'r') as f:
       transistor_data = json.load(f)

2. Run diagnostic report:
   report = analyze_data_completeness(transistor_data)
   print_diagnostic_report(report)

3. Extract all dataframes:
   df_iv_switch = json_to_iv_dataframe(transistor_data, 'switch')
   df_iv_diode = json_to_iv_dataframe(transistor_data, 'diode')
   df_capacitance = json_to_capacitance_dataframe(transistor_data)
   df_parameters = extract_device_parameters(df_iv_switch)
   df_thermal_voltage = compute_thermal_voltage(df_iv_switch)

4. Save organized files:
   device_name = transistor_data.get('name', 'device')
   save_device_data(
       device_name,
       df_iv_switch,
       df_iv_diode,
       df_capacitance,
       df_parameters,
       df_thermal_voltage
   )

This will create a folder structure:
   outputs/
   └── Fuji_2MBI300XBE120-50/
       ├── 1_iv_characteristics_switch.csv
       ├── 2_iv_characteristics_diode.csv
       ├── 3_capacitance_voltage.csv
       ├── 4_derived_parameters.csv
       └── 5_thermal_voltage.csv
""")
    print("="*80)