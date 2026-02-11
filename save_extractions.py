from json_extraction import *
import json
import sys
import glob


data_folder = './data'

files = glob.glob(os.path.join(data_folder,'*.json'))

if not files:
    print('Couldnt find any files')
    sys.exit(1)


for file in files:
    print(f"Working on {file}")
    

    # Load your JSON
    with open(file) as f:
        data = json.load(f)

    # Get diagnostic report first
    report = analyze_data_completeness(data)
    print_diagnostic_report(report)

    # Extract everything
    df1 = json_to_iv_dataframe(data, 'switch')
    df2 = json_to_iv_dataframe(data, 'diode')
    df3 = json_to_capacitance_dataframe(data)
    df4 = extract_device_parameters(df1)
    df5 = compute_thermal_voltage(df1)

    # Save organized
    save_device_data(data['name'], df1, df2, df3, df4, df5, base_path='./outputs')