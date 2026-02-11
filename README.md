# temp-sensitivity-analysis



Analyze effect of temperature on transistor parameters for future modeling

Just a small exploratory analysis. Looked at transistors in the transistor database

Goal was to identify potentially important variable impacted by temperature. Insight for later analytical modeling.


Limitations:



To RUN:
1. download .json data files from https://github.com/upb-lea/transistordatabase/tree/main/transistordatabase/examples/tdb_example  
2. place these files in /data
3. Create /outputs folder
4. run analysis.py and final_analysis.ipynb

Here are some calculated sensitivity values not included in the full analysis:    


Id_switch        0.001325  
Rds_on_switch    0.004724  
Id_diode         0.000898  
Rds_on_diode     0.003945  
Ciss                  NaN  
Coss                  NaN  
Crss                  NaN  
Vth                   NaN  
gm                    NaN  
gds             -0.002035  
Id_sat           0.003557  
Vt_thermal       0.002816  
dtype: float64

These values are extremely small, looking back, they were improperly scaled by temperature in Celsius, which is wrong for normalization. However the values show a decent picture for which parameters change the most due to temperature. From this basic analysis - Rds (switch and diode), Id_sat, Vt_thermal, and gds were affected by temperature
