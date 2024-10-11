# *Monitoring the success of nutrient reduction measures within eutrophic lakes using Sentinel-2*

# [Sentinel-2-basierte Erfolgskontrolle von Sanierungsmaßnahmen zur Nährstoffminderung in eutrophen Gewässern]

by: R. Riethmueller, 2024 (@rRie-code);
version: 1.0 (DE)

>This script provides time series analysis for Sentinel-2 based chlorophyll-a concentrations in water. <br>
First, satellite scenes were processed with the SeNtinel Applications Platform and the inbuilt Case 2 Regional CoastColor processor (C2RCC), using the C2X-Complex neural net. The results were exported with the PixEx operator (v1.3) without spatial aggregation. <br>
Here, the pixel information of each scene is aggregated and scene-wise statistics are calculated. Then, data is plotted as a time series (chl-a vs. time) and compared with in situ data (validation). Finally, trophic state indices are calculated and presented as a time series.

## **Software versions:**

### Developed on Spyder v5.5.1 with Python v3.10.14

### Non-standard Python packages used
- cdsapi v0.7.0
- matplotlib v3.8.4 
- pandas v2.2.2 
- requests v2.32.3
- scipy v1.13.1 
- seaborn 0.13.2 
- statsmodels 0.14.2 
- xarray 2024.7.0

### Related processing in SNAP
- SNAP v10.0.0 with optical toolbox v10.0.3 
- S2Resampling processor v1.0 
- IdePix S2-MSI v9.0.2
- C2RCC.MSI processor v1.2, C2X-Complex neural net
- PixEx operator v1.3
