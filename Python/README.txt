
# Time-of-flight Single Scatter Simulation (Python Implementation)
# Pre-alpha version | actively in development

## Overview
This Python script is an implementation of the Time-of-flight Single Scatter Simulation originally written in MATLAB
by Rodrigo Jose Santo, Andre Salomon, Hugo de Jong, Thibaut Merlin, Simon Stute, and Casper Beijst.
This implementation leverages parallel processing and Numba to enhance performance, making the code execution faster.

The Python translation was done by Thitiphat Klinsuwan, Adam Hopkins, Hamidreza Rashidy Kanan and Massimiliano Colarieti-Tosti.
Further enhancements and optimizations were done by Jeffrey Neele.


## Instructions

### Step 1: Data Conversion
If your data is in MATLAB format (.mat), you need to convert it to .npy format using the `mat2npy.py` script.
```bash
python mat2npy --load_path PATH_TO_MAT --save_path PATH_TO_NPY
```
This will convert the MATLAB file and save it as an .npy file to the specified path.

### Step 2: Data Organization
Organize your data in the following directory structure:
- Data/
  - Scanner/
  - Images/
  - AttenuationTable_AttenuationTable.npy
  - test/ (if needed)

### Step 3: Configuration
Edit the `parameters.txt` to set up the necessary parameters. This file will be used by the main script to configure the simulation settings.
Note: You can replace `parameters.txt` with e.g `parameters_Toyscanner.txt` in the current & next steps and achieve the same result

### Step 4: Running the Simulation
To start the simulation, run the following command:
```bash
python main.py --params parameters.txt
```

### Step 5: Results
The results will be stored as specified in `parameters.txt`. The output typically includes:
- Subfolders in the 'Images' directory containing sample slices.
- 'Scatters' directory containing:
  - AttenuationMask for tail fitting
  - Interpolated_scatters
  - Interpolated_scatters of each bin in binary format
- `Simulation_time.txt` will log the runtime of the simulation.
