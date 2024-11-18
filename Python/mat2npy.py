# This file is part of openSSS.
# 
#     openSSS is free software: you can redistribute it and/or modify it under the
#     terms of the GNU General Public License as published by the Free Software
#     Foundation, either version 3 of the License, or (at your option) any later
#     version.
# 
#     openSSS is distributed in the hope that it will be useful, but WITHOUT ANY
#     WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#     FOR A PARTICULAR PURPOSE.
# 
#     You should have received a copy of the License along with openSSS
# 
# Copyright 2022-2024 all openSSS contributors listed below:
# 
#     --> Rodrigo JOSE SANTO, Andre SALOMON, Hugo DE JONG, Thibaut MERLIN, Simon STUTE, Casper BEIJST,
#         Thitiphat KLINSUWAN, Jeffrey NEELE, Adam HOPKINS, Hamidreza RASHIDY KANAN, Massimiliano COLARIETI-TOSTI
# 
# This is openSSS version 0.2

import os
import numpy as np
from scipy.io import loadmat
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert matlab files(.mat) to numpy array files(.npy)")
    parser.add_argument('--load_path', type=str, required=True, help='Path where to load matlab files')
    parser.add_argument('--save_path', type=str, required=True, help='Path where to save numpy files')
    args = parser.parse_args()
    load_folder = args.load_path
    save_folder = args.save_folder

    # Check if the load_folder exists, if not raise an error
    if not os.path.exists(load_folder):
        raise FileNotFoundError(f"The specified load folder does not exist: {load_folder}")
    
    # Check if the save_folder exists, create if it does not and notify the user
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"The save folder was not found and has been created: {save_folder}")

    # Iterate through all files in the load_folder
    for filename in os.listdir(load_folder):
        if filename.endswith('.mat'):
            print('-'*50)
            # Construct the full path to the file
            file_path = os.path.join(load_folder, filename)
            print(f"Processing {file_path}")
            
            # Load the .mat file
            mat_data = loadmat(file_path)
            
            # Extract the base filename without extension for naming the npy files
            base_filename = os.path.splitext(filename)[0]
            
            # Save each variable from the .mat file as a separate .npy file and print status
            for key, value in mat_data.items():
                if key.startswith('__'):  # Skip meta keys
                    continue
                
                # Construct the npy file name
                npy_file_name = f"{base_filename}_{key}.npy"
                npy_file_path = os.path.join(save_folder, npy_file_name)
                
                # Save the numpy array as an .npy file
                np.save(npy_file_path, value)
                
                # Print status message
                print(f"{npy_file_name} is now created")
    print('-'*50)