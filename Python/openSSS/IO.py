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
# Copyright 2022-2025 all openSSS contributors listed below:
# 
#     --> Rodrigo JOSE SANTO, Andre SALOMON, Hugo DE JONG, Thibaut MERLIN, Simon STUTE, Casper BEIJST,
#         Thitiphat KLINSUWAN, Hamidreza RASHIDY KANAN, Massimiliano COLARIETI-TOSTI, Jeffrey NEELE
# 
# This is openSSS version 1.0

import os, shutil, argparse
import numpy as np
from Timber import Datafile

# Supporting functions
def ProcessInput(
    )->tuple[int, str]:

    """
    Processed the input parameters when calling openSSS from the terminal
    
    Returns:
    - args.iterations (int) : Number of iterations to run openSSS and reconstruct (on top of the step 0, reconstructing without scatters)
    - args.parms (str) : Path to the parameter file
    """

    parser = argparse.ArgumentParser(description="Run the Single Scatter Simulation with TOF and specify the parameters path.")
    parser.add_argument('--params', type=str, required=True, help='Path where to read setting parameters')
    parser.add_argument('--iterations', type=int, required=False, help='Number of iterations to perform')
    args = parser.parse_args()

    return args.iterations, args.params

def CreateDirectories(
        targetFolder : str, 
        folderImages : str, 
        folderScatters : str, 
        parameters : str
        ):

    """
    Creates the directories to save the different outputs of openSSS
    
    Parameters:
    - targetFolder (str): Directory path with the datafile and maps.
    - folderImages (str): Directory path to save the images of the downscaled maps, maks sinogram and scatter sinogram.
    - folderScatters (str): Directory path to save the estimated scatters.
    - parameters (str): Path to the parameter file.
    
    """

    # Create directories to store the result
    if not os.path.exists(os.path.join(targetFolder, folderImages)):
        os.makedirs(os.path.join(targetFolder, folderImages))
    if not os.path.exists(os.path.join(targetFolder, folderScatters)):
        os.makedirs(os.path.join(targetFolder, folderScatters))
    
    destination_path = os.path.join(targetFolder, 'parameters_backup.txt')
    try:
        shutil.copy(parameters, destination_path)
        print(f"Used parameters saved to {targetFolder}")
    except IOError as e:
        print(f"Unable to copy file. {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def GeneratePrompts(
        SavePath : str, 
        DataPath : str, 
        DataHeader : str, 
        LORCoordinates : np.ndarray, 
        SinogramIndex : np.ndarray, 
        LookUpTable : np.ndarray, 
        TOFbins : int, 
        TOFRange : float, 
        Span : int, 
        Mash : int, 
        Shift : int = 0
        ):
    
    """
    Generates the prompts, randoms and normalization from the CASToR list-mode datafile
    
    Parameters:
    - SavePath (str): Directory path to save the prompts read from the datafile.
    - DataPath (str): Directory path of the CASToR datafile.
    - DataHeader (str): Name of the CASToR list-mode datafile.
    - LORCoordinates (ndarray): Gives the sinogram coordinates for every detector pair (transaxially).
    - SinogramIndex (ndarray): Gives the sinogram slice for every ring pair.
    - LookUpTable (ndarray): Gives the sinogram slice for every real ring pair, for when span is used.
    - TOFbins (int): Number of TOF bins desired for the scatter estimation.
    - TOFRange (float): TOF range of the corresponding TOF bins.
    - Span (int): Value for the span.
    - Mash (int): Value for the mash.
    - Shift (int, optional): Number of crystals to skip on the first modules, depending on the order in the geometry.
    
    """
    
    # Create Prompts - No Span
    if Span == 1:
        if not os.path.isfile(f'{SavePath}/PromptsSinogram.npz') and Span == 1: #No Span applied:
            print('Creating Prompts - no span\n')
            Prompts, Randoms, Norm = Datafile.ExportPrompts(DataPath, DataHeader, LORCoordinates, SinogramIndex, TOFbins, TOFRange, mash=Mash, Shift=Shift)
            np.savez_compressed(f'{SavePath}/PromptsSinogram.npz', Prompts)
            np.savez_compressed(f'{SavePath}/RandomsSinogram.npz', Randoms)
            np.savez_compressed(f'{SavePath}/NormSinogram.npz', Norm)
            print('completed!\n')
        # else:
        #     Prompts = np.load(f'{SavePath}/PromptsSinogram.npz')['arr_0']
    else:
        # Create Prompts - With Span
        if not os.path.isfile(f'{SavePath}/PromptsSinogramMashed.npz') and Span > 1: #Span is applied
            print('Creating Prompts - Span\n')
            Prompts, Randoms, Norm = Datafile.ExportPrompts(DataPath, DataHeader, LORCoordinates, LookUpTable, TOFbins, TOFRange, mash=Mash, Shift=Shift)
            np.savez_compressed(f'{SavePath}/PromptsSinogramMashed.npz', Prompts)
            np.savez_compressed(f'{SavePath}/RandomsSinogram.npz', Randoms)
            np.savez_compressed(f'{SavePath}/NormSinogram.npz', Norm)
            print('completed!\n')
        # else:
        #     Prompts = np.load(f'{SavePath}/PromptsSinogramMashed.npz')['arr_0']