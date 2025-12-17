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

import numpy as np
from numpy import ndarray # for comment the function
from typing import Tuple # for comment the function

from numba import jit # for fast computation

##################################################################################################
#################### SUPPORT FUNCTIONS FOR Single Scatter Simulation #############################
##################################################################################################

@jit(nopython=True)
def SinogramToSpatial(
    NrSectorsTrans: int, 
    NrSectorsAxial: int, 
    NrModulesAxial: int, 
    NrModulesTrans: int, 
    NrCrystalsTrans: int, 
    NrCrystalsAxial: int,
    Geom: np.ndarray,
    MinSectorDifference: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Maps sinogram coordinates to spatial coordinates of detectors and rings in a PET scanner.
    
    Parameters:
    - NrSectorsTrans (int): Number of sectors in the transaxial direction.
    - NrSectorsAxial (int): Number of sectors in the axial direction.
    - NrModulesAxial (int): Number of modules in the axial direction within a sector.
    - NrModulesTrans (int): Number of modules in the transaxial direction within a sector.
    - NrCrystalsTrans (int): Number of crystals in the transaxial direction within a module.
    - NrCrystalsAxial (int): Number of crystals in the axial direction within a module.
    - Geom (np.ndarray): Geometry array containing spatial coordinates of detectors and rings.
    - MinSectorDifference (int, optional): Minimum sector difference required for calculations. Default is 0.
    
    Returns:
    - DetectorCoordinates (np.ndarray) : gives the spatial coordinates of detectors (Radial, Angular, Detector, Coordinate).
    - RingCoordinate (np.ndarray) : gives the spatial coordinates of rings per sinogram index.
    """
    NrRings = NrSectorsAxial * NrModulesAxial * NrCrystalsAxial
    NrCrystalsPerRing = NrSectorsTrans * NrModulesTrans * NrCrystalsTrans

    RadialSize = NrCrystalsPerRing - 1  # Simplified, assumes no min sector difference affecting this value
    AngularSize = NrCrystalsPerRing // 2

    DetectorCoordinates = np.zeros((RadialSize + 1, AngularSize, 2, 2))
    RingCoordinates = np.zeros((NrRings * NrRings, 2))

    offset = (NrModulesTrans * NrCrystalsTrans) // 2

    # Compute detector coordinates
    for d1 in range(1, NrCrystalsPerRing + 1):
        id1 = ((d1 - 1) % NrCrystalsPerRing) - offset
        id1 = id1 + NrCrystalsPerRing if id1 < 0 else id1

        for d2 in range(d1, NrCrystalsPerRing + 1):
            id2 = ((d2 - 1) % NrCrystalsPerRing) - offset
            id2 = id2 + NrCrystalsPerRing if id2 < 0 else id2

            min_id, max_id = sorted([id1, id2])
            radial = max_id - min_id
            angular = (min_id + max_id) % AngularSize

            if radial >= MinSectorDifference:  # Ensures non-zero radial distance meeting the minimum sector difference
                DetectorCoordinates[radial, angular, 0, :] = Geom[0, d1 - 1, :2]
                DetectorCoordinates[radial, angular, 1, :] = Geom[0, d2 - 1, :2]

    # Compute ring coordinates
    for r1 in range(1, NrRings + 1):
        for r2 in range(r1, NrRings + 1):
            ring_diff = abs(r2 - r1)
            sinogram_index = r1 - 1 if ring_diff == 0 else NrRings * (r1 - 1) + r2 - 1

            RingCoordinates[sinogram_index, 0] = Geom[r1 - 1, 0, 2]
            RingCoordinates[sinogram_index, 1] = Geom[r2 - 1, 0, 2]

    return DetectorCoordinates.astype(np.float32), RingCoordinates.astype(np.float32)

##################################################################################################
#################### SUPPORT FUNCTIONS FOR MANIPULATIONS #########################################
##################################################################################################

def ReadParameters(
        file_path : str
        )->dict:
    
    """
    Reads all parameters from the parameter file
    
    Parameters:
    - file_path (str): Path to the parameter file.
    
    Returns:
    - params (dict) : Dictionary with all the parameters
    """

    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue  # Skip comments and empty lines
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            # Remove surrounding single or double quotes from the value
            if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                value = value[1:-1]
            
            if (value.startswith("(") and value.endswith(")")) :
                value = value[1:-1]
                allValues = value.split(", ")

                try: 
                    value = (int(allValues[0]), int(allValues[1]), int(allValues[2]))
                except ValueError:
                    pass  # Keep as string if not convertible

                if type(value) == str:
                    try:
                        value = (float(allValues[0]), float(allValues[1]), float(allValues[2]))
                    except ValueError:
                        pass # Keep as string if not convertible
            
            if type(value) == str:
                if (value.startswith("[") and value.endswith("]")):
                    value = value[1:-1]
                    allValues = value.split(", ")

                    try: 
                        rest = []
                        for number in allValues:
                            rest.append(int(number))
                        
                        value = rest
                    except ValueError:
                        pass  # Keep as string if not convertible

                    if type(value) == str:
                        try:
                            rest = []
                            for number in allValues:
                                rest.append(float(number))
                            value = rest
                        except ValueError:
                            pass # Keep as string if not convertible

            if value == 'True':
                value = True
            elif value == 'False':
                value = False

            # Try to convert to integer if possible
            if type(value) == str:
                try:
                    value = int(value)
                except ValueError:
                    pass  # Keep as string if not convertible

            if type(value) == str:
                try:
                    value = float(value)
                except ValueError:
                    pass # Keep as string if not convertible

            params[key] = value
    return params

def NormalizeArray(
    array: ndarray,
    target_min: float = 0,
    target_max: float = 255
) -> ndarray:
    
    """
    Normalizes a 3D array to a specified range [target_min, target_max].

    Parameters:
    - array (ndarray): The input 3D array.
    - target_min (float, optional): The minimum value of the target range.
    - target_max (float, optional): The maximum value of the target range.

    Returns:
    - scaled_array (ndarray) : The normalized array scaled to [target_min, target_max].
    """
    # Find the minimum and maximum values in the array
    min_val = np.min(array)
    max_val = np.max(array)

    # Normalize the array to [0, 1]
    normalized_array = (array - min_val) / (max_val - min_val) if max_val > min_val else array

    # Scale to [target_min, target_max]
    scaled_array = target_min + normalized_array * (target_max - target_min)
    
    # Convert to uint8 if the target range is 0-255
    if target_min == 0 and target_max == 255:
        scaled_array = scaled_array.astype(np.uint8)
    
    return scaled_array