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

import time
import numpy as np
from skimage.transform import resize
from numpy import ndarray # for comment the function
from typing import Tuple, List, Union # for comment the function

from numba import jit # for fast computation
from RayTracing3DTOF import RayTracing3DTOF
from joblib import Parallel, delayed # for parallel processing

##################################################################################################
#################### SUPPORT FUNCTIONS FOR Single Scatter Simulation #############################
##################################################################################################

def CropAndDownscale(
    image: ndarray, 
    voxel_size: Union[Tuple[int, int, int], List[int]], 
    crop_size: Union[Tuple[int, int, int], List[int]], 
    downscaled_dimensions: Union[Tuple[int, int, int], List[int]], 
    clip: bool = True, 
    mode: str = 'edge', 
    preserve_range: bool = True, 
    anti_aliasing: bool = True, 
    order: int = 3
) -> ndarray:
    """
    Crops and downscales an image.
    Parameters:
    - image (ndarray): The original image to be cropped and downscaled.
    - voxel_size (tuple or list): Size of the voxels in the image [xLength, yLength, zLength] in mm.
    - crop_size (tuple or list): Desired size for the cropped image [xLength, yLength, zLength] in mm.
    - downscaled_dimensions (tuple or list): Dimensions for the downscaled image [xDim, yDim, zDim].
    - clip (bool, optional): Whether to clip the range at the end. Defaults to True.
    - mode (str, optional): How the input array is extended beyond its boundaries. Defaults to 'edge'.
    - preserve_range (bool, optional): Whether to keep the original range of values. Defaults to True.
    - anti_aliasing (bool, optional): Whether to apply a Gaussian filter to smooth the image prior to downsampling. Defaults to True.
    - order (int, optional): The order of the spline interpolation used when resizing. Defaults to 3 for cubic spline interpolation.

    Returns:
    - downscaled_image (ndarray): The cropped and downscaled image.
    """
    # Calculate half the thickness in pixels
    pixel_half_thickness = np.ceil(np.array(crop_size) / np.array(voxel_size) / 2).astype(int)
    center = np.array(image.shape) // 2

    # Initialize cropped image
    cropped_image = image

    # Cropping or padding each dimension as necessary
    for dim in range(3):
        start_idx = max(center[dim] - pixel_half_thickness[dim], 0)
        end_idx = min(center[dim] + pixel_half_thickness[dim], image.shape[dim])

        if start_idx == 0 and end_idx < pixel_half_thickness[dim] * 2:
            pad_width = [(0, 0)] * 3
            pad_width[dim] = (0, pixel_half_thickness[dim] * 2 - end_idx)
            cropped_image = np.pad(cropped_image, pad_width, mode='constant', constant_values=0)
        
        # Apply the slicing based on calculated indices
        slicing = [slice(None)] * 3
        slicing[dim] = slice(start_idx, end_idx)
        cropped_image = cropped_image[tuple(slicing)]
    
    # Resizing the image to the specified dimensions
    downscaled_image = resize(cropped_image, downscaled_dimensions, order=order, clip=clip,
                              mode=mode, preserve_range=preserve_range, anti_aliasing=anti_aliasing)
    return downscaled_image

@jit(nopython=True)
def SinogramCoordinates(
    NrSectorsTrans: int, 
    NrSectorsAxial: int, 
    NrModulesAxial: int, 
    NrModulesTrans: int, 
    NrCrystalsTrans: int, 
    NrCrystalsAxial: int,
    MinSectorDifference: int = 0
) -> Tuple[ndarray, ndarray]:
    """
    Calculates sinogram coordinates for every detector combination and sinogram indices for every ring combination.

    Parameters:
    NrSectorsTrans (int): Number of sectors in the transaxial direction.
    NrSectorsAxial (int): Number of sectors in the axial direction.
    NrModulesAxial (int): Number of modules inside a sector in the axial direction.
    NrModulesTrans (int): Number of modules inside a sector in the transaxial direction.
    NrCrystalsTrans (int): Number of crystals inside a module in the transaxial direction.
    NrCrystalsAxial (int): Number of crystals inside a module in the axial direction.
    MinSectorDifference (int, optional): Minimum sector difference required to consider two sectors. Default is 0.

    Returns:
        LORCoordinates (ndarray): Sinogram coordinates for every detector combination.
        SinogramIndex (ndarray): Sinogram coordinates for every ring combination.
    """
    NrRings = NrSectorsAxial * NrModulesAxial * NrCrystalsAxial
    NrCrystalsPerRing = NrSectorsTrans * NrModulesTrans * NrCrystalsTrans
    MinCrystalDifference = MinSectorDifference * NrModulesTrans * NrCrystalsTrans

    RadialSize = NrCrystalsPerRing - 2 * (MinCrystalDifference - 1) - 1
    AngularSize = NrCrystalsPerRing // 2
    NrSinograms = NrRings * NrRings

    DistanceCrystalId0toFirstSectorCenter = (NrModulesTrans * NrCrystalsTrans) // 2

    LORCoordinates = np.zeros((NrCrystalsPerRing, NrCrystalsPerRing, 2))

    for Detector1 in range(NrCrystalsPerRing):
        castorFullRingCrystalID1 = Detector1
        CrystalId1 = castorFullRingCrystalID1 % NrCrystalsPerRing - DistanceCrystalId0toFirstSectorCenter

        for Detector2 in range(NrCrystalsPerRing):
            castorFullRingCrystalID2 = Detector2
            CrystalId2 = castorFullRingCrystalID2 % NrCrystalsPerRing - DistanceCrystalId0toFirstSectorCenter

            if CrystalId1 < 0:
                CrystalId1 += NrCrystalsPerRing
            if CrystalId2 < 0:
                CrystalId2 += NrCrystalsPerRing

            IdA, IdB = sorted([CrystalId1, CrystalId2])
            RingIdA = castorFullRingCrystalID1 // NrCrystalsPerRing
            RingIdB = castorFullRingCrystalID2 // NrCrystalsPerRing

            Radial = Angular = 0

            if IdB - IdA < MinCrystalDifference:
                continue
            else:
                if IdA + IdB >= (3 * NrCrystalsPerRing) // 2 or IdA + IdB < NrCrystalsPerRing // 2:
                    if IdA == IdB:
                        Radial = -NrCrystalsPerRing // 2
                    else:
                        Radial = ((IdB - IdA - 1) // 2) - ((NrCrystalsPerRing - (IdB - IdA + 1)) // 2)
                else:
                    if IdA == IdB:
                        Radial = NrCrystalsPerRing // 2
                    else:
                        Radial = ((NrCrystalsPerRing - (IdB - IdA + 1)) // 2) - ((IdB - IdA - 1) // 2)

                Radial = int(np.floor(Radial))

                if IdA + IdB < NrCrystalsPerRing // 2:
                    Angular = (2 * IdA + NrCrystalsPerRing + Radial) // 2
                else:
                    if IdA + IdB >= (3 * NrCrystalsPerRing) // 2:
                        Angular = (2 * IdA - NrCrystalsPerRing + Radial) // 2
                    else:
                        Angular = (2 * IdA - Radial) // 2

                # Python Coordinate
                LORCoordinates[Detector1, Detector2, 0] = int(np.floor(Angular))
                LORCoordinates[Detector1, Detector2, 1] = int(np.floor(Radial + RadialSize // 2))

    SinogramIndex = np.zeros((NrRings, NrRings))

    for Ring1 in range(NrRings):
        for Ring2 in range(NrRings):
            RingDifference = abs(Ring2 - Ring1)
            if RingDifference == 0:
                CurrentSinogramIndex = Ring1
            else:
                CurrentSinogramIndex = NrRings
                if Ring1 < Ring2:
                    if RingDifference > 1:
                        for RingDistance in range(1, RingDifference):
                            CurrentSinogramIndex += 2 * (NrRings - RingDistance)
                    CurrentSinogramIndex += Ring1
                else:
                    if RingDifference > 1:
                        for RingDistance in range(1, RingDifference):
                            CurrentSinogramIndex += 2 * (NrRings - RingDistance)
                    CurrentSinogramIndex += NrRings - RingDifference + Ring1 - RingDifference

            # Python Coordinate
            SinogramIndex[Ring1, Ring2] = CurrentSinogramIndex

    return LORCoordinates.astype(np.int16), SinogramIndex.astype(np.int16)

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
      - DetectorCoordinates: np.ndarray detailing the spatial coordinates of detectors (Radial, Angular, Detector, Coordinate).
      - RingCoordinates: np.ndarray detailing the spatial coordinates of rings per sinogram index.
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


def MaskDetectors(Geometry, SinogramCoordinates, UseAttenuation, AccelerationFactor, 
                  GridSize, GridBounds, 
                  ActivityMap_flat, AttenuationMap_flat, 
                  Ring1):

    NrDetectors = Geometry.shape[1]
    NrRings = Geometry.shape[0]
    
    SinogramsTemp = np.zeros((NrDetectors + 1, NrDetectors // 2, NrRings), dtype=np.uint8)
    zDetector1 = Geometry[Ring1, 0, 2]

    for Detector1 in range(0, NrDetectors, AccelerationFactor):
            xDetector1 = Geometry[Ring1, Detector1, 0]
            yDetector1 = Geometry[Ring1, Detector1, 1]

            for Ring2 in range(NrRings):
                zDetector2 = Geometry[Ring2, 0, 2]

                if abs(Ring2 - Ring1) <= NrRings:
                    for Detector2 in range(0, NrDetectors, AccelerationFactor):
                        if (Detector1 == Detector2 and Ring2 == Ring1) or (Detector1 > Detector2):
                            continue
                        else:
                            xDetector2 = Geometry[Ring2, Detector2, 0]
                            yDetector2 = Geometry[Ring2, Detector2, 1]

                            AngularIndex = SinogramCoordinates[Detector1, Detector2, 0]
                            RadialIndex = SinogramCoordinates[Detector1, Detector2, 1]

                            LineCoordinates = [xDetector1, yDetector1, zDetector1, xDetector2, yDetector2, zDetector2]
                            Lenghts, Indexes, _ = RayTracing3DTOF(GridSize, GridBounds, LineCoordinates)
                            Indexes = Indexes - 1

                            if len(Lenghts) != 0:
                                # print('detected')
                                # in matlab, they retrieve the element at the linear index x of the ActivityMap array, regardless of its dimensions.
                                Activity = np.sum(ActivityMap_flat[Indexes] * Lenghts)
                                Attenuation = 1 / np.exp(-np.sum(AttenuationMap_flat[Indexes] * Lenghts))
                            else:
                                Activity = 0
                                Attenuation = 1

                            if Attenuation != 1:
                                Attenuation = 0

                            if Activity > 0:
                                Activity = 0
                            else:
                                Activity = 1

                            if UseAttenuation:
                                SinogramsTemp[RadialIndex, AngularIndex, Ring2] += np.uint8(Attenuation)
                            else:
                                SinogramsTemp[RadialIndex, AngularIndex, Ring2] += np.uint8(Activity)

    return SinogramsTemp

def MaskGenerator(ActivityMap, AttenuationMap, ImageSize, Geometry, SinogramCoordinates, SinogramIndex,
                     UseAttenuation, AccelerationFactor=1):
    if AccelerationFactor is None:
        AccelerationFactor = 1

    NrDetectors = Geometry.shape[1]
    NrRings = Geometry.shape[0]
    GridSize = list(ActivityMap.shape)
    GridBounds = ImageSize
    NrSinograms = NrRings ** 2

    Sinograms = np.zeros((NrDetectors + 1, NrDetectors // 2, NrRings, NrRings), dtype=np.uint8)

    ActivityMap[ActivityMap < 1e-5] = 0
    ActivityMap_flat = ActivityMap.flatten(order="F")
    AttenuationMap_flat = AttenuationMap.flatten(order="F")

    print('Generating tail mask... ')
    start_time = time.time()  # to time the algorithm

    results = Parallel(n_jobs = 20)(delayed
                                    (MaskDetectors)
                                    (Geometry, SinogramCoordinates, UseAttenuation, AccelerationFactor, GridSize, GridBounds, ActivityMap_flat, AttenuationMap_flat, i)
        for i in range(NrRings))
    
    for Ring1, result in enumerate(results):
        Sinograms[:,:,:,Ring1] = result

    SinogramOrder = SinogramIndex[:NrRings, :NrRings].T.flatten()
    SinogramOrder = np.argsort(SinogramOrder)

    Sinograms = np.reshape(Sinograms, (NrDetectors + 1, NrDetectors // 2, NrSinograms))
    Sinograms = Sinograms[:, :, SinogramOrder]

    end_time = time.time()
    print("Time for Generate Tail Mask:", end_time - start_time, "seconds")
    return Sinograms

##################################################################################################
#################### SUPPORT FUNCTIONS FOR MANIPULATIONS #########################################
##################################################################################################

def read_parameters(file_path):
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
            # Try to convert to integer if possible
            try:
                value = int(value)
            except ValueError:
                pass  # Keep as string if not convertible
            params[key] = value
    return params

def normalize_array(
    array: ndarray,
    target_min: int = 0,
    target_max: int = 255
) -> ndarray:
    """
    Normalizes a 3D array to a specified range [target_min, target_max].

    Parameters:
        array (ndarray): The input 3D array.
        target_min (int or float): The minimum value of the target range.
        target_max (int or float): The maximum value of the target range.

    Returns:
        ndarray: The normalized array scaled to [target_min, target_max].
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