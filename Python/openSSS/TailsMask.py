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

import os
import time
import numpy as np

from openSSS.RayTracing3DTOF import RayTracing3DTOF
from joblib import Parallel, delayed # for parallel processing

def GenerateTailsMask(
        SavePath : str, 
        ActivityMapDownscaled : np.ndarray, 
        AttenuationMapDownscaled : np.ndarray, 
        ImageSize : np.ndarray,
        Geometry : np.ndarray, 
        extendedGeometry : np.ndarray, 
        LORCoordinates : np.ndarray, 
        SinogramIndex : np.ndarray,
        AccelerationFactor : int, 
        UseAttenuation : bool = True, 
        Overwrite : bool = False
        )->np.ndarray:
    
    """
    Generates the tails mask for scaling, corresponding to regions where there should only be scatters
    
    Parameters:
    - SavePath (str): Directory path where to save the tails mask
    - ActivityMapDownscaled (ndarray): Attenuation map
    - AttenuationMapDownscaled (ndarray): Activity map
    - ImageSize (ndarray): Spatial coordinates of the boundaries of the maps
    - Geometry (ndarray): (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - extendedGeometry (ndarray): Geometry for when span is used (extended to include in-between virtual rings)
    - LORCoordinates (ndarray) : Gives the sinogram coordinates for every pair of detectors (transaxially) 
    - SinogramIndex (ndarray) : Gives the sinogram index (slice) for every pair of rings
    - AccelerationFactor (int): number of detectors to skip when forwardprojecting for faster results
    - UseAttenuation (bool, option) : Flag to indicate if tails is estimated from the attenuation map or activity. Default is attenuation
    - Overwrite (bool, option) : Flag to indicate if the existing tails mask should be overwritten if already exists. Default is false
    
    Returns:
    - AttenuationMask (ndarray): Tails mask in the sinogram space, for the sampled detectors used in scaling
    """
    
    # # Create tails mask
    if not os.path.isfile(f'{SavePath}/AttenuationMask.npy') or Overwrite:
        print('Start generating AttenuationMask')

        if extendedGeometry.shape[0] != 0:
            print('With span')
            AttenuationMask = MaskGeneratorMashing(ActivityMapDownscaled, AttenuationMapDownscaled, ImageSize, \
                                                   extendedGeometry, LORCoordinates, SinogramIndex, \
                                                    UseAttenuation, AccelerationFactor)
        else:
            print('No span')
            AttenuationMask = MaskGenerator(ActivityMapDownscaled, AttenuationMapDownscaled, ImageSize, \
                                            Geometry, LORCoordinates, SinogramIndex, \
                                            UseAttenuation, AccelerationFactor)
            
        np.save(f'{SavePath}/AttenuationMask.npy', AttenuationMask)
        print('completed!\n')
    else:
        AttenuationMask = np.load(f'{SavePath}/AttenuationMask.npy')

    return AttenuationMask
    
def MaskDetectors(
        Geometry : np.ndarray,
        SinogramCoordinates : np.ndarray,
        UseAttenuation : bool, 
        AccelerationFactor : int, 
        GridSize : np.ndarray, 
        GridBounds : np.ndarray, 
        ActivityMap_flat : np.ndarray,
        AttenuationMap_flat : np.ndarray, 
        Ring1 : int
        )->np.ndarray:
    
    """
    Parallelized back projection to calculate the tails mask with span

    Parameters:
    - Geometry (ndarray): (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - SinogramCoordinates (ndarray) : Gives the sinogram coordinates for every pair of detectors (transaxially) 
    - UseAttenuation (bool, option) : Flag to indicate if tails is estimated from the attenuation map or activity. Default is attenuation
    - AccelerationFactor (int): number of detectors to skip when forwardprojecting for faster results
    - GridSize (ndarray): Number of voxels of the maps
    - GridBounds (ndarray): Spatial coordinates of the boundaries of the maps
    - ActivityMapDownscaled (ndarray): Attenuation map flattened
    - AttenuationMapDownscaled (ndarray): Activity map flattened
    - Ring1 (int) : Index of the first ring index to combine with all others for forwardprojection

    Returns:
    - SinogramsTemp (ndarray): Tails mask in the sinogram space, for all the sinogram slices that include Ring1
    """

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

                            LineCoordinates = np.array([xDetector1, yDetector1, zDetector1, xDetector2, yDetector2, zDetector2])
                            Lenghts, Indexes, _ = RayTracing3DTOF(GridSize, GridBounds, LineCoordinates)
                            Indexes = Indexes - 1

                            if len(Lenghts) != 0:
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

def MaskGenerator(
        ActivityMap : np.ndarray,
        AttenuationMap : np.ndarray,
        ImageSize : np.ndarray,
        Geometry : np.ndarray,
        SinogramCoordinates : np.ndarray,
        SinogramIndex : np.ndarray,
        UseAttenuation : bool, 
        AccelerationFactor : int = 1
        )->np.ndarray:
    
    """
    Prepares forwardprojection in parallel way without span

    Parameters:
    - ActivityMapDownscaled (ndarray): Attenuation map
    - AttenuationMapDownscaled (ndarray): Activity map
    - ImageSize (ndarray): Spatial coordinates of the boundaries of the maps
    - Geometry (ndarray): (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - SinogramCoordinates (ndarray) : Gives the sinogram coordinates for every pair of detectors (transaxially) 
    - SinogramIndex (ndarray) : Gives the sinogram index (slice) for every pair of rings
    - UseAttenuation (bool) : Flag to indicate if tails is estimated from the attenuation map or activity. Default is attenuation
    - AccelerationFactor (int, optional): number of detectors to skip when forwardprojecting for faster results

    Returns:
    - Sinograms (ndarray): Tails mask in the sinogram space
    """

    if AccelerationFactor is None:
        AccelerationFactor = 1

    NrDetectors = Geometry.shape[1]
    NrRings = Geometry.shape[0]
    GridSize = np.array(ActivityMap.shape)
    GridBounds = ImageSize
    NrSinograms = NrRings ** 2

    Sinograms = np.zeros((NrDetectors + 1, NrDetectors // 2, NrRings, NrRings), dtype=np.uint8)

    ActivityMap_flat = ActivityMap.copy()
    ActivityMap_flat[ActivityMap < 1e-5] = 0
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

def MaskDetectorsMashing(
        Geometry : np.ndarray,
        SinogramCoordinates : np.ndarray,
        UseAttenuation : bool,
        AccelerationFactor : int, 
        GridSize : np.ndarray, 
        GridBounds : np.ndarray, 
        ActivityMap_flat : np.ndarray, 
        AttenuationMap_flat : np.ndarray, 
        Ring1 : int, 
        Ring2 : int):
    
    """
    Parallelized back projection to calculate the tails mask with span

    Parameters:
    - Geometry (ndarray): (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - SinogramCoordinates (ndarray) : Gives the sinogram coordinates for every pair of detectors (transaxially) 
    - UseAttenuation (bool) : Flag to indicate if tails is estimated from the attenuation map or activity. Default is attenuation
    - AccelerationFactor (int): number of detectors to skip when forwardprojecting for faster results
    - GridSize (ndarray): Number of voxels of the maps
    - GridBounds (ndarray): Spatial coordinates of the boundaries of the maps
    - ActivityMapDownscaled (ndarray): Attenuation map flattened
    - AttenuationMapDownscaled (ndarray): Activity map flattened
    - Ring1 (int) : Index of the first ring index
    - Ring1 (int) : Index of the second ring index

    Returns:
    - SinogramsTemp (ndarray): Tails mask in the sinogram space, for the sinogram slice of Ring1 and Ring2
    """
    
    NrDetectors = Geometry.shape[1]
    NrRings = Geometry.shape[0]
    
    SinogramsTemp = np.zeros((NrDetectors + 1, NrDetectors // 2), dtype=np.uint8)

    zDetector1 = Geometry[Ring1, 0, 2]
    zDetector2 = Geometry[Ring2, 0, 2]

    for Detector1 in range(0, NrDetectors, AccelerationFactor):
        xDetector1 = Geometry[Ring1, Detector1, 0]
        yDetector1 = Geometry[Ring1, Detector1, 1]

        if abs(Ring2 - Ring1) <= NrRings:
            for Detector2 in range(0, NrDetectors, AccelerationFactor):
                if (Detector1 == Detector2 and Ring2 == Ring1) or (Detector1 > Detector2):
                    continue
                else:
                    xDetector2 = Geometry[Ring2, Detector2, 0]
                    yDetector2 = Geometry[Ring2, Detector2, 1]

                    AngularIndex = SinogramCoordinates[Detector1, Detector2, 0]
                    RadialIndex = SinogramCoordinates[Detector1, Detector2, 1]

                    LineCoordinates = np.array([xDetector1, yDetector1, zDetector1, xDetector2, yDetector2, zDetector2])
                    Lenghts, Indexes, _ = RayTracing3DTOF(GridSize, GridBounds, LineCoordinates)
                    Indexes = Indexes - 1

                    if len(Lenghts) != 0:
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
                        SinogramsTemp[RadialIndex, AngularIndex] += np.uint8(Attenuation)
                    else:
                        SinogramsTemp[RadialIndex, AngularIndex] += np.uint8(Activity)

    return SinogramsTemp

def MaskGeneratorMashing(
        ActivityMap : np.ndarray, 
        AttenuationMap : np.ndarray, 
        ImageSize : np.ndarray, 
        Geometry : np.ndarray, 
        SinogramCoordinates : np.ndarray, 
        MashedSinogramIndices : np.ndarray,
        UseAttenuation : bool,
        AccelerationFactor : int = 1
        )->np.ndarray:
    
    """
    Prepares forwardprojection in parallel way with span

    Parameters:
    - ActivityMapDownscaled (ndarray): Attenuation map
    - AttenuationMapDownscaled (ndarray): Activity map
    - ImageSize (ndarray): Spatial coordinates of the boundaries of the maps
    - Geometry (ndarray): (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - SinogramCoordinates (ndarray) : Gives the sinogram coordinates for every pair of detectors (transaxially) 
    - MashedSinogramIndices (ndarray) : Gives the rings for each sinogram slice
    - UseAttenuation (bool) : Flag to indicate if tails is estimated from the attenuation map or activity. Default is attenuation
    - AccelerationFactor (int, optional): number of detectors to skip when forwardprojecting for faster results

    Returns:
    - Sinograms (ndarray): Tails mask in the sinogram space
    """
    
    if AccelerationFactor is None:
        AccelerationFactor = 1

    NrDetectors = Geometry.shape[1]
    NrRings = Geometry.shape[0]
    GridSize = np.array(ActivityMap.shape)
    GridBounds = ImageSize
    NrSinograms = MashedSinogramIndices.shape[0]

    Sinograms = np.zeros((NrDetectors + 1, NrDetectors // 2, NrSinograms), dtype=np.uint8)

    ActivityMap_flat = ActivityMap.copy()
    ActivityMap_flat[ActivityMap < 1e-5] = 0
    ActivityMap_flat = ActivityMap.flatten(order="F")
    AttenuationMap_flat = AttenuationMap.flatten(order="F")

    print('Generating tail mask... ')
    start_time = time.time()  # to time the algorithm

    results = Parallel(n_jobs = 20)(delayed
                                    (MaskDetectorsMashing)
                                    (Geometry, SinogramCoordinates, UseAttenuation, AccelerationFactor, 
                                     GridSize, GridBounds, ActivityMap_flat, AttenuationMap_flat,
                                     int(MashedSinogramIndices[i][0] * 2), int(MashedSinogramIndices[i][1] * 2))
                                    for i in range(MashedSinogramIndices.shape[0])
                                    )
    
    # for i in range(MashedSinogramIndices.shape[0]):
    #     results = MaskDetectorsMashing(Geometry, SinogramCoordinates, UseAttenuation, AccelerationFactor, 
    #                                     GridSize, GridBounds, ActivityMap_flat, AttenuationMap_flat,
    #                                     int(MashedSinogramIndices[i][0] * 2), int(MashedSinogramIndices[i][1] * 2))
    
    for i, result in enumerate(results):
        Sinograms[:,:,i] = result

    end_time = time.time()
    print("Time for Generate Tail Mask:", end_time - start_time, "seconds")

    return Sinograms