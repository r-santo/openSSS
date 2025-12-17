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

from openSSS.RayTracing3DTOF import RayTracing3DTOF
from joblib import Parallel, delayed # for parallel processing

##################################################################################################
#################### SUPPORT FUNCTIONS FOR Single Scatter Simulation #############################
##################################################################################################

def BackProjection(
    GridSize : ndarray, 
    ScattersMasked : ndarray,
    RandomsMasked : ndarray,
    PromptsMasked : ndarray, 
    Ring1 : int,
    Geometry : ndarray,
    AccelerationFactor : int,
    NrRings : int,
    NrDetectors : int, 
    GridBounds : ndarray,
    SinogramCoordinates : ndarray
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:

    """
    Parallelized backprojection of scatters, prompts and randoms

    Parameters:
    - GridSize (ndarray): Number of voxels in each dimension to backproject into
    - ScattersMasked (ndarray) : Scatters in the sinogram space, masked for the tails
    - RandomsMasked (ndarray) : Randoms in the sinogram space, masked for the tails
    - PromptssMasked (ndarray) : Prompts in the sinogram space, masked for the tails
    - Ring1 (int) : Index of the first ring to be combined with all other rings for backprojection
    - Geometry (ndarray): (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - AccelerationFactor (int): number of detectors to skip when backprojecting for faster results
    - NrRings (int) : Number of rings in the scanner
    - NrDetectors (int) : Number of detectors in the scanner
    - GridBounds (ndarray) : Spatial coordinates of the backprojected image
    - SinogramCoordinates (ndarray) : Gives the sinogram coordinates for every pair of detectors (transaxially) 

    Returns:
    - ActivityImageTemp (ndarray): Backprojected prompts
    - ScatterImageTemp (ndarray): Backprojected scatters
    - RandomImageTemp (ndarray): Backprojected randoms
    - SensitivityImageTemp (ndarray): Backprojected sensitivity (all ones in every projected line)
    """

    ActivityImageTemp = np.zeros(GridSize, np.float32).flatten(order="F")
    ScatterImageTemp = np.zeros(GridSize, np.float32).flatten(order="F")
    RandomImageTemp = np.zeros(GridSize, np.float32).flatten(order="F")
    SensitivityImageTemp = np.zeros(GridSize, np.float32).flatten(order="F")

    zDetector1 = Geometry[Ring1, 0, 2] # z-coordinate detector 1

    # Detector 1
    for Detector1 in range(0, NrDetectors, AccelerationFactor):
        xDetector1 = Geometry[Ring1, Detector1, 0]                  # x-coordinate detector 1
        yDetector1 = Geometry[Ring1, Detector1, 1]                  # y-coordinate detector 1

        for Ring2 in range(NrRings):
            zDetector2 = Geometry[Ring2, 0, 2]              # z-coordinate detector 2

            # Allowed ring difference
            if np.abs(Ring2 - Ring1) <= NrRings:
                for Detector2 in range(0, NrDetectors, AccelerationFactor):

                    # Check if LOR is possible
                    if (Detector1 == Detector2 and Ring2 == Ring1) or Detector1 > Detector2:
                        # Skip, LOR not possible
                        continue
                    else:
                        xDetector2 = Geometry[Ring2, Detector2, 0]                  # x-coordinate detector 2
                        yDetector2 = Geometry[Ring2, Detector2, 1]                  # y-coordinate detector 2

                        # Look up at which position in scatter sinogram the combination of the two detectors is stored
                        AngularIndex = SinogramCoordinates[Detector1, Detector2, 0]
                        RadialIndex = SinogramCoordinates[Detector1, Detector2, 1]

                        if ScattersMasked[RadialIndex, AngularIndex, Ring2] == 0:
                            continue
                        else:
                            LineCoordinates = np.array([xDetector1, yDetector1, zDetector1, xDetector2, yDetector2, zDetector2])

                            Lengths, Indexes, _ = RayTracing3DTOF(GridSize, GridBounds, LineCoordinates)
                            Indexes = Indexes - 1    # fix for Python indexing, Indexes is a NumPy array
                            if len(Lengths) > 0:

                                ActivityImageTemp[Indexes] = ActivityImageTemp[Indexes] + PromptsMasked[RadialIndex, AngularIndex, Ring2] /len(Lengths)#* Lengths / np.sum(Lengths)
                                ScatterImageTemp[Indexes] = ScatterImageTemp[Indexes] + ScattersMasked[RadialIndex, AngularIndex, Ring2] /len(Lengths)#* Lengths / np.sum(Lengths)
                                RandomImageTemp[Indexes] = RandomImageTemp[Indexes] + RandomsMasked[RadialIndex, AngularIndex, Ring2] /len(Lengths)#* Lengths / np.sum(Lengths)
                                SensitivityImageTemp[Indexes] = SensitivityImageTemp[Indexes] + 1
    
    return ActivityImageTemp, ScatterImageTemp, RandomImageTemp, SensitivityImageTemp

def BackProjectionScatterRandFit(
        PromptSinogram : ndarray,
        ScatterSinogram : ndarray, 
        RandomSinogram : ndarray, 
        MaskSinogram : ndarray, 
        GridSize : ndarray, 
        ImageSize : ndarray, 
        Geometry : ndarray, 
        SinogramCoordinates : ndarray, 
        SinogramOrder : ndarray, 
        AccelerationFactor : int
        ) -> float:
    
    """
    Fitting of the estimated scatters to the prompts and randoms

    Parameters:
    - PromptsSinogram (ndarray) : Prompts in the sinogram space
    - ScatterSinogram (ndarray) : Scatters in the sinogram space
    - RandomSinogram (ndarray) : Randoms in the sinogram space
    - MasSinogram (ndarray) : Tails mask in the sinogram space
    - GridSize (ndarray): Number of voxels in each dimension to backproject into
    - ImageSize (ndarray) : Spatial coordinates of the backprojected image
    - Geometry (ndarray): (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - SinogramCoordinates (ndarray) : Gives the sinogram coordinates for every pair of detectors (transaxially)
    - SinogramOrder (ndarray) : Order of the sinograms for the ring pair
    - AccelerationFactor (int): number of detectors to skip when backprojecting for faster results 

    Returns:
    - scaleFactorp0 (float) : scale values for the scatters to match the prompts at the tails
    """
    
    NrDetectors = Geometry.shape[1]
    NrRings = Geometry.shape[0]
    GridBounds = ImageSize

    #variable to save the backprojected sinograms 
    ActivityImage = np.zeros(GridSize, np.single).flatten(order="F")
    ScatterImage = np.zeros(GridSize, np.single).flatten(order="F")
    RandomImage = np.zeros(GridSize, np.single).flatten(order="F")
    SensitivityImage = np.zeros(GridSize, np.single).flatten(order="F")

    ScattersMasked = MaskSinogram * ScatterSinogram
    ScattersMasked = ScattersMasked[:,:,SinogramOrder[:]]
    ScattersMasked = ScattersMasked.reshape(ScattersMasked.shape[0], ScattersMasked.shape[1], NrRings, NrRings)

    RandomsMasked = MaskSinogram * RandomSinogram
    RandomsMasked = RandomsMasked[:,:,SinogramOrder[:]]
    RandomsMasked = RandomsMasked.reshape(RandomsMasked.shape[0], RandomsMasked.shape[1], NrRings, NrRings)

    PromptsMasked = MaskSinogram * np.float32(PromptSinogram)
    PromptsMasked = PromptsMasked[:,:,SinogramOrder[:]]
    PromptsMasked = PromptsMasked.reshape(PromptsMasked.shape[0], PromptsMasked.shape[1], NrRings, NrRings)

    # Main part backprojection of sinograms
    # print('Starting backprojection... ')

    # start_time = time.time()  # to time the algorithm

    results = Parallel(n_jobs=20)(delayed
                                  (BackProjection)(
                                      GridSize, ScattersMasked[:,:,Ring1,:], RandomsMasked[:,:,Ring1,:], PromptsMasked[:,:,Ring1,:], 
                                      Ring1, Geometry, AccelerationFactor, NrRings, NrDetectors, 
                                      GridBounds, SinogramCoordinates
                                  )
                                  for Ring1 in range(NrRings)
                                  )
    

    for result in results:
        ActivityImage += result[0]
        ScatterImage += result[1]
        RandomImage += result[2]
        SensitivityImage += result[3]


    ScatterBackProjected = ScatterImage / SensitivityImage
    ScatterBackProjected[ np.isnan(ScatterBackProjected) ] = 0
    ScatterBackProjected[ np.isinf(ScatterBackProjected) ] = 0

    ActivityBackProjected = ActivityImage / SensitivityImage
    ActivityBackProjected[ np.isnan(ActivityBackProjected)] = 0
    ActivityBackProjected[ np.isinf(ActivityBackProjected) ] = 0

    RandomsBackProjected = RandomImage / SensitivityImage
    RandomsBackProjected[ np.isnan(RandomsBackProjected)] = 0
    RandomsBackProjected[ np.isinf(RandomsBackProjected)] = 0

    TailsBackProjected = ActivityBackProjected - RandomsBackProjected
    TailsBackProjected[TailsBackProjected < 0] = 0

    # Fitting function - compute the slope. The slope is used to compute the scale factor
    x = ScatterBackProjected[:, np.newaxis] * 1e10
    a, _, _, _ = np.linalg.lstsq(x, TailsBackProjected[:], rcond=None)

    scaleFactorp0 = a * 1e10

    # end_time = time.time()
    # print("Time for backprojection:", end_time - start_time, "seconds")
    
    return scaleFactorp0

def BackProjectionMashing(
        GridSize : ndarray,
        ScattersTemp : ndarray,
        RandomsTemp : ndarray,
        PromptsTemp : ndarray, 
        Ring1 : int,
        Ring2 : int,
        Geometry : ndarray, 
        AccelerationFactor : int,
        NrRings : int,
        NrDetectors : int, 
        GridBounds : ndarray,
        SinogramCoordinates : ndarray
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:

    """
    Parallelized backprojection of scatters, prompts and randoms for when using span

    Parameters:
    - GridSize (ndarray): Number of voxels in each dimension to backproject into
    - ScattersTemp (ndarray) : Scatters in the sinogram space, masked for the tails
    - RandomsTemp (ndarray) : Randoms in the sinogram space, masked for the tails
    - PromptsTemp (ndarray) : Prompts in the sinogram space, masked for the tails
    - Ring1 (int) : Index of the first ring to be combined
    - Ring2 (int) : Index of the second ring to be combined
    - Geometry (ndarray): (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - AccelerationFactor (int): number of detectors to skip when backprojecting for faster results
    - NrRings (int) : Number of rings in the scanner
    - NrDetectors (int) : Number of detectors in the scanner
    - GridBounds (ndarray) : Spatial coordinates of the backprojected image
    - SinogramCoordinates (ndarray) : Gives the sinogram coordinates for every pair of detectors (transaxially) 

    Returns:
    - ActivityImageTemp (ndarray): Backprojected prompts
    - ScatterImageTemp (ndarray): Backprojected scatters
    - RandomImageTemp (ndarray): Backprojected randoms
    - SensitivityImageTemp (ndarray): Backprojected sensitivity (all ones in every projected line)
    """

    ActivityImageTemp = np.zeros(GridSize, np.float32).flatten(order="F")
    ScatterImageTemp = np.zeros(GridSize, np.float32).flatten(order="F")
    RandomImageTemp = np.zeros(GridSize, np.float32).flatten(order="F")
    SensitivityImageTemp = np.zeros(GridSize, np.float32).flatten(order="F")

    zDetector1 = Geometry[Ring1, 0, 2] # z-coordinate detector 1
    zDetector2 = Geometry[Ring2, 0, 2] # z-coordinate detector 2

    for Detector1 in range(0, NrDetectors, AccelerationFactor):
        xDetector1 = Geometry[Ring1, Detector1, 0]                  # x-coordinate detector 1
        yDetector1 = Geometry[Ring1, Detector1, 1]                  # y-coordinate detector 1

        # Allowed ring difference
        if np.abs(Ring2 - Ring1) <= NrRings:
            for Detector2 in range(0, NrDetectors, AccelerationFactor):

                # Check if LOR is possible
                if (Detector1 == Detector2 and Ring2 == Ring1) or Detector1 > Detector2:
                    # Skip, LOR not possible
                    continue
                else:
                    xDetector2 = Geometry[Ring2, Detector2, 0]                  # x-coordinate detector 2
                    yDetector2 = Geometry[Ring2, Detector2, 1]                  # y-coordinate detector 2

                    # Look up at which position in scatter sinogram the combination of the two detectors is stored
                    AngularIndex = SinogramCoordinates[Detector1, Detector2, 0]
                    RadialIndex = SinogramCoordinates[Detector1, Detector2, 1]

                   # if ScattersTemp[RadialIndex, AngularIndex, Ring2] == 0:
                    if ScattersTemp[RadialIndex, AngularIndex] == 0:
                        continue
                    else:
                        LineCoordinates = np.array([xDetector1, yDetector1, zDetector1, xDetector2, yDetector2, zDetector2])

                        Lengths, Indexes, Rays = RayTracing3DTOF(GridSize, GridBounds, LineCoordinates)
                        Indexes = Indexes - 1    # fix for Python indexing, Indexes is a NumPy array
                        if len(Lengths) > 0:

                            ActivityImageTemp[Indexes] = ActivityImageTemp[Indexes] + PromptsTemp[RadialIndex, AngularIndex] * Lengths / np.sum(Lengths)
                            ScatterImageTemp[Indexes] = ScatterImageTemp[Indexes] + ScattersTemp[RadialIndex, AngularIndex] * Lengths / np.sum(Lengths)
                            RandomImageTemp[Indexes] = RandomImageTemp[Indexes] + RandomsTemp[RadialIndex, AngularIndex] * Lengths / np.sum(Lengths)
                            SensitivityImageTemp[Indexes] = SensitivityImageTemp[Indexes] + 1


    return ActivityImageTemp, ScatterImageTemp, RandomImageTemp, SensitivityImageTemp

def BackProjectionScatterRandFitMashing(
        PromptSinogram : ndarray,
        ScatterSinogram : ndarray,
        RandomSinogram : ndarray,
        MaskSinogram : ndarray, 
        GridSize : ndarray, 
        ImageSize : ndarray,
        Geometry : ndarray,
        SinogramCoordinates : ndarray, 
        AccelerationFactor : int,
        MashedSinogramIndices : ndarray
        ) -> float:
    
    """
    Fitting of the estimated scatters to the prompts and randoms with span applied

    Parameters:
    - PromptsSinogram (ndarray) : Prompts in the sinogram space
    - ScatterSinogram (ndarray) : Scatters in the sinogram space
    - RandomSinogram (ndarray) : Randoms in the sinogram space
    - MasSinogram (ndarray) : Tails mask in the sinogram space
    - GridSize (ndarray): Number of voxels in each dimension to backproject into
    - ImageSize (ndarray) : Spatial coordinates of the backprojected image
    - Geometry (ndarray): (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - SinogramCoordinates (ndarray) : Gives the sinogram coordinates for every pair of detectors (transaxially)
    - AccelerationFactor (int): number of detectors to skip when backprojecting for faster results 
    - MashedSinogramIndices (ndarray) : Gives the sinogram index (slice) for every ring pair
    Returns:
    - scaleFactorp0 (float) : scale values for the scatters to match the prompts at the tails
    """
    
    NrDetectors = Geometry.shape[1]
    NrRings = Geometry.shape[0]
    NrSinograms = ScatterSinogram.shape[2]
    GridBounds = np.array(ImageSize)

    #variable to save the backprojected sinograms 
    ActivityImage = np.zeros(GridSize, np.single).flatten(order="F")
    ScatterImage = np.zeros(GridSize, np.single).flatten(order="F")
    RandomImage = np.zeros(GridSize, np.single).flatten(order="F")
    SensitivityImage = np.zeros(GridSize, np.single).flatten(order="F")
    
    ScattersMasked = MaskSinogram * ScatterSinogram
    ScattersMasked = ScattersMasked.reshape(ScattersMasked.shape[0], ScattersMasked.shape[1], NrSinograms)

    RandomsMasked = MaskSinogram * RandomSinogram
    RandomsMasked = RandomsMasked.reshape(RandomsMasked.shape[0], RandomsMasked.shape[1], NrSinograms)

    PromptsMasked = MaskSinogram * np.float32(PromptSinogram)
    PromptsMasked = PromptsMasked.reshape(PromptsMasked.shape[0], PromptsMasked.shape[1], NrSinograms)

    # print('Starting backprojection... ')
    # start_time = time.time()  # to time the algorithm
    results = Parallel(n_jobs=20)(delayed
                                  (BackProjectionMashing)(
                                      GridSize, ScattersMasked[:,:,i], RandomsMasked[:,:,i], PromptsMasked[:,:,i], 
                                      int(MashedSinogramIndices[i,0] * 2), int(MashedSinogramIndices[i,1] * 2), Geometry, AccelerationFactor, NrRings, NrDetectors, 
                                      GridBounds, SinogramCoordinates
                                  )
                                  for i in range(NrSinograms)
                                  )
    
    for result in results:
        ActivityImage += result[0]
        ScatterImage += result[1]
        RandomImage += result[2]
        SensitivityImage += result[3]

    
    ScatterBackProjected = ScatterImage / (SensitivityImage + 1e-15)
    ScatterBackProjected[ np.isnan(ScatterBackProjected) ] = 0
    ScatterBackProjected[ np.isinf(ScatterBackProjected) ] = 0

    ActivityBackProjected = ActivityImage / (SensitivityImage + 1e-15)
    ActivityBackProjected[ np.isnan(ActivityBackProjected)] = 0
    ActivityBackProjected[ np.isinf(ActivityBackProjected) ] = 0

    RandomsBackProjected = RandomImage / (SensitivityImage + 1e-15)
    RandomsBackProjected[ np.isnan(RandomsBackProjected)] = 0
    RandomsBackProjected[ np.isinf(RandomsBackProjected)] = 0

    TailsBackProjected = ActivityBackProjected - RandomsBackProjected
    TailsBackProjected[TailsBackProjected < 0] = 0

    x = ScatterBackProjected[:, np.newaxis] * 1e10
    a, _, _, _ = np.linalg.lstsq(x, TailsBackProjected[:], rcond=None)
    scaleFactorp0 = a * 1e10

    # end_time = time.time()
    # print("Time for backprojection:", end_time - start_time, "seconds")

    return scaleFactorp0