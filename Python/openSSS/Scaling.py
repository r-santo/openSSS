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

import time
import numpy as np

from numpy import ndarray # for comment the function
from openSSS.BackProjection import BackProjectionScatterRandFit, BackProjectionScatterRandFitMashing

def ScaleScattersToPrompts(
    SinogramIndex: ndarray,
    SinogramCoordinates: ndarray,
    SavePath: str,
    DesiredDimensions: ndarray,
    FittingSize: ndarray,
    Geometry: ndarray,
    ExtendedGeometry: ndarray,
    AccelerationFactor: int,
    SpanFlag: bool,
    LORCounts: ndarray,
) -> ndarray:
    
    """
    Scales the scatter distribution to the prompts
    
    Parameters:
    - SinogramIndex (ndarray): Array with the order of the sinograms for ring combinations
    - SinogramCoordinates (ndarray): Array with the sinogram coordinates for detector combinations
    - SavePath (str): Path where to scatter sinograms are saved
    - DesiredDimensions (ndarray): Voxel dimensions of the backprojected images
    - FittingSize (ndarray): Spatial size of the bakprojected images
    - Geometry (ndarray): (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - ExtendedGeometry (ndarray): Geometry for when span is used (extended to include in-between virtual rings)
    - AccelerationFactor (int): number of detectors to skip when backprojecting for faster results
    - SpanFlag (bool): flag to indicate if span is used (data reduction at the ring leval - axially)
    - LORCounts (ndarray): number of contributing LORs for each sinogram coordinate
    
    Returns:
    - ScaleFactors_A (ndarray): Scale factors for each TOF bin.
    """

    # Load the randoms prompts file to perform scaling right after
    Prompts = np.load(f'{SavePath}/PromptsSinogram.npz')['arr_0'] if not SpanFlag else np.load(f'{SavePath}/PromptsSinogramMashed.npz')['arr_0'] 
    TOFBins = Prompts.shape[-1]
    SinogramSize = Prompts.shape[:-1]

    try:
        Normalization = np.load(f'{SavePath}/NormSinogram.npz')['arr_0']
    except:
        Normalization = np.ones(SinogramSize, np.single)

    try:
        Randoms = np.load(f'{SavePath}/RandomsSinogram.npz')['arr_0']
        Randoms = Randoms*Normalization
    except:
        Randoms = np.zeros(SinogramSize, np.single)

    AttenuationMask = np.load(f'{SavePath}/AttenuationMask.npy')
    AttenuationMask = np.float32(AttenuationMask >= 1)

    ScaleFactors_A = np.zeros((TOFBins), dtype=np.float32)

    # Performing scaling with back projection
    # Start scaling factor computation
    print("Scaling for all bins")
    begin_time = time.time()
    for Bin in range(TOFBins):
        start_time = time.time()

        # print("Start: Scaling for bin {}".format(Bin))
        SinogramsInterpolatedCurrentBin = np.load(f'{SavePath}/SSS_mashed_bin{Bin}.npz')['arr_0']

        binPrompts = Prompts[:,:,:,Bin] / LORCounts
        
        if not SpanFlag:
            NrRings = SinogramIndex.shape[0]
            SinogramOrdering = SinogramIndex[0:NrRings,0:NrRings].T

            ScaleFactor = BackProjectionScatterRandFit(
                binPrompts, SinogramsInterpolatedCurrentBin, Randoms, AttenuationMask, DesiredDimensions, 
                FittingSize, Geometry, SinogramCoordinates, SinogramOrdering, AccelerationFactor)
        else:
            ScaleFactor = BackProjectionScatterRandFitMashing(
                binPrompts, SinogramsInterpolatedCurrentBin, Randoms, AttenuationMask, 
                DesiredDimensions, FittingSize, 
                ExtendedGeometry, SinogramCoordinates, 
                AccelerationFactor, SinogramIndex)


        end_time = time.time()
        bin_time = end_time - start_time
        print("Bin {} took {:.2f} seconds".format(Bin, bin_time))

        # Apply correction to scale factors
        ScaleFactors_A[Bin] = ScaleFactor #/ (2 * TOFRange / TOFBins)

    end_time = time.time()
    elapsed_time = end_time - begin_time
    print("Time for Scaling: {:.2f} seconds".format(elapsed_time))

    # print(ScaleFactors_A)
    # Save the scale factors
    np.save(f'{SavePath}/ScaleFactors.npy', ScaleFactors_A)

    return ScaleFactors_A