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
import os

from numpy import ndarray # for comment the function
from typing import List, Union # for comment the function
from scipy.interpolate import griddata, interp1d, RegularGridInterpolator # for sinogram interpolation
from joblib import Parallel, delayed # for parallel processing

interpolationMethod = 'nearest'

def InterpolateScatters(
    Scatters : ndarray,
    NrRings: int,
    NrRingsUsed: int,
    NrDetectors: int,
    NrDetectorsUsed: int,
    SinogramCoordinates: ndarray,
    SinogramIndex: ndarray,
    SavePath : str,
    SpanFlag: bool,
) -> ndarray:
    
    """
    Scatter Interpolation
    Interpolates the sampled scatters to obtain the scatter estimation for the full sinogram space
    
    Parameters:
    - Scatters (ndarray): Scatter estimation for only the sampled rings and detector combinations
    - NrRings (int): Number of rings in the scanner
    - NrRingsUsed (int): Number of rings to use for the scatter estimation before interpolation
    - NrDetectors (int): Number of detectors in the scanner
    - NrDetectorsUsed (int): Number of detectors to use for the scatter estimation before interpolation
    - SinogramCoordinates (ndarray): Array with the sinogram coordinates for detector combinations
    - SinogramIndex (ndarray): Gives the rings for each sinogram slice
    - SavePath (str): Path where to save the sinograms
    - SpanFlag (bool): flag to indicate if span is used (data reduction at the ring leval - axially)
    
    Returns:  
    - Interpolated_Scatters (ndarray): Interpolated Scatters for all rings and detectors (sinogram coordinates)
    """

    # Define which rings have been used
    Rings = np.floor(np.linspace(0, NrRings-1, NrRingsUsed) + 0.5).astype(int)

    # Define which detectors have been used per ring
    DetectorDifference = NrDetectors / NrDetectorsUsed
    Detectors = np.zeros((NrRings, NrDetectorsUsed), dtype=int)

    for RingIndex1 in range(NrRings):       #loop that defines which detectors are used
        for d in range(NrDetectorsUsed):
            if d == 0:
                Detectors[RingIndex1, d] = d    #make sure we use the first detector
            else:
                Detectors[RingIndex1, d] = int(np.floor(Detectors[RingIndex1, d-1] + DetectorDifference))
    
    # Interpolates to get the scatter estimation for the full sinogram space
    start_time = time.time()
    if SpanFlag:
        print("Start Interpolating with data reduction")
        InterpolatedScatters = InterpolateAllBinsSpan(Scatters, Rings, Detectors, SinogramCoordinates, SavePath, SinogramIndex)
    else:
        print("Start Interpolating")
        InterpolatedScatters = InterpolateAllBins(Scatters, Rings, SinogramIndex, Detectors, SinogramCoordinates, SavePath)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time for Interpolation: {:.2f} seconds".format(elapsed_time))

    with open(f'{SavePath}/Simulation_time.txt', 'w') as file:
        file.write("Time for Interpolation: {:.2f} seconds\n".format(elapsed_time))

    return InterpolatedScatters

def interpolate_radial_angular(
    Sinogram : ndarray,
    Ring1 : int,
    Ring2 : int,
    SinogramCoordinates : ndarray,
    Detectors : ndarray,
    grid_X : ndarray,
    grid_Y : ndarray
) -> ndarray:
    
    """
    Interpolation of Sample Detector
    
    Parameters:
    - Sinogram (ndarray): Sample Sinogram from ring 1 and ring 2.
    - Ring1 (int): Index of ring 1.
    - Ring2 (int): Index of ring 2.
    - SinogramCoordinates (ndarray): Sinogram coordinates.
    - Detectors (ndarray): Sample Detectors.
    - grid_X (ndarray): Full range of radial indexes.
    - grid_Y (ndarray): Full range of Angular indexes.
    
    Returns:
    - F_interpolated (ndarray): Interpolated Sinogram from ring 1 and ring 2.
    """
    
    # Iterate over each sinogram in the current bin
    RadialIndex = SinogramCoordinates[np.ix_(Detectors[Ring1], Detectors[Ring2], [1])][:, :, 0]  # 2D-Array
    AngularIndex = SinogramCoordinates[np.ix_(Detectors[Ring1], Detectors[Ring2], [0])][:, :, 0]  # 2D-Array

    # Prepare points and values for interpolation
    points = np.array([(y, x) for y, x in zip(RadialIndex.flatten(), AngularIndex.flatten())])
    values = Sinogram[RadialIndex.astype(int), AngularIndex.astype(int)].flatten()

    F = griddata(points, values, (grid_X, grid_Y), method=interpolationMethod)
    F_interpolated = fill_with_interp1d(F)
    return F_interpolated

def fill_with_interp1d(
    F: ndarray
) -> ndarray:
    
    """
    Extra-Interpolation outside convex hull of Sample Detectors
    
    Parameters:
    - F (ndarray): Interpolated in convex hull of Sinogram from ring 1 and ring 2.
    
    Returns:
    - F_interpolated (ndarray): Interpolated with row-wise extrapolated outside convex hull of Sinogram from ring 1 and ring 2.
    """

    F_interpolated = F.copy()
    for index in range(F.shape[0]):
        row = F[index, :]
        # Check if there are NaNs in the current row
        if np.any(np.isnan(row)):
            known_indices = np.where(~np.isnan(row))[0]
            if len(known_indices) == 0:
                nan_indices = np.where(np.isnan(row))[0]
                F_interpolated[index, nan_indices] = 0
            else:
                known_values = row[known_indices]
                interp_func = interp1d(known_indices, known_values, kind=interpolationMethod, fill_value='extrapolate')
                nan_indices = np.where(np.isnan(row))[0]
                F_interpolated[index, nan_indices] = interp_func(nan_indices)
            
    # bounding the extrapolation
    F_interpolated[F_interpolated<0] = 0
    return F_interpolated

def interpolate_chunk(
        RadialCoordinates : ndarray,
        AngularCoordinates : ndarray,
        Ring1 : int,
        Ring2 : int,
        interpolation_func : callable, 
        StepSize_Ring1 : int = 1, 
        StepSize_Ring2 : int = 1
        ) -> ndarray:
    
    """
    Processing of interpolation function in chunks for parallel processing
    
    Parameters:
    - RadialCoordinates (ndarray): Array of the radial coordinates of the sinograms.
    - RadialCoordinates (ndarray): Array of the angular coordinates of the sinograms.
    - Ring1 (int) : Index of first ring to interpolate.
    - Ring2 (int) : Index of second ring to interpolate.
    - interpolation_func (function) : Function to interpolate in parallel way
    - StepSize_Ring1 (int, optional) : Number of rings to consider first for this batch job
    - StepSize_Ring2 (int, optional) : Number of rings to consider second for this batch job
    
    Returns:
    - F_interpolated (ndarray): Interpolated with row-wise extrapolated outside convex hull of Sinogram from ring 1 and ring 2.
    """

    xi = np.meshgrid(RadialCoordinates, AngularCoordinates, np.arange(Ring1,Ring1 + StepSize_Ring1), np.arange(Ring2, Ring2 + StepSize_Ring2), indexing='ij')
    xi_flat = np.stack([x.flatten() for x in xi], axis=-1)

    return interpolation_func(xi_flat)

def InterpolateAllBins(
    SinogramsBinned: ndarray,
    Rings: ndarray,
    SinogramIndex: ndarray,
    Detectors: ndarray,
    SinogramCoordinates: ndarray,
    SavePath: str,
    Skip: bool = False,
) -> ndarray:
    
    """
    Interpolation of scatter estimation for the full sinogram space from the sampled LORs
    
    Parameters:
    - SinogramsBinned (ndarray): Sampled sinogram coordinates with the estimated scatters
    - Rings (ndarray): Index of the sampled rings used
    - SinogramIndex (ndarray): Gives the rings for each sinogram slice.
    - Detectors (ndarray): Index of the sampled detectors used
    - SinogramCoordinates (ndarray): Array with the sinogram coordinates for detector combinations
    - SavePath (str): Path where to save the sinograms
    - Skip (bool, optional): Flag to skip interpolating for the coordinates estimated by SSS
    
    Returns:
    - InterpolatedSinograms (ndarray): Scatter sinogram for all Sinogram coodinates
    """

    NrRingsUsed = len(Rings)
    NrRings = SinogramIndex.shape[0]

    InterpolatedSinograms = np.zeros((SinogramsBinned.shape[1], SinogramsBinned.shape[2], NrRings**2), dtype=np.float32)
    grid_X, grid_Y = np.mgrid[0:SinogramsBinned.shape[1], 0:SinogramsBinned.shape[2]]

    #Define the chunks of coordinates for interpolation
    points = (np.arange(SinogramsBinned.shape[1]), np.arange(SinogramsBinned.shape[2]), np.array(Rings), np.array(Rings))
    TOFBins = SinogramsBinned.shape[0]

    # print("Interpolate all bins")
    for Bin in range(TOFBins):
        start_time = time.time()
        SinogramsCurrentBin = SinogramsBinned[Bin, :, :, :].copy()

        # Interpolation to extend Radial and Angular index
        results = Parallel(n_jobs=20)(delayed(interpolate_radial_angular)(SinogramsCurrentBin[:, :, i],
                                                                             Rings[i // NrRingsUsed],
                                                                             Rings[i % NrRingsUsed],
                                                                             SinogramCoordinates,
                                                                             Detectors,
                                                                             grid_X,grid_Y) for i in range(SinogramsCurrentBin.shape[2]))

        # Update the SinogramsCurrentBin with interpolated results
        for i, result in enumerate(results):
            SinogramsCurrentBin[:, :, i] = result

        # Reshape, permute, and interpolate dimensions of SinogramsCurrentBin
        SinogramsCurrentBin = SinogramsCurrentBin.reshape(SinogramsBinned.shape[1], SinogramsBinned.shape[2],
                                                          NrRingsUsed, NrRingsUsed)
        SinogramsCurrentBin = np.transpose(SinogramsCurrentBin, (0, 1, 3, 2))

        # Setup interpolation variables
        interpolator_n = RegularGridInterpolator(points, SinogramsCurrentBin, method=interpolationMethod, bounds_error=False, fill_value=0)
        SinogramsInterpolatedCurrentBin = np.zeros((SinogramsBinned.shape[1], SinogramsBinned.shape[2], NrRings, NrRings), dtype=np.float32)

        # Feed the coordinates in chunks
        if not Skip:

            # Also compute the sinograms computed during SSS
            step_size_Ring1 = 5
            step_size_Ring2 = 5
            results = Parallel(n_jobs=20)(delayed(interpolate_chunk)
                                        (points[0], points[1], i, j, interpolator_n, step_size_Ring1, step_size_Ring2) for i in range(0,NrRings, step_size_Ring1) for j in range(0, NrRings, step_size_Ring2)
                                        )
            
            current = 0
            for i in range(0, NrRings, step_size_Ring1):
                for j in range(0, NrRings, step_size_Ring2):
                    SinogramsInterpolatedCurrentBin[:,:,i:i + step_size_Ring1,j:j+step_size_Ring2] = results[current].reshape(SinogramsBinned.shape[1], SinogramsBinned.shape[2], step_size_Ring1, step_size_Ring2)
                    current += 1
        else:
            # Skip sinograms computed during SSS, using parallelisation
            step_size_Ring2 = 8
            for k in range(0, NrRingsUsed-1):

                # Unsampled rings first
                currentRange = Rings[k+1] - (Rings[k]+1)
                results = Parallel(n_jobs=20)(delayed(interpolate_chunk)
                                            (points[0], points[1], Rings[k]+1, j, interpolator_n, currentRange, step_size_Ring2)
                                            for j in range(0, NrRings, step_size_Ring2)
                                            )
                
                count = 0
                for i in range(0, NrRings, step_size_Ring2):
                    SinogramsInterpolatedCurrentBin[:,:,Rings[k]+1:Rings[k+1], i:i+step_size_Ring2] = results[count].reshape(SinogramsBinned.shape[1], SinogramsBinned.shape[2], currentRange, step_size_Ring2)
                    count += 1

                # Sampled rings
                results = Parallel(n_jobs=6)(delayed(interpolate_chunk)
                                            (points[0], points[1], ring, Rings[k]+1, interpolator_n, 1, currentRange)
                                            for ring in Rings
                                            )
                
                for i in range(NrRingsUsed):
                    SinogramsInterpolatedCurrentBin[:,:,Rings[i], Rings[k]+1:Rings[k+1]] = results[i].reshape(SinogramsBinned.shape[1], SinogramsBinned.shape[2], currentRange)
                
                # Put in the SSS data
                for i in range(NrRingsUsed):
                    for j in range(NrRingsUsed):
                        SinogramsInterpolatedCurrentBin[:,:,Rings[i],Rings[j]] = SinogramsCurrentBin[:,:,i,j]
                

        SinogramsInterpolatedCurrentBin = SinogramsInterpolatedCurrentBin.reshape(SinogramsBinned.shape[1],
                                                                                  SinogramsBinned.shape[2],
                                                                                  NrRings ** 2)
        SinogramOrder = np.argsort(SinogramIndex[:NrRings, :NrRings].T.flatten())
        SinogramsInterpolatedCurrentBin = SinogramsInterpolatedCurrentBin[:,:,SinogramOrder]

        end_time = time.time()
        bin_time = end_time - start_time
        print("Bin {} took {:.2f} seconds".format(Bin, bin_time))

        np.savez_compressed(f'{SavePath}/SSS_mashed_bin{Bin}', SinogramsInterpolatedCurrentBin)
        InterpolatedSinograms += SinogramsInterpolatedCurrentBin

    return InterpolatedSinograms

def InterpolateAllBinsSpan(
    SinogramsBinned: ndarray,
    Rings: ndarray,
    Detectors: ndarray,
    SinogramCoordinates: ndarray,
    SavePath: str,
    MashedSinogramIndices: ndarray,
) -> ndarray:
    
    """
    Interpolation of scatter estimation for the full sinogram space from the sampled LORs
    
    Parameters:
    - SinogramsBinned (ndarray): Sample Sinogram.
    - Rings (ndarray or list): All ring indexes.
    - Detectors (ndarray): Sample Detectors.
    - SinogramCoordinates (ndarray): Sinogram coordinates.
    - SavePath (str): Path where to save the sinograms.
    - MashedSinogramIndices (ndarray): Gives the contributing rings for each sinogram index.
    
    Returns:
    - InterpolatedSinograms (ndarray): Scatter sinogram for all Sinogram coodinates.
    """

    NrRingsUsed = len(Rings)
    NrSinograms = MashedSinogramIndices.shape[0]

    InterpolatedSinograms = np.zeros((SinogramsBinned.shape[1], SinogramsBinned.shape[2], NrSinograms), dtype=np.float32)
    grid_X, grid_Y = np.mgrid[0:SinogramsBinned.shape[1], 0:SinogramsBinned.shape[2]]

    #Define the chunks of coordinates for interpolation
    points = (np.arange(SinogramsBinned.shape[1]), np.arange(SinogramsBinned.shape[2]), np.array(Rings), np.array(Rings))

    print("Interpolate all bins")
    for Bin in range(SinogramsBinned.shape[0]):
        start_time = time.time()

        SinogramsCurrentBin = SinogramsBinned[Bin, :, :, :].copy()

        # Interpolation to extend Radial and Angular index
        results = Parallel(n_jobs=-1)(delayed(interpolate_radial_angular)(SinogramsCurrentBin[:, :, i],
                                                                            Rings[i // NrRingsUsed],
                                                                            Rings[i % NrRingsUsed],
                                                                            SinogramCoordinates,
                                                                            Detectors,
                                                                            grid_X,grid_Y) for i in range(SinogramsCurrentBin.shape[2]))

        # Update the SinogramsCurrentBin with interpolated results
        for i, result in enumerate(results):
            SinogramsCurrentBin[:, :, i] = result

        # Reshape, permute, and interpolate dimensions of SinogramsCurrentBin
        SinogramsCurrentBin = SinogramsCurrentBin.reshape(SinogramsBinned.shape[1], SinogramsBinned.shape[2],
                                                        NrRingsUsed, NrRingsUsed)
        SinogramsCurrentBin = np.transpose(SinogramsCurrentBin, (0, 1, 3, 2))

        # Setup interpolation variables
        interpolator_n = RegularGridInterpolator(points, SinogramsCurrentBin, method=interpolationMethod, bounds_error=False, fill_value=0)
        SinogramsInterpolatedCurrentBin = np.zeros((SinogramsBinned.shape[1], SinogramsBinned.shape[2], MashedSinogramIndices.shape[0]), dtype=np.float32)
        
        # Feed the coordinates in chunks
        results = Parallel(n_jobs=20)(delayed
                                        (interpolate_chunk)
                                        (points[0], points[1], MashedSinogramIndices[i, 0], MashedSinogramIndices[i,1], interpolator_n)
                                        for i in range(MashedSinogramIndices.shape[0])
                                        )
        
        for i, result in enumerate(results):
            SinogramsInterpolatedCurrentBin[:, :, i] = result.reshape((SinogramsBinned.shape[1], SinogramsBinned.shape[2]))
        
        end_time = time.time()
        bin_time = end_time - start_time
        print("Bin {} took {:.2f} seconds".format(Bin, bin_time))

        # Save the bin
        np.savez_compressed(f'{SavePath}/SSS_mashed_bin{Bin}', SinogramsInterpolatedCurrentBin)
        InterpolatedSinograms += SinogramsInterpolatedCurrentBin[:, :, :]

    return InterpolatedSinograms


