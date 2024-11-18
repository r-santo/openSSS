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

from scipy.special import erf
from numpy import ndarray # for comment the function
from typing import List, Union # for comment the function
from RayTracing3DTOF import RayTracing3DTOF

from scipy.interpolate import griddata, interpn, interp1d, RegularGridInterpolator # for sinogram interpolation
from joblib import Parallel, delayed # for parallel processing
from numba import jit # for fast computation

ElectronMass = 9.10938356E-31       #mass electron (kg)
Lightspeed = 299792458.0            #speed of light (m/s)

interpolationMethod = 'nearest'

# Main function, the SSS algorithm with TOF
def SingleScatterSimulationTOF(
    ActivityMap: ndarray,
    AttenuationMap: ndarray,
    ImageSize: ndarray, 
    Geometry: ndarray, 
    SinogramCoordinates: ndarray, 
    SinogramIndex: ndarray,
    NormalVectors: ndarray, 
    DetectorSize: ndarray,
    AttenuationTable: ndarray, 
    EnergyResolution: float,
    EnergyThreshold: float,
    NrRingsUsed: int,
    NrDetectorsUsed: int,
    SampleStep: ndarray,
    TOFResolution: float,
    TOFRange: float,
    NrBins: int,
    SavePath: str):
    """
    3D Single Scatter Simulation (SSS)
    Performs TOF scatter estimation based on the SSS algorithm by Watson
    
    Parameters:
        ActivityMap: Activity maps estimation
        AttenuationMap: Attenuation map
        ImageSize: Attenuation and activity image size [-x -y -z x y z] (mm)
        Geometry: 3D array with the (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
        SinogramCoordinates: Array with the sinogram coordinates for detector combinations
        SinogramIndex: Array with the order of the sinograms for ring combinations
        NormalVectors: Vectors normal to the surface area of detectors
        DetectorSize: Size of the surface of the detectors [x, y] (mm)
        AttenuationTable: Linear Attenuation coefficient respect to the energy
        EnergyResolution: Energy resolution (%)
        EnergyThreshold: Energy threshold (keV)
        NrRingsUsed: Number of rins to use for the scatter estimation before interpolation
        NrDetectorsUsed: Number of detectors to use for the scatter estimation before interpolation
        SampleStep: Step size for the sample of scatter points [x, y, z] in voxels
        TOFResolution: Time resolution (ps)
        TOFRange: Range of TOF measured (ps)
        NrBins: Number of TOF bins
        SavePath: Path where to save the sinograms
    
    Returns:  
        Scatters: Estimated scatters from sample ring and detector
        Interpolated_Scatters: Interpolated Scatters
    """

    xDim, yDim, zDim = AttenuationMap.shape # dimensions of the contour image
    NrDetectors = Geometry.shape[1] # number of detectors used
    NrRings = Geometry.shape[0] # number of rings

    # the crop for the attenuation and activity map expands half the detector height for each side of the FOV
    zVoxelSize = (ImageSize[5] - ImageSize[2]) / zDim # real distance between middle two slices in contour_image

    EnergyReferenceJoule = 511.0E3 * 1.6E-19  # Photon energy (J)
    SmallNumber = 10 ** (-15)  # used to prevent dividing by zero
    EnergyReference = 511  # energy used for generating attenuation map (keV)
    GridSize = ActivityMap.shape  # dimensions for the ray tracing grid
    GridBounds = ImageSize  # real dimensions of the ray tracing grid

    NormalVectors = np.ascontiguousarray(NormalVectors)

    # TOF characteristics
    BinWidth = 2 * TOFRange / NrBins  # width of the bin (ps)
    EnergyIndex = np.where(AttenuationTable[:, 0] == EnergyReference)[0]

    # assign each voxel in the attenuation map an index to know what tissue it is, and how to convert the attenuation coefficient for ray 2
    AttenuationTissue = np.zeros((xDim, yDim, zDim))
    for zIndex in range(0, zDim):
        for xIndex in range(0, xDim):
            for yIndex in range(0, yDim):
                if AttenuationMap[xIndex, yIndex, zIndex] > 0:
                    j = 0
                    CurrentMinimum = 10 ** 15
                    for i in range(1, AttenuationTable.shape[1]):
                        CurrentDifference = abs(AttenuationTable[EnergyIndex, i] - AttenuationMap[xIndex, yIndex, zIndex])
                        if CurrentDifference < CurrentMinimum:
                            CurrentMinimum = CurrentDifference
                            j = i
                        AttenuationTissue[xIndex, yIndex, zIndex] = j

    # Compress the attenuation_table, so that it just includes the energy for 511 keV and the ratios for water of all other energies
    AttenuationRatios = np.zeros((AttenuationTable.shape[0], 2))
    AttenuationRatios[:, 0] = AttenuationTable[:, 0]  # Copying the energy column as it is
    # Calculating ratios for water for energies other than 511 keV
    AttenuationRatios[:, 1] = AttenuationTable[:, 1] / AttenuationTable[EnergyIndex, 1]

    # Find unique values in AttenuationTissue and remove 0
    Tissue = np.unique(AttenuationTissue)
    Tissue = Tissue[Tissue != 0].astype(int)

    # Replace occurrences of unique values in AttenuationTissue with their corresponding index + 1
    for i in range(len(Tissue)):
        AttenuationTissue[AttenuationTissue == Tissue[i]] = i + 1

    # Update AttenuationTable
    AttenuationTable = AttenuationTable[EnergyIndex, [0] + list(Tissue)]

    # Energy efficiency of detectors (ϵA/ϵB)
    # Extract energy values from AttenuationRatios
    EfficiencyTable = np.zeros_like(AttenuationRatios)
    EfficiencyTable[:, 0] = AttenuationRatios[:, 0]

    # Calculate detector efficiency
    EfficiencyTable[:, 1] = 0.5 * (1 - erf(
        (EnergyThreshold - EfficiencyTable[:, 0]) / (EnergyReference * EnergyResolution / 2 / np.sqrt(np.log(2)))))

    # Time of flight efficiency (εt) at different Δs
    # Define constants
    LightSpeed = 299792458.0 * 1e-10  # speed of light (cm/ps)
    TimeRange = 4000  # how wide the kernel is (ps)
    # Calculate the time range array
    time_array = np.arange(-TimeRange, TimeRange + 1)  # + 1 to include TimeRange itself
    # Call TOFEfficiencyTable function
    # TOFTable = [εt(Δs), Δs]
    TOFTable = TOFEfficiencyTable(time_array, BinWidth, NrBins, TOFResolution)

    # Define which rings will be used, #Python Index start 0
    Rings = np.floor(np.linspace(0, NrRings-1, NrRingsUsed) + 0.5).astype(int)  # use np.floor(a+0.5) instead of round

    # Calculate the number of sinograms that will be obtained (oblique + non-oblique)
    NrSinograms = NrRingsUsed ** 2

    # Define which detectors will be used per ring
    DetectorDifference = NrDetectors / NrDetectorsUsed
    Detectors = np.zeros((NrRings, NrDetectorsUsed), dtype=int)

    for RingIndex1 in range(NrRings):       #loop that defines which detectors are used
        for d in range(NrDetectorsUsed):
            if d == 0:
                Detectors[RingIndex1, d] = d    #make sure we use the first detector
            else:
                Detectors[RingIndex1, d] = int(np.floor(Detectors[RingIndex1, d-1] + DetectorDifference))


    # Initialize structure to save sinograms
    Scatters = np.zeros((NrBins, NrDetectors + 1, NrDetectors // 2, NrSinograms), dtype=np.float32)

    # -----------------------------------------------------
    # MAIN PART SSS
    # -----------------------------------------------------
    Width = - ImageSize[2]
    Radius = - ImageSize[0]
    # The z-coordinate in the cropped maps starts half width before the FOV
    zStart = zVoxelSize / 2 - Width  # z-coordinated of first slice

    # loop over all possible scatter points, using parallel loops
    zSamplePoints = range(0, zDim, SampleStep[2])

    start_time_scatters = time.time()  # to time the algorithm

    # prepare flat array to be called in bobySSS
    ActivityMap_flat = ActivityMap.flatten(order="F")
    AttenuationMap_flat = AttenuationMap.flatten(order="F")

    # Call the function for the main part of the SSS algorithm with Parallelto increase speed
    # The result will be store as list
    print("Start Scattering Simulation")
    Scatters = Parallel(n_jobs=-1)(
        delayed(bodySSS)(zIndex, zStart, zVoxelSize, NrBins, NrDetectors, NrSinograms, SampleStep,
                         Radius, xDim, yDim, zDim, AttenuationTissue, NrRings, NrRingsUsed, NrDetectorsUsed,
                         Rings, Geometry, Detectors, GridSize, GridBounds, ActivityMap_flat, AttenuationMap_flat, NormalVectors,
                         SinogramCoordinates, EnergyReference, EfficiencyTable, AttenuationRatios, LightSpeed, TOFTable,
                         TimeRange, DetectorSize, EnergyReferenceJoule, SmallNumber) for zIndex in zSamplePoints)

    # Sum contribution over sample points
    Scatters = np.sum(np.stack(Scatters), axis=0)
    end_time_scatters = time.time()
    
    # Post Processing
    print("Start Interpolating")
    InterpolatedScatters = InterpolateAllBins(Scatters, Rings, SinogramIndex, Detectors, SinogramCoordinates, SavePath)
    end_time_scatter_Interpolation = time.time()
    elapsed_time1 = end_time_scatters - start_time_scatters
    elapsed_time2 = end_time_scatter_Interpolation - start_time_scatters
    print("Time for SSS:", elapsed_time1, "seconds")
    print("Time for SSS+Interpolation:", elapsed_time2, "seconds")

    with open(f'{SavePath}/Simulation_time.txt', 'w') as file:
        file.write(f"Time for SSS: {elapsed_time1} seconds\n")
        file.write(f"Time for SSS+Interpolation: {elapsed_time2} seconds\n")
    
    return InterpolatedScatters


##################################################################################################
#################### SUPPORT FUNCTIONS FOR Single Scatter Simulation #############################
##################################################################################################

@jit(nopython=True)
def CalcAngleScatter(
    ScatterVector1: float,
    ScatterVector2: float,
    LOR: float
) -> float:

    """
    Calculate the Scattering angle from law of cosines.
    
    Parameters:
    - ScatterVector1 (float): Distance of Detector 1 to Scattering point.
    - ScatterVector2 (float): Distance of Detector 2 to Scattering point.
    - LOR (flaot): Distance of Detector 1 and Detector 2
    
    Returns:
    - ScatterAngle: Scattering angle in degree.
    """
    #Scatter angle
    ScatterAngle = 180 - np.degrees(np.arccos(((ScatterVector1) + (ScatterVector2) - (LOR)) / (2 * np.sqrt(ScatterVector1 * ScatterVector2))))
    return ScatterAngle

# Klein-Nishina probability
@jit(nopython=True)
def KleinNishina(
    Energy: float,
    Angle: float
) -> float:
    
    """
    Calculate the Probabity of scatter from Klein-Nishina formula
    
    Parameters:
    - Energy (float): Refference energy in Jules.
    - Angle (float): Angle of Scattered photon in degree.
    
    Returns:
    - probability: Probabity of scattering event.
    """

    Gamma = Energy / (ElectronMass * Lightspeed ** 2)
    Ratio = 1.0 / (1.0 + Gamma * ( 1.0 - np.cos(np.deg2rad(Angle)) ))
    probability = Ratio ** 2 * (Ratio + 1/Ratio - np.sin(np.deg2rad(Angle)) ** 2)
    return probability

@jit(nopython=True)
def TOFEfficiencyTable(
    Offset: ndarray,
    BinWidth: float,
    NrBins: int,
    Resolution: float
) -> float:

    """
    Pre-calculation of TOF-Detector efficiency
    
    Parameters:
    - Offset (ndarray): Time-offset (Δs) in ps.
    - BinWidth (float): width of the bin in ps.
    - NrBins (int): Number of TOF bins.
    - Resolution (float): Energy resolution of the system
    
    Returns:
    - probability (float): Probabity of scattering event.
    """

    # Calculate shifts for each bin
    shifts = ((np.arange(1, NrBins + 1) - (NrBins + 1) / 2) * BinWidth).reshape(-1, 1)

    # Calculate the probability based on the given formula
    probability = Offset.reshape(1, -1) - shifts  # .reshape(-1, 1)
    probability = np.exp(-(probability ** 2) / (Resolution ** 2 / 4 / np.log(2)))

    # Normalize the probability values
    probability = probability / np.sum(probability, axis=0)
    return probability

##################################################################################################
#################### MAIN FUNCTIONS OF THE SINGLE SCATTER SIMULATION #############################
##################################################################################################
@jit(nopython=True)
def bodySSS(zIndex, zStart, zVoxelSize, NrBins, NrDetectors, NrSinograms, SampleStep,
            Radius, xDim, yDim, zDim, AttenuationTissue, NrRings, NrRingsUsed, NrDetectorsUsed,
            Rings, Geometry, Detectors, GridSize, GridBounds, ActivityMap_flat, AttenuationMap_flat, NormalVectors,
            SinogramCoordinates, EnergyReference, EfficiencyTable, AttenuationRatios, LightSpeed, TOFTable,
            TimeRange, DetectorSize, EnergyReferenceJoule, SmallNumber):
    zScatterPoint = zStart + zIndex * zVoxelSize  # z-coordinate scatter point
    # Structure to save the sinograms of each slice
    ScatterSlice = np.zeros((NrBins, NrDetectors + 1, NrDetectors // 2, NrSinograms), dtype=np.float32)
    # Structure to save how many times a LOR is used
    ScatterCounts = np.zeros((NrBins, NrDetectors + 1, NrDetectors // 2, NrSinograms), dtype=np.float32)

    for yIndex in range(0, yDim, SampleStep[1]):
        if np.sum(AttenuationTissue[:, yIndex, zIndex]) != 0:  # Check if any of the indices is a scatter point
            yScatterPoint = ((yIndex+1) * 2 * Radius / yDim) - Radius  # y-coordinate scatter point

            for xIndex in range(0, xDim, SampleStep[0]):
                # Check if it is a scatterpoint
                if AttenuationTissue[xIndex, yIndex, zIndex] > 0:
                    xScatterPoint = ((xIndex+1) * 2 * Radius / xDim) - Radius  # x-coordinate scatter point
                    LinePaths = np.zeros((NrRingsUsed, NrDetectorsUsed, 2))
                    LineDistributions = [[[np.zeros(0), np.zeros(0)] for _ in range(NrDetectorsUsed)] for _ in range(NrRingsUsed)]
                    Angles = np.zeros((NrRingsUsed, NrDetectorsUsed))

                    # loop over all "half" LOR:s (scatterpoint to detectors)
                    for RingIndex1 in range(NrRingsUsed):
                        Ring1 = Rings[RingIndex1]
                        zDetector1 = Geometry[Ring1, 0, 2]

                        for DetectorIndex1 in range(NrDetectorsUsed):
                            Detector1 = Detectors[Ring1, DetectorIndex1]  # Assuming Detectors is a 2D array
                            xDetector1 = Geometry[Ring1, Detector1, 0]  # Assuming Geometry is a 3D array
                            yDetector1 = Geometry[Ring1, Detector1, 1]  # Assuming Geometry is a 3D array

                            LineCoordinates = [xScatterPoint, yScatterPoint, zScatterPoint,
                                               xDetector1, yDetector1, zDetector1]
                            # Ray Tracing function, Indexes returned match MATLAB indexes
                            # i.e. Indexes start from one
                            Lenghts, Indexes, Rays = RayTracing3DTOF(GridSize, GridBounds, LineCoordinates)
                            Indexes = Indexes - 1    # fix for Python indexing, Indexes is a NumPy array

                            # order="F" makes the operation column major, to comply with MATLAB
                            ActivityRay = ActivityMap_flat[Indexes] * Lenghts
                            ActivityIntegral = np.sum(ActivityRay)

                            AttenuationsIntegral = np.exp(-(np.sum(AttenuationMap_flat[Indexes] * Lenghts)))
                            LinePaths[RingIndex1, DetectorIndex1] = [AttenuationsIntegral, ActivityIntegral]
                            LineDistributions[RingIndex1][DetectorIndex1] = [ActivityRay, Rays]

                            ScatterVector = np.ascontiguousarray(np.array([xDetector1 - xScatterPoint,
                                             yDetector1 - yScatterPoint,
                                             zDetector1 - zScatterPoint]))

                            # Angles for all detectors, used later to calculate effective area of detectors
                            Angles[RingIndex1, DetectorIndex1] = np.abs(np.degrees(np.arccos(np.dot(ScatterVector, NormalVectors[Ring1, Detector1])
                                                                                                 / (np.linalg.norm(NormalVectors[Ring1, Detector1]) * np.linalg.norm(ScatterVector)))))
                    # Mix paths to generate LOR:s
                    # Create the first path (path1) (from the scatterpoint to the first detector
                    for RingIndex1 in range(0, NrRingsUsed):
                        Ring1 = Rings[RingIndex1]
                        zDetector1 = Geometry[Ring1, 0, 2]

                        for DetectorIndex1 in range(NrDetectorsUsed):
                            Detector1 = Detectors[Ring1, DetectorIndex1]
                            xDetector1 = Geometry[Ring1, Detector1, 0]
                            yDetector1 = Geometry[Ring1, Detector1, 1]

                            ScatterVector1 = np.sqrt((xDetector1 - xScatterPoint) ** 2
                                                     + (yDetector1 - yScatterPoint) ** 2
                                                     + (zDetector1 - zScatterPoint) ** 2)

                            # ActivityIntegral (emission line integral) and
                            # AttenuationIntegral of unscattered photon to Detector1
                            AttenuationPath1 = LinePaths[RingIndex1, DetectorIndex1, 0]
                            ActivityPath1 = LinePaths[RingIndex1, DetectorIndex1, 1]

                            if AttenuationPath1 == 0:
                                continue  # Skip to the next loop

                            # Activity and sample lenght distributions of unscattered photon in detector 1
                            ActivityRay1 = LineDistributions[RingIndex1][DetectorIndex1][0].T
                            Rays1 = LineDistributions[RingIndex1][DetectorIndex1][1].T

                            # Create the second path (path2) from the scatterpoint
                            for RingIndex2 in range(NrRingsUsed):
                                Ring2 = Rings[RingIndex2]
                                zDetector2 = Geometry[Ring2, 0, 2]

                                # Allowed ring difference
                                if np.abs((RingIndex2) - (RingIndex1)) <= NrRingsUsed:

                                    for DetectorIndex2 in range(NrDetectorsUsed):
                                        Detector2 = Detectors[Ring2, DetectorIndex2]
                                        # Check if the current LOR is possible
                                        if (Detector1 == Detector2 and RingIndex2 == RingIndex1) or Detector1 > Detector2:
                                            continue  # Skip, LOR not possible
                                        else:
                                            xDetector2 = Geometry[Ring2, Detector2, 0]
                                            yDetector2 = Geometry[Ring2, Detector2, 1]
                                            ScatterVector2 = np.sqrt((xDetector2 - xScatterPoint) ** 2 + (yDetector2 - yScatterPoint) ** 2 + (zDetector2 - zScatterPoint) ** 2)
                                            LOR = (((xDetector1-xDetector2)**2)+((yDetector1-yDetector2)**2)+((zDetector1-zDetector2)**2))

                                            # ActivityIntegral and AttenuationIntegral of unscattered
                                            # photon in Detector2
                                            AttenuationPath2 = LinePaths[RingIndex2, DetectorIndex2, 0]
                                            ActivityPath2 = LinePaths[RingIndex2, DetectorIndex2, 1]

                                            # Count the amount of Counts expected (Later used for asymetric correction)
                                            if AttenuationPath2 == 0 or (ActivityPath1 == 0 and ActivityPath2 == 0):
                                                continue  # No Activity on any path = no counts
                                            elif ActivityPath1 != 0 and ActivityPath2 != 0:
                                                Counts = 2  # Activity on both paths, 2 counts
                                            else:
                                                Counts = 1  # Activity on only one of the paths, 1 count

                                            # Look up where the probability should be added in the scatter sinograms
                                            AngularIndex = SinogramCoordinates[Detector1, Detector2, 0]
                                            RadialIndex = SinogramCoordinates[Detector1, Detector2, 1]

                                            # Calculate the scattering angle
                                            ScatterAngle = CalcAngleScatter(ScatterVector1**2, ScatterVector2**2, LOR)
                                            # Calculate the energy of scattered photon, dependent on the angle of scatter
                                            #Use floor(x + 0.5) to mimic mathematical round(), i.e. rounding up from 0.5
                                            EnergyScatter = np.floor(EnergyReference / (1 + (EnergyReference / 511.0)
                                                                                        * (1 - (np.cos(np.deg2rad(ScatterAngle))))) + 0.5)
                                            # Eneryindex needs to have 1 subtracted
                                            EnergyScatterIndex = int(EnergyScatter * 2) - 1  # np.where(attenuation_ratios[:,1] == energy_ray2)


                                            # Retrieve the EnnergyEfficiency for a specific path ASB for detector1 and 2
                                            EnnergyEfficiency = (EfficiencyTable[EnergyScatterIndex, 1]
                                                                 * EfficiencyTable[EnergyReference * 2 - 1, 1])
                                                                                    # subtract 1 for Python indexing
                                            if EnnergyEfficiency == 0:
                                                continue

                                            AttenuationScale = AttenuationRatios[EnergyScatterIndex, 1]
                                            # Calculate the scaled AttenuationIntegral for both paths
                                            AttenuationScaled1 = AttenuationPath1 ** AttenuationScale
                                            AttenuationScaled2 = AttenuationPath2 ** AttenuationScale

                                            # TOF detector efficiency
                                            # Defined so that time difference (time1 - time2) is positive closer to detector 2
                                            # Based on the CASTOR interpretation
                                            if ActivityPath1 != 0:
                                                SpatialOffset = np.floor(2 * ((ScatterVector1 - ScatterVector2) / 2 - Rays1) / LightSpeed + 0.5) + TimeRange
                                                ActivityBinned1 = TOFTable[:, SpatialOffset.astype(np.int16)]
                                                ActivityBinned1 = np.dot(ActivityRay1, ActivityBinned1.T)
                                            else:
                                                ActivityBinned1 = np.zeros(1)
                                            if ActivityPath2 != 0:
                                                SpatialOffset = np.floor(2 * ((ScatterVector1 - ScatterVector2) / 2 + LineDistributions[RingIndex2][DetectorIndex2][1].T) / LightSpeed + 0.5) + TimeRange
                                                ActivityBinned2 = TOFTable[:, SpatialOffset.astype(np.int16)]
                                                ActivityBinned2 = np.dot(LineDistributions[RingIndex2][DetectorIndex2][0].T,ActivityBinned2.T)
                                            else:
                                                ActivityBinned2 = np.zeros(1)

                                            # geometrical correction value (first component in formula Watson, without cross sections)
                                            GeometricalEfficiency = (((DetectorSize[0] * DetectorSize[1] * 1e-2) ** 2
                                                                      * np.abs(np.cos(np.deg2rad(Angles[RingIndex1, DetectorIndex1]))))
                                                                     * (np.abs(np.cos(np.deg2rad(Angles[RingIndex2, DetectorIndex2])))
                                                                        / (4 * np.pi * (ScatterVector1 ** 2) * (ScatterVector2 ** 2))))

                                            # Calculate probability
                                            Probability = (GeometricalEfficiency * EnnergyEfficiency * KleinNishina(EnergyReferenceJoule, ScatterAngle)
                                                           * (AttenuationPath1 * AttenuationScaled2 * ActivityBinned1 + AttenuationPath2 * AttenuationScaled1 * ActivityBinned2))
                                            
                                            # Probability becomes a vector containing the probabilities of scatter
                                            # contributions landing in a certain TOF bin along the associated LOR

                                            # Clarification: Both sides of the path is calculated simultaneously
                                            # Depending on if the emission event occures on the AS or BS side
                                            # of the scatter point S, or both.

                                            # add probability and count to the corresponding sinogram

                                            ScatterSlice[:, RadialIndex, AngularIndex, (RingIndex2 + RingIndex1 * NrRingsUsed)] += Probability.T  # add .T here

                                            ScatterCounts[:, RadialIndex, AngularIndex, (RingIndex2 + RingIndex1 * NrRingsUsed)] += Counts

    # add small number to prevent devided by 0
    dim1, dim2, dim3, dim4 = ScatterCounts.shape
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                for l in range(dim4):
                    if ScatterCounts[i, j, k, l] == 0:
                        ScatterCounts[i, j, k, l] = SmallNumber
    ScatterSlice = ScatterSlice / ScatterCounts
    return ScatterSlice

##################################################################################################
#################### SUPPORT FUNCTIONS FOR INTERPOLATION #########################################
##################################################################################################

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
            known_values = row[known_indices]
            interp_func = interp1d(known_indices, known_values, kind=interpolationMethod, fill_value='extrapolate')
            nan_indices = np.where(np.isnan(row))[0]
            F_interpolated[index, nan_indices] = interp_func(nan_indices)
    # bounding the extrapolation
    F_interpolated[F_interpolated<0] = 0
    return F_interpolated

def interpolate_chunk(d1, d2, i, j, interpolation_func, step_size_i = 1, step_size_j = 1):

    xi = np.meshgrid(d1, d2, np.arange(i,i + step_size_i), np.arange(j, j + step_size_j), indexing='ij')
    xi_flat = np.stack([x.flatten() for x in xi], axis=-1)

    return interpolation_func(xi_flat)

def InterpolateAllBins(
    SinogramsBinned: ndarray,
    Rings: Union[ndarray, List[int]],
    SinogramIndex: ndarray,
    Detectors: ndarray,
    SinogramCoordinates: ndarray,
    SavePath: str,
    Skip: bool = False
) -> ndarray:
    
    """
    Interpolation of Sample Detector
    
    Parameters:
    - SinogramsBinned (ndarray): Sample Sinogram.
    - Rings (ndarray or list): All ring indexes.
    - SinogramIndex (ndarray): Array with the order of the sinograms for ring combinations.
    - Detectors (ndarray): Sample Detectors.
    - SinogramCoordinates (ndarray): Sinogram coordinates.
    - Skip (bool): Boolean that defines whether we can skip generating the sinograms generated by SSS during interpolation
    
    Returns:
    - InterpolatedSinograms (ndarray): Full Interpolated Sinogram.
    """

    NrRingsUsed = len(Rings)
    NrRings = SinogramIndex.shape[0]

    InterpolatedSinograms = np.zeros((SinogramsBinned.shape[1], SinogramsBinned.shape[2], NrRings**2), dtype=np.float32)
    grid_X, grid_Y = np.mgrid[0:SinogramsBinned.shape[1], 0:SinogramsBinned.shape[2]]

    #Define the chunks of coordinates for interpolation
    points = (np.arange(SinogramsBinned.shape[1]), np.arange(SinogramsBinned.shape[2]), np.array(Rings), np.array(Rings))

    print("Interpolate all bins")
    for Bin in range(SinogramsBinned.shape[0]):

        SinogramsCurrentBin = SinogramsBinned[Bin, :, :, :].copy()

        # Interpolation to extend Radial and Angular index
        print("Start: Interpolate_radial_angular")

        results = Parallel(n_jobs=-1)(delayed(interpolate_radial_angular)(SinogramsCurrentBin[:, :, i],
                                                                             Rings[i // NrRingsUsed],
                                                                             Rings[i % NrRingsUsed],
                                                                             SinogramCoordinates,
                                                                             Detectors,
                                                                             grid_X,grid_Y) for i in range(SinogramsCurrentBin.shape[2]))

        # Update the SinogramsCurrentBin with interpolated results
        for i, result in enumerate(results):
            SinogramsCurrentBin[:, :, i] = result

        print("Done: Interpolate_radial_angular")

        # Reshape, permute, and interpolate dimensions of SinogramsCurrentBin
        SinogramsCurrentBin = SinogramsCurrentBin.reshape(SinogramsBinned.shape[1], SinogramsBinned.shape[2],
                                                          NrRingsUsed, NrRingsUsed)
        SinogramsCurrentBin = np.transpose(SinogramsCurrentBin, (0, 1, 3, 2))

        # Setup interpolation variables
        my_interpn = RegularGridInterpolator(points, SinogramsCurrentBin, method=interpolationMethod, bounds_error=False, fill_value=0)
        SinogramsInterpolatedCurrentBin = np.zeros((SinogramsBinned.shape[1], SinogramsBinned.shape[2], NrRings, NrRings), dtype=np.float32)

        # Feed the coordinates in chunks
        print("Start: Interpolation on all sinograms")

        if not Skip:
            # Also compute the sinograms computed during SSS
            step_size_Ring1 = 2
            step_size_Ring2 = 8
            results = Parallel(n_jobs=20)(delayed(interpolate_chunk)
                                        (points[0], points[1], i, j, my_interpn, step_size_Ring1, step_size_Ring2) for i in range(0,NrRings, step_size_Ring1) for j in range(0, NrRings, step_size_Ring2)
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
                                            (points[0], points[1], Rings[k]+1, j, currentRange, step_size_Ring2, my_interpn)
                                            for j in range(0, NrRings, step_size_Ring2)
                                            )
                
                count = 0
                for i in range(0, NrRings, step_size_Ring2):
                    SinogramsInterpolatedCurrentBin[:,:,Rings[k]+1:Rings[k+1], i:i+step_size_Ring2] = results[count].reshape(SinogramsBinned.shape[1], SinogramsBinned.shape[2], currentRange, step_size_Ring2)
                    count += 1

                # Sampled rings
                results = Parallel(n_jobs=6)(delayed(interpolate_chunk)
                                            (points[0], points[1], ring, Rings[k]+1, my_interpn, 1, currentRange)
                                            for ring in Rings
                                            )
                
                for i in range(NrRingsUsed):
                    SinogramsInterpolatedCurrentBin[:,:,Rings[i], Rings[k]+1:Rings[k+1]] = results[i].reshape(SinogramsBinned.shape[1], SinogramsBinned.shape[2], currentRange)   

        print("Done: Interpolation on all sinograms")

        SinogramsInterpolatedCurrentBin = SinogramsInterpolatedCurrentBin.reshape(SinogramsBinned.shape[1],
                                                                                  SinogramsBinned.shape[2],
                                                                                  NrRings ** 2)
        SinogramOrder = np.argsort(SinogramIndex[:NrRings, :NrRings].T.flatten())

        SinogramsInterpolatedCurrentBin[:, :, SinogramOrder].tofile(f'{SavePath}/SSS_bin{Bin}.bin')
        InterpolatedSinograms += SinogramsInterpolatedCurrentBin[:, :, SinogramOrder]

        print("Bin", Bin, "done!")

    return InterpolatedSinograms