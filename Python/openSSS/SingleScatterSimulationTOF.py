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

from scipy.special import erf
from numpy import ndarray # for comment the function
from openSSS.RayTracing3DTOF import RayTracing3DTOF

from joblib import Parallel, delayed # for parallel processing
from numba import jit # for fast computation

ElectronMass = 9.10938356E-31       #mass electron (kg)
Lightspeed = 299792458.0            #speed of light (m/s)

# Main function, the SSS algorithm with TOF
def SingleScatterSimulationTOF(
    ActivityMap: ndarray,
    AttenuationMap: ndarray,
    ImageSize: ndarray, 
    Geometry: ndarray, 
    SinogramCoordinates: ndarray, 
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
    SavePath: str,
    )->ndarray:
    """
    3D Single Scatter Simulation (SSS)
    Performs TOF scatter estimation based on the SSS algorithm by Watson
    
    Parameters:
    - ActivityMap (ndarray): Activity map (current estimation)
    - AttenuationMap (ndarray): Attenuation map
    - ImageSize (ndarray): Attenuation and activity image size [-x -y -z x y z] (mm)
    - Geometry (ndarray): 3D array with the (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
    - SinogramCoordinates (ndarray): Array with the sinogram coordinates for detector combinations
    - NormalVectors (ndarray): Vectors normal to the surface area of detectors
    - DetectorSize (ndarray): Size of the surface of the detectors [x, y] (mm)
    - AttenuationTable (ndarray): Linear Attenuation coefficient respect to the energy
    - EnergyResolution (float): Energy resolution (%)
    - EnergyThreshold (float): Energy threshold (keV)
    - NrRingsUsed (int): Number of rings to use for the scatter estimation before interpolation
    - NrDetectorsUsed (int): Number of detectors to use for the scatter estimation before interpolation
    - SampleStep (int): Step size for the sample of scatter points [x, y, z] in voxels
    - TOFResolution (float): Time resolution (ps)
    - TOFRange (float): Range of TOF measured (ps)
    - NrBins (int): Number of TOF bins
    - SavePath (str): Path where to save the sinograms
    
    Returns:  
    - Scatters (ndarray): Scatters for only the sampled rings and detectors (sinogram coordinates)
    """

    xDim, yDim, zDim = AttenuationMap.shape # dimensions of the contour image
    NrDetectors = Geometry.shape[1] # number of detectors used
    NrRings = Geometry.shape[0] # number of rings

    # the crop for the attenuation and activity map expands half the detector height for each side of the FOV
    zVoxelSize = (ImageSize[5] - ImageSize[2]) / zDim # real distance between middle two slices in contour_image

    EnergyReferenceJoule = 511.0E3 * 1.6E-19  # Photon energy (J)
    SmallNumber = 10 ** (-15)  # used to prevent dividing by zero
    EnergyReference = 511  # energy used for generating attenuation map (keV)
    GridSize = np.array(ActivityMap.shape)  # dimensions for the ray tracing grid
    GridBounds = np.array(ImageSize)  # real dimensions of the ray tracing grid

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

    # prepare flat array to be called in bodySSS
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
    elapsed_time1 = end_time_scatters - start_time_scatters
    print("Time for SSS: {:.2f} seconds".format(elapsed_time1))
    np.savez_compressed(os.path.join(SavePath, 'Scatters_SSS.npz'), Scatters)  

    return Scatters


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
    - LOR (float): Distance of Detector 1 and Detector 2
    
    Returns:
    - ScatterAngle (float): Scattering angle in degrees.
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
    - probability (float): Probabity of scattering event.
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

    # # Normalize the probability values
    # probability = probability / np.sum(probability, axis=0)

    # Normalize the probability values
    zero_mask = np.sum(probability, axis=0) == 0
    probability = probability / np.sum(probability, axis=0)
    probability[:,zero_mask] = 0
    return probability

##################################################################################################
#################### MAIN FUNCTIONS OF THE SINGLE SCATTER SIMULATION #############################
##################################################################################################
@jit(nopython=True)
def bodySSS(
    zIndex : int, 
    zStart : float, 
    zVoxelSize : float, 
    NrBins : int, 
    NrDetectors : int, 
    NrSinograms : int, 
    SampleStep : ndarray,
    Radius : float, 
    xDim : int, 
    yDim : int, 
    zDim : int, 
    AttenuationTissue : ndarray, 
    NrRings : int, 
    NrRingsUsed : int, 
    NrDetectorsUsed : int,
    Rings : ndarray, 
    Geometry : ndarray, 
    Detectors : ndarray, 
    GridSize : ndarray, 
    GridBounds : ndarray, 
    ActivityMap_flat : ndarray, 
    AttenuationMap_flat : ndarray, 
    NormalVectors : ndarray,
    SinogramCoordinates : ndarray, 
    EnergyReference : float, 
    EfficiencyTable : ndarray, 
    AttenuationRatios : ndarray, 
    LightSpeed : float, 
    TOFTable : ndarray,
    TimeRange : float, 
    DetectorSize : ndarray, 
    EnergyReferenceJoule : float, 
    SmallNumber : float
    )->ndarray:

    """
    Parallelized 3D Single Scatter Simulation (SSS)
    For each Z coordinate, performs TOF scatter estimation based on the SSS algorithm by Watson
    
    Parameters:
    - zIndex (int) : Voxel index in the Z (axial) direction.
    - zStart (float) : Spatial coordinate in the Z (axial) direction where the maps start,.
    - zVoxelSize (float) : Size of the voxels in the Z (axial) direction, 
    - NrBins (int) : Number of TOF bins 
    - NrDetectors (int) : Total number of detectors per ring in the scanner 
    - NrSinograms (int) : Number of sinograms (slices) 
    - SampleStep (ndarray) : Voxel steps to jump over when choosing scatter points in the maps
    - Radius (float) : Spatial size of the scanner in the transaxial direction (assumes it is the same in X and Y) 
    - xDim (int) : Number of voxels in the X direction 
    - yDim (int) : Number of voxels in the Y direction
    - zDim (int) : Number of voxels in the Z direction
    - AttenuationTissue (ndarray) : Attenuation map converted into tissue type, as an index of the AttenuationRatios table
    - NrRings (int) : Number of rings of the scanner 
    - NrRingsUsed (int) : Number of rings sampled for scatter estimation
    - NrDetectorsUsed (int) : Number of detectors sampled per ring for scatter estimation
    - Rings (ndarray) : Array with the index of the rings sampled 
    - Geometry (ndarray) : Geometry of the scanner, with the spatial coordinates of every crystal
    - Detectors (ndarray) : Array with the idex of the detectors sampled per ring
    - GridSize (ndarray) : Voxel dimensions of the activity and attenuation maps 
    - GridBounds (ndarray) : Spatial coordinates of the boundaries of the maps
    - ActivityMap_flat (ndarray) : Flattened activity map
    - AttenuationMap_flat (ndarray) : Flattened attenuation map
    - NormalVectors (ndarray) : Vectors normal to the face of the crystals, for all crystals of the scanner
    - SinogramCoordinates (ndarray) : Sinogram index (slice) for every ring pair
    - EnergyReference (float) : Reference energy of positron annihilation in keV 
    - EfficiencyTable (ndarray) : Array with the detection efficiency for different energies, in steps of 0.5 keV 
    - AttenuationRatios (ndarray) : Array with the scaling for the linear attenuation factors for different energies and tissues
    - LightSpeed (float) : speed of light
    - TOFTable (ndarray) : Array with the TOF efficiency for different TOF bins and TOF values
    - TimeRange (float) : TOF range based on the TOF bins and TOF bin size 
    - DetectorSize (ndarray) : size of the face of the crystals
    - EnergyReferenceJoule (float) : Reference energy of positron annihilation in Joule
    - SmallNumber (float) : Small number to prevent dividing by 0
    
    Returns:  
    - ScatterSlice (ndarray): Scatters for only the sampled rings and detectors coming from the slice zIndex of the maps
    """


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

                            LineCoordinates = np.array([xScatterPoint, yScatterPoint, zScatterPoint,
                                               xDetector1, yDetector1, zDetector1])
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
                                                ActivityBinned1 = TOFTable[:, SpatialOffset.astype(np.int32)]
                                                ActivityBinned1 = np.dot(ActivityRay1, ActivityBinned1.T)
                                            else:
                                                ActivityBinned1 = np.zeros(1)
                                            if ActivityPath2 != 0:
                                                SpatialOffset = np.floor(2 * ((ScatterVector1 - ScatterVector2) / 2 + LineDistributions[RingIndex2][DetectorIndex2][1].T) / LightSpeed + 0.5) + TimeRange
                                                ActivityBinned2 = TOFTable[:, SpatialOffset.astype(np.int32)]
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

    ScatterSlice = ScatterSlice / ScatterCounts
    # remove any nan value from ScatterCount being 0 for some sinogram coordinates
    ScatterSlice = np.nan_to_num(ScatterSlice)

    return ScatterSlice
