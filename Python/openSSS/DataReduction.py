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
from numpy import ndarray
import time

def ComputeLORCoordinates(
    NrSectorsTrans: int, 
    NrModulesTrans: int, 
    NrCrystalsTrans: int, 
    MinSectorDifference: int = 0
) -> np.ndarray:
    """
    Calculates sinogram coordinates for every detector combination

    Parameters:
    - NrSectorsTrans (int): Number of sectors in the transaxial direction.
    - NrModulesTrans (int): Number of modules inside a sector in the transaxial direction.
    - NrCrystalsTrans (int): Number of crystals inside a module in the transaxial direction.
    - MinSectorDifference (int, optional): Minimum sector difference required to consider two sectors. Default is 0.

    Returns:
    - LORCoordinates (ndarray): Sinogram coordinates for every detector combination.
    """

    NrCrystalsPerRing = NrSectorsTrans * NrModulesTrans * NrCrystalsTrans
    MinCrystalDifference = MinSectorDifference * NrModulesTrans * NrCrystalsTrans

    RadialSize = NrCrystalsPerRing - 2 * (MinCrystalDifference - 1) - 1

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

    return LORCoordinates.astype(np.int16)

def SinogramLUT(
        NrRings : int,
        AxialMashing : int,
        MaxRingDifference : int = None
        )->tuple[ndarray, ndarray, ndarray, ndarray]:

    '''
    Mashes the sinograms together according to the span (axial mashing) (LUT = Look-up Table)

    Parameters:
    - NrRings (int): Total Rings that the scanner contains
    - AxialMashing (int): Span value
    - MaxRingDifference (int, optional): Max difference between 2 rings to create a sinogram
    
    Returns:
    - SinogramIndex (ndarray): Gives the corresponding rings that contribute to each sinogram slice
    - RingLUT (ndarray): Boolean matrix that defines whether a ring pair is the mashed sinogram
    - LookUpTable (ndarray): For each ring-pairing refer to the mashed sinogram slice it got mashed into
    - SinogramCounts (ndarray): Gives the number of contributing LORs per sinogram slice
    '''

    NrSinograms = int(NrRings*2 - 1 + np.sum(4*(NrRings - (AxialMashing+1)/2 - AxialMashing*(np.arange(1,np.round(NrRings/AxialMashing)+1)-1)) - 2))
    SinogramIndex = np.zeros((NrSinograms, 2), dtype=np.float32)
    SinogramCounts= np.zeros((NrSinograms), dtype=np.float32)
    LookUpTable = np.full((NrRings, NrRings), -1, dtype=np.int16)


    for Ring1 in range(1,NrRings+1):
        for Ring2 in range(1,NrRings+1):
            if MaxRingDifference is not None:
                if abs(Ring1 - Ring2) > MaxRingDifference:
                    continue

            if abs(Ring2 - Ring1) < (AxialMashing + 1)/2:
                CurrentSinogramIndex = Ring1 + Ring2 - 1
            else:
                ShiftAmount = np.floor((abs(Ring2-Ring1) - (AxialMashing + 1)/2)/AxialMashing).astype(int) + 1
                ShiftPosition = 2*NrRings - 1
                
                if (Ring2 - Ring1 > 0):
                    ShiftBoundary = ShiftAmount + 1
                else:
                    ShiftBoundary = ShiftAmount

                for i in range(1,ShiftBoundary):
                    if (i == ShiftBoundary-1) and (Ring2 - Ring1 > 0):
                        RingDifferenceJump = 1
                    else:
                        RingDifferenceJump = 2
                    ShiftPosition = ShiftPosition + ((NrRings - (AxialMashing + 1)/2 - AxialMashing*(i-1))*2 - 1)*RingDifferenceJump
                CurrentSinogramIndex = (Ring1 + Ring2 - 1) - ShiftAmount*AxialMashing + (AxialMashing-1)/2 + ShiftPosition

            SinogramIndex[int(CurrentSinogramIndex - 1), 0] += Ring2-1
            SinogramIndex[int(CurrentSinogramIndex - 1), 1] += Ring1-1
            SinogramCounts[int(CurrentSinogramIndex - 1)] += 1

            LookUpTable[Ring1-1,Ring2-1] = int(CurrentSinogramIndex - 1)

    SinogramIndex[:,0] /= SinogramCounts + 1e-8
    SinogramIndex[:,1] /= SinogramCounts + 1e-8

    # For each sinogram coordinate, gives the corresponding ring
    SinogramIndex = SinogramIndex[SinogramCounts!=0,:]

    RingLUT = np.full((NrRings*2-1, NrRings*2-1), False, dtype=bool)
    for sinoID in range(SinogramIndex.shape[0]):
        Ring1 = int(SinogramIndex[sinoID, 0]*2)
        Ring2 = int(SinogramIndex[sinoID, 1]*2)
        RingLUT[Ring1,Ring2] = True

    return SinogramIndex, RingLUT, LookUpTable, SinogramCounts

def SumLORsCounts(
        Geometry : ndarray,
        LookUpTable : ndarray,
        SinogramIndex : ndarray,
        Mash : int,
        Span : int,
        NrSinograms : int
        )->np.ndarray:
    
    """
    Calculates the number of contributing LORs for each sinogram coordinate

    Parameters:
    - Geometry (ndarray): Goemetry of the scanner.
    - LookUpTable (int): Gives the real rings for each sinogram slice, in case span is applied.
    - SinogramIndex (int): Gives the rings for each sinogram slice.
    - Mash (int): Mash value.
    - Span (int): Span value.
    - NrSinogram (int): Number of sinograms

    Returns:
    - sumLORCounts (ndarray): Number of contributing LORs per sinogram coordinate.
    """

    # SinogramCounts, LookupTable, LORCoordinates
    NrDetectors = Geometry.shape[1]
    NrAngles = NrDetectors // 2
    NrRings = Geometry.shape[0]

    sinogramLORCountsRing = np.full((NrDetectors + 1, NrAngles), Mash**2, np.float32)
    
    if Span > 1:
        sumLORCounts = np.zeros((NrDetectors + 1, NrAngles, NrSinograms), np.float32)
    else:
        sumLORCounts = np.zeros((NrDetectors + 1, NrAngles, NrRings**2), np.float32)

    for ring1 in range(NrRings):
        for ring2 in range(NrRings):
            if Span > 1:
                currentSinogramIndex = LookUpTable[ring1,ring2]
            else:
                currentSinogramIndex = SinogramIndex[ring1,ring2]

            sumLORCounts[:,:,currentSinogramIndex] += sinogramLORCountsRing

    return sumLORCounts