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

@jit(nopython=True)
def SinogramCoordinates(
    NrSectorsTrans: int, 
    NrSectorsAxial: int, 
    NrModulesAxial: int, 
    NrModulesTrans: int, 
    NrCrystalsTrans: int, 
    NrCrystalsAxial: int,
    MinSectorDifference: int = 0,
    ID_Shidt: bool = True
) -> Tuple[ndarray, ndarray, int]:
    """
    Calculates sinogram coordinates for every detector combination and sinogram indices for every ring combination.

    Parameters:
    - NrSectorsTrans (int): Number of sectors in the transaxial direction.
    - NrSectorsAxial (int): Number of sectors in the axial direction.
    - NrModulesAxial (int): Number of modules inside a sector in the axial direction.
    - NrModulesTrans (int): Number of modules inside a sector in the transaxial direction.
    - NrCrystalsTrans (int): Number of crystals inside a module in the transaxial direction.
    - NrCrystalsAxial (int): Number of crystals inside a module in the axial direction.
    - MinSectorDifference (int, optional): Minimum sector difference required to consider two sectors. Default is 0.

    Returns:
    - LORCoordinates (ndarray): Sinogram coordinates for every detector combination.
    - SinogramIndex (ndarray): Sinogram coordinates for every ring combination.
    - DetectorShift (int): Gives the detector shift, used for export the prompts
    """
    NrRings = NrSectorsAxial * NrModulesAxial * NrCrystalsAxial
    NrCrystalsPerRing = NrSectorsTrans * NrModulesTrans * NrCrystalsTrans
    MinCrystalDifference = MinSectorDifference * NrModulesTrans * NrCrystalsTrans

    RadialSize = NrCrystalsPerRing - 2 * (MinCrystalDifference - 1) - 1
    AngularSize = NrCrystalsPerRing // 2
    NrSinograms = NrRings * NrRings

    DistanceCrystalId0toFirstSectorCenter = (NrModulesTrans * NrCrystalsTrans) // 2 if ID_Shidt else 0
    DetectorShift = int(DistanceCrystalId0toFirstSectorCenter)

    LORCoordinates = np.zeros((NrCrystalsPerRing, NrCrystalsPerRing, 2)) - 1

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

            if CrystalId1 < CrystalId2:
                IdA = CrystalId1
                IdB = CrystalId2

                RingIdA = castorFullRingCrystalID1 // NrCrystalsPerRing
                RingIdB = castorFullRingCrystalID2 // NrCrystalsPerRing
            else:
                IdA = CrystalId2
                IdB = CrystalId1

                RingIdA = castorFullRingCrystalID2 // NrCrystalsPerRing
                RingIdB = castorFullRingCrystalID1 // NrCrystalsPerRing
            

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
            if Ring1 == Ring2:
                CurrentSinogramIndex = Ring1
            else:
                if Ring2 < Ring1:
                    ringDifference = Ring1 - Ring2
                    CurrentSinogramIndex = (2 * ringDifference - 1) * NrRings - (ringDifference * (ringDifference - 1)) + Ring2
                else:
                    ringDifference = Ring2 - Ring1
                    CurrentSinogramIndex = (2 * ringDifference - 1) * NrRings - (ringDifference * (ringDifference - 1)) \
                        + (NrRings - ringDifference) + Ring1
            # Python Coordinate
            SinogramIndex[Ring1, Ring2] = CurrentSinogramIndex

    return LORCoordinates.astype(np.int16), SinogramIndex.astype(np.int16), DetectorShift