%% This file is part of openSSS.
% 
%     openSSS is free software: you can redistribute it and/or modify it under the
%     terms of the GNU General Public License as published by the Free Software
%     Foundation, either version 3 of the License, or (at your option) any later
%     version.
% 
%     openSSS is distributed in the hope that it will be useful, but WITHOUT ANY
%     WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
%     FOR A PARTICULAR PURPOSE.
% 
%     You should have received a copy of the License along with openSSS
% 
% Copyright 2022-2023 all openSSS contributors listed below:
% 
%     --> Rodrigo JOSE SANTO, Andre SALOMON, Hugo DE JONG, Thibaut MERLIN, Simon STUTE, Casper BEIJST
% 
% This is openSSS version 0.1
%
%__________________________________________________________________________________________________________________
%% SinogramCoordinates
% Creates the matrix for the transformation from detector index to sinogram coordinates
% 
% INPUT:    NrSectorsTrans                 - number of sector in the transaxial direction
%           NrSectorsAxial                 - number of sector in the axial direction
%           NrModulesAxial                 - number of modules inside a sector in the axial direction
%           NrModulesTrans                 - number of modules inside a sector in the transaxial direction
%           NrCrystalsTrans                - number of crystals inside a module in the transaxial direction
%           NrCrystalsAxial                - number of crystals inside a module in the axial direction
%
% OUTPUT:   LORCoordinates                 - sinogram coordinates for every detector combination
%           SinogramIndex                  - sinogram coordinates for every ring combination
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________

function [LORCoordinates, SinogramIndex] = SinogramCoordinates(NrSectorsTrans, NrSectorsAxial, NrModulesAxial, NrModulesTrans, NrCrystalsTrans, NrCrystalsAxial)

    NrRings = NrSectorsAxial * NrModulesAxial * NrCrystalsAxial;
    NrCrystalsPerRing = NrSectorsTrans * NrModulesTrans * NrCrystalsTrans;

    % Minimum sector difference: make sure only LORs involving crystals that are at least this amount of rsectors away from each other are used
    MinSectorDifference = 0;
    MinCrystalDifference = MinSectorDifference * NrModulesTrans * NrCrystalsTrans;

    % Sinogram dimensions (always defined for full ring system)
    RadialSize = NrCrystalsPerRing - 2 * (MinCrystalDifference - 1) - 1;
    
    % m_nbSinogramBins is half the acceptation angle (angle / 2) but multiplied by a factor 2 because the LORs of
    % angles phi and phi+1 are both mapped to the same sinogram row (interleaved, to increase sampling)
    % see Bailey 2005, PET Basic Sciences, Figure 3.5
    AngularSize = NrCrystalsPerRing / 2; % only need to cover 180 degrees (other 180 are the same LORs)
    NrSinograms = NrRings * NrRings;
    
    % determine transaxial ID of crystal (in its own ring) relative to the crystal at the center of the first rsector
    % this requires shifting all IDs by half the rsector size
    % as the first rsector is considered at the top of the scanner (positive y-axis pointing towards the ceiling)
    % which implies that the top crystal's ID is 0 and all LORs having phi=0 are aligned with the positive y-axis
    DistanceCrystalId0toFirstSectorCenter = (NrModulesTrans * NrCrystalsTrans) / 2;
    
    LORCoordinates = zeros(NrCrystalsPerRing, NrCrystalsPerRing, 2);

    % Generates first the coordinates on each sinogram
    for Detector1 = 1:NrCrystalsPerRing
        castorFullRingCrystalID1 = Detector1 - 1;
        CrystalId1 = mod(castorFullRingCrystalID1,NrCrystalsPerRing) - DistanceCrystalId0toFirstSectorCenter;

        for Detector2 = 1:NrCrystalsPerRing
            castorFullRingCrystalID2 = Detector2-1;
            CrystalId2 = mod(castorFullRingCrystalID2,NrCrystalsPerRing) - DistanceCrystalId0toFirstSectorCenter;
            
            if (CrystalId1 < 0)
                CrystalId1 = CrystalId1 + NrCrystalsPerRing;
            end
            if (CrystalId2 < 0)
                CrystalId2 = CrystalId2 + NrCrystalsPerRing;
            end
            
            IdA = 0; IdB = 0;
            if (CrystalId1 < CrystalId2)
                IdA = CrystalId1;
                IdB = CrystalId2;
                RingIdA = castorFullRingCrystalID1 / NrCrystalsPerRing;
                RingIdB = castorFullRingCrystalID2 / NrCrystalsPerRing;
            else
                IdA = CrystalId2;
                IdB = CrystalId1;
                RingIdA = castorFullRingCrystalID2 / NrCrystalsPerRing;
                RingIdB = castorFullRingCrystalID1 / NrCrystalsPerRing;
            end
            
            Radial = 0; Angular = 0;
            if (IdB - IdA < MinCrystalDifference)
                continue
            else
                if (IdA + IdB >= (3 * NrCrystalsPerRing) / 2 || IdA + IdB < NrCrystalsPerRing / 2)
                    if (IdA == IdB)
                        Radial = -NrCrystalsPerRing / 2;
                    else
                        Radial = ((IdB - IdA - 1) / 2) - ((NrCrystalsPerRing - (IdB - IdA + 1)) / 2);
                    end
                else
                    if (IdA == IdB)
                        Radial = NrCrystalsPerRing / 2;
                    else
                        Radial = ((NrCrystalsPerRing - (IdB - IdA + 1)) / 2) - ((IdB - IdA - 1) / 2);
                    end
                end
                
                Radial = floor(Radial);
                
                if (IdA + IdB < NrCrystalsPerRing / 2)
                    Angular = (2 * IdA + NrCrystalsPerRing + Radial) / 2;
                else
                    if (IdA + IdB >= (3 * NrCrystalsPerRing) / 2)
                        Angular = (2 * IdA - NrCrystalsPerRing + Radial) / 2;
                    else
                        Angular = (2 * IdA - Radial) / 2;
                    end
                end
                
                LORCoordinates(Detector1, Detector2, 1) = floor(Angular) + 1;
                LORCoordinates(Detector1, Detector2, 2) = floor(Radial + RadialSize / 2) + 1;
            end
        end
    end

    % Generates the order of the sinograms based on the ring difference
    % Increasing in absolute value, first the negative: 0, -1, 1, -2, 2, ...
    SinogramIndex = zeros(NrRings,NrRings);
    for Ring1 = 1:NrRings
        for Ring2 = 1:NrRings
            RingDifference = abs(Ring2 - Ring1);
            if RingDifference == 0
                CurrentSinogramIndex = Ring1;
            else
                CurrentSinogramIndex = NrRings;
                if Ring1 < Ring2
                    if RingDifference > 1
                        for RingDistance = 1:(RingDifference - 1)
                            CurrentSinogramIndex = CurrentSinogramIndex + 2*(NrRings - RingDistance);
                        end
                    end
                    CurrentSinogramIndex = CurrentSinogramIndex + Ring1;
                else
                    if RingDifference > 1
                        for RingDistance = 1:(RingDifference - 1)
                            CurrentSinogramIndex = CurrentSinogramIndex + 2*(NrRings - RingDistance);
                        end
                    end
                    CurrentSinogramIndex = CurrentSinogramIndex + NrRings - RingDifference + Ring1 - RingDifference;
                end
            end
            SinogramIndex(Ring1, Ring2) = CurrentSinogramIndex;
        end
    end
end