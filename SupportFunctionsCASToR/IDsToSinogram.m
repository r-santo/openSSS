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
%% IDsToSinogram
% Gives the sinogram coordinates for every detector combination, given by the CASToR ID for each one, always
% considering that the first ID is smaller than the second
% 
% INPUT:    ScannerGeometry                - geometry of the scanner as obtained from CASToR
%
% OUTPUT:   IDtoSinogramLUT                - LUT table that gives sinogram coordinates for every detector combination, represented by their CASToR IDs
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________

function [IDsToSinogramLUT] = IDsToSinogram(ScannerGeometryFile)
    [ScannerGeometry,] = GeometryCASToR(ScannerGeometryFile);
    [~, ~, ~, NrModulesTrans, ~, NrCrystalsTrans] = ScannerInfoCASToR(ScannerGeometryFile);
    
    NrRings = size(ScannerGeometry, 1);
    NrCrystalsPerRing = size(ScannerGeometry, 2);
    NrElements = NrRings*NrCrystalsPerRing;

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
    
    IDsToSinogramLUT = zeros(NrElements, NrElements, 3, 'int16') -1;

    % Goes over ever element, in the same way as the CASToRIDs are organized
    for CASToRID1 = 1:NrElements
        castorFullRingCrystalID1 = CASToRID1 - 1;
        CrystalId1 = mod(castorFullRingCrystalID1,NrCrystalsPerRing) - DistanceCrystalId0toFirstSectorCenter;

        % CASToRID1 is always considered smalled than CASToRID2
        for CASToRID2 = CASToRID1:NrElements
            castorFullRingCrystalID2 = CASToRID2-1;
            CrystalId2 = mod(castorFullRingCrystalID2,NrCrystalsPerRing) - DistanceCrystalId0toFirstSectorCenter;
            
            if (CrystalId1 < 0)
                CrystalId1 = CrystalId1 + NrCrystalsPerRing;
            end
            if (CrystalId2 < 0)
                CrystalId2 = CrystalId2 + NrCrystalsPerRing;
            end
            
            IdA = 0; IdB = 0;
            RingIdA = 0; RingIdB = 0;
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
            
            % Calculates the radial and angular coordinates
            Radial = -1; Angular = -1;
            if (IdB - IdA >= MinCrystalDifference)
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
            end

            % Calculates the sinogram coordinate
            CurrentSinogramIndex = -1;
            RingDifference = abs(RingIdB - RingIdA);
            if RingDifference == 0
                CurrentSinogramIndex = RingIdA;
            else
                CurrentSinogramIndex = NrRings;
                if RingIdA < RingIdB
                    if RingDifference > 1
                        for RingDistance = 1:(RingDifference - 1)
                            CurrentSinogramIndex = CurrentSinogramIndex + 2*(NrRings - RingDistance);
                        end
                    end
                    CurrentSinogramIndex = CurrentSinogramIndex + RingIdA;
                else
                    if RingDifference > 1
                        for RingDistance = 1:(RingDifference - 1)
                            CurrentSinogramIndex = CurrentSinogramIndex + 2*(NrRings - RingDistance);
                        end
                    end
                    CurrentSinogramIndex = CurrentSinogramIndex + NrRings - RingDifference + RingIdA - RingDifference;
                end
            end

            IDsToSinogramLUT(CASToRID1, CASToRID2, 1) = floor(Radial + RadialSize / 2) + 1;
            IDsToSinogramLUT(CASToRID1, CASToRID2, 2) = floor(Angular) + 1;
            IDsToSinogramLUT(CASToRID1, CASToRID2, 3) = CurrentSinogramIndex + 1;
        end
    end

end