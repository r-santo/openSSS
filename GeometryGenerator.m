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
% Copyright 2022-2024 all openSSS contributors listed below:
% 
%     --> Rodrigo JOSE SANTO, Andre SALOMON, Hugo DE JONG, Thibaut MERLIN, Simon STUTE, Casper BEIJST
% 
% This is openSSS version 0.1
%
%__________________________________________________________________________________________________________________
%% Generate lookup table PET-detectors
% Helper script to generate custom cylindrical geometries
% Note: all lengths are in cm 
%
%       Y                                        _________  
%       |                                       / _ \     \ 
%       |                                      | / \ |     |
%       |_____ Z                               | | | |     |
%        \                                     | | | |     |
%         \                                    | \_/ |     |
%          X                                    \___/_____/
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________
%% Information on PET-scanner
ScannerName = 'uExplorer';

NrSectorsAxial = 8;          % number of sectors axial - contains modules 
NrSectorsTrans = 24;         % number of sectors transaxial - contains modules side by side in the same slope angle
SectorGapAxial = 0.5;        % sector gap axial
AngleStart = 0;              % angle of the first sector
Radius = 40;                 % radius PET-scanner
NrModulesAxial = 13;          % number of modules axial - contains detectors
NrModulesTrans = 6;          % number of modules transaxial - contains detectors
ModulesGapAxial = 0.1;       % module gap axial
ModulesGapTrans = 0.1;       % module gap transaxial
NrCrystalsAxial = 6;         % number of detectors per module axial
NrCrystalsTrans = 6;         % number of detectors per module transaxial
DetectorSize = [0.27, 0.27];   % size of detector transaxial and axial
DetectorGap = [0.01, 0.01];  % detector gap transaxial and axial

% Parameters relevant for scatter estimation only
EnergyResolution = 0.15;     % energy resolution
TOFResolution = 400;         % TOF resolution (ps)

%% 

NrRings = NrSectorsAxial * NrModulesAxial * NrCrystalsAxial;        % number of rings
NrCrystalsPerSectorTrans = NrModulesTrans * NrCrystalsTrans;        % number of detectors per sector in one ring
NrCrystalsPerRing = NrCrystalsPerSectorTrans * NrSectorsTrans;      % number of detectors per ring
AngleStep = 360.0 / NrSectorsTrans;                                 % step size of angle to middle of a sector

Geometry = zeros(NrRings,NrCrystalsPerRing,3);                      % lookup table of the geometry
NormalVectors = zeros(NrRings,NrCrystalsPerRing,3);                     % lookup table of the orientation of the detectors

Zdetector = .5 * DetectorSize(2);  % z of first ring of detectors

% loop over all rings
for Ring = 1:NrRings
    
    % loop over all sectors
    for Sector = 1:NrSectorsTrans
        
        Angle = 90 + AngleStart + (Sector-1) * AngleStep;   % angle sector
        AngleSector = Angle - 90;                           % slope of sector
        Xmid = Radius * cosd(Angle);                        % x middle of sector
        Ymid = Radius * sind(Angle);                        % y middle of sector
        
        % shift for even number of modules
        if mod(NrModulesTrans,2) == 0
            Shift = .5*(ModulesGapTrans + DetectorSize(1));
        else
            Shift = .5 * (DetectorSize(1) + DetectorGap(1));
        end
        
        Xdetector = Xmid + Shift * cosd(AngleSector); % x of first detector after middle
        Ydetector = Ymid + Shift * sind(AngleSector); % y of first detector after middle 
        
        Detector = (Sector - .5) * NrCrystalsPerSectorTrans + 1;        % detector index first detector after middle

        % loop over all detectors after middle of current sector
        while Detector < (Sector * NrCrystalsPerSectorTrans + 1)
            
            % add x, y and z to lookup table for current detector
            Geometry(Ring,Detector,1) = Xdetector;
            Geometry(Ring,Detector,2) = Ydetector;
            Geometry(Ring,Detector,3) = Zdetector;

            % add orientation x, y and z to lookup table for current detector
            NormalVectors(Ring,Detector,1) = -cosd(AngleSector);
            NormalVectors(Ring,Detector,2) = -sind(AngleSector);
            NormalVectors(Ring,Detector,3) = 0;
            
            % check if the next detector is in a different module
            if round(Detector / NrCrystalsTrans) == (Detector / NrCrystalsTrans)
                Xdetector = Xdetector + (DetectorSize(1) + ModulesGapTrans) * cosd(AngleSector);  % add module gap
                Ydetector = Ydetector + (DetectorSize(1) + ModulesGapTrans) * sind(AngleSector);  % add module gap
            else
                Xdetector = Xdetector + (DetectorSize(1) + DetectorGap(1)) * cosd(AngleSector);  % only add detector gap
                Ydetector = Ydetector + (DetectorSize(1) + DetectorGap(1)) * sind(AngleSector);  % only add detector gap
            end
            
            Detector = Detector + 1;  % index next detector
        end
        
        Xdetector = Xmid - Shift * cosd(AngleSector); % x of first detector before middle
        Ydetector = Ymid - Shift * sind(AngleSector); % y of first detector before middle
        
        Detector = (Sector - .5) * NrCrystalsPerSectorTrans;             % detector index first detector before middle
        
        % loop over all detectors before middle of current sector
        while Detector > (Sector - 1) * NrCrystalsPerSectorTrans
            
            % add x, y and z to lookup table for current detector
            Geometry(Ring,Detector,1) = Xdetector;
            Geometry(Ring,Detector,2) = Ydetector;
            Geometry(Ring,Detector,3) = Zdetector;

            % add orientation x, y and z to lookup table for current detector
            NormalVectors(Ring,Detector,1) = -cosd(AngleSector);
            NormalVectors(Ring,Detector,2) = -sind(AngleSector);
            NormalVectors(Ring,Detector,3) = 0;
            
            % check if the next detector is in a different module
            if round(((Detector - 1) / NrCrystalsTrans)) == ((Detector - 1) / NrCrystalsTrans)
                Xdetector = Xdetector - (DetectorSize(1) + ModulesGapTrans) * cosd(AngleSector);  % add module gap
                Ydetector = Ydetector - (DetectorSize(1) + ModulesGapTrans) * sind(AngleSector);  % add module gap
            else
                Xdetector = Xdetector - (DetectorSize(1) + DetectorGap(1)) * cosd(AngleSector);  % only add detector gap
                Ydetector = Ydetector - (DetectorSize(1) + DetectorGap(1)) * sind(AngleSector);  % only add detector gap
            end
            
            Detector = Detector - 1;  % index next detector
        end
        
    end

    % check if the next ring of detectors is in a different sector
    if rem(Ring,NrCrystalsAxial*NrModulesAxial) == 0
        Zdetector = Zdetector + DetectorSize(2) + SectorGapAxial;
    else
        % check if the next ring of detectors is in a different module
        if rem(Ring,NrCrystalsAxial) == 0        
            Zdetector = Zdetector + DetectorSize(2) + ModulesGapAxial;  % add module gap
        else
            Zdetector = Zdetector + DetectorSize(2) + DetectorGap(2); % only add detector gap
        end
    end
    
end

% To center the scanner so that the origin is in its middle
ScannerWidth = (max(Geometry(:,:,3),[],'All') - min(Geometry(:,:,3),[],'All'));
Geometry(:,:,3) = Geometry(:,:,3) - ScannerWidth/2 - min(Geometry(:,:,3),[],'All');

FileName = sprintf('./Data/%s.mat',ScannerName);
DetectorSize = DetectorSize*10; % To convert from cm to mm as required for openSSS on this parameter
save(FileName, ...
    'Geometry', 'NormalVectors',...
    'NrSectorsAxial', 'NrSectorsTrans', 'NrModulesAxial', 'NrModulesTrans', 'NrCrystalsAxial', 'NrCrystalsTrans',...
    'DetectorSize', 'EnergyResolution', 'TOFResolution');
