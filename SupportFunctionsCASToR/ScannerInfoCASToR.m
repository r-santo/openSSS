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
%% ScannerInfoCASToR
% Read scanner information from a CASToR .geom file
% 
% INPUT:    ScannerGeomFile                - file path for the scanner file from CASToR (.geom)
%
% OUTPUT:   NrSectorsTrans                 - number of sector in the transaxial direction
%           NrSectorsAxial                 - number of sector in the axial direction
%           NrModulesAxial                 - number of modules inside a sector in the axial direction
%           NrModulesTrans                 - number of modules inside a sector in the transaxial direction
%           NrCrystalsTrans                - number of crystals inside a module in the transaxial direction
%           NrCrystalsAxial                - number of crystals inside a module in the axial direction
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________

function [NrSectorsTrans, NrSectorsAxial, NrModulesAxial, NrModulesTrans, NrCrystalsAxial, NrCrystalsTrans] = ScannerInfoCASToR(ScannerGeomFile)
    
    % Loads all the scanner information directly from the CASToR geometry file
    fileID = fopen(ScannerGeomFile,'r');
    header = fread(fileID,'*char')';
    fclose(fileID);     
    
    NrSectorsTrans = regexp(header,'.*number of rsectors: (\d*).*', 'tokens');
    NrSectorsTrans = str2double(NrSectorsTrans{1,1});
    
    NrSectorsAxial = regexp(header,'.*number of rsectors axial: (\d*).*', 'tokens');
    NrSectorsAxial = str2double(NrSectorsAxial{1,1});
    
    NrModulesAxial = regexp(header,'.*number of modules axial: (\d*).*', 'tokens');
    NrModulesAxial = str2double(NrModulesAxial{1,1});
    
    NrModulesTrans = regexp(header,'.*number of modules transaxial: (\d*).*', 'tokens');
    NrModulesTrans = str2double(NrModulesTrans{1,1});
    
    NrCrystalsAxial = regexp(header,'.*number of crystals axial: (\d*).*', 'tokens');
    NrCrystalsAxial = str2double(NrCrystalsAxial{1,1});
    
    NrCrystalsTrans = regexp(header,'.*number of crystals transaxial: (\d*).*', 'tokens');
    NrCrystalsTrans = str2double(NrCrystalsTrans{1,1});
end