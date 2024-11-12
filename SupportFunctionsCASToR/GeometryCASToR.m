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
%% GeometryCASToR
% Import the geometry based on a CASToR LUT
% 
% INPUT:    ScannerGeomFile        - file path for the scanner file from CASToR [.geom]
%
% OUTPUT:   Geometry               - spatial coordinates for all detectors, organized as [Rings, Crystals,Dimension]
%           NormalVectors          - vector normal to the face of the crystal, on the same organization as geometry
%           DetectorSize           - Size of each detector/crystal
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________

function [Geometry, NormalVectors, DetectorSize] = GeometryCASToR(ScannerGeomFile)
    
    % Loads all the scanner information directly from the CASToR geometry file
    fileID = fopen(ScannerGeomFile,'r');
    header = fread(fileID,'*char')';
    fclose(fileID);     
    
    NrElements = regexp(header,'.*number of elements: (\d*).*', 'tokens');
    NrElements = str2double(NrElements{1,1});
    
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

    DetectorSizeAxial = regexp(header,'.*crystals size axial: (\d*).*', 'tokens');
    DetectorSizeAxial = str2double(DetectorSizeAxial{1,1});

    DetectorSizeTrans = regexp(header,'.*crystals size trans: (\d*).*', 'tokens');
    DetectorSizeTrans = str2double(DetectorSizeTrans{1,1});

    DetectorSize = [DetectorSizeTrans, DetectorSizeAxial];
    
    NrRings = NrModulesAxial * NrCrystalsAxial * NrSectorsAxial;
    
    % Loads the LUT, composed of the crystal coordinates and the normal vectors
    % Should have the same filename but [.glut] file extension
    [FilePath, FileName, ]= fileparts(ScannerGeomFile);
    geomFile = sprintf('%s/%s.glut',FilePath, FileName);
    LUT = ReadBinaryFile(geomFile, [6, NrElements,1],'single','l');

    Geometry = permute(LUT(1:3,:), [2, 1])/10;
    Geometry = reshape(Geometry, NrElements/NrRings, NrRings, 3);
    Geometry = permute(Geometry, [2, 1, 3]);
    
    NormalVectors = permute(LUT(4:6,:), [2, 1]);
    NormalVectors = reshape(NormalVectors, NrElements/NrRings, NrRings, 3);
    NormalVectors = permute(NormalVectors, [2, 1, 3]);
end