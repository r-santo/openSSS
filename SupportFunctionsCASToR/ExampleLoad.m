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
%% Example of openSSS
% Example of how to load the scanner geometry from CASToR into openSSS and
% how to related CASToR IDs with openSSS sinogram coordinates
%
% Only provided as support to link with CASToR externally. For direct
% bridging between openSSS and CASToR, check the official CASToR gitlab
% (WORK IN PROGRESS - not yet available)
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________
%% Import information about the system
% Scanner geometry file from CASToR (.geom)
GeometryFile = 'foo/bar.geom';
[Geometry, NormalVectors, DetectorSize] = GeometryCASToR(GeometryFile);
[NrSectorsTrans, NrSectorsAxial, NrModulesAxial, NrModulesTrans, NrCrystalsAxial, NrCrystalsTrans] = ScannerInfoCASToR(GeometryFile);

% Plot the scanner geometry
plot3(Geometry(:,:,1), Geometry(:,:,2), Geometry(:,:,3), '.r');
axis equal;
%% Create LUT to convert from CASToR IDs to sinogram coordinates
% Slow to compute, depending on the number of total elements/crystals/detector
IDsToSinogramLUT = IDsToSinogram(GeometryFile);

%__________________________________________________________________________________________________________________