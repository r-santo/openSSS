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
% Example of how to use openSSS without TOF, applied to the Siemens Briograph Vision 600 geometry
% for a NEMA phantom simulated with Monte-Carlo
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________
%% Compile ray tracing function
% Only needed once
if ~isfile("./RayTracing3D.mexa64") && ~isfile("./RayTracing3D.mexw64") && ~isfile("./RayTracing3D.mexmaci64")
    mex RayTracing3D.cpp
end

%% Import information about the system
% Scanner name
ScannerName = 'ToyScanner';
FileName = sprintf('./Data/%s.mat', ScannerName);

% Geometry and normalization factors, from reconstruction library
% The format is [Ring, Detector, Coordinate], with coordinates ordered by [x, y, z]
% Units are in cm
load(FileName, 'Geometry', 'NormalVectors')
% Size of the scanner, in mm (here converted from cm to mm)
DeviceSize = [(max(Geometry(:,:,1),[],'All') - min(Geometry(:,:,1),[],'All')),...
              (max(Geometry(:,:,2),[],'All') - min(Geometry(:,:,2),[],'All')),...
              (max(Geometry(:,:,3),[],'All') - min(Geometry(:,:,3),[],'All'))]*10;

% Values from the scanner design
load(FileName, 'NrSectorsAxial', 'NrSectorsTrans', 'NrModulesAxial', 'NrModulesTrans', 'NrCrystalsAxial', 'NrCrystalsTrans')
NrRings = NrSectorsAxial * NrModulesAxial * NrCrystalsAxial;
NrCrystals = NrSectorsTrans * NrModulesTrans* NrCrystalsTrans;

% Size of each detector unit/crystal [x, y] in mm
load(FileName, 'DetectorSize');

% Energy resolution of the system, in decimal units (max 1)
load(FileName, 'EnergyResolution');

% Creates custom sinogram format, that indexes the radial and angular component of LORs relative to each other
% starting from the middle of the first sector and with format [radial, angular]
% Sinograms are ordered based on the ring difference, starting on the negative (0, -1, 1, -2, 2, ...)
[LORCoordinates, SinogramIndex] = SinogramCoordinates(NrSectorsTrans, NrSectorsAxial, NrModulesAxial, NrModulesTrans, NrCrystalsTrans, NrCrystalsAxial);
% Creates helper variables with coordinates of detectors and rings 
% associated with every sinogram coordinate and index
[DetectorCoordinates, RingCoordinates] = SinogramToSpatial(NrSectorsTrans, NrSectorsAxial, NrModulesAxial, NrModulesTrans, NrCrystalsTrans, NrCrystalsAxial, Geometry);

%% Information on the specific settings for the acqusition
% Energy threshold defined for the acquisition, in keV
EnergyThreshold = 435;

%% Read the images
% This includes both the actual image and the corresponding voxel size, given in mm
% Attenuation image must be in cm-1
load('./Data/AttenuationImage.mat', 'AttenuationMap','AttenuationSize');
% Activity image can be in arbitrary unites
load('./Data/ActivityImage.mat', 'ActivityMap','ActivitySize');

% It is possible to crop and downscale the images. This is recommended to avoid
% running out of memmory and crashing the computer
% Units in mm
DesiredDimensions = [60, 60 ,32];
DesiredSize = [size(ActivityMap,1)*ActivitySize(1), ...
               size(ActivityMap,1)*ActivitySize(2), ...
               DeviceSize(3)];

% Coordinates for the bounds of the image to be used to estimate scatters
% in the format [xStart, yStart, zStart, xEnd, yEnd, zEnd] and in cm
ImageSize = [-DesiredSize(1)/2, -DesiredSize(2)/2, -DesiredSize(3)/2, ...
              DesiredSize(1)/2, DesiredSize(2)/2, DesiredSize(3)/2]/10;

% Coordinates for the bounds of the backprojected data to be used in the scalling of scatters
% in the format [xStart, yStart, zStart, xEnd, yEnd, zEnd] and in cm
FittingSize = [-DesiredSize(1)/2, -DesiredSize(2)/2, -DeviceSize(3)/2, ...
              DesiredSize(1)/2, DesiredSize(2)/2, DeviceSize(3)/2]/10;

%% Crops and downscaled the attenuation map, which is used for the whole SSS
AttenuationMapDownscaled = CropAndDownscale(AttenuationMap, AttenuationSize, DesiredSize, DesiredDimensions);
% Very low attenuation values (such as air) do not influence the SSS significantly, so can be skipped by making them 0 (no attenuation)
AttenuationMapDownscaled(AttenuationMapDownscaled < 0.001) = 0;

ActivityMapDownscaled = CropAndDownscale(ActivityMap, ActivitySize, DesiredSize, DesiredDimensions);

%% Settings for SSS, to be balanced between speed and accuracy
% Number of rings  and detectors (per ring) to simulate, the rest being interpolated
NrRingsSimulated = 3;
NrDetectorsSimulated = NrCrystals/4;

% Step to sample scatter points in each diretion, in units of integer voxels
% Example means that scatter points are sampled every 3 voxels in x
% direction, 3 voxels in y direction and every 3 voxels in z direction
SampleStep = [3, 3, 2]; 

% Detectors to be skipped when generating the tail-mask and backprojecting
% events. It reduces the number of LORs considered, so it is faster but fitting 
% may be less representative of the full distrubution of events
AccelerationFactor = 2;

% Path where to save the scatter estimates of each time bin. These are not
% stored in memory to avoid crashing the computer
SavePath = '/mnt/sata1/temp';

%%
fprintf('Generating tail mask... ');
AttenuationMask = MaskGenerator(ActivityMapDownscaled, AttenuationMapDownscaled, ImageSize, Geometry, LORCoordinates, SinogramIndex, true, AccelerationFactor);
fprintf('completed!\n');

%% Scatter estimation
% Estimates the scatters of the current image
fprintf('Perform estimation of scatter...');
Scatters = SingleScatterSimulation(ActivityMapDownscaled, AttenuationMapDownscaled, ImageSize,...
                                   Geometry, LORCoordinates, SinogramIndex,...
                                   NormalVectors,  DetectorSize, EnergyResolution, EnergyThreshold,...
                                   NrRingsSimulated, NrDetectorsSimulated, SampleStep,...
                                   SavePath);
fprintf('completed\n');
    
% Scales the scatters appropriately, based on the prompts minus randoms
fprintf('Perform scaling of scatter...');
% Prompts and randoms set to 0 only for exemplification
Prompts = zeros(size(Scatters), 'uint8');
Randoms = zeros(size(Scatters), 'single');
% Calculates a single scaling factor for the non-TOF scatters
ScaleFactor = BackProjectionScatterRandFit(Prompts, Scatters, Randoms, AttenuationMask, DesiredDimensions, FittingSize, Geometry, LORCoordinates, SinogramIndex, AccelerationFactor);
fprintf('completed\n');

fprintf('Scatter estimation terminated\n\n');
