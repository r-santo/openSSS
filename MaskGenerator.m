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
%% MaskGenerator
% Generates the mask for the tail fitting of scatter sinograms
% 
% INPUT:    ActivityMap                 - activity maps estimation
%           AttenuationMap              - attenuation map
%           ImageSize                   - attenuation and activity image size [-x -y -z x y z] (mm)
%           Geometry                    - 3D array with the (x,y,z) positions of the detectors
%           SinogramCoordinates         - array with the sinogram positions for detector combinations
%           SinogramIndex               - array with the order of the sinograms for ring combinations
%           UseAttenuation              - operation mode for the mask delineation
%           AccelerationFactor             - step for the iteration over the detectors of the rings
%
% OUTPUT:   Sinograms                   - estimated scatters 
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________
function Sinograms = MaskGenerator(ActivityMap, AttenuationMap, ImageSize, Geometry, SinogramCoordinates, SinogramIndex, UseAttenuation, AccelerationFactor)
    if nargin == 7
        AccelerationFactor = 1;
    end
    %%  scanner information, input images, constants and lookup tables for main algorithm
    % define some constants
    NrDetectors = size(Geometry,2);                % number of detectors used
    NrRings = size(Geometry,1);                    % number of rings
    GridSize = size(ActivityMap);                  % voxel dimensions for the ray tracing grid
    GridBounds = ImageSize;                        % real dimensions for the ray tracing grid

    % calculate the number of sinograms that will be obtained (oblique + non-oblique)
    NrSinograms = NrRings^2;
    
    % structure to save the sinograms in the format used in the rest of the PET/MRI reconstruction pipeline
    Sinograms = zeros(NrDetectors+1, NrDetectors/2, NrRings, NrRings, 'uint8');
    
    % Calculates the line path which corresponds to the sinogram value at the given location
    % It is analogous to the line paths for the scatter points with the difference that one of the detector positions is interpreted as the
    % scatter point
    ActivityMap(ActivityMap < 1e-5) = 0;

    %%__________________________________________________________________________________________________________________
    %% Main part mask generation
 
    tic;
    parfor Ring1 = 1:NrRings
        SinogramsTemp = zeros(NrDetectors+1,NrDetectors/2, NrRings, 'uint8');
        zDetector1 = Geometry(Ring1,1,3);    % z-coordinate detector 1
    
        % Detector 1
        for Detector1 = 1:AccelerationFactor:NrDetectors
            xDetector1 = Geometry(Ring1,Detector1,1);                 % x-coordinate detector 1
            yDetector1 = Geometry(Ring1,Detector1,2);                 % y-coordinate detector 1
    
            for Ring2 = 1:NrRings
                zDetector2 = Geometry(Ring2,1,3);    % z-coordinate detector 2
                
                % Allowed ring difference
                if abs(Ring2 - Ring1) <= NrRings
                    for Detector2 = 1:AccelerationFactor:NrDetectors
                        % Check if LOR is possible (the first ring corresponds to the smallest id of the detector)
                        if (Detector1 == Detector2 && Ring2 == Ring1) || (Detector1 > Detector2)
                            % skip, LOR is not possible
                            continue
                        else
                            xDetector2 = Geometry(Ring2,Detector2,1);                 % x-coordinate detector 2
                            yDetector2 = Geometry(Ring2,Detector2,2);                 % y-coordinate detector 2
                            
                            % Look up at which position in scatter sinogram the probability should be added
                            AngularIndex = SinogramCoordinates(Detector1,Detector2,1);
                            RadialIndex = SinogramCoordinates(Detector1,Detector2,2);
    
                            LineCoordinates = [xDetector1 yDetector1 zDetector1 xDetector2 yDetector2 zDetector2];
                            [Lenghts, Indexes, ~] = RayTracing3DTOF(GridSize,GridBounds,LineCoordinates);
                            if ~isempty(Lenghts)
                                Activity = sum(ActivityMap(Indexes).*Lenghts);
                                Attenuation = 1/exp(-sum(AttenuationMap(Indexes).*Lenghts));
                            else
                                Activity = 0;
                                Attenuation = 1;
                            end
                            
                            if Attenuation ~= 1
                                Attenuation = 0;
                            end
                            
                            if Activity > 0
                                Activity = 0;
                            else
                                Activity = 1;
                            end
                            
                            if UseAttenuation
                                SinogramsTemp(RadialIndex,AngularIndex,Ring2) = SinogramsTemp(RadialIndex,AngularIndex,Ring2) + uint8(Attenuation);
                            else
                                SinogramsTemp(RadialIndex,AngularIndex,Ring2) = SinogramsTemp(RadialIndex,AngularIndex,Ring2) + uint8(Activity);
                            end
                        end
                    end
                end
            end
        end
        
        Sinograms(:,:,:,Ring1) = Sinograms(:,:,:,Ring1) + SinogramsTemp;
    end
    
    % order of sinograms in the file format used, to convert from two ring subscripts to a linear ring difference index
    SinogramOrder = SinogramIndex(1:NrRings,1:NrRings)';
    [~,SinogramOrder] = sort(SinogramOrder(:));

    Sinograms = reshape(Sinograms, NrDetectors+1,NrDetectors/2,NrSinograms);
    Sinograms = Sinograms(:,:,SinogramOrder);

    toc;
end

%%__________________________________________________________________________________________________________________