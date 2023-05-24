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
%% BackProjectionScatterRandFit
% Backprojects the tail-restricted sinograms and performs fitting of scatters to the prompts minus randoms
% 
% INPUT:    PromptSinogram                 - sinograms of measured prompts
%           ScatterSinogram                - sinograms of estimated scatters
%           RandomSinogram                 - sinograms of estimated randoms
%           MaskSinogram                   - sinogram mask representing the scatter tails
%           GridSize                       - voxel dimensions for the backprojected images [x y z] (voxels)
%           ImageSize                      - size for the backprojected image [-x -y -z x y z] (mm)
%           Geometry                       - 3D array with the (x,y,z) positions of the detectors
%           SinogramCoordinates            - array with the sinogram coordinates for detector combinations
%           SinogramIndex                  - array with the order of the sinograms for ring combinations
%           AccelerationFactor             - step for the iteration over the detectors of the rings
%
% OUTPUT:   ScaleFactor                    - fitted scaling factor for the estimated sinograms 
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________

function ScaleFactor = BackProjectionScatterRandFit(PromptSinogram, ScatterSinogram, RandomSinogram, MaskSinogram, GridSize, ImageSize, Geometry, SinogramCoordinates, SinogramIndex, AccelerationFactor)
    if nargin == 9
        AccelerationFactor = 1;
    end
    %%  scanner information, input images, constants and lookup tables for main algorithm
    % define some constants
    NrDetectors = size(Geometry,2);                % number of detectors used
    NrRings = size(Geometry,1);                    % number of rings
    GridBounds = ImageSize;                        % real dimensions for the ray tracing grid
    
    % variable to save the backprojected sinograms 
    ActivityImage = zeros(GridSize, 'single');
    ScatterImage = zeros(GridSize, 'single');
    RandomImage = zeros(GridSize, 'single');
    SensitivityImage = zeros(GridSize, 'single');

    % order of sinograms in the file format used, to convert from a linear ring difference index to two ring subscripts
    SinogramOrder = SinogramIndex(1:NrRings,1:NrRings)';

    % masked sinograms to the tails of the scatter, as calculated previously based on the attenuation/activity maps
    ScattersMasked = single(MaskSinogram >= 1).*ScatterSinogram;
    ScattersMasked = ScattersMasked(:,:,SinogramOrder(:));
    ScattersMasked = reshape(ScattersMasked, size(ScattersMasked, 1), size(ScattersMasked, 2), NrRings, NrRings);
    clearvars ScatterSinogram;

    RandomsMasked = single(MaskSinogram >= 1).*RandomSinogram;
    RandomsMasked = RandomsMasked(:,:,SinogramOrder(:));
    RandomsMasked = reshape(RandomsMasked, size(RandomsMasked, 1), size(RandomsMasked, 2), NrRings, NrRings);
    clearvars RandomSinogram;

    PromptsMasked = single(MaskSinogram >= 1).*single(PromptSinogram);
    PromptsMasked = PromptsMasked(:,:,SinogramOrder(:));
    PromptsMasked = reshape(PromptsMasked, size(PromptsMasked, 1), size(PromptsMasked, 2), NrRings, NrRings);
    clearvars PromptSinogram;
    clearvars MaskSinogram;

    %%__________________________________________________________________________________________________________________
    %% Main part backprojection of sinograms
 
    tic;
    parfor Ring1 = 1:NrRings
        ActivityImageTemp = zeros(GridSize, 'single');
        ScatterImageTemp = zeros(GridSize, 'single');
        RandomImageTemp = zeros(GridSize, 'single');
        SensitivityImageTemp = zeros(GridSize, 'single');
        
        % Access only the necessary sinograms, to optimize memory over the paralell processes
        ScattersTemp = squeeze(ScattersMasked(:,:,:,Ring1));
        RandomsTemp = squeeze(RandomsMasked(:,:,:,Ring1));
        PromptsTemp = squeeze(PromptsMasked(:,:,:,Ring1));
    
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
                        % Check if LOR is possible
                        if (Detector1 == Detector2 && Ring2 == Ring1) || Detector1 > Detector2
                            % skip, LOR is not possible
                            continue
                        else
                            xDetector2 = Geometry(Ring2,Detector2,1);                 % x-coordinate detector 2
                            yDetector2 = Geometry(Ring2,Detector2,2);                 % y-coordinate detector 2
                            
                            % Look up at which position in scatter sinogram the combination of the two detectors is stored
                            AngularIndex = SinogramCoordinates(Detector1,Detector2,1);
                            RadialIndex = SinogramCoordinates(Detector1,Detector2,2);

                            if ScattersTemp(RadialIndex,AngularIndex, Ring2) == 0
                                continue
                            else
                                LineCoordinates = [xDetector1, yDetector1, zDetector1, xDetector2, yDetector2, zDetector2];
                                
                                [Lenghts, Indexes, ~] = RayTracing3DTOF(GridSize,GridBounds,LineCoordinates);
                                if ~isempty(Lenghts)

                                    ActivityImageTemp(Indexes) = ActivityImageTemp(Indexes) + PromptsTemp(RadialIndex,AngularIndex, Ring2).*Lenghts./sum(Lenghts);
                                    ScatterImageTemp(Indexes) = ScatterImageTemp(Indexes) + ScattersTemp(RadialIndex,AngularIndex, Ring2).*Lenghts./sum(Lenghts);
                                    RandomImageTemp(Indexes) = RandomImageTemp(Indexes) + RandomsTemp(RadialIndex,AngularIndex, Ring2).*Lenghts./sum(Lenghts);
                                    SensitivityImageTemp(Indexes) = SensitivityImageTemp(Indexes) + 1;
                                end
                            end
                        end
                    end
                end
            end
        end
        
        ActivityImage = ActivityImage + ActivityImageTemp;
        ScatterImage = ScatterImage + ScatterImageTemp;
        RandomImage = RandomImage + RandomImageTemp;
        SensitivityImage = SensitivityImage + SensitivityImageTemp;
    end
    
    ScatterBackProjected = ScatterImage./SensitivityImage;
    ScatterBackProjected(isnan(ScatterBackProjected)) = 0;

    ActivityBackProjected = ActivityImage./SensitivityImage;
    ActivityBackProjected(isnan(ActivityBackProjected)) = 0;

    RandomsBackProjected = RandomImage./SensitivityImage;
    RandomsBackProjected(isnan(RandomsBackProjected)) = 0;

    TailsBackProjected = ActivityBackProjected - RandomsBackProjected;
    TailsBackProjected(TailsBackProjected < 0) = 0;

    % Fit a single scale factor
    [Fitted, ~] = fit(ScatterBackProjected(:)*1e10, TailsBackProjected(:), 'poly1', 'Lower', [0, 0], 'Upper', [Inf, 0]);
    ScaleFactor = Fitted.p1*1e10;
    
    toc;
end

%%__________________________________________________________________________________________________________________