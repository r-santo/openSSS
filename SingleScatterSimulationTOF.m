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
%% 3D Single Scatter Simulation (SSS)
% Performs TOF scatter estimation based on the SSS algorithm by Watson
%
% INPUT:    ActivityMap                 - activity maps estimation
%           AttenuationMap              - attenuation map
%           ImageSize                   - attenuation and activity image size [-x -y -z x y z] (mm)
%           Geometry                    - 3D array with the (x,y,z) positions of the detectors [rings, detectors, coordinates(x,y,z)]
%           SinogramCoordinates         - array with the sinogram coordinates for detector combinations
%           SinogramIndex               - array with the order of the sinograms for ring combinations
%           NormalVectors               - vectors normal to the surface area of detectors
%           DetectorSize                - size of the surface of the detectors [x, y] (mm)
%           EnergyResolution            - energy resolution (%)
%           EnergyThreshold             - energy threshold (keV)
%           NrRingsUsed                 - number of rins to use for the scatter estimation before interpolation
%           NrDetectorsUsed             - number of detectors to use for the scatter estimation before interpolation
%           SampleStep                  - step size for the sample of scatter points [x, y, z] in voxels
%           TOFResolution               - time resolution (ps)
%           TOFRange                    - range of TOF measured (ps)
%           NrBins                      - number of TOF bins
%           SavePath                    - path where to save the sinograms
%
% OUTPUT:   Scatters                   - estimated scatters summed scatter
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________
function Scatters = SingleScatterSimulationTOF(ActivityMap, AttenuationMap, ImageSize,...
                                               Geometry, SinogramCoordinates, SinogramIndex, NormalVectors, DetectorSize,...
                                               EnergyResolution, EnergyThreshold,...
                                               NrRingsUsed, NrDetectorsUsed, SampleStep, ...
                                               TOFResolution, TOFRange, NrBins,...
                                               SavePath)
    %%  scanner information, input images, constants and lookup tables for main algorithm
    % define some constants
    [xDim, yDim, zDim] = size(AttenuationMap);      % dimensions of the contour image
    NrDetectors = size(Geometry,2);                    % number of detectors used
    NrRings = size(Geometry,1);                        % number of rings

    % the crop for the attenuation and activity map expands half the detector height for each side of the FOV
    zVoxelSize = (ImageSize(6) - ImageSize(3))/zDim;     % real distance between middle two slices in contour_image
    
    EnergyReferenceJoule = 511.0E3 * 1.6E-19;                     % photon energy (J)
    SmallNumber = 10^(-15);                            % used to prevent dividing by zero
    EnergyReference = 511.0;                             % energy used for generation of attenuation map (keV)
    GridSize = size(ActivityMap);                   % dimensions for the ray tracing grid
    GridBounds = ImageSize;                        % real dimensions of the ray tracing grid

    % TOF characteristics
    BinWidth = 2*TOFRange/NrBins;                % width of the bin (ps)
    
    % import attenuation table and see where to find the row of current attenuation energy
    load('./Data/AttenuationTable.mat', 'AttenuationTable');
    EnergyIndex = find(AttenuationTable(:,1) == EnergyReference);
    
    % assign each voxel in the attenuation map an index to know what tissue it
    % is, and how to convert the attenuation coefficient for ray 2
    AttenuationTissue = zeros(xDim,yDim,zDim);
    for zIndex = 1:zDim
        for xIndex = 1:xDim
            for yIndex = 1:yDim
                if AttenuationMap(xIndex,yIndex,zIndex) > 0
                    j = 0;
                    CurrentMinimum = 10^15;
                    for i = 2:size(AttenuationTable,2)
                        CurrentDifference = abs(AttenuationTable(EnergyIndex,i) - AttenuationMap(xIndex,yIndex,zIndex));
                        if CurrentDifference < CurrentMinimum
                            CurrentMinimum = CurrentDifference;
                            j = i;
                        end
                        AttenuationTissue(xIndex,yIndex,zIndex) = j;
                    end
                end
            end
        end
    end
    
    % Compress the attenuation_table, so that it just includes the energy for 511kEV and the ratios for water of all other energies
    AttenuationRatios = zeros(size(AttenuationTable,1), 2);
    AttenuationRatios(:,1) = AttenuationTable(:,1);
    AttenuationRatios(:,2) = AttenuationTable(:,2)./AttenuationTable(EnergyIndex,2);
    
    Tissue = unique(AttenuationTissue);
    Tissue = Tissue(Tissue ~= 0);
    for i = 1:length(Tissue)
        AttenuationTissue(AttenuationTissue == Tissue(i)) = i+1;
    end
    AttenuationTable = AttenuationTable(EnergyIndex,[1;Tissue]); %To be complient with previous implementation (first column is energy)
    
    % calculate detector efficiency
    EfficiencyTable = AttenuationRatios(:,1); % For the different kinds of energy allowed
    EfficiencyTable(:,2) = 0.5*(1-erf((EnergyThreshold-EfficiencyTable(:,1))/(EnergyReference*EnergyResolution/2/sqrt(log(2)))));
    
    % calculate TOF efficiencency
    LightSpeed = 299792458.0*1e-10;            % speed of light (cm/ps)
    TimeRange = 4000;               % how wide the kernel is (ps)
    TOFTable = TOFEfficiencyTable(-TimeRange:TimeRange, BinWidth, NrBins, TOFResolution);

    % define which rings will be used
    Rings = round(linspace(1,NrRings,NrRingsUsed));  % defines which rings are used
    
    % calculate the number of sinograms that will be obtained (oblique + non-oblique)
    NrSinograms = NrRingsUsed^2;
    
    % define which detectors will be used per ring
    DetectorDifference = NrDetectors/NrDetectorsUsed;            % difference in detector index between two detectors
    Detectors = zeros(NrRings,NrDetectorsUsed);     % structure to save which detectors are used per ring
    Detector1 = 0;
    for RingIndex1 = 1:NrRings             % this loop defines which detectors are used per ring
        for d = 1:NrDetectorsUsed
            if d == 1
                Detectors(RingIndex1,d) = d + Detector1;
            else
                Detectors(RingIndex1,d) = floor(Detectors(RingIndex1,d-1) + DetectorDifference);
            end
        end
    end
    
    % structure to save the sinograms in the format used in the rest of the PET/MRI reconstruction pipeline
    Scatters = zeros(NrBins, NrDetectors+1,NrDetectors/2,NrSinograms, 'single');
    
    %%__________________________________________________________________________________________________________________
    %% Main part SSS
    
    Width = - ImageSize(3);
    Radius = - ImageSize(1);
    % The z-coordinate in the cropped maps starts half width before the FOV
    zStart = zVoxelSize/2 - Width;     % z-coordinate of first slice
    
    tic;
    % loop over all possible scatter points
    zSamplePoints = 1:SampleStep(3):zDim;
    parfor zSampleIndex = 1:numel(zSamplePoints)          % parellel for-loop to make algorithm faster
        zIndex = zSamplePoints(zSampleIndex);
        zScatterPoint = zStart + (zIndex-1)*zVoxelSize;    % z-coordinate scatter point
        
        % structure to save the sinograms of each slice
        ScatterSlice = zeros(NrBins, NrDetectors+1,NrDetectors/2,NrSinograms, 'single');
        
        % structure to save how many times a LOR is used
        ScatterCounts = zeros(NrBins, NrDetectors+1,NrDetectors/2,NrSinograms, 'single');
        
        for yIndex = 1:SampleStep(2):yDim
            if sum(AttenuationTissue(:,yIndex,zIndex)) ~= 0  % to check if any of the indices is a scatter point
               
                yScatterPoint = (yIndex * 2. * Radius / yDim) - Radius;   % y-coordinate scatter point
    
                for xIndex = 1:SampleStep(1):xDim
    
                    % Check if it is a scatter point
                    if AttenuationTissue(xIndex,yIndex,zIndex) > 0
    
                        xScatterPoint = (xIndex * 2. * Radius / xDim) - Radius;   % x-coordinate scatter point
                        LinePaths = zeros(NrRingsUsed, NrDetectorsUsed, 2);
                        LineDistributions = cell(NrRings, NrDetectorsUsed, 2);
                        Angles = zeros(NrRingsUsed, NrDetectorsUsed);
                        
                        % loop over all LORs
                        
                        % Ring detector 1
                        for RingIndex1 = 1:NrRingsUsed
                            Ring1 = Rings(RingIndex1);
                            zDetector1 = Geometry(Ring1,1,3);    % z-coordinate detector 1
                            
                            % Detector 1
                            for DetectorIndex1 = 1:NrDetectorsUsed
                                Detector1 = Detectors(Ring1,DetectorIndex1);                                % detector index detector 1
                                xDetector1 = Geometry(Ring1,Detector1,1);                 % x-coordinate detector 1
                                yDetector1 = Geometry(Ring1,Detector1,2);                 % y-coordinate detector 1
                                
                                LineCoordinates = [xScatterPoint yScatterPoint zScatterPoint xDetector1 yDetector1 zDetector1];
                                [Lenghts, Indexes, Rays] = RayTracing3DTOF(GridSize,GridBounds,LineCoordinates);
                                ActivityRay = ActivityMap(Indexes).*Lenghts;
                                ActivityIntegral = sum(ActivityRay);
                                AttenuationIntegral = exp(-(sum(AttenuationMap(Indexes).*Lenghts)));
                                
                                LinePaths(RingIndex1, DetectorIndex1, :) = [AttenuationIntegral, ActivityIntegral];
                                LineDistributions{RingIndex1, DetectorIndex1, 1} = ActivityRay;
                                LineDistributions{RingIndex1, DetectorIndex1, 2} = Rays;
                                
                                ScatterVector = [xDetector1-xScatterPoint, yDetector1-yScatterPoint, zDetector1-zScatterPoint];
                                Angles(RingIndex1,DetectorIndex1) = abs(acosd(dot(ScatterVector, squeeze(NormalVectors(Ring1, Detector1,:)))/norm(squeeze(NormalVectors(Ring1, Detector1,:)))/norm(ScatterVector)));
                            end
                        end
                        
                        % Mix paths to get LORs
                        for RingIndex1 = 1:NrRingsUsed
                            Ring1 = Rings(RingIndex1);
                            zDetector1 = Geometry(Ring1,1,3);    % z-coordinate detector 1
                            
                            for DetectorIndex1 = 1:NrDetectorsUsed
                                Detector1 = Detectors(Ring1,DetectorIndex1);                                % detector index detector 1
                                xDetector1 = Geometry(Ring1,Detector1,1);                 % x-coordinate detector 1
                                yDetector1 = Geometry(Ring1,Detector1,2);                 % y-coordinate detector 1
                                ScatterVector1 = sqrt((xDetector1-xScatterPoint)^2+(yDetector1-yScatterPoint)^2+(zDetector1-zScatterPoint)^2);   % distance detector 1 to scatter point
                                
                                % Activity and attenuation of unscattered photon in detector 1
                                AttenuationPath1 = LinePaths(RingIndex1, DetectorIndex1, 1);
                                ActivityPath1 = LinePaths(RingIndex1, DetectorIndex1, 2);
                                
                                if AttenuationPath1 == 0
                                    continue
                                end

                                % Activity and sample lenght distributions of unscattered photon in detector 1
                                ActivityRay1 = LineDistributions{RingIndex1,DetectorIndex1,1}';
                                Rays1 = LineDistributions{RingIndex1,DetectorIndex1,2}';
                                
                                for RingIndex2 = 1:NrRingsUsed
                                    Ring2 = Rings(RingIndex2);
                                    zDetector2 = Geometry(Ring2,1,3);    % z-coordinate
    
                                    % Allowed ring difference
                                    if abs(RingIndex2 - RingIndex1) <= NrRingsUsed
                                        
                                        % Detector 2
                                        for DetectorIndex2 = 1:NrDetectorsUsed
                                            Detector2 = Detectors(Ring2,DetectorIndex2);    % detector index
                                            
                                            % Check if LOR is possible
                                            if (Detector1 == Detector2 && RingIndex2 == RingIndex1) || Detector1 > Detector2
                                                % skip, LOR is not possible
                                                continue
                                            else
                                                xDetector2 = Geometry(Ring2,Detector2,1);                 % x-coordinate
                                                yDetector2 = Geometry(Ring2,Detector2,2);                 % y-coordinate
                                                ScatterVector2 = sqrt((xDetector2-xScatterPoint)^2+(yDetector2-yScatterPoint)^2+(zDetector2-zScatterPoint)^2);   % distance detector 2 to scatter point
                                                
                                                % Activity and attenuation of unscattered photon in detector 2
                                                AttenuationPath2 = LinePaths(RingIndex2, DetectorIndex2, 1);
                                                ActivityPath2 = LinePaths(RingIndex2, DetectorIndex2, 2);
                                                
                                                if AttenuationPath2 == 0 || (ActivityPath1 == 0 && ActivityPath2 == 0)
                                                    continue
                                                elseif ActivityPath1 ~= 0 && ActivityPath2 ~= 0
                                                    Counts = 2;
                                                else
                                                    Counts = 1;
                                                end
    
                                                % Look up at which position in scatter sinogram the probability should be added
                                                AngularIndex = SinogramCoordinates(Detector1,Detector2,1);
                                                RadialIndex = SinogramCoordinates(Detector1,Detector2,2);
    
                                                % calculate scatter angle
                                                ScatterAngle = CalcAngleScatter(xScatterPoint,yScatterPoint,zScatterPoint,xDetector1,yDetector1,zDetector1,xDetector2,yDetector2,zDetector2);
    
                                                % calculate energy of scattered photon, detector efficiency and attenuation scalling
                                                EnergyScatter = round(EnergyReference / (1 + (EnergyReference / 511.0)*(1 - cosd(ScatterAngle))));
                                                EnergyScatterIndex = EnergyScatter*2;%find(attenuation_ratios(:,1) == energy_ray2);
    
                                                EnnergyEfficiency = EfficiencyTable(EnergyScatterIndex,2)*EfficiencyTable(EnergyReference*2,2);
                                                if EnnergyEfficiency == 0
                                                    continue
                                                end

                                                AttenuationScale = AttenuationRatios(EnergyScatterIndex,2);
                                                % Attenuation of scattered photon in detector 1
                                                AttenuationScaled1 = AttenuationPath1^AttenuationScale;
                                                % Attenuation of scattered photon in detector 2
                                                AttenuationScaled2 = AttenuationPath2^AttenuationScale;

                                                % TOF detector efficiency
                                                % Defined so that time difference  (time1 - time2) is positive closer to detector 2
                                                % Based on the CASToR interpretation
                                                if ActivityPath1 ~= 0
                                                    SpatialOffset = round(2*((ScatterVector1 - ScatterVector2)/2 - Rays1)/LightSpeed) + TimeRange + 1;
                                                    ActivityBinned1 = TOFTable(:,SpatialOffset);
                                                    ActivityBinned1 = ActivityRay1*ActivityBinned1';
                                                else
                                                    ActivityBinned1 = 0;
                                                end
                                                
                                                if ActivityPath2 ~= 0
                                                    SpatialOffset = round(2*((ScatterVector1 - ScatterVector2)/2 + LineDistributions{RingIndex2,DetectorIndex2,2}')/LightSpeed) + TimeRange + 1;
                                                    ActivityBinned2 = TOFTable(:,SpatialOffset);
                                                    ActivityBinned2 = LineDistributions{RingIndex2,DetectorIndex2,1}'*ActivityBinned2';
                                                else
                                                    ActivityBinned2 = 0;
                                                end
                                                
                                                % geometrical correction value (first component in formula Watson, without cross sections)
                                                GeometricalEfficiency = power(DetectorSize(1)*DetectorSize(2)*1e-2,2)*abs(cosd(Angles(RingIndex1,DetectorIndex1)))*abs(cosd(Angles(RingIndex2,DetectorIndex2))) / (4*pi*(ScatterVector1^2)*(ScatterVector2^2));
    
                                                % calculate probability
                                                Probability = GeometricalEfficiency*EnnergyEfficiency*KleinNishina(EnergyReferenceJoule,ScatterAngle)*(AttenuationPath1*AttenuationScaled2*ActivityBinned1 + AttenuationPath2*ActivityBinned2*AttenuationScaled1);
                                                
                                                % add probability to the corresponding sinogram and add count
                                                ScatterSlice(:, RadialIndex,AngularIndex,RingIndex2+(RingIndex1-1)*NrRingsUsed) = ...
                                                    ScatterSlice(:, RadialIndex,AngularIndex,RingIndex2+(RingIndex1-1)*NrRingsUsed) + Probability';
                                                
                                                ScatterCounts(:, RadialIndex,AngularIndex,RingIndex2+(RingIndex1-1)*NrRingsUsed) = ...
                                                    ScatterCounts(:, RadialIndex,AngularIndex,RingIndex2+(RingIndex1-1)*NrRingsUsed) + Counts;

                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end     
        end
        
        % correct for differences in counts per sinogram point (this may occur due to asymmetric use of detectors in different rings)
        ScatterCounts(ScatterCounts == 0) = SmallNumber;     % prevent dividing by zero
        ScatterSlice = ScatterSlice;% ./ sinograms_counts_per_slice;
        
        % add sinograms of this slice to the other slices
        Scatters = Scatters + ScatterSlice;
        
    end
    
    %%__________________________________________________________________________________________________________________
    %% Postprocessing
    
    toc;
    % interpolate sinogram of current slice
    Scatters = InterpolateAllBins(Scatters, Rings, SinogramIndex, Detectors, SinogramCoordinates, SavePath);
    toc;
end
%%__________________________________________________________________________________________________________________
%% Helper functions
%%__________________________________________________________________________________________________________________
%% TOF detection efficiency function

function Probability = TOFEfficiencyTable(Offset, BinWidth, NrBins, Resolution)
    shifts = ((1:NrBins) - (NrBins+1)/2)*BinWidth;
    Probability = Offset - shifts';
    Probability = exp(-Probability.^2/(Resolution^2/4/log(2)));
    Probability = Probability./sum(Probability,1);
end

%%__________________________________________________________________________________________________________________
%% Klein nishina probability
    
function Probability = KleinNishina(Energy,Angle)
    
    ElectronMass = 9.10938356E-31;       % mass electron (kg)
    LightSpeed = 299792458.0;            % speed of light (m/s)
    
    Gamma = Energy / (ElectronMass * LightSpeed * LightSpeed); % used in Klein-Nishina formula
    Ratio = 1.0 / (1.0 + Gamma * ( 1.0 - cosd(Angle) ));
  
    % Klein-Nishina formula
    Probability = Ratio^2 * (Ratio + 1/Ratio - sind(Angle)^2);

end

%%__________________________________________________________________________________________________________________
%% Scatter angle based on the 3D-coordinates of the detectors and scatter point

function ScatterAngle = CalcAngleScatter(xScatter,yScatter,zScatter,xDetector1,yDetector1,zDetector1,xDetector2,yDetector2,zDetector2)

    % lengths between the three different points
    ScatterVector1 = (((xDetector1-xScatter)^2)+((yDetector1-yScatter)^2)+((zDetector1-zScatter)^2));
    ScatterVector2 = (((xDetector2-xScatter)^2)+((yDetector2-yScatter)^2)+((zDetector2-zScatter)^2));
    LOR = (((xDetector1-xDetector2)^2)+((yDetector1-yDetector2)^2)+((zDetector1-zDetector2)^2));
    
    % scatter angle
    ScatterAngle = 180 - acosd(((ScatterVector1)+(ScatterVector2)-(LOR))/(2*sqrt(ScatterVector1*ScatterVector2)));   % law of cosines
end

%%__________________________________________________________________________________________________________________
%% Interpolates the sinograms, all together

function InterpolatedSinograms = InterpolateAllBins(SinogramsBinned, Rings, SinogramIndex, Detectors, SinogramCoordinates, SavePath)
    NrRingsUsed = length(Rings);
    NrRings = size(SinogramIndex, 1);
    
    InterpolatedSinograms = zeros(size(SinogramsBinned,2), size(SinogramsBinned,3), NrRings^2, 'single');
    for Bin=1:size(SinogramsBinned,1)
        
        SinogramsCurrentBin = squeeze(SinogramsBinned(Bin,:,:,:));
        for i=1:size(SinogramsCurrentBin,3)
            Sinogram = SinogramsCurrentBin(:,:,i);
            Ring1 = Rings(fix((i-1)/NrRingsUsed) + 1);
            Ring2 = Rings(mod((i-1),NrRingsUsed) + 1);

            RadialIndex = SinogramCoordinates(Detectors(Ring1,:), Detectors(Ring2,:),2);
            AngularIndex = SinogramCoordinates(Detectors(Ring1,:), Detectors(Ring2,:),1);
            LinearIndex = sub2ind(size(Sinogram), RadialIndex(:), AngularIndex(:));
            Values = Sinogram(LinearIndex);

            [RadialDim,AngularDim] = size(Sinogram);
            F = scatteredInterpolant(RadialIndex(:),AngularIndex(:),double(Values),'linear');
            SinogramsCurrentBin(:,:,i) = F({1:RadialDim,1:AngularDim});
        end

        % Possible negative values result from extrapolation due to insufficient sampling range for the LORs
        % Ideally the corners of the sinogram should be set but this is difficult because of how the LORs are defined
        % Largest angle for opposing detectors, which my be impossible in this ring setup
        SinogramsCurrentBin(SinogramsCurrentBin < 0) = 0;

        [RadialDim,AngularDim,~] = size(SinogramsCurrentBin);
        SinogramsCurrentBin = reshape(SinogramsCurrentBin, RadialDim, AngularDim, NrRingsUsed, NrRingsUsed);
        SinogramsCurrentBin = permute(SinogramsCurrentBin, [1, 2, 4, 3]);
        % some grid vectors had their orientation changed to warn interpn that they correspond indeed to grid vectors
        SinogramsInerpolatedCurrentBin = single(interpn(1:RadialDim, 1:AngularDim, Rings(:), Rings(:), double(SinogramsCurrentBin), 1:RadialDim, 1:AngularDim, (1:NrRings)', (1:NrRings)', 'linear'));
        SinogramsInerpolatedCurrentBin = reshape(SinogramsInerpolatedCurrentBin, RadialDim, AngularDim, NrRings^2);

        % order in the file format used for the sinograms
        SinogramOrder = SinogramIndex(1:NrRings,1:NrRings)';
        [~,SinogramOrder] = sort(SinogramOrder(:));

        filename = sprintf('%s/SSS_bin%d.bin', SavePath, Bin);
        fileID = fopen(filename,'W');
        fwrite(fileID,SinogramsInerpolatedCurrentBin(:,:,SinogramOrder),'float32');
        fclose(fileID);

        InterpolatedSinograms = InterpolatedSinograms + SinogramsInerpolatedCurrentBin(:,:,SinogramOrder);
    end
end
%%_______________________________________________________________________________________________________________________________________________________________________________________________
%% function that samples scatter points from the attenuation maps
function sampled_points = SamplePoints(cubesize, subdivision)
    subcubesize = cubesize/subdivision;
    [x, y, z] = meshgrid(1:cubesize, 1:cubesize, 1:cubesize);
    xyz = [x(:), y(:), z(:)];

    % Defines to what subcube the samples belong to
    ijk = ceil(xyz/subcubesize);
    subcubes = unique(ijk,'rows');
    sampled_points = zeros(subdivision^3,3);
    for i=1:size(subcubes,1)
        index = find(ijk(:,1) == subcubes(i,1) & ijk(:,2) == subcubes(i,2) & ijk(:,3) == subcubes(i,3));
        sampled_points(i,:) = xyz(randi(length(index)),:);
    end

    n = accumarray(ijk,1,subdivision*ones(1,3));
    density = n/subdivision^3; % #points per m^3 in each of 27 subcubes
end