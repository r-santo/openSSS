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
%% CropAndDownscale
% Crops image and downscales it
% 
% INPUT:    ImageMap                 - original image to be cropped and downscaled
%           VoxelSize                - size of the voxels in the image [xLenght, yLenght, zLenght] (mm)
%           CroppedSize              - desired size for the cropped image [xLenght, yLenght, zLenght] (mm)
%           DownscaledDimensions     - dimensions for downscaled image [xDim, yDim, zDim]
%
% OUTPUT:   DownscaledImage          - fitted scaling factor for the estimated sinograms 
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________


function DownscaledImage = CropAndDownscale(ImageMap, VoxelSize, CroppedSize, DownscaledDimensions)
    PixelHalfThickness = ceil(CroppedSize ./ VoxelSize / 2);
    
    Center = size(ImageMap) / 2;
    DownscaledImage = ImageMap;

    if Center(1) > PixelHalfThickness(1)
        fprintf("Cropping X\n")
        DownscaledImage = DownscaledImage(Center(1) - PixelHalfThickness(1) : Center(1) + PixelHalfThickness(1),:,:);
    elseif Center(1) < PixelHalfThickness(1)
        fprintf("Padding X\n")
        DownscaledImage = padarray(DownscaledImage, [PixelHalfThickness(1) - Center(1), 0, 0], 0, 'both');
    end

    if Center(2) > PixelHalfThickness(2)
        fprintf("Cropping Y\n")
        DownscaledImage = DownscaledImage(:,Center(2) - PixelHalfThickness(2) : Center(2) + PixelHalfThickness(2),:);
    elseif Center(2) < PixelHalfThickness(2)
        fprintf("Padding Y\n")
        DownscaledImage = padarray(DownscaledImage, [0, PixelHalfThickness(2) - Center(2), 0], 0, 'both');
    end

    if Center(3) > PixelHalfThickness(3)
        fprintf("Cropping Z\n")
        DownscaledImage = DownscaledImage(:,:,Center(3) - PixelHalfThickness(3) : Center(3) + PixelHalfThickness(3));
    elseif Center(3) < PixelHalfThickness(3)
        fprintf("Padding Z\n")
        DownscaledImage = padarray(DownscaledImage, [0, 0, PixelHalfThickness(3) - Center(3)], 0, 'both');
    end
    
    DownscaledImage = imresize3(DownscaledImage, DownscaledDimensions);
end