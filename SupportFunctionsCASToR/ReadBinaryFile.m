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
%% ReadBinaryFile
% Reads binary file into appropriately sized matrix
% 
% INPUT:    FileName                 - Path to the file to be loaded
%           Dimensions               - Size of the corresponding matrix
%           DataType                 - Data type of the data stored
%           ByteOrder                - Order that bytes are saved
%
% OUTPUT:   ProcessedData            - Loaded data with the correct dimensions 
%
% Script by: 
% Rodrigo JOSE SANTO - UMC Utrecht
%__________________________________________________________________________________________________________________

function ProcessedData = ReadBinaryFile(FileName, Dimensions, DataType, ByteOrder)
    
    xDim = Dimensions(1); yDim = Dimensions(2); zDim = Dimensions(3);
    if length(Dimensions) == 4
        vDim = Dimensions(4);
    else
        vDim = 1;
    end

    NrDataPoints = xDim*yDim*zDim*vDim;

    FileID = fopen(FileName);
    RawData = fread(FileID,NrDataPoints,DataType,ByteOrder);
    fclose(FileID);
    
    if NrDataPoints ~= length(RawData)
        disp("Combination of dimensions and image type do not match the file size");
    end

    ProcessedData = reshape(RawData, Dimensions);
    ProcessedData = squeeze(ProcessedData);
end