# This file is part of openSSS.
# 
#     openSSS is free software: you can redistribute it and/or modify it under the
#     terms of the GNU General Public License as published by the Free Software
#     Foundation, either version 3 of the License, or (at your option) any later
#     version.
# 
#     openSSS is distributed in the hope that it will be useful, but WITHOUT ANY
#     WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#     FOR A PARTICULAR PURPOSE.
# 
#     You should have received a copy of the License along with openSSS
# 
# Copyright 2022-2024 all openSSS contributors listed below:
# 
#     --> Rodrigo JOSE SANTO, Andre SALOMON, Hugo DE JONG, Thibaut MERLIN, Simon STUTE, Casper BEIJST,
#         Thitiphat KLINSUWAN, Jeffrey NEELE, Adam HOPKINS, Hamidreza RASHIDY KANAN, Massimiliano COLARIETI-TOSTI
# 
# This is openSSS version 0.2

import os
import shutil
import numpy as np
from PIL import Image # to save image
import argparse # for user to specify path to parameters

# Load support functions
from functions import read_parameters, CropAndDownscale, SinogramCoordinates, SinogramToSpatial, normalize_array, MaskGenerator
from SingleScatterSimulationTOF import SingleScatterSimulationTOF

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the Single Scatter Simulation with TOF and specify the parameters path.")
    parser.add_argument('--params', type=str, required=True, help='Path where to read setting parameters')
    args = parser.parse_args()

    # Load Parameters from parameter files
    parameters = read_parameters(args.params)
    Path2Data = parameters['Path2Data']
    Path2Result = parameters['Path2Result']
    print(f"Data path is defined as {Path2Data}")
    print(f"Save path is defined as {Path2Result}")
    LoadFiles = parameters['LoadFiles']

    # Load the Scanner information
    ScannerName = parameters['ScannerName']
    scanners_path = f'{Path2Data}/Scanners'
    images_path = f'{Path2Data}/Images'

    Geometry = np.load(f'{scanners_path}/{ScannerName}_Geometry.npy')
    NormalVectors = np.load(f'{scanners_path}/{ScannerName}_NormalVectors.npy')

    NrSectorsAxial = np.load(f'{scanners_path}/{ScannerName}_NrSectorsAxial.npy')[0][0].astype(int)
    NrSectorsTrans = np.load(f'{scanners_path}/{ScannerName}_NrSectorsTrans.npy')[0][0].astype(int)
    NrModulesAxial = np.load(f'{scanners_path}/{ScannerName}_NrModulesAxial.npy')[0][0].astype(int)
    NrModulesTrans = np.load(f'{scanners_path}/{ScannerName}_NrModulesTrans.npy')[0][0].astype(int)
    NrCrystalsAxial = np.load(f'{scanners_path}/{ScannerName}_NrCrystalsAxial.npy')[0][0].astype(int)
    NrCrystalsTrans = np.load(f'{scanners_path}/{ScannerName}_NrCrystalsTrans.npy')[0][0].astype(int)

    DetectorSize = np.load(f'{scanners_path}/{ScannerName}_DetectorSize.npy')[0]
    EnergyResolution = np.load(f'{scanners_path}/{ScannerName}_EnergyResolution.npy')[0][0].astype(float)
    TOFResolution = np.load(f'{scanners_path}/{ScannerName}_TOFResolution.npy')[0][0].astype(int)

    # Create directory to store the result
    if not os.path.exists(os.path.join(Path2Result, ScannerName, 'Images')):
        os.makedirs(os.path.join(Path2Result, ScannerName, 'Images'))
    if not os.path.exists(os.path.join(Path2Result, ScannerName, 'Scatters')):
        os.makedirs(os.path.join(Path2Result, ScannerName, 'Scatters'))
    
    # Automatically save parameter to result path
    destination_path = os.path.join(Path2Result, ScannerName, args.params)
    try:
        shutil.copy(args.params, destination_path)
        print(f"Using Parameters is saved to {Path2Result}/{ScannerName}")
    except IOError as e:
        print(f"Unable to copy file. {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    ## Read the images
    # This includes both the actual image and the corresponding voxel size, given in mm
    AttenuationMap = np.load(f'{images_path}/AttenuationImage_AttenuationMap.npy')
    AttenuationSize = np.load(f'{images_path}/AttenuationImage_AttenuationSize.npy')[0]
    ActivityMap = np.load(f'{images_path}/ActivityImage_ActivityMap.npy')
    ActivitySize = np.load(f'{images_path}/ActivityImage_ActivitySize.npy')[0]

    # Import attenuation table and see where to find the row of current attenuation energy
    AttenuationTable = np.load(f'{Path2Data}/AttenuationTable_AttenuationTable.npy')

    DeviceSize = [
        (np.max(Geometry[:, :, 0]) - np.min(Geometry[:, :, 0])) * 10,
        (np.max(Geometry[:, :, 1]) - np.min(Geometry[:, :, 1])) * 10,
        (np.max(Geometry[:, :, 2]) - np.min(Geometry[:, :, 2])) * 10
        ]

    NrCrystals = NrSectorsTrans * NrModulesTrans* NrCrystalsTrans

    # It is possible to crop and downscale the images. This is recommended to avoid
    # running out of memmory and crashing the computer
    # Units in mm
    DesiredDimensions = np.array([parameters['DesiredDimensions_x'], parameters['DesiredDimensions_y'], parameters['DesiredDimensions_z']])
    DesiredSize = np.array([ActivityMap.shape[0]*ActivitySize[0], 
                ActivityMap.shape[0]*ActivitySize[1], 
                DeviceSize[2]])

    # Coordinates for the bounds of the image to be used to estimate scatters
    # in the format [xStart, yStart, zStart, xEnd, yEnd, zEnd] and in cm
    ImageSize = np.array([-DesiredSize[0]/2, -DesiredSize[1]/2, -DesiredSize[2]/2,
                DesiredSize[0]/2, DesiredSize[1]/2, DesiredSize[2]/2])

    ImageSize = np.array([x / 10 for x in ImageSize])  # Convert to cm

    # Parameter for SSS simulation
    NrRingsSimulated = parameters['NrRingsSimulated']
    NrDetectorsSimulated = parameters['NrDetectorsSimulated']
    SampleStep = np.array([parameters['SampleStep_x'], parameters['SampleStep_y'], parameters['SampleStep_z']])
    AccelerationFactor = parameters['AccelerationFactor']
    TOFRange = parameters['TOFRange']
    EnergyThreshold = parameters['EnergyThreshold']
    TOFbins = parameters['TOFbins']
    SavePath = os.path.join(Path2Result, ScannerName, 'Scatters')
    
    LORCoordinates, SinogramIndex = SinogramCoordinates(NrSectorsTrans, NrSectorsAxial, NrModulesAxial, NrModulesTrans, NrCrystalsTrans, NrCrystalsAxial)
    DetectorCoordinates, RingCoordinates = SinogramToSpatial(NrSectorsTrans, NrSectorsAxial, NrModulesAxial, NrModulesTrans, NrCrystalsTrans, NrCrystalsAxial, Geometry)

    AttenuationMapDownscaled = CropAndDownscale(AttenuationMap, AttenuationSize, DesiredSize, DesiredDimensions, True, 'edge', True, True, 3)
    # Very low attenuation values (such as air) do not influence the SSS significantly, so can be skipped by making them 0 (no attenuation)
    AttenuationMapDownscaled[AttenuationMapDownscaled < 0.001] = 0
    ActivityMapDownscaled = CropAndDownscale(ActivityMap, ActivitySize, DesiredSize, DesiredDimensions, True, 'edge', True, True, 3)

    # Save Attenuation and Activity Downscaled images
    Image.fromarray(normalize_array(ActivityMapDownscaled)[:, :, ActivityMapDownscaled.shape[2] // 2]).save(f'{Path2Result}/{ScannerName}/Images/ActivityMapDownscaled.png')
    Image.fromarray(normalize_array(AttenuationMapDownscaled)[:, :, AttenuationMapDownscaled.shape[2] // 2]).save(f'{Path2Result}/{ScannerName}/Images/AttenuationMapDownscaled.png')
    
    if not LoadFiles:
        # Generate Attenuation Mask & Interpolated scatters files
        print('Start generating Attenuation mask & Interpolated scatters')

        # Generate Attenuation Mask
        AttenuationMask = MaskGenerator(ActivityMapDownscaled, AttenuationMapDownscaled, ImageSize, Geometry, LORCoordinates, SinogramIndex, True, AccelerationFactor)
        Image.fromarray(normalize_array(AttenuationMask)[:, :, AttenuationMask.shape[2] // 2]).save(f'{Path2Result}/{ScannerName}/Images/AttenuationMask.png')
        np.save(f'{SavePath}/AttenuationMask.npy', AttenuationMask)
        print('completed!\n')
        
        InterpolatedScatters = SingleScatterSimulationTOF(ActivityMapDownscaled, AttenuationMapDownscaled, ImageSize, Geometry, LORCoordinates, SinogramIndex,
                                NormalVectors, DetectorSize, AttenuationTable, EnergyResolution, EnergyThreshold, NrRingsSimulated,
                                NrDetectorsSimulated, SampleStep, TOFResolution, TOFRange, TOFbins, SavePath)
        print('completed!\n')

        Image.fromarray(normalize_array(InterpolatedScatters)[:, :, InterpolatedScatters.shape[2] // 2]).save(f'{Path2Result}/{ScannerName}/Images/Interpolated_Scatters.png')

        # save interpolated scatter for further usage
        np.save(f'{SavePath}/Interpolated_Scatters.npy', InterpolatedScatters)
        
    else:
        # Load Attenuation mask & Interpolated scatter files
        print('Load files Attenuation mask & Interpolated scatters')

        AttenuationMask = np.load(f'{SavePath}/AttenuationMask.npy')
        print('AttenuationMask loaded')
        InterpolatedScatters = np.load(f'{SavePath}/Interpolated_Scatters.npy')
        print('Interpolated Scatters loaded')