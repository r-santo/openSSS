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
# Copyright 2022-2025 all openSSS contributors listed below:
# 
#     --> Rodrigo JOSE SANTO, Andre SALOMON, Hugo DE JONG, Thibaut MERLIN, Simon STUTE, Casper BEIJST,
#         Thitiphat KLINSUWAN, Hamidreza RASHIDY KANAN, Massimiliano COLARIETI-TOSTI, Jeffrey NEELE
# 
# This is openSSS version 1.0

import os
import numpy as np
from PIL import Image # to save image
import subprocess

from openSSS.Support import ReadParameters, NormalizeArray
from openSSS.Utils import PrepareImages, ComputePhisicalDimensions
from openSSS.IO import ProcessInput, CreateDirectories, GeneratePrompts

from openSSS.Geometry import PrepareGeometryAndSinograms
from openSSS.DataReduction import SumLORsCounts
from openSSS.TailsMask import GenerateTailsMask
from openSSS.Interpolation import InterpolateScatters
from openSSS.Scaling import ScaleScattersToPrompts

from openSSS.SingleScatterSimulationTOF import SingleScatterSimulationTOF
from Timber import Headerfile, Datafile

if __name__ == "__main__":

    totalIterations, params = ProcessInput()

    # Load parameters from parameter file
    parameters = ReadParameters(params)
    
    folderImages = parameters['FolderImages']
    folderScatters = parameters['FolderScatters']

    Path2Data = parameters['Path2Data']
    Path2Result = parameters['Path2Result']
    Path2Datafile = parameters['Path2Datafile']

    print(f"Data path is defined as {Path2Data}")
    print(f"Save path is defined as {Path2Result}")
    print(f"Datafile path is defined as {Path2Result}")
    
    # 1) Load scanner information
    # Header file
    DatafileName = parameters['DatafileName']
    ExperimentName = DatafileName[:-7] # filters _df.Cdh from DatafileName
    header = Headerfile.ReadHeaderFile(os.path.join(Path2Datafile, DatafileName))

    ScannerName = header['scanner_name']

    # Preparation to obtain scanner information
    targetFolder = os.path.join(Path2Result, 'Scatters')

    # Create directories to save files to
    CreateDirectories(targetFolder, folderImages, folderScatters, params)

    # Parameters for SSS simulation
    NrRingsSimulated = parameters['NrRingsSimulated']
    NrDetectorsSimulated = parameters['NrDetectorsSimulated']
    SampleStep = np.array([parameters['SampleStep_x'], parameters['SampleStep_y'], parameters['SampleStep_z']])
    AccelerationFactor = parameters['AccelerationFactor']
    
    EnergyThreshold = parameters['EnergyThreshold']
    EnergyResolution = parameters['EnergyResolution']

    TOFRange = parameters['TOFRange']
    TOFbins = parameters['TOFbins']
    TOFResolution = header['tof_resolution']
    
    MRD = parameters['MRD']
    SavePath = os.path.join(targetFolder, folderScatters) # Folder to save the scatter estimation

    # Create Mash and Span according to whether we apply data reduction
    DataReduction = parameters['DataReduction']
    if DataReduction == False:
        Mash = 1
        Span = 1
    else:
        Mash = parameters['AngularMashing']
        Span = parameters['AxialMashing']
    # print(f'Mash: {Mash}, Span: {Span}')

    # Scanner geometry, TOF resolution & Energy resolution with Sinogram indexing - OPTIONAL: Data reduction according to Mash & Span
    Geometry, NormalVectors, DetectorSize, LORCoordinates, SinogramIndex, LookUpTable, extendedGeometry, extendedNormalVectors, SinogramCounts, DetectorShift = \
        PrepareGeometryAndSinograms(ScannerName, Mash, Span, MRD, Path2Data)

    DesiredScale = np.array([parameters['DesiredDimensions_x'], parameters['DesiredDimensions_y'], parameters['DesiredDimensions_z']])
    DeviceSize = (np.max(Geometry, axis=(0,1)) - np.min(Geometry, axis=(0,1)))*10
    DeviceDimensions = (DeviceSize//DesiredScale).astype(int)
    DeviceSize = DeviceDimensions * DesiredScale

    # Import attenuation table and see where to find the row of current attenuation energy
    AttenuationTable = np.load(f'./openSSS/AttenuationTable_AttenuationTable.npy')

    # 2) Prepare activity & attenuation
    # Load activity & attenuation files
    attenuationFile = os.path.join(Path2Datafile, parameters['MuMapName'])
    # Loading updates activityFile
    activityFile = os.path.join(Path2Datafile, f'output-{ExperimentName}-step1_it1.hdr')

    # # Reconstruct step 1 - No scatter corrected image
    if not os.path.isfile(activityFile):
        command = [f'bash ./CASToR_recon.sh {ExperimentName} 1']
        processes = [subprocess.Popen(cmd, shell=True, cwd=Path2Datafile) for cmd in command]
        for proc in processes:
            proc.wait()

    # Crop and downscaled the maps
    ActivityMapDownscaled, AttenuationMapDownscaled, DesiredSize, DesiredDimensions = PrepareImages(activityFile, attenuationFile, [None, None, DeviceSize[2]], DesiredScale)
    ImageSize, FittingSize = ComputePhisicalDimensions(DesiredSize, DeviceSize)
    
    Image.fromarray(NormalizeArray(ActivityMapDownscaled)[:, ActivityMapDownscaled.shape[1] // 2, :]).save(os.path.join(targetFolder, folderImages, 'ActivityMapDownscaled.png'))
    Image.fromarray(NormalizeArray(AttenuationMapDownscaled)[:, AttenuationMapDownscaled.shape[1] // 2, :]).save(os.path.join(targetFolder, folderImages, 'AttenuationMapDownscaled.png'))

    # Generates the tails mask for the fitting
    AttenuationMask = GenerateTailsMask(SavePath, ActivityMapDownscaled, AttenuationMapDownscaled, ImageSize,\
                                        Geometry, extendedGeometry, LORCoordinates, SinogramIndex, AccelerationFactor)
    Image.fromarray(NormalizeArray(AttenuationMask)[:, :, AttenuationMask.shape[2] // 2]).save(f'{targetFolder}/{folderImages}/AttenuationMask.png')


    # Create Prompts
    sumLORCounts = SumLORsCounts(Geometry, LookUpTable, SinogramIndex, Mash, Span, SinogramCounts.shape[0])
    GeneratePrompts(SavePath, Path2Datafile, DatafileName, LORCoordinates, SinogramIndex, LookUpTable, TOFbins, TOFRange, Span, Mash, Shift=DetectorShift)

    # Perform scatter estimation, interpolation, scaling, injection and reconstruction (using CASToR)
    for i in range(totalIterations):
        
        # Outputs the estimated scatters for only the sampled rings and detectors combination
        SampledScatters = SingleScatterSimulationTOF(ActivityMapDownscaled, AttenuationMapDownscaled, ImageSize, Geometry, LORCoordinates,
                                                     NormalVectors, DetectorSize, AttenuationTable, EnergyResolution, EnergyThreshold, NrRingsSimulated,
                                                     NrDetectorsSimulated, SampleStep, TOFResolution, TOFRange, TOFbins, SavePath)
        
        # Interpolates the sampled scatters to the full sinogram space (only outputs summed for all TOF bins, individually are saved to disk)
        InterpolatedScatters = InterpolateScatters(SampledScatters, Geometry.shape[0], NrRingsSimulated, Geometry.shape[1], NrDetectorsSimulated,
                                                   LORCoordinates, SinogramIndex, SavePath, Span > 1)

        # Scales the interpolated scatters (loading them per bin from disk)
        ScaleFactors = ScaleScattersToPrompts(SinogramIndex, LORCoordinates, SavePath,
                                              DesiredDimensions, FittingSize, Geometry, extendedGeometry,
                                              AccelerationFactor, Span > 1, sumLORCounts)
        
        # Inject scatter to CASToR datafile
        Datafile.InjectScatters(Path2Datafile, DatafileName, LORCoordinates, SinogramIndex, SavePath, LookUpTable, Span > 1, TOFRange, mash=Mash, Shift=DetectorShift)

        # Runs the reconstruction
        command = [f'bash ./CASToR_recon.sh {ExperimentName} {i+2}']
        processes = [subprocess.Popen(cmd, shell=True, cwd=Path2Datafile) for cmd in command]
        for proc in processes:
            proc.wait()

        # Loads the new activity map
        activityFile = os.path.join(Path2Datafile, f'output-{ExperimentName}-step{i+2}_it1.hdr')
        ActivityMapDownscaled, _, _, _ = PrepareImages(activityFile, attenuationFile, [None, None, DeviceSize[2]], DesiredScale)
