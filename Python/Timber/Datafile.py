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

import numpy as np
import os

from tqdm import tqdm
import struct

from Timber import Coincidence, Headerfile

def ExportPrompts(
        DataPath : str, 
        DataHeader : str, 
        LORCoordinates : np.ndarray, 
        SinogramIndex : np.ndarray,  
        TOF_bins : int, 
        TOF_range : float = None, 
        mash : int = 1, 
        Shift : int = 0
        )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads the CASToR datafile and exports the prompts, randoms and normalization sinograms in the openSSS format 
    
    Parameters:
    - DataPath (str) : Path to directory where the CASToR datafile is
    - DataHeader (str) : Filename of the CASToR datafile header
    - LORCoordinates (ndarray): Sinogram coordinates for every detector combination.
    - SinogramIndex (ndarray): Sinogram coordinates for every ring combination.
    - TOF_bins (int) : Number of TOF bins
    - TOF_range (float) : Range of TOF values
    - mash (int) : Level of detector mashing (transaxial detector combination)
    - Shift (int) : Shift of crystal index in case the first crystal does not start at the beginning of the first module
    
    Returns:
    - promptsSinogram (ndarray) : Prompts in the sinogram format used by openSSS
    - randomsSinogram (ndarray) : Randoms correction factor in the sinogram format used by openSSS
    - normalizationSinogram (ndarray) : Normalization correction factors in the sinogram format used by openSSS
    """

    header = Headerfile.ReadHeaderFile(os.path.join(DataPath, DataHeader))

    NrCrystalsPerRing = LORCoordinates.shape[0] * mash
    event_struct, event_pos = Coincidence.GenerateEventClass(header)

    # Check datafile size
    data_size = os.path.getsize(os.path.join(DataPath, f'{header["data_filename"]}'))
    if data_size / struct.calcsize(event_struct) != header["nb_events"]:
        raise ValueError('The number of events in the file is not consistent with the header')
    
    if TOF_range is None:
        TOF_range = header["tof_range"]

    sinogramSize = [np.max(LORCoordinates[:,:,1])+1, np.max(LORCoordinates[:,:,0])+1, np.max(SinogramIndex)+1]
    promptsSinogram = np.zeros(sinogramSize + [TOF_bins], dtype=np.uint16)

    scan_duration = header["duration"]
    randomsSinogram = np.zeros(sinogramSize, dtype=np.single) if header["random_flag"] else None
    normalizationSinogram = np.ones(sinogramSize, dtype=np.single) if header["normalization_flag"] else None

    print(f'Sinogram size is: {promptsSinogram.shape}')
    print(f'Now processing prompts {"and randoms " if header["random_flag"] else ""}{"and norm " if header["normalization_flag"] else ""}...')

    nb_events = header["nb_events"]
    
    pbar = tqdm(total=nb_events, desc="Processing events", mininterval=2)
    with open(os.path.join(DataPath, f'{header["data_filename"]}'), 'rb') as file:
        for event in struct.iter_unpack(event_struct, file.read()):
            pbar.update()

            detector1, ring1, detector2, ring2, TOF = Coincidence.ParseCASToRIDs(event, event_pos, NrCrystalsPerRing, Shift)

            radialCoordinate = LORCoordinates[detector1 // mash, detector2 // mash, 0]
            angularCoordinate = LORCoordinates[detector1 // mash, detector2 // mash, 1]
            sinogram = SinogramIndex[ring1, ring2]

            if sinogram == -1 or radialCoordinate == -1 or angularCoordinate == -1:
                continue

            if header["random_flag"]:
                randomsSinogram[angularCoordinate, radialCoordinate, sinogram] = event[0]*scan_duration/TOF_bins

            if header["normalization_flag"]:
                normalizationSinogram[angularCoordinate, radialCoordinate, sinogram] = event[1]

            if abs(TOF) > TOF_range:
                continue

            time_bin = (TOF + TOF_range) / (2*TOF_range / TOF_bins)
            if time_bin < 0 or time_bin >= TOF_bins:
                continue
            else:
                time_bin = int(time_bin)

            promptsSinogram[angularCoordinate, radialCoordinate, sinogram, time_bin]+=1

    pbar.close()
    return promptsSinogram, randomsSinogram, normalizationSinogram

def InjectScatters(
        DataPath : str, 
        DataHeader : str, 
        LORCoordinates : np.ndarray, 
        SinogramIndex : np.ndarray, 
        ScatterPath : str, 
        LookUpTable : np.ndarray,  
        SpanFlag : bool, 
        TOF_range : float = None, 
        NewPath : str = None, 
        mash : int = 1, 
        Shift : int = 0
        ):
    """
    Reads the CASToR datafile and injects the estimated scatters in a new datafile 
    
    Parameters:
    - DataPath (str) : Path to directory where the CASToR datafile is
    - DataHeader (str) : Filename of the CASToR datafile header
    - LORCoordinates (ndarray) : Sinogram coordinates for every detector combination.
    - SinogramIndex (ndarray) : Sinogram coordinates for every ring combination.
    - ScatterPath (str) : Path to directory where the scatter correction factors have been saved to
    - LookUpTable (np.ndarray) : Sinogram coordinates for every ring combination for when span (axial compression) is utilized
    - SpanFlag (bool) : Flag to indicate if span has been applied (axial compression)
    - TOF_range (float) : Range of TOF values
    - NewPath (str) : Path to directory where to save the new datafile
    - mash (int) : Level of detector mashing (transaxial detector combination)
    - Shift (int) : Shift of crystal index in case the first crystal does not start at the beginning of the first module

    """

    # speed of light in mm/ps
    SPEED_OF_LIGHT_IN_MM_PER_PS = 0.299792458
    scaling_factor = 1./(SPEED_OF_LIGHT_IN_MM_PER_PS/2)

    header = Headerfile.ReadHeaderFile(os.path.join(DataPath, DataHeader))
    scan_duration = header["duration"]

    NrCrystalsPerRing = LORCoordinates.shape[0] * mash
    Scale = np.load(ScatterPath + '/ScaleFactors.npy')
    TOF_bins = Scale.shape[0]

    event_struct, event_pos = Coincidence.GenerateEventClass(header)

    # Check datafile size
    data_size = os.path.getsize(os.path.join(DataPath, f'{header["data_filename"]}'))
    if data_size / struct.calcsize(event_struct) != header["nb_events"]:
        raise ValueError('The number of events in the file is not consistent with the header')
    
    new_header = header.copy()
    new_header["scatter_flag"] = True

    original_datafile_name = header["data_filename"]
    new_datafile_name = original_datafile_name[:-6] + 'scatter_' + original_datafile_name[-6:]
    new_header_name = DataHeader[:-6] + 'scatter_' + DataHeader[-6:]
    new_header["data_filename"] = new_datafile_name
    scatter_event_struct, scatter_event_pos = Coincidence.GenerateEventClass(new_header)
    
    if TOF_range is None:
        TOF_range = header["tof_range"]
    else:
        new_header["tof_range"] = TOF_range
    if NewPath is None:
        NewPath = DataPath

    # This is required for CASToR when using list-mode data
    Scale /= (2 * TOF_range / TOF_bins)

    if SpanFlag: # Span applied
        sinogramSize = [np.max(LORCoordinates[:,:,1])+1, np.max(LORCoordinates[:,:,0])+1, SinogramIndex.shape[0]]
    else: # No Span
        sinogramSize = [np.max(LORCoordinates[:,:,1])+1, np.max(LORCoordinates[:,:,0])+1, int(np.max(SinogramIndex))+1]
    scatter_sinograms = np.zeros([TOF_bins] + sinogramSize, dtype=np.float32)
 
    print('Now loading scatters...')
    for i in tqdm(range(TOF_bins)):
        scatter_sinograms[i,...] = np.load(ScatterPath + f'/SSS_mashed_bin{i}.npz')['arr_0']

    nb_events = header["nb_events"]
    nb_events_filtered = 0

    print('Now injecting scatters...')
    pbar = tqdm(total=nb_events, desc="Processing events", mininterval=2)
    fileID = open(os.path.join(NewPath, f'{new_header["data_filename"]}'),'wb')
    with open(os.path.join(DataPath , f'{header["data_filename"]}'), 'rb') as file:
        for event in struct.iter_unpack(event_struct, file.read()):
            pbar.update()

            detector1, ring1, detector2, ring2, TOF = Coincidence.ParseCASToRIDs(event, event_pos, NrCrystalsPerRing, Shift)
            if abs(TOF) > TOF_range:
                continue

            radialCoordinate = LORCoordinates[detector1 // mash, detector2 // mash, 0]
            angularCoordinate = LORCoordinates[detector1 // mash, detector2 // mash, 1]

            if SinogramIndex.shape[0] == SinogramIndex.shape[1]:
                sinogram = SinogramIndex[ring1, ring2]
            else:
                sinogram = LookUpTable[ring1, ring2]


            if sinogram == -1 or radialCoordinate == -1 or angularCoordinate == -1:
                continue

            time_bin = (TOF + TOF_range) / (2*TOF_range / TOF_bins)
            if time_bin < 0 or time_bin >= TOF_bins:
                continue
            else:
                time_bin = int(time_bin)
                factor = Scale[time_bin]*scaling_factor/scan_duration#/getattr(event, 'normalization') #/ sumLORCounts[angularCoordinate,radialCoordinate,sinogram]
                scatter_rate = factor*scatter_sinograms[time_bin, angularCoordinate, radialCoordinate, sinogram]

                event_destination = struct.pack(scatter_event_struct, *event[:scatter_event_pos[0]],\
                                                scatter_rate, *event[scatter_event_pos[0]:])
                fileID.write(event_destination)

            nb_events_filtered += 1

    fileID.close()
    pbar.close()

    new_header["nb_events"] = nb_events_filtered
    Headerfile.WriteHeaderFile(new_header, os.path.join(NewPath,new_header_name))

    return