# This file is part of Timber.
# 
#     Timber is free software: you can redistribute it and/or modify it under the
#     terms of the GNU General Public License as published by the Free Software
#     Foundation, either version 3 of the License, or (at your option) any later
#     version.
# 
#     Timber is distributed in the hope that it will be useful, but WITHOUT ANY
#     WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#     FOR A PARTICULAR PURPOSE.
# 
#     You should have received a copy of the License along with Timber
# 
# Copyright 2025 all Timber contributors listed below:
# 
#     --> Rodrigo JOSE SANTO
# 
# This is Timber version 1.0

# Function to generate event class
def GenerateEventClass(
        DataHeader : dict
        )->tuple[str, tuple]:
    
    """
    Prepares the event type to read from the CASToR datafile
    
    Parameters:
    - DataHeader (dict) : Loaded header of the CASToR datafile
    
    Returns:
    - fields (str) : string of the different value sizes (bytes) depending on the corrections and contributing lines for each event
    - fields_pos (tuple) : index positions of the different event info once each event is loaded
    """

    if 'list' in DataHeader['data_mode']:
        fields, fields_pos = GenerateListEventInfo(DataHeader['attenuation_flag'], DataHeader['scatter_flag'], DataHeader['random_flag'],\
                                                   DataHeader['normalization_flag'], DataHeader['tof_flag'], DataHeader['nb_max_lines'])
    else:
        raise NotImplementedError('Handling of histogram data is not yet implemented')
        
    return fields, fields_pos

# Function to generate list event class
def GenerateListEventInfo(
        AttenuationFlag : bool,
        ScatterFlag : bool,
        RandomFlag : bool,
        NormalizationFlag : bool,
        TOFFlag : bool,
        MaxNbLines : int
        )->tuple[str, tuple]:
    
    """
    Prepares the list-mode event in the structure of the CASToR datafile 
    
    Parameters:
    - AttenuationFlag (bool) : Flag to signal the presence of attenuation correction factors
    - ScatterFlag (bool) : Flag to signal the presence of scatter correction factors
    - RandomFlag (bool) : Flag to signal the presence of random correction factors
    - NormalizationFlag (bool) : Flag to signal the presence of normalization correction factors
    - TOFFlag (bool) : Flag to signal the presence of TOF information
    - MaxNbLines (int) : Maximum number of LORs that contribute to this event
    
    Returns:
    - fields (str) : string of the different value sizes (bytes) depending on the corrections and contributing lines for each event
    - fields_pos (tuple) : index positions of the different event info once each event is loaded
    """

    NbLinesFlag = MaxNbLines > 1

    fields = '<1L' # Coincidence time (miliseconds) - uint32

    # Attenuation correction factor (optional), Scatter correction factor (optional)
    # Random correction factor (optional), Normalization correction factor (optional)
    # Time of flight information (ps) (optional)
    nb_corrections = AttenuationFlag + ScatterFlag + RandomFlag + NormalizationFlag + TOFFlag# Correction factors (floats)
    if nb_corrections > 0:
        fields += f'{nb_corrections}f'

    # Number of contributing lines (optional) - uint16
    if NbLinesFlag:
        fields += '1I'

    fields += f'{2*MaxNbLines}L' # Array of contributing crystals - uint32

    # Information on positions (Random, Normalization, TOF, Crystals)
    array_pos = (1+AttenuationFlag if ScatterFlag else None,\
                 1+AttenuationFlag+ScatterFlag if RandomFlag else None,\
                 1+AttenuationFlag+ScatterFlag+RandomFlag if NormalizationFlag else None,\
                 1+AttenuationFlag+ScatterFlag+RandomFlag+NormalizationFlag if TOFFlag else None,\
                 1+AttenuationFlag+ScatterFlag+RandomFlag+NormalizationFlag+TOFFlag+NbLinesFlag)
    
    return fields, array_pos
    
# Function to parse CASToRIDs into ring/detector pairs
def ParseCASToRIDs(
        event : tuple,
        field_pos : tuple,
        NrCrystalsPerRing : int,
        Shift : int = 0
        )->tuple[int, int, int, int, float]:
    
    """
    Converts the CASToR IDs into detector and ring information, with TOF 
    
    Parameters:
    - event (tuple) : event read from the CASToR datafile
    - field_pos (tuple) : index positions of the different event info once each event is loaded
    - NrCrystalsPerRing (int) : Number of crystals per ring
    - Shift (int) : Shift of crystal index in case the first crystal does not start at the beginning of the first module
    
    Returns:
    - detector1 (int) : Index of the transaxial crystal of the first detector of the corresponding LOR
    - ring1 (int) : Index of the ring of the first detector of the corresponding LOR
    - detector2 (int) : Index of the transaxial crystal of the second detector of the corresponding LOR
    - ring2 (int) : Index of the ring of the second detector of the corresponding LOR
    - TOF (float) : TOF information for the event, in the format time1-time2 (positive closer to detector 2)
    """

    castorIDs = [event[field_pos[4]], event[field_pos[4]+1]]
    TOF_info = event[field_pos[3]]

    ID1 = castorIDs[0] if castorIDs[0] < castorIDs[1] else castorIDs[1]
    ID2 = castorIDs[1] if castorIDs[0] < castorIDs[1] else castorIDs[0]
    TOF_temp = TOF_info if castorIDs[0] < castorIDs[1] else -TOF_info

    # Calculates without making any modification to the detector ID
    detector1_temp = ID1 % NrCrystalsPerRing
    detector2_temp = ID2 % NrCrystalsPerRing
    
    if detector1_temp < detector2_temp:
        detector1 = detector1_temp
        detector2 = detector2_temp
    else:
        detector1 = detector2_temp
        detector2 = detector1_temp
    
    # Takes into account this ID shift situation
    detector1_temp = ID1 % NrCrystalsPerRing - Shift
    detector2_temp = ID2 % NrCrystalsPerRing - Shift
    if detector1_temp < 0:
        detector1_temp += NrCrystalsPerRing
    if detector2_temp < 0:
        detector2_temp += NrCrystalsPerRing

    # determine crystal with smallest transaxial ID (A=smallest, B=largest) and corresponding ringIds
	# to distinguish LOR with ring difference X from LOR with same transaxial positions but with ring difference -X
    if detector1_temp < detector2_temp:
        ring1 = ID1 // NrCrystalsPerRing
        ring2 = ID2 // NrCrystalsPerRing

        TOF = TOF_temp
    else:
        ring1 = ID2 // NrCrystalsPerRing
        ring2 = ID1 // NrCrystalsPerRing

        TOF = -TOF_temp

    return detector1, ring1, detector2, ring2, TOF