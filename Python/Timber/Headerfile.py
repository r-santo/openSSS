import re

# Function to read the datafile header
def ReadHeaderFile(
        HeaderFilePath : str
        )->dict:
    """
    Reads the CASToR header  
    
    Parameters:
    - HeaderFilePath (str) : Path to CASToR datafile header
    
    Returns:
    - data_header (dict) : Dictionary populated with all the keywords of a typical CASToR datafile header
    """

    fileID = open(HeaderFilePath,'r')
    header = fileID.read()
    fileID.close()

    data_filename = re.findall(r'Data filename: (\S*)', header)[0]

    nb_events = int(re.findall(r'Number of events: (\d*)', header)[0])
    data_mode = re.findall(r'Data mode: (\w*[-]?\w*)', header)[0]
    data_type = re.findall(r'Data type: (\w*)', header)[0]

    start_time = float(re.findall(r'Start time \(s\): (\d*[.]?\d*)', header)[0])
    duration = float(re.findall(r'Duration \(s\): (\d*[.]?\d*)', header)[0])

    scanner_name = re.findall(r'Scanner name: (\S*)', header)[0]
    isotope = re.findall(r'Isotope: (\S*)', header)[0]
    calibration_factor = float(re.findall(r'Calibration factor: (\d*[.]?\d*)', header)[0])

    max_lines = re.findall(r'Maximum number of lines per event: (\d*)', header)
    max_lines = int(max_lines[0]) if len(max_lines) != 0 else 1

    attenuation_flag = re.findall(r'Attenuation correction flag: (\d)', header)
    attenuation_flag = bool(attenuation_flag[0]) if len(attenuation_flag) != 0 else False

    normalization_flag = re.findall(r'Normalization correction flag: (\d)', header)
    normalization_flag = bool(normalization_flag[0]) if len(normalization_flag) != 0 else False

    scatter_flag = re.findall(r'Scatter correction flag: (\d)', header)
    scatter_flag = bool(scatter_flag[0]) if len(scatter_flag) != 0 else False

    random_flag = re.findall(r'Random correction flag: (\d)', header)
    random_flag = bool(random_flag[0]) if len(random_flag) != 0 else False

    tof_flag = re.findall(r'TOF information flag: (\d)', header)
    tof_flag = bool(tof_flag[0]) if len(tof_flag) != 0 else False
    if tof_flag:
        tof_resolution = float(re.findall(r'TOF resolution \(ps\): (\d*[.]?\d*)', header)[0])

        if 'histo' in data_mode:
            tof_bins = int(re.findall(r'Histo TOF number of bins: (\d*)', header)[0])
            tof_size = float(re.findall(r'Histo TOF bin size \(ps\): (\d*[.]?\d*)', header)[0])
            tof_range = None

        elif 'list' in data_mode:
            tof_bins = None
            tof_size = None
            tof_range = float(re.findall(r'List TOF measurement range \(ps\): (\d*[.]?\d*)', header)[0])
    else:
        tof_resolution = None
        tof_bins = None
        tof_size = None
        tof_range = None

    data_header = {
        "data_filename": data_filename,
        "nb_events": nb_events,
        "data_mode": data_mode,
        "data_type": data_type,
        "start_time": start_time,
        "duration": duration,
        "scanner_name": scanner_name,
        "isotope": isotope,
        "calibration_factor": calibration_factor,
        "nb_max_lines": max_lines,
        "attenuation_flag": attenuation_flag,
        "normalization_flag": normalization_flag,
        "scatter_flag": scatter_flag,
        "random_flag": random_flag,
        "tof_flag": tof_flag,
        "tof_resolution": tof_resolution,
        "tof_bin_size": tof_size,
        "tof_nb_bins": tof_bins,
        "tof_range": tof_range
    }

    return data_header

def WriteHeaderFile(
        Header : dict, 
        HeaderFilePath : str
        ):
    """
    Writes a CASToR header to a text file
    
    Parameters:
    - Header (dict) : Dictionary of the CASToR datafile header
    - HeaderFilePath (str) : Path to desired CASToR datafile header
    
    """

    fileID = open(HeaderFilePath,'w')
    
    print(f'Data filename: {Header["data_filename"]}', file=fileID)

    print(f'Number of events: {Header["nb_events"]}', file=fileID)
    print(f'Data mode: {Header["data_mode"]}', file=fileID)
    print(f'Data type: {Header["data_type"]}', file=fileID)

    print(f'Start time (s): {Header["start_time"]}', file=fileID)
    print(f'Duration (s): {Header["duration"]}', file=fileID)

    print(f'Scanner name: {Header["scanner_name"]}', file=fileID)
    print(f'Isotope: {Header["isotope"]}', file=fileID)
    print(f'Calibration factor: {Header["calibration_factor"]}', file=fileID)

    if Header["nb_max_lines"] > 1:
        print(f'Maximum number of lines per event: {Header["nb_max_lines"]}', file=fileID)

    if Header["attenuation_flag"]:
        print('Attenuation correction flag: 1', file=fileID)
    if Header["normalization_flag"]:
        print('Normalization correction flag: 1', file=fileID)
    if Header["scatter_flag"]:
        print('Scatter correction flag: 1', file=fileID)
    if Header["random_flag"]:
        print('Random correction flag: 1', file=fileID)

    if Header["tof_flag"]:
        print('TOF information flag: 1', file=fileID)
        print(f'TOF resolution (ps): {Header["tof_resolution"]}', file=fileID)

        if 'histo' in Header["data_mode"]:
            print(f'Histo TOF number of bins: {Header["tof_nb_bins"]}', file=fileID)
            print(f'Histo TOF bin size (ps): {Header["tof_bin_size"]}', file=fileID)

        elif 'list' in Header["data_mode"]:
            print(f'List TOF measurement range (ps): {Header["tof_range"]}', file=fileID)
    
    fileID.close()