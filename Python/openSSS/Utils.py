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
import re, os
from scipy.ndimage import zoom

from skimage.transform import resize
from numpy import ndarray # for comment the function
from typing import Tuple, List, Union # for comment the function

# Supporting functions
def PrepareImages(
        ActivityFilename : str,
        AttenuationFilename : str,
        SizeDesired : np.ndarray,
        ScaleDesired : np.ndarray
        )->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Prepares the maps, by cropping and downscaling according to specified parameters
    
    Parameters:
    - ActivityFilename (str): Path to the Activity map file
    - AttenuationFilename (str): Path to the Attenuation map file
    - SizeDesired (np.ndarray): Spatial size desired for the maps
    - ScaleDesired (np.ndarray): Scale desired for the voxels of the maps
    
    Returns:
    - reference_activity (np.ndarray) : Cropped and downscaled activity map
    - reference_attenuation (np.ndarray) : Cropped and downscaled attenuation map
    - SizeDesired (np.ndarray) : Updated spatial size desired for the maps, based on voxel size rounding
    - DimsDesired (np.ndarray) : Number of voxels in all directions of the cropped and downscaled maps
    """

    # Load file information
    [activity, dims_act, size_act] = LoadImage(ActivityFilename)
    [attenuation, dims_atn, size_atn] = LoadImage(AttenuationFilename)

    for i,dim in enumerate(SizeDesired):
        if dim is None:
            SizeDesired[i] = dims_act[i]*size_act[i]

    DimsDesired = (SizeDesired // ScaleDesired).astype(int)
    SizeDesired = DimsDesired * ScaleDesired
    # Activity image
    # reference_activity = CropMaps(activity, dims_act, size_act, SizeDesired)
    # reference_activity = ScaleMaps(reference_activity, reference_activity.shape, DimsDesired)
    reference_activity = CropAndDownscale(activity, size_act, SizeDesired, DimsDesired)
    
    # # Attenuation image
    # reference_attenuation = CropMaps(attenuation, dims_atn, size_atn, SizeDesired)
    # reference_attenuation = ScaleMaps(reference_attenuation, reference_attenuation.shape, DimsDesired)
    reference_attenuation = CropAndDownscale(attenuation, size_atn, SizeDesired, DimsDesired)
    reference_attenuation[reference_attenuation < 0.001] = 0

    return reference_activity, reference_attenuation, SizeDesired, DimsDesired

def ComputePhisicalDimensions(
        DesiredSize : np.ndarray,
        DeviceSize : np.ndarray
        )->tuple[np.ndarray, np.ndarray]:
    
    """
    Calculates the spatial dimensions for the activity and attenuation maps, based on the desired and device sizes. 
    The fitting size, used in backprojection for scaling, is based on the scanner boundaries
    
    Parameters:
    - DesiredSize (np.ndarray) : Spatial coordinates of the desired boundaries for the maps
    - DeviceSize (str) : Spatial coordinates of the boundaries of the scanner
    
    Returns:
    - ImageSize (np.ndarray) : Spatial coordinates of the boundaries for the maps to be downscaled into
    - FittingSize (np.ndarray) : Spatial coordinates of the boundaries for the backprojected maps, used in scaling and based on the device size
    """

    # Coordinates for the bounds of the image to be used to estimate scatters
    # in the format [xStart, yStart, zStart, xEnd, yEnd, zEnd] and in cm
    ImageSize = [-DesiredSize[0]/2, -DesiredSize[1]/2, -DesiredSize[2]/2,
                DesiredSize[0]/2, DesiredSize[1]/2, DesiredSize[2]/2]

    ImageSize = np.array([x / 10 for x in ImageSize])  # Convert to cm

    # Coordinates for the bounds of the backprojected data to be used in the scaling of scatters
    # in the format [xStart, yStart, zStart, xEnd, yEnd, zEnd] and in cm
    FittingSize = [-DesiredSize[0]/2, -DesiredSize[1]/2, -DeviceSize[2]/2,
                DesiredSize[0]/2, DesiredSize[1]/2, DeviceSize[2]/2]

    FittingSize = [x / 10 for x in FittingSize]  # Convert to cm
    FittingSize = np.array(FittingSize)

    return ImageSize, FittingSize

def LoadImage(
        Filename : str
        )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Loads the image from the filename header, in the interfile format
    
    Parameters:
    - Filename (str) : Path to the filename of the interfile header of the image to load
    
    Returns:
    - image (np.ndarray) : Loaded image
    - size (np.ndarray) : Number of voxels in each direction
    - scale (np.ndarray) : Size of the voxels in each direction
    """

    fileID = open(Filename,'r')
    header = fileID.read()
    fileID.close()   

    scale = np.zeros(3, dtype=float)
    size = np.zeros(3, dtype=int)
    for i in range(3):
        temp = re.findall(rf'.*matrix size \[{i+1}\] := (\d*).*', header)
        size[i] = int(temp[0])

        temp = re.findall(rf'.*scaling factor \(mm/pixel\) \[{i+1}\] := (\d*\.*\d+).*', header)
        scale[i] = float(temp[0])
    
    temp = re.findall(rf'.*name of data file := (\S*).*', header)
    imagefile = temp[0]

    image = np.fromfile(os.path.dirname(Filename) + f'/{imagefile}', dtype=np.float32).reshape(size[2],size[1],size[0]).transpose((2,1,0))

    return image, size, scale

def CropMaps(
        image : np.ndarray, 
        existingDims : np.ndarray, 
        voxel_size : np.ndarray, 
        desiredSize : np.ndarray
        )->np.ndarray:
    
    """
    Crops the image to the desired size. If the image is smaller than the desired size, it is padded.
    
    Parameters:
    - image (np.ndarray) : Image to downscale
    - existingDims (np.ndarray) : Number of voxels in each dimension
    - voxel_size (np.ndarray) : Size of the voxels in each direction
    - desiredSize (np.ndarray) : Spatial coordinates of the desired boundaries for the cropped image
    
    Returns:
    - cropped_image (np.ndarray) : Cropped image
    """

    pixel_half_thickness = np.ceil(np.array(desiredSize) / np.array(voxel_size) / 2).astype(int)
    center = np.array(existingDims) // 2

    # Initialize cropped image
    cropped_image = image

    # Cropping or padding each dimension as necessary
    for dim in range(3):
        start_idx = max(center[dim] - pixel_half_thickness[dim], 0)
        end_idx = min(center[dim] + pixel_half_thickness[dim], image.shape[dim])

        if start_idx == 0 and end_idx < pixel_half_thickness[dim] * 2:
            pad_width = [(0, 0)] * 3
            pad_width[dim] = (0, pixel_half_thickness[dim] * 2 - end_idx)
            cropped_image = np.pad(cropped_image, pad_width, mode='constant', constant_values=0)
        
        # Apply the slicing based on calculated indices
        slicing = [slice(None)] * 3
        slicing[dim] = slice(start_idx, end_idx)
        cropped_image = cropped_image[tuple(slicing)]

    return cropped_image

def ScaleMaps(
        image : np.ndarray, 
        existingDims : np.ndarray, 
        DimsDesired : np.ndarray
        )->np.ndarray:
    
    """
    Scales the image to the desired number of voxels
    
    Parameters:
    - image (np.ndarray) : Image to downscale
    - existingDims (np.ndarray) : Number of voxels in each dimension
    - DimsDesired (np.ndarray) : Number of desired voxels in each direction
    
    Returns:
    - referenceImage (np.ndarray) : Scaled image
    """

    referenceImage = zoom(image, np.divide(DimsDesired, existingDims), prefilter=False)
    referenceImage[referenceImage < 0] = 0
    return referenceImage

def CropAndDownscale(
    image: ndarray, 
    voxel_size: Union[Tuple[int, int, int], List[int]], 
    crop_size: Union[Tuple[int, int, int], List[int]], 
    downscaled_dimensions: Union[Tuple[int, int, int], List[int]], 
    clip: bool = True, 
    mode: str = 'edge', 
    preserve_range: bool = True, 
    anti_aliasing: bool = True, 
    order: int = 3
    ) -> ndarray:

    """
    Crops and downscales an image.
    Parameters:
    - image (ndarray): The original image to be cropped and downscaled.
    - voxel_size (tuple or list): Size of the voxels in the image [xLength, yLength, zLength] in mm.
    - crop_size (tuple or list): Desired size for the cropped image [xLength, yLength, zLength] in mm.
    - downscaled_dimensions (tuple or list): Dimensions for the downscaled image [xDim, yDim, zDim].
    - clip (bool, optional): Whether to clip the range at the end. Defaults to True.
    - mode (str, optional): How the input array is extended beyond its boundaries. Defaults to 'edge'.
    - preserve_range (bool, optional): Whether to keep the original range of values. Defaults to True.
    - anti_aliasing (bool, optional): Whether to apply a Gaussian filter to smooth the image prior to downsampling. Defaults to True.
    - order (int, optional): The order of the spline interpolation used when resizing. Defaults to 3 for cubic spline interpolation.

    Returns:
    - downscaled_image (ndarray): The cropped and downscaled image.
    """
    
    # Calculate half the thickness in pixels
    pixel_half_thickness = np.ceil(np.array(crop_size) / np.array(voxel_size) / 2).astype(int)
    center = np.array(image.shape) // 2

    # Initialize cropped image
    cropped_image = image

    # Cropping or padding each dimension as necessary
    for dim in range(3):
        start_idx = center[dim] - pixel_half_thickness[dim]
        end_idx = center[dim] + pixel_half_thickness[dim]

        if start_idx < 0 and end_idx > image.shape[dim]:
            pad_width = [(0, 0)] * 3
            pad = pixel_half_thickness[dim]-center[dim]
            pad_width[dim] = (pad, pad)
            cropped_image = np.pad(cropped_image, pad_width, mode='constant', constant_values=0)
        else:
            # Apply the slicing based on calculated indices
            slicing = [slice(None)] * 3
            slicing[dim] = slice(start_idx, end_idx)
            cropped_image = cropped_image[tuple(slicing)]
    
    # Resizing the image to the specified dimensions
    downscaled_image = resize(cropped_image, downscaled_dimensions, order=order, clip=clip,
                              mode=mode, preserve_range=preserve_range, anti_aliasing=anti_aliasing)
    return downscaled_image
