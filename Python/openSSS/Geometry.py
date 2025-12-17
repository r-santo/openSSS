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
from numpy import ndarray
import re, os
from openSSS.Sinogram import SinogramCoordinates
from openSSS.DataReduction import SinogramLUT

def ImportCASToRGeometry(
        scanner : str, 
        ScannerPath : str
        ) -> tuple[ndarray, ndarray, int, int, int, int, int, int, ndarray]:
    
    """
    Geometry loading
    Imports the geometry from the .geom files as used by CASToR
    
    Parameters:
    - scanner (str) : Name of the scanner as in the CASToR .geom file
    - CASToRPath (str) : Path to the CASToR configuration folder where the .geom file are
    
    Returns:  
    - Geom (ndarray): Geometry of the scanner, organized as [Rings, Detectors, Coordinates (x,y,z)]
    - NormsVec (ndarray) : Vectors normal to the face of each crystal detector in the same organization structure
    - n_sectors_axial (int) : Number of sectors axially
    - n_sectors_tx (int) : Number of sectors transaxially
    - n_mod_ax (int) : Number of modules inside each sector axially
    - n_mod_tx (int) : Number of modules inside each sector transaxially
    - n_det_per_mod_ax (int) : Number of crystals/detectors inside each module axially
    - n_det_per_mod_tx (int) : Number of crystals/detectors inside each module transaxially
    - [det_size_ax, det_size_tx] (ndarray) : Detector/crystal face size
    """
    
    # Load geometrical information
    geomHeader = os.path.join(ScannerPath, f'{scanner}.geom')
    geomFile = os.path.join(ScannerPath , f'{scanner}.glut')

    fileID = open(geomHeader,'r')
    header = fileID.read()
    fileID.close()   

    nbElements = re.findall(r'.*number of elements.*: (\d*).*', header)
    nbElements = int(nbElements[0])

    # Ring information
    n_sectors_axial = re.findall(r'.*number of rsectors axial.*: (\d*).*', header)
    n_sectors_axial = int(n_sectors_axial[0])

    n_mod_ax = re.findall(r'.*number of modules axial.*: (\d*).*', header)
    n_mod_ax = int(n_mod_ax[0])

    n_det_per_mod_ax = re.findall(r'.*number of crystals axial.*: (\d*).*', header)
    n_det_per_mod_ax = int(n_det_per_mod_ax[0])
    
    det_size_ax = re.findall(r'.*crystals size axial.*: (\d*[.]?\d*).*', header)
    det_size_ax = float(det_size_ax[0])

    nbRings = n_mod_ax * n_det_per_mod_ax * n_sectors_axial

    # Detector information
    n_sectors_tx = re.findall(r'.*number of rsectors.*: (\d*).*', header)
    n_sectors_tx = int(n_sectors_tx[0])

    n_mod_tx = re.findall(r'.*number of modules transaxial.*: (\d*).*', header)
    n_mod_tx = int(n_mod_tx[0])

    n_det_per_mod_tx = re.findall(r'.*number of crystals transaxial.*: (\d*).*', header)
    n_det_per_mod_tx = int(n_det_per_mod_tx[0])

    det_size_tx = re.findall(r'.*crystals size transaxial.*: (\d*[.]?\d*).*', header)
    det_size_tx = float(det_size_tx[0])
    
    # Geometry - Division by 10 to convert from mm to cm
    LUT = np.fromfile(geomFile, dtype=np.float32).reshape(nbElements, 6)
    Geom = LUT[:, 0:3].astype(np.float64)
    Geom = np.reshape(Geom, (nbRings, int(nbElements/nbRings), 3)) / 10

    NormsVec = LUT[:, 3:6].astype(np.float64)
    NormsVec = np.reshape(NormsVec, (nbRings, int(nbElements/nbRings), 3)) / 10

    return Geom, NormsVec, n_sectors_axial, n_sectors_tx, n_mod_ax, n_mod_tx, n_det_per_mod_ax, n_det_per_mod_tx, np.array([det_size_ax, det_size_tx])

def PrepareGeometryAndSinograms(
        ScannerName : str, 
        Mash : int, 
        Span : int, 
        MRD : int, 
        ScannerPath : str, 
        Shift : bool = True
        )->tuple[ndarray, ndarray, ndarray]:
    
    """
    Prepares the geometry and the sinograms
    Processed the geometry from the .geom files as used by CASToR, adapted for Mash, Span and MRD and creates the needed LUTs for the sinogram coordinates
    
    Parameters:
    - ScannerName (str) : Name of the scanner as in the CASToR .geom file
    - Mash (int) : Mash value
    - Span (int) : Span value
    - MRD (int) : Maximum ring difference value
    - ScannerPath (str) : Path to the CASToR configuration folder where the .geom file are
    - Shift (bool, optional) : Flag if the detector shift in the first module is applied (so that detector index starts counting on the middle of the module)
    
    Returns:  
    - Geom (ndarray): Geometry of the scanner, organized as [Rings, Detectors, Coordinates (x,y,z)]
    - NormalVectors (ndarray) : Vectors normal to the face of each crystal detector in the same organization structure
    - DetectorSize (ndarray) : Detector/crystal face size
    - LORCoordinates (ndarray) : Gives the sinogram coordinates for each detector pair transaxially
    - SinogramIndex (ndarray) : Gives the sinogram index (slice) for each ring pair
    - LookUpTable (ndarray) : Gives the corresponding rings that contribute to each sinogram slice
    - extendedGeometryRings (ndarray) : Extended geometry in case span is used, to include rings inbetween
    - extendedNormalVectorsRings (ndarray) : Extended nomal bectors in case span is used, to include rings inbetween
    - SinogramCounts (ndarray) : Gives the corresponding rings that contribute to each sinogram slice
    - DetectorShift (int) : Number of detectors to shift in case the geometry does not start with the first crystal of the first module
    """

    Geometry, NormalVectors, NrSectorsAxial, NrSectorsTrans, NrModulesAxial, NrModulesTrans, NrCrystalsAxial, NrCrystalsTrans, DetectorSize = ImportCASToRGeometry(ScannerName, ScannerPath)
    NrRings = Geometry.shape[0]

    # Detector mashing - Mash
    if Mash > 1:
        print(f'Apply mash {Mash}')
        Geometry, NormalVectors = mashDetectors(Geometry, NormalVectors, Mash)
        totalDetectors = Geometry.shape[1]
        # No ID shift here because the inherent concept does not work
        LORCoordinates, SinogramIndex, DetectorShift = SinogramCoordinates(1, NrSectorsAxial, NrModulesAxial, 1, totalDetectors, NrCrystalsAxial, ID_Shidt=False)
    else:
        LORCoordinates, SinogramIndex, DetectorShift = SinogramCoordinates(NrSectorsTrans, NrSectorsAxial, NrModulesAxial, NrModulesTrans, NrCrystalsTrans, NrCrystalsAxial, ID_Shidt=Shift)

    # Sinogram Mashing - Span
    if Span > 1:
        print(f'Apply Span {Span}')
        SinogramIndex, _, LookUpTable, SinogramCounts = SinogramLUT(NrRings, Span, MRD) # Retrieve mashed sinograms + coordinates
        extendedGeometryRings, extendedNormalVectorsRings = expandRings(Geometry, NormalVectors)
    else:
        LookUpTable = np.array([])
        extendedGeometryRings = np.array([])
        extendedNormalVectorsRings = np.array([])
        SinogramCounts = np.array([])

    return Geometry, NormalVectors, DetectorSize, LORCoordinates, SinogramIndex, LookUpTable, extendedGeometryRings, extendedNormalVectorsRings, SinogramCounts, DetectorShift

def expandRings(
        Geometry : np.ndarray, 
        NormalVectors :np.ndarray
        )->tuple[np.ndarray, np.ndarray]:
    
    """
    Expands the geometry and the normal vectors when span is applied, to take into account the mid-ring positions (as in the michelogram)

    Parameters:
    - Geometry (ndarray): Geometry of the scanner, organized as [Rings, Detectors, Coordinates (x,y,z)]
    - NormalVectors (ndarray) : Vectors normal to the face of each crystal detector in the same organization structure
    
    Returns:  
    - newGeometry (ndarray): Geometry with double the ring minus 1, inserted with in-between rings
    - newNormalVectors (ndarray) : Vectors normal to the face of each crystal detector equally expanded
    """

    NrCrystals = Geometry.shape[1]
    totalRings = Geometry.shape[0] * 2 - 1

    newGeometry = np.zeros((totalRings, NrCrystals, 3))
    newNormalVectors = np.zeros_like(newGeometry)

    # Put original geometry in first
    newGeometry[0:totalRings:2,:,:] = Geometry

    # Take care of normals
    newNormalVectors = np.zeros_like(newGeometry)
    newNormalVectors[:,:,:] = NormalVectors[0,:,:]

    # Create geometry of new rings
    k = 0
    for i in range(1,totalRings,2):
        newGeometry[i,:,:] = (Geometry[k,:,:] + Geometry[k+1,:,:]) / 2
        k += 1

    return newGeometry, newNormalVectors


def mashDetectors(
        Geometry : np.ndarray, 
        NormalVectors : np.ndarray, 
        combineSize : int
        )->tuple[np.ndarray,np.ndarray]:
    
    """
    Collapses the geometry by reducing the number of detectors, merging them together (coordinates are averaged)

    Parameters:
    - Geometry (ndarray): Geometry of the scanner, organized as [Rings, Detectors, Coordinates (x,y,z)]
    - NormalVectors (ndarray) : Vectors normal to the face of each crystal detector in the same organization structure
    - combineSize (int) : Number of detectors to collapse together
    
    Returns:  
    - newGeometry (ndarray): Geometry with fraction of the detectors per ring
    - newNormalVectors (ndarray) : Vectors normal to the face of each crystal detector equally expanded
    """

    # Retrieve number of detectors in current geometry
    NrCrystals = Geometry.shape[1]
    if NrCrystals % combineSize != 0:
        raise Exception("Number of detectors not divisible by combineSize: ", NrCrystals, combineSize)

    # Decrease the geometry of the detectors
    NrCrystalsSiemens = Geometry.shape[1]
    newGeometry = np.zeros((Geometry.shape[0], int(NrCrystalsSiemens / combineSize), 3))
    newNormalVectors = np.zeros_like(newGeometry)

    k = 0
    for i in range(0, NrCrystalsSiemens, combineSize):
        newGeometry[:,k,:] = np.nanmean(Geometry[:,i:i+combineSize,:],1)
        newNormalVectors[:,k,:] = np.nanmean(NormalVectors[:,i:i+combineSize,:],1)
        k += 1

    return newGeometry, newNormalVectors

