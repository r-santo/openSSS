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
from numpy import ndarray # for comment the function
from numba import jit # for fast computation

##################################################################################################
#################### SUPPORT FUNCTIONS FOR Forward Projection ####################################
##################################################################################################

@jit(nopython=True)
def RayTracing3DTOF(
    GridSize: ndarray,
    GridBounds: ndarray,
    LineCoordinates: ndarray
    )->tuple[ndarray, ndarray, ndarray]:
    
    """
    Traces a line segment defined by two points through a voxel grid (using Woo's raytracing algorithm).
    Determines the linear indexes of voxels intersected by the line segment, the lenght and the corresponding lenght sample point

    Parameters:
    - GridSize (ndarray): dimensions of the voxel grid [Nx Ny Nz]
    - GridBounds (ndarray): 1-by-6 matrix with voxel grid boundaries [xMin yMin zMin xMax yMax zMax] (must satisfy xMax>xMin, yMax>yMin, zMax>zMin)
    - LineCoordinates (ndarray): 1-by-6 matrix with line segment coordinates [x1,y1,z1,x2,y2,z2]
    
    Returns:      
    - IntersectedVoxelData (ndarray): N-by-1 matrix containing lenghts inside the intersected voxels
    - IntersectedVoxelIndices (ndarray): N-by-1 matrix containing linear indeces of the intersected voxels
    - IntersectedVoxelSamples (ndarray): N-by-1 matrix containing the sample point at the middle of the intersected voxels
    """

    intersectTest, tMin, tMax = BoxIntersectTest(GridSize, GridBounds, LineCoordinates)
    if intersectTest:
        tMin = max(tMin, 0)
        tMax = min(tMax, 1)

        xVoxelSize = (GridBounds[3] - GridBounds[0]) / GridSize[0]
        yVoxelSize = (GridBounds[4] - GridBounds[1]) / GridSize[1]
        zVoxelSize = (GridBounds[5] - GridBounds[2]) / GridSize[2]

        xVec = LineCoordinates[3] - LineCoordinates[0]
        yVec = LineCoordinates[4] - LineCoordinates[1]
        zVec = LineCoordinates[5] - LineCoordinates[2]

        xGridStart = LineCoordinates[0] + xVec * tMin
        yGridStart = LineCoordinates[1] + yVec * tMin
        zGridStart = LineCoordinates[2] + zVec * tMin
        xGridEnd = LineCoordinates[0] + xVec * tMax
        yGridEnd = LineCoordinates[1] + yVec * tMax
        zGridEnd = LineCoordinates[2] + zVec * tMax

        StorageSize = int(GridSize[0] + GridSize[1] + GridSize[2])

        IntersectedVoxelData = np.zeros(StorageSize)
        IntersectedVoxelIndices = np.zeros(StorageSize)
        IntersectedVoxelSamples = np.zeros(StorageSize)

        X = max(1, min(GridSize[0], np.ceil((xGridStart - GridBounds[0]) / xVoxelSize)))
        xEnd = max(1, min(GridSize[0], np.ceil((xGridEnd - GridBounds[0]) / xVoxelSize)))

        if xVec > 0:
            xStep = 1
            xTDelta = xVoxelSize / xVec
            xTMax = tMin + (GridBounds[0] + X * xVoxelSize - xGridStart) / xVec
        elif xVec < 0:
            xStep = -1
            xTDelta = xVoxelSize / -xVec
            xTMax = tMin + (GridBounds[0] + (X - 1) * xVoxelSize - xGridStart) / xVec
        else:
            xStep = 0
            xTMax = tMax
            xTDelta = tMax

        Y = max(1, min(GridSize[1], np.ceil((yGridStart - GridBounds[1]) / yVoxelSize)))
        yEnd = max(1, min(GridSize[1], np.ceil((yGridEnd - GridBounds[1]) / yVoxelSize)))

        if yVec > 0:
            yStep = 1
            yTDelta = yVoxelSize / yVec
            yTMax = tMin + (GridBounds[1] + Y * yVoxelSize - yGridStart) / yVec
        elif yVec < 0:
            yStep = -1
            yTDelta = yVoxelSize / -yVec
            yTMax = tMin + (GridBounds[1] + (Y - 1) * yVoxelSize - yGridStart) / yVec
        else:
            yStep = 0
            yTMax = tMax
            yTDelta = tMax

        Z = min(max(1, int(np.ceil((zGridStart - GridBounds[2]) / zVoxelSize))), int(GridSize[2]))
        zEnd = min(max(1, int(np.ceil((zGridEnd - GridBounds[2]) / zVoxelSize))), int(GridSize[2]))

        if zVec > 0:
            zStep = 1
            zTDelta = zVoxelSize / zVec
            zTMax = tMin + (GridBounds[2] + Z * zVoxelSize - zGridStart) / zVec
        elif zVec < 0:
            zStep = -1
            zTDelta = zVoxelSize / -zVec
            zTMax = tMin + (GridBounds[2] + (Z - 1) * zVoxelSize - zGridStart) / zVec
        else:
            zStep = 0
            zTMax = tMax
            zTDelta = tMax

        IntersectedVoxelIndices[0] = sub2ind(X, Y, Z, GridSize)
        AddedVoxelCount = 1
        xGridStart = xVec * tMin
        yGridStart = yVec * tMin
        zGridStart = zVec * tMin
        previousLength = 0

        while X != xEnd or Y != yEnd or Z != zEnd:
            if xTMax < yTMax:
                if xTMax < zTMax:
                    X += xStep
                    length = xTMax
                    xTMax += xTDelta
                else:
                    Z += zStep
                    length = zTMax
                    zTMax += zTDelta
            else:
                if yTMax < zTMax:
                    Y += yStep
                    length = yTMax
                    yTMax += yTDelta
                else:
                    Z += zStep
                    length = zTMax
                    zTMax += zTDelta

            IntersectedVoxelData[AddedVoxelCount-1] = np.sqrt((xGridStart - xVec * length) ** 2 +
                                                                (yGridStart - yVec * length) ** 2 +
                                                                (zGridStart - zVec * length) ** 2)
            IntersectedVoxelSamples[AddedVoxelCount-1] = previousLength + IntersectedVoxelData[AddedVoxelCount-1] / 2
            previousLength = IntersectedVoxelSamples[AddedVoxelCount-1]

            xGridStart = xVec * length
            yGridStart = yVec * length
            zGridStart = zVec * length
            AddedVoxelCount += 1
            IntersectedVoxelIndices[AddedVoxelCount-1] = sub2ind(X, Y, Z, GridSize)

        IntersectedVoxelData[AddedVoxelCount-1] = np.sqrt((xGridStart - xVec * tMax) ** 2 +
                                                            (yGridStart - yVec * tMax) ** 2 +
                                                            (zGridStart - zVec * tMax) ** 2)
        IntersectedVoxelSamples[AddedVoxelCount-1] = previousLength + IntersectedVoxelData[AddedVoxelCount-1] / 2

        IntersectedVoxelIndices = IntersectedVoxelIndices.astype(np.int32)

        IntersectedVoxelData = IntersectedVoxelData[:AddedVoxelCount]
        IntersectedVoxelIndices = IntersectedVoxelIndices[:AddedVoxelCount]
        IntersectedVoxelSamples = IntersectedVoxelSamples[:AddedVoxelCount]

    else : 
        IntersectedVoxelData = np.zeros(0)
        IntersectedVoxelIndices = np.zeros(0).astype(np.int32)
        IntersectedVoxelSamples = np.zeros(0)

    return IntersectedVoxelData, IntersectedVoxelIndices, IntersectedVoxelSamples

@jit(nopython=True)
def sub2ind(
    X : int, 
    Y : int, 
    Z : int, 
    GridSize : ndarray
    )->int:
    """
    Converts the 3D coordinates index into a linear index

    Parameters:
    - X (int) : Voxel index of X direction
    - Y (int) : Voxel index of Y direction
    - Z (int) : Voxel index of Z direction
    - GridSize (ndarray) : Number of voxels of the grid in each direction
    
    Returns:      
    - Index (int) : Linear index
    """

    return X + (Y - 1) * GridSize[1] + (Z - 1) * GridSize[1] * GridSize[0]

@jit(nopython=True)
def BoxIntersectTest(
    grid_size : ndarray, 
    grid_bounds : ndarray, 
    line_coordinates : ndarray
    )->tuple[bool, float, float]:

    """
    Tests if the line intersects the grid

    Parameters:
    - grid_size (ndarray) : Number of voxels of the grid in each direction
    - grid_bounds (ndarray) : Size of the grid in distance units
    - line_coordinates (ndarray) : Coordinates of the line start and end
    
    Returns:      
    - Intersect (bool) : Flag to say it intersects or not
    - t_min (float) : Minimum fraction of line that is intersected
    - t_max (float) : Maximum fraction of line that is intersected
    """

    x_div = 1 / (line_coordinates[3] - line_coordinates[0] + 1e-20)
    
    if x_div >= 0:
        t_min = (grid_bounds[0] - line_coordinates[0]) * x_div
        t_max = (grid_bounds[3] - line_coordinates[0]) * x_div
    else:
        t_min = (grid_bounds[3] - line_coordinates[0]) * x_div
        t_max = (grid_bounds[0] - line_coordinates[0]) * x_div

    y_div = 1 / (line_coordinates[4] - line_coordinates[1] + 1e-20)
    
    if y_div >= 0:
        y_t_min = (grid_bounds[1] - line_coordinates[1]) * y_div
        y_t_max = (grid_bounds[4] - line_coordinates[1]) * y_div
    else:
        y_t_min = (grid_bounds[4] - line_coordinates[1]) * y_div
        y_t_max = (grid_bounds[1] - line_coordinates[1]) * y_div

    if t_min > y_t_max or y_t_min > t_max:
        return False, None, None
    if y_t_min > t_min:
        t_min = y_t_min
    if y_t_max < t_max:
        t_max = y_t_max

    z_div = 1 / (line_coordinates[5] - line_coordinates[2] + 1e-20)
    
    if z_div >= 0:
        z_t_min = (grid_bounds[2] - line_coordinates[2]) * z_div
        z_t_max = (grid_bounds[5] - line_coordinates[2]) * z_div
    else:
        z_t_min = (grid_bounds[5] - line_coordinates[2]) * z_div
        z_t_max = (grid_bounds[2] - line_coordinates[2]) * z_div

    if t_min > z_t_max or z_t_min > t_max:
        return False, None, None
    if z_t_min > t_min:
        t_min = z_t_min
    if z_t_max < t_max:
        t_max = z_t_max

    if (t_min >= 1 and t_max >= 1) or (t_min <= 0 and t_max <= 0):
        return False, None, None
    
    return True, t_min, t_max