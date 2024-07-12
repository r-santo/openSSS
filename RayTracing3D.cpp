/// This file is part of openSSS.
// 
//     openSSS is free software: you can redistribute it and/or modify it under the
//     terms of the GNU General Public License as published by the Free Software
//     Foundation, either version 3 of the License, or (at your option) any later
//     version.
// 
//     openSSS is distributed in the hope that it will be useful, but WITHOUT ANY
//     WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
//     FOR A PARTICULAR PURPOSE.
// 
//     You should have received a copy of the License along with openSSS
// 
// Copyright 2022-2023 all openSSS contributors listed below:
// 
//     --> Rodrigo JOSE SANTO, Andre SALOMON, Hugo DE JONG, Thibaut MERLIN, Simon STUTE, Casper BEIJST
// 
// This is openSSS version 0.1
//
//__________________________________________________________________________________________________________________
// RayTracing3D
// Traces a line segment defined by two points through a voxel grid (using Woo's raytracing algorithm).
// Determines the linear indexes of voxels intersected by the line segment, the lenght and the corresponding lenght sample point
//
// INPUT:       first           - dimensions of the voxel grid [Nx Ny Nz]
//              second          - 1-by-6 matrix with voxel grid boundaries [xMin yMin zMin xMax yMax zMax] (must satisfy xMax>xMin, yMax>yMin, zMax>zMin)
//              third           - 1-by-6 matrix with line segment coordinates [x1,y1,z1,x2,y2,z2]
//
// OUTPUT:      first           - N-by-1 matrix containing lenghts inside the intersected voxels
//              second          - N-by-1 matrix containing linear indeces of the intersected voxels
//              third           - N-by-1 matrix containing the sample point at the middle of the intersected voxels
//
// Script by:
// Rodrigo JOSE SANTO - UMC Utrecht
// Adapted from original code at MathWorks File Exchange by Ivan Klyuzhin

#include <string.h>
#include "mex.h"
#include <math.h>
#include <algorithm>
#include <iostream>

using namespace std;

#define MAX(a,b) ((a>=b)?a:b) 
#define MIN(a,b) ((a<=b)?a:b)

int sub2ind(const int X, const int Y, const int Z, const double * GridSize); // convert subscript to linear index - Y-dimension taken as first dimension
int BoxIntersectTest(double const *GridSize, double const *GridBounds, double const *LineCoordinates, double & tmin, double & tmax); // box-line intersect test

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

//Check the number of inputs and outputs
if (nlhs != 2) {
	mexErrMsgTxt("Error: Number of output arguments must be equal 2.");
}
if (nrhs != 3) {
	mexErrMsgTxt("Error: Number of input arguments must be equal 3.");
}
//Check sizes of the inputs
if ( (mxGetN(prhs[0])!=3) || (mxGetM(prhs[0])!=1) || (!mxIsClass(prhs[0],"double")) )
    mexErrMsgTxt("First input must be a 1-by-3 matrix of type double");
if ( (mxGetN(prhs[1])!=6) || (mxGetM(prhs[1])!=1) || (!mxIsClass(prhs[1],"double")) )
    mexErrMsgTxt("Second input must be a 1-by-6 matrix of type double");
if ( (mxGetN(prhs[2])!=6) || (mxGetM(prhs[2])!=1) || (!mxIsClass(prhs[2],"double")) )
    mexErrMsgTxt("Third input must be a 1-by-6 matrix of type double");

double const *GridSize, *GridBounds, *LineCoordinates;

GridSize = mxGetPr(prhs[0]); // [Nx Ny Nz]
GridBounds = mxGetPr(prhs[1]); // [xMin yMin zMin xMax yMax zMax],
LineCoordinates = mxGetPr(prhs[2]);// [x1 y1 z1 x2 y2 z2],

double xVoxelSize, yVoxelSize, zVoxelSize;
double xTDelta, yTDelta, zTDelta; // parametric step size along different dimensions
double xTMax, yTMax, zTMax; // used to determine the dimension of the next step along the line
double tMin, tMax; // maximum and minimum parameteric coordinates
int X, Y, Z, xEnd, yEnd, zEnd; // voxel subscripts
int xStep, yStep, zStep; // direction of grid traversal
int StorageSize; //initial storage size - dynamically updated

//storage for the indexes of the intersected voxels - dynamically allocated
float *IntersectedVoxelData;
int *IntersectedVoxelIndices;
int AddedVoxelCount;

double xGridStart, yGridStart, zGridStart, xGridEnd, yGridEnd, zGridEnd; // point where segment intersects the grid
double xVec, yVec, zVec; //direction of the line segment
double length; // direction in which the line is incremented

// Intersection test, find minimum and maximum parameteric intersection coordinate
int intersectTest = BoxIntersectTest(GridSize, GridBounds, LineCoordinates, tMin, tMax);
if (intersectTest==0) {
	//return empty array
	plhs[0] = mxCreateNumericMatrix(0,0,mxINT32_CLASS,mxREAL);
    plhs[1] = mxCreateNumericMatrix(0,0,mxINT32_CLASS,mxREAL);
	return;
}
tMin = MAX(tMin,0);
tMax = MIN(tMax,1);
// Compute helpful variables
xVoxelSize = (GridBounds[3] - GridBounds[0])/GridSize[0];
yVoxelSize = (GridBounds[4] - GridBounds[1])/GridSize[1];
zVoxelSize = (GridBounds[5] - GridBounds[2])/GridSize[2];
xVec = LineCoordinates[3] - LineCoordinates[0];
yVec = LineCoordinates[4] - LineCoordinates[1];
zVec = LineCoordinates[5] - LineCoordinates[2];
xGridStart = LineCoordinates[0] + xVec*tMin;
yGridStart = LineCoordinates[1] + yVec*tMin;
zGridStart = LineCoordinates[2] + zVec*tMin;
xGridEnd = LineCoordinates[0] + xVec*tMax;
yGridEnd = LineCoordinates[1] + yVec*tMax;
zGridEnd = LineCoordinates[2] + zVec*tMax;

// Allocate memory to store the indexes of the intersected voxels
StorageSize = GridSize[0] + GridSize[1] + GridSize[2]; 
plhs[0] = mxCreateNumericMatrix(StorageSize,1,mxSINGLE_CLASS,mxREAL); // initializes to zeros
IntersectedVoxelData = (float*)mxGetData(plhs[0]);
plhs[1] = mxCreateNumericMatrix(StorageSize,1,mxINT32_CLASS,mxREAL); // initializes to zeros
IntersectedVoxelIndices = (int*)mxGetData(plhs[1]);

// Determine initial voxel coordinates and line traversal directions
// X-dimension
X = MAX(1,MIN(GridSize[0],ceil((xGridStart-GridBounds[0])/xVoxelSize)));  // starting coordinate - include left boundary - index starts from 1
xEnd = MAX(1,MIN(GridSize[0],ceil((xGridEnd-GridBounds[0])/xVoxelSize))); // ending coordinate - stepping continues until we hit this index
if (xVec>0) 
{
    xStep = 1;
    xTDelta = xVoxelSize/xVec; //parametric step length between the x-grid planes
    xTMax = tMin + (GridBounds[0] + X*xVoxelSize - xGridStart)/xVec; // parametric distance until the first crossing with x-grid plane
}
else if (xVec<0)
{
    xStep = -1;
    xTDelta = xVoxelSize/-xVec; //parametric step length between the x-grid planes
    xTMax = tMin + (GridBounds[0] + (X-1)*xVoxelSize - xGridStart)/xVec; // parametric distance until the first crossing with x-grid plane
}
else
{
    xStep = 0;
    xTMax = tMax; // the line doesn't cross the next x-plane
    xTDelta = tMax; // set the parametric step to maximum
}
// Y-dimension
Y = MAX(1,MIN(GridSize[1],ceil((yGridStart-GridBounds[1])/yVoxelSize)));
yEnd = MAX(1,MIN(GridSize[1], ceil((yGridEnd-GridBounds[1])/yVoxelSize)));
if (yVec>0) 
{
    yStep = 1;
    yTDelta = yVoxelSize/yVec;
    yTMax = tMin + (GridBounds[1] + Y*yVoxelSize - yGridStart)/yVec;
}
else if (yVec<0)
{
    yStep = -1;
    yTDelta = yVoxelSize/-yVec;
    yTMax = tMin + (GridBounds[1] + (Y-1)*yVoxelSize - yGridStart)/yVec;
}
else
{
    yStep = 0;
    yTMax = tMax;
    yTDelta = tMax;
}
// Z-dimension
Z = MAX(1,MIN(GridSize[2],ceil((zGridStart-GridBounds[2])/zVoxelSize)));
zEnd = MAX(1,MIN(GridSize[2],ceil((zGridEnd-GridBounds[2])/zVoxelSize)));
if (zVec>0) 
{
    zStep = 1;
    zTDelta = zVoxelSize/zVec;
    zTMax = tMin + (GridBounds[2] + Z*zVoxelSize - zGridStart)/zVec;
}
else if (zVec<0)
{
    zStep = -1;
    zTDelta = zVoxelSize/-zVec;
    zTMax = tMin + (GridBounds[2] + (Z-1)*zVoxelSize - zGridStart)/zVec;
}
else
{
    zStep = 0;
    zTMax = tMax;
    zTDelta = tMax;
}

// Add initial voxel to the list
IntersectedVoxelIndices[0] = sub2ind(X, Y, Z, GridSize);
AddedVoxelCount = 1;

xGridStart = xVec*tMin;
yGridStart = yVec*tMin;
zGridStart = zVec*tMin;

// Step iteratively through the grid
while ((X!=xEnd)||(Y!=yEnd)||(Z!=zEnd))
{
    if (xTMax<yTMax)
    {
        if (xTMax<zTMax)
        {
            X += xStep;
            length = xTMax;
            xTMax += xTDelta;
        }
        else
        {
            Z += zStep;
            length = zTMax;
            zTMax += zTDelta;
        }
    }
    else
    {
        if (yTMax<zTMax)
        {
            Y += yStep;
            length = yTMax;
            yTMax += yTDelta;
        }
        else
        {
            Z += zStep;
            length = zTMax;
            zTMax += zTDelta;
        }
    }
    //must perform memory check - if the initial allocated array is large enough this step is not necessary
    if (AddedVoxelCount>StorageSize)
    {
        StorageSize = StorageSize+1000;
        IntersectedVoxelData = (float*)mxRealloc(IntersectedVoxelData, sizeof(float)*StorageSize);
        IntersectedVoxelIndices = (int*)mxRealloc(IntersectedVoxelIndices, sizeof(int)*StorageSize);
    }
    IntersectedVoxelData[AddedVoxelCount-1] = (float) sqrt(pow(xGridStart - xVec*length,2) + pow(yGridStart - yVec*length,2) + pow(zGridStart - zVec*length,2));
    xGridStart = xVec*length;
    yGridStart = yVec*length;
    zGridStart = zVec*length;
    AddedVoxelCount++;
    IntersectedVoxelIndices[AddedVoxelCount-1] = sub2ind(X, Y, Z, GridSize);
}
IntersectedVoxelData[AddedVoxelCount-1] = sqrt(pow(xGridStart - xVec*tMax,2) + pow(yGridStart - yVec*tMax,2) + pow(zGridStart - zVec*tMax,2));
// Update the size of the output matrix
IntersectedVoxelData = (float*)mxRealloc(IntersectedVoxelData, sizeof(float)*AddedVoxelCount);
IntersectedVoxelIndices = (int*)mxRealloc(IntersectedVoxelIndices, sizeof(int)*AddedVoxelCount);

mxSetM(plhs[0], AddedVoxelCount); //number of rows
mxSetN(plhs[0], 1); //number of columns
mxSetData(plhs[0], IntersectedVoxelData); // update pointer to the matrix data

mxSetM(plhs[1], AddedVoxelCount); //number of rows
mxSetN(plhs[1], 1); //number of columns
mxSetData(plhs[1], IntersectedVoxelIndices); // update pointer to the matrix data

}

int sub2ind(const int X, const int Y, const int Z, const double * GridSize)
{
	return (X + (Y - 1)*GridSize[1] + (Z - 1)*GridSize[1]*GridSize[0]);
}

int BoxIntersectTest(double const *GridSize, double const *GridBounds, double const *LineCoordinates, double & rTMin, double & rTMax)
{
double tMin, tMax, yTMin, yTMax, zTMin, zTMax;
double xDiv, yDiv, zDiv;

xDiv = 1/(LineCoordinates[3] - LineCoordinates[0] + 1e-23);

if (xDiv >= 0) // t-coordinate of box bounds
{
     tMin = (GridBounds[0] - LineCoordinates[0])*xDiv;
     tMax = (GridBounds[3] - LineCoordinates[0])*xDiv;
} 
else 
{
     tMin = (GridBounds[3] - LineCoordinates[0])*xDiv;
     tMax = (GridBounds[0] - LineCoordinates[0])*xDiv;
}

yDiv = 1/(LineCoordinates[4] - LineCoordinates[1] + 1e-23);

if (yDiv >= 0) 
{
     yTMin = (GridBounds[1] - LineCoordinates[1])*yDiv;
     yTMax = (GridBounds[4] - LineCoordinates[1])*yDiv;
} 
else 
{
     yTMin = (GridBounds[4] - LineCoordinates[1])*yDiv;
     yTMax = (GridBounds[1] - LineCoordinates[1])*yDiv;
}

if ( (tMin > yTMax) || (yTMin > tMax) ) // check if line misses the box
	return false;
if (yTMin > tMin)
	tMin = yTMin;
if (yTMax < tMax)
	tMax = yTMax;

zDiv = 1/(LineCoordinates[5] - LineCoordinates[2] + 1e-23);

if (zDiv >= 0) 
{
     zTMin = (GridBounds[2] - LineCoordinates[2])*zDiv;
     zTMax = (GridBounds[5] - LineCoordinates[2])*zDiv;
} 
else 
{
     zTMin = (GridBounds[5] - LineCoordinates[2])*zDiv;
     zTMax = (GridBounds[2] - LineCoordinates[2])*zDiv;
}

if ((tMin > zTMax) || (zTMin > tMax)) // check if line misses the box
	return false;
if (zTMin > tMin)
	tMin = zTMin;
if (zTMax < tMax)
	tMax = zTMax;
if ((tMin>=1)&&(tMax>=1))
    return false;
if ((tMin<=0)&&(tMax<=0))
    return false;
rTMin = tMin;
rTMax = tMax;
return 1;
}

