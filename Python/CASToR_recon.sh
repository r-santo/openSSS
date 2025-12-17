#!/bin/bash

##
##
##  Reconstructs GATE simulations of the UMCU PET/MR system
##
##

###########################
# Some checks
###########################

if [ $# -lt 1 ]
then
  echo "Usage: $0 DATAFILE_NAME ITERATION"
  exit 1
fi

# Set the CASToR reconstruction program
recon="/PATH/TO/CASTOR/RECON/EXECUTABLE"
# Test the existency of the CASToR reconstruction program in the PATH
type ${recon} > /dev/null
if [ $? != 0 ]
then
  echo "***** In order to run the benchmark script, please add the CASToR binary folder into your PATH !"
  exit 1
fi

###########################
# Command-line options
###########################

# General verbose level
verbose="-vb 2"

# Loads the original datafile if it is the first step, otherwise load the datafile with scatters injected
if [ $2 == 1 ]
then
	datafile="-df $1_df.Cdh"
else
	datafile="-df $1_scatter_df.Cdh"
fi

# Loads the previous image guess from previous iteration, to emulate resuming the iterative process
if [ $2 == 1 ]
then
	image=""
else
	let "iteration = $2 - 1"
	image="-img output-$1-step$((iteration))_it1.hdr"
fi

# Senstivity image is also loaded after the first iteration in order to speed up the reconstruction
if [ $2 == 1 ]
then
	sens=""
else
	sens="-sens output-$1-step1_sensitivity.hdr"
fi

# The output file base name
output="-fout output-$1-step$2"

# This is an option to specify that we want to save the image of the last iteration only
last_it="-oit -1"

# Number of iterations (1) and subsets (17)
# The total number of iterations is definied by how many times this script is called, which can be automatically controlled in the openSSS python code
iteration="-it 1:17"

# Number of voxels of the reconstructed image (X,Y,Z)
voxels_number="-dim 256,256,256"

# Size of the voxels, in mm
vox_size="-vox 2.,2.,2."

# The reconstruction algorithm
optimizer="-opti MLEM"

# The projection algorithm
projector="-proj classicSiddon"

# Attenuation correction image
umap="-atn ./muMap.h33"

# Parallel computation using the OpenMP library.
# Adjust to your own setup
thread="-th 8"


###########################
# Launch the reconstruction
###########################

echo "=============================================================================================="
echo "Reconstruction is going on. Should take from one to several minutes depending on the hardware."
echo "=============================================================================================="
${recon} ${verbose} ${datafile} ${output} ${last_it} ${iteration} ${voxels_number} ${vox_size} ${optimizer} ${projector} ${umap} ${thread} ${sens} ${image}

# Finished
echo ""
exit 0

