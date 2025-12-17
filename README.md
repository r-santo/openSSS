# openSSS

**Python library for scatter estimation in 3D TOF-PET**

openSSS is an open source Python library for scatter estimation in 3D TOF-PET based on the TOF-aware Single Scatter Simulation (SSS) algorithm [1]. It is technically compatible with any reconstruction platform, as it outputs scatter correction factors. It is primarily implemented in Python, with support and development of new features focused on this version. The original Matlab version is also made available.

## Examples

One example is provided with the corresponding raw data, as both a script and notebook, illustrating a _CASToR_-based pipeline. An additional example is planned for release soon, using an alternative implementation based on _ParallelProj_ projectors.  
The example dataset corresponds to GATE-simulated data of the NEMA IQ phantom for the geometry of the Siemens Biograph Vision, in the CASToR datafile format. It is publicaly available for download [here](https://doi.org/10.5281/zenodo.17649221).  
  
In order to run the example based on CASToR (assuming it is already installed):
* place the scanner geometry in a _Data_ folder that matches the _Path2Data_ folder in the parameter file
* place the datafile in a _Datafile_ folder that matches the _Path2Datafile_ folder in the parameter file
* update the CASToR reconstruction script with the _recon_ path to the CASToR reconstruction executable _castor-recon_ and the number of _threads_
* move the CASToR reconstruction script to the _Datafile_ folder
 
## Technical implementation
openSSS was validated for three PET system geometries, on Monte-Carlo simulated data and two vendor's specific reconstruction platforms. This is reported in our publication, which we recommend to read and cite if you are using openSSS: 
* R. José Santo, A. Salomon, H. de Jong, S. Stute, T. Merlin, C. Beijst. [**openSSS: an open-source implementation of scatter estimation for 3D TOF-PET**](https://ejnmmiphys.springeropen.com/articles/10.1186/s40658-025-00730-x). EJNMMI Phys 12, 17 (2025).

## Contributions
openSSS was initially developed with the contribution of:
* R. José Santo1, A. Salomon2, H. W.A.M. de Jong3, S. Stute4, 5, T. Merlin6, C. Beijst1
    1. UMC Utrecht, Imaging & Oncology, Utrecht, The Netherlands;
    2. Philips Research Europe, Eindhoven, Netherlands
    3. UMC Utrecht, Radiology & Nuclear Medicine, Utrecht, The Netherlands
    4. CHU de Nantes, Nuclear Medicine, Nantes, France
    5. Université d'Angers, Université de Nantes, CRCINA, Inserm, CNRS, Pays de la Loire, France
    6. University of Brest, LaTIM, INSERM, UMR 1101, Brest, France  

The Python version was developed with additional contributions by:
* T. Klinsuwan1, A. Hopkins1, H. R. Kanan1 and M. Colarieti-Tosti1
* J. Neele2
    1. KTH Royal Institute of Technology, Biomedical Engineering and Health Systems, Stockholm, Sweden
    2. UMC Utrecht, Imaging & Oncology, Utrecht, The Netherlands

## Support
For any questions and support, feel free to open an _Issue_ in _GitHub_ or contact directly the main maintainer:
* R. José Santo - r.josesanto@umcutrecht.nl

## References
[1] C. C. Watson. Extension of single scatter simulation to scatter correction of time-of-flight PET. IEEE Nuclear Science Symposium Conference Record 2005 (2005).
