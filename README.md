# openSSS

 openSSS is an open source implementation of scatter estimation for 3D TOF-PET based on the TOF-aware Single Scatter Simulation (SSS) algorithm proposed by Watson [1]. It is compatible with any reconstruction platform. Two versions are currently available in this repository:
 
 - Matlab: stable. It is the original implementation, tested on version R2021b
 - Python: under active development. It is fully based on the stable Matlab version
 
 openSSS was validated for three PET system geometries, on Monte-Carlo simulated data and two vendor's specific reconstruction platforms. The first results were presented at the IEEE MIC2023 and further developments were presented at the CASToR's User's Meeting at the IEEE MIC2024:
 
 openSSS: an open-source implementation of scatter estimation for 3D TOF-PET
 <br />R. José Santo1, A. Salomon2, H. W.A.M. de Jong3, S. Stute4, 5, T. Merlin6, C. Beijst1
 <br />https://www.eventclass.org/contxt_ieee2023/scientific/online-program/session?s=M-09
 <br />IEEE NSS MIC 2023 Conference proceedings

 1. UMC Utrecht, Imaging & Oncology, Utrecht, The Netherlands; r.josesanto@umcutrecht.nl
 2. Philips Research Europe, Eindhoven, Netherlands
 3. UMC Utrecht, Radiology & Nuclear Medicine, Utrecht, The Netherlands
 4. CHU de Nantes, Nuclear Medicine, Nantes, France
 5. Université d'Angers, Université de Nantes, CRCINA, Inserm, CNRS, Pays de la Loire, France
 6. University of Brest, LaTIM, INSERM, UMR 1101, Brest, France
 
 The Python version was developed with additional contributions by:
 <br />T. Klinsuwan1, A. Hopkins1, H. R. Kanan1 and M. Colarieti-Tosti1
 <br />J. Neele2
 
 1. KTH Royal Institute of Technology, Biomedical Engineering and Health Systems, Stockholm, Sweden
 2. UMC Utrecht, Imaging & Oncology, Utrecht, The Netherlands

## References
 [1] C. C. Watson, "Extension of single scatter simulation to scatter correction of time-of-flight PET," IEEE Nuclear Science Symposium Conference Record, 2005, Fajardo, PR, USA, pp. 2492-2496, 2005.
