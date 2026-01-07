# Introduction
Python code for Imaging Science capstone project, "Developing an Empirical Workflow for Cultural Heritage Imaging: Statistical vs. Geometrical Detectors". DOI ~never~.


# Setup
### **conda**

```bash
conda create --name env python=3.13 
conda activate env
conda install pip
pip install -r requirements.txt
```

### **venv (windows)**
```shell
python -m venv env 
env\Scripts\activate
pip install -r requirements.txt
```

### **venv (macOS/Linux)**
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```


# Algorithms
Summaries courtesy ENVI(R) Software Docs. GOSP summary courtesy Ren & Chang (2000).
### **ACE**
(**statistical**) Adaptive Coherence Estimator (Kraut & Scharf, 1999). Derived from the Generalized Likelihood Ratio (GLR) approach. The ACE is invariant to relative scaling of input spectra and has a Constant False Alarm Rate (CFAR) with respect to such scaling.

### **SAM**
(**geometrical**)
Uses an n-D angle to match pixels to reference spectra. The algorithm determines the spectral similarity between two spectra by calculating the angle between the spectra and treating them as vectors in a space with dimensionality equal to the number of bands.

### **OSP**
(**geometrical**)
Classification first designs an orthogonal subspace projector to eliminate the response of non-targets, then Matched Filter is applied to match the desired target from the data. OSP is efficient and effective when target signatures are distinct.

### **GOSP**
(**geometrical**)
GOSP relaxes the band number constraint inherent to OSP in
such a manner that OSP can be extended to multispectral image
processing in an unsupervised fashion. The idea of the GOSP is to
create a new set of additional bands that are generated nonlinearly
from original multispectral bands prior to the OSP classification. It is then followed by an unsupervised OSP classifier called automatic
target detection and classification algorithm (ATDCA)

# Tests
Test scripts are provided to test functionality of the entire code base. They can be used in sequence, or as a reference for automation. Each script includes command line interface options (CLI) and a help flag (-h). The purpose of each test script, and examples, are provided below. 

### test_build_datacube.py
A datacube is an image with multiple channels, called bands, usually in the BIP file structure i.e. `(rows, cols, bands)` or `(height, width, channels)`. A datacube object within the code refers to a NumPy array file (.npz) with BIP file structure.

To build a datacube, provide one of the following as the input flag (`-i`): 1) A directory containing single-band TIFF files, 2) A path to a multiband TIFF file, 3) A directory (or path) to a single-band or multiband H5 file.

Note: datacube is not normalized by default, this requires the `-n` flag, seen in example 2.

#### **Examples**
Build a datacube from f/93v-92r directory, specify output directory with filename
`tests/test_build_datacube.py -i data/raw_data/Arch_93v_92r/ -o data/datacubes/Arch_93v_92r.npy`

Build datacube as float64, normalizing output, increase program RAM to 13 GB
`tests/test_build_datacube.py -i data/raw_data/Arch_93v_92r/ -o data/datacubes/Arch_93v_92r.npy -d float64 -n -c 13312`


### test_bgp.py
The band generation process (BGP)

### test_extract_gui.py

### test_eda.py


### text_algorithms.py


# Acknowledgements
#### Author
Gian-Mateo (Mateo) Tifone
#### Advisors
David Messinger, Roger Easton Jr. 

