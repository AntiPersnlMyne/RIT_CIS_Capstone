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
Build a datacube from f/93v-92r directory, specify output directory with filename.

`tests/test_build_datacube.py -i data/raw_data/Arch_93v_92r/ -o data/datacubes/Arch_93v_92r.npy`

Build datacube as float64, normalizing output, increase program RAM to 13 GB.

`tests/test_build_datacube.py -i data/raw_data/Arch_93v_92r/ -o data/datacubes/Arch_93v_92r.npy -d float64 -n -c 13312`


### test_bgp.py
The band generation process (BGP) from Ren & Chang (2000). Generates pairwise combinations of the given datacube. E.g., given bands **A**, **B**, **C**, pairwise combinations would be **AA**, **AB**, **AC**, **BB**, **BC**, **CC**. The number of output bands are calculated with the equation **bands + bands(bands-1)/2**. The output file is always equal or larger than the input file. The output dtype always matches the input.

#### **Examples**
Input from datacube directory, output new datacube with original + pairwise bands.

`tests/test_bgp.py -i data/datacubes/archimedes_cubes/Arch_93v_92r.npy -o outputs/`

Create new datacube and specify output name with `-n`.

`tests/test_bgp.py -i data/datacubes/archimedes_cubes/Arch_93v_92r.npy -o outputs/ -n arch_pairwise`


### test_extract_gui.py
The extraction GUI uses a matplotlib.pyplot, with key controls, to select points on an image to be used as either "target" or "background" spectra for algorithms. **Left click** chooses a point, and creates a colored marker at the point. **Right click** undoes/deletes the most recently created point. Switch between "target" and "background" mode using '**t**' and '**b**' keys. Multiple points can be chosen and saved for each spectra class. So save and exit, press '**q**' or '**esc**'.

The program uses an RGB pseudocolor to display the GUI image. These are defined as band indices within the datacube. Spectra are saved as NumPy zip arrays (.npz) 

#### **Examples**

Define pseudocolor image as band indices of the datacube (**red=7, green=4, blue=2**) 

`tests/test_extract_gui.py -i data/datacubes/archimedes_cubes/Arch_93v_92r.npy -r 7 -g 4 -b 2`

Save results to file

`tests/test_extract_gui.py -i data/datacubes/archimedes_cubes/Arch_93v_92r.npy -r 7 -g 4 -b 2 -o results/arch_test.npz`


### test_eda.py
Exploratory data analysis (EDA) calculates and displays band statistics. The first result is an HTML formatted table, saved to project root, then immediately displayed in the default browser. The second result is the band correlation matrix. The HTML result is automatically saved, and the correlation matrix can be saved via the matplotlib window.

#### Examples
Plot statistics of f/93v-92r

`tests/test_eda.py -i data/datacubes/archimedes_cubes/Arch_93v_92r.npy`

### text_algorithms.py

The four algorithms are ACE, SAM, OSP, and GOSP. Only one algorithm can run per file execution. Mandated parameters are an input datacube, and for supervised algorithms, a spectra zip file. 

Displays each algorithm output (score map) as matplotlib subfigure. Due to the figure being low resolution, specifying an output directory saves each score map in full resolution individually.

All algorithms are memory-safe for any size datacube. The amount of memory used can be specified with the `-b` flag and **low**, **medium**, **high**. The default assumes a 16GB machine with few background processes running concurrently.

#### Examples

Run SAM

`test/text_algorithms -a sam -i data/datacubes/archimedes_cubes/Arch_93v_92r.npy -t results/arch_test.npz`

Run OSP  with batch processing

`test/text_algorithms -a osp -i data/datacubes/archimedes_cubes/Arch_93v_92r_bgp.npy -t results/arch_test.npz -O`

Run GOSP with HIGH (*warning*) memory usage for faster throughput 

`test/text_algorithms -a gosp -i data/datacubes/archimedes_cubes/Arch_93v_92r_bgp.npy -t results/arch_test.npz -b high`


# Acknowledgements
#### Author
Gian-Mateo (Mateo) Tifone
#### Advisors
David Messinger, Roger Easton Jr. 
#### Contributor
Douglas Tavolette

