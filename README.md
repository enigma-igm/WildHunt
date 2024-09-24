# WildHunt
A collection of python routines to aid in the hunt for quasars


## Installation 

- Step 1: Manually install the following packages (details below):
	- astro-datalab
	- pyqt
- Step 2: Install the package with `pip install -e .` in the main package directory 


### Install commands for the additional packages
#### astro-datalab 
"pip install --ignore-installed --no-cache-dir astro-datalab"

#### pyqt 
"pip install PyQt5"

### tables
Installation of tables under M1 Macs is a bit tricky.  One solution is to install hdf5 with brew and set the HDF5_DUR manually:

```
brew install hdf5
export HDF5_DIR=/opt/homebrew/opt/hdf5
```