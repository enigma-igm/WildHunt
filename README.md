# WildHunt
A collection of python routines to aid in the hunt for quasars


## Installation 

You need to manually install the following packages:

### astro-datalab 
"pip install --ignore-installed --no-cache-dir astro-datalab"

### pyqt 
"pip install PyQt5"


Then you can install wildhunt with:
"pip install -e ." in the main package directory


Installation of tables under M1 Macs is a bit tricky.  One solution is to install hdf5 with brew and set the HDF5_DUR manually:

brew install hdf5

export HDF5_DIR=/opt/homebrew/opt/hdf5 
