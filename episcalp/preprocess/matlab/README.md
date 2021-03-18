# Artifact Removal Using EEGLABS ICA
This portion of prepreprocessing is performed in MATLAB. Download the latest version [here](https://www.mathworks.com/downloads/)

## Installation Guide
EEGLAB requires some setup. First, install the package from [here](https://sccn.ucsd.edu/eeglab/download.php). The 
default ICA algorithm is runICA, which is a Matlab implementation of [infomax](https://www.tqmp.org/RegularArticles/vol06-1/p031/p031.pdf).
This algorithm is well regarded, but rather slow. A faster version of the algorithm, binICA,  has been developed, but is not
available out of the box. 

The setup is:
1. Based on your OS, download the binICA files [here](https://sccn.ucsd.edu/wiki/Binica)
2. Locate the path to the eeglab directory, which we will call <eeglab_path>
3. Extract the zipped folder from 1 into the folder `<eeglab_path>/functions/supportfiles/`
4. Check the file `<eeglab_path>/functions/sigprocfuncs/icadefs.m` to make sure the variable `ICABINARY` points to the
extracted location

If you fail to do this, this package will default to the slower version of the algorithm.

**Note** that binICA returns less information than the default runICA, but the other outputs are not currently used in
this analysis. 

## How to Use
This package achieves its goal in 2 steps, which are wrapped in run_me.m:
1. Apply frequency based filters. A notch filter around the powerline frequency and a bandpass filter from 0.5-90Hz are
applied.
2. Artifacts are removed based on the output from Independent Component Analysis.

To run the complete pipeline, follow these steps:
1. Put all the source files in one folder, e.g. `sourcedata/unfiltered`
2. Define an output directory for the filtered file, e.g. `sourcedata/filtered`
3. Modify EEGLabPath, folder (from 1), outputdir (from 2) and group_size in run_me.m`
4. Run the script
