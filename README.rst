Epilepsy Scalp Study
====================

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code style: black

Data Organization
-----------------

Data should be organized in the BIDS-iEEG/BIDS-EEG format:

https://github.com/bids-standard/bids-specification/blob/master/src/04-modality-specific-files/04-intracranial-electroencephalography.md


High-level Pipeline
-------------------

We will consistently obtain data from manufacturing, or EDF format. These files however, are not
sufficient for us to analyze the scalp EEG data. We require additional metadata, which will be
facilitated by following the BIDS protocol.

1. Raw EDF data is initially to be stored in the `<bids_root>/sourcedata/` folder.

2. BIDS-conversion (source EDF data): Using `mne-bids`, we will convert data to BIDS format of the raw EEG data.

3. (Temporary - Matlab Auto-ICA) Since auto-ICA currently sits in EEGLab with the MATLAB module, we can run auto-ICA on 
the raw EDF data. During this, data is notch filtered, and bandpass filtered between 1-30 Hz (to remove
higher frequency muscle artifacts). Then auto-ICA will try to remove stereotypical artifacts.
   - ICA preprocessing is done in `eeglab`, where data is then output in the `.set` data format. 
   - episcalp/preprocess/matlab/run_me.m
   
4. BIDS-conversion (ICA-cleaned EDF data): Using `mne-bids`, we will convert ICA-cleaned-data to BIDS format of the raw EEG data. Data
will be stored in EDF format.


4. ICA preprocessed data will be written to `<bids_root>/derivatives/` folder using `mne_bids` copy_eeglab function to convert to BIDS.
    episcalp/bids/run_bids_conversion_ica.py.
   
5. Further analysis will either start from the raw data, or from the ICA preprocessed data.

6. Feature generation code is stored in a subfolder of episcalp

7. Scripts that assist in IO of intermediate results (specifically for notebooks) is located within sample_code

High level details that remain to be sorted out are the inclusion of Persyst spikes:

- should we perform Persyst spike detection before, or after ICA cleaning?


Installation Guide
==================

Setup environment from pipenv

.. code-block::

    # create virtual environment
   python3.8 -m venv .venv

   pipenv install --dev

   # if dev versions are needed
   pip install https://api.github.com/repos/mne-tools/mne-python/zipball/master
   pipenv install https://api.github.com/repos/mne-tools/mne-bids/zipball/master
   pip install https://api.github.com/repos/mne-tools/mne-connectivity/zipball/master
   # pip install for oblique random forests
   pip install https://api.github.com/repos/neurodatadesign/manifold_random_forests/zipball/master
   
If you're using some private repos, such as ``eztrack``, here's some helper code
for installing.

.. code-block::

   # use pipenv to install private repo
   pipenv install -e git+git@github.com:adam2392/eztrack

   # or
   pipenv install -e /Users/adam2392/Documents/eztrack

Organization Guide
------------------

Development should occur in 3 main steps:

1. BIDS organization: any scripts to convert datasets and test BIDs-compliance should go into `bids/`.

2. Analysis: Each type of analysis should have their own directory


Setup Jupyter Kernel To Test
============================

You need to install ipykernel to expose your environment to jupyter notebooks.

.. code-block::

   python -m ipykernel install --name episcalp --user
   # now you can run jupyter lab and select a kernel
   jupyter lab
