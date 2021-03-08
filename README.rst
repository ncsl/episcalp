
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

1. (Optional) If data is not in BIDS format yet, a BIDS conversion script should be carried out using `mne-bids`.

2. Raw data is stored in the `<bids_root>` of the data 
directory. Raw data format is in BrainVision, or EDF format.

3. ICA preprocessing is done in `eeglab`, where data is then output in the `.set` data format. 

4. ICA preprocessed data will be written to `<bids_root>/derivatives/` folder using `mne_bids` copy_eeglab function to convert to BIDS.
   
5. Further analysis will either start from the raw data, or from the ICA preprocessed data.

Installation Guide
==================

Setup environment from pipenv

.. code-block::

   pipenv install --dev

   # If this step hangs, you can skip the locking phase
   pipenv install --dev --skip-lock
   

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
