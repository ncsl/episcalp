
Epilepsy Scalp Study
====================

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code style: black

Data Organization
-----------------

Data should be organized in the BIDS-iEEG/BIDS-EEG format:

https://github.com/bids-standard/bids-specification/blob/master/src/04-modality-specific-files/04-intracranial-electroencephalography.md


Installation Guide
==================

Setup environment from pipenv

.. code-block::

   pipenv install --dev

   # use pipenv to install private repo
   pipenv install -e git+git@github.com:adam2392/eztrack

   # or
   pipenv install -e /Users/adam2392/Documents/eztrack

   # if dev versions are needed
   pipenv install https://api.github.com/repos/mne-tools/mne-bids/zipball/master

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
