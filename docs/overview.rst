.. _overview:
.. currentmodule:: bayspar

##############
Quick Overview
##############

BAYSPAR (BAYesian SPAtially-varying Regression) is a Bayesian calibration model
for the TEX\ :sub:`86` temperature proxy.

There are two calibrations — one for sea-surface temperatures (SSTs) and one for subsurface (0-100 m) temperatures (subT) — and two methods to infer or predict past temperatures. The "standard" version of the model draws calibration parameters from the model gridpoint nearest to the core site to predict temperatures. The "Deep-Time" or "analog" version searches the calibration dataset for coretop TEX\ :sub:`86` values similar to those in the time series (within a user-set search tolerance) and uses the parameters from those analog locations to predict SSTs. The SST data in the calibration model are the statistical mean data from the `World Ocean Atlas 2009 <https://www.nodc.noaa.gov/OC5/WOA09/pr_woa09.html>`_.

For further details, refer to the `original publication in Geochimica et Cosmochimica Acta <https://doi.org/10.1016/j.gca.2013.11.026>`_, and the `updated calibration publication in Scientific Data <https://doi.org/10.1038/sdata.2015.29>`_.

The `original BAYSPAR MATLAB code <https://github.com/jesstierney/BAYSPAR>`_ is also available online.
