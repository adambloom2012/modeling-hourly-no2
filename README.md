## Modeling Hourly Ground Level NO2

This repository contains the code for the final project for Adam Bloom in the NCSU GeoAI course. This repository was originally cloned from the Toward Global Estimation of Ground-Level NO2 Pollution With Deep Learning and Remote Sensing paper that I am extending using a similar architecture but replacing a Sentinel 5p backbone with a TEMPO backbone, and adding additional features in order to effectively model hourly NO2 concentrations in the continental United States.

The submit_hourly_gpu.sh script in the satellite_model directory is used to train the model on the Hazel NC State HPC. The data used in this study is currently on that HPC and also in an external device. I've created a seperate repository with scripts that hit Copernicus and Earthdata API and process data used on this study which can be shared upon request. The data can also be made availablre upon request.
