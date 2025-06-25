# 🧠 T1map-for-Thalamus 🧠

This repository contains the source code for the paper:  
**"Segmenting Thalamic Nuclei: T1 Maps Provide a Reliable and Efficient Solution"**.


## 🔧 Data Processing Pipeline
The `src/data_processing` folder contains all scripts and dependencies for preprocessing T1-weighted MRI images.
We process paired **MPRAGE** and **FGATIR** images to generate quantitative **T1** and **PD maps**, and **multi-TI images** at specified inversion times (TIs). 
The entire pipeline has been packaged into a **Docker Image**, allowing users to run the full processing workflow on any pair of T1-weighted images.
The software will be released publicly upon acceptance of the paper.


## 🧪 Multi-TI Selection Module
The `src/multi-TI_selection` folder contains the code to identify the most informative 
TI images for thalamic nuclei segmentation.

## 🔬 Input Comparison Module
The `src/input_comparison` folder contains code to compare structural MRI sequences (MPRAGE, FGATIR) against quantitative maps (T1 maps, PD maps) and the selected multi-TI images.
It includes training, evaluation, and statistical validation scripts to assess segmentation performance across input types. 
