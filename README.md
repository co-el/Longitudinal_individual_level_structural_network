# Longitudinal_individual_level_structural_network

## Description  

This repository contains code aiming at identifying (cross-sectional and longitudinal) networks of grey matter (GM) representing covariations between regional volumes using Independent Component Analysis (ICA) and Lasso. 
The process involves running ICA to find 20 GM networks in the training cohort. If you wish to investigate a different number of networks, you can change the n_components parameter. 
The identified networks are then backprojected for visualization. 
Once the network values are obtained with ICA from the training cohort, the training cohort is randomly split into 70-30 for training and testing a Lasso model, and models are then applied to the validation cohort. 
The input for the Lasso model includes the network values identified by ICA and regional GM volumes. 


## Required Input Files:

- A CSV file with a column containing the subject ID ('session_label') and the remaining columns containing regional GM volumes. If longitudinal data are used, each timepoint of each participant should occupy one row in the CSV and the timepoint information should be included in the subject ID. 

- The code includes a manual division of the cohort into training and validation sets. 

- For this study, we segmented and parcellated brain images using GIF. If using a different segmentation atlas (e.g., Freesurfer), modify the names of regions and the regions to include in the "Subsetting regional volumes to include GM ones" section.

- region_numb: A CSV file containing the names of regions and their corresponding numbers in the parcellation atlas.

- temp: T1 template space (MNI or subject-specific) used for backprojection and visualisation.

- path_imParc: Parcellated map in the same space as the provided template.

- root: Directory to store the outputs.


## Output Files:

- demographic_discovery.csv: CSV file with the subject IDs of participants included in the training cohort.
- replication_external_cohort.csv: CSV file with the subject IDs of participants included in the validation cohort.
- loading_ica_discovery.csv: Network values for each participant from the discovery/training cohort in each of the identified networks.
- regions_in_each_component_discovery.csv: List of the regions involved in each network.
- Pickle files of Lasso models.
- train_cohort_loading_predicted*.csv: CSV files for each network containing the network values obtained through Lasso for the train set of the training cohort.
- test_cohort_loading_predicted*.csv: CSV files for each network containing the network values obtained through Lasso for the test set of the training cohort.
- replication_cohort_predicted_loading*.csv: CSV files for each network containing the network values obtained through Lasso for the validation cohort.

Please refer to the documentation within the code for further details on execution and parameter customization.



## Disclaimer

This code is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.
