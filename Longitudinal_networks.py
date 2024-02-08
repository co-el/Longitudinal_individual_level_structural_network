#Longitudinal_networks
#This code is meant for people who have a large database and want to apply the same technique to get longitudinal networks
# We will specify that if they want to use other regions/atlas, they need to change the selected regions below and the parcellated image for the back projection. 

#Importing libraries

import argparse
import os, glob
import numpy as np
import pandas as pd
import pickle
import nibabel as nb
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
np.random.RandomState(0)
from nilearn import image
from nilearn.plotting import (
    plot_stat_map,
    show
)

from processcohort import process_cohort


parser = argparse.ArgumentParser(description='ICA on longitudinal data')
parser.add_argument('--input', type=str, help='Input CSV with the participants and regional volumes')
args = parser.parse_args()

### Defining the input matrix for ICA
# A csv with one column with the subject ID and the remaining with the volume of brain regions
df = pd.read_csv(args.input) # TO BE PROVIDED. cvs with participants as rows, brain regional volumes as cols

train_df, test_df = train_test_split(df, test_size=0.3, random_state=0) #splitting the data into training and testing
        
process_cohort(train_df)
    
    
    
    
    
#Define external cohort - previous variables have been overwritten from here on
df = pd.read_csv('/data/elisa/SCRIPT/2019-07-23_02-38-42_df_completed.csv') #TO BE PROVIDED. reloading the same cvs that I loaded for the traiing cohort
dataFrame_opera1 = df[df['session_label'].str.contains('OPERA1')] #keep just participants from OPERA1
dataFrame_opera2 = df[df['session_label'].str.contains('OPERA2')] #keep just participants from OPERA2
dataFrame_oratorio = df[df['session_label'].str.contains('ORATORIO')] #keep just participants from ORATORIO - this approach worked for my study because I had to subset based on positive and negative trials. For other cohorts, as suggested at the beginning, could be best to split the 70% of the cohort to training and the remaining to the external

dataFrameOut = pd.concat ([dataFrame_opera2, dataFrame_opera1])
dataFrameOut = pd.concat ([dataFrameOut, dataFrame_oratorio])

dataFrameOut = dataFrameOut.drop_duplicates(subset= "session_label")


#As for the training cohort, subsetting the GM regions
#subsetting just regions of interest and participants' ID.

df= dataFrameOut[[ 'session_label', 
                  'Vol_prob_Brain_Stem',
                  'Vol_prob_Brain_Stem_and_Pons', 
                 'Vol_prob_Cerebellar_Vermal_Lobules_I-V',
 'Vol_prob_Cerebellar_Vermal_Lobules_VI-VII',
 'Vol_prob_Cerebellar_Vermal_Lobules_VIII-X',
'Vol_prob_Left_Accumbens_Area', 'Vol_prob_Right_Accumbens_Area', 
'Vol_prob_Left_ACgG_anterior_cingulate_gyrus','Vol_prob_Right_ACgG_anterior_cingulate_gyrus', 
                  'Vol_prob_Left_Amygdala',
                  'Vol_prob_Right_Amygdala', 
                  'Vol_prob_Left_AnG_angular_gyrus',
                  'Vol_prob_Right_AnG_angular_gyrus',
                  'Vol_prob_Left_Basal_Forebrain',
                  'Vol_prob_Right_Basal_Forebrain',
                  'Vol_prob_Left_CO_central_operculum',
                  'Vol_prob_Right_CO_central_operculum',
                  'Vol_prob_Left_Calc_calcarine_cortex',
                  'Vol_prob_Right_Calc_calcarine_cortex',
                  'Vol_prob_Right_TrIFG_triangular_part_of_the_inferior_frontal_gyrus', 
                  'Vol_prob_Left_TrIFG_triangular_part_of_the_inferior_frontal_gyrus', 
                  'Vol_prob_Right_Thalamus_Proper', 
                  'Vol_prob_Left_Thalamus_Proper', 
                  'Vol_prob_Right_TTG_transverse_temporal_gyrus',
                  'Vol_prob_Left_TTG_transverse_temporal_gyrus',
                  'Vol_prob_Right_TMP_temporal_pole',
                  'Vol_prob_Left_TMP_temporal_pole',
                  'Vol_prob_Pons',
                  'Vol_prob_Right_POrG_posterior_orbital_gyrus',
                  'Vol_prob_Left_POrG_posterior_orbital_gyrus',
                  'Vol_prob_Right_PP_planum_polare',
                  'Vol_prob_Left_PP_planum_polare',
                  'Vol_prob_Right_PT_planum_temporale',
                  'Vol_prob_Left_PT_planum_temporale',
                  'Vol_prob_Right_PoG_postcentral_gyrus',
                  'Vol_prob_Left_PoG_postcentral_gyrus',
                  'Vol_prob_Right_PrG_precentral_gyrus',
                  'Vol_prob_Left_PrG_precentral_gyrus',
                  'Vol_prob_Right_STG_superior_temporal_gyrus',
                  'Vol_prob_Left_STG_superior_temporal_gyrus',
                  'Vol_prob_Right_SPL_superior_parietal_lobule',
                  'Vol_prob_Left_SPL_superior_parietal_lobule',
                  'Vol_prob_Right_SOG_superior_occipital_gyrus',
                  'Vol_prob_Left_SOG_superior_occipital_gyrus',
                  'Vol_prob_Right_SMG_supramarginal_gyrus',
                  'Vol_prob_Left_SMG_supramarginal_gyrus',
                  'Vol_prob_Right_SMC_supplementary_motor_cortex',
                  'Vol_prob_Left_SMC_supplementary_motor_cortex',
                  'Vol_prob_Right_SFG_superior_frontal_gyrus',
                  'Vol_prob_Left_SFG_superior_frontal_gyrus',
                  'Vol_prob_Right_SCA_subcallosal_area',
                  'Vol_prob_Left_SCA_subcallosal_area',
                  'Vol_prob_Right_Putamen',
                  'Vol_prob_Left_Putamen',
                  'Vol_prob_Right_PO_parietal_operculum',
                  'Vol_prob_Left_PO_parietal_operculum',
                  'Vol_prob_Left_Caudate',
                  'Vol_prob_Right_Caudate',
                  'Vol_prob_Right_PIns_posterior_insula',
                  'Vol_prob_Left_PIns_posterior_insula',
                  'Vol_prob_Left_Pallidum',
                  'Vol_prob_Right_Pallidum',
                  'Vol_prob_Right_PCu_precuneus',
             'Vol_prob_Right_PHG_parahippocampal_gyrus',
                  'Vol_prob_Left_PCu_precuneus',
 'Vol_prob_Left_PHG_parahippocampal_gyrus',
                  'Vol_prob_Right_PCgG_posterior_cingulate_gyrus',
                  'Vol_prob_Left_PCgG_posterior_cingulate_gyrus',
                  'Vol_prob_Right_MFC_medial_frontal_cortex',
                  'Vol_prob_Left_MFC_medial_frontal_cortex',
                  'Vol_prob_Right_OrIFG_orbital_part_of_the_inferior_frontal_gyrus',
                  'Vol_prob_Left_OrIFG_orbital_part_of_the_inferior_frontal_gyrus',
 'Vol_prob_Right_OpIFG_opercular_part_of_the_inferior_frontal_gyrus',
 'Vol_prob_Left_OpIFG_opercular_part_of_the_inferior_frontal_gyrus',
 'Vol_prob_Right_OFuG_occipital_fusiform_gyrus',
                  'Vol_prob_Left_OFuG_occipital_fusiform_gyrus',
 'Vol_prob_Right_OCP_occipital_pole',
                  'Vol_prob_Left_OCP_occipital_pole',
                  'Vol_prob_Right_MTG_middle_temporal_gyrus',
                  'Vol_prob_Left_MTG_middle_temporal_gyrus',
                  'Vol_prob_Right_MSFG_superior_frontal_gyrus_medial_segment',
                  'Vol_prob_Left_MSFG_superior_frontal_gyrus_medial_segment',
 'Vol_prob_Right_MPrG_precentral_gyrus_medial_segment',
                  'Vol_prob_Left_MPrG_precentral_gyrus_medial_segment',
 'Vol_prob_Right_MPoG_postcentral_gyrus_medial_segment',
                  'Vol_prob_Left_MPoG_postcentral_gyrus_medial_segment',
 'Vol_prob_Right_MOrG_medial_orbital_gyrus',
                  'Vol_prob_Left_MOrG_medial_orbital_gyrus',
 'Vol_prob_Right_MOG_middle_occipital_gyrus',
                  'Vol_prob_Left_MOG_middle_occipital_gyrus',
 'Vol_prob_Right_MFG_middle_frontal_gyrus',
                  'Vol_prob_Left_MFG_middle_frontal_gyrus',
 'Vol_prob_Right_MCgG_middle_cingulate_gyrus',
                  'Vol_prob_Left_MCgG_middle_cingulate_gyrus',
 'Vol_prob_Right_LOrG_lateral_orbital_gyrus',
                  'Vol_prob_Left_LOrG_lateral_orbital_gyrus',
                  'Vol_prob_Right_ITG_inferior_temporal_gyrus',
                  'Vol_prob_Left_ITG_inferior_temporal_gyrus',
                  'Vol_prob_Right_IOG_inferior_occipital_gyrus',
                  'Vol_prob_Left_IOG_inferior_occipital_gyrus',
                  'Vol_prob_Right_Hippocampus',
                  'Vol_prob_Left_Hippocampus',
                  'Vol_prob_Right_GRe_gyrus_rectus',
                  'Vol_prob_Left_GRe_gyrus_rectus',
                  'Vol_prob_Right_FuG_fusiform_gyrus',
                  'Vol_prob_Left_FuG_fusiform_gyrus',
                  'Vol_prob_Right_FRP_frontal_pole',
                  'Vol_prob_Left_FRP_frontal_pole',
                  'Vol_prob_Right_FO_frontal_operculum',
                  'Vol_prob_Left_FO_frontal_operculum',
                  'Vol_prob_Right_Ent_entorhinal_area',
                  'Vol_prob_Left_Ent_entorhinal_area',
                  'Vol_prob_Right_Cun_cuneus',
                  'Vol_prob_Left_Cun_cuneus',
                  'Vol_prob_Right_Claustrum',
                  'Vol_prob_Left_Claustrum',
                  'Vol_prob_Right_Cerebellum_Exterior',
                  'Vol_prob_Left_Cerebellum_Exterior',
                  'Vol_prob_Right_AOrG_anterior_orbital_gyrus',
                  'Vol_prob_Left_AOrG_anterior_orbital_gyrus',
                  'Vol_prob_Right_AIns_anterior_insula',
                  'Vol_prob_Left_AIns_anterior_insula',
                  'Vol_prob_Left_LiG_lingual_gyrus',
                  'Vol_prob_Right_LiG_lingual_gyrus' ]] 

df1_test = df.dropna(axis= 0) #dropping NaN values
col_name= df1_test.columns.tolist()
col = pd.DataFrame(col_name) #matrix with col names to be used later
col.columns = ['REGION_Label']#to be consistent and be able to merge later 
col['REGION_Label'] = col['REGION_Label'].astype(str).str.replace('Vol_prob_', '') #to match the parcellation file where I don't have them
print ("shape col", col.shape)
type(col)
print (col)


region_numb = pd.read_csv('/data/elisa/Parcellated_regions_ica_modified.csv')
region_numb['REGION_Label'] = region_numb['REGION_Label'].astype(str).str.replace(' ', '_')
region_numb.head()

col = pd.merge (col, region_numb, on='REGION_Label')
print (col)
print ("shape col", col.shape)
type(col)

col_to_keep = col.REGION_Label.tolist ()


df1_test.columns = df1_test.columns.str.replace("Vol_prob_", "")
col_to_keep.append ('session_label')
df1_test_subj = df1_test[df1_test.columns.intersection(col_to_keep)] 
df1_test = df1_test_subj.iloc[:, 1:]
print (df1_test.head())  #Remove the ID column from the dataframe to create the final input to be used to run ICA
print (shape(df1_test))



pd.DataFrame.to_csv(df1_test_subj, '/data/elisa/RESULTS/Longitudinal_project/Dec2023/replication_external_cohort.csv')


list_disc = list(df_ica.columns)
list_rep = list(df1_test.columns)
print(shape(df1_test))
print (len(list_disc), len(list_rep))

    


#Validation method  for each component
#Here I am creating pickles, applying and validating the model

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle

#Defining list of ica components to go through
list_of_components = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14','v15', 'v16', 'v17', 'v88', 'v19', 'v20'] #CAPISCI XK V18 non andato

X= df_ica_subj #discovery cohort - reports subj id, vol measure for each region

for i in list_of_components: 
    print ('Starting analysis for: ', i)
    y = loading_[str(i)]
    print (y)
    

    
    #create training and testing 
    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state= 0)
    id_train = pd.DataFrame (X_train.iloc [:, 0:1])#saving id to dataframe
    print (len(id_train))
    print(id_train)  
    X_train = X_train.iloc[:, 1:] #removing the id column
    id_test = pd.DataFrame (X_test.iloc [:, 0:1])#saving id to dataframe
    print (len(id_test))
    X_test = X_test.iloc[:, 1:] #removing the id column
    
    #from sklearn.preprocessing.StandardScaler import class
    lasso = Lasso (alpha = 0.0, fit_intercept=True, normalize= True  )
    
    
    #Fit model 

    lasso.fit(X_train, y_train)
    model_name = "/data/elisa/RESULTS/Longitudinal_project/Dec2023/model_" + i + ".pkl" 
    print('The model is saved as: ', model_name)
    pickle.dump(lasso, open(model_name, 'wb'))
    
    #Create model score

    print (lasso.score(X_test, y_test))
    print (lasso.score(X_train, y_train))
    print (lasso.coef_)
    
    #Quality checking results for missing values

    cleanedList = [x for x in y if str(x) != 'NaN']
    cleanedList = [x for x in y_test if str(x) != 'NaN']
    cleanedList = [x for x in y_train if str(x) != 'NaN']
    
    
    #predicting loading values in the training cohort

    pred_train_lasso= lasso.predict(X_train)
    print ('The shape is: ', len(pred_train_lasso))
    ##print ('The shape of the matrix for the predicted loadings in the train sample is: ', shape(pred_train_lasso))
    pred_test_lasso= lasso.predict(X_test)
    ##print ('The shape of the matrix for the predicted loadings in the test sample is: ', shape(pred_test_lasso))

    #evaluating the performance of LASSO with mean squared error and R2
    print('The mean squared error for the ica loading vs. predicted loading in the train cohort is: ', np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
    print('The R2 score for the ica loading vs. predicted loading in the train cohort is: ', r2_score(y_train, pred_train_lasso))
    print('The mean squared error for the ica loading vs. predicted loading in the test cohort is: ', np.sqrt(mean_squared_error(y_test,pred_test_lasso)))
    print('The R2 score for the ica loading vs. predicted loading in the test cohort is: ', r2_score(y_test, pred_test_lasso))

    
    #saving loadings and predicted loading for stats analysis ICC



    ########
    ######## Saving training data 
    ########

    #print ('Original list is long: ', len(y_train))
    #checking for NaN
    #cleanedList = [x for x in y_train if str(x) != "NaN"] #checking for NaN
    #print ('list without NaN is long: ', len(cleanedList))
    col_load = str("ica_loading_" + i)
    col_pred = str("predicted_loading_" + i)
    subject = id_train.values.tolist()
    print(id_train)
    print(y_train)
    print(pred_train_lasso)
    save_train = pd.DataFrame(list(zip(subject, y_train, pred_train_lasso)),
               columns = ['session_label', col_load, col_pred])
    print(save_train)
    saving_name_train = "/data/elisa/RESULTS/Longitudinal_project/Dec2023/train_cohort_loading_predicted_" + i + ".csv"
    save_train.to_csv(saving_name_train)
    
    
    
    
    ########
    ######## Saving testing data 
    ########

    ##print ('Original list is long: ', len(y_test))
    cleanedList = [x for x in y_test if str(x) != "NaN"] #checking for NaN
    ##print ('list without NaN is long: ', len(cleanedList))
    
    subject_test = id_test.values.tolist()
    print(id_test)
    print(pred_test_lasso)
    
    save_test = pd.DataFrame(list(zip(subject_test, y_test, pred_test_lasso)), columns =['session_label', col_load, col_pred])
    print (save_test)
    saving_name_test = "/data/elisa/RESULTS/Longitudinal_project/Dec2023/test_cohort_loading_predicted_" + i + ".csv"
    save_test.to_csv(saving_name_test)
    
    #############
    ############# Replication cohort
    #############

    #Predict loading for replication cohort and save it 

    replication_cohort = (df1_test)
    print (shape(replication_cohort))
    predicted_loading_replication_cohort = lasso.predict(replication_cohort)
    print(len(predicted_loading_replication_cohort))
    print(predicted_loading_replication_cohort)
    print (shape(predicted_loading_replication_cohort))
    id_replication = df1_test_subj.iloc[:, 0]
    col_pred_id = "predicted_loading_" + i 
    predicted_loading_replication_cohort_save = pd.DataFrame(list(zip(id_replication, predicted_loading_replication_cohort)),
               columns =['session_label', col_pred_id])
    print (predicted_loading_replication_cohort_save)
    save_name_pred = "/data/elisa/RESULTS/Longitudinal_project/Dec2023/replication_cohort_predicted_loading_" + i + ".csv"

    predicted_loading_replication_cohort_save.to_csv(save_name_pred)


subj_list_replication = df1_test_subj['session_label'].tolist()
print (len(subj_list_replication)) #subj_list which I'll need later
subj_list_replication_ = pd.DataFrame(subj_list_replication)




    
