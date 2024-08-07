import glob
import os
import pathlib
import pickle

import nibabel as nb

from nilearn.plotting import plot_stat_map, show

import numpy as np

import pandas as pd

from sklearn.decomposition import FastICA
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from constants import GM_REGIONS


def process_cohort(dataFrameOut: pd.DataFrame, region_numb: pd.DataFrame, template: pathlib.Path, parcellated_map, output_folder: pathlib.Path):
    #Subsetting regional volumes to include GM ones
    df= dataFrameOut[GM_REGIONS]  #selecting the GM regions to keep from the atlas


    #In the final dataframe I keep just regions of interest and participants ID.
    #I need participants ID here to get back whose those values belongs to

    df1 = df.dropna(axis=0) #dropping NaN values


    #Defining variables for later 
    subj_list = df1['session_label'].tolist()
    print (len(subj_list)) #subj_list which I'll need later
    subj_list_ = pd.DataFrame(subj_list)
    col_name= df1.columns.tolist()
    col = pd.DataFrame(col_name) #matrix with col names to be used later
    col.columns = ['REGION_Label']#to be consistent and be able to merge later 
    col['REGION_Label'] = col['REGION_Label'].astype(str).str.replace('Vol_prob_', '') #to match the parcellation file where I don't have them


    region_numb['REGION_Label'] = region_numb['REGION_Label'].astype(str).str.replace(' ', '_')
    region_numb.head()

    col = pd.merge (col, region_numb, on='REGION_Label')
    
    col_to_keep = col.REGION_Label.tolist ()
    df1.columns = df1.columns.str.replace("Vol_prob_", "")
    col_to_keep.append ('session_label')
    df_ica_subj = df1[df1.columns.intersection(col_to_keep)] #I'll need this later on for the replication stage
    df_ica = df_ica_subj.iloc[:, 1:]
    

    pd.DataFrame.to_csv(df_ica, os.path.join(output_folder, 'demographic_discovery.csv'))



    #Running ICA on the training cohort
    np.random.RandomState(0)
    df = df.reset_index()
    ica = FastICA(n_components= 20, algorithm= 'parallel', fun= 'logcosh', random_state=0 ) #random_state=42
    components_masked = ica.fit_transform(df_ica.T).T 
    # Unused? comp = ica.components_
    loading = pd.DataFrame(ica.mixing_)
    loading_ = pd.concat([subj_list_.reset_index(drop=True), loading.reset_index(drop=True)], axis=1)
    loading_.columns=['subj_id', 'v1','v2','v3','v4', 'v5', 
                            'v6', 'v7', 'v8', 'v9', 'v10', 
                    'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20']


    pd.DataFrame.to_csv(loading_, os.path.join(output_folder, 'loading_ica_discovery.csv'))
    print (loading_.shape)
    print (loading_)


    #Visualize components identified from ICA from the training group
    # Normalize estimated components, for thresholding to make sense
    components_masked -= components_masked.mean(axis=0)
    components_masked /= components_masked.std(axis=0)

    # Threshold
    components_masked[np.abs(components_masked) < .8] = 0
    #####components_masked[np.abs(components_masked) < .8] = 
    a = components_masked.T
    a.size
    
    data= pd.DataFrame(a)
    data_wregion_names =  pd.concat([col.reset_index(drop=True), data.reset_index(drop=True)], axis=1)
    data_wregion_names #matrix with name of the regions and value in each component 
    data_wregion_names.columns=['region', 'region_number','v1','v2','v3','v4', 'v5', 
                            'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 
                            'v19', 'v20']
    data_wregion_names['region'] = data_wregion_names['region'].astype(str).str.replace('Vol_cat_', '')
    data_wregion_names['region'] = data_wregion_names['region'].astype(str).str.replace('Vol_prob_', '')

    data_wregion_names


    #For visualization purposes: 
    data_wregion_names['v1'] = np.where(data_wregion_names['v1'].between(-1.5,1.5), 0, data_wregion_names['v1'])
    data_wregion_names['v2'] = np.where(data_wregion_names['v2'].between(-1.5,1.5), 0, data_wregion_names['v2'])
    data_wregion_names['v3'] = np.where(data_wregion_names['v3'].between(-1.5,1.5), 0, data_wregion_names['v3'])
    data_wregion_names['v4'] = np.where(data_wregion_names['v4'].between(-1.5,1.5), 0, data_wregion_names['v4'])
    data_wregion_names['v5'] = np.where(data_wregion_names['v5'].between(-1.5,1.5), 0, data_wregion_names['v5'])
    data_wregion_names['v6'] = np.where(data_wregion_names['v6'].between(-1.5,1.5), 0, data_wregion_names['v6'])
    data_wregion_names['v7'] = np.where(data_wregion_names['v7'].between(-1.5,1.5), 0, data_wregion_names['v7'])
    data_wregion_names['v8'] = np.where(data_wregion_names['v8'].between(-1.5,1.5), 0, data_wregion_names['v8'])
    data_wregion_names['v9'] = np.where(data_wregion_names['v9'].between(-1.5,1.5), 0, data_wregion_names['v9'])
    data_wregion_names['v10'] = np.where(data_wregion_names['v10'].between(-1.5,1.5), 0, data_wregion_names['v10'])
    data_wregion_names['v11'] = np.where(data_wregion_names['v11'].between(-1.5,1.5), 0, data_wregion_names['v11'])
    data_wregion_names['v12'] = np.where(data_wregion_names['v12'].between(-1.5,1.5), 0, data_wregion_names['v12'])
    data_wregion_names['v13'] = np.where(data_wregion_names['v13'].between(-1.5,1.5), 0, data_wregion_names['v13'])
    data_wregion_names['v14'] = np.where(data_wregion_names['v14'].between(-1.5,1.5), 0, data_wregion_names['v14'])
    data_wregion_names['v15'] = np.where(data_wregion_names['v15'].between(-1.5,1.5), 0, data_wregion_names['v15'])
    data_wregion_names['v16'] = np.where(data_wregion_names['v16'].between(-1.5,1.5), 0, data_wregion_names['v16'])
    data_wregion_names['v17'] = np.where(data_wregion_names['v17'].between(-1.5,1.5), 0, data_wregion_names['v17'])
    data_wregion_names['v88'] = np.where(data_wregion_names['v88'].between(-1.5,1.5), 0, data_wregion_names['v88'])
    data_wregion_names['v19'] = np.where(data_wregion_names['v19'].between(-1.5,1.5), 0, data_wregion_names['v19'])
    data_wregion_names['v20'] = np.where(data_wregion_names['v20'].between(-1.5,1.5), 0, data_wregion_names['v20'])

    pd.DataFrame.to_csv(data_wregion_names, os.path.join(output_folder, 'regions_in_each_component_discovery.csv')) #file to get which regions were most involved in each network


    #Visualisation 

    path_imParc = parcellated_map #TO BE PROVIDED importing parcellated map
    temp = nb.load(template) # TO BE PROVIDED importing template in the same space of the parcellated map
    parcel_image = nb.load(path_imParc)
    parc_as_int = parcel_image.get_fdata().astype(np.int32)
    reg_keep = data_wregion_names['region_number'].astype(np.int32).tolist()
    heatmap = np.zeros_like(parcel_image.get_fdata())
    
    h = parc_as_int [parc_as_int > 0]
    h = h.astype(np.int32).tolist()
    h= set (h)
    h= list(h)
        
    for i in h:
        #print(type(i))
        if i not in reg_keep: 
            #print('no')
            parc_as_int[parc_as_int == i] = 0
            
            
    region_col = data_wregion_names.loc[:, 'region']
    #print (region_col)
    for column in data_wregion_names.columns[2:]:
        heatmap = np.zeros_like(parcel_image.get_fdata())
        
        name_output = output_folder + str(column) + '_longitudinal_ica.nii.gz'
        subset = data_wregion_names[["region", "region_number", column]]

        for i, row in data_wregion_names.iterrows(): 
            n_region = int(row['region_number'])
            heatmap[parc_as_int == n_region] = float(row[column])
        heatmap_image= nb.Nifti1Image(heatmap, parcel_image.affine, parcel_image.header)
        nb.save(heatmap_image, name_output)


    masked_compoents = glob.glob(os.path.join( output_folder, "v*.nii.gz")) 
    temp = nb.load(template)


    for f in masked_compoents: 
        component = f
        niftiImage = f.split('/')[-1]
        no_compo = niftiImage.split('_')[0]
        img_file = nb.load(f)
        img_array = img_file.get_data() 
        img_array[ ( ( img_array >= -1.5) & (img_array <= 1.5) ) ] = 0   
        np.any(img_array <= 0.8)
        affine = img_file.affine
        img = nb.Nifti1Image(img_array, affine)
        final_name = 'thresholded_masked_ICA_GM_' + no_compo + '.nii.gz'
        final= os.path.join (output_folder, final_name)
        nb.save(img, final)   


    #save as jpeg
    temp = nb.load(template)
    thres_compoents = glob.glob(os.path.join( output_folder, "thresholded_masked_ICA_GM_*.nii.gz"))
    for f in thres_compoents: 
        component = f
        niftiImage = f.split('/')[-1]
        
        no_compo = niftiImage.split('_')[-1]
        no_compo = no_compo.split('.')[0]
        num = no_compo.split('v')[1]
        #print (no_component)
        name = "Network " + num
        #mean_img = image.mean_img (scans2)
        a= plot_stat_map(component, temp, dim= -0.5, title= name)
        show()
        name = 'thresholded_masked_ICA_GM_' + no_compo + '.png'
        a.savefig (os.path.join (output_folder, name))


    #Lasso 
    
    #validation_df =  df[GM_REGIONS] 
    df1_validation = df.dropna(axis= 0) #dropping NaN values
    col_name = df1_validation.columns.tolist()
    col = pd.DataFrame(col_name) #matrix with col names to be used later
    col.columns = ['REGION_Label']#to be consistent and be able to merge later 
    col['REGION_Label'] = col['REGION_Label'].astype(str).str.replace('Vol_prob_', '') #to match the parcellation file where I don't have them
    
    region_numb['REGION_Label'] = region_numb['REGION_Label'].astype(str).str.replace(' ', '_')
    region_numb.head()

    col = pd.merge (col, region_numb, on='REGION_Label')
    
    col_to_keep = col.REGION_Label.tolist ()


    df1_validation.columns = df1_validation.columns.str.replace("Vol_prob_", "")
    col_to_keep.append ('session_label')
    df1_validation_subj = df1_validation[df1_validation.columns.intersection(col_to_keep)] 
    df1_validation = df1_validation.iloc[:, 1:]
    pd.DataFrame.to_csv(df1_validation_subj, os.path.join(output_folder, 'replication_external_cohort.csv'))


    list_disc = list(df_ica.columns)
    list_rep = list(df1_validation.columns)
    
    #Validation method  for each component
    ##Here I am creating pickles, applying and validating the model

    #Defining list of ica components to go through
    list_of_components = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14','v15', 'v16', 'v17', 'v18', 'v19', 'v20'] 
    
    X= df_ica_subj #discovery cohort - reports subj id, vol measure for each region
    
    for i in list_of_components: 
        print ('Starting analysis for: ', i)
        y = loading_[str(i)]
        
        #create training and testing 
        X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state= 0)
        id_train = pd.DataFrame (X_train.iloc [:, 0:1])#saving id to dataframe
        X_train = X_train.iloc[:, 1:] #removing the id column
        id_test = pd.DataFrame (X_test.iloc [:, 0:1])#saving id to dataframe
        X_test = X_test.iloc[:, 1:] #removing the id column
        
        #from sklearn.preprocessing.StandardScaler import class
        lasso = Lasso (alpha = 0.0, fit_intercept=True, normalize= True  )
        
        #Fit model 
    
        lasso.fit(X_train, y_train)
        model_name = output_folder + "model_" + i + ".pkl" 
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
    
        
        #Saving loadings and predicted loading for stats analysis ICC
    
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
        save_train = pd.DataFrame(list(zip(subject, y_train, pred_train_lasso)),
                   columns = ['session_label', col_load, col_pred])
        saving_name_train = output_folder + "train_cohort_loading_predicted_" + i + ".csv"
        save_train.to_csv(saving_name_train)
        
    
        ########
        ######## Saving testing data 
        ########
    
        
        subject_test = id_test.values.tolist()
        
        save_test = pd.DataFrame(list(zip(subject_test, y_test, pred_test_lasso)), columns =['session_label', col_load, col_pred])
        saving_name_test = output_folder + "test_cohort_loading_predicted_" + i + ".csv"
        save_test.to_csv(saving_name_test)
        
        #############
        ############# Replication cohort
        #############
    
        #Predict loading for replication cohort and save it 
    
        replication_cohort = (df1_validation)
        predicted_loading_replication_cohort = lasso.predict(replication_cohort)
        id_replication = df1_validation_subj.iloc[:, 0]
        col_pred_id = "predicted_loading_" + i 
        predicted_loading_replication_cohort_save = pd.DataFrame(list(zip(id_replication, predicted_loading_replication_cohort)),
                   columns =['session_label', col_pred_id])
        save_name_pred = output_folder + "replication_cohort_predicted_loading_" + i + ".csv"
    
        predicted_loading_replication_cohort_save.to_csv(save_name_pred)
    
    
   
    



