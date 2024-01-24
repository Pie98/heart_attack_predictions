from sklearn.utils import resample
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

#############################################################

# ------------------- undersamle_data ----------------------#

#############################################################

def undersample_data(df, target, random_state=42):
    # Finding the numerosity of each class
    df_class_0 = df[df[target] == 'No']
    df_class_1 = df[df[target] == 'Yes']

    #Undersampling
    num_instances_minority_class = len(df_class_1)
    df_class_0_undersampled = resample(df_class_0, replace=False, n_samples=num_instances_minority_class, random_state=random_state)
    df = pd.concat([df_class_0_undersampled, df_class_1])

    return df


#################################################################

# ------------------- preprocess_features ----------------------#

#################################################################


def preprocess_features(df, target, categorical_columns, numerical_columns, 
                        numerical_tranformer = 'min_max', test_size=0.07, valid_size=0.08):
    
    #Train test split
    Train_df, Test_df, Train_labels, Test_labels = train_test_split(df.drop(target, axis=1), df[target],
                                                                        test_size=test_size, random_state=41)

    Train_df, Valid_df, Train_labels, Valid_labels = train_test_split(Train_df, Train_labels, test_size=valid_size, random_state=41)

    Train_labels = Train_labels
    Valid_labels = Valid_labels
    Test_labels = Test_labels

    # encoding features
    if numerical_tranformer == 'min_max':
        ct_features = make_column_transformer(
            (MinMaxScaler(), numerical_columns),
            (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns )
        )
    elif numerical_tranformer == 'log_min_max':
        df[numerical_columns] = np.log(df[numerical_columns])
        ct_features = make_column_transformer(
            (MinMaxScaler(), numerical_columns),
            (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns )
        )
    
    ct_features.fit(Train_df)

    Train_df_encoded = ct_features.transform(Train_df)
    Valid_df_encoded = ct_features.transform(Valid_df)
    Test_df_encoded = ct_features.transform(Test_df)

    # Create an OrdinalEncoder
    ordinal_encoder = OneHotEncoder(sparse=False)
    # Training the OrdinalEncoder on Train_labels
    Train_labels_encoded = ordinal_encoder.fit(np.array(Train_labels).reshape(-1, 1))

    Train_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Train_labels).reshape(-1, 1)))
    Valid_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Valid_labels).reshape(-1, 1)))
    Test_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Test_labels).reshape(-1, 1)))  

    return (Train_df_encoded, Train_labels_encoded), (Valid_df_encoded, Valid_labels_encoded), (Test_df_encoded, Test_labels_encoded)

