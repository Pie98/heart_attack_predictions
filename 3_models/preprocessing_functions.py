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


def preprocess_features(df, target, categorical_columns, numerical_columns):
    #Train test split
    Train_df, Test_df, Train_labels, Test_labels = train_test_split(df.drop(target, axis=1), df[target],
                                                                        test_size=0.07, random_state=41)

    Train_df, Valid_df, Train_labels, Valid_labels = train_test_split(Train_df, Train_labels, test_size=0.08, random_state=41)

    # encoding categorical features
    teams_transf = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), numerical_columns),
    sparse_threshold=0  
    )

    teams_transf.fit(Train_df)

    Train_teams_encoded = teams_transf.transform(Train_df)
    Valid_teams_encoded = teams_transf.transform(Valid_df)
    Test_teams_encoded = teams_transf.transform(Test_df)

    #encoding numerical features
    Train_dict_features_norm = Train_dict_features.copy()
    Valid_dict_features_norm = Valid_dict_features.copy()
    Test_dict_features_norm = Test_dict_features.copy()

    for feature in list(Train_dict_features.keys()):
        feature_list = Train_dict_features_norm[feature].columns
        numerical_transf = make_column_transformer(
            (MinMaxScaler(), feature_list), 
            sparse_threshold=0  
        )
        numerical_transf.fit(Train_dict_features[feature])
    Train_dict_features_norm[feature] = numerical_transf.transform(Train_dict_features_norm[feature])
    Valid_dict_features_norm[feature] = numerical_transf.transform(Valid_dict_features_norm[feature])
    Test_dict_features_norm[feature] = numerical_transf.transform(Test_dict_features_norm[feature])

    #encoding labels
    label_transf = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), [target]),
    sparse_threshold=0  
    )

    label_transf.fit(Train_labels)

    Train_labels_encoded = label_transf.transform(Train_labels)
    Valid_labels_encoded = label_transf.transform(Valid_labels)
    Test_labels_encoded = label_transf.transform(Test_labels)

    return df

