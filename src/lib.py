import os
from transformers import num_cat_transform
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

def merge_files(filepath):
    csv_files = os.listdir(filepath)

    df_concat = pd.concat(
        [pd.read_csv(f"{filepath}\\{f}") for f in csv_files],
        ignore_index=True
        )

    return df_concat

def plot_hist(dataframe):
    hist = dataframe.hist(bins=50, figsize=(20,15))
    plt.show()


def split_train_test(data, test_ratio):
    shuffled_idices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_idices[:test_set_size]
    train_indices = shuffled_idices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def strata_build(df, category_name, df_col_name, bin_list, label_list=[1, 2, 3, 4, 5]):
    df[category_name] = pd.cut(
        df[df_col_name],
        bins=bin_list,
        labels=[1, 2, 3, 4, 5]
        )
    df[category_name].hist()
    plt.show()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df[category_name]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    category_proportions = strat_test_set[category_name].value_counts() / len(strat_test_set)
    print(category_proportions)

    for set_ in (strat_train_set, strat_test_set):
        set_.drop(category_name, axis=1, inplace=True)

def data_explorer(train_set):
    df_s = "population"
    df_c = "median_house_value"
    df_train = train_set.copy()
    df_train.plot(kind="scatter", x="longitude", y="latitude")
    plt.show()
    df_train.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show()
    df_train.plot(
        kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=df_train[df_s]/100, label=df_s, figsize=(10,7), c=df_c,
        cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    plt.show()

def calc_correlation(df, target_var, correlation_attr_list):
    df_i = df.copy()
    df_no_str = df_i.drop("ocean_proximity", axis=1)
    corr_matrix = df_no_str.corr()
    print(corr_matrix[target_var].sort_values(ascending=False))

    scatter_matrix(df_no_str[correlation_attr_list], figsize=(12,8))
    plt.show()

def attr_join(df, attr_target_dict, target_var):
    df_i = df.copy()
    df_no_str = df_i.drop("ocean_proximity", axis=1)
    corr_matrix = df_no_str.corr()
    for k, v in attr_target_dict.items():
        df_no_str[k] = df_no_str[v[0]]/df_no_str[v[1]]
    
    corr_matrix = df_no_str.corr()
    print(corr_matrix[target_var].sort_values(ascending=False))

def fill_missing_values(df, attr, choice):
    match choice:
        case 1:
            df.dropna(subset=[attr])
        case 2:
            df.drop(attr, axis=1)
        case 3:
            median = df[attr].median()
            df[attr].fillna(median, inplace=True)
        case 4:
            imputer = SimpleImputer(strategy="median")
            df_num = df.drop("ocean_proximity", axis=1)
            imputer.fit(df_num)
            X = imputer.transform(df_num)
            return X

    return df
    
def cat_processing(df, cat_col_name):
    df_cat = df[[cat_col_name]]
    print(df_cat.head(10))

    ordinal_encoder = OrdinalEncoder()
    df_cat_encoded = ordinal_encoder.fit_transform(df_cat)
    print(df_cat_encoded[:10])

    cat_encoder = OneHotEncoder()
    df_cat_1hot = cat_encoder.fit_transform(df_cat)
    print(df_cat_1hot)

def lin_reg_train(df, train_set, predictor, cat_col_name=None, impute_strategy="median"):
    df_labels = train_set[predictor].copy()
    df = train_set.drop(predictor, axis=1)

    if cat_col_name:
        df_num = df.drop(cat_col_name, axis=1)
        df_prepared, full_pipeline = num_cat_transform(
            df,
            df_num,
            cat_col_name,
            impute_strategy
            )

        lin_reg = LinearRegression()
        lin_reg.fit(df_prepared, df_labels)

        some_data = df.iloc[:5]
        some_labels = df_labels.iloc[:5]
        some_data_prepared = full_pipeline.transform(some_data)
        print("Predictions: ", lin_reg.predict(some_data_prepared))
        print("Labels: ", list(some_labels))
    
