import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Combine attributes & prepare data for numerical and categorical transformation
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household,
                population_per_household, bedrooms_per_room
                ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# df_num to indicate dataframe with only numerical values (no categories)
def standardization(df_num, impute_strategy):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ])

    df_num_tr = num_pipeline.fit_transform(df_num)

    return num_pipeline

# Transform numerical and categorical values simultaneously
def num_cat_transform(df, df_num, cat_col_name, impute_strategy):
    num_attrs = list(df_num)
    num_pipeline = standardization(df_num, impute_strategy)

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attrs),
        ("cat", OneHotEncoder(), [cat_col_name]),
    ])

    df_prepared = full_pipeline.fit_transform(df)
    return df_prepared, full_pipeline
