# module required
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error



dataset = pd.read_csv("https://raw.githubusercontent.com/changdaeoh/HandsOn_ML/main/datasets/housing.csv")
d_copy = dataset


# binning
d_copy["income_cat"] = pd.cut(d_copy['median_income'], 
                              bins = [0., 1.5, 3.0, 4.5, 6., np.inf], 
                              labels = [1,2,3,4,5]) 

# data split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(d_copy, d_copy["income_cat"]):
    train = d_copy.loc[train_index]
    test = d_copy.loc[test_index]

train = train.drop(['income_cat'], axis = 1)
test = test.drop(['income_cat'], axis = 1)

# seperate y from entire
X_train = train.drop("median_house_value", axis = 1)
y_train = train["median_house_value"].copy()
X_test = test.drop("median_house_value", axis = 1)
y_test = test["median_house_value"].copy()


# ------------------------------------------------------------------------------------------
# Preprocess Pipeline 
# ------------------------------------------------------------------------------------------

# global variables
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

# creating Combined Attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]


a = X_train.drop(['ocean_proximity'],axis = 1)
num_list = list(a.columns)
cat_list = ['ocean_proximity']

# preprocess pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    ('attr_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

preprocess_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_list),
    ("cat", OneHotEncoder(), cat_list)
])

# save feature names
col_names = list(X_train.columns[:-1]) +\
["rooms_per_household", "population_per_household", "bedrooms_per_room"] +\
['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

#-----------------------------------------------------------------------------------------
# Attach model
#-----------------------------------------------------------------------------------------

# FeatureSelector 
def indices_of_top_k(importances, k):
    return np.sort(np.argpartition(np.array(importances), -k)[-k:])

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y = None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

# Random Search CV
params = {'kernel':['linear','rbf'],
          'C':np.logspace(-5,5,20),
          'gamma':np.logspace(-3,3,15)}
svr = SVR()
rs = RandomizedSearchCV(svr, params, scoring = "neg_mean_squared_error",
                        cv = 5, n_iter = 20, n_jobs = -1)
rs.fit(preprocess_pipeline.fit_transform(X_train), y_train)

# fitting Random Forest to get feature importances
rf = RandomForestRegressor(random_state = 42)
rf.fit(preprocess_pipeline.fit_transform(X_train), y_train)

k = 10

# full pipeline
full_pipe = Pipeline([
    ('preprocessing', preprocess_pipeline),
    ('feature_selection', FeatureSelector(rf.feature_importances_, k)),
    ('svr', SVR(**rs.best_params_))
])

#-----------------------------------------------------------------------------------------
# optimal option searching
#-----------------------------------------------------------------------------------------
 
param_grid = [{
    'preprocessing__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(rf.feature_importances_) + 1))
}]

gs_prep = GridSearchCV(full_pipe, param_grid, cv=5,
                       scoring='neg_mean_squared_error', verbose=2)
gs_prep.fit(X_train, y_train)
print(gs_prep.best_params_)