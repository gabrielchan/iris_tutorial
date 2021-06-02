# This file is following the example from https://jamesrledoux.com/code/randomized_parameter_search
# Parallelizing 

# Import RandomSearchCV hyperparamter tuning from sklearn, Random Forest Classifer model, our Iris dataset, and distributions
from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, truncnorm, randint

# Pretty Printing of Model
from pprint import pprint

# Imports for Joblibspark for distrubte joblib onto Spark
from sklearn.utils import parallel_backend
from joblibspark import register_spark

# Register Spark as Backend
register_spark()

# Note that Iris data is our sample dataset for the RF classifer tutorial
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Initial estimates on hyperparamters. Setup as a dictionary to be passed to RandomSearchCV for RF classifer tutorial
model_params = {
  # Randomly sample numbers from 4 to 200
  'n_estimators': randint(4, 200),
  # normally distrubte max_features with a mean of 0.25, stddev of 0.1, and bound between 0 and 1
  'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
  # uniform distribution from 0.01 to 0.2 (0.01+0.199)
  'min_samples_split': uniform(0.01, 0.199)
}

# Create RandomForestClassifer Model
rf_model = RandomForestClassifier()

# random search metaestimator - train n_iter=100 models over cv=5 folds per iteration on rf_model using param_distributions
clf = RandomizedSearchCV(
  estimator=rf_model,
  param_distributions=model_params,
  n_iter=100,
  cv = 5,
  random_state = 1
)

# train the metaestimator/tune it find the best model of the 100 iterations
# Note that this is what we want to parallelize
# model = clf.fit(X, y)

# parallelize the tuning process by distributing it on Spark. When this runs you can check localhost:4040 for the UI
with parallel_backend('spark', n_jobs=3):
  model = clf.fit(X, y)

pprint(model.get_params())
pprint(model.best_estimator_)

predictions = model.predict(X)
print(predictions)
