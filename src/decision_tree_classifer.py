# Decision Tree Classifer Example

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.utils import parallel_backend
from joblibspark import register_spark
from pprint import pprint

register_spark()

iris = datasets.load_iris()
X = iris.data
y = iris.target