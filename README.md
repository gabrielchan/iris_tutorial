# Random Forest Classifer on the Famous Iris Dataset
**PLEASE NOTE** that this README is WIP

## The Problem & The Dataset
We attempt to predict the type of a iris given it's length of width of the petal and sepal (*sepal is the bigger petal at the bottom of a flower*). These values that we predict are known as **response**.

A given iris can be:
- Setosa (id=0)
- Versicolor (id=1)
- Virginica (id=2)

A model is provided a dataset that contains a table of the following columns (denoted **feature** in machine learning) `['sepal_length', 'sepal_width', 'petal_length', 'petal_width']`. Therefore a dataset contains **features** and it's **observations**.

As stated above, the goal of this exercise is to guess what type of iris is based on it's features. *We will attempt to guess the type of flower using supervised learning*. A machine learning model can do this through a form of supervised learning called **classification**. As the name suggests, it is the process of predicting the correct label of a given observation.

## Model Choice
The choice of your model ultimately depends on what type of data you have and what kind of predictions you're attempting to make. As the iris dataset is really a training wheels dataset for *supervised learning*, that is what we will do.

### Decision Tree Classifier vs Random Forest Classifier
Two important models in supervised learning for classification is the decision tree classifier and random forest classifier. [This article](https://towardsdatascience.com/understanding-random-forest-58381e0602d2) does a really good job of explaining and comparing the two.

Put very simply, a [**Decision Tree Classifier**](https://www.youtube.com/watch?v=LDRbO9a6XPU) is a supervised machine learning model where it recursively builds a tree by identifying the “best boolean question to ask” at a node on the tree to split the data set in two until all the observations at each leaf is identical (that is, unmixed). A leaf is then created saying what the % chance a given set of data equates to a label. It uses **Gini Impuirity** (if we were to randomly attempt to match between data and labels, what is our probability of guessing wrong) and Information Gain (`IG = gini_impurity(parent) - cumulative_avg(gini_impurity(true_child), gini_impurity(false_child)`. How much does a boolean question reduce the impurity)

A **Random Forest Classifer** is an **ensemble**. It a type of "model" that operates be making a bunch of *uncorrelated* (or as little as possible) non-ensemble models (e.g., decision tree classifier) work together. The goal of this is to achieve greater predictions by having multiple models work to provide a result, and the most common result is the final result returned. The theory of this is similar to building an investment portfolio, where it's safety and performance consistency comes from diversifying (that is, uncorrelated) your assets such that individual downfalls don't impact the overall result.

> NOTE
>
> In the tutorial that was followed, it used a Random Forest Classifer, but it's clear that the model of choice doesn't impact other key factors like hyperparameter tuning and improving performance by distrbuting the tuning onto a Spark cluster

## Hyperparameters
"**Parameters**" is a very common term in computer science. It refers to an input that a function requires. For example, `def some_function(int: important_number, string: cool_string)`:
- The parameters in this context is `important_number` and `important_string`
- When someone writes `some_function(100, "hello world")`, the integer 100 and string "hello world" are **arguments**

However, in machine learning, and in particular a "machine learning model's parameters" may refer to it's *hyperparameter*. Unfortunately in machine learning, these terms are used quite interchangably (for example, `model.get_params()` would imply getting a model's parameters, but it's clear that it is it's hyperparameters). As such, context usually defines what someone is refering to. Therefore, it is important to understand the differences between the two.

**Model Hyperparameter** (often refered too simply as, hyperparameter): Parameters in a machine learning model that control and define how a model learns (e.g., `n_iter`, the number of iterations to train on, is a hyperparameter as it configures how many iterations a model needs to go through during learning)
**Model Parammeters**: Variables that are learned and predicted by the model (e.g., the response matrix as a result of the predict on the iris dataset is a model parameter, as the model guessed it)

In an simplified summary, hyperparameter's is the configuration and the model parameters is the results. 

You can see the following is the output of `model.get_params()` from `hyperparam_tune_example.py`. 
```
{
  "cv": 5,                                                                       
  "error_score": nan,
  "estimator": RandomForestClassifier(),
  "estimator__bootstrap": True,
  "estimator__ccp_alpha": 0.0,
  "estimator__class_weight": None,
  "estimator__criterion": "gini",
  "estimator__max_depth": None,
  "estimator__max_features": "auto",
  "estimator__max_leaf_nodes": None,
  "estimator__max_samples": None,
  "estimator__min_impurity_decrease": 0.0,
  "estimator__min_impurity_split": None,
  "estimator__min_samples_leaf": 1,
  "estimator__min_samples_split": 2,
  "estimator__min_weight_fraction_leaf": 0.0,
  "estimator__n_estimators": 100,
  "estimator__n_jobs": None,
  "estimator__oob_score": False,
  "estimator__random_state": None,
  "estimator__verbose": 0,
  "estimator__warm_start": False,
  "n_iter": 100,
  "n_jobs": None,
  "param_distributions": {
    "max_features": <scipy.stats._distn_infrastructure.rv_frozen object at 0x7ffcbe3e3340>,
    "min_samples_split": <scipy.stats._distn_infrastructure.rv_frozen object at 0x7ffcbe1c1940>,
    "n_estimators": <scipy.stats._distn_infrastructure.rv_frozen object at 0x7ffcb9132d60>},
  "pre_dispatch": "2*n_jobs",
  "random_state": 1,
  "refit": True,
  "return_train_score": False,
  "scoring": None,
  "verbose": 0
}
```
As you can see, many of these parameters have nothing to do with flowers. Therefore it is clear that `model.get_params()` returns an object containing the model's hyperparameters.

### Tuning/Optimization

# High-Level Usage of Scikit-Learn
Typically in a `scikit-learn` model:
- `X = feature matrix`, that is the *learning dataset*
- `y = target matrix`, that is the *associated labels*

> In particular to the Iris dataset, the feature matrix is data, the target matrix is target
>
> `iris.target` -> `List[Int]` where:
> - `iris.target.shape == (length_x, 1)` (a 1 dimensional array with length_x elements)
> - returns a list of integers that represent an iris type label (e.g., Setosa = 0)
>
> `iris.data` -> `List[List[String]]` where:
> - `iris.data.shape == (length_x, num_of_features)` (a length_x dimension array with num_of_features elements each)

Then you can use a supervised learning model (from the `scikit-learn` library) to injest this data set then make future predictions

# Appendix
## Terminology
**Feature**: A column in a dataset
- Also known as predictor, attribute, independent variable, input, regressor and/or covariate

**Observation**: A row in a dataset
- Also known as sample, example, instance and/or record

**Response**: Values that are being predicted
- Also known as target, outcome, label, dependent variable

**Supervised Learning**: Machine learning by making predictions from labelled data

**Unsupervised Learning**: Machine learning by extracting structure from unstructured data

**Classification**: Supervised learning where the response are *categorical data*. The categories are usually the labels

**Label**: A meaning to an observation

**Regression**: Supervised learning where the response are *continous data*

**Decision Tree Classifier**

**Gini Impurity**

**Random Forest Classifier**

**Ensemble**

## High-Level Process of Supervised Learning
1. Train a machine learning model using *labeled data*
  - Labeled data is just the dataset where each observation has a label assigned
  - The model will learn the relationship between the label and features
2. Make predictions on *unlabeled data*
