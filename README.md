# changepointmodel 
--- 
[![CI](https://github.com/cunybpl/changepointmodel/actions/workflows/unittests.yaml/badge.svg)](https://github.com/cunybpl/changepointmodel/actions/workflows/unittests.yaml)

Changepoint modeling, load disaggregation and savings methodologies consistent with ashrae guidelines. 


## About 
---

This is a small toolkit for processing building energy data centered around changepoint modeling. Much of our work is based on industry standard methodologies taken from Ashrae. 

We have found over time that while doing these individual calculations is not difficult, doing them at scale in a well defined way can be. Therefore we tried to design a flexible library that can be used in a variety of contexts such as data science, machine learning and batch processing.


### Features 
----

* Loosely coupled, interface oriented and extensible (`changepoint.core`)
* Ready built high level application tooling (`changepoint.bema`)
* Wrapper classes for `scipy.optimize.curve_fit` that conform to the `scikit-learn` interface. 
* PEP484 complient (from 2.2.0). Overall strong typing approach. 
* Heavily tested and production ready 


The `core` package consists of the lower level machinery and calculation APIs. The `app` package contains higher level components organized into a ready made application for common building energy modeling tasks and batch processing. 

We have heavily documented each module's public interface. Below is a brief outline of what is inside 

__core__
----
The core APIs are a loosely coupled set of classes that work together to build changepoint models, load aggregations and savings calculations.

* `estimator` - Wrapper classes around `scipy.optimize.curve_fit` with some bells and whistles. 
    * These classes interop with scikit learn and can be incorporated into the sklearn API (`cross_val_score`, `GridSearchCV` etc.) for ML projects.
* `pmodels` - Interfaces for defining types around parameter model functions. 
* `loads` - Load calculations for various ParameterModels
* `predsum` - Predicted Sum (used for nac calculations)
* `savings` - High level interfaces for savings calculations.
* `schemas` - Input validation 
* `scoring` - Standard statistical reporting for model validation.

__core.calc__ 
----
Model dependent calculations and ashrae savings formulas. 

* `models` - the standard parameter modeling functions. We use these with `scipy.optimize.curve_fit`'s "trf" algorithm that allows for approximate derivitives and bounded search. 
* `bounds` - a standard set of bounds functions to use. We determine these by analyzing the X value input. 
* `metrics` - scoring metrics borrowed from `sklearn` 
* `loads` - loads calculations for base, heating and cooling 
* `savings` - ashrae savings formulas for monthly data 
* `uncertainties` - ashrae uncertainty calculations for savings 

__app__ 
--- 

This is application level code that is provided as a convenience for batch processing data. 

* `main` - Run an option-c or baseline set of models. 
* `config` - A standard configuration of the `core` components 
* `filter_` - A module we use for filtering data based on procedural constraints 
* `extras` - Extra model filtering
* `models` - Pydantic model validators and parsers 






