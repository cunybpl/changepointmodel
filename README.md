# changepointmodel 
--- 
[![pytest](https://github.com/cunybpl/changepointmodel/actions/workflows/unittests.yaml/badge.svg)](https://github.com/cunybpl/changepointmodel/actions/workflows/unittests.yaml) <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="131.5" height="20"><linearGradient id="smooth" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="round"><rect width="131.5" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#round)"><rect width="65.5" height="20" fill="#555"/><rect x="65.5" width="66.0" height="20" fill="#007ec6"/><rect width="131.5" height="20" fill="url(#smooth)"/></g><g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110"><image x="5" y="3" width="14" height="14" xlink:href="https://dev.w3.org/SVG/tools/svgweb/samples/svg-files/python.svg"/><text x="422.5" y="150" fill="#010101" fill-opacity=".3" transform="scale(0.1)" textLength="385.0" lengthAdjust="spacing">python</text><text x="422.5" y="140" transform="scale(0.1)" textLength="385.0" lengthAdjust="spacing">python</text><text x="975.0" y="150" fill="#010101" fill-opacity=".3" transform="scale(0.1)" textLength="560.0" lengthAdjust="spacing">3.10, 3.11</text><text x="975.0" y="140" transform="scale(0.1)" textLength="560.0" lengthAdjust="spacing">3.10, 3.11</text><a xlink:href=""><rect width="65.5" height="20" fill="rgba(0,0,0,0)"/></a><a xlink:href="https://www.python.org/"><rect x="65.5" width="66.0" height="20" fill="rgba(0,0,0,0)"/></a></g></svg>

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
* PEP484 complient (from 3.0.0). Overall strong typing approach. 
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






