# CyberDataAnalytics: lab assignment 2
Group 1: Daan van der Valk and Sandesh Manganahalli Jayaprakash

Lab assignment 2 for Cyber Data Analytics, the TU Delft course.

## Installation
All included scripts should be run with Python 3. We used Python 3.6.4 to be specific, but hopefully any Python 3 version would suffice.

The following packages should be installed, which can be done using pip (`pip install <package>`) or Conda (`conda install <package>`), whatever you prefer.

* `matplotlib`
* `scipy`
* `sklearn`
* `pydotplus`
* `graphviz` (depending on the environment, also `python-graphviz`)
* `imblearn`
* `joblib`
* `seaborn`
* `pandas`

Some of these packages maybe already installed.

## Highlights
### Familiarization
* Map: [all sensors in the dataset (generated by EPANET)](https://github.com/DaanvanderValk/CDA2/blob/master/Data/junctions_map.png)
* Graph: [cyclic behavior and correlation in some of the signals](https://github.com/DaanvanderValk/CDA2/blob/master/Exploration/cyclic_behavior.svg)
* Predicting the L_T1 series using moving average (MA) is done in [prediction.py](https://github.com/DaanvanderValk/CDA2/blob/master/Exploration/prediction.py)
  * Graphs for a window size of 3: [predictions](https://github.com/DaanvanderValk/CDA2/blob/master/Exploration/moving_average_windowsize3.svg) and [errors](https://github.com/DaanvanderValk/CDA2/blob/master/Exploration/squared_errors_windowsize3.svg)
  * Graphs for multiple window sizes: [predictions](https://github.com/DaanvanderValk/CDA2/blob/master/Exploration/moving_average.svg) and [errors](https://github.com/DaanvanderValk/CDA2/blob/master/Exploration/squared_errors.svg)


### ARMA
* Order estimation: [arma_parameters.py](https://github.com/DaanvanderValk/CDA2/blob/master/ARMA/arma_parameters.py) resulted in [arma_parameter_results.txt](https://github.com/DaanvanderValk/CDA2/blob/master/ARMA/arma_parameter_results.txt)
  * Graphs: [autocorrelation](https://github.com/DaanvanderValk/CDA2/blob/master/ARMA/autocorrelation.svg), [ARMA predictions](https://github.com/DaanvanderValk/CDA2/blob/master/ARMA/armapredictions.svg). 
* Testing for all signals: [arma_all_features.py](https://github.com/DaanvanderValk/CDA2/blob/master/ARMA/arma_all_features.py) resulted in [arma_detection_results_all.txt](https://github.com/DaanvanderValk/CDA2/blob/master/ARMA/arma_detection_results_all.txt)


### Discrete models (N-Grams)
* Discritization in percentiles: [Percentiles.py](https://github.com/DaanvanderValk/CDA2/blob/master/Discretisation/Percentiles.py) resulted in [discretization.svg](https://github.com/DaanvanderValk/CDA2/blob/master/Discretisation/discretization.svg)
* [trigrams.py](https://github.com/DaanvanderValk/CDA2/blob/master/N-Grams/trigrams.py)
* [fivegrams.py](https://github.com/DaanvanderValk/CDA2/blob/master/N-Grams/fivegrams.py)


### PCA
* First execute: [training_data03_residuals_plot.py](https://github.com/DaanvanderValk/CDA2/blob/master/PCA/training_data03_residuals_plot.py)
* Then run in the same environment: [pca_all.py](https://github.com/DaanvanderValk/CDA2/blob/master/PCA/pca_all.py)

