# A minimal example combining H2O’s AutoML and Shapley’s decomposition in R

This script depicts a minimal example using R — one of the most used languages for Data Science — for fitting machine learning models using H2O’s AutoML and Shapley’s value. Since H2O’s AutoML tool has a wide range of predictive models, the key point of this approach is to limit the model search to only tree-based by setting include_algos = c('DRF', 'GBM', 'XGBoost') . Furthermore, using the Shapley’s values, global importance plots, and partial dependence plots were presented in order to illustrate the interpretability of this approach.

More info available at https://medium.com/@renanxaviercortes/a-minimal-example-combining-h2os-automl-and-shapley-s-decomposition-in-r-ba4481282c3c
