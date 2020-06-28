library(h2o)
library(tidyverse)
library(ggbeeswarm)

df <- tibble(y = rep(c(0,1), c(1000,1000)), 
             x1 = rnorm(2000),
             x2 = rf(2000, df1 = 5, df2 = 2),
             x3 = runif(2000),
             x4 = c(sample(rep(c('A', 'B', 'C'), c(300, 300, 400))), 
                    sample(c('A', 'B', 'C'), 1000, prob = c(0.25, 0.25, 0.5), replace = T)),
             x5 = c(rnorm(1000), rnorm(1000, 0.25)))

# For classification, the y column must be a factor.
# Source: https://www.rdocumentation.org/packages/h2o/versions/3.26.0.2/topics/h2o.automl
df <- df %>% 
  mutate(y = as.factor(y)) %>% 
  mutate_if(is.character, factor)

h2o.init()

df_frame <- as.h2o(df)

# Note that when splitting frames, H2O does not give an exact split. 
# It is designed to be efficient on big data using a probabilistic splitting method rather than an exact split.
# Source: http://h2o-release.s3.amazonaws.com/h2o/master/3552/docs-website/h2o-docs/datamunge/splitdatasets.html
df_frame_split <- h2o.splitFrame(df_frame, ratios = 0.8)

# Metric for binary classification (deviance is the default). Check documentation here http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
automl_model <- h2o.automl(#x = x, 
                           y = 'y',
                           balance_classes = TRUE,
                           training_frame = df_frame_split[[1]],
                           nfolds = 4,
                           #validation_frame = df_frame_split[[2]], # read help(h2o.automl) !!!Optional. This argument is ignored unless the user sets nfolds = 0!!! By default and when nfolds > 1, cross-validation metrics will be used for early stopping and thus validation_frame will be ignored.
                           leaderboard_frame = df_frame_split[[2]],
                           max_runtime_secs = 60 * 2, # Two minutes
                           #exclude_algos = "StackedEnsemble", # Global Importance of Stacked models is tricky
                           include_algos = c('DRF', 'GBM', 'XGBoost'), # Complete List help(h2o.automl)
                           sort_metric = "AUC")

lb <- as.data.frame(automl_model@leaderboard)
aml_leader <- automl_model@leader


# SHAP values: http://docs.h2o.ai/h2o/latest-stable/h2o-r/docs/reference/predict_contributions.H2OModel.html

SHAP_values <- predict_contributions.H2OModel(aml_leader, df_frame_split[[2]])

# Wrangling inspired here: https://bradleyboehmke.github.io/HOML/iml.html

shap_df <- SHAP_values %>%
  as.data.frame() %>%
  select(-BiasTerm) %>%
  gather(feature, shap_value) %>%
  group_by(feature) %>%
  mutate(shap_importance = mean(abs(shap_value)),
         shap_force = mean(shap_value)) %>% 
  ungroup()

# SHAP contribution plot
p1 <- ggplot(shap_df, aes(x = shap_value, y = reorder(feature, shap_importance))) +
  ggbeeswarm::geom_quasirandom(groupOnX = FALSE, varwidth = TRUE, size = 0.9, alpha = 0.5, width = 0.15) +
  xlab("SHAP value") +
  ylab(NULL) +
  theme_minimal(base_size = 15)

# SHAP importance plot
p2 <- shap_df %>% 
  select(feature, shap_importance) %>%
  distinct() %>% 
  ggplot(aes(x = reorder(feature, shap_importance), 
             y = shap_importance)) +
  geom_col(fill = 'black') +
  coord_flip() +
  xlab(NULL) +
  ylab("mean(|SHAP value|)") +
  theme_minimal(base_size = 15)

# Combine plots
gridExtra::grid.arrange(p1, p2, nrow = 1)

# Shapley-based dependence plots for a numerical feature
SHAP_values %>%
  as.data.frame() %>%
  select(-BiasTerm) %>% 
  mutate(x5_feature_values = as.vector(df_frame_split[[2]]$x5)) %>% 
  ggplot(aes(x = x5_feature_values, y = x5)) +
  geom_point(aes(color = x5), width = 0.1) +
  scale_colour_gradient(low = "red", high = "blue", name = 'SHAP values') +
  ylab('Shapley\'s values for x5 feature') +
  xlab('x5 values') +
  theme_minimal(base_size = 15) +
  geom_smooth()

# Shapley-based dependence plots for a categorical feature
SHAP_values %>%
  as.data.frame() %>%
  select(-BiasTerm) %>% 
  mutate(x4_feature_values = as.vector(df_frame_split[[2]]$x4)) %>% 
  ggplot(aes(x = x4_feature_values, y = x4)) +
  geom_jitter(aes(color = x4), width = 0.1) +
  scale_colour_gradient(low = "red", high = "blue", name = 'SHAP values') +
  ylab('Shapley\'s values for x4 feature') +
  xlab('x4 feature classes') +
  theme_minimal(base_size = 15)