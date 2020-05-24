library(h2o)
library(tidyverse)
library(ggbeeswarm)

df <- tibble(y = rep(c(0,1), c(1000,1000)), 
             x1 = rnorm(2000),
             x2 = rf(2000, df1 = 5, df2 = 2),
             x3 = sample(rep(c('A', 'B', 'C'), c(500,500, 1000))))

# For classification, the y column must be a factor.
# Source: https://www.rdocumentation.org/packages/h2o/versions/3.26.0.2/topics/h2o.automl
df <- df %>% 
  mutate(y = as.factor(y)) %>% 
  mutate_if(is.character, factor)

h2o.init()

df_frame <- as.h2o(df)

# Note that when splitting frames, H2O does not give an exact split. 
# It’s designed to be efficient on big data using a probabilistic splitting method rather than an exact split.
# Source: http://h2o-release.s3.amazonaws.com/h2o/master/3552/docs-website/h2o-docs/datamunge/splitdatasets.html
df_frame_split <- h2o.splitFrame(df_frame, ratios = 0.8)


# Metric for binary classification (deviance is the default). Check documentation here http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
automl_model <- h2o.automl(#x = x, 
                           y = 'y',
                           balance_classes = TRUE,
                           training_frame = df_frame_split[[1]],
                           nfolds = 5, # Default is nfolds = 5!!!
                           #validation_frame = df_frame_split[[2]], # read help(h2o.automl) !!!Optional. This argument is ignored unless the user sets nfolds = 0!!!
                           leaderboard_frame = df_frame_split[[2]],
                           max_runtime_secs = 60 * 2, # Two minutes
                           #exclude_algos = "StackedEnsemble", # Global Importance of Stacked models are tricky
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
  mutate(shap_importance = mean(abs(shap_value))) %>% 
  ungroup()


# SHAP contribution plot
p1 <- ggplot(shap_df, aes(x = shap_value, y = reorder(feature, shap_importance))) +
  ggbeeswarm::geom_quasirandom(groupOnX = FALSE, varwidth = TRUE, size = 0.4, alpha = 0.25) +
  xlab("SHAP value") +
  ylab(NULL)


# SHAP importance plot
p2 <- shap_df %>% 
  select(feature, shap_importance) %>%
  distinct() %>% 
  ggplot(aes(x = reorder(feature, shap_importance), y = shap_importance)) +
  geom_col() +
  coord_flip() +
  xlab(NULL) +
  ylab("mean(|SHAP value|)")

# Combine plots
gridExtra::grid.arrange(p1, p2, nrow = 1)
