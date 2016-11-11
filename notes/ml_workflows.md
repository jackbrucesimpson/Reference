# [Visual Diagnostics for More Informed Machine Learning](https://www.youtube.com/watch?v=Nt5HaoVKcvY)

![workflow](../resources/ml_steps.png)

## Model Selection Triple
When selecting a model, rather than going with your default favourite method, take 3 things into account:
- Feature analysis: intelligent feature selection and engineering
- Model selection: model that makes most sense for problem/domain space
- Hyperparameter Tuning: once model and features have been selected, select the parameters that result in optimal performance.

## Visual Feature Analysis
- Boxplots are a useful starting tool for looking at all features as they show you:
	- Central tendency
	- Distribution
	- Outliers
- Histograms let you examine the distribution of a feature
- Sploms: Pairwise plots of features to identify:
	- pairwise linear, quadratic and exponential relationships between variables
	- Homo/heteroscedasticity
	- How features are distributed relative to each other
- Raduiz: Plot features around a circle and show how much pull they have
- Parallel coordinates: lets you visualise multiple variables as line segments - you want to find separating chords which can help with classification

## Evaluation Tools
- Classification heat maps: show you areas where model is performing best
- ROC-AUC and Prediction Error Plots: Show you which models are performing better
- Residual plots: Show you which models are doing best and why
- Gridsearch and validation curves: shows you the performance of a model along the parameters. You can create a visual heatmap for grid search