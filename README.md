# Red Wine Quality Analysis

## Interactive Dashboard

To explore the data interactively with detailed charts, visit the Looker Studio dashboard:

[Interactive Dashboard on Looker Studio](https://lookerstudio.google.com/u/0/reporting/c4d22105-252e-422d-bb88-c76c667a7f78/page/Aa5yD)

Or alternatively, on a [Personal Website](https://bunevicius.com/project-pages/red-wine-analysis)

## Setup Guide

To replicate the analysis locally, follow these setup steps:

- Clone the Repository:

      git clone https://github.com/vytautas-bunevicius/red-wine-quality-analysis.git

- Navigate to the repository directory:

      cd red-wine-quality

- Install necessary Python libraries using the command:

      pip install -r requirements.txt

- Launch Jupyter Notebook to interact with the analysis:

      jupyter notebook

## Project Overview

This project aims to analyze the quality of red wine using **Python** and various data analysis libraries such as **Pandas** and **Scikit-Learn**. The dataset features physicochemical properties of red wine and correlates them with sensory quality ratings.

## Research Objectives

1. **Key Physicochemical Properties:** Explore how various properties like acidity, sugar, and alcohol content influence wine quality.
2. **Predictive Modeling:** Develop models to predict wine quality based on its chemical properties.
3. **Statistical Analysis:** Perform statistical tests to confirm or reject hypotheses about factors affecting wine quality.
4. **Outlier Impact:** Assess how outliers in the data affect the overall analysis and modeling.

## Hypotheses

a. **Null Hypothesis (H0)**: There is no significant correlation between alcohol content and the quality of red wine.

b. **Alternative Hypothesis (H1)**: Higher alcohol content in red wine is associated with higher quality.

## Exploratory Data Analysis Questions

1. What are the distributions of key physicochemical properties in the dataset?
2. How do these properties correlate with the quality ratings of red wine?
3. Are there identifiable trends or patterns that can help predict wine quality?

## Findings and Insights

### 1. Distribution of Properties
- Key physicochemical properties of wines show varied distribution patterns, often right-skewed or bimodal, indicating diverse wine characteristics.

### 2. Influence on Wine Quality
- Alcohol content and acidity levels are significant predictors of wine quality, showing strong correlations in the data analysis.

### 3. Predictive Modeling of Wine Quality
- The linear regression model used in this analysis provided an R-squared value of 35.4%, indicating a moderate level of predictability.

### 4. Outliers and Their Impact
- Identified outliers using the IQR method were initially retained to potentially uncover valuable insights but were later reevaluated for their impact on the analysis.

## Future Improvements

1. **Model Enhancements:** Explore non-linear models or machine learning techniques to improve prediction accuracy.
2. **Data Preprocessing:** Further refine data cleaning and preprocessing to handle outliers and multicollinearity more effectively.
3. **Extended Data Collection:** Incorporate additional data points or features to enrich the dataset and potentially uncover new insights into wine quality.
