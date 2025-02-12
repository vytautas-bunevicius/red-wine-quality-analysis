# Red Wine Quality Analysis

## Table of Contents

- [Overview](#overview)
- [Dashboard](#dashboard)
- [Installation](#installation)
  - [Using uv (Recommended)](#using-uv-recommended)
  - [Using pip (Alternative)](#using-pip-alternative)
- [Data Analysis](#data-analysis)
  - [Research Objectives](#research-objectives)
  - [Hypotheses](#hypotheses)
  - [Exploratory Data Analysis Questions](#exploratory-data-analysis-questions)
- [Findings and Insights](#findings-and-insights)
- [Future Improvements](#future-improvements)

## Overview

Red Wine Quality Analysis is a comprehensive project that explores the physicochemical properties of red wine and builds predictive models to assess quality. Leveraging Python with libraries like Pandas and Scikit-Learn, the analysis dives into key variables such as acidity, sugar, and alcohol content to uncover their relationships with wine quality.

## Dashboard

Explore the data interactively via our Looker Studio dashboard:

[Interactive Dashboard on Looker Studio](https://lookerstudio.google.com/u/0/reporting/c4d22105-252e-422d-bb88-c76c667a7f78/page/Aa5yD)

For additional context and resources, visit our [Project Page](https://bunevicius.com/project-pages/red-wine-analysis).

## Installation

Set up the project locally using one of the following methods:

### Using uv (Recommended)

1. **Install uv:**

   ```bash
   # On Unix/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   ```

2. **Clone the Repository:**

   ```bash
   git clone https://github.com/vytautas-bunevicius/red-wine-quality-analysis.git
   cd red-wine-quality-analysis
   ```

3. **Create and Activate a Virtual Environment:**

   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

4. **Install Dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

5. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

### Using pip (Alternative)

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/vytautas-bunevicius/red-wine-quality-analysis.git
   cd red-wine-quality-analysis
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

## Data Analysis

### Research Objectives

- **Key Physicochemical Properties:** Analyze how variables like acidity, sugar, and alcohol contribute to wine quality.
- **Predictive Modeling:** Develop and validate statistical models to forecast wine quality.
- **Statistical Validation:** Use hypothesis testing and correlation analysis to confirm relationships in the data.
- **Outlier Impact:** Evaluate the effect of outliers on the analytical insights and model performance.

### Hypotheses

- **Null Hypothesis (H0):** There is no significant correlation between alcohol content and the quality of red wine.
- **Alternative Hypothesis (H1):** Higher alcohol content in red wine is associated with better quality ratings.

### Exploratory Data Analysis Questions

1. What are the distributions of key chemical properties in the dataset?
2. How do different physicochemical parameters correlate with the sensory quality ratings?
3. What data patterns can be identified to reliably predict wine quality?

## Findings and Insights

- **Diverse Distributions:** Key properties display varied distributions (e.g., right-skewed, bimodal), indicating complex characteristics within the dataset.
- **Significant Predictors:** Both alcohol content and acidity are strong predictors of the wine quality outcomes.
- **Model Performance:** A linear regression model achieved an R-squared value of 35.4%, suggesting moderate predictability.
- **Outlier Considerations:** Initial analyses that included outliers provided valuable insights, though subsequent adjustments improved model robustness.

## Future Improvements

- **Advanced Modeling:** Experiment with non-linear models and machine learning techniques to enhance prediction accuracy.
- **Improved Data Preprocessing:** Apply more refined strategies for managing outliers and addressing multicollinearity.
- **Dataset Expansion:** Incorporate additional features and data points to further enrich the analysis and uncover deeper insights.