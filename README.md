# Red Wine Quality Analysis

## Table of Contents

- [Overview](#overview)
- [Interface](#interface)
- [Features](#features)
- [Model Details](#model-details)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
    - [Using uv (Recommended)](#using-uv-recommended)
    - [Using pip (Alternative)](#using-pip-alternative)
- [Development Setup](#development-setup)
    - [Environment Configuration](#environment-configuration)
- [Local Execution](#local-execution)
- [Data Analysis](#data-analysis)
    - [Research Objectives](#research-objectives)
    - [Hypotheses](#hypotheses)
    - [Exploratory Data Analysis Questions](#exploratory-data-analysis-questions)
- [Findings and Insights](#findings-and-insights)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

Data analysis project exploring the physicochemical properties of red wine and
their correlation with quality ratings. Uses Python, Pandas, and Scikit-Learn to
investigate variables such as acidity, sugar, and alcohol content, and builds
predictive models to assess wine quality.

## Interface

Explore the data interactively via our Looker Studio dashboard:

[Interactive Dashboard on Looker Studio](https://lookerstudio.google.com/u/0/reporting/c4d22105-252e-422d-bb88-c76c667a7f78/page/Aa5yD)

For additional context and resources, visit
our [Project Page](https://bunevicius.com/project-pages/red-wine-analysis).

## Features

- Exploratory data analysis with statistical validation
- Physicochemical property correlation analysis
- Feature engineering
- Multiple regression model evaluation
- Model optimization
- Interactive Jupyter notebook analysis
- Data visualization
- Statistical hypothesis testing

## Model Details

Regression models evaluated:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression

## Prerequisites

- Python 3.12+ (check `.python-version` file for the current required version)
- Jupyter Notebook
- scikit-learn
- Pandas
- NumPy
- Matplotlib/Seaborn

## Installation

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

3. **Install Dependencies and Set Up Virtual Environment:**

   ```bash
   uv sync
   ```

   This command creates a virtual environment and installs all dependencies from
   `pyproject.toml`.

4. **Activate the Virtual Environment:**

   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

### Using pip (Alternative)

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/vytautas-bunevicius/red-wine-quality-analysis.git
   cd red-wine-quality-analysis
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -e .
   ```

## Development Setup

### Environment Configuration

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open the analysis notebooks:
    - `notebooks/wine_analysis.ipynb` - Main analysis notebook
    - `notebooks/model_evaluation.ipynb` - Model comparison and evaluation

## Local Execution

Run the Jupyter notebook for interactive data analysis:

```bash
jupyter notebook
```

Access the notebooks in your browser at `http://localhost:8888`.

## Data Analysis

### Research Objectives

- **Key Physicochemical Properties:** Analyze how variables like acidity, sugar,
  and alcohol contribute to wine quality.
- **Predictive Modeling:** Develop and validate statistical models to forecast
  wine quality.
- **Statistical Validation:** Use hypothesis testing and correlation analysis to
  confirm relationships in the data.
- **Outlier Impact:** Evaluate the effect of outliers on the analytical insights
  and model performance.

### Hypotheses

- **Null Hypothesis (H0):** There is no significant correlation between alcohol
  content and the quality of red wine.
- **Alternative Hypothesis (H1):** Higher alcohol content in red wine is
  associated with better quality ratings.

### Exploratory Data Analysis Questions

1. What are the distributions of key chemical properties in the dataset?
2. How do different physicochemical parameters correlate with the sensory
   quality ratings?
3. What data patterns can be identified to reliably predict wine quality?

## Findings and Insights

### Dataset Analysis

We analyzed 1,599 red wine samples after removing 240 duplicates, leaving 1,359
unique wines for analysis. The dataset spans quality scores from 3 to 8 out of
10, with the majority of wines clustered in the 5-6 quality range. This
represents a typical distribution in commercial wine production. The average
wine in our sample contains 10.4% alcohol by volume, ranging from as low as 8.4%
to as high as 14.9%. Most wines show low volatile acidity levels around 0.5,
which is ideal for avoiding spoilage aromas. The typical pH level sits around
3.3, indicating an appropriately acidic environment that preserves wine quality
and prevents bacterial growth.

### What Drives Quality - The Four Main Factors

**Alcohol Content: The Strongest Quality Driver**

Alcohol content emerges as the single most influential factor affecting wine
quality. The statistical correlation is 0.48, indicating a moderate-to-strong
positive relationship. In practical terms, wines with 11-12% alcohol content
consistently score 1-2 points higher on the 10-point scale compared to wines
with 9-10% alcohol. A concrete example illustrates this: a wine with 10% alcohol
and average properties across other factors receives approximately a 5/10
rating. The identical wine, when fermented to 12% alcohol while maintaining all
other properties, would likely achieve a 6/10 rating. This improvement stems
from the natural ripeness of grapes and the fermentation process. Winemakers
looking to improve their product should focus on harvesting grapes at full
ripeness to maximize natural sugars that ferment into alcohol. This single
factor can be the difference between a commercial-grade wine and a premium
product.

**Volatile Acidity: The Quality Killer**

Volatile acidity represents the negative side of the quality equation, with a
correlation of -0.40. This chemical compound creates the characteristic "
vinegar" smell that emerges from spoilage and oxidation during production or
storage. When volatile acidity is high, consumers immediately detect the
off-flavors that ruin the drinking experience. A wine with volatile acidity of
0.5 loses approximately 1.4 quality points compared to an identical wine with
volatile acidity of 0.3. This is a substantial penalty that directly translates
to lower market value and customer satisfaction. Production facilities can
control this factor through careful sanitation practices, proper storage
conditions, and minimizing oxygen exposure during fermentation and aging.
Implementing rigorous hygiene protocols and investing in anaerobic storage
systems pays dividends in final product quality.

**Sulphates: The Preservation Advantage**

Sulphates function as preservatives in wine and show a positive correlation of
0.28 with quality ratings. These compounds prevent unwanted microbial growth and
oxidation, effectively extending shelf life while maintaining flavor integrity.
When sulphate levels increase from 0.5 to 0.8 units, the wine gains
approximately 0.2 quality points in ratings. While this seems modest compared to
alcohol's impact, it represents a measurable improvement that accumulates in the
consumer's perception of product reliability. Standard industry practice calls
for adding appropriate sulphite levels during the bottling stage. This is an
inexpensive intervention with proven benefits that every winemaker should
implement as part of their quality control process.

**Chlorides: The Hidden Contaminant**

Chlorides, essentially salt content in wine, show a weak-to-moderate negative
correlation of -0.14 with quality. Even small increases in chloride levels
noticeably harm consumer perception. Wines with high chloride content can lose
0.5-1.0 quality points regardless of other positive factors. This contamination
typically results from unclean production equipment, mineral-heavy water
sources, or poor sanitation practices. Unlike other factors that require
specific expertise, preventing chloride contamination is straightforward: ensure
production equipment is properly cleaned, use filtered or distilled water in the
process, and maintain strict sanitation standards. This is a low-cost prevention
measure that eliminates a completely avoidable quality deficiency.

### Secondary Factors Worth Understanding

Fixed acidity and citric acid demonstrate a strong relationship with a
correlation of 0.66, meaning they move together in wine composition. More citric
acid leads to more fixed acidity, and this combination affects how tart and
fresh a wine tastes to consumers. The relationship between pH and citric acid is
even stronger at -0.55, representing a fundamental chemical principle: wines
with higher citric acid content naturally have lower pH values. This is
important because pH directly impacts microbial stability and perceived acidity
in the final product. Interestingly, residual sugar—the sweetness remaining
after fermentation—has nearly zero correlation with quality at only 0.02. This
finding suggests that consumers prioritize the chemical balance and complexity
of wine over simple sweetness, contradicting the common assumption that sweeter
wines automatically rate higher.

### Understanding Model Performance and Limitations

Our quality prediction model explains 35% of the variation in wine ratings. This
means that while the four main factors we identified are genuinely important,
they account for only one-third of what determines a wine's final quality score.
The remaining 65% comes from factors outside this analysis: grape variety,
vineyard location and soil conditions, aging process duration, winemaking
techniques specific to each producer, and the subjective preferences of tasters.
This 35% figure is realistic and useful for production improvement, but it
clarifies that chemistry alone cannot explain all quality differences. The model
works best for making relative comparisons—if you change certain parameters, you
can predict how quality will shift relative to where it was before.

In contrast, our alcohol prediction model achieves 72% accuracy. Using density,
pH, residual sugar, and fixed acidity measurements, we can reliably estimate
alcohol content with high precision. This model is particularly useful for
quality control and fermentation monitoring, where accurate alcohol predictions
help winemakers determine when fermentation is complete or whether their process
is progressing normally.

### Actionable Recommendations for Wine Production

Based on these findings, winemakers should prioritize five key actions. First,
maximize grape ripeness at harvest to increase natural sugars available for
fermentation into alcohol. This single change drives the largest quality
improvement. Second, prevent oxidation and bacterial spoilage through proper
production techniques and storage conditions—this directly controls volatile
acidity levels. Third, implement appropriate sulphite additions during bottling
as standard practice. Fourth, maintain a clean production environment and use
quality water sources to avoid salt and chloride contamination. Finally, monitor
pH and acidity balance throughout production since these factors affect both
consumer perception and long-term shelf stability. These five actionable steps
directly target the factors proven to drive wine quality in this analysis.

## Future Improvements

- **Advanced Modeling:** Experiment with non-linear models and machine learning
  techniques to enhance prediction accuracy.

## License

This project is released under the [Unlicense](https://unlicense.org/). This
means you can copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.

See the [UNLICENSE](UNLICENSE) file for more details.