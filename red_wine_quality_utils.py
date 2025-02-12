"""
red_wine_quality_utils.py

This module provides utility functions for analyzing the red wine quality dataset.
It offers data manipulation, visualization, statistical testing, and model evaluation functionalities.
All functions in this module conform to Google's Python Style Guidelines.

Functions:
  - get_columns: Retrieve the list of column names from a DataFrame.
  - remove_duplicates: Remove duplicate rows from a DataFrame.
  - plot_box_chart: Create box plots for each column in a DataFrame.
  - identify_outliers: Identify outliers in numeric columns using the IQR method.
  - plot_histograms: Plot histograms for selected features.
  - plot_heatmap: Display a heatmap based on a correlation matrix.
  - test_correlation: Calculate and test the Pearson correlation between two variables.
  - plot_coefficients: Visualize model coefficients with their confidence intervals.
  - plot_correlation: Create a scatter plot with a regression line and statistical annotations.
  - log_transform_features: Apply logarithmic transformation to reduce skewness.
  - train_linear_model: Train a linear regression model using OLS.
  - plot_model_predictions: Plot residuals from the model predictions with a trendline.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import OLS

# Primary brand colors from brandbook
PRIMARY_BLUE = "#3A5CED"
SECONDARY_BLUE = "#7E7AE6"
LIGHT_BLUE = "#7BC0FF"
SOFT_BLUE = "#B8CCF4"
DARK_BLUE = "#18407F"
SKY_BLUE = "#85A2FF"
LAVENDER = "#C2A9FF"
DEEP_PURPLE = "#3D3270"
AQUA = "#82E5E8"
OCEAN_BLUE = "#1C8BA5"
SOFT_AQUA = "#C0E0E2"

# Technical colors from brandbook
WHITE = "#FFFFFF"
BLACK = "#000000"
GRAY_LIGHT = "#E5E8EF"
TEXT_DARK = "#1A1E21"
TEXT_MID = "#48494C"
ALERT_RED = "#D30B3B"
HIGHLIGHT_BLUE = "#5684F7"
BACKGROUND_TRANSPARENT = "rgba(255, 255, 255, 0.9)"

# Typography specifications from brandbook
FONT_FAMILY = "Gordita, Figtree, sans-serif"
FONT_SIZE_TITLE = 24
FONT_SIZE_SUBTITLE = 20
FONT_SIZE_AXIS = 16
FONT_SIZE_TICK = 14
FONT_SIZE_LEGEND = 14
CORNER_RADIUS = 4

# Default plot dimensions
PLOT_HEIGHT = 600
PLOT_WIDTH_PER_SUBPLOT = 400
PLOT_MARGINS = {"l": 60, "r": 150, "t": 100, "b": 80, "pad": 10}

# Base plot layout configuration
BASE_LAYOUT = {
    "paper_bgcolor": WHITE,
    "plot_bgcolor": WHITE,
    "font": {
        "family": FONT_FAMILY,
        "color": TEXT_DARK,
        "size": FONT_SIZE_AXIS,
    },
    "xaxis": {
        "gridcolor": GRAY_LIGHT,
        "linecolor": GRAY_LIGHT,
        "zerolinecolor": GRAY_LIGHT,
        "showline": True,
        "linewidth": 1,
    },
    "yaxis": {
        "gridcolor": GRAY_LIGHT,
        "linecolor": GRAY_LIGHT,
        "zerolinecolor": GRAY_LIGHT,
        "showline": True,
        "linewidth": 1,
    },
}



def get_columns(df: pd.DataFrame) -> List[str]:
    """Retrieve column names from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame from which to extract column names.

    Returns:
        List[str]: A list containing the column names of the DataFrame.
    """
    return df.columns.tolist()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame.

    This function removes duplicate rows from the provided DataFrame and prints
    the number of duplicates found.

    Args:
        df (pd.DataFrame): The DataFrame from which duplicates will be removed.

    Returns:
        pd.DataFrame: A DataFrame with duplicate rows removed.
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    print(f"Removed {initial_count - final_count} duplicate rows")
    return df


def plot_box_chart(
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    chart_title: str,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Create a box plot for each column in the DataFrame.

    This function generates a box plot for every column in the DataFrame and displays 
    the plot using Plotly. Optionally, the plot can be saved as an image file.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        chart_title (str): The title of the chart.
        save_path (Optional[str], optional): File path to save the plot image. Defaults to None.

    Returns:
        go.Figure: The Plotly figure object containing the box plots.
    """
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                boxpoints="outliers",
                fillcolor=PRIMARY_BLUE,  # main box color set to blue
                marker=dict(color=LAVENDER),  # outlier markers set to lavender
                line=dict(color=PRIMARY_BLUE),  # border color set to blue
            )
        )
    # Remove xaxis and yaxis settings from BASE_LAYOUT to avoid duplicate keyword issues
    base_layout = dict(BASE_LAYOUT)
    base_layout.pop("xaxis", None)
    base_layout.pop("yaxis", None)
    base_layout.update({
        "title": chart_title,
        "xaxis_title": x_label,
        "yaxis_title": y_label,
        "title_x": 0.5,
        "title_font": dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE, color=TEXT_DARK),
        "showlegend": False,
        "height": 500,
        "width": 1600,
        "margin": dict(l=50, r=50, t=80, b=50),
    })
    fig.layout.update(base_layout)
    fig.update_xaxes(title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_SUBTITLE, color=TEXT_DARK))
    fig.update_yaxes(title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_SUBTITLE, color=TEXT_DARK))

    if save_path:
        fig.write_image(save_path)
    return fig


def identify_outliers(df: pd.DataFrame) -> Dict[str, Union[pd.Series, int]]:
    """Identify outliers in numeric columns using the IQR method.

    This function calculates outliers for each numeric column in the DataFrame 
    and computes both the count of outliers per column and the total count across all columns.

    Args:
        df (pd.DataFrame): The DataFrame to analyze for outliers.

    Returns:
        Dict[str, Union[pd.Series, int]]: A dictionary with keys:
            - 'outliers_per_column': Series containing count of outliers per column.
            - 'total_outliers': Total number of outliers detected across the DataFrame.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df < lower_bound) | (df > upper_bound)
    outliers_count = outliers.sum()
    total_outliers = outliers.values.sum()

    return {"outliers_per_column": outliers_count, "total_outliers": total_outliers}


def plot_histograms(
    df: pd.DataFrame, features: List[str], nbins: int = 40, save_path: Optional[str] = None
) -> go.Figure:
    """Plot histograms for the specified features in the DataFrame.

    This function creates histograms for the given features, displaying them side by side.
    The x-axis of each histogram is labeled with the corresponding feature name.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        features (List[str]): List of feature names for which to plot histograms.
        nbins (int, optional): Number of bins for each histogram. Defaults to 40.
        save_path (Optional[str], optional): File path to save the image. Defaults to None.

    Returns:
        go.Figure: The Plotly figure object containing the histograms.
    """
    title = f"Distribution of {', '.join(features)}"
    rows = 1
    cols = len(features)

    fig = sp.make_subplots(rows=rows, cols=cols, horizontal_spacing=0.1)

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Histogram(
                x=df[feature],
                nbinsx=nbins,
                name=feature,
                marker=dict(
                    color=([PRIMARY_BLUE, SECONDARY_BLUE, LAVENDER])[i % 3],
                    line=dict(color="#000000", width=1),
                ),
            ),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(title_text=feature, row=1, col=i + 1, title_font=dict(size=14))
        fig.update_yaxes(title_text="Count", row=1, col=i + 1, title_font=dict(size=14))

    fig.update_layout(
        **BASE_LAYOUT,
        title_text=title,
        title_x=0.5,
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE, color=TEXT_DARK),
        showlegend=False,
        height=500,
        width=400 * cols,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    if save_path:
        fig.write_image(save_path)
    return fig


def plot_heatmap(corr_matrix: pd.DataFrame, save_path: Optional[str] = None) -> go.Figure:
    """Display a heatmap of the provided correlation matrix.

    This function creates a heatmap visualization of the correlation matrix using Plotly,
    applying a custom color scale.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix to visualize.
        save_path (Optional[str], optional): File path to save the image. Defaults to None.

    Returns:
        go.Figure: The Plotly figure object containing the heatmap.
    """
    colorscale = [
        [0.0, PRIMARY_BLUE],
        [0.5, LAVENDER],
        [1.0, SECONDARY_BLUE],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=colorscale,
            colorbar=dict(
                title={'text': 'Correlation', 'font': {'family': FONT_FAMILY, 'size': FONT_SIZE_AXIS, 'color': TEXT_DARK}},
                tickfont={'family': FONT_FAMILY, 'size': FONT_SIZE_TICK, 'color': TEXT_DARK}
            ),
        )
    )

    # Remove xaxis and yaxis settings from BASE_LAYOUT to avoid duplicate keyword issues
    base_layout = dict(BASE_LAYOUT)
    base_layout.pop("xaxis", None)
    base_layout.pop("yaxis", None)
    base_layout.update({
        "title": {
            "text": "Correlation Matrix",
            "x": 0.5,
            "xanchor": "center",
            "font": {"family": FONT_FAMILY, "size": FONT_SIZE_TITLE, "color": TEXT_DARK},
        },
        "xaxis": {
            'title': {'text': 'Features', 'font': {'family': FONT_FAMILY, 'size': FONT_SIZE_AXIS, 'color': TEXT_DARK}},
            'tickfont': {'family': FONT_FAMILY, 'size': FONT_SIZE_AXIS, 'color': TEXT_DARK},
            'tickangle': 45,
        },
        "yaxis": {
            'title': {'text': 'Features', 'font': {'family': FONT_FAMILY, 'size': FONT_SIZE_AXIS, 'color': TEXT_DARK}},
            'tickfont': {'family': FONT_FAMILY, 'size': FONT_SIZE_AXIS, 'color': TEXT_DARK},
        },
        "height": 500,
        "width": 1600,
        "margin": {"l": 100, "r": 100, "t": 100, "b": 100},
    })
    fig.layout.update(base_layout)

    # Add correlation annotations to each cell for improved readability
    for i, row in enumerate(corr_matrix.values):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=corr_matrix.columns[j],
                y=corr_matrix.index[i],
                text=str(round(val, 2)),
                showarrow=False,
                font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK, color=TEXT_DARK),
                xref="x", yref="y"
            )

    if save_path:
        fig.write_image(save_path)
    return fig


def test_correlation(
    data: pd.DataFrame, var1: str, var2: str
) -> Dict[str, Union[float, bool, Tuple[float, float]]]:
    """Calculate and test the Pearson correlation between two variables.

    This function computes the Pearson correlation coefficient between two variables,
    evaluates the statistical significance, and determines whether to reject the null hypothesis
    (that the correlation is zero). It also calculates the 95% confidence interval for the correlation.

    Args:
        data (pd.DataFrame): The DataFrame containing the variables.
        var1 (str): The name of the first variable.
        var2 (str): The name of the second variable.

    Returns:
        Dict[str, Union[float, bool, Tuple[float, float]]]: A dictionary containing:
            - "Correlation": The Pearson correlation coefficient.
            - "P-Value": The p-value of the correlation.
            - "Reject H0": True if the null hypothesis is rejected, else False.
            - "95% CI": A tuple with the lower and upper bounds of the 95% confidence interval.
    """
    correlation, p_value = pearsonr(data[var1], data[var2])
    alpha = 0.05
    reject_h0 = p_value < alpha
    ci_low, ci_high = np.percentile(
        [
            correlation - (1.96 * np.sqrt((1 - correlation ** 2) / (len(data) - 3))),
            correlation + (1.96 * np.sqrt((1 - correlation ** 2) / (len(data) - 3))),
        ],
        [2.5, 97.5],
    )
    return {
        "Correlation": correlation,
        "P-Value": p_value,
        "Reject H0": reject_h0,
        "95% CI": (ci_low, ci_high),
    }


def plot_coefficients(
    model: OLS, title: str = "Coefficient Estimates and Confidence Intervals"
) -> go.Figure:
    """Visualize model coefficients and their confidence intervals.

    This function creates a bar plot of the coefficients from a fitted OLS model,
    complete with their corresponding confidence intervals.

    Args:
        model (OLS): The fitted statsmodels OLS model.
        title (str, optional): The title of the plot. Defaults to "Coefficient Estimates and Confidence Intervals".

    Returns:
        go.Figure: The Plotly figure object containing the bar plot.
    """
    coefficients = model.params
    conf_int = model.conf_int()
    conf_int_df = pd.DataFrame(conf_int, columns=["Lower CI", "Upper CI"])
    coefficients_df = pd.DataFrame(coefficients, columns=["Coefficient"])
    coefficients_summary = coefficients_df.join(conf_int_df)

    fig = px.bar(
        coefficients_summary,
        y="Coefficient",
        error_y="Upper CI",
        error_y_minus="Lower CI",
        title=title,
        color_discrete_sequence=[SECONDARY_BLUE],
    )
    fig.update_layout(
        **BASE_LAYOUT,
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE, color=TEXT_DARK),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def plot_correlation(
    data: pd.DataFrame, var1: str, var2: str, save_path: Optional[str] = None
) -> go.Figure:
    """Visualize the correlation between two variables with a regression line.

    This function creates a scatter plot of two variables from the DataFrame,
    fits a linear regression model, and annotates the plot with statistical details
    including Pearson correlation, p-value, 95% confidence interval, and the line equation.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        var1 (str): The name of the first variable.
        var2 (str): The name of the second variable.
        save_path (Optional[str], optional): File path to save the plot image. Defaults to None.

    Returns:
        go.Figure: The Plotly figure object containing the scatter plot and regression line.
    """
    stats_dict = test_correlation(data, var1, var2)

    fig = px.scatter(
        data,
        x=var1,
        y=var2,
        trendline="ols",
        labels={var1: var1, var2: var2},
        title=f"Scatter Plot of {var1} vs {var2}",
        color_discrete_sequence=[PRIMARY_BLUE],
    )

    results = px.get_trendline_results(fig)
    slope = results.iloc[0]["px_fit_results"].params[1]
    intercept = results.iloc[0]["px_fit_results"].params[0]

    fig.data[-1].line.color = SECONDARY_BLUE

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            text=[
                f"Correlation: {stats_dict['Correlation']:.3f}<br>P-value: {stats_dict['P-Value']:.3g}<br>"
                f"95% CI: [{stats_dict['95% CI'][0]:.3f}, {stats_dict['95% CI'][1]:.3f}]<br>"
                f"Line Equation: y = {slope:.3f}x + {intercept:.3f}"
            ],
            mode="text",
            textposition="bottom right",
            showlegend=False,
            textfont=dict(family="sans serif", size=12, color="black"),
        )
    )

    fig.update_layout(
        **BASE_LAYOUT,
        title={
            "text": f"Scatter Plot of {var1} vs {var2}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"family": FONT_FAMILY, "size": FONT_SIZE_TITLE, "color": TEXT_DARK},
        },
        xaxis_title=var1,
        yaxis_title=var2,
        xaxis_title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS, color=TEXT_DARK),
        yaxis_title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS, color=TEXT_DARK),
        margin=dict(l=50, r=50, t=80, b=50),
        height=500,
        width=1600,
        annotations=[
            {
                "text": f"Statistical Details:<br>Correlation: {stats_dict['Correlation']:.3f}<br>P-value: {stats_dict['P-Value']:.3g}<br>"
                        f"95% CI: [{stats_dict['95% CI'][0]:.3f}, {stats_dict['95% CI'][1]:.3f}]<br>"
                        f"Line Equation: y = {slope:.3f}x + {intercept:.3f}",
                "align": "left",
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 1.05,
                "y": 0.5,
                "bordercolor": "black",
                "borderwidth": 1,
            }
        ],
    )

    if save_path:
        fig.write_image(save_path)
    return fig


def log_transform_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply logarithmic transformation to specified columns to reduce skewness.

    This function applies a logarithmic transformation (using log1p) to the given columns 
    in the DataFrame. If a column contains non-positive values, a small positive offset is added 
    to enable the logarithmic transformation.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (List[str]): A list of column names to transform.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns log-transformed.
    """
    transformed_df = df.copy()

    for column in columns:
        if (transformed_df[column] <= 0).any():
            min_positive = transformed_df[column][transformed_df[column] > 0].min()
            transformed_df[column] = transformed_df[column] + (min_positive * 0.1)

        transformed_df[column] = np.log1p(transformed_df[column])

    return transformed_df


def train_linear_model(
    data: pd.DataFrame, target_column: str
) -> Tuple[OLS, pd.DataFrame, pd.Series, np.ndarray]:
    """Train a linear regression model using Ordinary Least Squares (OLS).

    This function splits the data into training and testing sets, adds a constant term
    to the features, and fits an OLS model on the training data. It then predicts values 
    on the test set and prints model evaluation metrics such as Mean Squared Error and 
    R-squared.

    Args:
        data (pd.DataFrame): The input DataFrame containing features and the target column.
        target_column (str): The name of the target column.

    Returns:
        Tuple[OLS, pd.DataFrame, pd.Series, np.ndarray]: A tuple containing:
            - The fitted OLS model.
            - The test features (X_test).
            - The test target values (y_test).
            - The predicted target values for the test features (y_pred).
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = OLS(y_train, X_train)
    results = model.fit()

    y_pred = results.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

    print(f"Model summary:\n{results.summary()}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print(f"Adjusted R-squared: {adj_r2}")

    return results, X_test, y_test, y_pred


def plot_model_predictions(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    title: str,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Plot residuals from model predictions.

    This function creates a scatter plot of the residuals (difference between actual and predicted values)
    with an overlaid trendline. It is useful for visualizing the performance of a regression model.

    Args:
        x_test (pd.DataFrame): The test features.
        y_test (pd.Series): The actual target values.
        y_pred (pd.Series): The predicted target values.
        title (str): The title for the plot.
        save_path (Optional[str], optional): File path to save the plot image. Defaults to None.

    Returns:
        go.Figure: The Plotly figure object containing the scatter plot and trendline.
    """
    residuals = y_test - y_pred
    results_df = pd.DataFrame({"Predicted": y_pred, "Residuals": residuals}).reset_index(drop=True)

    fig = px.scatter(
        results_df,
        x="Predicted",
        y="Residuals",
        title=title,
        labels={"Predicted": "Predicted Values", "Residuals": "Residuals"},
        trendline="ols",
    )

    marker_color = PRIMARY_BLUE
    trendline_color = SECONDARY_BLUE

    fig.update_traces(marker=dict(color=marker_color))

    # Update layout by merging BASE_LAYOUT without xaxis and yaxis keys to avoid conflicts
    base_layout = dict(BASE_LAYOUT)
    base_layout.pop("xaxis", None)
    base_layout.pop("yaxis", None)
    base_layout.update({
        "title": {
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"family": FONT_FAMILY, "size": FONT_SIZE_TITLE, "color": TEXT_DARK},
        },
        "xaxis_title": "Predicted Values",
        "yaxis_title": "Residuals",
        "xaxis_title_font": dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS, color=TEXT_DARK),
        "yaxis_title_font": dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS, color=TEXT_DARK),
        "margin": {"l": 50, "r": 50, "t": 80, "b": 50},
        "height": 500,
        "width": 1600,
    })
    fig.layout.update(base_layout)

    fig.data[1].line.color = trendline_color

    if save_path:
        fig.write_image(save_path)
    return fig
