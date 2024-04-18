from typing import List, NoReturn

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy import stats
from scipy.stats import pearsonr

# Define a consistent color scheme for visualizations
PRIMARY_COLORS = ["#5684F7", "#3A5CED", "#7E7AE6"]
SECONDARY_COLORS = ["#7BC0FF", "#B8CCF4", "#18407F", "#85A2FF", "#C2A9FF", "#3D3270"]

# Combine both color lists for extended options
ALL_COLORS = PRIMARY_COLORS + SECONDARY_COLORS


def get_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns a list of column names from the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to extract column names from.

    Returns:
    - List[str]: Column names from the DataFrame.
    """
    return df.columns.tolist()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame and prints the number of duplicates found.

    Parameters:
    - df (pd.DataFrame): The DataFrame to remove duplicates from.

    Returns:
    - pd.DataFrame: DataFrame with duplicates removed.
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    print(f"Removed {initial_count - final_count} duplicate rows")
    return df


def plot_box_chart(
    df: pd.DataFrame, x_label: str, y_label: str, chart_title: str
) -> None:
    """
    Creates a box plot for each column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - x_label (str): The label for the x-axis.
    - y_label (str): The label for the y-axis.
    - chart_title (str): The title of the chart.
    """
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                boxpoints="outliers",
                marker_color=SECONDARY_COLORS[2],
                line_color=PRIMARY_COLORS[0],
            )
        )
    fig.update_layout(
        title=chart_title, xaxis_title=x_label, yaxis_title=y_label, title_x=0.5
    )
    fig.show()


def identify_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies outliers using the IQR method for each numeric column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame to analyze for outliers.

    Returns:
    - pd.DataFrame: Count of outliers in each column.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df < lower_bound) | (df > upper_bound)).sum()
    return outliers


def plot_histograms(df: pd.DataFrame, features: List[str], nbins: int = 40) -> None:
    """
    Plots histograms for specified features in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame to plot.
    - features (List[str]): List of features to plot histograms for.
    - nbins (int): Number of bins to use in histograms.
    """
    title = f"Distribution of {', '.join(features)}"

    fig = sp.make_subplots(rows=1, cols=len(features))

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Histogram(
                x=df[feature],
                nbinsx=nbins,
                name=feature,
                marker=dict(
                    color=ALL_COLORS[i % len(ALL_COLORS)],
                    line=dict(color="#000000", width=1),
                ),
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title_text=title, title_x=0.5, showlegend=False  # Centers the title
    )

    fig.show()


def plot_heatmap(corr_matrix: pd.DataFrame) -> None:
    """
    Plots a heatmap based on a correlation matrix using a specified color scheme.

    Parameters:
    - corr_matrix (pd.DataFrame): Correlation matrix to plot.
    """
    colorscale = [
        [0.0, SECONDARY_COLORS[5]],  # Dark blue for low correlations
        [0.2, SECONDARY_COLORS[4]],  # Lighter blue
        [0.4, PRIMARY_COLORS[2]],  # Light purple
        [0.6, PRIMARY_COLORS[1]],  # Medium purple
        [0.8, PRIMARY_COLORS[0]],  # Rich blue
        [1.0, SECONDARY_COLORS[0]],  # Light blue for high correlations
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=colorscale,
        )
    )

    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        title_x=0.5,
    )
    fig.show()


def plot_top_correlations(
    df: pd.DataFrame, corr_matrix: pd.DataFrame, top_n: int
) -> None:
    """
    Plots the top N positive and negative correlations from the correlation matrix.

    Parameters:
    - df (pd.DataFrame): DataFrame to analyze.
    - corr_matrix (pd.DataFrame): Correlation matrix of the DataFrame.
    - top_n (int): Number of top correlations to plot.
    """
    correlations = corr_matrix.unstack().reset_index()
    correlations.columns = ["Feature 1", "Feature 2", "Correlation"]
    correlations = correlations[correlations["Feature 1"] != correlations["Feature 2"]]
    top_pos_corr = correlations.nlargest(top_n, "Correlation")
    top_neg_corr = correlations.nsmallest(top_n, "Correlation")

    for _, row in top_pos_corr.iterrows():
        fig = px.scatter(
            df,
            x=row["Feature 1"],
            y=row["Feature 2"],
            trendline="ols",
            title=f"Positive: {row['Feature 1']} vs {row['Feature 2']}",
            color_discrete_sequence=[PRIMARY_COLORS[0]],
        )
        fig.update_layout(title_x=0.5)
        fig.show()

    for _, row in top_neg_corr.iterrows():
        fig = px.scatter(
            df,
            x=row["Feature 1"],
            y=row["Feature 2"],
            trendline="ols",
            title=f"Negative: {row['Feature 1']} vs {row['Feature 2']}",
            color_discrete_sequence=[PRIMARY_COLORS[1]],
        )
        fig.update_layout(title_x=0.5)
        fig.show()


def test_correlation(data, var1, var2):
    """
    Calculate the Pearson correlation between two variables and test the null hypothesis that the correlation is zero.

    Parameters:
    data (DataFrame): The dataset containing the variables.
    var1 (str): The name of the first variable.
    var2 (str): The name of the second variable.

    Returns:
    dict: A dictionary containing the correlation, p-value, a boolean indicating whether to reject the null hypothesis,
    and the 95% confidence interval of the correlation.
    """
    correlation, p_value = pearsonr(data[var1], data[var2])
    alpha = 0.05
    reject_h0 = p_value < alpha
    ci_low, ci_high = np.percentile(
        [
            correlation - (1.96 * np.sqrt((1 - correlation**2) / (len(data) - 3))),
            correlation + (1.96 * np.sqrt((1 - correlation**2) / (len(data) - 3))),
        ],
        [2.5, 97.5],
    )  # 95% CI
    return {
        "Correlation": correlation,
        "P-Value": p_value,
        "Reject H0": reject_h0,
        "95% CI": (ci_low, ci_high),
    }
