import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from typing import List, NoReturn
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

color_scheme = ['#dcb0f2', '#f6a5e6', '#ff99d1', '#ff8fb4', '#ff8b91', '#ff8e6b', '#ff9842', '#ffa600']


def get_columns(df: pd.DataFrame) -> List[str]:
    """
    This function returns a list of column names from a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which to extract column names.

    Returns:
    list: A list containing the names of all columns in the DataFrame.
    """
    return [col for col in df]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function checks for duplicate rows in a DataFrame and removes them.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which to remove duplicates.

    Returns:
    pandas.DataFrame: The DataFrame after removing duplicates.
    """
    # Check for duplicates
    duplicates = df.duplicated()

    # If there are duplicates, print the number of duplicates and remove them
    if duplicates.any():
        print(f"Number of duplicate rows: {duplicates.sum()}")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found")

    # Print the number of rows after removing duplicates
    print(f"Number of rows after removing duplicates: {len(df)}")

    return df


def plot_box_chart(df: pd.DataFrame, x_label: str, y_label: str, chart_title: str) -> None:
    fig = go.Figure()

    for col in df.columns:
        fig.add_trace(go.Box(
            y=df[col],
            name=col,
            boxpoints='outliers',
            marker_color='#18407F',
            line_color='#5684F7',
            whiskerwidth=0.2
        ))

    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title_text=chart_title, title_x=0.5)
    fig.show()


def identify_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function identifies the outliers in a DataFrame using the Interquartile Range (IQR) method and returns the count of outliers in each column.

    Parameters:
    df (pandas.DataFrame): The DataFrame in which to find outliers.

    Returns:
    pandas.DataFrame: A DataFrame where each cell contains the count of outliers in the corresponding column.
    """

    # Calculate the first quartile (Q1) for each column in the DataFrame
    Q1 = df.quantile(0.25)

    # Calculate the third quartile (Q3) for each column in the DataFrame
    Q3 = df.quantile(0.75)

    # Calculate the Interquartile Range (IQR) for each column in the DataFrame
    IQR = Q3 - Q1

    # Determine the lower bound for outliers
    lower_bound = Q1 - 1.5 * IQR

    # Determine the upper bound for outliers
    upper_bound = Q3 + 1.5 * IQR

    # Identify the outliers in the DataFrame
    outliers = df[(df < lower_bound) | (df > upper_bound)]

    # Count the number of outliers in each column
    outliers_count = outliers.count()

    return outliers_count


def plot_histograms(df: pd.DataFrame, features: List[str], nbins: int = 40) -> None:
    """
    This function plots histograms for the given features of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which to extract data.
    features (List[str]): A list of column names for which to plot histograms.
    nbins (int, optional): The number of bins to use in the histogram. Defaults to 40.

    Returns:
    None
    """
    color_palette: list[str] = ['#5684F7', '#3A5CED', '#7E7AE6', '#C2A9FF']
    fig = sp.make_subplots(rows=1, cols=len(features))

    for i, feature in enumerate(features):
        hist = go.Histogram(x=df[feature], nbinsx=nbins, name=feature, marker=dict(color=color_palette[i % len(color_palette)], line=dict(color='#000000', width=1)))
        fig.add_trace(hist, row=1, col=i+1)
        fig.update_xaxes(title_text=feature, row=1, col=i+1)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_layout(height=400, width=800, title_text=f"Distribution of {', '.join(features)}", title_x=0.5, showlegend=False)
    fig.show()


def plot_heatmap(corr_matrix: pd.DataFrame, colors_matrix: List[str]) -> None:
    """
    This function plots a heatmap using the correlation matrix and a color matrix.

    Parameters:
    corr_matrix (DataFrame): A correlation matrix.
    colors_matrix (List[str]): A list of colors for the heatmap.

    Returns:
    None
    """
    color_positions = [i/(len(colors_matrix)-1) for i in range(len(colors_matrix))]
    color_scale = [[pos, color] for pos, color in zip(color_positions, colors_matrix)]

    heatmap = go.Heatmap(
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        z=corr_matrix.values,
        colorscale=color_scale,
    )

    fig = go.Figure(data=heatmap)

    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        title_x=0.5,
    )

    fig.show()


def plot_top_correlations(df: pd.DataFrame, corr_matrix: pd.DataFrame, top_n: int) -> None:
    """
    This function plots the top positive and negative correlations in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    corr_matrix (pd.DataFrame): The correlation matrix of the DataFrame.
    top_n (int): The number of top correlations to plot.

    Returns:
    None
    """
    correlations = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(np.bool_))
    correlations = correlations.stack().reset_index()
    correlations.columns = ['Feature 1', 'Feature 2', 'Correlation']

    top_pos_corr = correlations.sort_values(by='Correlation', ascending=False).head(top_n)
    top_neg_corr = correlations.sort_values(by='Correlation').head(top_n)

    for i in range(top_n):
        fig = px.scatter(df, x=top_pos_corr.iloc[i, 0], y=top_pos_corr.iloc[i, 1],
                         trendline="ols",
                         title=f"Positive Correlation: {top_pos_corr.iloc[i, 0]} vs {top_pos_corr.iloc[i, 1]}",
                         color_discrete_sequence=['#5684F7'],
                         trendline_color_override="#18407F")
        fig.update_layout(title_x=0.5)
        fig.show()

    for i in range(top_n):
        fig = px.scatter(df, x=top_neg_corr.iloc[i, 0], y=top_neg_corr.iloc[i, 1],
                         trendline="ols",
                         title=f"Negative Correlation: {top_neg_corr.iloc[i, 0]} vs {top_neg_corr.iloc[i, 1]}",
                         color_discrete_sequence=['#3A5CED'],
                         trendline_color_override="#18407F")
        fig.update_layout(title_x=0.5)
        fig.show()
