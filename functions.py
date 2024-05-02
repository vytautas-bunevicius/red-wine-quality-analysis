import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import OLS
from typing import Dict, List, Tuple, Union

import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio

PRIMARY_COLORS = ["#5684F7", "#3A5CED", "#7E7AE6"]
SECONDARY_COLORS = ["#7BC0FF", "#B8CCF4", "#18407F", "#85A2FF", "#C2A9FF", "#3D3270"]
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


def plot_box_chart(df: pd.DataFrame, x_label: str, y_label: str, chart_title: str, save_path: str = None) -> None:
    """
    Creates a box plot for each column in the DataFrame and optionally saves it as an image.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - x_label (str): The label for the x-axis.
    - y_label (str): The label for the y-axis.
    - chart_title (str): The title of the chart.
    - save_path (str, optional): Path to save the plot as an image. If None, the plot is not saved.

    Returns:
    - None

    Example usage:
    plot_box_chart(df, "X Label", "Y Label", "Box Chart Title", save_path="plot.png")
    """
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                boxpoints="outliers",
                marker_color=SECONDARY_COLORS[4],
                line_color=SECONDARY_COLORS[0],
            )
        )
    fig.update_layout(
        title=chart_title, xaxis_title=x_label, yaxis_title=y_label, title_x=0.5
    )
    fig.show()

    if save_path:
        fig.write_image(save_path)


def identify_outliers(df: pd.DataFrame) -> Dict[str, Union[pd.Series, int]]:
    """
    Identifies outliers using the IQR method for each numeric column in the DataFrame and calculates the total count of outliers.

    Parameters:
    - df (pd.DataFrame): DataFrame to analyze for outliers.

    Returns:
    - dict: A dictionary containing:
        - 'outliers_per_column': Count of outliers in each column.
        - 'total_outliers': Total number of outliers across all columns.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((df < lower_bound) | (df > upper_bound))
    outliers_count = outliers.sum()

    total_outliers = outliers.values.sum()

    return {
        'outliers_per_column': outliers_count,
        'total_outliers': total_outliers
    }


def plot_histograms(df: pd.DataFrame, features: List[str], nbins: int = 40, save_path: str = None) -> None:
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
        title_text=title, title_x=0.5, showlegend=False
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_heatmap(corr_matrix: pd.DataFrame, save_path: str = None) -> None:
    """
    Plots a heatmap based on a correlation matrix using a specified color scheme.

    Parameters:
    - corr_matrix (pd.DataFrame): Correlation matrix to plot.
    - save_path (str, optional): Path to save the plot as an image. If None, the plot is not saved.

    Returns:
    - None
    """
    colorscale = [
        [0.0, SECONDARY_COLORS[5]], 
        [0.2, SECONDARY_COLORS[4]],  
        [0.4, PRIMARY_COLORS[2]],  
        [0.6, PRIMARY_COLORS[1]],  
        [0.8, PRIMARY_COLORS[0]],  
        [1.0, SECONDARY_COLORS[0]],
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
    if save_path:
        fig.write_image(save_path)


def test_correlation(data: pd.DataFrame, var1: str, var2: str) -> Dict[str, Union[float, bool, Tuple[float, float]]]:
    """
    Calculate the Pearson correlation between two variables and test the null hypothesis that the correlation is zero.

    Parameters:
    - data (DataFrame): The dataset containing the variables.
    - var1 (str): The name of the first variable.
    - var2 (str): The name of the second variable.

    Returns:
    - dict: A dictionary containing the following keys:
        - "Correlation" (float): The Pearson correlation coefficient between var1 and var2.
        - "P-Value" (float): The p-value associated with the correlation coefficient.
        - "Reject H0" (bool): A boolean indicating whether to reject the null hypothesis.
        - "95% CI" (tuple): A tuple containing the lower and upper bounds of the 95% confidence interval of the correlation coefficient.
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
    )
    return {
        "Correlation": correlation,
        "P-Value": p_value,
        "Reject H0": reject_h0,
        "95% CI": (ci_low, ci_high),
    }


def plot_coefficients(model: OLS, title: str = "Coefficient Estimates and Confidence Intervals") -> None:
    """
    Creates a bar plot of model coefficients and their confidence intervals using predefined color codes.

    Parameters:
    - model (statsmodels OLS model): The fitted model object.
    - title (str): Title of the plot.

    Returns:
    - None
    """
    coefficients = model.params
    conf_int = model.conf_int()
    conf_int_df = pd.DataFrame(conf_int, columns=['Lower CI', 'Upper CI'])
    coefficients_df = pd.DataFrame(coefficients, columns=['Coefficient'])
    coefficients_summary = coefficients_df.join(conf_int_df)

    fig = px.bar(coefficients_summary, y='Coefficient', error_y='Upper CI', error_y_minus='Lower CI', title=title,
                 color_discrete_sequence=[PRIMARY_COLORS[1]])
    fig.show()


def plot_correlation(data: pd.DataFrame, var1: str, var2: str, save_path: str = None) -> go.Figure:
    """
    Visualize the Pearson correlation between two variables using Plotly, with improved annotation placement and specified trendline color.

    Parameters:
    - data (pd.DataFrame): The dataset containing the variables.
    - var1 (str): The name of the first variable.
    - var2 (str): The name of the second variable.
    - save_path (str): The file path to save the plot as a PNG image.

    Returns:
    - go.Figure: A Plotly figure object with the scatter plot, regression line, and improved annotations.
    """
    stats = test_correlation(data, var1, var2)

    fig = px.scatter(data, x=var1, y=var2, trendline="ols", labels={var1: var1, var2: var2},
                     title=f"Scatter Plot of {var1} vs {var2}", color_discrete_sequence=[SECONDARY_COLORS[0]])

    results = px.get_trendline_results(fig)
    slope = results.iloc[0]["px_fit_results"].params[1]
    intercept = results.iloc[0]["px_fit_results"].params[0]

    fig.data[-1].line.color = SECONDARY_COLORS[4]

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        text=[f"Correlation: {stats['Correlation']:.3f}<br>P-value: {stats['P-Value']:.3g}<br>"
              f"95% CI: [{stats['95% CI'][0]:.3f}, {stats['95% CI'][1]:.3f}]<br>"
              f"Line Equation: y = {slope:.3f}x + {intercept:.3f}"],
        mode="text",
        textposition="bottom right",
        showlegend=False,
        textfont=dict(family="sans serif", size=12, color="black")
    ))

    fig.update_layout(
        xaxis=dict(domain=[0, 0.85]),
        title_x=0.5,
        annotations=[{
            'text': f"Statistical Details:<br>Correlation: {stats['Correlation']:.3f}<br>P-value: {stats['P-Value']:.3g}<br>"
                    f"95% CI: [{stats['95% CI'][0]:.3f}, {stats['95% CI'][1]:.3f}]<br>"
                    f"Line Equation: y = {slope:.3f}x + {intercept:.3f}",
            'align': 'left',
            'showarrow': False,
            'xref': 'paper',
            'yref': 'paper',
            'x': 1.05,
            'y': 0.5,
            'bordercolor': 'black',
            'borderwidth': 1
        }]
    )

    fig.show()
    if save_path:
        fig.write_image(save_path)


def log_transform_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply log transformation to specified columns in a DataFrame to reduce skewness.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - columns (List[str]): List of column names to apply the log transformation.

    Returns:
    - pd.DataFrame: A DataFrame with the log-transformed values in the specified columns.
    """
    transformed_df = df.copy()

    for column in columns:
        if (transformed_df[column] <= 0).any():
            min_positive = transformed_df[column][transformed_df[column] > 0].min()
            transformed_df[column] = transformed_df[column] + (min_positive * 0.1)

        transformed_df[column] = np.log1p(transformed_df[column])

    return transformed_df


def train_linear_model(data: pd.DataFrame, target_column: str) -> Tuple[OLS, pd.DataFrame, pd.Series, np.ndarray]:
    """
    Trains a linear regression model using the given data and target column.

    Parameters:
    - data (pd.DataFrame): The input data containing the features and target column.
    - target_column (str): The name of the target column in the data.

    Returns:
    - results (statsmodels.regression.linear_model.RegressionResultsWrapper): The trained model results.
    - X_test (pd.DataFrame): The test features.
    - y_test (pd.Series): The test target values.
    - y_pred (np.ndarray): The predicted target values for the test features.
    """

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = OLS(y_train, X_train)
    results = model.fit()

    y_pred = results.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)

    print(f"Model summary:\n{results.summary()}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print(f"Adjusted R-squared: {adj_r2}")

    return results, X_test, y_test, y_pred


def plot_model_predictions(x_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series, title: str, save_path: str = None) -> None:
    """
    Plot the residuals from model predictions with a trendline, using specified color palette.

    Parameters:
    - x_test (pd.DataFrame): The input DataFrame containing the test data.
    - y_test (pd.Series): The actual target values.
    - y_pred (pd.Series): The predicted target values.
    - title (str): The title of the plot.
    - save_path (str): The file path to save the plot as a PNG image.

    Returns:
    - None
    """
    residuals = y_test - y_pred
    results_df = pd.DataFrame({
        'Predicted': y_pred,
        'Residuals': residuals
    }).reset_index(drop=True)

    fig = px.scatter(results_df, x='Predicted', y='Residuals',
                     title=title, labels={'Predicted': 'Predicted Values', 'Residuals': 'Residuals'},
                     trendline="ols")
    
    marker_color = SECONDARY_COLORS[0]
    trendline_color = SECONDARY_COLORS[4]

    fig.update_traces(marker=dict(color=marker_color))
    fig.update_layout(
        title_x=0.5,
        plot_bgcolor='white',
        xaxis=dict(title='Predicted Values'),
        yaxis=dict(title='Residuals'),
        showlegend=False
    )

    fig.data[1].line.color = trendline_color

    fig.show()
    
    if save_path:
        pio.write_image(fig, save_path)