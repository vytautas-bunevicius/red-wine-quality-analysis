"""
Unit tests for the red_wine_quality_utils module using pytest.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import statsmodels.api as sm

import red_wine_quality_utils as utils


@pytest.fixture
def df():
    """Fixture returning a simple DataFrame for testing."""
    return pd.DataFrame(
        {"A": [1, 2, 2, 3, 4], "B": [5, 6, 6, 7, 8], "C": [9, 10, 10, 11, 12]}
    )


@pytest.fixture
def numeric_df():
    """Fixture returning a numeric DataFrame for testing outlier detection."""
    return pd.DataFrame(
        {
            "col1": [1, 2, 3, 100],
            "col2": [10, 20, 30, -100],
            "col3": [100, 200, 300, 400],
        }
    )


def test_get_columns(df):
    """Test that get_columns returns the correct list of DataFrame columns."""
    cols = utils.get_columns(df)
    assert cols == ["A", "B", "C"]


def test_remove_duplicates(df):
    """Test that remove_duplicates correctly removes duplicate rows from a DataFrame."""
    initial_length = len(df)
    df_no_dup = utils.remove_duplicates(df)
    expected_length = initial_length - 1
    assert len(df_no_dup) == expected_length


def test_identify_outliers(numeric_df):
    """Test that identify_outliers returns a dictionary with outlier counts using the IQR method."""
    results = utils.identify_outliers(numeric_df)
    assert "total_outliers" in results
    assert "outliers_per_column" in results
    assert isinstance(results["total_outliers"], (int, np.integer))
    assert hasattr(results["outliers_per_column"], "to_dict")


def test_log_transform_features():
    """Test that log_transform_features applies a logarithmic transformation to specified columns."""
    df_local = pd.DataFrame(
        {
            "A": [1, 10, 100],
            "B": [0, 5, 10],
        }
    )
    transformed = utils.log_transform_features(df_local, columns=["A", "B"])
    np.testing.assert_allclose(
        transformed["A"], np.log1p(df_local["A"]), rtol=1e-5
    )
    assert (transformed["B"] >= 0).all()


def test_train_linear_model():
    """Test that train_linear_model returns a fitted model and makes predictions with expected shapes."""
    df_local = pd.DataFrame({"X": np.arange(10), "Y": np.arange(10) * 2 + 1})
    result, X_test, y_test, y_pred = utils.train_linear_model(
        df_local, target_column="Y"
    )
    assert hasattr(result, "summary")
    assert len(X_test) == len(y_test)
    assert len(y_test) == len(y_pred)


def test_plot_functions(tmp_path):
    """Test that plotting functions execute without errors and save files when a save path is provided."""
    df_local = pd.DataFrame(
        {
            "A": np.random.rand(50),
            "B": np.random.rand(50),
            "C": np.random.rand(50),
        }
    )

    tmp_file = tmp_path / "box_chart.png"
    utils.plot_box_chart(
        df_local,
        x_label="X",
        y_label="Y",
        chart_title="Test Box Chart",
        save_path=str(tmp_file),
    )
    assert tmp_file.exists()

    tmp_file = tmp_path / "histograms.png"
    utils.plot_histograms(
        df_local, features=["A", "B"], nbins=10, save_path=str(tmp_file)
    )
    assert tmp_file.exists()

    corr_matrix = df_local.corr()
    tmp_file = tmp_path / "heatmap.png"
    utils.plot_heatmap(corr_matrix, save_path=str(tmp_file))
    assert tmp_file.exists()

    fig = utils.plot_correlation(df_local, var1="A", var2="B", save_path=None)
    assert isinstance(fig, go.Figure)


def test_plot_coefficients():
    """Test that plot_coefficients executes without errors for a fitted model."""
    df_local = pd.DataFrame({"X": np.arange(10), "Y": np.arange(10) * 3 + 2})
    X = sm.add_constant(df_local["X"])
    model = sm.OLS(df_local["Y"], X).fit()
    try:
        utils.plot_coefficients(model, title="Test Coefficients")
    except Exception as e:
        pytest.fail(f"plot_coefficients raised an exception: {e}")


def test_plot_model_predictions(tmp_path):
    """Test that plot_model_predictions executes without errors and saves an image when a save path is provided."""
    df_local = pd.DataFrame({"X": np.arange(10), "Y": np.arange(10) * 4 + 1})
    result, X_test, y_test, y_pred = utils.train_linear_model(
        df_local, target_column="Y"
    )
    tmp_file = tmp_path / "model_predictions.png"
    utils.plot_model_predictions(
        X_test,
        y_test,
        y_pred,
        title="Test Model Predictions",
        save_path=str(tmp_file),
    )
    assert tmp_file.exists()
