## ---- Load required libraries ------------------------------------------------


import pandas as pd
import calendar
import numpy as np

# ==============================================================================
#                       FUNCTION TO PLOT TREND SUBSERIES
# ==============================================================================


def plot_trend_subseries(decomposed, disease_name=""):
    """
    Plot a trend subseries (one line per year) from an STL decomposition.

    Parameters
    ----------
    decomposed : STL decomposition result
        The object returned by STL(...).fit()
        Must have a MultiIndex with levels ["disease", "time"].

    disease_name : str
        Name of the disease to extract (optional if already sliced).
    """

    ## Extract seasonal component ----
    trend = decomposed.trend.copy()

    ### If MultiIndex, drop the disease level ----
    if isinstance(trend.index, pd.MultiIndex):
        if "time" in trend.index.names:
            trend.index = trend.index.get_level_values("time")
        else:
            raise ValueError("Expected MultiIndex with a 'time' level.")

    ## Build tidy DataFrame ----
    df = pd.DataFrame(
        {
            "trend": trend,
            "year": trend.index.year,
            "month": trend.index.month,
        }
    )

    ## Pivot to get one line per year ----
    pivot = df.pivot(index="month", columns="year", values="trend")

    ## Replace month numbers with abbreviations ----
    pivot.index = pivot.index.map(lambda m: calendar.month_abbr[m])

    ## Plot ----
    ax = pivot.plot(
        figsize=(12, 6.5),
        title=f"Trend Component by Year — {disease_name}",
        xlabel="Time [M]",
        ylabel="Trend",
        legend=True,
    )

    return ax



# ==============================================================================
#                       FUNCTION TO ESTIMATE ARC
# ==============================================================================



def estimate_arc(ts, months):

    """
    Estimate ARC for a seasonal window defined by a set of months.
    
    The ARC (Average Rate of Change) quantifies the typical rate of change over
    a specified time interval. It is calculated from the trend line and 
    represents the rate of difference between the average occurrence of a given 
    measurement at the end of the time series and its average occurrence at the start. 
    This difference is divided by the total duration of the interval. 
    The ARC is expressed in the same units as the original measurement scale, 
    providing an interpretable rate of change.

    Parameters

    ----------
    ts : DataFrame with a 'trend' column and DatetimeIndex
    months : list of integers (1–12) defining the seasonal window
    
    Returns
    -------
    dict with ARC, initial value, end value, and time interval

    """

  # Filter to the months of interest
    seasonal_trend = ts.loc[ts.index.month.isin(months), "trend"].sort_index()

    if seasonal_trend.empty:
        return {
            "arc": np.nan,
            "initial_value": np.nan,
            "end_value": np.nan,
            "time_interval": 0
        }

    # Start and end values
    init_value = seasonal_trend.iloc[0]
    end_value  = seasonal_trend.iloc[len(seasonal_trend)-1]

    # Count unique months (not rows)
    n_months = seasonal_trend.index.to_period("M").nunique()

    # ARC per month
    arc = (end_value - init_value) / n_months

    return {
        "arc": arc,
        "initial_value": init_value,
        "end_value": end_value,
        "time_interval": n_months
    }