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

  # Slice all months matching the seasonal window (across all years)
    seasonal_trend = ts["trend"][ts.index.month.isin(months)]

    # Pull start and end values
    init_value = seasonal_trend.iloc[0]
    end_value  = seasonal_trend.iloc[-1]

    # Time interval (number of months)
    time_interval = len(seasonal_trend)

    # ARC calculation
    arc = (end_value - init_value) / time_interval

    return {
        "arc": arc,
        "initial_value": init_value,
        "end_value": end_value,
        "time_interval": time_interval
    }