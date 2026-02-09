def estimate_arc(ts, start="", end=""):

    """
    Calculate Seasonal Average Rate of Change
    
    The ARC (Average Rate of Change) quantifies the typical rate of change over
    a specified time interval. It is calculated from the trend line and 
    represents the rate of difference between the average occurrence of a given 
    measurement at the end of the time series and its average occurrence at the start. 
    This difference is divided by the total duration of the interval. 
    The ARC is expressed in the same units as the original measurement scale, 
    providing an interpretable rate of change.

    Parameters

    ----------
    ts : A decomposed time-series object with trend component.

    start : str
    Date a the biginning of the seasonal slope given as follows: '2021-02'.

    end : str
    Date a the end of the seasonal slope.

    Returns 
    A dictionary containing the ARC, slope-initial value, slope-end value and the 
    time interval.

    """

    ## Slice seasonal-based slope ----
    seasonal_trend = ts.trend[start:end]

    ## Pull value at beginning and end of the slope ----
    init_value = seasonal_trend.iloc[0]
    end_value = seasonal_trend.iloc[len(seasonal_trend)-1]

    ## Estimate time interval of the slope ----
    time_interval = len(seasonal_trend)

    ## Calculate ARC ----
    arc = (end_value - init_value) / time_interval

    results = {
        "arc": arc,
        "initial_value": init_value,
        "end_value": end_value,
        "time_interval": time_interval
    }

    return results

