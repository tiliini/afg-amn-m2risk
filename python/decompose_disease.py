import pandas as pd
import calendar


def plot_seasonal_subseries(decomposed, disease_name=""):
    """
    Plot a seasonal subseries (one line per year) from an STL decomposition.
    
    Parameters
    ----------
    decomposed : STL decomposition result
        The object returned by STL(...).fit()
        Must have a MultiIndex with levels ["disease", "time"].
    
    disease_name : str
        Name of the disease to extract (optional if already sliced).
    """

    ## Extract seasonal component ----
    seasonal = decomposed.seasonal.copy()

    ### If MultiIndex, drop the disease level ----
    if isinstance(seasonal.index, pd.MultiIndex):
        if "time" in seasonal.index.names:
            seasonal.index = seasonal.index.get_level_values("time")
        else:
            raise ValueError("Expected MultiIndex with a 'time' level.")

    ## Build tidy DataFrame ----
    df = pd.DataFrame({
        "seasonal_effect": seasonal,
        "year": seasonal.index.year,
        "month": seasonal.index.month
    })

    ## Pivot to get one line per year ----
    pivot = df.pivot(
        index="month",
        columns="year",
        values="seasonal_effect"
    )

    ## Replace month numbers with abbreviations ----
    pivot.index = pivot.index.map(lambda m: calendar.month_abbr[m])

    ## Plot ----
    ax = pivot.plot(
        figsize=(12, 6.5),
        title=f"Seasonal Component by Year — {disease_name}",
        xlabel="Time [M]",
        ylabel="Seasonal effect",
        legend=True
    )

    return ax


def summarise_disease(data, ts_index, date_format="%B %Y", time_period="M"):
        """
        Summarise admissions and make a time-series dataset.
        
        Parameters
        
        ----------
        data : Admissions data to be summarised for downstream analysis
        
        ts_index : str
        A variable to be used to set DateTimeIndex.

        date_format: str
        The date format expressed in your data.

        time_period: str
        Whether monthl-, week-, day, quarter-based admissions. Defaults to "M" 
        for month. 

        """
        ts = (
        data
        .assign(
        time=lambda x: pd.to_datetime(x[ts_index], format=date_format)\
            .dt.to_period(time_period)
            )
        .query("time.dt.year != 2025")
        .groupby([ts_index], as_index=False)
        .agg({"admission": "sum"})
        .set_index([ts_index])
        .sort_index()
    )
    
        ts.index = ts.index.to_timestamp()
        ts
        return ts


def create_time_plot(data, start, end, disease="ARI", time="M"):
        """
        Plot a seasonal subseries (one line per year) from an STL decomposition.
        
        Parameters
        
        ----------
        decomposed : A time-series data returned from `summarise_disease()`.

        start: str
        The starting period of the time series: 'Month Year'

        end: str
        The end period of the time series: 'Month Year'

        disease: str
        Indicate the disease under analysis.
        
        time : str
        Whether monthl-, week-, day, quarter-based admissions. Defaults to "M" 
        for month.
        
        """
        plot = data.plot(
        kind="line",
        title=f"Admission of {disease} from {start} to {end}",
        ylabel="# of cases",
        xlabel=f"Time [{time}]",
        subplots=False,
        fontsize=12,
        figsize=[12, 6.5],
        legend=False
    )
    
        return plot