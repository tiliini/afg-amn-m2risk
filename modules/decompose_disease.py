import pandas as pd
import calendar
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.seasonal import STL


# ==============================================================================
#           FUNCTION SUMMARISE ADMISSIONS AND MAKE A TIME-SERIES OBJECT
# ==============================================================================


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
        data.assign(
            time=lambda x: pd.to_datetime(x[ts_index], format=date_format).dt.to_period(
                time_period
            )
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


# ==============================================================================
#                       FUNCTION TO MAKE A TIME PLOT
# ==============================================================================


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
        legend=False,
    )

    return plot


# ==============================================================================
#                       FUNCTION TO PLOT SEASONAL SUBSERIES
# ==============================================================================


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
    df = pd.DataFrame(
        {
            "seasonal_effect": seasonal,
            "year": seasonal.index.year,
            "month": seasonal.index.month,
        }
    )

    ## Pivot to get one line per year ----
    pivot = df.pivot(index="month", columns="year", values="seasonal_effect")

    ## Replace month numbers with abbreviations ----
    pivot.index = pivot.index.map(lambda m: calendar.month_abbr[m])

    ## Plot ----
    ax = pivot.plot(
        figsize=(12, 6.5),
        title=f"Seasonal Component by Year — {disease_name}",
        xlabel="Time [M]",
        ylabel="Seasonal effect",
        legend=True,
    )

    return ax


# ==============================================================================
#                      FUNCTION TO APPLY STL DECOMPOSITION
# ==============================================================================


def apply_stl_decomposition(
    data,
    decompose,
    seasonal,
    period,
    scope="single",
    analysis_unit="",
):
    """
    Apply STL decomposition dynamically

    Parameters
    ----------
    data: DataFrame to be decomposed, returned by `summarise_admissions()`.

    decompose: str
        A variable name holding the phenomenon to be decomposed.

    seasonal: int
        Length of the seasonal smoother. It should be >= 7.

    period: int
        Periodicity of the sequence of the phenomenon to be decomposed.

    scope: str
        The scope of the decomposition. Whether a single-area or multiple-area decomposition.
    """

    ## ---- Helper function to comply with DRY -------------------------------------

    def decompose_series(series):
        """Decompose a single time series with Box-Cox decision."""

        ### Estimate lamba and its 95% confidence intervals ----
        data["box_coxed"], lmbda, ci = boxcox(x=data[decompose], lmbda=None, alpha=0.05)

        ### Decide whether to transform ----
        ci_contains_1 = ci[0] <= 1 <= ci[1]

        if ci_contains_1:
            decomposed = STL(
                data[decompose], seasonal=seasonal, period=period, robust=False
            ).fit()

            ### Return trend ----
            return decomposed.trend

        else:
            ### Decompose Box-Cox-transformed data ----
            decomposed = STL(
                data.box_coxed, seasonal=seasonal, period=period, robust=False
            ).fit()

            ### Reverse transformation to original scale ----
            decomposed = pd.DataFrame(
                {
                    "observed": inv_boxcox(decomposed.observed, lmbda),
                    "trend": inv_boxcox(decomposed.trend, lmbda),
                    "seasonal": inv_boxcox(decomposed.seasonal, lmbda),
                    "resid": inv_boxcox(decomposed.resid, lmbda),
                }
            )

            ### Return trend ----
            return decomposed.trend

    ## ---- Single-area decomposition ------------------------------------------

    if scope == "single":
        return decompose_series(series=data[decompose])

    ## ---- Single-area decomposition ------------------------------------------

    elif scope == "multiple":
        if analysis_unit is None:
            raise ValueError("analysis_unit must be provided for multiple-area scope.")

        ### List of unique analysis units ----
        units = data.analysis_unit.unique()

        ### Initialise results container ----
        results = {}

        ### Loop over ----
        for unit in units:
            subset = data.query(f"{analysis_unit} == @unit")
            results[unit] = decompose_series(subset[decompose])

        return results

    else:
        raise ValueError("scope must be 'single' or 'multiple'")
