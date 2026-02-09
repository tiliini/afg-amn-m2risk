## ---- Load required libraries ------------------------------------------------


import pandas as pd
import calendar


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
