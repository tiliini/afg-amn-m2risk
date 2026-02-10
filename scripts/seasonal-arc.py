# ==============================================================================
#                          SEASONAL AVERAGE RATE OF CHANGE
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


import importlib
import numpy as np
import pandas as pd
import modules.trend as trend
from modules import utils

import sys
from scripts.decompose_diseases_admissions import ari, awd, measles, pneumonia
sys.path.append("python")
import matplotlib.pyplot as plt

plt.style.use("ggplot")
importlib.reload(utils)
importlib.reload(trend)


## ---- ARI  -------------------------------------------------------------------


### Decompose by province ----
dec_ari_province = utils.apply_stl_decomposition(
    data=ari,
    decompose="admission",
    index="time",
    seasonal=7,
    period=12,
    scope="multiple",
    date_format="%B %Y",
    frequency="M",
    analysis_unit="province"
)

### Pull out trend ----
ari_trend = utils.pull_component_and_concatenate(dec_ari_province, "trend")

## Prepara data ----
ari_trend = (
    ari_trend
    .assign(
        year=ari_trend.index.year,
        month=ari_trend.index.month,
        season=lambda s: np.where(
            s.index.month.isin([8, 9, 10, 11, 12, 2, 3]), "High", "Low"
        ),
    )
)

# Now assign slope WITHIN season
ari_trend["slope"] = np.where(
    ari_trend["season"] == "Low",
    "Low",   # force Low season to always be Low slope
    np.select(
        [
            ari_trend["month"].isin([8, 9, 10, 11, 12]),
            ari_trend["month"].isin([1, 2, 3])
        ],
        ["Increase", "Decrease"],
        default="Low"
    )
)

## Define Seasonal Windows ----
high_windows = {
    "Increase": [8, 9, 10, 11, 12],
    "Decrease": [1, 2, 3],
    "Low":      []   # no Low-slope months in High season
}
low_windows = {
    "Low": [4, 5, 6, 7, 1]   # all Low-season months
}

season_slope_months = {
    "High": high_windows,
    "Low":  low_windows
}

## Make groups ----
groups = ari_trend.groupby(["province", "year", "season", "slope"])

results = []

for (province, year, season, slope), group in groups:
    months = season_slope_months[season][slope]
    arc_info = trend.estimate_arc(group, months=months)

    results.append({
        "province": province,
        "year": year,
        "season": season,
        "slope": slope,
        **arc_info
    })

arc_ari = pd.DataFrame(results)

### Median ARC ----
med_arc_ari = trend.get_absolute_and_median(
    arc_ari, ["province", "season", "slope"]
)

## ---- AWD  -------------------------------------------------------------------


### Decompose by province ----
dec_awd_province = utils.apply_stl_decomposition(
    data=awd,
    decompose="admission",
    index="time",
    seasonal=7,
    period=12,
    scope="multiple",
    date_format="%B %Y",
    frequency="M",
    analysis_unit="province"
)

### Pull out trend ----
awd_trend = utils.pull_component_and_concatenate(dec_awd_province, "trend")

## Prepara data ----
awd_trend = (
    awd_trend
    .assign(
        year=awd_trend.index.year,
        month=awd_trend.index.month,
        season=lambda s: np.where(
            s.index.month.isin([12, 1, 2]), "Low","High"
            )
))

awd_trend["slope"] = np.where(awd_trend["season"] == "Low", "Low",
        np.select(
        [
            awd_trend["month"].isin([3, 4, 5, 6]),
            awd_trend["month"].isin([7, 8]),
            awd_trend["month"].isin([9, 10, 11])
        ],
        ["Increase", "Flat", "Decrease"],
        default="Low"
    )
)

## Define Seasonal Windows ----
high_windows_awd = {
    "Increase": [3, 4, 5, 6],
    "Flat":     [7, 8],
    "Decrease": [9, 10, 11],
    "Low":      []
}

low_windows_awd = {
    "Low": [1, 2, 12]
}

flat_seasons_awd = {
    "Flat": [7, 8]
}

season_slope_months_awd = {
    "High": high_windows_awd,
    "Low": low_windows_awd
}

## Make groups ----
groups = awd_trend.groupby(["province", "year", "season", "slope"])

## Apply ARC to each group ----
results = []

for (province, year, season, slope), group in groups:
    months = season_slope_months_awd[season][slope] 
    arc_info = trend.estimate_arc(group, months=months)

    results.append({
        "province": province,
        "year": year,
        "season": season,
        "slope": slope,
        **arc_info
    })

arc_awd = pd.DataFrame(results)

### Median ARC ----
med_arc_awd = trend.get_absolute_and_median(
    arc_awd, ["province", "season", "slope"]
)

## ---- Measles ----------------------------------------------------------------


### Decompose by province ----
dec_measles_province = utils.apply_stl_decomposition(
    data=measles,
    decompose="admission",
    index="time",
    seasonal=7,
    period=12,
    scope="multiple",
    date_format="%B %Y",
    frequency="M",
    analysis_unit="province"
)

### Pull out trend ----
measles_trend = utils.pull_component_and_concatenate(dec_measles_province, "trend")

## Prepara data ----
measles_trend = (
    measles_trend
    .assign(
        year=measles_trend.index.year,
        month=measles_trend.index.month,
        season=lambda s: np.where(
            s.index.month.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), "High", "Low"
            )
    )
)


measles_trend["slope"] = np.where(measles_trend["season"] == "Low", "Low",
    np.select(
        [
            measles_trend["month"].isin([1, 2, 3, 4]),
            measles_trend["month"].isin([5, 6, 7, 8, 9]),
            measles_trend["month"].isin([10, 11, 12])
        ],
        ["Increase", "Decrease", "Low"],
        default="Low"
    )
)

## Define Seasonal Windows ----
high_windows_measles = {
    "Increase": [1, 2, 3, 4],
    "Decrease": [5, 6, 7, 8, 9],
    "Low":      []
}

low_windows_measles = {
    "Low": [10, 11, 12]
}

season_slope_months_measles = {
    "High": high_windows_measles,
    "Low": low_windows_measles
}

## Make groups ----
groups = measles_trend.groupby(["province", "year", "season", "slope"])

## Apply ARC to each group ----
results = []

for (province, year, season, slope), group in groups:
    months = season_slope_months_measles[season][slope]
    arc_info = trend.estimate_arc(group, months=months)

    results.append({
        "province": province,
        "year": year,
        "season": season,
        "slope": slope,
        **arc_info
    })

arc_measles = pd.DataFrame(results)

### Median ARC ----
med_arc_measles = trend.get_absolute_and_median(
    arc_measles, ["province", "season", "slope"]
)

## ---- Pneumonia --------------------------------------------------------------


### Decompose by province ----
dec_pneumonia_province = utils.apply_stl_decomposition(
    data=pneumonia,
    decompose="admission",
    index="time",
    seasonal=7,
    period=12,
    scope="multiple",
    date_format="%B %Y",
    frequency="M",
    analysis_unit="province"
)

### Pull out trend ----
pneumonia_trend = utils.pull_component_and_concatenate(dec_pneumonia_province, "trend")

## Prepara data ----
pneumonia_trend = (
    pneumonia_trend
    .assign(
        year=pneumonia_trend.index.year,
        month=pneumonia_trend.index.month,
        season=lambda s: np.where(
            s.index.month.isin([8, 9, 10, 11, 12, 2, 3]), "High", "Low"
            )
    )
)

pneumonia_trend["slope"] = np.where(pneumonia_trend["season"] == "Low", "Low",
    np.select(
        [
            pneumonia_trend["month"].isin([8, 9, 10, 11, 12]),
            pneumonia_trend["month"].isin([1, 2, 3])
        ],

        ["Increase", "Decrease"], default="Low"
    )

)


## Define Seasonal Windows ----
high_slopes_pneumonia = {
    "Increase": [8, 9, 10, 11, 12],
    "Decrease": [1, 2, 3],
    "Low":      []
}

low_slopes_pneumonia = {"Low": [4, 5, 6, 7]}

season_slope_months_pneumonia = {
    "High": high_slopes_pneumonia,
    "Low": low_slopes_pneumonia
}

## Make groups ----
groups = pneumonia_trend.groupby(["province", "year", "season", "slope"])

## Apply ARC to each group ----
results = []

for (province, year, season, slope), group in groups:
    months = season_slope_months_pneumonia[season][slope]
    arc_info = trend.estimate_arc(group, months=months)

    results.append({
        "province": province,
        "year": year,
        "season": season,
        "slope": slope,
        **arc_info
    })

arc_pneumonia = pd.DataFrame(results)

### Median ARC ----
med_arc_pneumonia = trend.get_absolute_and_median(
    arc_pneumonia, ["province", "season", "slope"]
)
