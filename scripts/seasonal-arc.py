# ==============================================================================
#                          SEASONAL AVERAGE RATE OF CHANGE
# ==============================================================================


## ---- Load required libraries ------------------------------------------------
import importlib
import numpy as np
import pandas as pd
import modules.arc as arc
import modules.decompose_disease as dec

import sys
from scripts.decompose_diseases_admissions import ari, awd, measles, pneumonia
sys.path.append("python")
import matplotlib.pyplot as plt

plt.style.use("ggplot")
importlib.reload(dec)

## ---- ARI  -------------------------------------------------------------------

### Decompose by province ----
dec_ari_province = dec.apply_stl_decomposition(
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
ari_trend = dec.pull_component_and_concatenate(dec_ari_province, "trend")

## Prepara data ----
ari_trend = (
    ari_trend
    .assign(
        year=ari_trend.index.year,
        month=ari_trend.index.month,
        season=lambda s: np.where(
            s.index.month.isin([8, 9, 10, 11, 12, 2, 3]), "High", "Low"
            ),
        slope = lambda s: np.select(
        [
            s.month.isin([8, 9, 10, 11, 12]),
            s.month.isin([1, 2, 3])
        ],
        ["Increase", "Decrease"],
        default="Low"
    )
    )
)

## Define Seasonal Windows ----
slope_months_ari = {
    "Increase": [8, 9, 10, 11, 12],
    "Decrease": [1, 2, 3],
    "Low":      [4, 5, 6, 7]
}

## Make groups ----
groups = ari_trend.groupby(["province", "season", "slope"])

## Apply ARC to each group ----
results = []

for (province, season, slope), group in groups:
    months = slope_months_ari[slope]
    arc_info = arc.estimate_arc(group, months=months)

    results.append({
        "province": province,
        "season": season,
        "slope": slope,
        **arc_info
    })

arc_ari = pd.DataFrame(results).iloc[:,0:4]


## ---- AWD  -------------------------------------------------------------------


### Decompose by province ----
dec_awd_province = dec.apply_stl_decomposition(
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
awd_trend = dec.pull_component_and_concatenate(dec_awd_province, "trend")

## Prepara data ----
awd_trend = (
    awd_trend
    .assign(
        year=awd_trend.index.year,
        month=awd_trend.index.month,
        season=lambda s: np.where(
            s.index.month.isin([12, 1, 2]), "Low","High"
            ),
        slope = lambda s: np.select(
        [
            s.month.isin([3, 4, 5, 6]),
            s.month.isin([7, 8]),
            s.month.isin([9, 10, 11])
        ],
        ["Increase", "Flat", "Decrease"],
        default="Low"
    )
))


## Define Seasonal Windows ----
slope_months = {
    "Increase": [3, 4, 5, 6],
    "Flat":     [7, 8],
    "Decrease": [9, 10, 11],
    "Low":      [12, 1, 2]
}

## Make groups ----
groups = awd_trend.groupby(["province", "season", "slope"])

## Apply ARC to each group ----
results = []

for (province, season, slope), group in groups:
    months = slope_months[slope] 
    arc_info = arc.estimate_arc(group, months=months)

    results.append({
        "province": province,
        "season": season,
        "slope": slope,
        **arc_info
    })

arc_awd = pd.DataFrame(results).iloc[:, 0:4]


## ---- Measles ----------------------------------------------------------------


### Decompose by province ----
dec_measles_province = dec.apply_stl_decomposition(
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
measles_trend = dec.pull_component_and_concatenate(dec_measles_province, "trend")

## Prepara data ----
measles_trend = (
    measles_trend
    .assign(
        year=measles_trend.index.year,
        month=measles_trend.index.month,
        season=lambda s: np.where(
            s.index.month.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), "High","Low"
            ), 
        slope = lambda s: np.select(
        [
            s.month.isin([1, 2, 3, 4]),
            s.month.isin([5, 6, 7, 8, 9])
        ],
        ["Increase", "Decrease"],
        default="Low"
    )
    )
)

## Define Seasonal Windows ----
slope_months = {
    "Increase": [1, 2, 3, 4],
    "Decrease": [5, 6, 7, 8, 9],
    "Low":      [10, 11, 12]
}

## Make groups ----
groups = measles_trend.groupby(["province", "season", "slope"])

## Apply ARC to each group ----
results = []

for (province, season, slope), group in groups:
    months = slope_months[slope]
    arc_info = arc.estimate_arc(group, months=months)

    results.append({
        "province": province,
        "season": season,
        "slope": slope,
        **arc_info
    })

arc_measles = pd.DataFrame(results).iloc[:,0:4]


## ---- Pneumonia --------------------------------------------------------------


### Decompose by province ----
dec_pneumonia_province = dec.apply_stl_decomposition(
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
pneumonia_trend = dec.pull_component_and_concatenate(dec_pneumonia_province, "trend")

## Prepara data ----
pneumonia_trend = (
    pneumonia_trend
    .assign(
        year=pneumonia_trend.index.year,
        month=pneumonia_trend.index.month,
        season=lambda s: np.where(
            s.index.month.isin([8, 9, 10, 11, 12, 2, 3]), "High", "Low"
            ),
        slope = lambda s: np.select(
        [
            s.month.isin([8, 9, 10, 11, 12]),
            s.month.isin([1, 2, 3])
        ],
        ["Increase", "Decrease"],
        default="Low"
    )
    )
)

## Define Seasonal Windows ----
slope_months_pneumonia = {
    "Increase": [8, 9, 10, 11, 12],
    "Decrease": [1, 2, 3],
    "Low":      [4, 5, 6, 7]
}

## Make groups ----
groups = pneumonia_trend.groupby(["province", "season", "slope"])

## Apply ARC to each group ----
results = []

for (province, season, slope), group in groups:
    months = slope_months_pneumonia[slope]
    arc_info = arc.estimate_arc(group, months=months)

    results.append({
        "province": province,
        "season": season,
        "slope": slope,
        **arc_info
    })

arc_pneumonia = pd.DataFrame(results).iloc[:,0:4]

