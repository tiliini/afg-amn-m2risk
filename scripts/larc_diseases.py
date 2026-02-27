# ==============================================================================
#                          SEASONAL AVERAGE RATE OF CHANGE
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


import importlib
import numpy as np
import modules.trend as trend
from modules import utils

import sys
from scripts.decompose_disease_admissions import ari, awd, measles, pneumonia
sys.path.append("python")
import matplotlib.pyplot as plt

plt.style.use("ggplot")
importlib.reload(utils)
importlib.reload(trend)


## ---- ARI  -------------------------------------------------------------------


### Decompose by province ----
ari_dec_province = utils.apply_stl_decomposition(
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
ari_trend = utils.pull_component_and_concatenate(ari_dec_province, "trend")

### Prepare data ----
ari_trend = (
    ari_trend
    .assign(
        year=ari_trend.index.year,
        month=ari_trend.index.month,
        season=lambda s: np.where(
            s.index.month.isin([8, 9, 10, 11, 12, 1, 2, 3]), "High", "Low"
        ),
    )
)


### LARC: 2021 to 2024 ----
ari_larc_2021_2024 = (
    ari_trend
    .query("year != 2025")
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)

### LARC: 2025 ----
ari_larc_2025 = (
    ari_trend
    .query("year == 2025")
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)

## ---- AWD  -------------------------------------------------------------------


### Decompose by province ----
awd_dec_province = utils.apply_stl_decomposition(
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
awd_trend = utils.pull_component_and_concatenate(awd_dec_province, "trend")

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


### LARC: 2021 to 2024 ----
awd_larc_2021_2024 = (
    awd_trend
    .query("year != 2025")
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)

### LARC: 2025 ----
awd_larc_2025 = (
    awd_trend
    .query("year == 2025")
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)

## ---- Measles ----------------------------------------------------------------


### Decompose by province ----
measles_dec_province = utils.apply_stl_decomposition(
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
measles_trend = utils.pull_component_and_concatenate(measles_dec_province, "trend")

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

### LARC: 2021 to 2024 ----
measles_larc_2021_2024 = (
    measles_trend
    .query("year != 2025")
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)

### LARC: 2025 ----
measles_larc_2025 = (
    measles_trend
    .query("year == 2025")
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)

## ---- Pneumonia --------------------------------------------------------------


### Decompose by province ----
pneumonia_dec_province = utils.apply_stl_decomposition(
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
pneumonia_trend = utils.pull_component_and_concatenate(pneumonia_dec_province, "trend")

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

### LARC: 2021 to 2024 ---
pneumonia_larc_2021_2024 = (
    pneumonia_trend
    .query("year != 2025")
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)

### LARC: 2025---
pneumonia_larc_2024 = (
    pneumonia_trend
    .query("year == 2025")
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)


# ============================== End of Workflow ===============================