# ==============================================================================
#                                  DISEASES
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


import importlib
import pandas as pd
from modules import seasonality as snl
from modules import utils
import sys

sys.path.append("python")
import matplotlib.pyplot as plt

plt.style.use("ggplot")
importlib.reload(utils)



## ---- Wrangle ----------------------------------------------------------------


### Read in disease admission. Extension is .xls, != engine to be used ----
data_feb25 = pd.read_excel(
    "data-raw/afg-morbidity-2021-2025-updated24Feb2025.xls",
    sheet_name=0,
    index_col=None,
    parse_dates=False,
    header=0, 
    skiprows=1
)

### Rename and exclude non-disease-related values ----
ts = (
    data_feb25
    .rename(columns={"Indicator ": "disease", "Province": "province"})
    .melt(id_vars=["disease", "province"], var_name="time", value_name="admission")
    .query("~`disease`.str.contains('Patients/Clients')")
)

### Recode diseases for easy manipulation ----
ts["disease"] = ts["disease"].replace(
    {
        "HMIS-MIAR-OPD- New Acute Watery Diarrhea <5 yrs, Female": "awd_female",
        "HMIS-MIAR-OPD- New Acute Watery Diarrhea <5 yrs, Male": "awd_male",
        "HMIS-MIAR-OPD- New Cough and Cold (ARI) <5 yrs, Female": "ari_female",
        "HMIS-MIAR-OPD- New Cough and Cold (ARI) <5 yrs, Male": "ari_male",
        "HMIS-MIAR-OPD- New Measles <5 yrs, Female": "measles_female",
        "HMIS-MIAR-OPD- New Measles <5 yrs, Male": "measles_male",
        "HMIS-MIAR-OPD- New Malaria <5 yrs, Female": "malaria_female",
        "HMIS-MIAR-OPD- New Malaria <5 yrs, Male": "malaria_male",
        "HMIS-MIAR-OPD- New Pneumonia (ARI) <5 yrs, Female": "pneumonia_female",
        "HMIS-MIAR-OPD- New Pneumonia (ARI) <5 yrs, Male": "pneumonia_male"
    }
)

### Pivot data to make values for females and males in a sep column ----
ts = (
    ts
    .pivot(index=["province", "time"], columns="disease", values="admission")
    .assign(
        ari=lambda a: a["ari_female"] + a["ari_male"],
        awd=lambda d: d["awd_female"] + d["awd_male"],
        malaria=lambda m: m["malaria_female"] + m["malaria_male"],
        measles=lambda m: m["measles_female"] + m["measles_male"],
        pneumonia=lambda p: p["pneumonia_female"] + p["pneumonia_male"]
    )
    .reset_index()
    .rename_axis(None, axis=1)
)

### Drop columns ----
ts = ts.loc[:, ~ts.columns.str.contains("female|male")]

### Check for missing values ----
utils.check_missing_values(ts)

### Exclude malaria for many missing values across months and years ----
ts = ts.drop(columns="malaria")

### Split disease-specific time seris ----
ari = (
    ts[["province", "time", "ari"]]
    .rename(columns={"ari": "admission"})
    .pipe(utils.apply_univariate_nocb_inputation)
)

awd = (
    ts[["province", "time", "awd"]]
    .rename(columns={"awd": "admission"})
    .pipe(utils.apply_univariate_nocb_inputation)
)

measles = (
    ts[["province", "time", "measles"]]
    .rename(columns={"measles": "admission"})
    .pipe(utils.apply_univariate_nocb_inputation)
)

pneumonia = (
    ts[["province", "time", "pneumonia"]]
    .rename(columns={"pneumonia": "admission"})
    .pipe(utils.apply_univariate_nocb_inputation)
)


## ---- ARI Decomposition ------------------------------------------------------


### Make a time-series object and plot ----
ari_plot_ts = (
    ari
    .pipe(utils.summarise_disease, "time", "%B %Y", "M")
    .pipe(utils.create_time_plot, "Jan 2021", "Dec 2025", "ARI", "M")
)

### Decompose ---- 
ari_dec = utils.apply_stl_decomposition(
    data=ari,
    decompose="admission",
    index="time",
    seasonal=7,
    period=12,
    scope="single",
    date_format="%B %Y",
    frequency="M",
    analysis_unit=""
)

### Plot decomposed components ----
plt.clf()
plt.rcParams["figure.figsize"] = (12, 6.5)
ari_dec.plot()

### Plot seasonal componet by year ----
snl.plot_seasonal_subseries(ari_dec, disease_name="ARI")


## ---- AWD Decomposition ------------------------------------------------------


### Make a time-series object and plot for inspection ----
awd_plot_ts = (
    awd
    .pipe(utils.summarise_disease, "time", "%B %Y", "M")
    .pipe(utils.create_time_plot, "Jan 2021", "Dec 2025", "AWD", "M")
)

### Decompose ---- 
awd_dec = utils.apply_stl_decomposition(
    data=awd,
    decompose="admission",
    index="time",
    seasonal=7,
    period=12,
    scope="single",
    date_format="%B %Y",
    frequency="M",
    analysis_unit=""
)

### Plot decomposed components ----
plt.clf()
plt.rcParams["figure.figsize"] = (12, 6.5)
awd_dec.plot()

### Plot seasonal componet by year ----
snl.plot_seasonal_subseries(awd_dec, disease_name="AWD")


## ---- Measles Decomposition --------------------------------------------------


### Make a time-series object and plot for inspection ----
measles_plot_ts = (
    measles
    .pipe(utils.summarise_disease, "time", "%B %Y", "M")
    .pipe(utils.create_time_plot, "Jan 2021", "Dec 2025", "ARI", "M")
)

### Decompose ---- 
measles_dec = utils.apply_stl_decomposition(
    data=measles,
    decompose="admission",
    index="time",
    seasonal=7,
    period=12,
    scope="single",
    date_format="%B %Y",
    frequency="M",
    analysis_unit=""
)

### Plot decomposed components ----
plt.clf()
plt.rcParams["figure.figsize"] = (12, 6.5)
measles_dec.plot()

### Plot seasonal componet by year ----
snl.plot_seasonal_subseries(measles_dec, disease_name="Measles")


## ---- Pneumonia Decomposition ------------------------------------------------


### Make a time-series object and plot for inspection ----
pneummonia_plot_ts = (
    pneumonia
    .pipe(utils.summarise_disease, "time", "%B %Y", "M")
    .pipe(utils.create_time_plot, "Jan 2021", "Dec 2025", "Pneumonia", "M")
)

### Decompose ---- 
pneumonia_dec = utils.apply_stl_decomposition(
    data=ari,
    decompose="admission",
    index="time",
    seasonal=7,
    period=12,
    scope="single",
    date_format="%B %Y",
    frequency="M",
    analysis_unit=""
)


### Plot decomposed components ----
plt.clf()
plt.rcParams["figure.figsize"] = (12, 6.5)
pneumonia_dec.plot()

### Plot seasonal componet by year ----
snl.plot_seasonal_subseries(pneumonia_dec, disease_name="Pneumonia")


# ============================== End of Workflow ===============================