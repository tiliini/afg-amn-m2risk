# ==============================================================================
#                                  DISEASES
# ==============================================================================


## ---- Load required libraries ------------------------------------------------

import pandas as pd
import modules.decompose_disease as dec
import importlib
from statsmodels.tsa.seasonal import STL
import sys

sys.path.append("python")
import matplotlib.pyplot as plt

plt.style.use("ggplot")


## ---- Wrangle ----------------------------------------------------------------


### Read in disease admission. Extension is .xls, != engine to be used ----
ts_diseases = pd.read_excel(
    "data-raw/afg-morbidity-admission-2021-2025.xls",
    sheet_name=0,
    index_col=None,
    parse_dates=False,
    header=0
)

### Rename and exclude non-disease-related values and year 2025 ----
ts = (
    ts_diseases.rename(columns={"OPD morbidity": "disease", "Province ": "province"})
    .melt(id_vars=["disease", "province"], var_name="time", value_name="admission")
    .query("disease != 'HMIS-MIAR-OPD- New Patients/Clients'")
    .query("~`time`.str.contains('2025')")
)

### Recode diseases for easy manipulation ----
ts["disease"] = ts["disease"].replace(
    {
        "HMIS-MIAR-OPD- New Acute Watery Diarrhea": "AWD",
        "HMIS-MIAR-OPD- New Cough and Cold (ARI)": "ARI",
        "HMIS-MIAR-OPD- New Measles": "Measles",
        "HMIS-MIAR-OPD- New Malaria": "Malaria",
        "HMIS-MIAR-OPD- New Pneumonia (ARI)": "New Pneumonia"
    }
)


### Check for missing values ----
dec.check_missing_values(ts)

### Exclude malaria for many missing values across months and years ----
ts.query("`disease` != 'Malaria'", inplace=True)

### Check for missing values ----
dec.check_missing_values(ts)

### Apply univariate NOCB imputation for missing values ----
dfs = []
provinces = ts.province.unique()

for province in provinces:
    subset = ts.query("`province` == @province")

    diseases = ts.disease.unique()
    for disease in diseases:
        subset_disease = subset.query("`disease` == @disease")
        x = subset_disease.isnull().values.any()

        if x:
            p = subset_disease.bfill()
            dfs.append(p)
        else:
            dfs.append(subset_disease)

# Combine everything into one long DataFrame
ts = pd.concat(dfs, ignore_index=True)
del [dfs, x, disease, provinces, diseases, subset, subset_disease, p, province]

### Split disease-specific time seris ----
ari = ts.query("disease == 'ARI'")
awd = ts.query("disease == 'AWD'")
measles = ts.query("disease == 'Measles'")
pneumonia = ts.query("disease == 'New Pneumonia'")


## ---- ARI Decomposition ------------------------------------------------------


### Make a time-series object and plot ----
plot_ari_ts = (
    ari
    .pipe(
        dec.summarise_disease, 
        ts_index="time", date_format="%B %Y", time_period="M"
    )
    .pipe(
        dec.create_time_plot, 
        start="Jan 2021", end="Dec 2024", disease="ARI", time="M"
    )
)

### Decompose ---- 
dec_ari = dec.apply_stl_decomposition(
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

### Plot decomposed components ----
plt.clf()
plt.rcParams["figure.figsize"] = (12, 6.5)
dec_ari.plot()

### Plot seasonal componet by year ----
dec.plot_seasonal_subseries(dec_ari, disease_name="ARI")


## ---- AWD Decomposition ------------------------------------------------------


### Make a time-series object and plot for inspection ----
plot_awd_ts = (
    awd
    .pipe(
        dec.summarise_disease, 
        ts_index="time", 
        date_format="%B %Y", 
        time_period="M"
    )
    .pipe(
        dec.create_time_plot, 
        start="Jan 2021", 
        end="Dec 2024", 
        disease="AWD", 
        time="M"
    )
)

### Decompose ---- 
dec_awd = dec.apply_stl_decomposition(
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
dec_awd.plot()

### Plot seasonal componet by year ----
dec.plot_seasonal_subseries(dec_awd, disease_name="AWD")


## ---- Measles Decomposition --------------------------------------------------


### Make a time-series object and plot for inspection ----
plot_measles_ts = (
    measles
    .pipe(
        dec.summarise_disease,
        ts_index="time", date_format="%B %Y", time_period="M"
    )
    .pipe(
        dec.create_time_plot,
        start="Jan 2021", end="Dec 2024", disease="ARI", time="M"
    )
)

### Decompose ---- 
dec_measles = dec.apply_stl_decomposition(
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
dec_measles.plot()

### Plot seasonal componet by year ----
dec.plot_seasonal_subseries(dec_measles, disease_name="Measles")


## ---- Pneumonia Decomposition ------------------------------------------------


### Make a time-series object and plot for inspection ----
plot_pneummonia_ts = (
    pneumonia
    .pipe(
        dec.summarise_disease,
        ts_index="time", date_format="%B %Y", time_period="M"
    )
    .pipe(
        dec.create_time_plot,
        start="Jan 2021", end="Dec 2024", disease="Pneumonia", time="M"
    )
)

### Decompose ---- 
dec_pneumonia = dec.apply_stl_decomposition(
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
dec_pneumonia.plot()

### Plot seasonal componet by year ----
dec.plot_seasonal_subseries(dec_pneumonia, disease_name="Pneumonia")


# ============================== End of Workflow ===============================