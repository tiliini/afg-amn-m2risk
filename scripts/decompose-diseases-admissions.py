# ==============================================================================
#                                  DISEASES                                  
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


import pandas as pd 
import decompose_disease as dd
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

### Rename and exclude non-disease-related values ----
ts = (
    ts_diseases
    .rename(
        columns={
            "OPD morbidity": "disease",
            "Province ": "province"
            }
        )
    .melt(
        id_vars=["disease", "province"],
        var_name="time",
        value_name="admission"
        )
    .query("disease != 'HMIS-MIAR-OPD- New Patients/Clients'")
)

### Recode diseases for easy manipulation ----
ts["disease"] = ts["disease"].replace({
    "HMIS-MIAR-OPD- New Acute Watery Diarrhea": "AWD",
    "HMIS-MIAR-OPD- New Cough and Cold (ARI)": "ARI",
    "HMIS-MIAR-OPD- New Measles": "Measles",
    "HMIS-MIAR-OPD- New Malaria": "Malaria",
    "HMIS-MIAR-OPD- New Pneumonia (ARI)": "New Pneumonia"
})

### Split disease-specific time seris ----
ari = ts.query("disease == 'ARI'")
awd = ts.query("disease == 'AWD'")
measles = ts.query("disease == 'Measles'")
malaria = ts.query("disease == 'Malaria'")
pneumonia = ts.query("disease == 'New Pneumonia'")


## ---- ARI Decomposition ------------------------------------------------------


### Make a time-series data ----
ari = dd.summarise_disease(
    data=ari, ts_index="time", date_format="%B %Y", time_period="M"
)

### Inspect the time series ----

#### Plot a time plot ----
plot = dd.create_time_plot(
    data=ari,
    start="Jan 2021",
    end="Dec 2024",
    disease="ARI",
    time="M"
)
## No need for box-cox transformatio ##

### Decompose using LOES ----
decomposed_ari = STL(
    ari["admission"],
    seasonal=7,
    period = 12,
    robust=False
).fit()

### Visualise results ----

### Plot decomposed components ----
plt.clf()
plt.rcParams["figure.figsize"] = (12, 6.5)
decomposed_ari.plot()

### Plot seasonal componet by year ----
dd.plot_seasonal_subseries(decomposed_ari, disease_name="ARI")


## ---- AWD Decomposition ------------------------------------------------------


### Make a time-series data ----
awd = dd.summarise_disease(
    data=awd, ts_index="time", date_format="%B %Y", time_period="M"
)

### Inspect the time series ----

#### Plot a time plot ----
time_plot_awd = dd.create_time_plot(
    data=awd,
    start="Jan 2021",
    end="Dec 2024",
    disease="AWD",
    time="M"
)

## No transformation required ##

### Decompose using LOES ----
decomposed_awd = STL(
    awd["admission"],
    seasonal=7,
    period = 12,
    robust=False
).fit()

### Plot decomposed components ----
plt.rcParams["figure.figsize"] = (12, 6.5)
decomposed_awd.plot()

### Plot seasonal componet by year ----
dd.plot_seasonal_subseries(decomposed_awd, disease_name="AWD")


## ---- Measles Decomposition --------------------------------------------------


### Make a time-series data ----
measles = dd.summarise_disease(
    data=measles, ts_index="time", date_format="%B %Y", time_period="M"
)

### Inspect the time series ----

#### Plot a time plot ----
time_plot_measles = dd.create_time_plot(
    data=measles,
    start="Jan 2021",
    end="Dec 2024",
    disease="Measles",
    time="M"
)

## No transformation required ##

### Decompose using LOES ----
decomposed_measles = STL(
    measles["admission"],
    seasonal=7,
    period = 12,
    robust=False
).fit()

### Plot decomposed components ----
plt.rcParams["figure.figsize"] = (12, 6.5)
decomposed_measles.plot()

### Plot seasonal componet by year ----
dd.plot_seasonal_subseries(decomposed_measles, disease_name="Measles")


## ---- Measles Decomposition --------------------------------------------------


### Make a time-series data ----
malaria = dd.summarise_disease(
    data=malaria, ts_index="time", date_format="%B %Y", time_period="M"
)

### Inspect the time series ----

#### Plot a time plot ----
time_plot_malaria = dd.create_time_plot(
    data=malaria,
    start="Jan 2021",
    end="Dec 2024",
    disease="Malaria",
    time="M"
)

## No transformation required ##

### Decompose using LOES ----
decomposed_malaria = STL(
    malaria["admission"],
    seasonal=7,
    period = 12,
    robust=False
).fit()

### Plot decomposed components ----
plt.rcParams["figure.figsize"] = (12, 6.5)
decomposed_malaria.plot()

### Plot seasonal componet by year ----
dd.plot_seasonal_subseries(decomposed_malaria, disease_name="Malaria")


## ---- Measles Decomposition --------------------------------------------------


### Make a time-series data ----
pneumonia = dd.summarise_disease(
    data=pneumonia, ts_index="time", date_format="%B %Y", time_period="M"
)

### Inspect the time series ----

#### Plot a time plot ----
time_plot_pneumonia = dd.create_time_plot(
    data=pneumonia,
    start="Jan 2021",
    end="Dec 2024",
    disease="Pneumonia",
    time="M"
)

## No transformation required ##

### Decompose using LOES ----

#### Decompose ----
decomposed_pneumonia = STL(
    pneumonia["admission"],
    seasonal=7,
    period=12,
    robust=False
).fit()


### Plot decomposed components ----
plt.rcParams["figure.figsize"] = (12, 6.5)
decomposed_pneumonia.plot()

### Plot seasonal componet by year ----
dd.plot_seasonal_subseries(decomposed_pneumonia, disease_name="Pneumonia")


# ============================== End of Workflow ===============================