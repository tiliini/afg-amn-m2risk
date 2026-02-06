# ==============================================================================
#                                  DISEASES
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


from os import pipe
import pandas as pd
import modules.decompose_disease as dec
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
    ts_diseases.rename(columns={"OPD morbidity": "disease", "Province ": "province"})
    .melt(id_vars=["disease", "province"], var_name="time", value_name="admission")
    .query("disease != 'HMIS-MIAR-OPD- New Patients/Clients'")
)

### Recode diseases for easy manipulation ----
ts["disease"] = ts["disease"].replace(
    {
        "HMIS-MIAR-OPD- New Acute Watery Diarrhea": "AWD",
        "HMIS-MIAR-OPD- New Cough and Cold (ARI)": "ARI",
        "HMIS-MIAR-OPD- New Measles": "Measles",
        "HMIS-MIAR-OPD- New Malaria": "Malaria",
        "HMIS-MIAR-OPD- New Pneumonia (ARI)": "New Pneumonia",
    }
)

### Split disease-specific time seris ----
ari = ts.query("disease == 'ARI'")
awd = ts.query("disease == 'AWD'")
measles = ts.query("disease == 'Measles'")
malaria = ts.query("disease == 'Malaria'")
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
    scope="single",
    date_format="%B %Y",
    frequency="M",
    analysis_unit=""
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
        disease="ARI", 
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


## ---- Malaria Decomposition --------------------------------------------------


### Make a time-series object and plot for inspection ----
plot_malaria_ts = (
    malaria
    .pipe(
        dec.summarise_disease,
        ts_index="time", date_format="%B %Y", time_period="M"
    )
    .pipe(
        dec.create_time_plot,
        start="Jan 2021", end="Dec 2024", disease="Malaria", time="M"
    )
)

### Decompose ---- 
dec_malaria = dec.apply_stl_decomposition(
    data=malaria,
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
dec_malaria.plot()

### Plot seasonal componet by year ----
dec.plot_seasonal_subseries(dec_malaria, disease_name="Malaria")


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