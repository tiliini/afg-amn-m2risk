# ==============================================================================
#                               WRANGLE ADMISSIONS                                  
# ==============================================================================

## ---- Load required libraries ------------------------------------------------

import pandas as pd 
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
plt.style.use("ggplot")

## ---- Acute Malnutrition -----------------------------------------------------

### Read acute malnutrition admissions ----
amn_admissions = pd.read_excel(
    "data-raw/afg-amn-monthly-admission-2012-2025.xlsx",
    engine='openpyxl',
    sheet_name='Data',
    index_col=None,
    parse_dates=False,
    header=0
)

### Select variables to include ----
column_names = ["Province", "Year", "Month", "samAdmittedTotal", "mamU5"]

### Summarise admissions by year-month ----
ts_amn = (
    amn_admissions[column_names]
    .rename(
        columns={
        "samAdmittedTotal": "sam",
        "mamU5": "mam",
        "Month": "month",
        "Year": "year",
        "Province": "province"
        }
    )
    .query("year != [2012, 2025]")
    .assign(
        time=lambda x: pd.PeriodIndex.from_fields(
            year=x["year"], 
            month=x["month"], 
            freq="M"
        ),
        gam=lambda g: g["sam"] + g["mam"]
    )
    .groupby("time", as_index=False)
    .agg({"gam": "sum"})
    .set_index("time")
    .sort_index()
)

## ---- Inspect the time series ------------------------------------------------

### Plot a time plot ----
plt.figure()
plot_amn_time = ts_amn.plot(
    kind="line",
    title="Global acute malnutrition admissions from Jan 2012 - Dec 2024",
    ylabel="# of cases",
    xlabel="Time [M]",
    subplots=False, 
    fontsize=12,
    figsize=[12, 6.5],
    legend=False
)

### Apply Box-Cox transformation to stabilise variance ----
ts_amn["bx_gam"], lmbda = boxcox(ts_amn.gam, lmbda=None)

### Plot Box-Cox-transformed data ----
plt.clf()
ts_amn["bx_gam"].plot(
    kind="line",
    title="Box-Cox transformed GAM admissions",
    ylabel="# of cases (transformed)",
    xlabel="Time [M]",
    fontsize=12,
    figsize=[12, 6.5],
    legend=False
)
plt.show()


## ---- Decompostion -----------------------------------------------------------

### Decompose using LOES ----
amn_decomposed = STL(
    ts_amn["bx_gam"],
    seasonal=7,
    period = 12,
    robust=False
).fit()


## ---- Visualise results ------------------------------------------------------

### Plot decomposed components ----
plt.clf()
plt.rcParams["figure.figsize"] = (12, 6.5)
plot_amn_components = amn_decomposed.plot()
plt.show()

# ============================== End of Workflow ===============================