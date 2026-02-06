# ==============================================================================
#                               ACUTE MALNUTRITION
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.seasonal import STL
import calendar
import matplotlib.pyplot as plt

plt.style.use("ggplot")


## ---- Acute Malnutrition -----------------------------------------------------


### Read acute malnutrition admissions ----
amn_admissions = pd.read_excel(
    "data-raw/afg-amn-monthly-admission-2012-2025.xlsx",
    engine="openpyxl",
    sheet_name="Data",
    index_col=None,
    parse_dates=False,
    header=0,
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
            "Province": "province",
        }
    )
    .query("year != [2012, 2025]")
    .assign(
        time=lambda x: pd.PeriodIndex.from_fields(
            year=x["year"], month=x["month"], freq="M"
        ),
        gam=lambda g: g["sam"] + g["mam"],
    )
    .groupby("time", as_index=False)
    .agg({"gam": "sum"})
    .set_index("time")
    .sort_index()
)


## ---- Inspect the time series ------------------------------------------------


### Plot a time plot ----
plot_amn_time = ts_amn.plot(
    kind="line",
    title="Global acute malnutrition admissions from Jan 2012 - Dec 2024",
    ylabel="# of cases",
    xlabel="Time [M]",
    subplots=False,
    fontsize=12,
    figsize=[12, 6.5],
    legend=False,
)

### Apply Box-Cox transformation to stabilise variance ----
ts_amn["bx_gam"], lmbda = boxcox(ts_amn.gam, lmbda=None)

### Plot Box-Cox-transformed data ----
ts_amn["bx_gam"].plot(
    kind="line",
    title="Box-Cox transformed GAM admissions",
    ylabel="# of cases (transformed)",
    xlabel="Time [M]",
    fontsize=12,
    figsize=[12, 6.5],
    legend=False,
)
plt.show()


## ---- Decompostion -----------------------------------------------------------


### Decompose using LOES ----
amn_decomposed = STL(ts_amn["bx_gam"], seasonal=7, period=12, robust=False).fit()

### Back transform Box-Cox to its original unit ----
amn_decomposed = pd.DataFrame(
    {
        "observed": inv_boxcox(amn_decomposed.observed, lmbda),
        "trend": inv_boxcox(amn_decomposed.trend, lmbda),
        "seasonal": inv_boxcox(amn_decomposed.seasonal, lmbda),
        "resid": inv_boxcox(amn_decomposed.resid, lmbda),
    }
)


## ---- Visualise results ------------------------------------------------------


### Plot decomposed components ----
plt.rcParams["figure.figsize"] = (12, 6.5)
plot_amn_components = amn_decomposed.plot(
    subplots=True, title="Decomposed components", xlabel="Time [M]"
)

### Plot seasonal componet by year ----
seasonal = amn_decomposed.seasonal.copy()

### Extract Year and Month
seasonal = pd.DataFrame(
    {
        "seasonal_effect": seasonal,
        "year": seasonal.index.year,
        "month": seasonal.index.month,
    }
).pivot(index="month", columns="year", values="seasonal_effect")

### Replace numeric index with month abbreviations ----
seasonal.index = seasonal.index.map(lambda m: calendar.month_abbr[m])

### Plot subseries ----
seasonal_plot = seasonal.plot(
    figsize=(12, 6.5),
    title="Seasonal Component by Year",
    xlabel="Time [M]",
    ylabel="Seasonal effect",
    legend=True,
)


# ============================== End of Workflow ===============================
