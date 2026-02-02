# ==============================================================================
#                               WRANGLE ADMISSIONS                                  
# ==============================================================================

## ---- Load required libraries ------------------------------------------------

import pandas as pd 

## ---- Acute Malnutrition -----------------------------------------------------

### Read acute malnutrition admissions ----
ts_amn = pd.read_excel(
    "data-raw/afg-amn-monthly-admission-2012-2025.xlsx",
    engine='openpyxl',
    sheet_name='Data',
    index_col=None,
    parse_dates=False,
    header=0
)

### Select variables to include ----
column_names = ["Province", "Year", "Month", "samAdmittedTotal", "mamU5"]

### Summarise admissions by year-mont and province ----
ts_amn = (
    ts_amn[column_names]
    .rename(
        columns={
        "samAdmittedTotal": "sam",
        "mamU5": "mam",
        "Month": "month",
        "Year": "year",
        "Province": "province"
        }
    )
    .assign(
        time=lambda x: pd.PeriodIndex.from_fields(
            year=x["year"], 
            month=x["month"], 
            freq="M"
        ),
        gam=lambda g: g["sam"] + g["mam"]
    )
    .groupby(["province", "time"], as_index=False)
    .agg({"gam": "sum"})
    .set_index(["province", "time"])
    .sort_index()
)

# ============================== End of Workflow ===============================