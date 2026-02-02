# ==============================================================================
#                                  READ DATA                                   
# ==============================================================================


## ---- Load required libraries ------------------------------------------------

import pandas as pd


## ---- Acute malnutrition admission data --------------------------------------

### Read acute malnutrition admissions ----
ts_amn = pd.read_excel(
    "data-raw/afg-amn-monthly-admission-2012-2025.xlsx",
    engine='openpyxl',
    sheet_name='Data',
    index_col=None,
    parse_dates=False,
    header=0
)

### Read in disease admission. Extension is .xls, different engine to be used ----
ts_diseases = pd.read_excel(
    "data-raw/afg-morbidity-admission-2021-2025.xls",
    sheet_name=0,
    index_col=None,
    parse_dates=False,
    header=0
)


# ============================== End of Workflow ===============================