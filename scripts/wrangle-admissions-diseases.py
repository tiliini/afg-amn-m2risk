# ==============================================================================
#                               WRANGLE ADMISSIONS                                  
# ==============================================================================


## ---- Load required libraries ------------------------------------------------

import pandas as pd

### Read in disease admission. Extension is .xls, different engine to be used ----
ts_diseases = pd.read_excel(
    "data-raw/afg-morbidity-admission-2021-2025.xls",
    sheet_name=0,
    index_col=None,
    parse_dates=False,
    header=0
)

ts_d = (
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
ts_d["disease"] = ts_d["disease"].replace({
    "HMIS-MIAR-OPD- New Acute Watery Diarrhea": "AWD",
    "HMIS-MIAR-OPD- New Cough and Cold (ARI)": "ARI",
    "HMIS-MIAR-OPD- New Measles": "Measles",
    "HMIS-MIAR-OPD- New Malaria": "Malaria",
    "HMIS-MIAR-OPD- New Pneumonia (ARI)": "New Pneumonia"
})

### 
ts_d = (
    ts_d
    .assign(
        time=lambda x: pd.to_datetime(x.time, format="%B %Y")\
            .dt.to_period("M")
    )
    .groupby(["province", "disease", "time"], as_index=False)
    .agg({"admission": "sum"})
    .set_index(["province", "disease", "time"])
    .sort_index()
)


# ============================== End of Workflow ===============================