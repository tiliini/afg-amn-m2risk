# ==============================================================================
#         AN IMPLEMENTATION OF M2RISK FRAMEWORK ON AFGHANISTAN DATA
# ==============================================================================


## ---- Decompose admission data on acute malnutrition -------------------------


exec(open("scripts/decompose-admissions-amn.py").read())


## ---- Decompose admission data on childhood diseases -------------------------


exec(open(file="scripts/decompose_diseases_admissions.py").read())


## ---- Analyse the trend component --------------------------------------------


exec(open("scripts/seasonal_arc.py").read())
exec(open("scripts/local_average_rate_of_change.py").read())


# ============================== End of Workflow ===============================