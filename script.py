# ==============================================================================
#         AN IMPLEMENTATION OF M2RISK FRAMEWORK ON AFGHANISTAN DATA
# ==============================================================================


## ---- Decompose admission data on acute malnutrition -------------------------


exec(open("scripts/decompose-admissions-amn.py").read())


## ---- Decompose admission data on childhood diseases -------------------------


exec(open(file="scripts/decompose_disease_admissions_updated.py").read())


## ---- Estimate Average Rate of Change ----------------------------------------


exec(open("scripts/larc_diseases.py").read())


# ============================== End of Workflow ===============================