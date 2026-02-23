# ==============================================================================
#                          LOCAL AVERAGE RATE OF CHANGE
# ==============================================================================


## ---- Load required libraries ------------------------------------------------


import importlib
import modules.trend as trend

import sys
from scripts.seasonal_arc import (
    ari_trend, awd_trend, measles_trend, pneumonia_trend
)
sys.path.append("python")

importlib.reload(trend)


## ---- ARI  -------------------------------------------------------------------


ari_larc = (
    ari_trend
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)


## ---- AWD  -------------------------------------------------------------------


awd_larc = (
    awd_trend
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)


## ---- Measles ----------------------------------------------------------------


measles_larc = (
    measles_trend
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)


## ---- Pneumonia Decomposition ------------------------------------------------


pneumonia_larc = (
    pneumonia_trend
    .pipe(trend.estimate_local_arc, "province")
    .pipe(trend.get_min_max_larc, "province", "season", "larc")
    .round()
)


# ============================== End of Workflow ===============================