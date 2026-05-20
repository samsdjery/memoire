# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:34:43 2026

@author: admin123
"""


#%% INSTALLATION PACKAGES
# pip install --upgrade statsmodels
# pip install xlrd
# pip install scipy==1.11.4
# pip install statsmodels==0.14.4 scipy==1.13.1
# pip install linearmodels 
# pip install cdsapi xarray netCDF4
# pip install netcdf4


import os
import math
import statistics
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg'
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.model_selection import KFold
from stargazer.stargazer import Stargazer
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
from linearmodels.iv import IV2SLS as IV2SLS_lm   # IV robuste (remplace sandbox IV2SLS)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from linearmodels.panel import PanelOLS
from linearmodels import RandomEffects
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from statsmodels.tsa.stattools    import adfuller, kpss, coint
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, acorr_breusch_godfrey, linear_reset)
from statsmodels.stats.stattools  import durbin_watson, jarque_bera

#%% CHEMIN DE TRAVAIL
# #############################################################################
# Définition des chemins
# ###################################################################################

baseDir = r"C:\Users\admin123\Desktop\Maitrise"

#fig_dir = os.path.join(baseDir, "Figure")
#tab_dir = os.path.join(baseDir, "Tableau")


fig_dir = os.path.join(baseDir, "Figure\PUR")
tab_dir = os.path.join(baseDir, "Tableau\PUR")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(tab_dir, exist_ok=True)
#%% CHARGEMENT DES DONNEES
# ##################################################################################
# Chargement des donnéeS
# #####################################################################################

VIIRS_CANADA_REGION = pd.read_csv(
    os.path.join(baseDir, "arcgis", "resultat", "viirs_zonal_2012_2024.csv")
)

VIIRS_CANADA_REGION_RESAMPL = pd.read_csv(
    os.path.join(baseDir, "arcgis", "resultat", "viirs_zonal_2012_2024_moll_resampl.csv")
)

VIIRS_CANADA_REGION_Filt_03 = pd.read_csv(
    os.path.join(baseDir, "arcgis", "resultat", "viirs_zonal_2012_2024_Filt_03.csv")
)

VIIRS_NUNAVUT_MINE_VILLE = pd.read_csv(
    os.path.join(baseDir, "arcgis", "resultat", "viirs_zonal_mine_2012_2024.csv")
)

#VIIRS_NUNAVUT_MINE_VILLE_RESAMPL = pd.read_csv(
#    os.path.join(baseDir, "arcgis", "resultat", "viirs_zonal_mine_2012_2024_moll_resampl.csv")
#)

VIIRS_NUNAVUT_MINE_VILLE_RESAMPL = pd.read_csv(
    os.path.join(baseDir, "arcgis", "resultat", "viirs_zonal_mine_2012_2024_moll_resampl_pur.csv")
)


#------------------- CHANGEMENT DE NOM SUM A VIIRS SUM
VIIRS_CANADA_REGION = VIIRS_CANADA_REGION.rename(columns={"SUM": "VIIRS_SUM"})

VIIRS_CANADA_REGION_Filt_03 = VIIRS_CANADA_REGION_Filt_03.rename(
    columns={"SUM": "VIIRS_SUM_F"})

VIIRS_CANADA_REGION_RESAMPL = VIIRS_CANADA_REGION_RESAMPL.rename(
    columns={"SUM": "VIIRS_SUM_RESAMPL"})

VIIRS_NUNAVUT_MINE_VILLE = VIIRS_NUNAVUT_MINE_VILLE.rename(
    columns={"SUM": "VIIRS_SUM_RESAMPL"})

VIIRS_NUNAVUT_MINE_VILLE_RESAMPL = VIIRS_NUNAVUT_MINE_VILLE_RESAMPL.rename(
    columns={"SUM": "VIIRS_SUM_RESAMPL"})



##################### Chargement des données DMSP

DMSP_CANADA_REGION = pd.read_csv(
    os.path.join(baseDir, "arcgis", "resultat", "dmsp_zonal_2000_2013_moll.csv")
)

#DMSP_NUNAVUT_MINE_VILLE = pd.read_csv(
#    os.path.join(baseDir, "arcgis", "resultat", "dmsp_zonal_mine_2000_2013_moll.csv")
#)

DMSP_NUNAVUT_MINE_VILLE = pd.read_csv(
    os.path.join(baseDir, "arcgis", "resultat", "dmsp_zonal_mine_2000_2013_moll_pur.csv")
)

#------------- CHANGEMENT DE NOM SUM A DMSP_SUM
DMSP_CANADA_REGION = DMSP_CANADA_REGION.rename(columns={"SUM": "DMSP_SUM"})

DMSP_NUNAVUT_MINE_VILLE = DMSP_NUNAVUT_MINE_VILLE.rename(
    columns={"SUM": "DMSP_SUM"}
)

####################### Chargement des données économiques

POPULATION = pd.read_excel(
    os.path.join(baseDir, "donnees_ent_stat", "DONNEES_POPULATION.xlsx")
)

EMPLOI_MINE = pd.read_excel(
    os.path.join(baseDir, "donnees_ent_stat", "EMPLOI_MINE.xlsx")
)

DONNEES_NUNAVUT = pd.read_excel(
    os.path.join(baseDir, "donnees_ent_stat", "DONNEES_NUNAVUT.xlsx")
)

DONNEES_CANADA = pd.read_excel(
    os.path.join(baseDir, "donnees_ent_stat", "PIB_CANADA.xlsx")
)


INF_REGION_CANADA = pd.read_excel(
    os.path.join(baseDir, "arcgis", "resultat", "DONNEES_REGION_CANADA.xls")
)

INF_MINE_CANADA = pd.read_excel(
    os.path.join(baseDir, "arcgis", "resultat", "MINES_INFORMATIONS.xls")
)

cover_total = pd.read_csv(
    os.path.join(baseDir, "donnees_meteo", "cloud_cover_nunavut_annuel.csv")
)

# Instrument composite (cloud_cover + part_clair + h_nuit_astro + part_nuit
# + h_obs + observabilite) — utilisé comme instrument du modèle IV via la
# variable `observabilite` (part annuelle d'heures d'observation effective).
instrument_composite = pd.read_csv(
    os.path.join(baseDir, "donnees_meteo",
                 "instrument_composite_nunavut_annuel.csv")
)


DONNEES_CANADA['Nom_Fr'] = DONNEES_CANADA['Nom_Fr'].replace(
    'Île-du-Prince-Édouard',
    'Île du Prince-Édouard'
)




#%% DOLLARS ENCHAINES 2017

####################################################################################
### MANIPULATION DES DONNÉES
#####################################################################################

################################## CONVERSION VARIABLE EN NUMERIQUE


vars_EN = [
    "PIB_EN_2017","PIB", "VAEMP", "VAEM", "VAF", "VAOA", 
    "VAEMP_EN_2017", "VAEM_EN_2017", "VAF_EN_2017", "VAOA_EN_2017","VANC",
    "SAMP", "SAM", "DI_MINIER", "EXP_MV_CM", "AMEN_CM","SAMP_EN_2017",
    "IMM_CM", "REP_CM","RMUS", "RHBUS", "RBUS", "TAUSCAD"
]


vars_EN = [
    "PIB_EN_2017","PIB", "VAEMP", "VAEM", "VAF", "VAOA", 
    "VAEMP_EN_2017", "VAEM_EN_2017", "VAF_EN_2017", "VAOA_EN_2017","VANC",
    "SAMP", "SAM", "DI_MINIER", "EXP_MV_CM", "AMEN_CM","SAMP_EN_2017",
    "IMM_CM", "REP_CM","RMUS", "RHBUS", "RBUS", "TAUSCAD"
]

for col in vars_EN:
    if col in DONNEES_NUNAVUT.columns:
        DONNEES_NUNAVUT[col] = (
            DONNEES_NUNAVUT[col]
            .astype(str)
            .str.replace("\xa0", "", regex=False)  # espace insécable
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        DONNEES_NUNAVUT[col] = pd.to_numeric(DONNEES_NUNAVUT[col], errors="coerce")
        
##############################   CALCUL IPC AJUSTE

DONNEES_NUNAVUT["IPCAJUSTE"] = (
    DONNEES_NUNAVUT["IPCN"] /
    DONNEES_NUNAVUT.loc[DONNEES_NUNAVUT["Annee"] == 2017, "IPCN"].values[0]
)


########## CONVERSION DOLLARS US EN CAD
DONNEES_NUNAVUT["PFR_CA"] = DONNEES_NUNAVUT["PFR_US"] * DONNEES_NUNAVUT["TAUSCAD"]
DONNEES_NUNAVUT["PPET_CA"] = DONNEES_NUNAVUT["PPET_US"] * DONNEES_NUNAVUT["TAUSCAD"]

DONNEES_NUNAVUT["RMBCA"] = DONNEES_NUNAVUT["RMBUS"] * DONNEES_NUNAVUT["TAUSCAD"]
DONNEES_NUNAVUT["RMCA"]  = DONNEES_NUNAVUT["RMUS"]  * DONNEES_NUNAVUT["TAUSCAD"]
DONNEES_NUNAVUT["RHBCA"] = DONNEES_NUNAVUT["RHBUS"] * DONNEES_NUNAVUT["TAUSCAD"]
DONNEES_NUNAVUT["RBCA"]  = DONNEES_NUNAVUT["RBUS"]  * DONNEES_NUNAVUT["TAUSCAD"]


##### CALCUL DE LA VALEUR DE LA PRODUCTION
DONNEES_NUNAVUT["GPMB_CAD"] = DONNEES_NUNAVUT["GPMB_O"] * DONNEES_NUNAVUT["POR_CA"]
DONNEES_NUNAVUT["GPM_CAD"]  = DONNEES_NUNAVUT["GPM_O"]  * DONNEES_NUNAVUT["POR_CA"]
DONNEES_NUNAVUT["GPHB_CAD"] = DONNEES_NUNAVUT["GPHB_O"] * DONNEES_NUNAVUT["POR_CA"]
DONNEES_NUNAVUT["IPB_CAD"]  =  DONNEES_NUNAVUT["IPB_T"] * DONNEES_NUNAVUT["PFR_CA"]

## CALCUL PRODUCTION D'OR EN OUNCE
DONNEES_NUNAVUT["GPNV_O"] = (
    DONNEES_NUNAVUT["GPMB_O"] +
    DONNEES_NUNAVUT["GPM_O"] +
    DONNEES_NUNAVUT["GPHB_O"]
)





#%% INTERPOLATION DES DONNEES DE STATISTIQUE CANADA ANNEE 2023 ET 2024
##########################################################################################
## conversion donnée pib en dollars enchainé en dollars courant de 2022 à 2024
##############################################################################################

# Filtrer période d’estimation
DONNEES_NUNAVUT_2000_2022 = DONNEES_NUNAVUT[
    (DONNEES_NUNAVUT["Annee"] >= 2000) &
    (DONNEES_NUNAVUT["Annee"] <= 2022)
]

# Liste des variables à traiter
variables = ["PIB","VAEMP", "VAEM", "VAOA", "VAF", "SAMP"]

models = {}

for var in variables:

    # Conversion numérique explicite (évite dtype object)
    Y_EN = pd.to_numeric(DONNEES_NUNAVUT_2000_2022[var], errors='coerce')
    X_EN = pd.to_numeric(DONNEES_NUNAVUT_2000_2022[f"{var}_EN_2017"], errors='coerce')

    # Supprimer les lignes avec NaN dans Y ou X
    _mask_valid = Y_EN.notna() & X_EN.notna()
    Y_EN = Y_EN[_mask_valid]
    X_EN = X_EN[_mask_valid]
    X_EN = sm.add_constant(X_EN)

    # Estimation avec covariance robuste (HC3)
    model = sm.OLS(Y_EN, X_EN).fit(cov_type='HC3')
    models[var] = model

    print(f"\n===== Résultats pour {var} =====")
    print(model.summary())


# Prédiction pour 2023-2024
mask_future = DONNEES_NUNAVUT["Annee"].isin([2023, 2024])

for var in variables:

    X_future = pd.to_numeric(
        DONNEES_NUNAVUT.loc[mask_future, f"{var}_EN_2017"], errors='coerce'
    )
    X_future = sm.add_constant(X_future)

    pred = models[var].predict(X_future)

    DONNEES_NUNAVUT.loc[mask_future, var] = pred


# =============================================================================
# TABLE STARGAZER — 6 régressions d'interpolation (dollars nominaux ~ chaînés)
# =============================================================================
# Modèle par variable :  Y (nominal, courant) = α + β · X (chaîné 2017) + ε
# Estimateur : MCO, covariance HC3 (robuste à l'hétéroscédasticité).
# Période d'estimation : 2000–2022. Prédiction utilisée pour 2023–2024.
#
# Pour obtenir une présentation compacte (1 seule ligne de pente partagée par
# les 6 colonnes), on refit chaque modèle avec un nom de colonne unifié
# `XEN2017`. Le dict `models` original (utilisé pour la prédiction ci-dessus)
# reste intact.
# =============================================================================
models_star = {}
for var in variables:
    _y_s = pd.to_numeric(DONNEES_NUNAVUT_2000_2022[var], errors='coerce')
    _x_s = pd.to_numeric(
        DONNEES_NUNAVUT_2000_2022[f"{var}_EN_2017"], errors='coerce'
    )
    _mask_s = _y_s.notna() & _x_s.notna()
    _y_s = _y_s[_mask_s]
    _x_s = _x_s[_mask_s].rename("XEN2017").to_frame()
    _x_s = sm.add_constant(_x_s)
    models_star[var] = sm.OLS(_y_s, _x_s).fit(cov_type='HC3')

stargazer_interp = Stargazer([models_star[v] for v in variables])
stargazer_interp.title(
    "Interpolation des séries de Statistique Canada pour 2023--2024 -- "
    "régressions MCO du nominal sur le chaîné 2017 (covariance HC3)"
)
stargazer_interp.custom_columns(variables, [1] * len(variables))
stargazer_interp.rename_covariates({
    "const":    "Constante",
    "XEN2017":  "$X$ (chaîné 2017)",
})
stargazer_interp.dependent_variable_name("Variable nominale (dollars courants)")
stargazer_interp.covariate_order(["const", "XEN2017"])
stargazer_interp.add_custom_notes([
    "Période d'estimation : 2000--2022. Prédiction pour 2023 et 2024.",
    "Variables (Y, X) appariées par série : "
    "PIB, VAEMP, VAEM, VAOA, VAF, SAMP.",
    "Y : dollars courants (nominal). X : dollars enchaînés de 2017.",
    "Covariance HC3 (correction de White, robuste à "
    "l'hétéroscédasticité).",
])

# Export HTML
_path_interp_html = os.path.join(
    tab_dir, "stargazer_interpolation_StatCan.html"
)
try:
    with open(_path_interp_html, "w", encoding="utf-8") as _f:
        _f.write(stargazer_interp.render_html())
    print(f"\n  ✔ Stargazer interpolation HTML exporté : {_path_interp_html}")
except Exception as _e:
    print(f"  ⚠ Export HTML interpolation échoué : {_e}")

# Export LaTeX (avec \footnotesize et label injectés)
_path_interp_tex = os.path.join(
    tab_dir, "stargazer_interpolation_StatCan.tex"
)
try:
    _latex_interp = stargazer_interp.render_latex()
    _latex_interp = _latex_interp.replace(
        r"\begin{tabular}",
        r"\footnotesize" + "\n" + r"\begin{tabular}",
    )
    _latex_interp = _latex_interp.replace(
        r"\end{table}",
        r"  \label{tab:interpolation_statcan_2023_2024}" + "\n" + r"\end{table}",
    )
    with open(_path_interp_tex, "w", encoding="utf-8") as _f:
        _f.write(_latex_interp)
    print(f"  ✔ Stargazer interpolation LaTeX exporté : {_path_interp_tex}")
except Exception as _e:
    print(f"  ⚠ Export LaTeX interpolation échoué : {_e}")


#%% MANIPULATION DES DONNEES
#########################################################################
# CALCUL DE LA VALEUR PRODUCTION ET DU REVENU  MINIÈRE NUNAVUT
################################################################################


DONNEES_NUNAVUT["PROD_MINE_CAD"] = (
    DONNEES_NUNAVUT["GPMB_CAD"] +
    DONNEES_NUNAVUT["GPM_CAD"] +
    DONNEES_NUNAVUT["GPHB_CAD"] +
    DONNEES_NUNAVUT["IPB_CAD"]
)

DONNEES_NUNAVUT["RMINE"] = (
    DONNEES_NUNAVUT["RMBCA"] +
    DONNEES_NUNAVUT["RMCA"] +
    DONNEES_NUNAVUT["RHBCA"] +
    DONNEES_NUNAVUT["RBCA"]
)

########################################################################################
##### CALCUL DOLLARS ENCHAINÉS EN 2017
######################################################################################
DONNEES_NUNAVUT = DONNEES_NUNAVUT [
     (DONNEES_NUNAVUT["Annee"] >= 2000) &
     (DONNEES_NUNAVUT["Annee"] <= 2024)
 ]

LISTE = [
    "PIB_EN","VAEMP_EN","VAEM_EN","VAF_EN","VAOA_EN",
    "SAMP_EN","SAM_EN","DI_MINIER_EN","EXP_MV_CM_EN",
    "AMEN_CM_EN","IMM_CM_EN","REP_CM_EN","RMBCA_EN",
    "RMCA_EN","RHBCA_EN","RBCA_EN","PROD_MINE_CAD_EN",
    "GPMB_CAD_EN","GPM_CAD_EN","GPHB_CAD_EN",
    "IPB_CAD_EN","POR_CA","POP","RMINE_EN"
]

# appliquer : var_EN = var / IPCAJUSTE si pas déjà fait
for var in LISTE:
    if var not in DONNEES_NUNAVUT.columns:
        DONNEES_NUNAVUT[var] = DONNEES_NUNAVUT[var.replace("_EN", "")] / DONNEES_NUNAVUT["IPCAJUSTE"]

    
LISTE_1 = [
    "PIB_EN","VAEMP_EN","VAEM_EN","VAF_EN","VAOA_EN",
    "SAMP_EN","SAM_EN","DI_MINIER_EN","PROD_MINE_CAD_EN",
    "GPMB_CAD_EN","GPM_CAD_EN","GPHB_CAD_EN",
    "IPB_CAD_EN","POR_CA","POP","RMINE_EN"
]
for var in LISTE_1:
    x = DONNEES_NUNAVUT[var]

    # Différences simples
    DONNEES_NUNAVUT[f"Diff_{var}"] = x.diff()

    # Log : gestion des zéros
    if (x <= 0).any():
        DONNEES_NUNAVUT[f"Log_{var}"] = np.log(x + 1)
        DONNEES_NUNAVUT[f"Diff_Log_{var}"] = np.log(x + 1).diff()
    else:
        DONNEES_NUNAVUT[f"Log_{var}"] = np.log(x)
        DONNEES_NUNAVUT[f"Diff_Log_{var}"] = np.log(x).diff()

    
  
#################################################################################
# FONCTION POUR CALCULER LOG DIFF ET TAUX DE CROISSANCE
#########################################################################
def compute_log_and_diff(nom_base, value_col, group_col):
    """
    Ajoute :
       - log(value_col)
       - diff(value_col) par groupe
       - diff(log) par groupe
    """
    # Remplacer zeros / valeurs négatives (comme en MATLAB)
    clean_col = nom_base[value_col].copy()
    clean_col[clean_col <= 0] = 1e-10

    # Ajouter log
    nom_base[f"Log_{value_col}"] = np.log(clean_col)

    # Trier pour un calcul correct
    nom_base = nom_base.sort_values([group_col, "Annee"])

    # Différences par groupe
    nom_base[f"Diff_{value_col}"] = nom_base.groupby(group_col)[value_col].diff()
    nom_base[f"Diff_Log_{value_col}"] = nom_base.groupby(group_col)[f"Log_{value_col}"].diff()

    return nom_base

def compute_log_and_diff_lum(nom_base, value_col, group_col):
    """
    Ajoute :
       - log(value_col)
       - diff(value_col) par groupe
       - diff(log) par groupe
    """
    # Remplacer zeros / valeurs négatives (comme en MATLAB)
    clean_col = nom_base[value_col].copy()
    clean_col[clean_col <= 0] = 1

    # Ajouter log
    nom_base[f"Log_{value_col}"] = np.log(clean_col)

    # Trier pour un calcul correct
    nom_base = nom_base.sort_values([group_col, "Annee"])

    # Différences par groupe
    nom_base[f"Diff_{value_col}"] = nom_base.groupby(group_col)[value_col].diff()
    nom_base[f"Diff_Log_{value_col}"] = nom_base.groupby(group_col)[f"Log_{value_col}"].diff()

    return nom_base


########################    CONVERSION EN LOG DIFF ET TAUX DE CROISSANCE

VIIRS_CANADA_REGION = compute_log_and_diff_lum(
    VIIRS_CANADA_REGION, value_col="VIIRS_SUM", group_col="Nom_Fr"
)


VIIRS_CANADA_REGION_Filt_03 = compute_log_and_diff_lum(
    VIIRS_CANADA_REGION_Filt_03, value_col="VIIRS_SUM_F", group_col="Nom_Fr"
)


VIIRS_CANADA_REGION_RESAMPL = compute_log_and_diff_lum(
    VIIRS_CANADA_REGION_RESAMPL, value_col="VIIRS_SUM_RESAMPL", group_col="Nom_Fr"
)

DMSP_CANADA_REGION = compute_log_and_diff_lum(
    DMSP_CANADA_REGION, value_col="DMSP_SUM", group_col="Nom_Fr"
)


VIIRS_NUNAVUT_MINE_VILLE = compute_log_and_diff_lum(
    VIIRS_NUNAVUT_MINE_VILLE, value_col="VIIRS_SUM_RESAMPL", group_col="Nom"
)


VIIRS_NUNAVUT_MINE_VILLE_RESAMPL = compute_log_and_diff_lum(
    VIIRS_NUNAVUT_MINE_VILLE_RESAMPL, 
    value_col="VIIRS_SUM_RESAMPL", 
    group_col="Nom"
)


DMSP_NUNAVUT_MINE_VILLE = compute_log_and_diff_lum(
    DMSP_NUNAVUT_MINE_VILLE, value_col="DMSP_SUM", group_col="Nom"
)



##########################################################################################
#     SELECTION DES DONNÉES DES QUATRES MINES
##############################################################################################

# Liste des mines à garder
mines_selection = [
    'Mine_Meliadine',
    'Mine_Meadowbank',
    'Mine_Hope_Bay',
    'Mine_Baffinland'
]

# Filtrer la base
DMSP_NUNAVUT_MINE = DMSP_NUNAVUT_MINE_VILLE[
    DMSP_NUNAVUT_MINE_VILLE["Nom"].isin(mines_selection)
].copy()



# Filtrer la base
VIIRS_NUNAVUT_MINE = VIIRS_NUNAVUT_MINE_VILLE[
    VIIRS_NUNAVUT_MINE_VILLE["Nom"].isin(mines_selection)
].copy()



# Filtrer la base
VIIRS_NUNAVUT_MINE_RESAMPL = VIIRS_NUNAVUT_MINE_VILLE_RESAMPL[
    VIIRS_NUNAVUT_MINE_VILLE_RESAMPL["Nom"].isin(mines_selection)
].copy()



#########################################################################
####         GESTION DES DONNÉES DU CANADA
######################################################################
TABLE_IPCAJUSTE = DONNEES_NUNAVUT[['Annee', 'IPCAJUSTE']].copy()
DONNEES_CANADA = pd.merge(
    DONNEES_CANADA,
    TABLE_IPCAJUSTE,
    on='Annee',
    how='left'
)
### CONVERSION EN DONNÉE NUMERIQUE
DONNEES_CANADA['PIB'] = pd.to_numeric(DONNEES_CANADA['PIB'], errors='coerce')
DONNEES_CANADA['VAEMP'] = pd.to_numeric(DONNEES_CANADA['VAEMP'], errors='coerce')


#### CONVERSION EN DOLLARS ENCHAINÉS
DONNEES_CANADA['PIB_EN'] = DONNEES_CANADA['PIB'] / DONNEES_CANADA['IPCAJUSTE']
DONNEES_CANADA['VAEMP_EN'] = DONNEES_CANADA['VAEMP'] / DONNEES_CANADA['IPCAJUSTE']

### TRI DES DONNÉES
DONNEES_CANADA = DONNEES_CANADA.sort_values(
    ['Nom_Fr', 'Annee']
).reset_index(drop=True)

### CALCUL LOG DIFF ET TAUX DE CROISSANCE
DONNEES_CANADA['Log_PIB_EN'] = np.log(DONNEES_CANADA['PIB_EN'])
DONNEES_CANADA['Diff_PIB_EN'] = DONNEES_CANADA.groupby('Nom_Fr')['PIB_EN'].diff()
DONNEES_CANADA['Diff_Log_PIB_EN'] = DONNEES_CANADA.groupby('Nom_Fr')['Log_PIB_EN'].diff()
DONNEES_CANADA['Log_VAEMP_EN'] = np.log(DONNEES_CANADA['VAEMP_EN'])
DONNEES_CANADA['Diff_VAEMP_EN'] = DONNEES_CANADA.groupby('Nom_Fr')['VAEMP_EN'].diff()
DONNEES_CANADA['Diff_Log_VAEMP_EN'] = DONNEES_CANADA.groupby('Nom_Fr')['Log_VAEMP_EN'].diff()

POPULATION['Log_EMPL'] = np.log(POPULATION['EMPL'])
POPULATION['Diff_EMPL'] = POPULATION['EMPL'].diff()
POPULATION['Diff_Log_EMPL'] = POPULATION['Log_EMPL'].diff()
##########################################################################################
#     SELECTION DES DONNÉES DU NUNAVUT
##############################################################################################

# DONNÉES VIIRS
VIIRS_NUNAVUT = VIIRS_CANADA_REGION[VIIRS_CANADA_REGION["Nom_Fr"] == "Nunavut"]
VIIRS_NUNAVUT_RESAMPL = VIIRS_CANADA_REGION_RESAMPL[VIIRS_CANADA_REGION_RESAMPL["Nom_Fr"] == "Nunavut"]

# DONNÉES DMSP
DMSP_NUNAVUT = DMSP_CANADA_REGION[DMSP_CANADA_REGION["Nom_Fr"] == "Nunavut"]



DONNEES_NUNAVUT_2012_2024 = DONNEES_NUNAVUT [
     (DONNEES_NUNAVUT["Annee"] >= 2012) &
     (DONNEES_NUNAVUT["Annee"] <= 2024)
 ]

DONNEES_NUNAVUT_2000_2011 = DONNEES_NUNAVUT [
     (DONNEES_NUNAVUT["Annee"] >= 2000) &
     (DONNEES_NUNAVUT["Annee"] <= 2011)
 ]

POPULATION_2009_2024 = POPULATION[(POPULATION["Annee"] >= 2009) & (POPULATION["Annee"] <= 2024)]
POPULATION_2000_2024 = POPULATION[(POPULATION["Annee"] >= 2000) & (POPULATION["Annee"] <= 2024)]
POPULATION_2005_2024 = POPULATION[(POPULATION["Annee"] >= 2005) & (POPULATION["Annee"] <= 2024)]

NUNAVUT_DM = DMSP_CANADA_REGION[DMSP_CANADA_REGION['Nom_Fr'] == 'Nunavut']  # Ajustez le nom exact si nécessaire
NUNAVUT_VI = VIIRS_CANADA_REGION_RESAMPL[VIIRS_CANADA_REGION_RESAMPL['Nom_Fr'] == 'Nunavut']  # Ajustez le nom exact si nécessaire



#%% MACHINE LEARNING CONVERSION
###############################################################################################################
                                #### Modele conversion DMSP en VIIRS CANADA
###########################################################################################################

###################################### --- Filtrer données 2013 ---
DMSP_2013 = DMSP_CANADA_REGION[(DMSP_CANADA_REGION["Annee"] >= 2013) & (DMSP_CANADA_REGION["Annee"] <= 2013)]
VIIRS_2013 = VIIRS_CANADA_REGION_RESAMPL[(VIIRS_CANADA_REGION_RESAMPL["Annee"] >= 2013) & (VIIRS_CANADA_REGION_RESAMPL["Annee"] <= 2013)]

# --- Jointure interne sur Nom_Fr ---
DATA_2013 = pd.merge(DMSP_2013[["Nom_Fr", "DMSP_SUM","Log_DMSP_SUM", "Longitude", "Latitude", "AREA"]], VIIRS_2013[["Nom_Fr","VIIRS_SUM_RESAMPL", "Log_VIIRS_SUM_RESAMPL"]], on="Nom_Fr", how="inner")

print(DATA_2013.head())

# --- Enlever NaN ou ±inf ---
mask_valide = np.isfinite(DATA_2013["DMSP_SUM"]) & np.isfinite(DATA_2013["Log_VIIRS_SUM_RESAMPL"])
DATA_2013 = DATA_2013[mask_valide].reset_index(drop=True)

#DMSP_X =  DMSP_CANADA_REGION[["DMSP_SUM", "X", "Y" ]]

DATA_2013.describe()


################################# CREATION MODEL MACHINE LEARNING

model = LinearRegression()

# Split des données
X_train, X_test, y_train, y_test = train_test_split(
    DATA_2013[["Log_DMSP_SUM", "Longitude", "Latitude"]], 
    DATA_2013[["Log_VIIRS_SUM_RESAMPL"]], 
    test_size=0.2, 
    random_state=2  # REPLICABILITÉ
)

model.fit(X_train,y_train)
# Score sur le train
train_score = model.score(X_train, y_train)
print("Train R² :", train_score)

# Score sur le test (vraie évaluation)
test_score = model.score(X_test, y_test)
print("Test  R² :", test_score)

kf = KFold(n_splits=2, shuffle=True)

cv_scores = cross_val_score(
    model,
    DATA_2013[["Log_DMSP_SUM", "Longitude", "Latitude"]],
    DATA_2013["Log_VIIRS_SUM_RESAMPL"],
    scoring='r2',
    cv=kf
)

print("Scores CV :", cv_scores)
print("Moyenne CV R² :", cv_scores.mean())

################################################## PREDICTION AVEC VERIFICATION

model.fit(DATA_2013[["Log_DMSP_SUM", "Longitude", "Latitude"]],DATA_2013[["Log_VIIRS_SUM_RESAMPL"]])
model.score(DATA_2013[["Log_DMSP_SUM", "Longitude", "Latitude"]],DATA_2013[["Log_VIIRS_SUM_RESAMPL"]])
Log_predictions = model.predict(DATA_2013[["Log_DMSP_SUM", "Longitude", "Latitude"]]).reshape(-1,1)
predictions=np.exp(Log_predictions)
print(predictions)




#%% CONVERSION DMSP 2000 - 2012 en VIIRS

########################################################################################################
################## CONVERSION DMSP 2000 - 2012 en VIIRS
########################################################################################################

#################################################### DONNEES MINES
##########  CONVERSION DMSP MINE EN VIIRS
Log_VIIRS_predictions_mine = model.predict( DMSP_NUNAVUT_MINE[["Log_DMSP_SUM", "Longitude", "Latitude"]]).reshape(-1,1)
DMSP_NUNAVUT_MINE["VIIRS_SUM_RESAMPL"] = np.exp(Log_VIIRS_predictions_mine)

######################### SELECTION ANNEE 2000 A 2011
DMSP_NUNAVUT_MINE_2000_2011 = DMSP_NUNAVUT_MINE[
    (DMSP_NUNAVUT_MINE["Annee"] >= 2000) &
    (DMSP_NUNAVUT_MINE["Annee"] <= 2011)
]

######################## COMBINER DONNEES DMSP ET VIIRS MINE
DMSP_VIIRS_MINE_2000_2024 = pd.concat([DMSP_NUNAVUT_MINE_2000_2011[['Nom', 'Annee', 'VIIRS_SUM_RESAMPL']], VIIRS_NUNAVUT_MINE_RESAMPL[['Nom', 'Annee', 'VIIRS_SUM_RESAMPL']]], ignore_index=True)

############################################## CALCULE LOG DIFF ET TAUX DE CROISSANCE
DMSP_VIIRS_MINE_2000_2024 = compute_log_and_diff(
    DMSP_VIIRS_MINE_2000_2024, value_col="VIIRS_SUM_RESAMPL", group_col="Nom"
)


######################## COMBINER DONNEES DMSP ET VIIRS MINE + VILLE (2000-2024)
##########  CONVERSION DMSP MINE_VILLE EN VIIRS (2000-2011)
Log_VIIRS_pred_mv = model.predict(
    DMSP_NUNAVUT_MINE_VILLE[["Log_DMSP_SUM", "Longitude", "Latitude"]]
).reshape(-1, 1)
DMSP_NUNAVUT_MINE_VILLE["VIIRS_SUM_RESAMPL"] = np.exp(Log_VIIRS_pred_mv)

DMSP_NUNAVUT_MINE_VILLE_2000_2011 = DMSP_NUNAVUT_MINE_VILLE[
    (DMSP_NUNAVUT_MINE_VILLE["Annee"] >= 2000) &
    (DMSP_NUNAVUT_MINE_VILLE["Annee"] <= 2011)
]

DMSP_VIIRS_MINE_VILLE_2000_2024 = pd.concat([
    DMSP_NUNAVUT_MINE_VILLE_2000_2011[['Nom', 'Annee', 'VIIRS_SUM_RESAMPL']],
    VIIRS_NUNAVUT_MINE_VILLE_RESAMPL[['Nom', 'Annee', 'VIIRS_SUM_RESAMPL']]
], ignore_index=True)

DMSP_VIIRS_MINE_VILLE_2000_2024 = compute_log_and_diff(
    DMSP_VIIRS_MINE_VILLE_2000_2024, value_col="VIIRS_SUM_RESAMPL", group_col="Nom"
)

################ SOMME ANNUELLE DES DN POUR LES MINES (mines_selection)
DN_SUM_MINE_ANNUEL = (
    DMSP_VIIRS_MINE_VILLE_2000_2024[
        DMSP_VIIRS_MINE_VILLE_2000_2024["Nom"].isin(mines_selection)
    ]
    .groupby("Annee")["VIIRS_SUM_RESAMPL"]
    .sum()
    .reset_index()
    .rename(columns={"VIIRS_SUM_RESAMPL": "VIIRS_SUM_MINE_ANNUEL"})
    .sort_values("Annee")
    .reset_index(drop=True)
)

DN_SUM_MINE_ANNUEL["Log_VIIRS_SUM_MINE_ANNUEL"] = np.log(
    DN_SUM_MINE_ANNUEL["VIIRS_SUM_MINE_ANNUEL"].clip(lower=1e-10)
)
DN_SUM_MINE_ANNUEL["Diff_Log_VIIRS_SUM_MINE_ANNUEL"] = (
    DN_SUM_MINE_ANNUEL["Log_VIIRS_SUM_MINE_ANNUEL"].diff()
)


######################################################### DONNEES NUNAVUT

##########  CONVERSION DMSP NUNAVUT EN VIIRS
Log_VIIRS_predictions = model.predict( DMSP_CANADA_REGION[["Log_DMSP_SUM", "Longitude", "Latitude" ]]).reshape(-1,1)
print(DMSP_CANADA_REGION)
DMSP_CANADA_REGION["VIIRS_SUM_RESAMPL"] = np.exp(Log_VIIRS_predictions)


####################### SELECTION DONNEE 2000 A 2011
DMSP_CANADA_REGION_2000_2011 = DMSP_CANADA_REGION[
    (DMSP_CANADA_REGION["Annee"] >= 2000) &
    (DMSP_CANADA_REGION["Annee"] <= 2011)
]

############# SELECTION DONNEES NUNAVUT
DMSP_NUNAVUT_2000_2011 = DMSP_CANADA_REGION_2000_2011[DMSP_CANADA_REGION_2000_2011["Nom_Fr"] == "Nunavut"]

# Combiner les deux DataFrames
DMSP_VIIRS_2000_2024 = pd.concat([DMSP_CANADA_REGION_2000_2011[['Nom_Fr', 'Annee', 'VIIRS_SUM_RESAMPL']], VIIRS_CANADA_REGION_RESAMPL[['Nom_Fr', 'Annee', 'VIIRS_SUM_RESAMPL']]], ignore_index=True)

print(DMSP_VIIRS_2000_2024.head())
print(DMSP_VIIRS_2000_2024.tail())


DMSP_VIIRS_2000_2024 = compute_log_and_diff(
    DMSP_VIIRS_2000_2024, value_col="VIIRS_SUM_RESAMPL", group_col="Nom_Fr"
)

########################## TRI DES DONNÉES

DMSP_VIIRS_2000_2024_NUNAVUT = DMSP_VIIRS_2000_2024[DMSP_VIIRS_2000_2024["Nom_Fr"] == "Nunavut"]
print(DMSP_VIIRS_2000_2024_NUNAVUT.head())


#%% ROBUSTESSE — CALIBRATION SANS COORDONNÉES GÉOGRAPHIQUES
################################################################################
# TEST DE ROBUSTESSE : CONVERSION DMSP → VIIRS SANS COORDONNÉES (Lat/Lon)
#
# Objectif : vérifier que les estimations de luminosité (et donc les PIB
# calculés en aval) ne dépendent pas de manière critique de l'inclusion des
# coordonnées géographiques dans la régression de calibration.
#
# Spécification :
#   - Modèle 1 (référence) : Log_VIIRS ~ Log_DMSP + Longitude + Latitude
#   - Modèle 2 (robustesse) : Log_VIIRS ~ Log_DMSP   (sans coordonnées)
#
# Même échantillon d'entraînement 2013, même split train/test (random_state=2),
# même validation croisée. On compare ensuite les estimations VIIRS pour le
# Nunavut et pour chaque mine sur 2000–2011 (période DMSP convertie).
################################################################################

print("\n" + "=" * 78)
print("  TEST DE ROBUSTESSE — CALIBRATION SANS COORDONNÉES GÉOGRAPHIQUES")
print("=" * 78)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Entraînement du modèle SANS coordonnées
# ─────────────────────────────────────────────────────────────────────────────
_FEATURES_NC = ["Log_DMSP_SUM"]   # NC = "no coordinates"

model_sans_coords = LinearRegression()

X_train_NC, X_test_NC, y_train_NC, y_test_NC = train_test_split(
    DATA_2013[_FEATURES_NC],
    DATA_2013[["Log_VIIRS_SUM_RESAMPL"]],
    test_size=0.2,
    random_state=2,   # même graine que le modèle de référence → comparaison équitable
)

model_sans_coords.fit(X_train_NC, y_train_NC)

train_score_NC = model_sans_coords.score(X_train_NC, y_train_NC)
test_score_NC  = model_sans_coords.score(X_test_NC,  y_test_NC)

print(f"  Train R² (sans coord.) : {train_score_NC:.4f}")
print(f"  Test  R² (sans coord.) : {test_score_NC:.4f}")

kf_NC = KFold(n_splits=2, shuffle=True, random_state=2)
cv_scores_NC = cross_val_score(
    model_sans_coords,
    DATA_2013[_FEATURES_NC],
    DATA_2013["Log_VIIRS_SUM_RESAMPL"],
    scoring="r2",
    cv=kf_NC,
)
print(f"  CV R² moyen (sans coord.) : {cv_scores_NC.mean():.4f}")

# Réajustement sur l'ensemble complet (cohérent avec le modèle de référence)
model_sans_coords.fit(
    DATA_2013[_FEATURES_NC],
    DATA_2013[["Log_VIIRS_SUM_RESAMPL"]],
)

# Coefficients estimés (interprétation)
_coef_NC      = float(model_sans_coords.coef_[0][0])
_intercept_NC = float(model_sans_coords.intercept_[0])
print(f"  Équation (sans coord.) : Log_VIIRS = {_intercept_NC:.4f} + {_coef_NC:.4f} × Log_DMSP")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Prédictions parallèles — Mines, Mine+Ville, Nunavut/Canada
# ─────────────────────────────────────────────────────────────────────────────
# /!\ On stocke les prédictions dans des colonnes suffixées "_SANS_COORDS"
#     pour ne PAS écraser celles du modèle de référence.

# --- Mines (DMSP_NUNAVUT_MINE) ---
_log_pred_mine_NC = model_sans_coords.predict(
    DMSP_NUNAVUT_MINE[_FEATURES_NC]
).reshape(-1, 1)
DMSP_NUNAVUT_MINE["VIIRS_SUM_RESAMPL_SANS_COORDS"] = np.exp(_log_pred_mine_NC)

# --- Mines + Villes (DMSP_NUNAVUT_MINE_VILLE) ---
_log_pred_mv_NC = model_sans_coords.predict(
    DMSP_NUNAVUT_MINE_VILLE[_FEATURES_NC]
).reshape(-1, 1)
DMSP_NUNAVUT_MINE_VILLE["VIIRS_SUM_RESAMPL_SANS_COORDS"] = np.exp(_log_pred_mv_NC)

# --- Régions canadiennes (incl. Nunavut) ---
_log_pred_can_NC = model_sans_coords.predict(
    DMSP_CANADA_REGION[_FEATURES_NC]
).reshape(-1, 1)
DMSP_CANADA_REGION["VIIRS_SUM_RESAMPL_SANS_COORDS"] = np.exp(_log_pred_can_NC)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Tableau comparatif — Métriques d'ajustement
# ─────────────────────────────────────────────────────────────────────────────
_metrics_comp = pd.DataFrame({
    "Métrique": ["R² entraînement", "R² test", "R² CV moyen",
                 "Coef. Log_DMSP", "Coef. Longitude", "Coef. Latitude",
                 "Constante (intercept)"],
    "Avec coordonnées": [
        round(train_score, 4),
        round(test_score, 4),
        round(cv_scores.mean(), 4),
        round(float(model.coef_[0][0]), 4),
        round(float(model.coef_[0][1]), 4),
        round(float(model.coef_[0][2]), 4),
        round(float(model.intercept_[0]), 4),
    ],
    "Sans coordonnées": [
        round(train_score_NC, 4),
        round(test_score_NC, 4),
        round(cv_scores_NC.mean(), 4),
        round(_coef_NC, 4),
        np.nan,
        np.nan,
        round(_intercept_NC, 4),
    ],
})
_metrics_comp["Écart absolu"] = (
    _metrics_comp["Sans coordonnées"] - _metrics_comp["Avec coordonnées"]
)

print("\n  ----- Comparaison des métriques de calibration -----")
print(_metrics_comp.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Tableau comparatif — Estimations VIIRS pour les mines (période 2000–2011)
# ─────────────────────────────────────────────────────────────────────────────
_mask_mine_period = (
    (DMSP_NUNAVUT_MINE["Annee"] >= 2000)
    & (DMSP_NUNAVUT_MINE["Annee"] <= 2011)
)

_estim_mines_comp = (
    DMSP_NUNAVUT_MINE.loc[_mask_mine_period,
        ["Nom", "Annee",
         "VIIRS_SUM_RESAMPL",            # avec coordonnées
         "VIIRS_SUM_RESAMPL_SANS_COORDS"]
    ]
    .rename(columns={
        "VIIRS_SUM_RESAMPL":             "VIIRS_AVEC_COORDS",
        "VIIRS_SUM_RESAMPL_SANS_COORDS": "VIIRS_SANS_COORDS",
    })
    .sort_values(["Nom", "Annee"])
    .reset_index(drop=True)
)

_estim_mines_comp["Ecart_absolu"] = (
    _estim_mines_comp["VIIRS_SANS_COORDS"]
    - _estim_mines_comp["VIIRS_AVEC_COORDS"]
)
_estim_mines_comp["Ecart_pct"] = (
    100.0 * _estim_mines_comp["Ecart_absolu"]
    / _estim_mines_comp["VIIRS_AVEC_COORDS"].replace(0, np.nan)
)

# Synthèse par mine
_synthese_mines = (
    _estim_mines_comp
    .groupby("Nom")
    .agg(
        VIIRS_AVEC_COORDS_moy=("VIIRS_AVEC_COORDS", "mean"),
        VIIRS_SANS_COORDS_moy=("VIIRS_SANS_COORDS", "mean"),
        Ecart_pct_moy=("Ecart_pct", "mean"),
        Ecart_pct_abs_moy=("Ecart_pct", lambda s: s.abs().mean()),
    )
    .reset_index()
)

print("\n  ----- Estimations VIIRS par mine (moyennes 2000–2011) -----")
print(_synthese_mines.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Tableau comparatif — Estimation VIIRS pour le Nunavut (2000–2011)
# ─────────────────────────────────────────────────────────────────────────────
_mask_nun_period = (
    (DMSP_CANADA_REGION["Nom_Fr"] == "Nunavut")
    & (DMSP_CANADA_REGION["Annee"] >= 2000)
    & (DMSP_CANADA_REGION["Annee"] <= 2011)
)
_estim_nun_comp = (
    DMSP_CANADA_REGION.loc[_mask_nun_period,
        ["Annee", "VIIRS_SUM_RESAMPL", "VIIRS_SUM_RESAMPL_SANS_COORDS"]
    ]
    .rename(columns={
        "VIIRS_SUM_RESAMPL":             "VIIRS_AVEC_COORDS",
        "VIIRS_SUM_RESAMPL_SANS_COORDS": "VIIRS_SANS_COORDS",
    })
    .sort_values("Annee")
    .reset_index(drop=True)
)
_estim_nun_comp["Ecart_absolu"] = (
    _estim_nun_comp["VIIRS_SANS_COORDS"] - _estim_nun_comp["VIIRS_AVEC_COORDS"]
)
_estim_nun_comp["Ecart_pct"] = (
    100.0 * _estim_nun_comp["Ecart_absolu"]
    / _estim_nun_comp["VIIRS_AVEC_COORDS"].replace(0, np.nan)
)
print("\n  ----- Estimation VIIRS Nunavut — Avec vs Sans coordonnées -----")
print(_estim_nun_comp.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Export Excel (3 feuilles : métriques, mines, Nunavut)
# ─────────────────────────────────────────────────────────────────────────────
_xl_robust_path = os.path.join(
    tab_dir, "tableau_robustesse_calibration_sans_coords.xlsx"
)
try:
    with pd.ExcelWriter(_xl_robust_path, engine="openpyxl") as _writer:
        _metrics_comp.to_excel(_writer, sheet_name="metriques", index=False)
        _estim_mines_comp.to_excel(_writer, sheet_name="estim_mines", index=False)
        _synthese_mines.to_excel(_writer, sheet_name="synthese_mines", index=False)
        _estim_nun_comp.to_excel(_writer, sheet_name="estim_Nunavut", index=False)
    print(f"\n  ✔ Tableau Excel sauvegardé : {_xl_robust_path}")
except Exception as _e:
    print(f"  ⚠ Export Excel échoué : {_e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6 bis. Table Stargazer — Coefficients comparés des deux calibrations
#        Refit en statsmodels OLS pour récupérer std. errors, t-stats, p-values.
#        Les coefficients sont identiques à ceux de sklearn (mêmes données).
# ─────────────────────────────────────────────────────────────────────────────
# (i) Données partagées
_y_calib = DATA_2013["Log_VIIRS_SUM_RESAMPL"].astype(float)

# (ii) Modèle 1 — AVEC coordonnées (Log_DMSP + Longitude + Latitude)
_X_avec = sm.add_constant(
    DATA_2013[["Log_DMSP_SUM", "Longitude", "Latitude"]].astype(float),
    has_constant="add",
)
MODEL_CALIB_AVEC = sm.OLS(_y_calib, _X_avec).fit()

# (iii) Modèle 2 — SANS coordonnées (Log_DMSP seul)
_X_sans = sm.add_constant(
    DATA_2013[["Log_DMSP_SUM"]].astype(float),
    has_constant="add",
)
MODEL_CALIB_SANS = sm.OLS(_y_calib, _X_sans).fit()

# (iv) Construction Stargazer
stargazer_calib = Stargazer([MODEL_CALIB_AVEC, MODEL_CALIB_SANS])
stargazer_calib.title(
    "Calibration DMSP $\\rightarrow$ VIIRS --- Comparaison avec et sans "
    "coordonnées géographiques (régressions OLS, données 2013)"
)
stargazer_calib.custom_columns(
    ["Avec coordonnées", "Sans coordonnées"], [1, 1]
)
stargazer_calib.rename_covariates({
    "const":         "Constante",
    "Log_DMSP_SUM":  "Log(DMSP)",
    "Longitude":     "Longitude",
    "Latitude":      "Latitude",
})
stargazer_calib.dependent_variable_name("Log(VIIRS rééchantillonné)")
stargazer_calib.covariate_order([
    "const", "Log_DMSP_SUM", "Longitude", "Latitude",
])
stargazer_calib.show_model_numbers(True)
stargazer_calib.significant_digits(4)
stargazer_calib.add_custom_notes([
    "Échantillon : 13 régions canadiennes (DATA\\_2013), année 2013.",
    "Variables géographiques en degrés décimaux (Lon/Lat).",
    "Coefficients identiques à la version sklearn LinearRegression.",
])

# (v) Export HTML
_path_star_html = os.path.join(
    tab_dir, "stargazer_calibration_avec_vs_sans_coords.html"
)
try:
    with open(_path_star_html, "w", encoding="utf-8") as _f:
        _f.write(stargazer_calib.render_html())
    print(f"  ✔ Stargazer HTML exporté : {_path_star_html}")
except Exception as _e:
    print(f"  ⚠ Export HTML stargazer échoué : {_e}")

# (vi) Export LaTeX (avec \footnotesize et label injectés)
_path_star_tex = os.path.join(
    tab_dir, "stargazer_calibration_avec_vs_sans_coords.tex"
)
try:
    _latex_calib = stargazer_calib.render_latex()
    _latex_calib = _latex_calib.replace(
        r"\begin{tabular}",
        r"\footnotesize" + "\n" + r"\begin{tabular}",
    )
    _latex_calib = _latex_calib.replace(
        r"\end{table}",
        r"  \label{tab:calibration_avec_vs_sans_coords}" + "\n" + r"\end{table}",
    )
    with open(_path_star_tex, "w", encoding="utf-8") as _f:
        _f.write(_latex_calib)
    print(f"  ✔ Stargazer LaTeX exporté : {_path_star_tex}")
except Exception as _e:
    print(f"  ⚠ Export LaTeX stargazer échoué : {_e}")

# (vii) Affichage console des coefficients pour vérification rapide
print("\n  ----- Coefficients comparés (statsmodels OLS) -----")
print("  Modèle AVEC coordonnées :")
print(MODEL_CALIB_AVEC.summary().tables[1])
print("\n  Modèle SANS coordonnées :")
print(MODEL_CALIB_SANS.summary().tables[1])
print(f"\n  R² ajusté : avec coord. = {MODEL_CALIB_AVEC.rsquared_adj:.4f}  |  "
      f"sans coord. = {MODEL_CALIB_SANS.rsquared_adj:.4f}")
print(f"  AIC       : avec coord. = {MODEL_CALIB_AVEC.aic:,.2f}  |  "
      f"sans coord. = {MODEL_CALIB_SANS.aic:,.2f}")
print(f"  BIC       : avec coord. = {MODEL_CALIB_AVEC.bic:,.2f}  |  "
      f"sans coord. = {MODEL_CALIB_SANS.bic:,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Figure comparative — Séries VIIRS Nunavut (avec vs sans coordonnées)
# ─────────────────────────────────────────────────────────────────────────────
fig_rc, ax_rc = plt.subplots(figsize=(10, 5.5))

ax_rc.plot(
    _estim_nun_comp["Annee"], _estim_nun_comp["VIIRS_AVEC_COORDS"],
    color="#2C5F8A", linewidth=2.0, marker="o", markersize=5,
    label="Avec coordonnées (Log_DMSP + Lon + Lat)",
)
ax_rc.plot(
    _estim_nun_comp["Annee"], _estim_nun_comp["VIIRS_SANS_COORDS"],
    color="#C0392B", linewidth=2.0, marker="s", markersize=5,
    linestyle="--",
    label="Sans coordonnées (Log_DMSP seul)",
)

ax_rc.set_xlabel("Année", fontsize=11)
ax_rc.set_ylabel(
    r"VIIRS estimé — Nunavut (nW$\cdot$cm$^{-2}\cdot$sr$^{-1}$)", fontsize=11
)
ax_rc.set_title(
    "Robustesse de la calibration DMSP → VIIRS — Nunavut (2000–2011)",
    fontsize=12, fontweight="bold",
)
ax_rc.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax_rc.legend(fontsize=10, framealpha=0.92, edgecolor="#CCCCCC")
ax_rc.spines["top"].set_visible(False)
ax_rc.spines["right"].set_visible(False)
ax_rc.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC")
ax_rc.set_axisbelow(True)

plt.tight_layout()
_fp_robust = os.path.join(
    fig_dir, "robustesse_calibration_sans_coords_Nunavut.png"
)
plt.savefig(_fp_robust, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
print(f"  ✔ Figure sauvegardée : {_fp_robust}")

# Figure 2×2 — Comparaison par mine
_mines_rc = ["Mine_Meadowbank", "Mine_Meliadine", "Mine_Hope_Bay", "Mine_Baffinland"]
fig_rcm, axes_rcm = plt.subplots(2, 2, figsize=(13, 8))
for ax_m, _mine in zip(axes_rcm.flatten(), _mines_rc):
    _sub = _estim_mines_comp[_estim_mines_comp["Nom"] == _mine]
    if _sub.empty:
        ax_m.set_visible(False)
        continue
    ax_m.plot(_sub["Annee"], _sub["VIIRS_AVEC_COORDS"],
              color="#2C5F8A", marker="o", lw=1.8, label="Avec coord.")
    ax_m.plot(_sub["Annee"], _sub["VIIRS_SANS_COORDS"],
              color="#C0392B", marker="s", lw=1.8, ls="--", label="Sans coord.")
    ax_m.set_title(_mine.replace("_", " "), fontsize=10, fontweight="bold")
    ax_m.set_xlabel("Année", fontsize=9)
    ax_m.set_ylabel("VIIRS estimé", fontsize=9)
    ax_m.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_m.legend(fontsize=8, framealpha=0.9)
    ax_m.spines["top"].set_visible(False)
    ax_m.spines["right"].set_visible(False)
    ax_m.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC")
    ax_m.set_axisbelow(True)
plt.tight_layout()
_fp_robust_mines = os.path.join(
    fig_dir, "robustesse_calibration_sans_coords_mines.png"
)
plt.savefig(_fp_robust_mines, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
print(f"  ✔ Figure mines sauvegardée : {_fp_robust_mines}")

print("\n  ✔ Test de robustesse (sans coordonnées) terminé.")
print("=" * 78 + "\n")


#%% GRAPHIQUES — VALIDATION DE LA CONVERSION DMSP → VIIRS
################################################################################
# GRAPHIQUES — VALIDATION DE LA CONVERSION DMSP → VIIRS
#
#  G1 : Nuage de points Log(DMSP) ~ Log(VIIRS) — données 2013, train vs test
#  G2 : VIIRS observé vs VIIRS prédit (échelle originale) — courbe 45°
#  G3 : Série temporelle Nunavut — DMSP→VIIRS estimé (2000–2011)
#                                  + VIIRS observé (2012–2024)
#  G4 : Série temporelle par mine (2 × 2 sous-graphiques)
################################################################################

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        12,                    # ↑ 10 → 12
    'axes.titlesize':   13,                    # ↑ 11 → 13
    'axes.labelsize':   12,                    # ↑ 10 → 12
    'xtick.labelsize':  11,                    # ↑ 9  → 11
    'ytick.labelsize':  11,                    # ↑ 9  → 11
    'axes.linewidth':   0.8,
    'axes.edgecolor':   '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'grid.color':       '#CCCCCC',
    'grid.linewidth':   0.6,
})

_C_TRAIN = '#2C5F8A'   # bleu acier  — entraînement
_C_TEST  = '#C0392B'   # rouge brique — test
_C_DMSP  = '#8A4A2C'   # brun rouille — DMSP estimé
_C_VIIRS = '#2C7A4B'   # vert sauge  — VIIRS observé
_C_FIT   = '#222222'   # gris foncé  — droite ajustée

# ─────────────────────────────────────────────────────────────────────────────
# G1 : Nuage de points Log(DMSP) ~ Log(VIIRS) — entraînement vs test
# ─────────────────────────────────────────────────────────────────────────────
_idx_train = X_train.index
_idx_test  = X_test.index

# Droite ajustée : varier Log_DMSP sur toute la plage, coordonnées moyennes
_x_range_g1 = np.linspace(
    DATA_2013["Log_DMSP_SUM"].min(),
    DATA_2013["Log_DMSP_SUM"].max(), 200
)
_X_fit_df = pd.DataFrame({
    "Log_DMSP_SUM": _x_range_g1,
    "Longitude": np.full(200, DATA_2013["Longitude"].mean()),
    "Latitude": np.full(200, DATA_2013["Latitude"].mean()),
})
_y_fit_g1 = model.predict(_X_fit_df).flatten()

fig_g1, ax_g1 = plt.subplots(figsize=(7.5, 6))

ax_g1.scatter(
    DATA_2013.loc[_idx_train, "Log_DMSP_SUM"],
    DATA_2013.loc[_idx_train, "Log_VIIRS_SUM_RESAMPL"],
    color=_C_TRAIN, alpha=0.70, s=32, zorder=3,
    label=f"Entraînement (n = {len(_idx_train)})"
)
ax_g1.scatter(
    DATA_2013.loc[_idx_test, "Log_DMSP_SUM"],
    DATA_2013.loc[_idx_test, "Log_VIIRS_SUM_RESAMPL"],
    color=_C_TEST, alpha=0.85, s=45, marker='^', zorder=4,
    label=f"Test (n = {len(_idx_test)})"
)
ax_g1.plot(
    _x_range_g1, _y_fit_g1,
    color=_C_FIT, linewidth=1.7, linestyle='-', zorder=2,
    label='Droite ajustée'
)

_annot_g1 = (
    f"$R^2$ entraîn. = {train_score:.4f}\n"
    f"$R^2$ test      = {test_score:.4f}\n"
    f"$R^2$ CV moy.  = {cv_scores.mean():.4f}"
)
ax_g1.text(
    0.04, 0.97, _annot_g1,
    transform=ax_g1.transAxes, va='top', fontsize=10,    # ↑ 8.5 → 10
    fontfamily='serif',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
              edgecolor='#CCCCCC', linewidth=0.8, alpha=0.92)
)

ax_g1.set_xlabel("Log(DMSP — Digital Number)", fontsize=12)      # ↑ explicite
ax_g1.set_ylabel(r"Log(VIIRS — nW$\cdot$cm$^{-2}\cdot$sr$^{-1}$, rééchantillonné)",
                 fontsize=12)                                      # ↑ explicite
#ax_g1.set_title(
#    "Relation Log(DMSP) ~ Log(VIIRS) — données 2013\n"
#    "Modèle de conversion : régression linéaire (Log_DMSP, X, Y)",
#    fontsize=13, fontweight='bold', pad=8, fontfamily='serif'      # ↑ 11 → 13
#)
ax_g1.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC')   # ↑ 9 → 10
ax_g1.spines['top'].set_visible(False)
ax_g1.spines['right'].set_visible(False)
ax_g1.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC', zorder=0)
ax_g1.xaxis.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC', zorder=0)
ax_g1.set_axisbelow(True)

plt.tight_layout()
_fp_g1 = os.path.join(fig_dir, "conversion_DMSP_VIIRS_scatter_log.png")
plt.savefig(_fp_g1, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ G1 sauvegardé : {_fp_g1}")
# ─────────────────────────────────────────────────────────────────────────────
# G2 : VIIRS observé vs VIIRS prédit (échelle originale) — droite 1:1
# ─────────────────────────────────────────────────────────────────────────────
_y_obs_g2  = DATA_2013["VIIRS_SUM_RESAMPL"].values.astype(float)
_y_pred_g2 = predictions.flatten()

_rmse_g2 = np.sqrt(np.mean((_y_obs_g2 - _y_pred_g2) ** 2))
_ss_res   = np.sum((_y_obs_g2 - _y_pred_g2) ** 2)
_ss_tot   = np.sum((_y_obs_g2 - _y_obs_g2.mean()) ** 2)
_r2_g2    = 1 - _ss_res / _ss_tot if _ss_tot > 0 else np.nan

_bisect_g2 = np.linspace(
    min(_y_obs_g2.min(), _y_pred_g2.min()),
    max(_y_obs_g2.max(), _y_pred_g2.max()), 200
)

fig_g2, ax_g2 = plt.subplots(figsize=(6.5, 6))

ax_g2.scatter(
    _y_obs_g2, _y_pred_g2,
    color=_C_TRAIN, alpha=0.70, s=32, zorder=3,
    label='Régions canadiennes (2013)'
)
ax_g2.plot(
    _bisect_g2, _bisect_g2,
    color='#555555', linewidth=1.5, linestyle='--', zorder=2,
    label='Droite 1:1 (prédiction parfaite)'
)

_annot_g2 = f"$R^2$ = {_r2_g2:.4f}\nRMSE = {_rmse_g2:,.2f}"
ax_g2.text(
    0.04, 0.97, _annot_g2,
    transform=ax_g2.transAxes, va='top', fontsize=10,           # ↑ 11 → 10 (cohérent avec G1)
    fontfamily='serif',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
              edgecolor='#CCCCCC', linewidth=0.8, alpha=0.92)
)

ax_g2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax_g2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax_g2.set_xlabel(r"VIIRS observé (nW$\cdot$cm$^{-2}\cdot$sr$^{-1}$)", fontsize=12)   # ↑ 10 → 12
ax_g2.set_ylabel(
    r"VIIRS prédit par conversion DMSP (nW$\cdot$cm$^{-2}\cdot$sr$^{-1}$)",
    fontsize=12                                                                        # ↑ 10 → 12
)
#ax_g2.set_title(
#    "Valeurs observées vs valeurs prédites — VIIRS 2013\n"
#    "Validation du modèle de conversion DMSP → VIIRS",
#    fontsize=13, fontweight='bold', pad=8, fontfamily='serif'                          # ↑ 11 → 13
#)
ax_g2.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC')                        # ↑ 11 → 10
ax_g2.spines['top'].set_visible(False)
ax_g2.spines['right'].set_visible(False)
ax_g2.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC', zorder=0)
ax_g2.xaxis.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC', zorder=0)
ax_g2.set_axisbelow(True)

plt.tight_layout()
_fp_g2 = os.path.join(fig_dir, "conversion_DMSP_VIIRS_obs_vs_pred.png")
plt.savefig(_fp_g2, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ G2 sauvegardé : {_fp_g2}")
# ─────────────────────────────────────────────────────────────────────────────
# G3 : Série temporelle Nunavut — DMSP→VIIRS estimé (2000–2011)
#                                 + VIIRS observé (2012–2024)
# ─────────────────────────────────────────────────────────────────────────────
_dmsp_nun = (
    DMSP_NUNAVUT_2000_2011[['Annee', 'VIIRS_SUM_RESAMPL']]
    .dropna(subset=['VIIRS_SUM_RESAMPL'])
    .sort_values('Annee')
    .reset_index(drop=True)
)
_viirs_nun = (
    VIIRS_NUNAVUT_RESAMPL[['Annee', 'VIIRS_SUM_RESAMPL']]
    .dropna(subset=['VIIRS_SUM_RESAMPL'])
    .sort_values('Annee')
    .reset_index(drop=True)
)

fig_g3, ax_g3 = plt.subplots(figsize=(13, 5.5))

ax_g3.plot(
    _dmsp_nun['Annee'], _dmsp_nun['VIIRS_SUM_RESAMPL'],
    color=_C_DMSP, linewidth=2.0,
    marker='s', markersize=5.5, markeredgecolor='white', markeredgewidth=0.5,
    linestyle='--', zorder=3,
    label='VIIRS estimé depuis DMSP (2000–2011)'
)
ax_g3.fill_between(
    _dmsp_nun['Annee'], _dmsp_nun['VIIRS_SUM_RESAMPL'],
    alpha=0.10, color=_C_DMSP
)

ax_g3.plot(
    _viirs_nun['Annee'], _viirs_nun['VIIRS_SUM_RESAMPL'],
    color=_C_VIIRS, linewidth=2.0,
    marker='o', markersize=5.5, markeredgecolor='white', markeredgewidth=0.5,
    linestyle='-', zorder=3,
    label='VIIRS observé (2012–2024)'
)
ax_g3.fill_between(
    _viirs_nun['Annee'], _viirs_nun['VIIRS_SUM_RESAMPL'],
    alpha=0.10, color=_C_VIIRS
)

# Zone de chevauchement DMSP / VIIRS (2012–2013)
ax_g3.axvspan(2011.55, 2013.45, alpha=0.07, color='#888888', zorder=1)
_y_max_g3 = max(
    _dmsp_nun['VIIRS_SUM_RESAMPL'].max() if not _dmsp_nun.empty else 0,
    _viirs_nun['VIIRS_SUM_RESAMPL'].max() if not _viirs_nun.empty else 0,
)
ax_g3.text(
    2012.5, _y_max_g3 * 0.97,
    'Chevauchement\nDMSP–VIIRS',
    ha='center', va='top', fontsize=10, fontstyle='italic',                # ↑ inchangé (11→10)
    color='#555555', fontfamily='serif'
)

_all_ann_g3 = sorted(
    set(_dmsp_nun['Annee'].astype(int).tolist()) |
    set(_viirs_nun['Annee'].astype(int).tolist())
)
ax_g3.set_xticks(_all_ann_g3)
ax_g3.set_xticklabels([str(a) for a in _all_ann_g3], rotation=50, ha='right')
ax_g3.set_xlim(_all_ann_g3[0] - 0.6, _all_ann_g3[-1] + 0.6)
ax_g3.set_ylim(bottom=0)

ax_g3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax_g3.set_xlabel('Année', fontsize=12, labelpad=5)                        # ↑ 10 → 12
ax_g3.set_ylabel(
    r'Luminosité VIIRS rééchantillonnée (nW$\cdot$cm$^{-2}\cdot$sr$^{-1}$)',
    fontsize=12, labelpad=5                                                # ↑ 10 → 12
)
#ax_g3.set_title(
#    "Série temporelle de luminosité nocturne — Nunavut (2000–2024)\n"
#    "DMSP converti en VIIRS (2000–2011) et VIIRS observé (2012–2024)",
#    fontsize=13, fontweight='bold', pad=8, fontfamily='serif'              # ↑ 11 → 13
#)
ax_g3.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC')           # ↑ 11 → 10
ax_g3.spines['top'].set_visible(False)
ax_g3.spines['right'].set_visible(False)
ax_g3.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC', zorder=0)
ax_g3.set_axisbelow(True)

plt.tight_layout()
_fp_g3 = os.path.join(fig_dir, "conversion_serie_temporelle_Nunavut.png")
plt.savefig(_fp_g3, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ G3 sauvegardé : {_fp_g3}")
# ─────────────────────────────────────────────────────────────────────────────
# G4 : Série temporelle par mine (2 × 2 sous-graphiques)
# ─────────────────────────────────────────────────────────────────────────────
_mines_g4 = {
    'Mine_Meadowbank': 'Mine Meadowbank (or, Agnico Eagle)',
    'Mine_Meliadine':  'Mine Meliadine (or, Agnico Eagle)',
    'Mine_Hope_Bay':   'Mine Hope Bay (or, TMAC Resources)',
    'Mine_Baffinland': 'Mine Mary River — Baffinland (fer)',
}

fig_g4, axes_g4 = plt.subplots(2, 2, figsize=(14, 8.5))
#fig_g4.suptitle(
#    "Série temporelle de luminosité nocturne par mine (2000–2024)\n"
#    "DMSP converti en VIIRS (pointillés) et VIIRS observé (trait continu)",
#    fontsize=13, fontweight='bold', y=1.01, fontfamily='serif'           # ↑ 11 → 13
#)

for ax_m, (mine_nom, mine_label) in zip(axes_g4.flatten(), _mines_g4.items()):

    _d_m = (
        DMSP_NUNAVUT_MINE_2000_2011[DMSP_NUNAVUT_MINE_2000_2011["Nom"] == mine_nom]
        [['Annee', 'VIIRS_SUM_RESAMPL']]
        .dropna(subset=['VIIRS_SUM_RESAMPL'])
        .sort_values('Annee')
        .reset_index(drop=True)
    )
    _v_m = (
        VIIRS_NUNAVUT_MINE_RESAMPL[VIIRS_NUNAVUT_MINE_RESAMPL["Nom"] == mine_nom]
        [['Annee', 'VIIRS_SUM_RESAMPL']]
        .dropna(subset=['VIIRS_SUM_RESAMPL'])
        .sort_values('Annee')
        .reset_index(drop=True)
    )

    if not _d_m.empty:
        ax_m.plot(
            _d_m['Annee'], _d_m['VIIRS_SUM_RESAMPL'],
            color=_C_DMSP, linewidth=1.8,
            marker='s', markersize=4.5,
            markeredgecolor='white', markeredgewidth=0.5,
            linestyle='--', zorder=3,
            label='VIIRS estimé (DMSP)'
        )
        ax_m.fill_between(
            _d_m['Annee'], _d_m['VIIRS_SUM_RESAMPL'],
            alpha=0.12, color=_C_DMSP
        )

    if not _v_m.empty:
        ax_m.plot(
            _v_m['Annee'], _v_m['VIIRS_SUM_RESAMPL'],
            color=_C_VIIRS, linewidth=1.8,
            marker='o', markersize=4.5,
            markeredgecolor='white', markeredgewidth=0.5,
            linestyle='-', zorder=3,
            label='VIIRS observé'
        )
        ax_m.fill_between(
            _v_m['Annee'], _v_m['VIIRS_SUM_RESAMPL'],
            alpha=0.12, color=_C_VIIRS
        )

    # Zone chevauchement
    ax_m.axvspan(2011.55, 2013.45, alpha=0.07, color='#888888', zorder=1)

    # Ticks X adaptés
    _ann_m = sorted(
        set(_d_m['Annee'].astype(int).tolist() if not _d_m.empty else []) |
        set(_v_m['Annee'].astype(int).tolist() if not _v_m.empty else [])
    )
    if _ann_m:
        ax_m.set_xticks(_ann_m)
        ax_m.set_xticklabels(
            [str(a) for a in _ann_m], rotation=55, ha='right', fontsize=8   # ↑ 7.5 → 8
        )
        ax_m.set_xlim(_ann_m[0] - 0.6, _ann_m[-1] + 0.6)

    ax_m.set_ylim(bottom=0)
    ax_m.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}')
    )
    ax_m.set_title(mine_label, fontsize=10, fontweight='bold',               # ↑ 9.5 → 10
                   pad=5, fontfamily='serif')
    ax_m.set_xlabel('Année', fontsize=9)                                     # ↑ 8.5 → 9
    ax_m.set_ylabel(r'VIIRS (nW$\cdot$cm$^{-2}\cdot$sr$^{-1}$)', fontsize=9) # ↑ 8.5 → 9
    ax_m.legend(fontsize=8, framealpha=0.9, edgecolor='#CCCCCC',             # ↑ 7.5 → 8
                borderpad=0.5, loc='upper left')
    ax_m.spines['top'].set_visible(False)
    ax_m.spines['right'].set_visible(False)
    ax_m.yaxis.grid(True, linestyle=':', linewidth=0.5,
                    color='#CCCCCC', zorder=0)
    ax_m.set_axisbelow(True)

plt.tight_layout()
_fp_g4 = os.path.join(fig_dir, "conversion_serie_temporelle_mines.png")
plt.savefig(_fp_g4, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ G4 sauvegardé : {_fp_g4}")

plt.rcdefaults()
print("\n  ✔ Tous les graphiques de validation DMSP → VIIRS générés.")
####################### MERGE DN A DONNEE NUNAVUT

DN_DONNEES_NUNAVUT_2000_2024 = pd.merge(
    DMSP_VIIRS_2000_2024_NUNAVUT,
    DONNEES_NUNAVUT,   # toutes les colonnes
    on="Annee",
    how="left"
)

################ AJOUT SOMME ANNUELLE DN MINES (pour correction VAEMP et SAMP)
DN_DONNEES_NUNAVUT_2000_2024 = DN_DONNEES_NUNAVUT_2000_2024.merge(
    DN_SUM_MINE_ANNUEL[["Annee", "VIIRS_SUM_MINE_ANNUEL",
                        "Log_VIIRS_SUM_MINE_ANNUEL",
                        "Diff_Log_VIIRS_SUM_MINE_ANNUEL"]],
    on="Annee", how="left"
)

###################   SELECTION DONNEES DE 2005 A 2024
DN_DONNEES_NUNAVUT_2005_2024 = DN_DONNEES_NUNAVUT_2000_2024[(DN_DONNEES_NUNAVUT_2000_2024["Annee"] >= 2005) & (DN_DONNEES_NUNAVUT_2000_2024["Annee"] <= 2024)]
# ─────────────────────────────────────────────────────────────────────────────
# G5 : Série temporelle par mine (2 × 2 sous-graphiques) — VERSION AMÉLIORÉE
# ─────────────────────────────────────────────────────────────────────────────

COULEURS = {
    'Mine_Meadowbank' : '#1b7837',
    'Mine_Baffinland' : '#2166ac',
    'Mine_Hope_Bay'   : '#d6604d',
    'Mine_Meliadine'  : '#762a83',
}


mine_open_yr = {
    'Mine_Meadowbank': 2010,
    'Mine_Hope_Bay':   2017,
    'Mine_Baffinland': 2014,
    'Mine_Meliadine':  2019,
}

fig_g4, axes_g4 = plt.subplots(2, 2, figsize=(14, 9), facecolor='white')
#fig_g4.suptitle(
#    "Série temporelle de luminosité nocturne par mine (2000–2024)\n"
#    "DMSP converti en VIIRS (trait plein) et VIIRS observé (tirets)",
#    fontsize=13, fontweight='bold', y=1.02, fontfamily='serif'           # ↑ 12 → 13
#)

for ax_m, (mine_nom, mine_label) in zip(axes_g4.flatten(), _mines_g4.items()):

    c = COULEURS[mine_nom]   # ← couleur spécifique à la mine (comme Fig. 5)

    _d_m = (
        DMSP_NUNAVUT_MINE_2000_2011[DMSP_NUNAVUT_MINE_2000_2011["Nom"] == mine_nom]
        [['Annee', 'VIIRS_SUM_RESAMPL']]
        .dropna(subset=['VIIRS_SUM_RESAMPL'])
        .sort_values('Annee')
        .reset_index(drop=True)
    )
    _v_m = (
        VIIRS_NUNAVUT_MINE_RESAMPL[VIIRS_NUNAVUT_MINE_RESAMPL["Nom"] == mine_nom]
        [['Annee', 'VIIRS_SUM_RESAMPL']]
        .dropna(subset=['VIIRS_SUM_RESAMPL'])
        .sort_values('Annee')
        .reset_index(drop=True)
    )

    # Calcul ymax pour positionner les annotations
    ymax_d = _d_m['VIIRS_SUM_RESAMPL'].max() if not _d_m.empty else 0
    ymax_v = _v_m['VIIRS_SUM_RESAMPL'].max() if not _v_m.empty else 0
    ymax   = max(ymax_d, ymax_v) * 1.18 if max(ymax_d, ymax_v) > 0 else 1

    # ── DMSP converti (trait plein, cercles) ──────────────────────────────
    if not _d_m.empty:
        ax_m.fill_between(
            _d_m['Annee'], _d_m['VIIRS_SUM_RESAMPL'],
            alpha=0.20, color=c, zorder=2
        )
        ax_m.plot(
            _d_m['Annee'], _d_m['VIIRS_SUM_RESAMPL'],
            color=c, linewidth=2.0,
            marker='o', markersize=5,
            markeredgecolor='white', markeredgewidth=0.6,
            linestyle='-', zorder=4,
            label='VIIRS estimé (DMSP)'
        )

    # ── VIIRS observé (tirets, carrés) ───────────────────────────────────
    if not _v_m.empty:
        ax_m.fill_between(
            _v_m['Annee'], _v_m['VIIRS_SUM_RESAMPL'],
            alpha=0.14, color=c, zorder=2
        )
        ax_m.plot(
            _v_m['Annee'], _v_m['VIIRS_SUM_RESAMPL'],
            color=c, linewidth=2.2,
            marker='s', markersize=5,
            markeredgecolor='white', markeredgewidth=0.6,
            linestyle='--', zorder=4,
            label='VIIRS observé'
        )

    # ── Rupture DMSP → VIIRS ─────────────────────────────────────────────
    ax_m.axvline(2011.5, color='#777777', lw=0.9, ls=':', alpha=0.9, zorder=3)
    ax_m.text(2011.8, ymax * 0.96,
              'DMSP→VIIRS', fontsize=8, color='#666666',               # ↑ 7.5 → 8
              va='top', ha='left', style='italic')

    # ── Ouverture de la mine ──────────────────────────────────────────────
    open_yr = mine_open_yr.get(mine_nom)
    if open_yr:
        ax_m.axvline(open_yr, color=c, lw=1.4, ls='-.', alpha=0.80, zorder=3)
        ax_m.text(open_yr + 0.15, ymax * 0.85,
                  f'Ouv. {open_yr}', fontsize=8.5,                      # ↑ 8 → 8.5
                  color=c, fontweight='bold', va='top', ha='left')

    # ── Axes & mise en forme ──────────────────────────────────────────────
    _ann_m = sorted(
        set(_d_m['Annee'].astype(int).tolist() if not _d_m.empty else []) |
        set(_v_m['Annee'].astype(int).tolist() if not _v_m.empty else [])
    )
    if _ann_m:
        ax_m.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax_m.set_xlim(_ann_m[0] - 0.6, _ann_m[-1] + 0.6)
        ax_m.tick_params(axis='x', labelsize=9.5)                      # ↑ 9 → 9.5

    ax_m.set_ylim(bottom=0, top=ymax)
    ax_m.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}')
    )
    ax_m.set_title(mine_label, fontsize=11, fontweight='bold',          # ↑ 10.5 → 11
                   color=c, pad=6, fontfamily='serif')
    ax_m.set_xlabel('Année', fontsize=10)                               # ↑ 9.5 → 10
    ax_m.set_ylabel(r'Luminosité (unités VIIRS)', fontsize=10)          # ↑ 9.5 → 10
    ax_m.legend(fontsize=9, framealpha=0.90, edgecolor='#CCCCCC',       # ↑ 8.5 → 9
                borderpad=0.6, loc='upper left')
    ax_m.spines['top'].set_visible(False)
    ax_m.spines['right'].set_visible(False)
    ax_m.yaxis.grid(True, linestyle=':', linewidth=0.5,
                    color='#CCCCCC', zorder=0)
    ax_m.set_axisbelow(True)

plt.tight_layout(h_pad=2.5, w_pad=2.5)
_fp_g4 = os.path.join(fig_dir, "conversion_serie_temporelle_mines.png")
plt.savefig(_fp_g4, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ G4 sauvegardé : {_fp_g4}")

#%% ALLOCATION DE LA VALEUR AJOUTÉE AURIFÈRE (VAOA_EN) PAR MINE D'OR
################################################################################
# ALLOCATION DE LA VALEUR AJOUTÉE AURIFÈRE (VAOA_EN) PAR MINE D'OR
# Méthode : partage proportionnel à la production en onces par mine
#
#   VAOA_mine(t) = [prod_mine_oz(t) / prod_totale_or_oz(t)] × VAOA_EN(t)
#
#   — Mine Meadowbank  : GPMB_O   (or, Agnico Eagle, opérationnelle 2010)
#   — Mine Meliadine   : GPM_O    (or, Agnico Eagle, opérationnelle 2019)
#   — Mine Hope Bay    : GPHB_O   (or, TMAC Resources, opérationnelle 2017)
#   ⚠ Mine Baffinland (fer) exclue — VAOA_EN couvre l'or uniquement
################################################################################

TABLE_VAOA = DONNEES_NUNAVUT[['Annee', 'VAOA_EN', 'PIB_EN','VAF_EN','GPMB_O', 'GPM_O', 'GPHB_O', 'GPNV_O','POR_CA']].copy()
# Note the double brackets - outer brackets for indexing, inner brackets for the list of column names
TABLE_VAOA['Share_MBK'] = (TABLE_VAOA['GPMB_O'] / TABLE_VAOA['GPNV_O'])
TABLE_VAOA['Share_MEL'] = (TABLE_VAOA['GPM_O'] / TABLE_VAOA['GPNV_O'])
TABLE_VAOA['Share_HB'] = (TABLE_VAOA['GPHB_O'] / TABLE_VAOA['GPNV_O'])


# ── VAOA allouée par mine (M$ CAD, dollars enchaînés 2017) ───────────────────
TABLE_VAOA['VAOA_MBK'] = TABLE_VAOA['Share_MBK'] * TABLE_VAOA['VAOA_EN']
TABLE_VAOA['VAOA_MEL'] = TABLE_VAOA['Share_MEL'] * TABLE_VAOA['VAOA_EN']
TABLE_VAOA['VAOA_HB']  = TABLE_VAOA['Share_HB']  * TABLE_VAOA['VAOA_EN']

# ── Part en % de la VAOA totale ──────────────────────────────────────────────
TABLE_VAOA['TX_VAOA_MBK'] = TABLE_VAOA['Share_MBK'] * 100
TABLE_VAOA['TX_VAOA_MEL'] = TABLE_VAOA['Share_MEL'] * 100
TABLE_VAOA['TX_VAOA_HB']  = TABLE_VAOA['Share_HB']  * 100

# ══════════════════════════════════════════════════════════════════════════════
# TABLEAU VAOA — Méthode proportionnelle (valeurs et taux par mine)
# ══════════════════════════════════════════════════════════════════════════════
_tab_vaoa = TABLE_VAOA[
    (TABLE_VAOA['Annee'] >= 2010) & (TABLE_VAOA['Annee'] <= 2024)
][['Annee', 'VAOA_EN',
   'VAOA_MBK', 'TX_VAOA_MBK',
   'VAOA_MEL', 'TX_VAOA_MEL',
   'VAOA_HB',  'TX_VAOA_HB']].copy()

_tab_vaoa.rename(columns={
    'Annee':       'Année',
    'VAOA_EN':     'VAOA totale\n(M\\$ CAD)',
    'VAOA_MBK':    'Meadowbank\n(M\\$ CAD)',
    'TX_VAOA_MBK': 'Meadowbank\n(\\%)',
    'VAOA_MEL':    'Meliadine\n(M\\$ CAD)',
    'TX_VAOA_MEL': 'Meliadine\n(\\%)',
    'VAOA_HB':     'Hope Bay\n(M\\$ CAD)',
    'TX_VAOA_HB':  'Hope Bay\n(\\%)',
}, inplace=True)

print("\n===== VAOA_EN allouée par mine — méthode proportionnelle (M$ CAD 2017) =====")
print(_tab_vaoa.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))

# ── Export Excel (feuille 1) ──────────────────────────────────────────────────
_xl_vaoa_path = os.path.join(tab_dir, 'tableau_VAOA_allocation.xlsx')
with pd.ExcelWriter(_xl_vaoa_path, engine='openpyxl') as _writer:
    _tab_vaoa.to_excel(_writer, sheet_name='VAOA_proportionnelle', index=False)
print(f"  ✔ Excel sauvegardé : {_xl_vaoa_path}")

# ── Helper : formatage français des nombres pour LaTeX ──────────────────────
def _fmt_fr(val, decimals=1):
    """Formate un nombre en convention française pour LaTeX :
       virgule décimale protégée {,} + espace fine \\, pour les milliers.
       Retourne '---' pour NaN."""
    if pd.isna(val):
        return "---"
    s = f"{val:,.{decimals}f}"               # ex. "1,898.9"
    s = s.replace(",", "§").replace(".", ",").replace("§", r"\,")
    s = s.replace(",", "{,}")                # protéger toutes les virgules
    s = s.replace(r"\{,}", r"\,")            # restaurer le séparateur milliers
    return s

def _is_pct_col(col_name):
    """Détecte une colonne de pourcentage (en-tête contient '\\%' ou '(%)')."""
    return r"\%" in col_name or "(%)" in col_name

# ── Export LaTeX générique au format « mines × (M$ | %) + Total » ────────────
def _pivot_mines_to_latex(df, mines_def, caption, label, fichier,
                          note='', decimals_m=1, decimals_p=1):
    """Génère un tableau LaTeX au format :
       Année | Mine1 (M$ | %) | Mine2 (M$ | %) | ... | Total (M$ | %)

       df         : DataFrame contenant 'Annee' et les colonnes M$/% requises
       mines_def  : liste de tuples (label, col_M$, col_pct) — le dernier est
                    typiquement ('Total', col_total_M$, col_total_pct)
       decimals_m : décimales pour les colonnes M$
       decimals_p : décimales pour les colonnes %
    """
    n_grp   = len(mines_def)
    col_fmt = 'l' + ('rr' * n_grp)

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(r'  \footnotesize')
    lines.append(r'  \setlength{\tabcolsep}{4pt}')
    lines.append(r'  \caption{' + caption + r'}')
    lines.append(r'  \label{' + label + r'}')
    lines.append(r'  \begin{tabular}{' + col_fmt + r'}')
    lines.append(r'    \hline\hline')

    # En-tête niveau 1 : Année + groupes de mines + Total
    _h1 = [r'\multirow{2}{*}{Année}']
    for (lbl, _, _) in mines_def:
        _h1.append(r'\multicolumn{2}{c}{' + lbl + '}')
    lines.append('    ' + ' & '.join(_h1) + r' \\')

    # cmidrule pour chaque paire (M$ | %)
    _cmid = ''
    for i in range(n_grp):
        c1 = 2 + 2 * i
        c2 = c1 + 1
        _cmid += r'\cmidrule(lr){' + f'{c1}-{c2}' + '}'
    lines.append('    ' + _cmid)

    # En-tête niveau 2 : M$ / %
    _h2 = ['']
    for _ in range(n_grp):
        _h2.append(r'M\$')
        _h2.append(r'\%')
    lines.append('    ' + ' & '.join(_h2) + r' \\')
    lines.append(r'    \hline')

    # Données
    for _, row in df.iterrows():
        cells = [str(int(row['Annee']))]
        for (_, col_m, col_p) in mines_def:
            vm = row.get(col_m, np.nan) if col_m else np.nan
            vp = row.get(col_p, np.nan) if col_p else np.nan
            cells.append(_fmt_fr(vm, decimals_m))
            cells.append(_fmt_fr(vp, decimals_p))
        lines.append('    ' + ' & '.join(cells) + r' \\')

    lines.append(r'    \hline\hline')
    lines.append(r'  \end{tabular}')
    if note:
        lines.append(r'  \begin{minipage}{\linewidth}')
        lines.append(r'    \vspace{2pt}\footnotesize ' + note)
        lines.append(r'  \end{minipage}')
    lines.append(r'\end{table}')

    _path = os.path.join(tab_dir, fichier)
    with open(_path, 'w', encoding='utf-8') as _f:
        _f.write('\n'.join(lines))
    print(f"  ✔ LaTeX sauvegardé : {_path}")

# ── Préparation des données VAOA : Total M$ (=VAOA_EN) + Total % (=100 par construction)
_vaoa_for_tex = TABLE_VAOA[
    (TABLE_VAOA['Annee'] >= 2010) & (TABLE_VAOA['Annee'] <= 2024)
].copy()
_vaoa_for_tex['Total_TX_VAOA'] = (
    _vaoa_for_tex[['TX_VAOA_MBK', 'TX_VAOA_MEL', 'TX_VAOA_HB']]
    .fillna(0).sum(axis=1)
)

_pivot_mines_to_latex(
    _vaoa_for_tex,
    mines_def=[
        ('Meadowbank', 'VAOA_MBK', 'TX_VAOA_MBK'),
        ('Meliadine',  'VAOA_MEL', 'TX_VAOA_MEL'),
        ('Hope Bay',   'VAOA_HB',  'TX_VAOA_HB'),
        ('Total',      'VAOA_EN',  'Total_TX_VAOA'),
    ],
    caption=(r'Allocation de la valeur ajoutée aurifère (VAOA) par mine d\'or '
             r'--- Valeurs (M\$ CAD 2017) et taux (\%), 2010--2024'),
    label='tab:vaoa_proportionnelle',
    fichier='tableau_VAOA_proportionnelle.tex',
    note=(r'\textit{Note :} Méthode de partage proportionnel à la production '
          r'en onces par mine ($\text{VAOA}_{\text{mine}} = '
          r'\frac{\text{prod}_{\text{mine}}}{\text{prod}_{\text{totale}}} '
          r'\times \text{VAOA}_{\text{EN}}$). '
          r'La mine Baffinland (fer) est exclue car la VAOA couvre uniquement '
          r'le secteur aurifère. La colonne « Total \% » reflète la somme des '
          r'parts (100\,\% par construction). Valeurs en millions de dollars '
          r'enchaînés de 2017. Requiert '
          r'\texttt{\textbackslash usepackage\{multirow,booktabs\}}.')
)

# ============================================================
# RÉGRESSION IV2SLS LOG-LOG : Log(VAOA_EN) ~ Log(GPNV_O) | Log(POR_CA)
#   Y  (dépendante)  : Log(VAOA_EN)  — log valeur ajoutée aurifère
#   X  (endogène)    : Log(GPNV_O)   — log production totale d'or (oz)
#   Z  (instrument)  : Log(POR_CA)   — log prix de l'or en CAD/oz
#   Interprétation β : élasticité — si prod. +1 %, VAOA + β %
#   Justification    : le prix de l'or détermine la production mais est
#                      exogène aux conditions locales du Nunavut
# ============================================================
TABLE_VAOA['Log_VAOA_EN'] = np.log(TABLE_VAOA['VAOA_EN'])
TABLE_VAOA['Log_GPNV_O']  = np.log(TABLE_VAOA['GPNV_O'])
TABLE_VAOA['Log_POR_CA']  = np.log(TABLE_VAOA['POR_CA'])
TABLE_VAOA['Log_GPM_O'] = np.log(TABLE_VAOA['GPM_O'])
TABLE_VAOA['Log_GPHB_O'] = np.log(TABLE_VAOA['GPHB_O'])

# Garder seulement les lignes où les deux colonnes ont des valeurs finies
mask = (np.isfinite(TABLE_VAOA['Log_GPNV_O'])) & (np.isfinite(TABLE_VAOA['Log_VAOA_EN']))
TABLE_VAOA_LOG = TABLE_VAOA[mask]

Log_VAOA_EN   = TABLE_VAOA_LOG['Log_VAOA_EN']
Log_VAOA_EN_CT = pd.DataFrame({'const': 1.0}, index=Log_VAOA_EN.index)
Log_GPNV_O  = TABLE_VAOA_LOG[['Log_GPNV_O']]
Log_POR_VAO = TABLE_VAOA_LOG[['Log_POR_CA']]

MODEL_IV_VAOA = IV2SLS_lm(Log_VAOA_EN, Log_VAOA_EN_CT, Log_GPNV_O, Log_POR_VAO).fit(cov_type='robust')

print("\n" + "="*70)
print("  IV2SLS LOG-LOG — Log(VAOA_EN) ~ Log(GPNV_O) | Log(POR_CA)")
print("="*70)
print(MODEL_IV_VAOA.summary)

# ── Coefficients IV ───────────────────────────────────────────────────────────
_beta_iv  = MODEL_IV_VAOA.params['Log_GPNV_O']  # élasticité prod → VAOA
_alpha_iv = MODEL_IV_VAOA.params['const']        # constante (log)

print(f"\n  Constante α            : {_alpha_iv:,.4f}")
print(f"  Élasticité β (Log_GPNV_O) : {_beta_iv:,.4f}")
print(f"  Interprétation : une hausse de 1 % de la production d'or "
      f"entraîne une variation de {_beta_iv:.4f} % de la VAOA aurifère")

# ── Prédiction VAOA pour chaque mine (retour en niveaux) ─────────────────────
# Log(VAOA_mine) = α + β × Log(prod_mine)
# → VAOA_mine = exp(α + β × Log(prod_mine))  si prod_mine > 0, sinon 0

for var_prod, _col_iv in [
    ('GPMB_O', 'VAOA_MBK_IV'),
    ('GPM_O',  'VAOA_MEL_IV'),
    ('GPHB_O', 'VAOA_HB_IV'),
]:
    prod_sel = TABLE_VAOA[var_prod].copy()
    TABLE_VAOA[_col_iv] = np.where(
        prod_sel > 0,
        np.exp(_alpha_iv + _beta_iv * np.log(prod_sel.clip(lower=1e-10))),
        0.0
    )

# Parts IV en %
for _col_iv, _col_tx in [
    ('VAOA_MBK_IV', 'TX_VAOA_MBK_IV'),
    ('VAOA_MEL_IV', 'TX_VAOA_MEL_IV'),
    ('VAOA_HB_IV',  'TX_VAOA_HB_IV'),
]:
    TABLE_VAOA[_col_tx] = np.where(
        TABLE_VAOA['VAOA_EN'] > 0,
        TABLE_VAOA[_col_iv] / TABLE_VAOA['VAOA_EN'] * 100,
        np.nan
    )

# ── Tableau comparatif : méthode proportionnelle vs IV log-log ───────────────
Table_VAO_METH = TABLE_VAOA[TABLE_VAOA['GPNV_O'] > 0][
    ['Annee', 'VAOA_EN',
     'VAOA_MBK',    'VAOA_MEL',    'VAOA_HB',
     'VAOA_MBK_IV', 'VAOA_MEL_IV', 'VAOA_HB_IV']
].copy()

Table_VAO_METH = Table_VAO_METH.set_index('Annee')
Table_VAO_METH.columns = pd.MultiIndex.from_tuples([
    ('VAOA totale (M$)', 'Officielle'),
    ('Meadowbank (M$)', 'Proportion.'), ('Meliadine (M$)', 'Proportion.'), ('Hope Bay (M$)', 'Proportion.'),
    ('Meadowbank (M$)', 'IV log-log'),  ('Meliadine (M$)', 'IV log-log'),  ('Hope Bay (M$)', 'IV log-log'),
])

print("\n" + "="*90)
print("  TABLEAU COMPARATIF — VAOA par mine (M$ CAD 2017)")
print("  Méthode proportionnelle  vs  IV2SLS log-log  (instrument : Log(prix de l'or))")
print("="*90)
print(Table_VAO_METH.to_string(float_format=lambda x: f'{x:>10,.1f}'))

# ══════════════════════════════════════════════════════════════════════════════
# TABLEAU VAOA — Comparatif méthode proportionnelle vs IV2SLS log-log
# ══════════════════════════════════════════════════════════════════════════════

# ── Export Excel (feuille 2 ajoutée au même fichier) ─────────────────────────
_tab_comp = TABLE_VAOA[(TABLE_VAOA['GPNV_O'] > 0) &
                       (TABLE_VAOA['Annee'] >= 2010)][[
    'Annee', 'VAOA_EN',
    'VAOA_MBK',    'TX_VAOA_MBK',
    'VAOA_MEL',    'TX_VAOA_MEL',
    'VAOA_HB',     'TX_VAOA_HB',
    'VAOA_MBK_IV', 'TX_VAOA_MBK_IV',
    'VAOA_MEL_IV', 'TX_VAOA_MEL_IV',
    'VAOA_HB_IV',  'TX_VAOA_HB_IV',
]].copy()

_tab_comp.rename(columns={
    'Annee':          'Année',
    'VAOA_EN':        'VAOA totale (M$)',
    'VAOA_MBK':       'MBK — Prop. (M$)',
    'TX_VAOA_MBK':    'MBK — Prop. (%)',
    'VAOA_MEL':       'MEL — Prop. (M$)',
    'TX_VAOA_MEL':    'MEL — Prop. (%)',
    'VAOA_HB':        'HB — Prop. (M$)',
    'TX_VAOA_HB':     'HB — Prop. (%)',
    'VAOA_MBK_IV':    'MBK — IV (M$)',
    'TX_VAOA_MBK_IV': 'MBK — IV (%)',
    'VAOA_MEL_IV':    'MEL — IV (M$)',
    'TX_VAOA_MEL_IV': 'MEL — IV (%)',
    'VAOA_HB_IV':     'HB — IV (M$)',
    'TX_VAOA_HB_IV':  'HB — IV (%)',
}, inplace=True)

with pd.ExcelWriter(_xl_vaoa_path, engine='openpyxl', mode='a') as _writer:
    _tab_comp.to_excel(_writer, sheet_name='VAOA_comparatif', index=False)
print(f"  ✔ Excel (feuille comparatif) ajouté : {_xl_vaoa_path}")

# ── Export LaTeX (tableau comparatif) ────────────────────────────────────────
_tab_comp_tex = TABLE_VAOA[(TABLE_VAOA['GPNV_O'] > 0) &
                           (TABLE_VAOA['Annee'] >= 2010)][[
    'Annee', 'VAOA_EN',
    'VAOA_MBK', 'VAOA_MBK_IV',
    'VAOA_MEL', 'VAOA_MEL_IV',
    'VAOA_HB',  'VAOA_HB_IV',
]].copy()

_lines_comp = []
_lines_comp.append(r'\begin{table}[htbp]')
_lines_comp.append(r'  \centering')
_lines_comp.append(r'  \caption{Allocation de la VAOA par mine --- Comparaison '
                   r'méthode proportionnelle et IV2SLS log-log (M\$ CAD 2017)}')
_lines_comp.append(r'  \label{tab:vaoa_comparatif}')
_lines_comp.append(r'  \small')
_lines_comp.append(r'  \begin{tabular}{crrrrrrrr}')
_lines_comp.append(r'    \toprule')
_lines_comp.append(r'    & & \multicolumn{2}{c}{Meadowbank} & \multicolumn{2}{c}{Meliadine} & \multicolumn{2}{c}{Hope Bay} \\')
_lines_comp.append(r'    \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}')
_lines_comp.append(r'    Année & VAOA (M\$) & Prop. & IV & Prop. & IV & Prop. & IV \\')
_lines_comp.append(r'    \midrule')
for _, row in _tab_comp_tex.iterrows():
    _yr   = str(int(row['Annee']))
    _tot  = _fmt_fr(row['VAOA_EN'], 1)
    _cells = [_yr, _tot]
    for col in ['VAOA_MBK', 'VAOA_MBK_IV',
                'VAOA_MEL', 'VAOA_MEL_IV',
                'VAOA_HB',  'VAOA_HB_IV']:
        v = row[col]
        _cells.append(_fmt_fr(v, 1) if pd.notna(v) and v > 0 else '---')
    _lines_comp.append('    ' + ' & '.join(_cells) + r' \\')
_lines_comp.append(r'    \bottomrule')
_lines_comp.append(r'  \end{tabular}')
_lines_comp.append(r'  \begin{minipage}{\linewidth}')
_lines_comp.append(r'    \vspace{2pt}\footnotesize \textit{Note :} '
                   r'Prop. = méthode de partage proportionnel à la production (oz). '
                   r'IV = prédiction par IV2SLS log-log instrumenté par le prix de '
                   r"l'or ($\log P_{\text{or}}$). "
                   r'Valeurs en millions de dollars enchaînés de 2017.')
_lines_comp.append(r'  \end{minipage}')
_lines_comp.append(r'\end{table}')

_path_comp_tex = os.path.join(tab_dir, 'tableau_VAOA_comparatif.tex')
with open(_path_comp_tex, 'w', encoding='utf-8') as _f:
    _f.write('\n'.join(_lines_comp))
print(f"  ✔ LaTeX sauvegardé : {_path_comp_tex}")



#%% GRAPHIQUE DE LA VALEUR AJOUTEE
# ── Graphique : barres empilées VAOA par mine + ligne VAOA totale ─────────────
plt.rcParams.update({
    'font.family':   'serif',
    'font.serif':    ['Times New Roman', 'DejaVu Serif'],
    'font.size':     12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'grid.color':       '#CCCCCC',
    'grid.linewidth':   0.6,
})

_C_MBK  = '#2C5F8A'   # bleu acier  — Meadowbank
_C_MEL  = '#2C7A4B'   # vert sauge  — Meliadine
_C_HB   = '#8A4A2C'   # brun rouille — Hope Bay
_C_TOT  = '#222222'   # gris foncé  — VAOA totale
# Filtrer les années avec au moins une mine active
_tv = TABLE_VAOA[TABLE_VAOA['GPNV_O'] > 0].sort_values('Annee').reset_index(drop=True)
_annees_v = _tv['Annee'].astype(int).values
_x_v = np.arange(len(_annees_v))

fig_vaoa, ax_vaoa = plt.subplots(figsize=(13, 6))

# Barres empilées
_b1 = ax_vaoa.bar(
    _x_v, _tv['VAOA_MBK'],
    color=_C_MBK, alpha=0.82, edgecolor='white', linewidth=0.3,
    width=0.65, zorder=2, label='Meadowbank'
)
_b2 = ax_vaoa.bar(
    _x_v, _tv['VAOA_MEL'],
    bottom=_tv['VAOA_MBK'],
    color=_C_MEL, alpha=0.82, edgecolor='white', linewidth=0.3,
    width=0.65, zorder=2, label='Meliadine'
)
_b3 = ax_vaoa.bar(
    _x_v, _tv['VAOA_HB'],
    bottom=_tv['VAOA_MBK'] + _tv['VAOA_MEL'],
    color=_C_HB, alpha=0.82, edgecolor='white', linewidth=0.3,
    width=0.65, zorder=2, label='Hope Bay'
)

# Ligne VAOA totale
ax_vaoa.plot(
    _x_v, _tv['VAOA_EN'],
    color=_C_TOT, linewidth=1.8, linestyle='--',
    marker='D', markersize=4.5, markeredgecolor='white', markeredgewidth=0.5,
    zorder=4, label='VAOA totale (officielle)'
)

# Axes
ax_vaoa.set_xticks(_x_v)
ax_vaoa.set_xticklabels([str(a) for a in _annees_v], rotation=50, ha='right')
ax_vaoa.set_xlim(-0.6, len(_x_v) - 0.4)
ax_vaoa.set_ylim(bottom=0)
ax_vaoa.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v:,.0f}')
)
ax_vaoa.set_xlabel('Année', fontsize=12, labelpad=5)                    # ↑ 10 → 12
ax_vaoa.set_ylabel('Valeur ajoutée aurifère (M$ CAD, base 2017)',
                   fontsize=12, labelpad=5)                             # ↑ 10 → 12
#ax_vaoa.set_title(
#    "Allocation de la valeur ajoutée aurifère (VAOA) par mine d'or — Nunavut\n"
#    "Partage proportionnel à la production en onces",
#    fontsize=13, fontweight='bold', pad=8, fontfamily='serif'           # ↑ 11 → 13
#)
ax_vaoa.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC',       # ↑ 9 → 10
               loc='upper left', borderpad=0.6)
ax_vaoa.spines['top'].set_visible(False)
ax_vaoa.spines['right'].set_visible(False)
ax_vaoa.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
ax_vaoa.set_axisbelow(True)

plt.tight_layout()
_fp_vaoa = os.path.join(fig_dir, "vaoa_allocation_mines_or.png")
plt.savefig(_fp_vaoa, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Figure VAOA sauvegardée : {_fp_vaoa}")

# ── Graphique 2 : parts (%) en barres empilées ────────────────────────────────
fig_vaoa_pct, ax_pct = plt.subplots(figsize=(13, 5.5))

ax_pct.bar(
    _x_v, _tv['TX_VAOA_MBK'],
    color=_C_MBK, alpha=0.82, edgecolor='white', linewidth=0.3,
    width=0.65, zorder=2, label='Meadowbank'
)
ax_pct.bar(
    _x_v, _tv['TX_VAOA_MEL'],
    bottom=_tv['TX_VAOA_MBK'],
    color=_C_MEL, alpha=0.82, edgecolor='white', linewidth=0.3,
    width=0.65, zorder=2, label='Meliadine'
)
ax_pct.bar(
    _x_v, _tv['TX_VAOA_HB'],
    bottom=_tv['TX_VAOA_MBK'] + _tv['TX_VAOA_MEL'],
    color=_C_HB, alpha=0.82, edgecolor='white', linewidth=0.3,
    width=0.65, zorder=2, label='Hope Bay'
)

ax_pct.axhline(100, color='#555555', linewidth=0.9,
               linestyle=':', zorder=3)

ax_pct.set_xticks(_x_v)
ax_pct.set_xticklabels([str(a) for a in _annees_v], rotation=50, ha='right')
ax_pct.set_xlim(-0.6, len(_x_v) - 0.4)
ax_pct.set_ylim(0, 108)
ax_pct.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v:.0f} %')
)
ax_pct.set_xlabel('Année', fontsize=12, labelpad=5)                     # ↑ 10 → 12
ax_pct.set_ylabel('Part de la VAOA aurifère (%)', fontsize=12, labelpad=5)  # ↑ 10 → 12
#ax_pct.set_title(
#    "Part de chaque mine dans la valeur ajoutée aurifère (VAOA) — Nunavut\n"
#    "Basée sur la production en onces d'or",
#    fontsize=13, fontweight='bold', pad=8, fontfamily='serif'           # ↑ 11 → 13
#)
ax_pct.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC',        # ↑ 9 → 10
              loc='upper left', borderpad=0.6)
ax_pct.spines['top'].set_visible(False)
ax_pct.spines['right'].set_visible(False)
ax_pct.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
ax_pct.set_axisbelow(True)

plt.tight_layout()
_fp_vaoa_pct = os.path.join(fig_dir, "vaoa_parts_mines_or.png")
plt.savefig(_fp_vaoa_pct, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Figure parts VAOA sauvegardée : {_fp_vaoa_pct}")

plt.rcdefaults()
print("\n  ✔ Allocation VAOA par mine terminée.")


#%% ALLOCATION SOUTIENT DES ACTIVITES MINIERES

TABLE_SAM = DONNEES_NUNAVUT[['Annee', 'SAMP_EN','GPMB_CAD_EN','GPM_CAD_EN' ,'GPHB_CAD_EN','IPB_CAD_EN','PROD_MINE_CAD_EN']].copy()
# Filtrer la base
TABLE_DN_SAM_Long = DMSP_VIIRS_MINE_VILLE_2000_2024[['Annee','Nom','VIIRS_SUM_RESAMPL']].copy()

TABLE_DN_SAM = TABLE_DN_SAM_Long.pivot_table(
    index='Annee',
    columns='Nom',
    values='VIIRS_SUM_RESAMPL',
    aggfunc='mean'
)

TABLE_DN_SAM['Total_DN'] = TABLE_DN_SAM.sum(axis=1)

TABLE_DN_SAM = (
    TABLE_DN_SAM
    .merge(TABLE_SAM, on='Annee', how='inner')
    .merge(TABLE_VAOA, on='Annee', how='inner')
)


TABLE_DN_SAM['Total_Share'] = TABLE_DN_SAM['Total_DN'] + TABLE_DN_SAM['PROD_MINE_CAD_EN']


TABLE_DN_SAM['Share_SAM_MBK'] = ((TABLE_DN_SAM['Mine_Meadowbank'] + TABLE_DN_SAM['GPMB_CAD_EN'])/ TABLE_DN_SAM['Total_Share'])
TABLE_DN_SAM['Share_SAM_MEL'] = ((TABLE_DN_SAM['Mine_Meliadine'] + TABLE_DN_SAM['GPM_CAD_EN'])/ TABLE_DN_SAM['Total_Share'])
TABLE_DN_SAM['Share_SAM_HB'] = ((TABLE_DN_SAM['Mine_Hope_Bay'] + TABLE_DN_SAM['GPHB_CAD_EN'])/ TABLE_DN_SAM['Total_Share'])
TABLE_DN_SAM['Share_SAM_BAF'] = ((TABLE_DN_SAM['Mine_Baffinland'] + TABLE_DN_SAM['IPB_CAD_EN'])/ TABLE_DN_SAM['Total_Share'])


# ── VAOA allouée par mine (M$ CAD, dollars enchaînés 2017) ───────────────────
TABLE_DN_SAM['SAM_MBK'] = TABLE_DN_SAM['Share_SAM_MBK'] * TABLE_DN_SAM['SAMP_EN']
TABLE_DN_SAM['SAM_MEL'] = TABLE_DN_SAM['Share_SAM_MEL'] * TABLE_DN_SAM['SAMP_EN']

TABLE_DN_SAM['SAM_HB'] = TABLE_DN_SAM['Share_SAM_HB'] * TABLE_DN_SAM['SAMP_EN']
TABLE_DN_SAM['SAM_BAF'] = TABLE_DN_SAM['Share_SAM_BAF'] * TABLE_DN_SAM['SAMP_EN']

# ── Part en % de la VAOA totale ──────────────────────────────────────────────
TABLE_DN_SAM['TX_SAM_MBK'] = TABLE_DN_SAM['Share_SAM_MBK'] * 100

TABLE_DN_SAM['TX_SAM_MEL'] = TABLE_DN_SAM['Share_SAM_MEL'] * 100
TABLE_DN_SAM['TX_SAM_HB'] = TABLE_DN_SAM['Share_SAM_HB'] * 100

TABLE_DN_SAM['TX_SAM_BAF'] = TABLE_DN_SAM['Share_SAM_BAF'] * 100

TABLE_DN_SAM['Total_TX'] = TABLE_DN_SAM['TX_SAM_BAF'] + TABLE_DN_SAM['TX_SAM_HB'] + TABLE_DN_SAM['TX_SAM_MEL'] + TABLE_DN_SAM['TX_SAM_MBK']




#%% MODÈLE STATISTIQUE CANADA

TABLE_DN_SAM['PIB_MEL_SC'] = TABLE_DN_SAM['SAM_MEL'] + TABLE_DN_SAM['VAOA_MEL'].fillna(0)
TABLE_DN_SAM['PIB_MBK_SC'] = TABLE_DN_SAM['SAM_MBK'] + TABLE_DN_SAM['VAOA_MBK'].fillna(0)
TABLE_DN_SAM['PIB_HB_SC'] = TABLE_DN_SAM['SAM_HB'] + TABLE_DN_SAM['VAOA_HB'].fillna(0)
TABLE_DN_SAM['PIB_BAF_SC'] = TABLE_DN_SAM['SAM_BAF'] + TABLE_DN_SAM['VAF_EN']


TABLE_DN_SAM['TX_PIB_MEL_SC'] = (TABLE_DN_SAM['PIB_MEL_SC'] / TABLE_DN_SAM['PIB_EN'])*100
TABLE_DN_SAM['TX_PIB_MBK_SC'] = (TABLE_DN_SAM['PIB_MBK_SC'] / TABLE_DN_SAM['PIB_EN'])*100
TABLE_DN_SAM['TX_PIB_HB_SC'] = (TABLE_DN_SAM['PIB_HB_SC'] / TABLE_DN_SAM['PIB_EN'])*100
TABLE_DN_SAM['TX_PIB_BAF_SC'] = (TABLE_DN_SAM['PIB_BAF_SC'] / TABLE_DN_SAM['PIB_EN'])*100

TABLE_DN_SAM['Total_TX_PIB'] = TABLE_DN_SAM['TX_PIB_MEL_SC'] + TABLE_DN_SAM['TX_PIB_MBK_SC'] + TABLE_DN_SAM['TX_PIB_HB_SC'] + TABLE_DN_SAM['TX_PIB_BAF_SC']

# ── Restriction du modèle StatCan : démarre en 2005 ─────────────────────────
# Les colonnes _SC sont mises à NaN avant 2005 ; les tableaux et figures qui
# utilisent dropna(subset=...) excluront automatiquement ces années.
_sc_mask_pre2005 = TABLE_DN_SAM['Annee'] < 2005
_sc_cols_to_mask = [
    'PIB_MEL_SC', 'PIB_MBK_SC', 'PIB_HB_SC', 'PIB_BAF_SC',
    'TX_PIB_MEL_SC', 'TX_PIB_MBK_SC', 'TX_PIB_HB_SC', 'TX_PIB_BAF_SC',
    'Total_TX_PIB',
]
TABLE_DN_SAM.loc[_sc_mask_pre2005, _sc_cols_to_mask] = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# TABLEAU SAM : Allocation du soutien aux activités minières par mine
# ══════════════════════════════════════════════════════════════════════════════

# ── Tableau A : Valeurs et taux par mine ─────────────────────────────────────
_tab_sam = TABLE_DN_SAM[
    (TABLE_DN_SAM['Annee'] >= 2005) & (TABLE_DN_SAM['Annee'] <= 2024)
][['Annee', 'SAMP_EN',
   'SAM_MBK', 'TX_SAM_MBK',
   'SAM_MEL', 'TX_SAM_MEL',
   'SAM_HB',  'TX_SAM_HB',
   'SAM_BAF', 'TX_SAM_BAF',
   'Total_TX']].copy()

_tab_sam.rename(columns={
    'Annee':       'Année',
    'SAMP_EN':     'SAMP total\n(M\\$ CAD)',
    'SAM_MBK':     'Meadowbank\n(M\\$ CAD)',
    'TX_SAM_MBK':  'Meadowbank\n(\\%)',
    'SAM_MEL':     'Meliadine\n(M\\$ CAD)',
    'TX_SAM_MEL':  'Meliadine\n(\\%)',
    'SAM_HB':      'Hope Bay\n(M\\$ CAD)',
    'TX_SAM_HB':   'Hope Bay\n(\\%)',
    'SAM_BAF':     'Baffinland\n(M\\$ CAD)',
    'TX_SAM_BAF':  'Baffinland\n(\\%)',
    'Total_TX':    'Total\n(\\%)',
}, inplace=True)

print("\n===== Allocation SAM par mine (valeurs M$ CAD 2017 et taux %) =====")
print(_tab_sam.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))

# ── Export Excel ──────────────────────────────────────────────────────────────
_xl_sam_path = os.path.join(tab_dir, 'tableau_SAM_allocation.xlsx')
with pd.ExcelWriter(_xl_sam_path, engine='openpyxl') as _writer:
    _tab_sam.to_excel(_writer, sheet_name='SAM_allocation', index=False)
print(f"  ✔ Excel sauvegardé : {_xl_sam_path}")

# ── Export LaTeX SAMP (même format que VAOA : mines × M$|% + Total) ──────────
_sam_for_tex = TABLE_DN_SAM[
    (TABLE_DN_SAM['Annee'] >= 2005) & (TABLE_DN_SAM['Annee'] <= 2024)
].copy()
# Total M$ = somme des SAM des 4 mines (≈ SAMP_EN par construction)
_sam_for_tex['Total_SAM'] = (
    _sam_for_tex[['SAM_MBK', 'SAM_MEL', 'SAM_HB', 'SAM_BAF']]
    .fillna(0).sum(axis=1)
)

_pivot_mines_to_latex(
    _sam_for_tex,
    mines_def=[
        ('Meadowbank', 'SAM_MBK',   'TX_SAM_MBK'),
        ('Meliadine',  'SAM_MEL',   'TX_SAM_MEL'),
        ('Hope Bay',   'SAM_HB',    'TX_SAM_HB'),
        ('Baffinland', 'SAM_BAF',   'TX_SAM_BAF'),
        ('Total',      'Total_SAM', 'Total_TX'),
    ],
    caption=(r'Allocation du soutien aux activités minières (SAMP) par mine '
             r'--- Valeurs (M\$ CAD 2017) et taux (\%), 2005--2024'),
    label='tab:sam_allocation',
    fichier='tableau_SAM_allocation.tex',
    note=(r'\textit{Note :} Valeurs en millions de dollars enchaînés de 2017. '
          r'Les taux (\%) représentent la part de chaque mine dans le soutien '
          r'total aux activités minières (SAMP), calculée par la méthode de '
          r'luminosité nocturne. La colonne « Total » somme les contributions '
          r'des quatre mines. Requiert '
          r'\texttt{\textbackslash usepackage\{multirow,booktabs\}}.'),
    decimals_m=2, decimals_p=1
)
#%% CALCUL PIB CORRIGE
# ============================================================
# PARAMÈTRES
# ============================================================
rho       = 0.94   # Canada = pays développé
alpha     = 0.45   # élasticité lumière-PIB
base_year = 2024

# ============================================================
# RÉGRESSIONS
# ============================================================

# Régression en différences logarithmiques (pour alpha)
Diff_Log_pib          = DN_DONNEES_NUNAVUT_2005_2024["Diff_Log_PIB_EN"]
Diff_Log_viirs        = DN_DONNEES_NUNAVUT_2005_2024["Diff_Log_VIIRS_SUM_RESAMPL"]
Diff_Log_viirs_const  = sm.add_constant(Diff_Log_viirs)
model_diff            = sm.OLS(Diff_Log_pib, Diff_Log_viirs_const).fit(cov_type='HC3')
print(model_diff.summary())

# Régression en niveaux logarithmiques (pour valeur de base 2024)
Log_pib         = DN_DONNEES_NUNAVUT_2005_2024["Log_PIB_EN"]
Log_viirs       = DN_DONNEES_NUNAVUT_2005_2024["Log_VIIRS_SUM_RESAMPL"]
Log_viirs_const = sm.add_constant(Log_viirs)
model_niveaux   = sm.OLS(Log_pib, Log_viirs_const).fit(cov_type='HC3')
print(model_niveaux.summary())

# ============================================================
# TRIER PAR ANNÉE
# ============================================================
DN_DONNEES_NUNAVUT_2005_2024 = (
    DN_DONNEES_NUNAVUT_2005_2024
    .sort_values('Annee', ascending=True)
    .reset_index(drop=True)
)

# ============================================================
# ÉQUATION 2 — Taux de croissance révisé (niveau régional)
# y*_t = rho * y_t + (1-rho) * alpha * Diff_Log_VIIRS_t
# Pas de condition sur DN — équation nationale
# ============================================================
DN_DONNEES_NUNAVUT_2005_2024['g_revisee'] = (
    rho * DN_DONNEES_NUNAVUT_2005_2024['Diff_Log_PIB_EN']
    + (1 - rho) * alpha
    * DN_DONNEES_NUNAVUT_2005_2024['Diff_Log_VIIRS_SUM_RESAMPL']
)

# ============================================================
# VALEUR DE BASE 2024 — même méthode que Modele Chen
# base_pib = PIB_EN[2024] directement (valeur réelle pour l'année de référence)
# ============================================================
DN_DONNEES_NUNAVUT_2005_2024['PIB_REEL'] = np.nan

base_pib = DN_DONNEES_NUNAVUT_2005_2024.loc[
    DN_DONNEES_NUNAVUT_2005_2024['Annee'] == base_year, 'PIB_EN'
].values[0]

DN_DONNEES_NUNAVUT_2005_2024.loc[
    DN_DONNEES_NUNAVUT_2005_2024['Annee'] == base_year, 'PIB_REEL'
] = base_pib

print(f"\nPIB de base {base_year} : {base_pib:,.0f}")

# ============================================================
# ÉQUATION 21 — Calcul rétrospectif
# PIB_t = PIB_{t+1} / (1 + g_t)   si VIIRS_t != 0
# PIB_t = 0                         si VIIRS_t == 0
# ============================================================
for i in range(len(DN_DONNEES_NUNAVUT_2005_2024) - 1, -1, -1):

    annee_t = DN_DONNEES_NUNAVUT_2005_2024.loc[i, 'Annee']

    # Sauter l'année de référence
    if annee_t == base_year:
        continue

    # Équation 21 : VIIRS_t = 0 → PIB_t = 0
    viirs_t = DN_DONNEES_NUNAVUT_2005_2024.loc[i, 'VIIRS_SUM_RESAMPL']
    if pd.isna(viirs_t) or viirs_t == 0:
        DN_DONNEES_NUNAVUT_2005_2024.loc[i, 'PIB_REEL'] = 0
        continue

    # Trouver l'index de t+1
    next_idx = DN_DONNEES_NUNAVUT_2005_2024[
        DN_DONNEES_NUNAVUT_2005_2024['Annee'] == annee_t + 1
    ].index

    if len(next_idx) == 0:
        DN_DONNEES_NUNAVUT_2005_2024.loc[i, 'PIB_REEL'] = 0
        continue

    next_idx  = next_idx[0]
    pib_next  = DN_DONNEES_NUNAVUT_2005_2024.loc[next_idx, 'PIB_REEL']

    # ✅ Taux de l'année t (pas t+1)
    g_current = DN_DONNEES_NUNAVUT_2005_2024.loc[i, 'g_revisee']

    # Vérifications
    if pd.isna(pib_next) or pd.isna(g_current) or g_current == -1:
        DN_DONNEES_NUNAVUT_2005_2024.loc[i, 'PIB_REEL'] = 0
        continue

    # ✅ Équation 21 : PIB_t = PIB_{t+1} / (1 + g_t)
    DN_DONNEES_NUNAVUT_2005_2024.loc[i, 'PIB_REEL'] = (
        pib_next / (1 + g_current)
    )

# ============================================================
# VÉRIFICATION
# ============================================================
print(
    DN_DONNEES_NUNAVUT_2005_2024[[
        'Annee', 'PIB_EN', 'VIIRS_SUM_RESAMPL',
        'g_revisee', 'PIB_REEL'
    ]].to_string()
)

################################# CALCUL LOG DIFF ET TAUX DE CROISSANCE
DN_DONNEES_NUNAVUT_2005_2024['Log_PIB_REEL'] = np.log(DN_DONNEES_NUNAVUT_2005_2024['PIB_REEL'])
DN_DONNEES_NUNAVUT_2005_2024['Diff_PIB_REEL'] =  DN_DONNEES_NUNAVUT_2005_2024['PIB_REEL'].diff()
DN_DONNEES_NUNAVUT_2005_2024['Diff_Log_PIB_REEL'] =  DN_DONNEES_NUNAVUT_2005_2024['Log_PIB_REEL'].diff()







#%% CORRELATION


###########################################################################################
                           ############ CORRELATION 
##########################################################################################                           

auto_PIB  = DN_DONNEES_NUNAVUT_2005_2024["Diff_PIB_REEL"].autocorr(lag=1)
auto_VIIRS = DN_DONNEES_NUNAVUT_2005_2024["Diff_VIIRS_SUM_RESAMPL"].autocorr(lag=1)
auto_POR  = DN_DONNEES_NUNAVUT_2005_2024["Diff_POR_CA"].autocorr(lag=1)

print("Autocorr PIB   :", auto_PIB)
print("Autocorr VIIRS :", auto_VIIRS)
print("Autocorr POR   :", auto_POR)


DN_DONNEES_NUNAVUT_2005_2024[
    ["Diff_PIB_REEL", "Diff_VIIRS_SUM_RESAMPL", "POR_CA"]
].corr()


#%% MODELE EMPLOI

#####################################################################################################
###############          CALCUL A LADE DE LA POPULATION
###########################################################################

DATA_PIB_DN_2009_2024 = DN_DONNEES_NUNAVUT_2005_2024[(DN_DONNEES_NUNAVUT_2005_2024["Annee"] >= 2009) & (DN_DONNEES_NUNAVUT_2005_2024["Annee"] <= 2024)]


POPULATION_DN_PIB_2009_2024 = (
    DATA_PIB_DN_2009_2024
        .merge(EMPLOI_MINE, on="Annee", how="left")
        .merge(POPULATION_2009_2024, on="Annee", how="left")
)


POPULATION_DN_PIB_2009_2024['PIB_TETE'] = (POPULATION_DN_PIB_2009_2024['PIB_REEL'] / POPULATION_DN_PIB_2009_2024['EMPL'])

POPULATION_DN_PIB_2009_2024['PIB_MBK'] = (POPULATION_DN_PIB_2009_2024['Total_Empl_MK'] * POPULATION_DN_PIB_2009_2024['PIB_TETE'])

POPULATION_DN_PIB_2009_2024['PIB_MEL'] = (POPULATION_DN_PIB_2009_2024['Total_Empl_MEL'] * POPULATION_DN_PIB_2009_2024['PIB_TETE'])

POPULATION_DN_PIB_2009_2024['PIB_HB'] = (POPULATION_DN_PIB_2009_2024['Total_Empl_HB'] * POPULATION_DN_PIB_2009_2024['PIB_TETE'])

POPULATION_DN_PIB_2009_2024['PIB_MR'] = (POPULATION_DN_PIB_2009_2024['Total_Empl_MR'] * POPULATION_DN_PIB_2009_2024['PIB_TETE'])

POPULATION_DN_PIB_2009_2024['PIB_BR'] = (POPULATION_DN_PIB_2009_2024['Total_Empl_BR'] * POPULATION_DN_PIB_2009_2024['PIB_TETE'])

POPULATION_DN_PIB_2009_2024['VAEM_RV_TETE']  = POPULATION_DN_PIB_2009_2024['PIB_MBK']  + POPULATION_DN_PIB_2009_2024['PIB_MEL'] + POPULATION_DN_PIB_2009_2024['PIB_HB'] + POPULATION_DN_PIB_2009_2024['PIB_BR'] + POPULATION_DN_PIB_2009_2024['PIB_MR']  


POPULATION_DN_PIB_2009_2024['TX_PIB_MBK'] = ((POPULATION_DN_PIB_2009_2024['Total_Empl_MK'] * POPULATION_DN_PIB_2009_2024['PIB_TETE'])/POPULATION_DN_PIB_2009_2024['PIB_REEL'])*100

POPULATION_DN_PIB_2009_2024['TX_PIB_MEL'] = ((POPULATION_DN_PIB_2009_2024['Total_Empl_MEL'] * POPULATION_DN_PIB_2009_2024['PIB_TETE'])/POPULATION_DN_PIB_2009_2024['PIB_REEL'])*100

POPULATION_DN_PIB_2009_2024['TX_PIB_HB'] = ((POPULATION_DN_PIB_2009_2024['Total_Empl_HB'] * POPULATION_DN_PIB_2009_2024['PIB_TETE'])/POPULATION_DN_PIB_2009_2024['PIB_REEL'])*100

POPULATION_DN_PIB_2009_2024['TX_PIB_MR'] = ((POPULATION_DN_PIB_2009_2024['Total_Empl_MR'] * POPULATION_DN_PIB_2009_2024['PIB_TETE'])/POPULATION_DN_PIB_2009_2024['PIB_REEL'])*100

POPULATION_DN_PIB_2009_2024['TX_PIB_BR'] = ((POPULATION_DN_PIB_2009_2024['Total_Empl_BR'] * POPULATION_DN_PIB_2009_2024['PIB_TETE'])/POPULATION_DN_PIB_2009_2024['PIB_REEL'])*100

POPULATION_DN_PIB_2009_2024['TX_VAEM_RV_TETE']  = POPULATION_DN_PIB_2009_2024['TX_PIB_MBK']  + POPULATION_DN_PIB_2009_2024['TX_PIB_MEL'] + POPULATION_DN_PIB_2009_2024['TX_PIB_HB'] + POPULATION_DN_PIB_2009_2024['TX_PIB_BR'] + POPULATION_DN_PIB_2009_2024['TX_PIB_MR']  




#%% REGRESSION LOG

###########################################################################################
                           ############ REGRESSION 
##########################################################################################                           

#################################### CREATION VARIABLE DE RETARD 
## 2. Créer les variables retardées
# Pour le retard d'une période (lag 1)
DN_DONNEES_NUNAVUT_2005_2024['POR_CA_lag1'] = DN_DONNEES_NUNAVUT_2005_2024['Diff_POR_CA'].shift(1)
DN_DONNEES_NUNAVUT_2005_2024['Log_POR_CA_lag1'] = DN_DONNEES_NUNAVUT_2005_2024['Diff_Log_POR_CA'].shift(1)

################################# SELECTION ANNEE 2005 A 2022
# Supprimer les lignes avec NaN dans les variables d'intérêt
DATA_PIB_DN_SELECT_LOG = DN_DONNEES_NUNAVUT_2005_2024.dropna(subset=['Log_PIB_REEL', 'Log_VIIRS_SUM_RESAMPL'])
DATA_PIB_DN_SELECT_DIFF = DN_DONNEES_NUNAVUT_2005_2024.dropna(subset=['Diff_PIB_REEL', 'Diff_VIIRS_SUM_RESAMPL'])
DATA_PIB_DN_SELECT_DIFF_LOG = DN_DONNEES_NUNAVUT_2005_2024.dropna(subset=['Diff_Log_PIB_REEL', 'Diff_Log_VIIRS_SUM_RESAMPL' ])





#################################################################################
######################### REGRESSSION LOG ##########################################

Log_POP_NUN = DATA_PIB_DN_SELECT_LOG["Log_POP"]
Log_DN = DATA_PIB_DN_SELECT_LOG[["Log_VIIRS_SUM_RESAMPL"]]
Log_PIB = DATA_PIB_DN_SELECT_LOG["Log_PIB_EN"]
Log_VAEMP = DATA_PIB_DN_SELECT_LOG["Log_VAEMP_EN"]
Log_POR = DATA_PIB_DN_SELECT_LOG["Log_POR_CA"]
POR = DATA_PIB_DN_SELECT_LOG["POR_CA"]
Log_PROD_MINE = DATA_PIB_DN_SELECT_LOG["Log_PROD_MINE_CAD_EN"]

POR_CT = sm.add_constant(POR)
Log_POR_CT = sm.add_constant(Log_POR)
Log_DN_CT = sm.add_constant(Log_DN)
Log_VAEMP_CT = sm.add_constant(Log_VAEMP)
Log_PROD_MINE_CT = sm.add_constant(Log_PROD_MINE)
Log_POP_NUN_CT = sm.add_constant(Log_POP_NUN)

MODEL_LOG_PIB_DN = sm.OLS(Log_PIB, Log_DN_CT).fit(cov_type='HC0')  # White
print("Méthode MCO (covariance robuste de White / HC0):")
print(MODEL_LOG_PIB_DN.summary())

MODEL_LOG_PIB_POP = sm.OLS(Log_PIB, Log_POP_NUN_CT).fit(cov_type='HC3')
print(MODEL_LOG_PIB_POP.summary())

MODEL_LOG_DN_POP = sm.OLS(Log_DN, Log_POP_NUN_CT).fit(cov_type='HC3')
print(MODEL_LOG_DN_POP.summary())

MODEL_LOG_PROD_DN = sm.OLS(Log_PROD_MINE, Log_DN_CT).fit(cov_type='HC3')
print(MODEL_LOG_PROD_DN.summary())

MODEL_LOG_PIB_PROD = sm.OLS(Log_PIB, Log_PROD_MINE_CT).fit(cov_type='HC3')
print(MODEL_LOG_PIB_PROD.summary())

MODEL_LOG_VAEMP_DN = sm.OLS(Log_VAEMP, Log_DN_CT).fit(cov_type='HC3')
print(MODEL_LOG_VAEMP_DN.summary())

# Méthode IV2SLS avec covariance robuste (HC, via linearmodels)
# exog = constante seule | endog = Log_DN | instrument = Log_POR
_exog_log = pd.DataFrame({'const': 1.0}, index=Log_PIB.index)

MODEL_IV_PIB_DN = IV2SLS_lm(
    dependent=Log_PIB,
    exog=_exog_log,
    endog=Log_DN,
    instruments=Log_POR.to_frame(name='Log_POR')
).fit(cov_type='robust')

print("\nMéthode IV2SLS (covariance robuste à l’hétéroscédasticité):")
print(MODEL_IV_PIB_DN.summary)
# Créer un DataFrame avec toutes les variables explicatives
#X = pd.DataFrame({
#    'Log_DN_CT': Log_DN_CT,
#    'Log_POP_NUN_CT': Log_POP_NUN_CT
#})

# Ajouter une constante (intercept)
#X = sm.add_constant(X)

#MODEL_LOG_PIB_DN = sm.OLS(Log_PIB, X).fit()
#

#print(MODEL_LOG_PIB_DN.summary())
# Matrice des variables explicatives : Log_DN + Log_PROD_MINE
Log_DN_PROD = DATA_PIB_DN_SELECT_LOG[["Log_VIIRS_SUM_RESAMPL", "Log_PROD_MINE_CAD_EN"]]
# Ajout de la constante
Log_DN_PROD_CT = sm.add_constant(Log_DN_PROD)
# Régression
MODEL_LOG_PIB_DN_PROD = sm.OLS(Log_PIB, Log_DN_PROD_CT).fit(cov_type='HC3')
# Résultats
print(MODEL_LOG_PIB_DN_PROD.summary())



#%% MODÈLE LOG ET IV
# =============================================================================
# 1. Définir les phases minières
# =============================================================================
VIIRS_MINE = DMSP_VIIRS_MINE_2000_2024[["Nom","Annee", "VIIRS_SUM_RESAMPL"]].copy()

VIIRS_MINE = compute_log_and_diff_lum(
    VIIRS_MINE, value_col="VIIRS_SUM_RESAMPL", group_col="Nom"
)

PHASES_MINES = {
    "Meadowbank": [
        {"phase": "exploration",   "debut": 2000, "fin": 2006},
        {"phase": "production",    "debut": 2007, "fin": None},
    ],
    "Baffinland": [
        {"phase": "exploration",   "debut": 2006, "fin": 2011},
        {"phase": "production",    "debut": 2012, "fin": None},
    ],
    "Meliadine": [
        {"phase": "exploration",   "debut": 2010, "fin": 2016},
        {"phase": "production",    "debut": 2017, "fin": None},
    ],
    "Hope Bay": [
        {"phase": "exploration",   "debut": 2008, "fin": 2012},
        {"phase": "production",    "debut": 2013, "fin": 2021},
        {"phase": "exploration",   "debut": 2022, "fin": None},
    ],
    # Mines historiques actives dans la période 2000-2024
    "Lupin": [
        {"phase": "production",    "debut": 2000, "fin": 2004},
        {"phase": "exploration",   "debut": 2005, "fin": None},
    ],
    "Jericho": [
        {"phase": "exploration",   "debut": 2000, "fin": 2005},
        {"phase": "production",    "debut": 2006, "fin": 2008},
    ],
    # Projets majeurs récents
    "Goose (Back River)": [
        {"phase": "exploration",   "debut": 2009, "fin": 2024},
        {"phase": "production",    "debut": 2025, "fin": None},
    ],
}

# =============================================================================
# 2. Convertir le dictionnaire en DataFrame
# =============================================================================

rows_phases = []
for mine, phases in PHASES_MINES.items():
    for p in phases:
        rows_phases.append({
            "mine": mine,
            "phase": p["phase"],
            "debut": p["debut"],
            "fin": p["fin"]
        })

table_phases = pd.DataFrame(rows_phases)
table_phases["debut"] = table_phases["debut"].astype(int)
table_phases["fin"]   = table_phases["fin"].astype("Int64")

# =============================================================================
# 3. Développer les phases par année
# =============================================================================

annee_max = int(DATA_PIB_DN_SELECT_LOG["Annee"].max())

rows_time = []
for _, r in table_phases.iterrows():
    debut = int(r["debut"])
    fin   = int(r["fin"]) if pd.notna(r["fin"]) else annee_max

    for an in range(debut, fin + 1):
        rows_time.append({
            "Annee": an,
            "mine":  r["mine"],
            "phase": r["phase"]
        })

table_time = pd.DataFrame(rows_time)

# =============================================================================
# 4. Créer les comptages (nombre de mines par phase et par année)
# =============================================================================

table_dummies = pd.get_dummies(table_time["phase"], prefix="phase", dtype=int)
table_time_full = pd.concat([table_time, table_dummies], axis=1)

# Base annuelle agrégée → comptage : nombre de mines en exploration/production
table_time_year = (
    table_time_full
    .groupby("Annee", as_index=False)[["phase_exploration", "phase_production"]]
    .sum()
    .rename(columns={
        "phase_exploration": "n_mines_exploration",
        "phase_production":  "n_mines_production"
    })
)

# Base mine-année → pour chaque mine, 0 ou 1 selon sa phase propre
table_time_mine = (
    table_time_full
    .groupby(["Annee", "mine"], as_index=False)[["phase_exploration", "phase_production"]]
    .sum()
    .rename(columns={
        "phase_exploration": "n_mines_exploration",
        "phase_production":  "n_mines_production"
    })
)

# =============================================================================
# 5. Fusion avec la base principale
# =============================================================================

table_MCO = DATA_PIB_DN_SELECT_LOG.copy()
table_MCO = table_MCO.merge(table_time_year, on="Annee", how="left")

for col in ["n_mines_exploration", "n_mines_production"]:
    if col not in table_MCO.columns:
        table_MCO[col] = 0
    table_MCO[col] = table_MCO[col].fillna(0)

# =============================================================================
# 6. Définir Y et X
# =============================================================================

table_MCO["Log_PIB"] = table_MCO["Log_PIB_EN"]

y = table_MCO["Log_PIB"]
X = table_MCO[["Log_VIIRS_SUM_RESAMPL", "n_mines_exploration", "n_mines_production"]]
X = sm.add_constant(X, has_constant="add")

# =============================================================================
# 7. Nettoyer
# =============================================================================

data_model = pd.concat([y, X], axis=1)
data_model = data_model.replace([np.inf, -np.inf], np.nan).dropna()

y_clean = data_model["Log_PIB"]
X_clean = data_model.drop(columns=["Log_PIB"])

# =============================================================================
# 8. Estimation
# =============================================================================

MODEL_LOG_PIB_DN_PH = sm.OLS(y_clean, X_clean).fit(cov_type="HAC", cov_kwds={"maxlags": 2})

print("Méthode MCO — comptage de mines par phase (covariance robuste HC0) :")
print(MODEL_LOG_PIB_DN_PH.summary())

# =============================================================================
# 9. Préparation de VIIRS_MINE
# =============================================================================

nom_to_mine = {
    "Mine_Baffinland": "Baffinland",
    "Mine_Meadowbank": "Meadowbank",
    "Mine_Meliadine":  "Meliadine",
    "Mine_Hope_Bay":   "Hope Bay",
}

VIIRS_MINE["mine"] = VIIRS_MINE["Nom"].map(nom_to_mine)

# Supprimer les anciennes colonnes si elles existent déjà
cols_phase = ["n_mines_exploration", "n_mines_production",
              "phase_exploration",   "phase_production"]   # nettoyage défensif
VIIRS_MINE = VIIRS_MINE.drop(columns=[c for c in cols_phase if c in VIIRS_MINE.columns])

VIIRS_MINE = VIIRS_MINE.merge(
    table_time_mine[["Annee", "mine", "n_mines_exploration", "n_mines_production"]],
    on=["Annee", "mine"],
    how="left"
)

for col in ["n_mines_exploration", "n_mines_production"]:
    VIIRS_MINE[col] = VIIRS_MINE[col].fillna(0)

# =============================================================================
# 10. Prédiction sur VIIRS_MINE
# =============================================================================

X_pred = VIIRS_MINE[[
    "Log_VIIRS_SUM_RESAMPL",
    "n_mines_exploration",
    "n_mines_production",
]].copy()

X_pred = X_pred.replace([np.inf, -np.inf], np.nan)
X_pred = sm.add_constant(X_pred, has_constant="add")

mask_pred = X_pred.notna().all(axis=1)

VIIRS_MINE["PRED_PIB_LOG_MINE"] = np.nan
VIIRS_MINE.loc[mask_pred, "PRED_PIB_LOG_MINE"] = MODEL_LOG_PIB_DN_PH.predict(X_pred.loc[mask_pred])

sigma2 = MODEL_LOG_PIB_DN_PH.mse_resid
VIIRS_MINE["PRED_PIB_MINE_LOG"] = np.exp(
    VIIRS_MINE["PRED_PIB_LOG_MINE"] + sigma2 / 2
)

PRED_PIB_MINE = VIIRS_MINE[['Nom', 'Annee', 'PRED_PIB_MINE_LOG']]





# =============================================================================
# MODÈLE IV — observabilité (instrument composite) de Log_VIIRS_SUM_RESAMPL
# =============================================================================
# Hypothèse : observabilite → Log_VIIRS_SUM_RESAMPL (mesure satellite
#               favorisée par les conditions d'observation : part claire ×
#               part de l'année en nuit astronomique)
#             observabilite ✗→ Log_PIB  (instrument valide — exogène)
#
# Construction de l'observabilité (fichier instrument_composite_nunavut_annuel) :
#   observabilite = (1 - cloud_cover) × part_nuit × h_nuit_astro / 8760
#                 = h_obs / heures_an_total
#
# Signe attendu en première étape : β_observabilité > 0
#   (contrairement à cloud_cover où le signe attendu était négatif).
# =============================================================================


# ---------------------------------------------------------------------------
# 1. Fusionner instrument_composite avec le panel principal
# ---------------------------------------------------------------------------

table_iv = table_MCO.merge(
    instrument_composite[["Annee", "observabilite"]], on="Annee", how="left"
)

table_iv["Log_PIB"] = table_iv["Log_PIB_EN"]

# ---------------------------------------------------------------------------
# 2. Construire les matrices (variables)
# ---------------------------------------------------------------------------

cols_needed = ["Log_PIB", "Log_VIIRS_SUM_RESAMPL", "observabilite",
               "n_mines_exploration", "n_mines_production"]
data_iv = (
    table_iv[cols_needed]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

dep    = data_iv["Log_PIB"]
endog  = data_iv[["Log_VIIRS_SUM_RESAMPL"]]                                  # variable endogène
exog   = sm.add_constant(                                      # exogènes incluses
    data_iv[["n_mines_exploration", "n_mines_production"]],
    has_constant="add"
)
instr  = sm.add_constant(                                      # instruments externes
    data_iv[["observabilite"]],
    has_constant="add"
)

# ---------------------------------------------------------------------------
# 3. Première étape  (diagnostic de pertinence)
# ---------------------------------------------------------------------------

first_stage = sm.OLS(
    data_iv["Log_VIIRS_SUM_RESAMPL"],
    pd.concat([exog, data_iv[["observabilite"]]], axis=1)
).fit(cov_type="HAC", cov_kwds={"maxlags": 2})

print("=== PREMIÈRE ÉTAPE : Log_VIIRS_SUM_RESAMPL ~ observabilite + contrôles ===")
print(first_stage.summary())
print(f"\nStatistique F (instrument) : {first_stage.fvalue:.3f}  "
      f"(seuil de pertinence usuel : F > 10)")

# ---------------------------------------------------------------------------
# 4. Modèle IV-2SLS  — CORRECTION
# ---------------------------------------------------------------------------
# ⚠️ linearmodels combine exog + instruments en interne.
#    → NE PAS ajouter de constante dans instruments (elle est déjà dans exog)

MODEL_IV = IV2SLS(
    dependent   = dep,
    exog        = exog,                          # const + n_mines_* (inchangé)
    endog       = endog,                         # Log_VIIRS_SUM_RESAMPL
    instruments = data_iv[["observabilite"]],    # ← observabilité, sans const
).fit(cov_type="kernel")

print("\n=== DEUXIÈME ÉTAPE : IV-2SLS ===")
print(MODEL_IV.summary)

# ---------------------------------------------------------------------------
# 5. Test de Hausman (endogénéité)  — Wu-Hausman inclus dans linearmodels
# ---------------------------------------------------------------------------

print("\n=== TEST D'ENDOGÉNÉITÉ (Wu-Hausman) ===")
print(MODEL_IV.wu_hausman())   # H0 : Log_VIIRS_SUM_RESAMPL est exogène

# ---------------------------------------------------------------------------
# 6. Prédiction sur VIIRS_MINE avec les coefficients IV
# ---------------------------------------------------------------------------

VIIRS_MINE = VIIRS_MINE.merge(
    instrument_composite[["Annee", "observabilite"]], on="Annee", how="left"
)

X_pred_iv = VIIRS_MINE[[
    "Log_VIIRS_SUM_RESAMPL",
    "n_mines_exploration",
    "n_mines_production",
]].copy()

X_pred_iv = X_pred_iv.replace([np.inf, -np.inf], np.nan)
X_pred_iv = sm.add_constant(X_pred_iv, has_constant="add")

# Réordonner pour correspondre à l'ordre du modèle IV (const, mines, Log_VIIRS_SUM_RESAMPL)
X_pred_iv = X_pred_iv[["const", "n_mines_exploration",
                        "n_mines_production", "Log_VIIRS_SUM_RESAMPL"]]

mask_iv = X_pred_iv.notna().all(axis=1)

VIIRS_MINE["PRED_PIB_MINE_IV_LOG"] = np.nan
VIIRS_MINE.loc[mask_iv, "PRED_PIB_MINE_IV_LOG"] = (
    MODEL_IV.params["const"]
    + MODEL_IV.params["n_mines_exploration"] * X_pred_iv.loc[mask_iv, "n_mines_exploration"]
    + MODEL_IV.params["n_mines_production"]  * X_pred_iv.loc[mask_iv, "n_mines_production"]
    + MODEL_IV.params["Log_VIIRS_SUM_RESAMPL"]              * X_pred_iv.loc[mask_iv, "Log_VIIRS_SUM_RESAMPL"]
)

sigma2_iv = MODEL_IV.resids.var()
VIIRS_MINE["PRED_PIB_MINE_LOG_IV"] = np.exp(
    VIIRS_MINE["PRED_PIB_MINE_IV_LOG"] + sigma2_iv / 2
)
PRED_PIB_MINE_IV = VIIRS_MINE[['Nom', 'Annee', 'PRED_PIB_MINE_LOG_IV']]

print("\n=== Prédictions IV (extrait) ===")
print(VIIRS_MINE[["Nom", "Annee", "PRED_PIB_MINE_LOG_IV"]].head(10).to_string())

PIB_ANNUEL_IV = PRED_PIB_MINE_IV.groupby('Annee')['PRED_PIB_MINE_LOG_IV'].sum().reset_index()
PIB_ANNUEL_LOG = PRED_PIB_MINE.groupby('Annee')['PRED_PIB_MINE_LOG'].sum().reset_index()
PIB_PAR_MINE = PRED_PIB_MINE.groupby('Nom')['PRED_PIB_MINE_LOG'].mean().reset_index()

VIIRS_MINE_2005_2024 = VIIRS_MINE[(VIIRS_MINE["Annee"] >= 2005) & (VIIRS_MINE["Annee"] <= 2024)]
PIB_REEL_SEL = DN_DONNEES_NUNAVUT_2005_2024[["Annee", "PIB_EN"]]

# Fusion sur l'année
TABLE_VIIRS_MINE_2005_2024 = VIIRS_MINE_2005_2024.merge(PIB_REEL_SEL, on="Annee", how="left")

TABLE_VIIRS_MINE_2005_2024["TX_PRED_PIB_MINE_LOG"] = (TABLE_VIIRS_MINE_2005_2024["PRED_PIB_MINE_LOG"] / TABLE_VIIRS_MINE_2005_2024["PIB_EN"])*100

TABLE_VIIRS_MINE_2005_2024["TX_PRED_PIB_MINE_LOG_IV"] = (TABLE_VIIRS_MINE_2005_2024["PRED_PIB_MINE_LOG_IV"] / TABLE_VIIRS_MINE_2005_2024["PIB_EN"])*100




#%% REGRESION TAUX DE CROISSANCE

##################################################################################
######################### REGRESION TAUX DE CROISSANCE #####################################
Diff_Log_POP_NUN = DATA_PIB_DN_SELECT_DIFF_LOG["Diff_Log_POP"]
Diff_Log_DN = DATA_PIB_DN_SELECT_DIFF_LOG[["Diff_Log_VIIRS_SUM_RESAMPL"]]
Diff_Log_PIB = DATA_PIB_DN_SELECT_DIFF_LOG["Diff_Log_PIB_REEL"]
Diff_Log_VAEMP = DATA_PIB_DN_SELECT_DIFF_LOG["Diff_Log_VAEMP_EN"]
Diff_Log_POR = DATA_PIB_DN_SELECT_DIFF_LOG["Diff_Log_POR_CA"]
Diff_POR = DATA_PIB_DN_SELECT_DIFF_LOG["POR_CA"]
Diff_Log_PROD_MINE = DATA_PIB_DN_SELECT_DIFF_LOG["Diff_Log_PROD_MINE_CAD_EN"]

POR_CT = sm.add_constant(POR)
Diff_Log_POR_CT = sm.add_constant(Diff_Log_POR)
Diff_Log_DN_CT = sm.add_constant(Diff_Log_DN)
Diff_Log_VAEMP_CT = sm.add_constant(Diff_Log_VAEMP)
Diff_Log_PROD_MINE_CT = sm.add_constant(Diff_Log_PROD_MINE)
Diff_Log_POP_NUN_CT = sm.add_constant(Diff_Log_POP_NUN)

MODEL_Diff_LOG_PIB_DN = sm.OLS(Diff_Log_PIB, Diff_Log_DN_CT).fit(cov_type='HC3')
print(MODEL_Diff_LOG_PIB_DN.summary())

#MODEL_Diff_LOG_VAEMP_DN = sm.OLS(Diff_Log_VAEMP, Diff_Log_DN_CT).fit(cov_type='HC3')
#print(MODEL_Diff_LOG_VAEMP_DN.summary())

#MODEL_Diff_LOG_PROD_DN = sm.OLS(Diff_Log_DN, Diff_Log_PROD_MINE_CT).fit(cov_type='HC3')
#print(MODEL_Diff_LOG_PROD_DN.summary())

MODEL_Diff_LOG_PIB_PROD = sm.OLS(Diff_Log_PIB, Diff_Log_PROD_MINE_CT).fit(cov_type='HC3')
print(MODEL_Diff_LOG_PIB_PROD.summary())

MODEL_Diff_LOG_POP_NUNAVUT = sm.OLS(Diff_Log_PIB, Diff_Log_POP_NUN_CT).fit(cov_type='HC3')
print(MODEL_Diff_LOG_POP_NUNAVUT.summary())
# Méthode IV2SLS avec covariance robuste (HC, via linearmodels)
_exog_dlog = pd.DataFrame({'const': 1.0}, index=Diff_Log_PIB.index)
MODEL_IV_Diff_Log_PIB_DN = IV2SLS_lm(
    Diff_Log_PIB, _exog_dlog, Diff_Log_DN, Diff_Log_POR.to_frame()
).fit(cov_type='robust')
print("Méthode IV2SLS (covariance robuste HC):")
print(MODEL_IV_Diff_Log_PIB_DN.summary)





#%% REGRESSION DIFF
###########################################################################################
######################## REGRESSION DIFF ##########################
                           
Diff_DN = DATA_PIB_DN_SELECT_DIFF[["Diff_VIIRS_SUM_RESAMPL"]]
Diff_PIB = DATA_PIB_DN_SELECT_DIFF["Diff_PIB_REEL"]
Diff_VAEMP = DATA_PIB_DN_SELECT_DIFF["Diff_VAEMP_EN"]
Diff_POR = DATA_PIB_DN_SELECT_DIFF["Diff_POR_CA"]
Diff_PROD_MINE = DATA_PIB_DN_SELECT_DIFF["Diff_PROD_MINE_CAD_EN"]

POR_CT = sm.add_constant(POR)
Diff_POR_CT = sm.add_constant(Diff_POR)
Diff_DN_CT = sm.add_constant(Diff_DN)
Diff_VAEMP_CT = sm.add_constant(Diff_VAEMP)
Diff_PROD_MINE_CT = sm.add_constant(Diff_PROD_MINE)

MODEL_Diff_PIB_DN = sm.OLS(Diff_PIB, Diff_DN_CT).fit(cov_type='HC3')
print(MODEL_Diff_PIB_DN.summary())

MODEL_Diff_VAEMP_DN = sm.OLS(Diff_VAEMP, Diff_DN_CT).fit(cov_type='HC3')
print(MODEL_Diff_VAEMP_DN.summary())

MODEL_Diff_PROD_DN = sm.OLS(Diff_DN, Diff_PROD_MINE_CT).fit(cov_type='HC3')
print(MODEL_Diff_PROD_DN.summary())

MODEL_Diff_PIB_PROD = sm.OLS(Diff_PIB, Diff_PROD_MINE_CT).fit(cov_type='HC3')
print(MODEL_Diff_PIB_PROD.summary())

# Méthode IV2SLS avec covariance robuste (HC, via linearmodels)
_exog_diff = pd.DataFrame({'const': 1.0}, index=Diff_PIB.index)
MODEL_IV_Diff_PIB_DN = IV2SLS_lm(
    Diff_PIB, _exog_diff, Diff_DN, Diff_POR.to_frame()
).fit(cov_type='robust')
print("Méthode IV2SLS (covariance robuste HC):")
print(MODEL_IV_Diff_PIB_DN.summary)

#%% TABLEAU RÉGRÉSSION
#
stargazer_PIB_Log = Stargazer([MODEL_LOG_PIB_DN, MODEL_LOG_PIB_DN_PH,
                               first_stage, MODEL_IV])
stargazer_PIB_Log.title("Régression du PIB du Nunavut sur la luminosité nocturne -- MCO et 2SLS")
stargazer_PIB_Log.custom_columns(
    ["OLS simple", "OLS phases", "IV 1re étape", "IV 2e étape"],
    [1, 1, 1, 1]
)
stargazer_PIB_Log.rename_covariates({
    "const": "Const",
    "Log_VIIRS_SUM_RESAMPL": "Log(DN)",
    "n_mines_exploration": "Mines expl.",
    "n_mines_production":  "Mines prod.",
    "observabilite":       "Observabilité"
})
stargazer_PIB_Log.dependent_variable_name("Log(PIB) / Log(DL) pour 1re étape")
stargazer_PIB_Log.covariate_order([
    "const",
    "Log_VIIRS_SUM_RESAMPL",
    "n_mines_exploration",
    "n_mines_production",
    "observabilite",
])
stargazer_PIB_Log.add_custom_notes([
    "Mines expl. : nombre de mines en exploration.",
    "Mines prod. : nombre de mines en production.",
    "DN : données de luminosité.",
])
# Chemin complet du fichier
file_path = os.path.join(os.path.join(baseDir, "Tableau"), "comparaison_modeles.html")

# Enregistrement
with open(file_path, "w", encoding="utf-8") as f:
    f.write(stargazer_PIB_Log.render_html())

print("Fichier enregistré ici :")
print(file_path)

# Export LaTeX dans tab_dir — ajusté en \footnotesize pour tenir dans la largeur
file_path_tex = os.path.join(tab_dir, "comparaison_modeles.tex")
_latex_src = stargazer_PIB_Log.render_latex()
# Injecter \footnotesize avant \begin{tabular} pour réduire la taille
_latex_src = _latex_src.replace(
    r'\begin{tabular}',
    r'\footnotesize' + '\n' + r'\begin{tabular}'
)
_latex_src = _latex_src.replace(
    r'\end{tabular}',
    r'\end{tabular}'
)
# Injection du label pour permettre les renvois \ref{} dans Overleaf
_latex_src = _latex_src.replace(
    r'\end{table}',
    r'  \label{tab:comparaison_modeles}' + '\n' + r'\end{table}'
)
with open(file_path_tex, "w", encoding="utf-8") as f:
    f.write(_latex_src)
print("Fichier LaTeX enregistré ici :")
print(file_path_tex)



#% TABLEAU RÉGRESSION — TAUX DE CROISSANCE
###############################################################################
# Régression MCO sur le taux de croissance du PIB (Δlog)
###############################################################################

stargazer_PIB_Diff = Stargazer([MODEL_Diff_LOG_PIB_DN])
stargazer_PIB_Diff.title("Croissance du PIB du Nunavut et luminosité nocturne -- MCO")
stargazer_PIB_Diff.custom_columns(["MCO"], [1])
stargazer_PIB_Diff.rename_covariates({
    "const": "Const",
    "Diff_Log_VIIRS_SUM_RESAMPL": "ΔLog(DN)",
})
stargazer_PIB_Diff.dependent_variable_name("ΔLog(PIB)")

# Export HTML
file_path_diff_html = os.path.join(tab_dir, "comparaison_modeles_croissance.html")
with open(file_path_diff_html, "w", encoding="utf-8") as f:
    f.write(stargazer_PIB_Diff.render_html())
print("Fichier HTML enregistré ici :")
print(file_path_diff_html)

# Export LaTeX
file_path_diff_tex = os.path.join(tab_dir, "comparaison_modeles_croissance.tex")
_latex_diff = stargazer_PIB_Diff.render_latex()
# Injection du label pour permettre les renvois \ref{} dans Overleaf
_latex_diff = _latex_diff.replace(
    r'\end{table}',
    r'  \label{tab:comparaison_modeles_croissance}' + '\n' + r'\end{table}'
)
with open(file_path_diff_tex, "w", encoding="utf-8") as f:
    f.write(_latex_diff)
print("Fichier LaTeX enregistré ici :")
print(file_path_diff_tex)

#%% TESTS DIAGNOSTIQUES
###############################################################################
# Tests diagnostiques sur les deux modèles principaux (Log-Log)
###############################################################################

import warnings
warnings.filterwarnings("ignore")

# ── Résidus OLS (MODEL_LOG_PIB_DN_PH) ────────────────────────────────────────
resid_ols  = MODEL_LOG_PIB_DN_PH.resid
exog_ols   = MODEL_LOG_PIB_DN_PH.model.exog   # matrice [const, Log_DN, n_mines_expl, n_mines_prod]
nobs_ols   = int(MODEL_LOG_PIB_DN_PH.nobs)

# ── Résidus IV (MODEL_IV) ────────────────────────────────────────────────────
resid_iv   = MODEL_IV.resids.values
nobs_iv    = len(resid_iv)

rows = []   # liste de dict → DataFrame final

# ─────────────────────────────────────────────────────────────────────────────
# A)  TESTS SUR (MCO-HC3)
# ─────────────────────────────────────────────────────────────────────────────

# 1. Normalité — Jarque-Bera
jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(resid_ols)
rows.append({
    "Modèle"      : "MCO (Log-Log)",
    "Test"        : "Jarque-Bera (normalité)",
    "Statistique" : round(float(jb_stat), 4),
    "p-valeur"    : round(float(jb_pval), 4),
    "H0"          : "Résidus normaux",
    "Décision (5%)": "Rejet H0" if jb_pval < 0.05 else "Non-rejet H0"
})

# 2. Hétéroscédasticité — Breusch-Pagan
bp_lm, bp_pval, bp_fstat, bp_fpval = het_breuschpagan(resid_ols, exog_ols)
rows.append({
    "Modèle"      : "MCO (Log-Log)",
    "Test"        : "Breusch-Pagan (hétérosc.)",
    "Statistique" : round(float(bp_lm), 4),
    "p-valeur"    : round(float(bp_pval), 4),
    "H0"          : "Homoscédasticité",
    "Décision (5%)": "Rejet H0" if bp_pval < 0.05 else "Non-rejet H0"
})

# 3. Hétéroscédasticité — White
wh_lm, wh_pval, wh_fstat, wh_fpval = het_white(resid_ols, exog_ols)
rows.append({
    "Modèle"      : "MCO (Log-Log)",
    "Test"        : "White (hétérosc.)",
    "Statistique" : round(float(wh_lm), 4),
    "p-valeur"    : round(float(wh_pval), 4),
    "H0"          : "Homoscédasticité",
    "Décision (5%)": "Rejet H0" if wh_pval < 0.05 else "Non-rejet H0"
})

# 4. Autocorrélation — Durbin-Watson
dw_stat = durbin_watson(resid_ols)
rows.append({
    "Modèle"      : "MCO (Log-Log)",
    "Test"        : "Durbin-Watson (autocorr.)",
    "Statistique" : round(float(dw_stat), 4),
    "p-valeur"    : "n.d.",
    "H0"          : "Pas d'autocorr. (DW approx. 2)",
    "Décision (5%)": "Suspicion" if (dw_stat < 1.5 or dw_stat > 2.5) else "Non-rejet H0"
})

# 5. Autocorrélation — Breusch-Godfrey (lag 1)
bg_lm, bg_pval, bg_fstat, bg_fpval = acorr_breusch_godfrey(MODEL_LOG_PIB_DN_PH, nlags=1)
rows.append({
    "Modèle"      : "MCO (Log-Log)",
    "Test"        : "Breusch-Godfrey (autocorr., lag 1)",
    "Statistique" : round(float(bg_lm), 4),
    "p-valeur"    : round(float(bg_pval), 4),
    "H0"          : "Pas d'autocorr.",
    "Décision (5%)": "Rejet H0" if bg_pval < 0.05 else "Non-rejet H0"
})

# 6. Multicolinéarité — VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor
try:
    _X_vif = pd.DataFrame(exog_ols, columns=MODEL_LOG_PIB_DN_PH.model.exog_names)
    for _i, _col in enumerate(_X_vif.columns):
        if _col == "const":
            continue
        _vif = variance_inflation_factor(_X_vif.values, _i)
        if _vif > 10:
            _dec = "Multicolinéarité forte"
        elif _vif > 5:
            _dec = "À surveiller"
        else:
            _dec = "Pas de multicolinéarité"
        rows.append({
            "Modèle"      : "MCO (Log-Log)",
            "Test"        : f"VIF — {_col}",
            "Statistique" : round(float(_vif), 4),
            "p-valeur"    : "n.d.",
            "H0"          : "Pas de multicolin. (VIF < 10)",
            "Décision (5%)": _dec
        })
except Exception as _e:
    rows.append({
        "Modèle"      : "MCO (Log-Log)",
        "Test"        : "VIF (multicolinéarité)",
        "Statistique" : "n.d.",
        "p-valeur"    : "n.d.",
        "H0"          : "Pas de multicolin. (VIF < 10)",
        "Décision (5%)": f"Échec : {_e}"
    })

# 7. Spécification — RESET de Ramsey
try:
    reset_res  = linear_reset(MODEL_LOG_PIB_DN_PH, power=2, use_f=True)
    reset_stat = float(reset_res.statistic)
    reset_pval = float(reset_res.pvalue)
except Exception:
    reset_stat, reset_pval = float("nan"), float("nan")
rows.append({
    "Modèle"      : "MCO (Log-Log)",
    "Test"        : "RESET Ramsey (spécification)",
    "Statistique" : round(reset_stat, 4),
    "p-valeur"    : round(reset_pval, 4) if not np.isnan(reset_pval) else "n.d.",
    "H0"          : "Pas de non-linéarité omise",
    "Décision (5%)": "Rejet H0" if (not np.isnan(reset_pval) and reset_pval < 0.05) else "Non-rejet H0"
})

# ─────────────────────────────────────────────────────────────────────────────
# B)  TESTS SUR  (IV-2SLS)
# ─────────────────────────────────────────────────────────────────────────────

# 7. Instruments faibles — F de première étape
fs_fstat, fs_fpval = float("nan"), float("nan")
try:
    fs_diag = MODEL_IV.first_stage.diagnostics
    # Essai 1 : colonnes = noms des endogènes, index = noms des stats
    _col = fs_diag.columns[0]
    try:
        fs_fstat = float(fs_diag.loc["f.stat", _col])
        fs_fpval = float(fs_diag.loc["f.pval", _col])
    except KeyError:
        # Essai 2 : lignes = noms des endogènes, colonnes = noms des stats
        fs_fstat = float(fs_diag.loc[_col, "f.stat"])
        fs_fpval = float(fs_diag.loc[_col, "f.pval"])
except Exception:
    pass

# Fallback : reprendre le first_stage déjà estimé (Log_DN ~ const + n_mines_* + observabilite)
if np.isnan(fs_fstat):
    try:
        fs_fstat = float(first_stage.fvalue)
        fs_fpval = float(first_stage.f_pvalue)
    except Exception:
        pass

rows.append({
    "Modèle"      : "IV-2SLS (Log-Log)",
    "Test"        : "F première étape (instruments faibles)",
    "Statistique" : round(fs_fstat, 4) if not np.isnan(fs_fstat) else "n.d.",
    "p-valeur"    : round(fs_fpval, 4) if not np.isnan(fs_fpval) else "n.d.",
    "H0"          : "Instruments faibles (F < 10)",
    "Décision (5%)": "Instruments forts" if (not np.isnan(fs_fstat) and fs_fstat >= 10) else "Instruments faibles"
})

# 8. Endogénéité — Wu-Hausman (Durbin)
try:
    wh_test  = MODEL_IV.wu_hausman()
    wh_s     = float(wh_test.stat)
    wh_p     = float(wh_test.pval)
except Exception:
    try:
        wh_test  = MODEL_IV.durbin_wu_hausman()
        wh_s     = float(wh_test.stat)
        wh_p     = float(wh_test.pval)
    except Exception:
        wh_s, wh_p = float("nan"), float("nan")
rows.append({
    "Modèle"      : "IV-2SLS (Log-Log)",
    "Test"        : "Wu-Hausman (endogénéité)",
    "Statistique" : round(wh_s, 4) if not np.isnan(wh_s) else "n.d.",
    "p-valeur"    : round(wh_p, 4) if not np.isnan(wh_p) else "n.d.",
    "H0"          : "Variable exogène (MCO consistent)",
    "Décision (5%)": "Rejet H0" if (not np.isnan(wh_p) and wh_p < 0.05) else "Non-rejet H0"
})

# 9. Normalité résidus IV — Jarque-Bera
jb_iv_stat, jb_iv_pval, _, _ = jarque_bera(resid_iv)
rows.append({
    "Modèle"      : "IV-2SLS (Log-Log)",
    "Test"        : "Jarque-Bera (normalité résidus IV)",
    "Statistique" : round(float(jb_iv_stat), 4),
    "p-valeur"    : round(float(jb_iv_pval), 4),
    "H0"          : "Résidus normaux",
    "Décision (5%)": "Rejet H0" if jb_iv_pval < 0.05 else "Non-rejet H0"
})

# 10. Autocorrélation résidus IV — Durbin-Watson
dw_iv = durbin_watson(resid_iv)
rows.append({
    "Modèle"      : "IV-2SLS (Log-Log)",
    "Test"        : "Durbin-Watson (autocorr. résidus IV)",
    "Statistique" : round(float(dw_iv), 4),
    "p-valeur"    : "n.d.",
    "H0"          : "Pas d'autocorr. (DW approx. 2)",
    "Décision (5%)": "Suspicion" if (dw_iv < 1.5 or dw_iv > 2.5) else "Non-rejet H0"
})

# ─────────────────────────────────────────────────────────────────────────────
# C)  TABLEAU FINAL
# ─────────────────────────────────────────────────────────────────────────────
TABLE_TESTS_DIAG = pd.DataFrame(rows)

print("\n" + "="*80)
print("TABLEAU DES TESTS DIAGNOSTIQUES — MODEL_LOG_PIB_DN_PH & MODEL_IV")
print("="*80)
print(TABLE_TESTS_DIAG.to_string(index=False))
print("="*80 + "\n")

# Export Excel
_path_tests = os.path.join(tab_dir, "tests_diagnostiques_modeles_log.xlsx")
with pd.ExcelWriter(_path_tests, engine="openpyxl") as _writer:
    TABLE_TESTS_DIAG.to_excel(_writer, sheet_name="Tests_Diag", index=False)
    _ws = _writer.sheets["Tests_Diag"]
    # Largeur automatique des colonnes
    for _col in _ws.columns:
        _max = max(len(str(_cell.value)) if _cell.value else 0 for _cell in _col)
        _ws.column_dimensions[_col[0].column_letter].width = min(_max + 4, 50)
print(f"Tableau tests diagnostiques enregistré : {_path_tests}")

# Export CSV (optionnel)
_path_tests_csv = os.path.join(tab_dir, "tests_diagnostiques_modeles_log.csv")
TABLE_TESTS_DIAG.to_csv(_path_tests_csv, index=False, sep=";", encoding="utf-8-sig")
print(f"CSV tests diagnostiques enregistré    : {_path_tests_csv}")

# Export LaTeX (ajusté en \footnotesize)
_path_tests_tex = os.path.join(tab_dir, "tests_diagnostiques_modeles_log.tex")
_latex_tests = TABLE_TESTS_DIAG.to_latex(
    index=False,
    escape=True,
    caption="Tests diagnostiques -- MODEL\\_LOG\\_PIB\\_DN\\_PH et MODEL\\_IV",
    label="tab:tests_diag_modeles_log",
    column_format="l" * TABLE_TESTS_DIAG.shape[1],
)
_latex_tests = _latex_tests.replace(
    r'\begin{tabular}',
    r'\footnotesize' + '\n' + r'\begin{tabular}'
)
_latex_tests = _latex_tests.replace(
    r'\end{tabular}',
    r'\end{tabular}'
)
with open(_path_tests_tex, "w", encoding="utf-8") as _f:
    _f.write(_latex_tests)
print(f"LaTeX tests diagnostiques enregistré  : {_path_tests_tex}")



#%% CALCUL VAEMP CORRIGER
# ============================================================
# TABLEAU DN MINING — sommes annuelles toutes mines
# ============================================================
somme_annuelle_totale = (
    DMSP_VIIRS_MINE_VILLE_2000_2024
    .groupby('Annee')['VIIRS_SUM_RESAMPL']
    .sum()
    .reset_index()
)
somme_annuelle_totale.columns = ['Annee', 'VIIRS_SUM_RESAMPL_MINING']

TABLEAU_DN_MINING = somme_annuelle_totale[
    (somme_annuelle_totale['Annee'] >= 2005) &
    (somme_annuelle_totale['Annee'] <= 2024)
].copy().sort_values('Annee').reset_index(drop=True)

print("Sommes annuelles totales VIIRS (2005-2024):")
print(TABLEAU_DN_MINING)

# Log et Diff_Log
TABLEAU_DN_MINING['Log_VIIRS_SUM_RESAMPL_MINING'] = np.log(
    TABLEAU_DN_MINING['VIIRS_SUM_RESAMPL_MINING'].replace(0, np.nan)
)

TABLEAU_DN_MINING['Diff_Log_VIIRS_SUM_RESAMPL_MINING'] = (
    TABLEAU_DN_MINING['Log_VIIRS_SUM_RESAMPL_MINING'].diff()
)

# ============================================================
# RÉGRESSIONS — VAEMP
# ============================================================

# Différences logarithmiques
Diff_Log_vaemp_of  = DN_DONNEES_NUNAVUT_2005_2024["Diff_Log_VAEMP_EN"]
Diff_Log_mining    = TABLEAU_DN_MINING["Diff_Log_VIIRS_SUM_RESAMPL_MINING"]
mask_v             = Diff_Log_vaemp_of.notna() & Diff_Log_mining.notna()
model_VAEMP_diff   = sm.OLS(
    Diff_Log_vaemp_of[mask_v],
    sm.add_constant(Diff_Log_mining[mask_v])
).fit(cov_type='HC3')
print(model_VAEMP_diff.summary())

# Niveaux logarithmiques
Log_vaemp_of        = DN_DONNEES_NUNAVUT_2005_2024["Log_VAEMP_EN"]
Log_mining_v        = TABLEAU_DN_MINING["Log_VIIRS_SUM_RESAMPL_MINING"]
mask_lv             = Log_vaemp_of.notna() & Log_mining_v.notna()
model_VAEMP_niveaux = sm.OLS(
    Log_vaemp_of[mask_lv],
    sm.add_constant(Log_mining_v[mask_lv])
).fit(cov_type='HC3')
print(model_VAEMP_niveaux.summary())

# ============================================================
# ÉQUATION 2 — Taux de croissance révisé VAEMP
# ============================================================
TABLEAU_DN_MINING['g_revisee_VAEMP'] = (
    rho * DN_DONNEES_NUNAVUT_2005_2024['Diff_Log_VAEMP_EN'].values
    + (1 - rho) * alpha
    * TABLEAU_DN_MINING['Diff_Log_VIIRS_SUM_RESAMPL_MINING']
)
#TABLEAU_DN_MINING['Log_VAEMP_CALCULER'] = (
#    rho * DN_DONNEES_NUNAVUT_2005_2024['Log_VAEMP_EN'].values
#    + (1 - rho) * 0.6
#    * TABLEAU_DN_MINING['Log_VIIRS_SUM_RESAMPL_MINING']
#)
#TABLEAU_DN_MINING['VAEMP_CALCULER'] = np.exp(TABLEAU_DN_MINING['Log_VAEMP_CALCULER'])
# ============================================================
# VALEUR DE BASE — année 2024 (même méthode que Modele Chen)
# base = valeur directe VAEMP_EN pour 2024 (pas équation 25)
# ============================================================
base_year_vaemp = 2024

base_vaemp = DN_DONNEES_NUNAVUT_2005_2024.loc[
    DN_DONNEES_NUNAVUT_2005_2024['Annee'] == base_year_vaemp, 'VAEMP_EN'
].values[0]

TABLEAU_DN_MINING['VAEMP_CORRIGE'] = np.nan
TABLEAU_DN_MINING.loc[
    TABLEAU_DN_MINING['Annee'] == base_year_vaemp, 'VAEMP_CORRIGE'
] = base_vaemp

print(f"\nVAEMP_CORRIGE de base {base_year_vaemp} : {base_vaemp:,.0f}")



# ============================================================
# ÉQUATION 21 — Reconstruction rétrospective
# VAEMP_t = VAEMP_{t+1} / (1 + g_t)   si VIIRS_t != 0
# VAEMP_t = 0                           si VIIRS_t == 0
# ============================================================
for i in range(len(TABLEAU_DN_MINING) - 1, -1, -1):

    annee_t = TABLEAU_DN_MINING.loc[i, 'Annee']

    if annee_t == base_year_vaemp:
        continue

    # Condition DN=0 pendant la boucle
    viirs_t = TABLEAU_DN_MINING.loc[i, 'VIIRS_SUM_RESAMPL_MINING']
    if pd.isna(viirs_t) or viirs_t == 0:
        TABLEAU_DN_MINING.loc[i, 'VAEMP_CORRIGE'] = 0
        continue

    # Trouver t+1
    next_idx = TABLEAU_DN_MINING[
        TABLEAU_DN_MINING['Annee'] == annee_t + 1
    ].index

    if len(next_idx) == 0:
        TABLEAU_DN_MINING.loc[i, 'VAEMP_CORRIGE'] = 0
        continue

    next_idx   = next_idx[0]
    vaemp_next = TABLEAU_DN_MINING.loc[next_idx, 'VAEMP_CORRIGE']
    g_current  = TABLEAU_DN_MINING.loc[i, 'g_revisee_VAEMP']

    if pd.isna(vaemp_next) or pd.isna(g_current) or g_current == -1:
        TABLEAU_DN_MINING.loc[i, 'VAEMP_CORRIGE'] = 0
        continue

    # ✅ Équation 21
    TABLEAU_DN_MINING.loc[i, 'VAEMP_CORRIGE'] = (
        vaemp_next / (1 + g_current)
    )

# ============================================================
# COLONNES DÉRIVÉES
# ============================================================
TABLEAU_DN_MINING['Log_VAEMP_CORRIGE'] = np.log(
    TABLEAU_DN_MINING['VAEMP_CORRIGE'].replace(0, np.nan)
)
TABLEAU_DN_MINING['Diff_VAEMP_CORRIGE'] = (
    TABLEAU_DN_MINING['VAEMP_CORRIGE'].diff()
)
TABLEAU_DN_MINING['Diff_Log_VAEMP_CORRIGE'] = (
    TABLEAU_DN_MINING['Log_VAEMP_CORRIGE'].diff()
)

# ============================================================
# VÉRIFICATION
# ============================================================
print(
    TABLEAU_DN_MINING[[
        'Annee',
        'VIIRS_SUM_RESAMPL_MINING',
        'g_revisee_VAEMP',
        'VAEMP_CORRIGE'
    ]].to_string()
)
#%% CALCUL SAM CORRIGER
# ============================================================
# RÉGRESSIONS — SAMP
# ============================================================

# Différences logarithmiques
Diff_Log_samp_of  = DN_DONNEES_NUNAVUT_2005_2024["Diff_Log_SAMP_EN"]
Diff_Log_mining_s = TABLEAU_DN_MINING["Diff_Log_VIIRS_SUM_RESAMPL_MINING"]
mask_s            = Diff_Log_samp_of.notna() & Diff_Log_mining_s.notna()
model_SAMP_diff   = sm.OLS(
    Diff_Log_samp_of[mask_s],
    sm.add_constant(Diff_Log_mining_s[mask_s])
).fit(cov_type='HC3')
print(model_SAMP_diff.summary())

# Niveaux logarithmiques
Log_samp_of        = DN_DONNEES_NUNAVUT_2005_2024["Log_SAMP_EN"]
Log_mining_s       = TABLEAU_DN_MINING["Log_VIIRS_SUM_RESAMPL_MINING"]
mask_ls            = Log_samp_of.notna() & Log_mining_s.notna()
model_SAMP_niveaux = sm.OLS(
    Log_samp_of[mask_ls],
    sm.add_constant(Log_mining_s[mask_ls])
).fit(cov_type='HC3')
print(model_SAMP_niveaux.summary())

# ============================================================
# ÉQUATION 2 — Taux de croissance révisé SAMP
# ============================================================
TABLEAU_DN_MINING['g_revisee_SAMP'] = (
    rho * DN_DONNEES_NUNAVUT_2005_2024['Diff_Log_SAMP_EN'].values
    + (1 - rho) * alpha
    * TABLEAU_DN_MINING['Diff_Log_VIIRS_SUM_RESAMPL_MINING']
)

# ============================================================
# VALEUR DE BASE — année 2024 (même méthode que Modele Chen)
# base = valeur directe SAMP_EN pour 2024 (pas équation 25)
# ============================================================
base_year_samp = 2024

base_samp = DN_DONNEES_NUNAVUT_2005_2024.loc[
    DN_DONNEES_NUNAVUT_2005_2024['Annee'] == base_year_samp, 'SAMP_EN'
].values[0]

TABLEAU_DN_MINING['SAMP_CORRIGE'] = np.nan
TABLEAU_DN_MINING.loc[
    TABLEAU_DN_MINING['Annee'] == base_year_samp, 'SAMP_CORRIGE'
] = base_samp

print(f"\nSAMP_CORRIGE de base {base_year_samp} : {base_samp:,.0f}")

# ============================================================
# ÉQUATION 21 — Reconstruction rétrospective
# SAMP_t = SAMP_{t+1} / (1 + g_t)   si VIIRS_t != 0
# SAMP_t = 0                          si VIIRS_t == 0
# ============================================================
for i in range(len(TABLEAU_DN_MINING) - 1, -1, -1):

    annee_t = TABLEAU_DN_MINING.loc[i, 'Annee']

    if annee_t == base_year_samp:
        continue

    # Condition DN=0 pendant la boucle
    viirs_t = TABLEAU_DN_MINING.loc[i, 'VIIRS_SUM_RESAMPL_MINING']
    if pd.isna(viirs_t) or viirs_t == 0:
        TABLEAU_DN_MINING.loc[i, 'SAMP_CORRIGE'] = 0
        continue

    # Trouver t+1
    next_idx = TABLEAU_DN_MINING[
        TABLEAU_DN_MINING['Annee'] == annee_t + 1
    ].index

    if len(next_idx) == 0:
        TABLEAU_DN_MINING.loc[i, 'SAMP_CORRIGE'] = 0
        continue

    next_idx  = next_idx[0]
    samp_next = TABLEAU_DN_MINING.loc[next_idx, 'SAMP_CORRIGE']
    g_current = TABLEAU_DN_MINING.loc[i, 'g_revisee_SAMP']

    if pd.isna(samp_next) or pd.isna(g_current) or g_current == -1:
        TABLEAU_DN_MINING.loc[i, 'SAMP_CORRIGE'] = 0
        continue

    # ✅ Équation 21
    TABLEAU_DN_MINING.loc[i, 'SAMP_CORRIGE'] = (
        samp_next / (1 + g_current)
    )

# ============================================================
# COLONNES DÉRIVÉES
# ============================================================
TABLEAU_DN_MINING['Log_SAMP_CORRIGE'] = np.log(
    TABLEAU_DN_MINING['SAMP_CORRIGE'].replace(0, np.nan)
)
TABLEAU_DN_MINING['Diff_SAMP_CORRIGE'] = (
    TABLEAU_DN_MINING['SAMP_CORRIGE'].diff()
)
TABLEAU_DN_MINING['Diff_Log_SAMP_CORRIGE'] = (
    TABLEAU_DN_MINING['Log_SAMP_CORRIGE'].diff()
)

# ============================================================
# VÉRIFICATION
# ============================================================
print(
    TABLEAU_DN_MINING[[
        'Annee',
        'VIIRS_SUM_RESAMPL_MINING',
        'g_revisee_SAMP',
        'SAMP_CORRIGE'
    ]].to_string()
)



#%% MODELE CHEN
alpha_mine = 1.33

## Modele Chen pour predire le PIB Minier

VIIRS_NUNAVUT_SEL = DN_DONNEES_NUNAVUT_2000_2024[["Nom_Fr","Annee", "VIIRS_SUM_RESAMPL"]].copy()
VIIRS_NUNAVUT_SEL = VIIRS_NUNAVUT_SEL.rename(columns={"Nom_Fr": "Nom"})

VIIRS_MINE_NUNAVUT = pd.concat([VIIRS_MINE, VIIRS_NUNAVUT_SEL], ignore_index=True)


# Pivoter le DataFrame
VIIRS = VIIRS_MINE_NUNAVUT.pivot(index='Annee', columns='Nom', values='VIIRS_SUM_RESAMPL')
VIIRS = VIIRS.reset_index()

# Calculer la somme des mines pour chaque année

# Calculer nunavut_reste
VIIRS['nunavut_reste'] = VIIRS['Nunavut'] - VIIRS['Mine_Baffinland'] - VIIRS['Mine_Hope_Bay'] - VIIRS['Mine_Meliadine'] - VIIRS['Mine_Meadowbank']
VIIRS['Total_Mine'] = VIIRS['Mine_Baffinland'] + VIIRS['Mine_Hope_Bay'] + VIIRS['Mine_Meliadine'] + VIIRS['Mine_Meadowbank']



# Sélectionner uniquement les colonnes Annee et PIB_EN dans DATA_PIB_DN_2012_2024

# Fusionner par année
#VIIRS = VIIRS.merge(DATA_PIB_DN_2012_2024[["Annee", "PIB_EN"]], on="Annee", how="left")


# Méthode correcte et recommandée (syntaxe pandas standard)
VIIRS_MINE_SEL = VIIRS[['Annee', 'Mine_Baffinland', 'Mine_Hope_Bay', 
                        'Mine_Meliadine', 'Mine_Meadowbank', 'nunavut_reste']].copy()


# Format long avec pd.melt()
VIIRS_MINE_SEL_LONG = VIIRS_MINE_SEL.melt(
    id_vars='Annee',              # variable(s) qui restent fixes
    value_vars=['Mine_Baffinland', 'Mine_Hope_Bay', 
                'Mine_Meliadine', 'Mine_Meadowbank', 'nunavut_reste'],  # colonnes à "fondre"
    var_name='Mine',              # nom de la nouvelle colonne avec les noms des mines
    value_name='VIIRS_SUM_RESAMPL'           # nom de la nouvelle colonne avec les valeurs
)

# 3. Trier pour que le calcul soit correct
VIIRS_MINE_SEL_LONG = VIIRS_MINE_SEL_LONG.sort_values(['Mine', 'Annee']).reset_index(drop=True)

VIIRS_MINE_SEL_LONG = compute_log_and_diff_lum(
    VIIRS_MINE_SEL_LONG, value_col="VIIRS_SUM_RESAMPL", group_col="Mine"
)



MINE_SEL = VIIRS_MINE_SEL_LONG.copy()
PIB_SEL = DN_DONNEES_NUNAVUT_2000_2024[["Annee", "PIB_EN", "Diff_Log_PIB_EN"]]

# Fusion sur l'année
TABLE_MINE_PIB_SEL_1 = MINE_SEL.merge(PIB_SEL, on="Annee", how="left")

# Trier par mine et année
TABLE_MINE_PIB_SEL_1 = TABLE_MINE_PIB_SEL_1.sort_values(["Mine", "Annee"]).reset_index(drop=True)



annee_ref = 2024

TABLE_MINE_PIB_SEL_2024 = TABLE_MINE_PIB_SEL_1[TABLE_MINE_PIB_SEL_1["Annee"] == annee_ref].copy()

total_viirs_2024 = TABLE_MINE_PIB_SEL_2024["VIIRS_SUM_RESAMPL"].sum()
pib_total_2024 = TABLE_MINE_PIB_SEL_2024["PIB_EN"].iloc[0]

TABLE_MINE_PIB_SEL_2024["PIB_MINE"] = (
    pib_total_2024 *
    TABLE_MINE_PIB_SEL_2024["VIIRS_SUM_RESAMPL"] / total_viirs_2024
)

TABLE_MINE_PIB_SEL_1["PIB_MINE"] = None

TABLE_MINE_PIB_SEL_1.loc[TABLE_MINE_PIB_SEL_1["Annee"] == annee_ref, "PIB_MINE"] = TABLE_MINE_PIB_SEL_2024["PIB_MINE"].values



# Équation 2 : y*_it = rho * y_it + (1 - rho) * y'_it
DN_DONNEES_NUNAVUT_2000_2024["y_star"] = rho * DN_DONNEES_NUNAVUT_2000_2024["Diff_Log_PIB_EN"] + (1 - rho) * DN_DONNEES_NUNAVUT_2000_2024["Diff_Log_VIIRS_SUM_RESAMPL"]
print(DN_DONNEES_NUNAVUT_2000_2024[["Diff_Log_PIB_EN", "Diff_Log_VIIRS_SUM_RESAMPL", "y_star"]].head(10))

TABLE_MINE_PIB_SEL = TABLE_MINE_PIB_SEL_1[["Annee", "Mine", "VIIRS_SUM_RESAMPL","PIB_MINE","PIB_EN",'Diff_Log_VIIRS_SUM_RESAMPL','Diff_Log_PIB_EN']].copy()

# Trier par Mine et Annee
TABLE_MINE_PIB_SEL = TABLE_MINE_PIB_SEL.sort_values(["Mine", "Annee"]).reset_index(drop=True)

# Calculer DN_t-1 par groupe Mine
TABLE_MINE_PIB_SEL["DN_lag"] = TABLE_MINE_PIB_SEL.groupby("Mine")["VIIRS_SUM_RESAMPL"].shift(1)

# (DN_t - DN_t-1) / DN_t-1
TABLE_MINE_PIB_SEL["delta_DN"] = (
    (TABLE_MINE_PIB_SEL["VIIRS_SUM_RESAMPL"] - TABLE_MINE_PIB_SEL["DN_lag"])
    / TABLE_MINE_PIB_SEL["DN_lag"]
)

# Fusionner avec y_star national (Nunavut)
TABLE_MINE_PIB_SEL = TABLE_MINE_PIB_SEL.merge(
    DN_DONNEES_NUNAVUT_2000_2024[["Annee", "y_star"]],
    on="Annee", how="left"
)

# Equation 20
# gy*_ij,t = rho * y*_i,t + (1-rho) * alpha * delta_DN   si DN_t-1 != 0
#          = y*_i,t                                        si DN_t-1 == 0


def eq20(row):
    if pd.isna(row["DN_lag"]) or row["DN_lag"] == 0:
        return row["y_star"]  # DN_t-1 = 0 => taux national
    return rho * row["y_star"] + (1 - rho) * alpha_mine * row["delta_DN"]

TABLE_MINE_PIB_SEL["gy_star"] = TABLE_MINE_PIB_SEL.apply(eq20, axis=1)

print(TABLE_MINE_PIB_SEL[["Annee", "Mine", "VIIRS_SUM_RESAMPL", "DN_lag", "delta_DN", "y_star", "gy_star"]].head(20))



#def eq20(row):
#    if pd.isna(row["VIIRS_SUM_RESAMPL"]) or row["VIIRS_SUM_RESAMPL"] == 0:
#        return 0  # DN_t = 0 => pas d'activité économique
#    return rho * row["y_star"] + (1 - rho) * alpha * row["delta_DN"]
#
#["gy_star"] = TABLE_MINE_PIB_TEST.apply(eq20, axis=1)

#print(TABLE_MINE_PIB_TEST[["Annee", "Mine", "VIIRS_SUM_RESAMPL", "DN_lag", "delta_DN", "y_star", "gy_star"]].head(20))



# Equation 21
# RGY*_ij,t = RGY*_ij,t+1 / (1 + gy*_ij,t)   si DN_ij,t != 0
#           = 0                                  si DN_ij,t == 0

# La valeur de base est PIB_MINE en 2024 (dernière année connue)
# On remonte en arrière à partir de 2024
# Equation 21 - on remonte à partir de 2024
TABLE_MINE_PIB_SEL = TABLE_MINE_PIB_SEL.sort_values(["Mine", "Annee"]).reset_index(drop=True)

for mine, group in TABLE_MINE_PIB_SEL.groupby("Mine"):
    idx = sorted(group.index.tolist(), reverse=True)  # du plus récent au plus ancien
    
    for pos, i in enumerate(idx):
        dn  = TABLE_MINE_PIB_SEL.loc[i, "VIIRS_SUM_RESAMPL"]
        gy  = TABLE_MINE_PIB_SEL.loc[i, "gy_star"]
        pib = TABLE_MINE_PIB_SEL.loc[i, "PIB_MINE"]
        
        # Année de base 2024 : déjà calculée, on ne touche pas
        if pd.notna(pib):
            continue
        
        # Année suivante (i+1 dans le dataframe trié)
        next_i = idx[pos - 1]  # pos-1 car on est en ordre décroissant
        next_pib = TABLE_MINE_PIB_SEL.loc[next_i, "PIB_MINE"]
        
        if pd.isna(dn) or dn == 0:
            TABLE_MINE_PIB_SEL.loc[i, "PIB_MINE"] = 0
        elif pd.isna(gy) or pd.isna(next_pib):
            TABLE_MINE_PIB_SEL.loc[i, "PIB_MINE"] = None
        else:
            TABLE_MINE_PIB_SEL.loc[i, "PIB_MINE"] = next_pib / (1 + gy)

print(TABLE_MINE_PIB_SEL[["Annee", "Mine", "VIIRS_SUM_RESAMPL", "gy_star", "PIB_MINE"]])



TABLE_MINE_PIB_SEL["POURCENTAGE_PIB"] = (
    TABLE_MINE_PIB_SEL["PIB_MINE"] /
    TABLE_MINE_PIB_SEL.groupby("Annee")["PIB_MINE"].transform("sum")
) * 100



#%% ROBUSTESSE — SENSIBILITÉ DU MODÈLE CHEN AUX PARAMÈTRES (ρ, α)
################################################################################
# TEST DE ROBUSTESSE : SENSIBILITÉ DU MODÈLE CHEN À ρ ET α
#
# Référence : Chen & Nordhaus (2011) « Using luminosity data as a proxy for
# economic statistics », PNAS 108(21); Henderson, Storeygard & Weil (2012)
# « Measuring Economic Growth from Outer Space », AER 102(2).
#
# Le modèle Chen agrège deux signaux (taux statistique + taux satellitaire)
# via deux paramètres clés :
#
#   y*_t      = ρ × Diff_Log_PIB_t  +  (1-ρ) × Diff_Log_VIIRS_t      (eq. 2)
#   gy*_i,t   = ρ × y*_t            +  (1-ρ) × α × delta_DN_i,t       (eq. 20)
#
#   ρ : poids accordé à la mesure statistique du taux national
#   α : élasticité lumière → PIB au niveau mine (delta_DN → croissance)
#
# Grille testée :
#   ρ ∈ {0.80, 0.90, 0.94, 0.98}
#   α ∈ {0.30, 0.45, 0.60}     (gamme Henderson 2012 / Pinkovskiy & SiM 2016)
#   → 12 scénarios × 4 mines × 25 années
#
# Cas central de référence pour le calcul des écarts : (ρ=0.94, α=0.45),
# soit les valeurs les plus citées dans la littérature.
# NB : le déploiement principal du modèle Chen utilise α_mine = 1.33,
# valeur calibrée hors de cette grille — la sensibilité est explorée ici
# autour des valeurs canoniques de la littérature.
#
# Sortie : Excel multi-feuilles + heatmap d'écart + trajectoires + table LaTeX.
################################################################################

print("\n" + "=" * 78)
print("  TEST DE ROBUSTESSE — SENSIBILITÉ CHEN AUX PARAMÈTRES (ρ, α)")
print("=" * 78)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Paramètres de la grille
# ─────────────────────────────────────────────────────────────────────────────
RHO_GRID   = [0.80, 0.90, 0.94, 0.98]
ALPHA_GRID = [0.30, 0.45, 0.60]
_RHO_REF, _ALPHA_REF = 0.94, 0.45    # cas central pour le calcul des écarts

_MINES_CHEN = ["Mine_Baffinland", "Mine_Hope_Bay",
               "Mine_Meliadine",  "Mine_Meadowbank"]
_LABELS_CHEN = {
    "Mine_Meadowbank": "Mine Meadowbank",
    "Mine_Meliadine":  "Mine Meliadine",
    "Mine_Hope_Bay":   "Mine Hope Bay",
    "Mine_Baffinland": "Mine Baffinland",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fonction qui réplique le pipeline Chen avec (ρ, α) paramétriques
# ─────────────────────────────────────────────────────────────────────────────
def _chen_estim_param(rho_val, alpha_val, base_table, nat_data, annee_ref=2024):
    """
    Reproduit le pipeline du modèle Chen avec ρ et α paramétriques.

    base_table : TABLE_MINE_PIB_SEL (Annee, Mine, VIIRS_SUM_RESAMPL,
                 DN_lag, delta_DN, PIB_MINE pour l'année d'ancrage)
    nat_data   : DN_DONNEES_NUNAVUT_2000_2024 (Diff_Log_PIB_EN,
                 Diff_Log_VIIRS_SUM_RESAMPL)
    Retourne   : DataFrame [Annee, Mine, PIB_MINE].
    """
    # (a) y_star national paramétrique (eq. 2)
    _dn = nat_data[["Annee", "Diff_Log_PIB_EN",
                    "Diff_Log_VIIRS_SUM_RESAMPL"]].copy()
    _dn["y_star_p"] = (
        rho_val * _dn["Diff_Log_PIB_EN"]
        + (1 - rho_val) * _dn["Diff_Log_VIIRS_SUM_RESAMPL"]
    )

    # (b) Copie de la table mine, on garde uniquement l'ancrage à annee_ref
    _tmp = base_table[["Annee", "Mine", "VIIRS_SUM_RESAMPL",
                       "DN_lag", "delta_DN", "PIB_MINE"]].copy()
    _tmp.loc[_tmp["Annee"] != annee_ref, "PIB_MINE"] = np.nan
    _tmp["PIB_MINE"] = pd.to_numeric(_tmp["PIB_MINE"], errors="coerce")

    # (c) Joindre y_star paramétrique
    _tmp = _tmp.merge(_dn[["Annee", "y_star_p"]], on="Annee", how="left")

    # (d) gy_star paramétrique (eq. 20)
    def _eq20_param(row):
        if pd.isna(row["DN_lag"]) or row["DN_lag"] == 0:
            return row["y_star_p"]
        return (rho_val * row["y_star_p"]
                + (1 - rho_val) * alpha_val * row["delta_DN"])
    _tmp["gy_star_p"] = _tmp.apply(_eq20_param, axis=1)

    # (e) Remontée temporelle depuis annee_ref
    _tmp = _tmp.sort_values(["Mine", "Annee"]).reset_index(drop=True)
    for _mine, _grp in _tmp.groupby("Mine"):
        _idx = sorted(_grp.index.tolist(), reverse=True)
        for _pos, _i in enumerate(_idx):
            _dn_v = _tmp.loc[_i, "VIIRS_SUM_RESAMPL"]
            _gy   = _tmp.loc[_i, "gy_star_p"]
            _pib  = _tmp.loc[_i, "PIB_MINE"]
            if pd.notna(_pib):
                continue
            _next_i   = _idx[_pos - 1]
            _next_pib = _tmp.loc[_next_i, "PIB_MINE"]
            if pd.isna(_dn_v) or _dn_v == 0:
                _tmp.loc[_i, "PIB_MINE"] = 0
            elif pd.isna(_gy) or pd.isna(_next_pib):
                _tmp.loc[_i, "PIB_MINE"] = np.nan
            else:
                _tmp.loc[_i, "PIB_MINE"] = _next_pib / (1 + _gy)

    return _tmp[["Annee", "Mine", "PIB_MINE"]]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Exécution sur la grille 4 × 3 = 12 scénarios
# ─────────────────────────────────────────────────────────────────────────────
_RESULTS = []
for _rho in RHO_GRID:
    for _alpha in ALPHA_GRID:
        _df = _chen_estim_param(
            _rho, _alpha,
            TABLE_MINE_PIB_SEL, DN_DONNEES_NUNAVUT_2000_2024
        )
        _df["rho"]      = _rho
        _df["alpha"]    = _alpha
        _df["scenario"] = f"ρ={_rho:.2f}, α={_alpha:.2f}"
        _RESULTS.append(_df)
        print(f"  ✔ Scénario ρ={_rho:.2f}, α={_alpha:.2f} calculé")

SENS_CHEN = pd.concat(_RESULTS, ignore_index=True)
SENS_CHEN["PIB_MINE"] = pd.to_numeric(SENS_CHEN["PIB_MINE"], errors="coerce")
SENS_CHEN = SENS_CHEN[SENS_CHEN["Mine"].isin(_MINES_CHEN)].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Référence = cas central (ρ=0.94, α=0.45) → calcul des écarts
# ─────────────────────────────────────────────────────────────────────────────
_REF = (
    SENS_CHEN[(SENS_CHEN["rho"] == _RHO_REF) & (SENS_CHEN["alpha"] == _ALPHA_REF)]
    [["Annee", "Mine", "PIB_MINE"]]
    .rename(columns={"PIB_MINE": "PIB_REF"})
)
SENS_CHEN = SENS_CHEN.merge(_REF, on=["Annee", "Mine"], how="left")
SENS_CHEN["Ecart_abs"] = SENS_CHEN["PIB_MINE"] - SENS_CHEN["PIB_REF"]
SENS_CHEN["Ecart_pct"] = (
    100 * SENS_CHEN["Ecart_abs"] / SENS_CHEN["PIB_REF"].replace(0, np.nan)
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Synthèses (période 2010–2024, post-ouverture des mines)
# ─────────────────────────────────────────────────────────────────────────────
_PERIODE = (SENS_CHEN["Annee"] >= 2010) & (SENS_CHEN["Annee"] <= 2024)

_PER_MINE = (
    SENS_CHEN[_PERIODE]
    .groupby(["Mine", "rho", "alpha"])
    .agg(
        PIB_moy_2010_2024 = ("PIB_MINE",  "mean"),
        Ecart_pct_moy     = ("Ecart_pct", lambda s: s.abs().mean()),
        Ecart_pct_max     = ("Ecart_pct", lambda s: s.abs().max()),
    )
    .reset_index()
    .sort_values(["Mine", "rho", "alpha"])
)

_PER_SCEN = (
    SENS_CHEN[_PERIODE]
    .groupby(["rho", "alpha"])
    .agg(
        Ecart_pct_moy = ("Ecart_pct", lambda s: s.abs().mean()),
        Ecart_pct_max = ("Ecart_pct", lambda s: s.abs().max()),
        PIB_global    = ("PIB_MINE",  "mean"),
    )
    .reset_index()
    .sort_values(["rho", "alpha"])
)

print("\n  --- Écart |%| moyen vs cas central (ρ=0.94, α=0.45) — par scénario ---")
print(_PER_SCEN.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Export Excel multi-feuilles
# ─────────────────────────────────────────────────────────────────────────────
_path_xl_sens = os.path.join(tab_dir, "tableau_sensibilite_chen_rho_alpha.xlsx")
try:
    with pd.ExcelWriter(_path_xl_sens, engine="openpyxl") as _xl:
        _PER_SCEN.to_excel(_xl, sheet_name="synthese_grille",   index=False)
        _PER_MINE.to_excel(_xl, sheet_name="synthese_par_mine", index=False)
        SENS_CHEN.to_excel(_xl, sheet_name="series_completes",  index=False)
        for _m in _MINES_CHEN:
            _piv = (_PER_MINE[_PER_MINE["Mine"] == _m]
                    .pivot(index="rho", columns="alpha",
                           values="Ecart_pct_moy").round(2))
            _piv.to_excel(_xl, sheet_name=f"ecart_pct_{_m[:25]}")
    print(f"\n  ✔ Excel exporté : {_path_xl_sens}")
except Exception as _e:
    print(f"  ⚠ Export Excel échoué : {_e}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Heatmap (2×2) — écart |%| moyen par mine × (ρ, α)
# ─────────────────────────────────────────────────────────────────────────────
fig_sens_h, axes_sens_h = plt.subplots(2, 2, figsize=(13, 10), facecolor="white")
_vmax_h = max(_PER_MINE["Ecart_pct_moy"].max(), 1.0)

for _ax_h, _mine in zip(axes_sens_h.flatten(), _MINES_CHEN):
    _piv_m = (_PER_MINE[_PER_MINE["Mine"] == _mine]
              .pivot(index="rho", columns="alpha", values="Ecart_pct_moy"))
    _im = _ax_h.imshow(_piv_m.values, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=_vmax_h)
    for _i in range(_piv_m.shape[0]):
        for _j in range(_piv_m.shape[1]):
            _v = _piv_m.values[_i, _j]
            _col_t = "white" if _v > _vmax_h * 0.55 else "#222222"
            _ax_h.text(_j, _i, f"{_v:.1f}%", ha="center", va="center",
                       color=_col_t, fontsize=10, fontfamily="serif")
    _ax_h.set_xticks(range(len(_piv_m.columns)))
    _ax_h.set_xticklabels([f"{a:.2f}" for a in _piv_m.columns])
    _ax_h.set_yticks(range(len(_piv_m.index)))
    _ax_h.set_yticklabels([f"{r:.2f}" for r in _piv_m.index])
    _ax_h.set_xlabel(r"$\alpha$ (élasticité lumière→PIB)", fontsize=10)
    _ax_h.set_ylabel(r"$\rho$ (poids stat. national)",      fontsize=10)
    _ax_h.set_title(_LABELS_CHEN.get(_mine, _mine), fontsize=11, fontweight="bold")
    plt.colorbar(_im, ax=_ax_h, fraction=0.04, pad=0.04,
                 label="|Écart| % moyen")

plt.suptitle("Sensibilité du modèle Chen — Écart |%| moyen vs (ρ=0.94, α=0.45)\n"
             "Période 2010–2024, par mine",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
_fp_heat_sens = os.path.join(fig_dir, "robustesse_chen_heatmap_rho_alpha.png")
plt.savefig(_fp_heat_sens, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
print(f"  ✔ Heatmap sauvegardée : {_fp_heat_sens}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Trajectoires PIB par mine sous les 12 scénarios (2×2)
# ─────────────────────────────────────────────────────────────────────────────
fig_sens_t, axes_sens_t = plt.subplots(2, 2, figsize=(15, 9.5), facecolor="white")
_n_scen   = len(RHO_GRID) * len(ALPHA_GRID)
_palette  = plt.get_cmap("viridis", _n_scen)
_color_map = {}
_ii = 0
for _r in RHO_GRID:
    for _a in ALPHA_GRID:
        _color_map[(_r, _a)] = _palette(_ii)
        _ii += 1

for _ax_t, _mine in zip(axes_sens_t.flatten(), _MINES_CHEN):
    _sub_m = SENS_CHEN[SENS_CHEN["Mine"] == _mine]
    for (_r, _a), _g in _sub_m.groupby(["rho", "alpha"]):
        _g = _g.sort_values("Annee")
        _is_ref = (_r == _RHO_REF and _a == _ALPHA_REF)
        _ax_t.plot(
            _g["Annee"], _g["PIB_MINE"],
            color=_color_map[(_r, _a)],
            linewidth=2.6 if _is_ref else 1.1,
            alpha=1.0   if _is_ref else 0.60,
            zorder=6    if _is_ref else 3,
            label=f"ρ={_r:.2f}, α={_a:.2f}" + (" (réf.)" if _is_ref else "")
        )
    _ax_t.set_title(_LABELS_CHEN.get(_mine, _mine),
                    fontsize=11, fontweight="bold")
    _ax_t.set_xlabel("Année", fontsize=10)
    _ax_t.set_ylabel("PIB mine (M$ CAD)", fontsize=10)
    _ax_t.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _ax_t.spines["top"].set_visible(False)
    _ax_t.spines["right"].set_visible(False)
    _ax_t.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC")
    _ax_t.set_axisbelow(True)
    _ax_t.legend(fontsize=7, ncol=2, framealpha=0.9, loc="best")

plt.suptitle("Sensibilité Chen — Trajectoires du PIB par mine sous 12 scénarios "
             "(ρ × α)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
_fp_traj_sens = os.path.join(fig_dir, "robustesse_chen_trajectoires_rho_alpha.png")
plt.savefig(_fp_traj_sens, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
print(f"  ✔ Trajectoires sauvegardées : {_fp_traj_sens}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Table LaTeX de synthèse
# ─────────────────────────────────────────────────────────────────────────────
_path_tex_sens = os.path.join(tab_dir, "tableau_sensibilite_chen.tex")
try:
    # Listes formatées en français pour caption/notes
    _rho_str_fr   = ";".join(_fmt_fr(_r, 2) for _r in RHO_GRID)
    _alpha_str_fr = ";".join(_fmt_fr(_a, 2) for _a in ALPHA_GRID)
    _rho_ref_fr   = _fmt_fr(_RHO_REF,   2)
    _alpha_ref_fr = _fmt_fr(_ALPHA_REF, 2)

    with open(_path_tex_sens, "w", encoding="utf-8") as _f:
        _f.write("\\begin{table}[htbp]\n\\centering\n")
        _f.write("\\caption{Sensibilité du modèle Chen aux paramètres "
                 "$\\rho$ et $\\alpha$ --- écart $|\\%|$ moyen 2010--2024 "
                 f"vs cas central ($\\rho={_rho_ref_fr}$, "
                 f"$\\alpha={_alpha_ref_fr}$).}}\n")
        _f.write("\\label{tab:sensibilite_chen_rho_alpha}\n")
        _f.write("\\begin{tabular}{l" + "r" * len(ALPHA_GRID) + "}\n\\hline\\hline\n")
        _f.write("$\\rho$\\,\\textbackslash\\,$\\alpha$")
        for _a in ALPHA_GRID:
            _f.write(f" & $\\alpha={_fmt_fr(_a, 2)}$")
        _f.write(" \\\\\n\\hline\n")
        for _r in RHO_GRID:
            _f.write(f"$\\rho={_fmt_fr(_r, 2)}$")
            for _a in ALPHA_GRID:
                _v = _PER_SCEN[
                    (_PER_SCEN["rho"] == _r) & (_PER_SCEN["alpha"] == _a)
                ]["Ecart_pct_moy"].values
                _vv = _v[0] if len(_v) else np.nan
                _f.write(f" & {_fmt_fr(_vv, 1)}\\,\\%")
            _f.write(" \\\\\n")
        _f.write("\\hline\\hline\n")
        _f.write("\\end{tabular}\n")
        # Note sortie du tabular pour permettre le retour à la ligne automatique
        _f.write("\\begin{minipage}{\\textwidth}\\footnotesize "
                 "\\textit{Note :} écart en valeur absolue moyenné sur les 4 mines "
                 "et la période 2010--2024, par rapport au scénario central "
                 f"($\\rho={_rho_ref_fr}$, $\\alpha={_alpha_ref_fr}$). "
                 f"Grille testée : $\\rho \\in \\{{{_rho_str_fr}\\}}$, "
                 f"$\\alpha \\in \\{{{_alpha_str_fr}\\}}$."
                 "\\end{minipage}\n")
        _f.write("\\end{table}\n")
    print(f"  ✔ Table LaTeX exportée : {_path_tex_sens}")
except Exception as _e:
    print(f"  ⚠ Export LaTeX échoué : {_e}")

print("\n  ✔ Test de robustesse Chen (ρ, α) terminé.")
print("=" * 78 + "\n")


#%% MODELE SCORE
    
## Modele score



TABLE_MINE_PIB_SEL_2009_2022 = TABLE_MINE_PIB_SEL[(TABLE_MINE_PIB_SEL["Annee"] >= 2009) & (TABLE_MINE_PIB_SEL["Annee"] <= 2022)]
POPULATION_2009_2022 = POPULATION_2005_2024[(POPULATION_2005_2024["Annee"] >= 2009) & (POPULATION_2005_2024["Annee"] <= 2022)]



# CRÉATION TABLEAU MINE

TABLE_VIIRS_MINE_SHARE = DMSP_VIIRS_MINE_2000_2024[["Nom","Annee", "VIIRS_SUM_RESAMPL"]].copy()

TABLE_VIIRS_MINE_SHARE = compute_log_and_diff_lum(
    TABLE_VIIRS_MINE_SHARE, value_col="VIIRS_SUM_RESAMPL", group_col="Nom"
)


TABLE_VIIRS_MINE_SHARE_2009_2022 = TABLE_VIIRS_MINE_SHARE[(TABLE_VIIRS_MINE_SHARE["Annee"] >= 2009) & (TABLE_VIIRS_MINE_SHARE["Annee"] <= 2022)]
DN_DONNEES_NUNAVUT_2009_2022 = DN_DONNEES_NUNAVUT_2005_2024[(
    DN_DONNEES_NUNAVUT_2005_2024["Annee"] >= 2009) & (
        DN_DONNEES_NUNAVUT_2005_2024["Annee"] <= 2022)]
EMPLOI_MINE = EMPLOI_MINE[(EMPLOI_MINE["Annee"] >= 2009) & (EMPLOI_MINE["Annee"] <= 2022)]

TABLE_EMPLOI = EMPLOI_MINE[
    ['Annee', 'Total_Empl_MK', 'Total_Empl_MEL', 'Total_Empl_HB', 'Total_Empl_MR']
].copy()

TABLE_EMPLOI = TABLE_EMPLOI.rename(columns={
    'Total_Empl_MK': 'Mine_Meadowbank',
    'Total_Empl_MEL': 'Mine_Meliadine',
    'Total_Empl_HB': 'Mine_Hope_Bay',
    'Total_Empl_MR': 'Mine_Baffinland'
})
# Sélectionne seulement les colonnes utiles dans Table_share_2009_2022
pop_active_nunavut = POPULATION_2009_2022[['Annee', 'POP_ACTIVE']]

# Fusion avec TABLE_EMPLOI par année
TABLE_EMPLOI = TABLE_EMPLOI.merge(
    pop_active_nunavut,
    on='Annee',
    how='left'
)

# Renommer la colonne pour Nunavut
TABLE_EMPLOI.rename(columns={'POP_ACTIVE': 'Nunavut'}, inplace=True)

# Calculer nunavut_reste en restant aligné sur les années
TABLE_EMPLOI['nunavut_reste'] = (
    TABLE_EMPLOI['Nunavut']
    - TABLE_EMPLOI['Mine_Baffinland']
    - TABLE_EMPLOI['Mine_Hope_Bay']
    - TABLE_EMPLOI['Mine_Meliadine']
    - TABLE_EMPLOI['Mine_Meadowbank']
)
TABLE_EMPLOI_LONG = TABLE_EMPLOI.melt(
    id_vars=['Annee'],
    value_vars=['Mine_Meadowbank', 'Mine_Meliadine', 'Mine_Hope_Bay', 'Mine_Baffinland','nunavut_reste'],
    var_name='Mine',
    value_name='POP_ACTIVE'
)



TABLE_PRODUCTION = DN_DONNEES_NUNAVUT_2009_2022[['Annee', 'GPMB_CAD_EN', 'GPM_CAD_EN', 'GPHB_CAD_EN', 'IPB_CAD_EN']].copy()

TABLE_PRODUCTION = TABLE_PRODUCTION.rename(columns={
    'GPMB_CAD_EN': 'Mine_Meadowbank',
    'GPM_CAD_EN': 'Mine_Meliadine',
    'GPHB_CAD_EN': 'Mine_Hope_Bay',
    'IPB_CAD_EN': 'Mine_Baffinland'
})
TABLE_PRODUCTION['nunavut_reste'] = 0


TABLE_PRODUCTION_LONG = TABLE_PRODUCTION.melt(
    id_vars=['Annee'],
    value_vars=['Mine_Meadowbank', 'Mine_Meliadine', 'Mine_Hope_Bay', 'Mine_Baffinland','nunavut_reste'],
    var_name='Mine',
    value_name='PROD_MINE_CAD_EN'
)



TABLEAU_MINE = (
    TABLE_EMPLOI_LONG
    .merge(
        TABLE_PRODUCTION_LONG,
        on=['Annee', 'Mine'],
        how='outer'
    )
    .merge(
        TABLE_MINE_PIB_SEL_2009_2022[
            ['Annee', 'Mine', 'PIB_EN', 'VIIRS_SUM_RESAMPL','Diff_Log_VIIRS_SUM_RESAMPL','Diff_Log_PIB_EN']
        ],
        on=['Annee', 'Mine'],
        how='outer'
    )
)

# Trier par mine et année
TABLEAU_MINE = TABLEAU_MINE.sort_values(['Mine', 'Annee']).reset_index(drop=True)
TABLEAU_MINE = compute_log_and_diff(
    TABLEAU_MINE, value_col="POP_ACTIVE", group_col="Mine"
)

TABLEAU_MINE = compute_log_and_diff_lum(
    TABLEAU_MINE, value_col="PROD_MINE_CAD_EN", group_col="Mine"
)


## calcul du pib par mine par allocation
# --- 0) Garde seulement les 4 mines productrices
mines = ['Mine_Meadowbank','Mine_Meliadine','Mine_Hope_Bay','Mine_Baffinland','nunavut_reste']
TABLEAU_MINE_SEL = TABLEAU_MINE[TABLEAU_MINE['Mine'].isin(mines)].copy()

# --- 1) Shares annuels (Production, VIIRS)
TABLEAU_MINE_SEL['Total_Prod_annee'] = TABLEAU_MINE_SEL.groupby('Annee')['PROD_MINE_CAD_EN'].transform('sum')
TABLEAU_MINE_SEL['Share_Prod'] = TABLEAU_MINE_SEL['PROD_MINE_CAD_EN'] / TABLEAU_MINE_SEL['Total_Prod_annee']

TABLEAU_MINE_SEL['Total_VIIRS_annee'] = TABLEAU_MINE_SEL.groupby('Annee')['VIIRS_SUM_RESAMPL'].transform('sum')
TABLEAU_MINE_SEL['Share_VIIRS'] = TABLEAU_MINE_SEL['VIIRS_SUM_RESAMPL'] / TABLEAU_MINE_SEL['Total_VIIRS_annee']


# Shares emploi
TABLEAU_MINE_SEL['Total_Empl_annee'] = TABLEAU_MINE_SEL.groupby('Annee')['POP_ACTIVE'].transform('sum')
TABLEAU_MINE_SEL['Share_Empl'] = TABLEAU_MINE_SEL['POP_ACTIVE'] / TABLEAU_MINE_SEL['Total_Empl_annee']

#w_prod = 0.3
#w_viirs = 0.7
# --- 3) Score pondéré + normalisation (sécurité numérique)
#TABLEAU_MINE_SEL['Score_raw'] = w_prod*TABLEAU_MINE_SEL['Share_Prod'] + w_viirs*TABLEAU_MINE_SEL['Share_VIIRS']
#TABLEAU_MINE_SEL['Score'] = TABLEAU_MINE_SEL['Score_raw'] / TABLEAU_MINE_SEL.groupby('Annee')['Score_raw'].transform('sum')


# --- 4) PIB Nunavut (total) -> à partir de DN_DONNEES_NUNAVUT_2009_2022
# IMPORTANT: remplace 'PIB_EN' par le nom exact de la colonne PIB total dans DN_DONNEES_NUNAVUT_2009_2022
PIB_total = DN_DONNEES_NUNAVUT_2009_2022[['Annee','PIB_REEL']].rename(columns={'PIB_REEL':'PIB_Nunavut'})

TABLEAU_MINE_SEL = TABLEAU_MINE_SEL.merge(PIB_total, on='Annee', how='left')

#TABLEAU_MINE_SEL['Score_raw'] = (w_prod*TABLEAU_MINE_SEL['Share_Prod'] + w_viirs*TABLEAU_MINE_SEL['Share_VIIRS'] + w_empl*TABLEAU_MINE_SEL['Share_Empl'])
#TABLEAU_MINE_SEL['Score'] = TABLEAU_MINE_SEL['Score_raw'] / TABLEAU_MINE_SEL.groupby('Annee')['Score_raw'].transform('sum')

#TABLEAU_MINE_SEL['PIB_mine_estime'] = TABLEAU_MINE_SEL['Score'] * TABLEAU_MINE_SEL['PIB_Nunavut']
#TABLEAU_MINE_SEL['Contribution_pct'] = 100 * TABLEAU_MINE_SEL['Score']


# Résultat final
TABLE_FINAL_SCORE = TABLEAU_MINE_SEL.sort_values(['Annee','Mine']).reset_index(drop=True)


###########################################################################################
#####   GRILLE DE POIDS -- MODELE SCORE PONDERE vs VAEMP
###########################################################################################

# ============================================================
# REFERENCE : VAEMP_EN (Valeur Ajoutee Miniere officielle)
# ============================================================
VAEMP_REF = DN_DONNEES_NUNAVUT_2009_2022[['Annee', 'VAEMP_EN']].copy()
VAEMP_REF['Annee'] = VAEMP_REF['Annee'].astype(int)

# ============================================================
# BASE : 5 entites (4 mines + nunavut_reste) pour normalisation correcte
# Les shares ont ete calcules dans TABLEAU_MINE_SEL avec la somme de toutes entites
# ============================================================
all_entities  = ['Mine_Meadowbank', 'Mine_Meliadine', 'Mine_Hope_Bay',
                 'Mine_Baffinland', 'nunavut_reste']
mines_list_4  = ['Mine_Meadowbank', 'Mine_Meliadine', 'Mine_Hope_Bay', 'Mine_Baffinland']

TABLEAU_MINE_SCORE = TABLEAU_MINE_SEL[TABLEAU_MINE_SEL['Mine'].isin(all_entities)][
    ['Annee', 'Mine', 'Share_Prod', 'Share_VIIRS', 'Share_Empl', 'PIB_Nunavut']
].copy()
TABLEAU_MINE_SCORE[['Share_Prod', 'Share_VIIRS', 'Share_Empl']] = (
    TABLEAU_MINE_SCORE[['Share_Prod', 'Share_VIIRS', 'Share_Empl']].fillna(0)
)

# ============================================================
# FONCTION UTILITAIRE : PIB total des 4 mines pour un jeu de poids
# ============================================================
def compute_pib_4mines(df_b, mines_4, w_p, w_v, w_e):
    df_w = df_b.copy()
    df_w['Score_raw'] = (
        w_p * df_w['Share_Prod'] +
        w_v * df_w['Share_VIIRS'] +
        w_e * df_w['Share_Empl']
    )
    total_score = df_w.groupby('Annee')['Score_raw'].transform('sum')
    df_w['Score']       = df_w['Score_raw'] / total_score
    df_w['PIB_estime']  = df_w['Score'] * df_w['PIB_Nunavut']
    pib = (df_w[df_w['Mine'].isin(mines_4)]
           .groupby('Annee')['PIB_estime'].sum()
           .reset_index()
           .rename(columns={'PIB_estime': 'PIB_4mines'}))
    return pib

# ============================================================
# GRILLE DES POIDS  (pas = 0.1,  w_prod + w_viirs + w_empl = 1)
# ============================================================
step = 0.1
vals = [round(k * step, 1) for k in range(0, 11)]
weights_grid = [
    (wp, wv, round(1.0 - wp - wv, 1))
    for wp in vals for wv in vals
    if round(1.0 - wp - wv, 1) >= 0.0
]
print(f"Nombre de combinaisons testees : {len(weights_grid)}")

# ============================================================
# BOUCLE -- calcul RMSE, MAE, R2, Corr pour chaque combinaison
# ============================================================
results_poids = []
for w_p, w_v, w_e in weights_grid:
    pib_pred = compute_pib_4mines(TABLEAU_MINE_SCORE, mines_list_4, w_p, w_v, w_e)
    comp = pib_pred.merge(VAEMP_REF, on='Annee', how='inner').dropna()
    if len(comp) < 3:
        continue
    y_true = comp['VAEMP_EN'].values
    y_pred = comp['PIB_4mines'].values
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    results_poids.append({
        'w_prod': w_p, 'w_viirs': w_v, 'w_empl': w_e,
        'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Corr': corr,
    })

TABLE_POIDS = (pd.DataFrame(results_poids)
               .sort_values('RMSE')
               .reset_index(drop=True))

print("\n===== TOP 15 combinaisons (RMSE minimal) =====")
print(TABLE_POIDS[['w_prod','w_viirs','w_empl','RMSE','MAE','R2','Corr']]
      .head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best_rmse = TABLE_POIDS.iloc[0]
best_r2   = TABLE_POIDS.sort_values('R2', ascending=False).iloc[0]
print(f"\n  Meilleur RMSE : w_prod={best_rmse['w_prod']:.1f}  w_viirs={best_rmse['w_viirs']:.1f}"
      f"  w_empl={best_rmse['w_empl']:.1f}  |  RMSE={best_rmse['RMSE']:,.1f}  R2={best_rmse['R2']:.4f}")
print(f"  Meilleur R2   : w_prod={best_r2['w_prod']:.1f}   w_viirs={best_r2['w_viirs']:.1f}"
      f"  w_empl={best_r2['w_empl']:.1f}  |  RMSE={best_r2['RMSE']:,.1f}  R2={best_r2['R2']:.4f}")

# ============================================================
# EXPORT EXCEL : grille complete
# ============================================================
output_poids = os.path.join(tab_dir, "tableau_grille_poids.xlsx")
with pd.ExcelWriter(output_poids, engine='openpyxl') as writer:
    TABLE_POIDS[['w_prod','w_viirs','w_empl','RMSE','MAE','R2','Corr']].to_excel(
        writer, sheet_name='Grille_poids', index=False)
print(f"\nTableau des poids exporte : {output_poids}")


# ============================================================
# EXPORT LATEX : grille complete
# ============================================================
output_poids = os.path.join(tab_dir, "tableau_grille_poids.tex")

_cols_lt    = ['w_prod', 'w_viirs', 'w_empl', 'RMSE', 'MAE', 'R2', 'Corr']
_headers_lt = [r'$w_{\mathrm{prod}}$', r'$w_{\mathrm{viirs}}$', r'$w_{\mathrm{empl}}$',
               'RMSE', 'MAE', r'$R^2$', 'Corr']
_n_cols_lt  = len(_headers_lt)
_caption_lt = ("Grille de pondérations testées et indicateurs de performance "
               "(RMSE, MAE, $R^2$, corrélation) du score composite.")

with open(output_poids, 'w', encoding='utf-8') as _f:
    _f.write(r"\begin{longtable}{" + "r" * _n_cols_lt + "}" + "\n")
    _f.write(r"\caption{" + _caption_lt + r"}\label{tab:grille_poids}\\" + "\n")
    _f.write(r"\toprule" + "\n")
    _f.write(" & ".join(_headers_lt) + r" \\" + "\n")
    _f.write(r"\midrule" + "\n")
    _f.write(r"\endfirsthead" + "\n")
    _f.write(r"\multicolumn{" + str(_n_cols_lt)
             + r"}{l}{\textit{(suite du tableau \ref{tab:grille_poids})}}\\" + "\n")
    _f.write(r"\toprule" + "\n")
    _f.write(" & ".join(_headers_lt) + r" \\" + "\n")
    _f.write(r"\midrule" + "\n")
    _f.write(r"\endhead" + "\n")
    _f.write(r"\midrule" + "\n")
    _f.write(r"\multicolumn{" + str(_n_cols_lt)
             + r"}{r}{\textit{(suite à la page suivante)}}\\" + "\n")
    _f.write(r"\endfoot" + "\n")
    _f.write(r"\bottomrule" + "\n")
    _f.write(r"\endlastfoot" + "\n")
    for _, _row in TABLE_POIDS[_cols_lt].iterrows():
        _f.write(
            f"{_row['w_prod']:.1f} & {_row['w_viirs']:.1f} & {_row['w_empl']:.1f} & "
            f"{_row['RMSE']:.4f} & {_row['MAE']:.4f} & {_row['R2']:.4f} & {_row['Corr']:.4f}"
            r" \\" + "\n"
        )
    _f.write(r"\end{longtable}" + "\n")

print(f"\nTableau des poids exporte : {output_poids}")


#%% GRAPHIQUE SCORE

# ============================================================
# GRAPHIQUE  : Serie temporelle  VAEMP vs PIB predit
#               Panneau superieur : niveaux  |  inferieur : residus
# ============================================================
series_to_plot = [
    {
        'label': (f"Meilleur RMSE  "
                  f"(w_p={best_rmse['w_prod']:.1f}, w_v={best_rmse['w_viirs']:.1f},"
                  f" w_e={best_rmse['w_empl']:.1f})"),
        'wp': best_rmse['w_prod'], 'wv': best_rmse['w_viirs'], 'we': best_rmse['w_empl'],
        'ls': '-',  'lw': 2.0, 'marker': 's', 'color': '#e41a1c',
    },
    {
        'label': (f"Meilleur R\u00b2    "
                  f"(w_p={best_r2['w_prod']:.1f}, w_v={best_r2['w_viirs']:.1f},"
                  f" w_e={best_r2['w_empl']:.1f})"),
        'wp': best_r2['w_prod'], 'wv': best_r2['w_viirs'], 'we': best_r2['w_empl'],
        'ls': '--', 'lw': 1.8, 'marker': '^', 'color': '#377eb8',
    },
    {
        'label': "Poids actuels  (w_p=0.3, w_v=0.4, w_e=0.3)",
        'wp': 0.3, 'wv': 0.4, 'we': 0.3,
        'ls': '-.', 'lw': 1.5, 'marker': 'D', 'color': '#4daf4a',
    },
    {
        'label': "Poids egaux    (w_p=0.3, w_v=0.3, w_e=0.4)",
        'wp': 0.3, 'wv': 0.3, 'we': 0.4,
        'ls': ':',  'lw': 1.5, 'marker': 'o', 'color': '#984ea3',
    },
]

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 9,  'figure.titlesize': 13,
    'axes.grid': True, 'grid.alpha': 0.3,
})

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(13, 9),
    gridspec_kw={'height_ratios': [2.5, 1]},
    sharex=True
)
#fig.suptitle(
#    "Sensibilite aux poids du Score pondere\n"
#    "PIB minier estime (4 mines) vs Valeur Ajoutee de l'Extraction Miniere (VAEM)",
#    fontsize=13, fontweight='bold', y=0.99
#)

vaemp_ann  = VAEMP_REF['Annee'].astype(int).values
vaemp_vals = VAEMP_REF['VAEMP_EN'].values

ax1.plot(vaemp_ann, vaemp_vals, 'k-o', lw=2.5, ms=6, zorder=5,
         label='VAEM officielle (Statistique Canada)')

for s in series_to_plot:
    pib_df = compute_pib_4mines(TABLEAU_MINE_SCORE, mines_list_4, s['wp'], s['wv'], s['we'])
    pib_df = pib_df.merge(VAEMP_REF[['Annee']], on='Annee').sort_values('Annee')
    ax1.plot(pib_df['Annee'], pib_df['PIB_4mines'],
             linestyle=s['ls'], lw=s['lw'], color=s['color'],
             marker=s['marker'], ms=5, label=s['label'])
    vaemp_align = (VAEMP_REF.set_index('Annee')['VAEMP_EN']
                   .reindex(pib_df['Annee'].values))
    resid = pib_df['PIB_4mines'].values - vaemp_align.values
    ax2.plot(pib_df['Annee'], resid,
             linestyle=s['ls'], lw=1.2, color=s['color'],
             marker=s['marker'], ms=4)

ax2.axhline(0, color='k', lw=1.5, linestyle='-')
ax2.fill_between(vaemp_ann, 0, 0)

ax1.set_ylabel('Valeur ajoutee miniere (M$ CAD 2017)', fontsize=11)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

ax2.set_ylabel('Residu\n(Predit - VAEM)', fontsize=10)
ax2.set_xlabel('Annee', fontsize=11)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax2.set_xticks(range(int(vaemp_ann.min()), int(vaemp_ann.max()) + 1))
ax2.set_xticklabels(
    range(int(vaemp_ann.min()), int(vaemp_ann.max()) + 1),
    rotation=45, ha='right')

plt.tight_layout()
fig1_path = os.path.join(fig_dir, "sensibilite_poids_score.png")
plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure 1 enregistree : {fig1_path}")
plt.show()




#%% TABLEAU COMPARATIF PIB PAR MINE ET PAR MODELE 
###########################################################################################
#####   TABLEAU COMPARATIF PIB PAR MINE ET PAR MODELE -- Annee en ligne, Mine en colonne
#####   Chaque modele conserve sa propre plage de donnees disponibles
###########################################################################################

mines_list = ['Mine_Meadowbank', 'Mine_Meliadine', 'Mine_Hope_Bay', 'Mine_Baffinland']
mines_labels = {
    'Mine_Meadowbank': 'Meadowbank',
    'Mine_Meliadine':  'Meliadine',
    'Mine_Hope_Bay':   'Hope Bay',
    'Mine_Baffinland': 'Baffinland',
}


# Plages reelles par methode :
#   Emploi   : POPULATION_DN_PIB_2009_2024  -> 2009-2024
#   OLS/IV   : TABLE_VIIRS_MINE_2005_2024   -> 2005-2024
#   Chen     : TABLE_MINE_PIB_SEL           -> 2000-2024
#   Score*   : TABLEAU_MINE_SCORE + best_rmse    -> 2009-2022  (meilleurs poids RMSE)
#   StatCan  : TABLE_DN_SAM (SAM + VAOA)    -> 2005-2024

# ============================================================
# 1. METHODE EMPLOI  (toute la plage de POPULATION_DN_PIB_2009_2024)
# ============================================================
cols_map = {
    'Mine_Meadowbank': ('PIB_MBK', 'TX_PIB_MBK'),
    'Mine_Meliadine':  ('PIB_MEL', 'TX_PIB_MEL'),
    'Mine_Hope_Bay':   ('PIB_HB',  'TX_PIB_HB'),
    'Mine_Baffinland': ('PIB_MR',  'TX_PIB_MR'),
}

rows_empl = []
for mine, (col_pib, col_tx) in cols_map.items():
    for _, row in POPULATION_DN_PIB_2009_2024.iterrows():
        rows_empl.append({
            'Annee':      int(row['Annee']),
            'Mine':       mine,
            'PIB_Emploi': pd.to_numeric(row[col_pib], errors='coerce'),
            'TX_Emploi':  pd.to_numeric(row[col_tx],  errors='coerce'),
        })
TABLE_EMPL_LONG = pd.DataFrame(rows_empl)

# ============================================================
# 2. METHODES LOG-OLS et LOG-IV  (toute la plage 2005-2024)
# ============================================================
df_log_iv = TABLE_VIIRS_MINE_2005_2024[
    TABLE_VIIRS_MINE_2005_2024['Nom'].isin(mines_list)
][['Annee', 'Nom',
   'PRED_PIB_MINE_LOG',    'TX_PRED_PIB_MINE_LOG',
   'PRED_PIB_MINE_LOG_IV', 'TX_PRED_PIB_MINE_LOG_IV']].copy()

df_log_iv = df_log_iv.rename(columns={
    'Nom':                     'Mine',
    'PRED_PIB_MINE_LOG':       'PIB_OLS',
    'TX_PRED_PIB_MINE_LOG':    'TX_OLS',
    'PRED_PIB_MINE_LOG_IV':    'PIB_IV',
    'TX_PRED_PIB_MINE_LOG_IV': 'TX_IV',
})

# ============================================================
# 3. METHODE CHEN  (toute la plage disponible de TABLE_MINE_PIB_SEL)
#    TX = PIB_MINE / PIB_EN  (% du PIB officiel en dollars 2017)
# ============================================================
df_chen_sel = TABLE_MINE_PIB_SEL[
    TABLE_MINE_PIB_SEL['Mine'].isin(mines_list)
][['Annee', 'Mine', 'PIB_MINE', 'PIB_EN']].copy()

df_chen_sel['PIB_MINE'] = pd.to_numeric(df_chen_sel['PIB_MINE'], errors='coerce')
df_chen_sel['TX_CHEN']  = (TABLE_MINE_PIB_SEL["POURCENTAGE_PIB"])
df_chen_sel = df_chen_sel.rename(columns={'PIB_MINE': 'PIB_CHEN'})
df_chen_sel = df_chen_sel[['Annee', 'Mine', 'PIB_CHEN', 'TX_CHEN']]

# ============================================================
# 4. METHODE SCORE PONDERE -- MEILLEURS POIDS (RMSE minimal)
#    best_rmse deja calcule dans la GRILLE DE POIDS (section precedente)
#    TABLEAU_MINE_SCORE contient 5 entites (4 mines + nunavut_reste)
#    pour que la normalisation du score soit correcte (somme = 1 par annee)
# ============================================================
_sb = TABLEAU_MINE_SCORE.copy()
_sb['Score_raw'] = (best_rmse['w_prod']  * _sb['Share_Prod']  +
                    best_rmse['w_viirs'] * _sb['Share_VIIRS'] +
                    best_rmse['w_empl']  * _sb['Share_Empl'])
_sb['Score']     = (_sb['Score_raw']
                    / _sb.groupby('Annee')['Score_raw'].transform('sum'))
_sb['PIB_SCORE'] = _sb['Score'] * _sb['PIB_Nunavut']
_sb['TX_SCORE']  = _sb['Score'] * 100   # % du PIB total Nunavut

df_score_sel = (
    _sb[_sb['Mine'].isin(mines_list)]
    [['Annee', 'Mine', 'PIB_SCORE', 'TX_SCORE']]
    .copy()
    .reset_index(drop=True)
)
print(f"  Score* meilleurs poids : w_prod={best_rmse['w_prod']:.1f},"
      f" w_viirs={best_rmse['w_viirs']:.1f},"
      f" w_empl={best_rmse['w_empl']:.1f}"
      f" | RMSE={best_rmse['RMSE']:,.1f} M$")

# ============================================================
# 5. METHODE STATISTIQUE CANADA  (SAM alloué par VIIRS + VAOA)
#    PIB_mine = SAM_mine + VAOA_mine  (or) / SAM_mine + VAF_EN (fer)
#    Source   : TABLE_DN_SAM  — plage 2005-2024 (valeurs NaN avant 2005)
# ============================================================
_sc_cols_map = {
    'Mine_Meadowbank': ('PIB_MBK_SC', 'TX_PIB_MBK_SC'),
    'Mine_Meliadine':  ('PIB_MEL_SC', 'TX_PIB_MEL_SC'),
    'Mine_Hope_Bay':   ('PIB_HB_SC',  'TX_PIB_HB_SC'),
    'Mine_Baffinland': ('PIB_BAF_SC', 'TX_PIB_BAF_SC'),
}
_rows_sc = []
for _mine, (_col_pib, _col_tx) in _sc_cols_map.items():
    for _, _row in TABLE_DN_SAM.reset_index().iterrows():
        _rows_sc.append({
            'Annee':  int(_row['Annee']),
            'Mine':   _mine,
            'PIB_SC': pd.to_numeric(_row.get(_col_pib), errors='coerce'),
            'TX_SC':  pd.to_numeric(_row.get(_col_tx),  errors='coerce'),
        })
df_statcan_long = pd.DataFrame(_rows_sc)

# ============================================================
# FUSION COMPLETE (format long, outer join sur toutes les annees)
# NaN = methode non disponible pour cette annee
# ============================================================
TABLE_COMP = (
    TABLE_EMPL_LONG
    .merge(df_log_iv,       on=['Annee', 'Mine'], how='outer')
    .merge(df_chen_sel,     on=['Annee', 'Mine'], how='outer')
    .merge(df_score_sel,    on=['Annee', 'Mine'], how='outer')
    .merge(df_statcan_long, on=['Annee', 'Mine'], how='outer')
)
TABLE_COMP = TABLE_COMP.sort_values(['Annee', 'Mine']).reset_index(drop=True)
TABLE_COMP['Mine_Label'] = TABLE_COMP['Mine'].map(mines_labels).fillna(TABLE_COMP['Mine'])

# ============================================================
# TABLEAUX PIVOTES : Annee (lignes) x Mine (colonnes)
# Chaque tableau ne montre QUE les annees ou la methode a des donnees
# Colonnes alternees M$ / %  |  une feuille Excel par modele
# ============================================================
modeles_info = {
    'Emploi':  ('PIB_Emploi', 'TX_Emploi'),
    'OLS':     ('PIB_OLS',    'TX_OLS'),
    'IV':      ('PIB_IV',     'TX_IV'),
    'Chen':    ('PIB_CHEN',   'TX_CHEN'),
    'Score':   ('PIB_SCORE',  'TX_SCORE'),
    'StatCan': ('PIB_SC',     'TX_SC'),
}

mines_order = list(mines_labels.values())
pivots = {}

for modele, (col_pib, col_tx) in modeles_info.items():

    # Annees pour lesquelles ce modele a au moins une valeur non-NaN
    df_mod     = TABLE_COMP.dropna(subset=[col_pib])
    annees_mod = sorted(df_mod['Annee'].dropna().unique().astype(int))

    if not annees_mod:
        print(f"  MODELE {modele} : aucune donnee disponible.")
        continue

    df_filtre = TABLE_COMP[TABLE_COMP['Annee'].isin(annees_mod)]

    # Pivot M$
    pib_piv = df_filtre.pivot_table(
        index='Annee', columns='Mine_Label', values=col_pib, aggfunc='mean'
    ).reindex(columns=mines_order)
    pib_piv['Total (M$)'] = pib_piv.sum(axis=1, min_count=1)

    # Pivot %
    tx_piv = df_filtre.pivot_table(
        index='Annee', columns='Mine_Label', values=col_tx, aggfunc='mean'
    ).reindex(columns=mines_order)
    tx_piv['Total (%)'] = tx_piv.sum(axis=1, min_count=1)

    # MultiIndex alternee : (Mine, M$), (Mine, %), ..., (Total, M$), (Total, %)
    df_pib = pib_piv.copy()
    df_pib.columns = pd.MultiIndex.from_tuples(
        [(c, 'M$') for c in mines_order] + [('Total', 'M$')]
    )
    df_tx = tx_piv.copy()
    df_tx.columns = pd.MultiIndex.from_tuples(
        [(c, '%') for c in mines_order] + [('Total', '%')]
    )
    interleaved_cols = []
    for m in mines_order:
        interleaved_cols += [(m, 'M$'), (m, '%')]
    interleaved_cols += [('Total', 'M$'), ('Total', '%')]

    combined = pd.concat([df_pib, df_tx], axis=1)[interleaved_cols]
    combined.index.name = 'Annee'
    pivots[modele] = combined

    # Affichage console
    print("\n" + "="*75)
    print(f"  MODELE : {modele}  |  {min(annees_mod)}-{max(annees_mod)}"
          f"  ({len(annees_mod)} annees)  |  M$ CAD 2017")
    print("="*75)
    print(pib_piv.to_string(float_format=lambda x: f"{x:>10,.1f}"))
    print(f"\n  Pourcentages du PIB Nunavut (%):")
    print(tx_piv.to_string(float_format=lambda x: f"{x:>8.1f}"))

# ============================================================
# EXPORT EXCEL -- une feuille par modele + feuille Donnees_Long complete
# ============================================================
output_path = os.path.join(tab_dir, "tableau_pib_mines_pivot.xlsx")

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for modele, pivot_df in pivots.items():
        pivot_df.to_excel(writer, sheet_name=modele)

    TABLE_COMP_OUT = (TABLE_COMP
                      .drop(columns='Mine')
                      .rename(columns={'Mine_Label': 'Mine'}))
    TABLE_COMP_OUT.to_excel(writer, sheet_name='Donnees_Long', index=False)

print(f"\nTableaux pivotes exportes : {output_path}")

# ============================================================
# EXPORT LATEX AMELIORE -- un fichier .tex par modele + Donnees_Long
# ============================================================
latex_dir = tab_dir
os.makedirs(latex_dir, exist_ok=True)

for modele, pivot_df in pivots.items():
    output_path = os.path.join(latex_dir, f"tableau_pib_mines_{modele}.tex")

    _mines_cols = [m for m in mines_order if (m, 'M$') in pivot_df.columns]
    n_groupes   = len(_mines_cols) + 1  # mines + Total
    col_fmt     = 'l' + ('rr' * n_groupes)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"  \centering" + "\n")
        f.write(r"  \footnotesize" + "\n")
        f.write(r"  \setlength{\tabcolsep}{4pt}" + "\n")
        f.write(f"  \\caption{{PIB des mines du Nunavut --- Méthode {modele} "
                f"(M\\$ CAD 2017)}}\n")
        f.write(r"  \label{tab:pib_mines_" + modele.lower() + "}\n")
        f.write(r"  \begin{tabular}{" + col_fmt + "}\n")
        f.write(r"    \hline\hline" + "\n")

        # En-tête niveau 1 : noms de mines + Total
        _h1 = [r"\multirow{2}{*}{Année}"]
        for m in _mines_cols:
            _h1.append(r"\multicolumn{2}{c}{" + m + "}")
        _h1.append(r"\multicolumn{2}{c}{Total}")
        f.write("    " + " & ".join(_h1) + r" \\" + "\n")

        # cmidrule pour chaque paire (M$ | %)
        _cmid = []
        for i in range(n_groupes):
            c1 = 2 + 2 * i
            c2 = c1 + 1
            _cmid.append(r"\cmidrule(lr){" + f"{c1}-{c2}" + "}")
        f.write("    " + "".join(_cmid) + "\n")

        # En-tête niveau 2 : M$ / %
        _h2 = [""]
        for _ in range(n_groupes):
            _h2.append(r"M\$")
            _h2.append(r"\%")
        f.write("    " + " & ".join(_h2) + r" \\" + "\n")
        f.write(r"    \hline" + "\n")

        # Lignes de données
        for annee in pivot_df.index:
            cells = [str(int(annee))]
            for m in _mines_cols + ['Total']:
                val_m = pivot_df.loc[annee, (m, 'M$')] if (m, 'M$') in pivot_df.columns else np.nan
                val_p = pivot_df.loc[annee, (m, '%')]  if (m, '%')  in pivot_df.columns else np.nan
                cells.append(_fmt_fr(val_m, 1))
                cells.append(_fmt_fr(val_p, 1))
            f.write("    " + " & ".join(cells) + r" \\" + "\n")

        f.write(r"    \hline\hline" + "\n")
        f.write(r"  \end{tabular}" + "\n")
        f.write(r"  \begin{minipage}{\linewidth}" + "\n")
        f.write(r"    \vspace{2pt}\footnotesize \textit{Note :} "
                r"Valeurs en millions de dollars canadiens enchaînés de 2017. "
                r"Les colonnes \%\ donnent la part du PIB territorial du Nunavut. "
                r"Requiert \texttt{\textbackslash usepackage\{multirow,booktabs\}}." + "\n")
        f.write(r"  \end{minipage}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"  Tableau LaTeX exporté : {output_path}")

    # Export Excel individuel par modèle
    _xl_mod_path = os.path.join(tab_dir, f"tableau_pib_mines_{modele}.xlsx")
    with pd.ExcelWriter(_xl_mod_path, engine='openpyxl') as _writer:
        # Feuille M$ — pivot Annee × Mine
        _pib_xl = pivot_df[[c for c in pivot_df.columns if c[1] == 'M$']].copy()
        _pib_xl.columns = [m for m, _ in _pib_xl.columns]
        _pib_xl.reset_index(inplace=True)
        _pib_xl.to_excel(_writer, sheet_name='Valeurs_M$', index=False)

        # Feuille % — pivot Annee × Mine
        _tx_xl = pivot_df[[c for c in pivot_df.columns if c[1] == '%']].copy()
        _tx_xl.columns = [m for m, _ in _tx_xl.columns]
        _tx_xl.reset_index(inplace=True)
        _tx_xl.to_excel(_writer, sheet_name='Part_PIB_pct', index=False)

        # Feuille combinée (M$ et % interleaved)
        _comb_xl = pivot_df.copy()
        _comb_xl.columns = [f"{m} ({u})" for m, u in _comb_xl.columns]
        _comb_xl.reset_index(inplace=True)
        _comb_xl.to_excel(_writer, sheet_name='Combine', index=False)
    print(f"  Tableau Excel exporté  : {_xl_mod_path}")

# Export du tableau long avec améliorations LaTeX
output_path_long = os.path.join(latex_dir, "tableau_pib_mines_donnees_long.tex")

TABLE_COMP_OUT = (
    TABLE_COMP
    .drop(columns='Mine')
    .rename(columns={'Mine_Label': 'Mine'})
)

# Formatage des nombres pour le tableau long (français : virgule + espace fine)
df_long_latex = TABLE_COMP_OUT.copy()
cols_pib = ['PIB_Emploi', 'PIB_OLS', 'PIB_IV', 'PIB_CHEN', 'PIB_SCORE', 'PIB_SC']
cols_tx  = ['TX_Emploi',  'TX_OLS',  'TX_IV',  'TX_CHEN',  'TX_SCORE',  'TX_SC']

for col in cols_pib:
    if col in df_long_latex.columns:
        df_long_latex[col] = df_long_latex[col].apply(lambda x: _fmt_fr(x, 1))

for col in cols_tx:
    if col in df_long_latex.columns:
        df_long_latex[col] = df_long_latex[col].apply(lambda x: _fmt_fr(x, 1))

# Renommer les colonnes pour un affichage propre en LaTeX (échappe $ et %)
_rename_long = {
    'PIB_Emploi': r'Emploi (M\$)', 'TX_Emploi': r'Emploi (\%)',
    'PIB_OLS':    r'OLS (M\$)',    'TX_OLS':    r'OLS (\%)',
    'PIB_IV':     r'IV (M\$)',     'TX_IV':     r'IV (\%)',
    'PIB_CHEN':   r'Chen (M\$)',   'TX_CHEN':   r'Chen (\%)',
    'PIB_SCORE':  r'Score (M\$)',  'TX_SCORE':  r'Score (\%)',
    'PIB_SC':     r'StatCan (M\$)','TX_SC':     r'StatCan (\%)',
}
df_long_latex = df_long_latex.rename(columns=_rename_long)

# Export du tableau long avec en-tête LaTeX
with open(output_path_long, 'w', encoding='utf-8') as f:
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Données longues du PIB des mines du Nunavut --- "
            "Toutes méthodes (M\\$ CAD 2017 et \\% du PIB territorial)}\n")
    f.write("\\label{tab:pib_mines_long}\n")
    f.write("\\scriptsize\n")  # Police plus petite pour le tableau long

    latex_str = df_long_latex.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'r' * len(df_long_latex.columns)
    )

    latex_str = latex_str.replace('\\toprule',    '\\hline\\hline')
    latex_str = latex_str.replace('\\midrule',    '\\hline')
    latex_str = latex_str.replace('\\bottomrule', '\\hline\\hline')

    f.write(latex_str)
    f.write("\\end{table}\n")

print("\nExport LaTeX amélioré terminé.")

# ============================================================
# FONCTION D'EXPORT POUR TABLEAUX COMPARATIFS MULTI-MÉTHODES
# ============================================================
def export_tableau_comparatif_latex(pivots, output_path):
    """
    Exporte un tableau comparatif de toutes les méthodes pour une année sélectionnée.
    Format : Mine et Total sur 1 colonne, valeur 'M$ (%)' formatée en français.
    """
    n_cols = len(mines_order) + 2  # Année + N mines + Total
    col_fmt = 'l' + 'r' * (n_cols - 1)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"  \centering" + "\n")
        f.write(r"  \footnotesize" + "\n")
        f.write(r"  \setlength{\tabcolsep}{4pt}" + "\n")
        f.write(r"  \caption{Comparaison des méthodes d'estimation du PIB minier "
                r"(M\$ CAD 2017, part en \% du PIB territorial)}" + "\n")
        f.write(r"  \label{tab:pib_comparatif_methodes}" + "\n")
        f.write(r"  \begin{tabular}{" + col_fmt + "}" + "\n")
        f.write(r"    \hline\hline" + "\n")

        # En-tête
        _hdr = ["Année"] + list(mines_order) + ["Total"]
        f.write("    " + " & ".join(_hdr) + r" \\" + "\n")
        f.write(r"    \hline" + "\n")

        # Données pour chaque méthode
        for methode in pivots.keys():
            df = pivots[methode]
            f.write(r"    \multicolumn{" + str(n_cols) + r"}{l}{\textbf{Méthode "
                    + methode + r"}} \\" + "\n")

            for annee in df.index:
                cells = [str(int(annee))]
                for mine in list(mines_order) + ['Total']:
                    val_m = df.loc[annee, (mine, 'M$')] if (mine, 'M$') in df.columns else np.nan
                    val_p = df.loc[annee, (mine, '%')]  if (mine, '%')  in df.columns else np.nan
                    if pd.notna(val_m) and pd.notna(val_p):
                        cells.append(f"{_fmt_fr(val_m, 1)} ({_fmt_fr(val_p, 1)}\\%)")
                    else:
                        cells.append("---")
                f.write("    " + " & ".join(cells) + r" \\" + "\n")
            f.write(r"    \hline" + "\n")

        f.write(r"    \hline" + "\n")
        f.write(r"  \end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

# Export du tableau comparatif
output_path_comp = os.path.join(latex_dir, "tableau_pib_comparatif_methodes.tex")
export_tableau_comparatif_latex(pivots, output_path_comp)
print(f"Tableau comparatif des méthodes exporté : {output_path_comp}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT EXCEL — PIB PAR MINE (toutes méthodes + données longues)
# ══════════════════════════════════════════════════════════════════════════════
_xl_pib_path = os.path.join(tab_dir, "tableau_pib_mines_all_modeles.xlsx")
with pd.ExcelWriter(_xl_pib_path, engine='openpyxl') as _writer:
    # Une feuille par modèle (pivot Annee × Mine, M$ et %)
    for _modele, _pivot_df in pivots.items():
        # Aplatir le MultiIndex de colonnes pour Excel
        _df_xl = _pivot_df.copy()
        _df_xl.columns = [f"{m} ({u})" for m, u in _df_xl.columns]
        _df_xl.reset_index(inplace=True)
        _sheet = _modele[:31]  # Excel limite les noms de feuilles à 31 caractères
        _df_xl.to_excel(_writer, sheet_name=_sheet, index=False)

    # Feuille données longues (toutes méthodes, format long)
    TABLE_COMP_OUT.to_excel(_writer, sheet_name='Donnees_longues', index=False)

    # Feuille résumé comparatif (moyenne M$ par mine × méthode)
    _rows_comp = []
    for _modele, _pivot_df in pivots.items():
        for _mine in mines_order:
            if (_mine, 'M$') in _pivot_df.columns:
                _avg = _pivot_df[(_mine, 'M$')].mean(skipna=True)
                _tx  = _pivot_df[(_mine, '%')].mean(skipna=True) if (_mine, '%') in _pivot_df.columns else np.nan
                _rows_comp.append({'Modèle': _modele, 'Mine': _mine,
                                   'PIB moy. (M$)': round(_avg, 2),
                                   'Part moy. (%)': round(_tx, 2)})
    pd.DataFrame(_rows_comp).to_excel(_writer, sheet_name='Resume_comparatif', index=False)

print(f"  ✔ Excel PIB par mine sauvegardé : {_xl_pib_path}")

#%% GRAPHIQUES PIB PRÉDIT PAR MINE — UN GRAPHIQUE PAR MODÈLE
################################################################################
# GRAPHIQUES PIB PRÉDIT PAR MINE — UN GRAPHIQUE PAR MODÈLE  (style mémoire)
# Structure : 1 figure par modèle  ×  4 sous-graphiques (2×2), un par mine
# Chaque sous-graphique : diagramme en barres  Année → PIB prédit (M$ CAD 2017)
################################################################################

# ── Style académique global ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        12,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'axes.linewidth':   0.8,
    'axes.edgecolor':   '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'grid.color':       '#CCCCCC',
    'grid.linewidth':   0.6,
})

# ── Palette sobre (gris neutre + teinte distinctive par mine) ─────────────────
_COULEURS_MINE = {
    'Meadowbank': '#2C5F8A',   # bleu acier
    'Meliadine':  '#8A4A2C',   # brun rouille
    'Hope Bay':   '#2C7A4B',   # vert sauge
    'Baffinland': '#5C3D8A',   # indigo
}

# ── Titres concis style article scientifique ──────────────────────────────────
_TITRES_MODELE = {
    'Emploi': (
        "Contribution au PIB estimée par la méthode de l'emploi",
        "Part de l'emploi minier rapportée au PIB réel du Nunavut (dollars de 2017)"
    ),
    'OLS': (
        "Contribution au PIB estimée par la méthode OLS log-log",
        "Régression Log(PIB) ~ Log(luminosité VIIRS) — erreurs robustes HC3 (dollars de 2017)"
    ),
    'IV': (
        "Contribution au PIB estimée par la méthode IV2SLS log-log",
        "Variable instrumentale : prix de l'or — erreurs robustes HC (dollars de 2017)"
    ),
    'Chen': (
        "Contribution au PIB estimée par la méthode de Chen et al. (2011)",
        "Taux de croissance pondéré : luminosité VIIRS × croissance du PIB du Nunavut (dollars de 2017)"
    ),
    'Score': (
        "Contribution au PIB estimée par la méthode du score pondéré",
        f"Poids optimaux (RMSE minimal) : w_prod = {best_rmse['w_prod']:.1f}, "
        f"w_viirs = {best_rmse['w_viirs']:.1f}, "
        f"w_empl = {best_rmse['w_empl']:.1f} (dollars de 2017)"
    ),
}

# ── Noms officiels pour les sous-titres de sous-graphiques ───────────────────
_NOMS_MINE = {
    'Meadowbank': 'Mine Meadowbank (or, Agnico Eagle)',
    'Meliadine':  'Mine Meliadine (or, Agnico Eagle)',
    'Hope Bay':   'Mine Hope Bay (or, TMAC Resources)',
    'Baffinland': 'Mine Mary River — Baffinland (fer)',
}
# ── Boucle principale : un graphique par modèle ───────────────────────────────
for modele, (col_pib, _col_tx) in modeles_info.items():

    df_m = (
        TABLE_COMP[TABLE_COMP['Mine_Label'].isin(mines_order)]
        .dropna(subset=[col_pib])
        [['Annee', 'Mine_Label', col_pib]]
        .copy()
    )

    if df_m.empty:
        print(f"  [{modele}] Aucune donnée — graphique ignoré.")
        continue

    titre_principal, sous_titre = _TITRES_MODELE.get(
        modele, (modele, ""))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Titre principal + sous-titre
#    fig.text(0.5, 0.995, titre_principal,
#             ha='center', va='top',
#             fontsize=13, fontweight='bold', fontstyle='normal',        # ↑ 12 → 13
#             fontfamily='serif')
#    fig.text(0.5, 0.965, sous_titre,
#             ha='center', va='top',
#             fontsize=9, fontstyle='italic', color='#444444',           # ↑ 8.5 → 9
#             fontfamily='serif')

    # Calcul de l'échelle Y commune (optionnel : facilite la comparaison inter-mines)
    y_max_global = df_m[col_pib].max() if not df_m.empty else 1

    for ax, mine_label in zip(axes.flatten(), mines_order):

        df_mine = (df_m[df_m['Mine_Label'] == mine_label]
                   .sort_values('Annee')
                   .copy())

        nom_complet = _NOMS_MINE.get(mine_label, mine_label)
        color       = _COULEURS_MINE.get(mine_label, '#555555')

        if df_mine.empty:
            ax.set_title(nom_complet, fontsize=11, fontweight='bold',    # ↑ 10 → 11
                         pad=6, color=color)
            ax.text(0.5, 0.5, 'Données non disponibles pour ce modèle',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='#888888', fontstyle='italic')    # ↑ 9 → 10
            ax.set_axis_off()
            continue

        x_ann  = df_mine['Annee'].astype(int).values
        y_vals = df_mine[col_pib].values

        # ── Barres ────────────────────────────────────────────────────────────
        ax.bar(
            x_ann, y_vals,
            color=color,
            alpha=0.75,
            edgecolor='white',
            linewidth=0.4,
            width=0.75,
            zorder=2
        )

        # ── Axes ──────────────────────────────────────────────────────────────
        ax.set_title(nom_complet, fontsize=11, fontweight='bold',        # ↑ 10 → 11
                     pad=6, color=color)
        ax.set_xlabel('Année', fontsize=10, labelpad=4)                  # ↑ 9 → 10
        ax.set_ylabel('PIB estimé (M$ CAD, base 2017)', fontsize=10, labelpad=4)  # ↑ 9 → 10

        # Ticks en X : afficher toutes les années disponibles
        ax.set_xticks(x_ann)
        ax.set_xticklabels(
            [str(a) for a in x_ann],
            rotation=55, ha='right', fontsize=9                          # ↑ 8 → 9
        )

        # Ticks en Y : formatés avec séparateur de milliers
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:,.0f}')
        )
        ax.set_ylim(bottom=0, top=y_max_global * 1.08)

        # ── Grille horizontale uniquement ─────────────────────────────────────
        ax.yaxis.grid(True, linestyle=':', linewidth=0.6,
                      color='#AAAAAA', zorder=0)
        ax.set_axisbelow(True)

        # ── Spines : garder seulement bas + gauche ────────────────────────────
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.7)
        ax.spines['bottom'].set_linewidth(0.7)

        # ── Ligne de base à y = 0 ─────────────────────────────────────────────
        ax.axhline(0, color='#333333', linewidth=0.7, zorder=1)

    plt.tight_layout(rect=[0, 0, 1, 0.955])

    # ── Sauvegarde haute résolution (300 dpi pour impression) ─────────────────
    fig_path = os.path.join(fig_dir, f"PIB_mine_{modele}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✔ [{modele}] Figure sauvegardée : {fig_path}")

# ── Réinitialiser les paramètres matplotlib par défaut après les figures ──────
plt.rcdefaults()

print("\n  ✔ Tous les graphiques PIB par mine ont été générés (style mémoire).")

#%% GRAPHIQUES PIB PRÉDIT — UN GRAPHIQUE PAR MINE, TOUS LES MODÈLES
################################################################################
# GRAPHIQUES PIB PRÉDIT — UN GRAPHIQUE PAR MINE, TOUS LES MODÈLES  (style mémoire)
# Structure : 1 figure par mine (4 figures)
# Chaque figure : diagramme en barres groupées
#   – Axe X   : années (toutes les années disponibles pour la mine)
#   – Groupes  : 1 groupe par année, 1 barre par modèle
#   – Couleur  : une couleur distincte par modèle
#   – Axe Y    : PIB estimé en M$ CAD dollars de 2017
################################################################################

# ── Style académique global ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        10,
    'axes.titlesize':   13,                    # ↑ 12 → 13
    'axes.labelsize':   12,                    # ↑ 10 → 12
    'xtick.labelsize':  10,                    # ↑ 8.5 → 10
    'ytick.labelsize':  11,                    # ↑ 9 → 11
    'axes.linewidth':   0.8,
    'axes.edgecolor':   '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'grid.color':       '#CCCCCC',
    'grid.linewidth':   0.6,
})

# ── Modèles : libellé → colonne PIB dans TABLE_COMP ──────────────────────────
_MODELES_G = {
    'Emploi':  'PIB_Emploi',
    'OLS':     'PIB_OLS',
    'IV':      'PIB_IV',
    'Chen':    'PIB_CHEN',
    'Score':   'PIB_SCORE',
    'StatCan': 'PIB_SC',
}

# ── Palette de couleurs par modèle ────────────────────────────────────────────
_COULEURS_MODELE = {
    'Emploi':  '#2C5F8A',   # bleu acier
    'OLS':     '#C0392B',   # rouge
    'IV':      '#2C7A4B',   # vert sauge
    'Chen':    '#7D3C98',   # violet
    'Score':   '#D68910',   # ocre doré
    'StatCan': '#17A589',   # turquoise
}

# ── Motifs de hachures pour accessibilité en N&B ─────────────────────────────
_HACHURES_MODELE = {
    'Emploi':  '',
    'OLS':     '///',
    'IV':      '\\\\\\',
    'Chen':    '...',
    'Score':   'xxx',
    'StatCan': '---',
}

# ── Noms complets pour les titres ─────────────────────────────────────────────
_NOMS_MINE_COMPLET = {
    'Meadowbank': 'Mine Meadowbank (or, Agnico Eagle)',
    'Meliadine':  'Mine Meliadine (or, Agnico Eagle)',
    'Hope Bay':   'Mine Hope Bay (or, TMAC Resources)',
    'Baffinland': 'Mine Mary River — Baffinland (fer)',
}

# ── Plages temporelles officielles d'activité par mine ───────────────────────
_ANNEE_OUVERTURE = {
    'Meadowbank': 2010,
    'Meliadine':  2019,
    'Hope Bay':   2017,
    'Baffinland': 2014,
}

n_modeles = len(_MODELES_G)
bar_width  = 0.15                          # largeur d'une barre
offsets    = (np.arange(n_modeles) - (n_modeles - 1) / 2) * bar_width

# ── Boucle principale : un graphique par mine ─────────────────────────────────
for mine_label in mines_order:

    nom_complet  = _NOMS_MINE_COMPLET.get(mine_label, mine_label)
    annee_ouv    = _ANNEE_OUVERTURE.get(mine_label, 0)

    # Filtrer les données pour cette mine — garder les lignes avec ≥1 modèle disponible
    cols_pib_all = list(_MODELES_G.values())
    df_mine = (
        TABLE_COMP[TABLE_COMP['Mine_Label'] == mine_label]
        .copy()
        .dropna(subset=cols_pib_all, how='all')
        .sort_values('Annee')
        .reset_index(drop=True)
    )

    if df_mine.empty:
        print(f"  [{mine_label}] Aucune donnée — graphique ignoré.")
        continue

    annees_dispo = sorted(a for a in df_mine['Annee'].astype(int).unique() if a >= 2005)
    n_annees     = len(annees_dispo)
    x_pos        = np.arange(n_annees)

    # Largeur de figure adaptée au nombre d'années
    fig_width  = max(14, n_annees * 1.1)
    fig_height = 6.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # ── Barres groupées : une barre par modèle, groupées par année ────────────
    handles_legende = []
    for i, (modele, col_pib) in enumerate(_MODELES_G.items()):

        y_vals = np.full(n_annees, np.nan)
        for j, annee in enumerate(annees_dispo):
            row = df_mine[df_mine['Annee'] == annee]
            if not row.empty:
                val = pd.to_numeric(row[col_pib].values[0], errors='coerce')
                if pd.notna(val) and val > 0:
                    y_vals[j] = val

        bars = ax.bar(
            x_pos + offsets[i],
            y_vals,
            width=bar_width,
            label=modele,
            color=_COULEURS_MODELE[modele],
            hatch=_HACHURES_MODELE[modele],
            alpha=0.82,
            edgecolor='white',
            linewidth=0.3,
            zorder=2
        )
        # Patch pour la légende (avec hachure visible)
        handles_legende.append(
            mpatches.Patch(
                facecolor=_COULEURS_MODELE[modele],
                hatch=_HACHURES_MODELE[modele],
                edgecolor='#444444',
                linewidth=0.5,
                alpha=0.85,
                label=modele
            )
        )

    # ── Ligne verticale à l'année d'ouverture ─────────────────────────────────
    if annee_ouv in annees_dispo:
        x_ouv = annees_dispo.index(annee_ouv)
        ax.axvline(
            x_ouv - 0.5,
            color='#333333', linewidth=1.1,
            linestyle='--', zorder=3
        )
        ax.text(
            x_ouv - 0.45, ax.get_ylim()[1] * 0.97,
            f'Ouverture ({annee_ouv})',
            fontsize=8, color='#333333',                      # ↑ 7.5 → 8
            fontstyle='italic', va='top'
        )

    # ── Titres et axes ────────────────────────────────────────────────────────
    ax.set_title(
        nom_complet,
        fontsize=13, fontweight='bold', pad=10,               # ↑ 12 → 13
        fontfamily='serif'
    )
    ax.set_xlabel('Année', fontsize=12, labelpad=5)           # ↑ 10 → 12
    ax.set_ylabel('PIB estimé (M$ CAD, base 2017)', fontsize=12, labelpad=5)  # ↑ 10 → 12

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [str(a) for a in annees_dispo],
        rotation=50, ha='right', fontsize=10                  # ↑ 8.5 → 10
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}')
    )
    ax.set_ylim(bottom=0)
    # Recalculer le max après le tracé (NaN ignorés)
    y_max = df_mine[cols_pib_all].max().max()
    if pd.notna(y_max) and y_max > 0:
        ax.set_ylim(top=y_max * 1.15)

    # ── Grille horizontale légère ─────────────────────────────────────────────
    ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#BBBBBB', zorder=0)
    ax.set_axisbelow(True)

    # ── Spines : bas + gauche uniquement ──────────────────────────────────────
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)

    # ── Légende ───────────────────────────────────────────────────────────────
    ax.legend(
        handles=handles_legende,
        title='Modèle d\'estimation',
        title_fontsize=10,                                    # ↑ 9 → 10
        fontsize=10,                                          # ↑ 9 → 10
        loc='upper left',
        framealpha=0.92,
        edgecolor='#CCCCCC',
        borderpad=0.7,
        labelspacing=0.4
    )

    # ── Note de bas de figure ─────────────────────────────────────────────────
    fig.text(
        0.5, -0.01,
        ("Note : Les barres absentes signifient que le modèle ne dispose pas "
         "de données pour cette année. "
         "PIB exprimé en millions de dollars canadiens, base 2017."),
        ha='center', fontsize=8, fontstyle='italic', color='#555555',  # ↑ 7.5 → 8
        fontfamily='serif'
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    # ── Sauvegarde haute résolution (300 dpi) ─────────────────────────────────
    safe_name = mine_label.replace(' ', '_').replace('—', '-')
    fig_path  = os.path.join(fig_dir, f"PIB_modeles_{safe_name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✔ [{mine_label}] Figure sauvegardée : {fig_path}")

# ── Réinitialiser les paramètres matplotlib par défaut ────────────────────────
plt.rcdefaults()

print("\n  ✔ Tous les graphiques comparatifs (1 figure/mine, tous modèles) ont été générés.")
#%% GRAPHIQUE : Luminosité nocturne du Nunavut
################################################################################
# GRAPHIQUE : Luminosité nocturne du Nunavut
#   — DMSP_SUM          (2000–2013, axe droit)
#   — VIIRS_SUM_RESAMPL (2012–2024, axe gauche)
#   Double axe Y pour respecter les unités hétérogènes des deux capteurs
################################################################################

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        12,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'axes.linewidth':   0.8,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'grid.color':       '#CCCCCC',
    'grid.linewidth':   0.6,
})

_COL_VIIRS = '#2C5F8A'   # bleu acier
_COL_DMSP  = '#C0392B'   # rouge brique

# ── Préparer les séries ───────────────────────────────────────────────────────
_viirs = (
    VIIRS_NUNAVUT_RESAMPL[['Annee', 'VIIRS_SUM_RESAMPL']]
    .dropna(subset=['VIIRS_SUM_RESAMPL'])
    .sort_values('Annee')
    .reset_index(drop=True)
)
_dmsp = (
    DMSP_NUNAVUT[['Annee', 'DMSP_SUM']]
    .dropna(subset=['DMSP_SUM'])
    .sort_values('Annee')
    .reset_index(drop=True)
)

# ── Figure & double axe ───────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(13, 5.5))
ax2 = ax1.twinx()

# DMSP (axe gauche) — ligne tiretée + marqueurs carrés
l_dmsp, = ax1.plot(
    _dmsp['Annee'], _dmsp['DMSP_SUM'],
    color=_COL_DMSP, linewidth=2.0,
    marker='s', markersize=5.5, markeredgecolor='white', markeredgewidth=0.6,
    linestyle='--', zorder=4,
    label='DMSP (2000–2013)'
)
ax1.fill_between(
    _dmsp['Annee'], _dmsp['DMSP_SUM'],
    alpha=0.10, color=_COL_DMSP, zorder=2
)

# VIIRS (axe droit) — ligne pleine + marqueurs ronds
l_viirs, = ax2.plot(
    _viirs['Annee'], _viirs['VIIRS_SUM_RESAMPL'],
    color=_COL_VIIRS, linewidth=2.0,
    marker='o', markersize=5.5, markeredgecolor='white', markeredgewidth=0.6,
    linestyle='-', zorder=4,
    label=r'VIIRS rééchantillonné (2012–2024)'
)
ax2.fill_between(
    _viirs['Annee'], _viirs['VIIRS_SUM_RESAMPL'],
    alpha=0.10, color=_COL_VIIRS, zorder=2
)

# ── Zone de chevauchement DMSP/VIIRS (2012–2013) ─────────────────────────────
ax1.axvspan(2011.55, 2013.45, alpha=0.06, color='#888888', zorder=1)
_y_txt = _dmsp['DMSP_SUM'].max()
ax1.text(
    2012.5, _y_txt * 0.96,
    'Chevauchement\nDMSP–VIIRS',
    ha='center', va='top', fontsize=7.5,
    color='#555555', fontstyle='italic', fontfamily='serif'
)

# ── Axe X : toutes les années disponibles ────────────────────────────────────
_all_annees = sorted(
    set(_viirs['Annee'].astype(int).tolist()) |
    set(_dmsp['Annee'].astype(int).tolist())
)
ax1.set_xticks(_all_annees)
ax1.set_xticklabels([str(a) for a in _all_annees], rotation=50, ha='right')
ax1.set_xlim(_all_annees[0] - 0.6, _all_annees[-1] + 0.6)

# ── Étiquettes des axes Y ─────────────────────────────────────────────────────
ax1.set_ylabel(
    'DMSP — Somme de luminosité\n(Digital Number)',
    color=_COL_DMSP, fontsize=10, labelpad=6
)
ax2.set_ylabel(
    r'VIIRS — Somme de luminosité rééchantillonnée' '\n'
    r'(nW$\cdot$cm$^{-2}\cdot$sr$^{-1}$)',
    color=_COL_VIIRS, fontsize=10, labelpad=6
)
ax1.set_xlabel('Année', fontsize=10, labelpad=5)

# ── Couleur des ticks selon l'axe ─────────────────────────────────────────────
ax1.tick_params(axis='y', labelcolor=_COL_DMSP)
ax2.tick_params(axis='y', labelcolor=_COL_VIIRS)

# ── Format numérique des ticks Y ─────────────────────────────────────────────
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

# ── Grille horizontale (axe gauche uniquement) ────────────────────────────────
ax1.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
ax1.set_axisbelow(True)
ax2.yaxis.grid(False)

# ── Spines ────────────────────────────────────────────────────────────────────
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['left'].set_color(_COL_DMSP)
ax1.spines['left'].set_linewidth(1.0)
ax2.spines['right'].set_color(_COL_VIIRS)
ax2.spines['right'].set_linewidth(1.0)
ax1.spines['bottom'].set_linewidth(0.8)

# ── Titre ─────────────────────────────────────────────────────────────────────
#ax1.set_title(
#    "Évolution de la luminosité nocturne du Nunavut\n"
#    "Données satellitaires DMSP (2000–2013) et VIIRS (2012–2024)",
#    fontsize=11, fontweight='bold', pad=10, fontfamily='serif'
#)

# ── Légende commune ───────────────────────────────────────────────────────────
ax1.legend(
    handles=[l_dmsp, l_viirs],
    loc='upper left',
    fontsize=10,
    framealpha=0.92,
    edgecolor='#CCCCCC',
    borderpad=0.7
)

plt.tight_layout()

# ── Sauvegarde ────────────────────────────────────────────────────────────────
_fig_path = os.path.join(fig_dir, "luminosite_DMSP_VIIRS_Nunavut.png")
plt.savefig(_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

plt.rcdefaults()
print(f"  ✔ Figure sauvegardée : {_fig_path}")



#%% FIGURES ET TABLEAUX DE DESCRIPTION DES DONNEES  (version amelioree)
# ###############################################################################
# FIGURES ET TABLEAUX DE DESCRIPTION DES DONNEES  (version amelioree)
# ###############################################################################
# ── Style publication ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family'        : 'serif',
    'font.size'          : 12,                     # ↑ 11 → 12
    'axes.titlesize'     : 13,                     # ↑ 13 → 13 (conservé)
    'axes.titleweight'   : 'bold',
    'axes.labelsize'     : 12,                     # ↑ 11 → 12
    'xtick.labelsize'    : 11,                     # ↑ 10 → 11
    'ytick.labelsize'    : 11,                     # ↑ 10 → 11
    'legend.fontsize'    : 10,                     # ↑ 9.5 → 10
    'legend.framealpha'  : 0.93,
    'legend.edgecolor'   : '#cccccc',
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
})

def add_grid(ax, axis='y'):
    """Grille horizontale legere."""
    if axis == 'y':
        ax.yaxis.grid(True, ls='--', lw=0.45, alpha=0.4, color='#aaaaaa', zorder=0)
    else:
        ax.xaxis.grid(True, ls='--', lw=0.45, alpha=0.4, color='#aaaaaa', zorder=0)
    ax.set_axisbelow(True)

def shade_event(ax, x0, x1, label, color='#fee090', alpha=0.25, ytext=0.97):
    """Zone ombree pour un evenement (crise, COVID, etc.)."""
    ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=0)
    ax.text((x0 + x1) / 2, ytext, label,
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8,                    # ↑ 7.5 → 8
            color='#555555', style='italic')

def fmt_M(x, _):
    """Formateur axe y en M$."""
    return f'{x:,.0f}'

COULEURS = {
    'Mine_Meadowbank' : '#1b7837',
    'Mine_Baffinland' : '#2166ac',
    'Mine_Hope_Bay'   : '#d6604d',
    'Mine_Meliadine'  : '#762a83',
}
LABELS_MINE = {
    'Mine_Meadowbank' : 'Meadowbank (or)',
    'Mine_Baffinland' : 'Baffinland / Mary River (fer)',
    'Mine_Hope_Bay'   : 'Hope Bay (or)',
    'Mine_Meliadine'  : 'Meliadine (or)',
}

fig_dir = os.path.join(baseDir, "Figure", "PUR")
tab_dir = os.path.join(baseDir, "Tableau", "PUR")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(tab_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 : Chronologie des mines — Gantt ameliore
# ══════════════════════════════════════════════════════════════════════════════
mines_gantt = [
    ('Meadowbank',                    2010, 2024, '#1b7837', 'Or',  'Agnico Eagle'),
    ('Hope Bay',                      2017, 2024, '#d6604d', 'Or',  'Agnico Eagle'),
    ('Baffinland / Mary River',       2014, 2024, '#2166ac', 'Fer', 'Baffinland Iron'),
    ('Meliadine',                     2019, 2024, '#762a83', 'Or',  'Agnico Eagle'),
]

fig1, ax1 = plt.subplots(figsize=(12, 4.2), facecolor='white')
ax1.set_facecolor('#f9f9f9')

for i, (label, debut, fin, coul, mineral, proprio) in enumerate(mines_gantt):
    duree = fin - debut + 1

    # Barre principale
    bar = mpatches.FancyArrow(
        debut, i, duree, 0,
        width=0.42, length_includes_head=True,
        head_width=0.42, head_length=0.4,
        fc=coul, ec='white', lw=0.7, alpha=0.90,
        zorder=3
    )
    ax1.add_patch(bar)

    # Hachure pour Fer
    if mineral == 'Fer':
        rect_h = mpatches.FancyArrow(
            debut, i, duree, 0,
            width=0.42, length_includes_head=True,
            head_width=0.42, head_length=0.4,
            fc='none', ec='white', lw=0.5, alpha=0.35,
            zorder=4, hatch='///'
        )
        ax1.add_patch(rect_h)

    # Nom de la mine (centre de la barre)
    cx = debut + duree / 2
    ax1.text(cx, i, f'{label}  ({mineral})',
             ha='center', va='center', fontsize=11,               # ↑ 10.5 → 11
             color='white', fontweight='bold', zorder=5)

    # Annee d'ouverture
    ax1.text(debut - 0.2, i, str(debut),
             ha='right', va='center', fontsize=10,                # ↑ 9.5 → 10
             color=coul, fontweight='bold')

    # Proprietaire (dessous)
    ax1.text(cx, i - 0.32, proprio,
             ha='center', va='center', fontsize=8,                # ↑ 7.5 → 8
             color='#444444', style='italic')

# Grille annuelle legere
for yr in range(2000, 2026, 2):
    ax1.axvline(yr, color='#cccccc', lw=0.55, ls='-', zorder=1)
for yr in range(2001, 2026, 2):
    ax1.axvline(yr, color='#e5e5e5', lw=0.35, ls='-', zorder=1)

# Annee courante
ax1.axvline(2024.5, color='#555555', lw=1.2, ls='--', alpha=0.7, zorder=2)

# Legende mineral
pat_or  = mpatches.Patch(facecolor='#888888', label='Minerai : Or', alpha=0.85)
pat_fer = mpatches.Patch(facecolor='#888888', hatch='///', label='Minerai : Fer',
                          edgecolor='white', alpha=0.55)
ax1.legend(handles=[pat_or, pat_fer],
           loc='lower right', fontsize=9.5, framealpha=0.92)      # ↑ 9 → 9.5

ax1.set_xlim(1999, 2026)
ax1.set_ylim(-0.65, len(mines_gantt) - 0.35)
ax1.set_yticks([])
ax1.set_xlabel('Année', fontsize=12)                               # ↑ 11 → 12
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_color('#888888')
ax1.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(1))
ax1.tick_params(axis='x', which='minor', length=3, color='#bbbbbb')

#ax1.set_title(
#    'Figure 1 — Chronologie des activités minières au Nunavut (2000–2024)',
#    fontsize=13, pad=12, loc='left')
#fig1.text(0.01, -0.04,
#          'Sources : Agnico Eagle Mines, Baffinland Iron Mines Corp.',
#          fontsize=8.5, color='#666666', style='italic',          # ↑ 8 → 8.5
#          transform=ax1.transAxes)

plt.tight_layout(pad=1.5)
p1 = os.path.join(fig_dir, "fig1_chronologie_mines_gantt.png")
plt.savefig(p1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure 1 enregistree : {p1}")


#%% ANALYSE D'INCERTITUDE MULTI-MODÈLES — STATISTIQUES D'ENSEMBLE
###############################################################################
# Référence méthodologique : Timmermann (2006) « Forecast Combinations »,
#   Handbook of Economic Forecasting ; Steel (2020) « Model Averaging and
#   its Use in Economics », J. Economic Literature.
#
# Contexte : aucun modèle ne constitue la référence absolue (PIB minier
#   non publié par Statistique Canada au niveau des mines individuelles).
#   On mobilise donc un moyennage équipondéré des N modèles disponibles
#   pour chaque cellule (Mine × Année), ce qui fournit :
#     — une estimation centrale robuste (médiane / moyenne)
#     — une mesure d'incertitude inter-modèles (écart-type, CV%, IQR)
#
# Sortie : tableaux Excel + LaTeX + fan charts par mine + carte thermique CV
###############################################################################

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────
_COLS_PIB    = ['PIB_Emploi', 'PIB_OLS', 'PIB_IV', 'PIB_CHEN', 'PIB_SCORE', 'PIB_SC']
_COLS_TX     = ['TX_Emploi',  'TX_OLS',  'TX_IV',  'TX_CHEN',  'TX_SCORE',  'TX_SC']
_NOMS_MOD    = ['Emploi', 'OLS (MCO)', 'IV2SLS', 'Chen', 'Score pondéré', 'StatCan']

# Couleurs cohérentes avec le reste du mémoire
_C_MINE = {
    'Meadowbank': '#1b7837',
    'Meliadine':  '#762a83',
    'Hope Bay':   '#d6604d',
    'Baffinland': '#2166ac',
}
# Couleurs des modèles pour les fan charts
_C_MOD = {
    'Emploi':        '#2C5F8A',   # bleu acier  — cohérent avec _COULEURS_MODELE
    'OLS (MCO)':     '#C0392B',   # rouge
    'IV2SLS':        '#2C7A4B',   # vert sauge
    'Chen':          '#7D3C98',   # violet
    'Score pondéré': '#D68910',   # ocre doré
    'StatCan':       '#17A589',   # turquoise
}

# Forcer la conversion numérique de toutes les colonnes PIB/TX
for _c in _COLS_PIB + _COLS_TX:
    TABLE_COMP[_c] = pd.to_numeric(TABLE_COMP[_c], errors='coerce')

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CALCUL DES STATISTIQUES D'ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
def _ens_stats(vals_raw, prefix):
    """
    Calcule les statistiques d'ensemble sur un vecteur (NaN ignorés).
    Retourne un dict avec préfixe pour éviter les collisions de colonnes.
    """
    v = np.array(pd.to_numeric(vals_raw, errors='coerce'), dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {f'{prefix}N': 0, f'{prefix}Moyenne': np.nan,
                f'{prefix}Mediane': np.nan, f'{prefix}Ecart_type': np.nan,
                f'{prefix}CV_pct': np.nan, f'{prefix}Min': np.nan,
                f'{prefix}Max': np.nan, f'{prefix}Q25': np.nan,
                f'{prefix}Q75': np.nan, f'{prefix}IQR': np.nan}
    mu  = float(np.mean(v))
    sig = float(np.std(v, ddof=0))          # ddof=0 : on mesure la dispersion
                                             # sur la population des modèles
    return {
        f'{prefix}N':          len(v),
        f'{prefix}Moyenne':    mu,
        f'{prefix}Mediane':    float(np.median(v)),
        f'{prefix}Ecart_type': sig,
        f'{prefix}CV_pct':     sig / mu * 100 if mu != 0 else np.nan,
        f'{prefix}Min':        float(np.min(v)),
        f'{prefix}Max':        float(np.max(v)),
        f'{prefix}Q25':        float(np.percentile(v, 25)),
        f'{prefix}Q75':        float(np.percentile(v, 75)),
        f'{prefix}IQR':        float(np.percentile(v, 75) - np.percentile(v, 25)),
    }

_TABLE_COMP_2005 = TABLE_COMP[TABLE_COMP['Annee'] >= 2005].reset_index(drop=True)

_pib_rows, _tx_rows = [], []
for _, _row in _TABLE_COMP_2005.iterrows():
    _pib_rows.append(_ens_stats(_row[_COLS_PIB].values, 'PIB_'))
    _tx_rows.append( _ens_stats(_row[_COLS_TX].values,  'TX_'))

_DF_PIB = pd.DataFrame(_pib_rows)
_DF_TX  = pd.DataFrame(_tx_rows)

TABLE_ENSEMBLE = pd.concat(
    [_TABLE_COMP_2005[['Annee', 'Mine', 'Mine_Label'] + _COLS_PIB + _COLS_TX]
     .reset_index(drop=True),
     _DF_PIB.reset_index(drop=True),
     _DF_TX .reset_index(drop=True)],
    axis=1
).sort_values(['Mine', 'Annee']).reset_index(drop=True)

# Affichage console récapitulatif — M$
print("\n" + "=" * 105)
print("  STATISTIQUES D'ENSEMBLE — PIB par mine, Nunavut (M$ CAD, base 2017)")
print("  Approche : moyennage équipondéré  |  N = nombre de modèles disponibles par cellule")
print("=" * 105)
_aff_pib = ['Annee', 'Mine_Label', 'PIB_N', 'PIB_Moyenne', 'PIB_Mediane',
            'PIB_Ecart_type', 'PIB_CV_pct', 'PIB_Q25', 'PIB_Q75', 'PIB_Min', 'PIB_Max']
print(TABLE_ENSEMBLE[_aff_pib].to_string(index=False,
      float_format=lambda x: f'{x:,.1f}'))

# Affichage console récapitulatif — % du PIB Nunavut
print("\n" + "=" * 105)
print("  STATISTIQUES D'ENSEMBLE — Part du PIB par mine (% du PIB Nunavut)")
print("  Approche : moyennage équipondéré  |  N = nombre de modèles disponibles par cellule")
print("=" * 105)
_aff_tx = ['Annee', 'Mine_Label', 'TX_N', 'TX_Moyenne', 'TX_Mediane',
           'TX_Ecart_type', 'TX_CV_pct', 'TX_Q25', 'TX_Q75', 'TX_Min', 'TX_Max']
print(TABLE_ENSEMBLE[_aff_tx].to_string(index=False,
      float_format=lambda x: f'{x:,.2f}'))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  TABLEAU RÉCAPITULATIF : CONSENSUS FINAL (N ≥ 2 modèles)
# ─────────────────────────────────────────────────────────────────────────────
_TE2 = TABLE_ENSEMBLE[TABLE_ENSEMBLE['PIB_N'] >= 2].copy()

# Résumé moyen par mine (toutes années) — M$ et %
TABLE_RESUME_MINE = (
    _TE2.groupby('Mine_Label')
    [['PIB_N',
      'PIB_Moyenne', 'PIB_Mediane', 'PIB_Ecart_type', 'PIB_CV_pct', 'PIB_Q25', 'PIB_Q75',
      'TX_Moyenne',  'TX_Mediane',  'TX_Ecart_type',  'TX_CV_pct',  'TX_Q25',  'TX_Q75']]
    .mean()
    .round(2)
    .reset_index()
)

print("\n" + "=" * 95)
print("  RÉSUMÉ PAR MINE — M$ CAD 2017 (moyenne sur toutes années 2005-2024, N ≥ 2 modèles)")
print("=" * 95)
_cols_resume_pib = ['Mine_Label', 'PIB_N', 'PIB_Moyenne', 'PIB_Mediane',
                    'PIB_Ecart_type', 'PIB_CV_pct', 'PIB_Q25', 'PIB_Q75']
print(TABLE_RESUME_MINE[_cols_resume_pib].to_string(
    index=False, float_format=lambda x: f'{x:,.2f}'))

print("\n" + "=" * 95)
print("  RÉSUMÉ PAR MINE — % du PIB Nunavut (moyenne sur toutes années 2005-2024, N ≥ 2 modèles)")
print("=" * 95)
_cols_resume_tx = ['Mine_Label', 'PIB_N', 'TX_Moyenne', 'TX_Mediane',
                   'TX_Ecart_type', 'TX_CV_pct', 'TX_Q25', 'TX_Q75']
print(TABLE_RESUME_MINE[_cols_resume_tx].to_string(
    index=False, float_format=lambda x: f'{x:,.2f}'))

# ─────────────────────────────────────────────────────────────────────────────
# 2 bis. CALCUL DU TOTAL AGRÉGÉ DES 4 MINES (méthode rigoureuse)
#   Pour chaque (Année × Modèle), on SOMME les 4 mines, puis on applique
#   _ens_stats sur les 6 sommes annuelles → l'incertitude est correctement
#   propagée (≠ somme des médianes qui sous-estime la dispersion).
#   Doit être calculé AVANT les exports Excel et LaTeX qui l'utilisent.
# ─────────────────────────────────────────────────────────────────────────────
_MINES_AGG = ['Meadowbank', 'Meliadine', 'Hope Bay', 'Baffinland']
_BASE_AGG  = _TABLE_COMP_2005[_TABLE_COMP_2005['Mine_Label'].isin(_MINES_AGG)]

_TOT_PIB = _BASE_AGG.groupby('Annee')[_COLS_PIB].sum(min_count=1).reset_index()
_TOT_TX  = _BASE_AGG.groupby('Annee')[_COLS_TX ].sum(min_count=1).reset_index()

_tot_rows_pib, _tot_rows_tx = [], []
for _, _r in _TOT_PIB.iterrows():
    _stats = _ens_stats(_r[_COLS_PIB].values, 'PIB_')
    _stats['Annee'] = int(_r['Annee'])
    _tot_rows_pib.append(_stats)
for _, _r in _TOT_TX.iterrows():
    _stats = _ens_stats(_r[_COLS_TX].values, 'TX_')
    _stats['Annee'] = int(_r['Annee'])
    _tot_rows_tx.append(_stats)

ENSEMBLE_TOTAL = (
    pd.DataFrame(_tot_rows_pib)
    .merge(pd.DataFrame(_tot_rows_tx), on='Annee')
    .sort_values('Annee').reset_index(drop=True)
)

# Statistiques pluriannuelles du Total 4 mines (moyennes sur les années
# où N >= 2 modèles sont disponibles).
_ENS_TOT_VALIDES = ENSEMBLE_TOTAL[ENSEMBLE_TOTAL['PIB_N'] >= 2]
_TOTAL_RES = {
    'PIB_N':         _ENS_TOT_VALIDES['PIB_N'].mean(),
    'PIB_Moyenne':   _ENS_TOT_VALIDES['PIB_Moyenne'].mean(),
    'PIB_Mediane':   _ENS_TOT_VALIDES['PIB_Mediane'].mean(),
    'PIB_Ecart_type':_ENS_TOT_VALIDES['PIB_Ecart_type'].mean(),
    'PIB_CV_pct':    _ENS_TOT_VALIDES['PIB_CV_pct'].mean(),
    'PIB_Q25':       _ENS_TOT_VALIDES['PIB_Q25'].mean(),
    'PIB_Q75':       _ENS_TOT_VALIDES['PIB_Q75'].mean(),
    'TX_Moyenne':    _ENS_TOT_VALIDES['TX_Moyenne'].mean(),
    'TX_Mediane':    _ENS_TOT_VALIDES['TX_Mediane'].mean(),
    'TX_Ecart_type': _ENS_TOT_VALIDES['TX_Ecart_type'].mean(),
    'TX_CV_pct':     _ENS_TOT_VALIDES['TX_CV_pct'].mean(),
    'TX_Q25':        _ENS_TOT_VALIDES['TX_Q25'].mean(),
    'TX_Q75':        _ENS_TOT_VALIDES['TX_Q75'].mean(),
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  EXPORT EXCEL — TABLEAUX COMPLETS
# ─────────────────────────────────────────────────────────────────────────────
_path_ens_xl = os.path.join(tab_dir, "tableau_ensemble_incertitude.xlsx")

# Index par année pour récupérer les valeurs "Total 4 mines" rapidement
_TOT_BY_YEAR = ENSEMBLE_TOTAL.set_index('Annee')

with pd.ExcelWriter(_path_ens_xl, engine='openpyxl') as _xl:

    # Feuille 1 — données longues + lignes "Total 4 mines" (1 par année)
    _tot_long = ENSEMBLE_TOTAL.copy()
    _tot_long.insert(0, 'Mine',       'Total_4_mines')
    _tot_long.insert(1, 'Mine_Label', 'Total 4 mines')
    _ens_long_aug = pd.concat([TABLE_ENSEMBLE, _tot_long],
                              ignore_index=True, sort=False)
    _ens_long_aug.to_excel(_xl, sheet_name='Ensemble_Long', index=False)

    # Feuilles M$ — pivots Annee × Mine + colonne 'Total 4 mines'
    for _stat, _lbl in [('PIB_Moyenne',    'Moyenne_M$'),
                        ('PIB_Mediane',    'Mediane_M$'),
                        ('PIB_CV_pct',     'CV_pct_M$'),
                        ('PIB_IQR',        'IQR_M$')]:
        _piv = _TE2.pivot_table(index='Annee', columns='Mine_Label',
                                values=_stat, aggfunc='mean').round(2)
        _piv['Total 4 mines'] = _TOT_BY_YEAR[_stat].round(2)
        _piv.to_excel(_xl, sheet_name=_lbl)

    # Feuilles % — pivots Annee × Mine + colonne 'Total 4 mines'
    for _stat, _lbl in [('TX_Moyenne',     'Moyenne_Pct'),
                        ('TX_Mediane',     'Mediane_Pct'),
                        ('TX_CV_pct',      'CV_pct_Pct'),
                        ('TX_IQR',         'IQR_Pct'),
                        ('TX_Q25',         'Q25_Pct'),
                        ('TX_Q75',         'Q75_Pct')]:
        _piv = _TE2.pivot_table(index='Annee', columns='Mine_Label',
                                values=_stat, aggfunc='mean').round(2)
        _piv['Total 4 mines'] = _TOT_BY_YEAR[_stat].round(2)
        _piv.to_excel(_xl, sheet_name=_lbl)

    # Résumé par mine (M$ et %) + ligne 'Total 4 mines'
    _tot_resume_row = pd.DataFrame([{
        'Mine_Label': 'Total 4 mines',
        **{_k: _v for _k, _v in _TOTAL_RES.items()}
    }])
    _resume_aug = pd.concat(
        [TABLE_RESUME_MINE, _tot_resume_row],
        ignore_index=True, sort=False
    )
    _resume_aug.to_excel(_xl, sheet_name='Resume_par_mine', index=False)

    # N modèles disponibles par cellule + colonne Total
    _piv_n = TABLE_ENSEMBLE.pivot_table(index='Annee', columns='Mine_Label',
                                        values='PIB_N', aggfunc='mean').round(0)
    _piv_n['Total 4 mines'] = _TOT_BY_YEAR['PIB_N'].round(0)
    _piv_n.to_excel(_xl, sheet_name='N_modeles')

    # NOUVELLE FEUILLE dédiée — totaux annuels (4 mines agrégées)
    ENSEMBLE_TOTAL.round(2).to_excel(
        _xl, sheet_name='Total_4_mines_annuel', index=False
    )

print(f"\n  ✔ Tableau d'ensemble exporté : {_path_ens_xl}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXPORT LATEX — TABLEAU CONSENSUS (médiane + IQR + CV)
#     Deux fichiers séparés : un pour M$ CAD, un pour % du PIB
# ─────────────────────────────────────────────────────────────────────────────
_path_ens_tex_msc = os.path.join(tab_dir, "tableau_ensemble_incertitude_valeurs.tex")
_path_ens_tex_pct = os.path.join(tab_dir, "tableau_ensemble_incertitude_parts.tex")

_mines_ord_l = [m for m in ['Meadowbank', 'Meliadine', 'Hope Bay', 'Baffinland']
                if m in _TE2['Mine_Label'].unique()]
_annees_l    = sorted(_TE2['Annee'].unique().astype(int))

# ENSEMBLE_TOTAL est déjà calculé plus haut (section 2 bis) avant l'export Excel.

def _write_ens_table_latex(f, TE2, mines_ord, annees, col_med, col_q25, col_q75,
                           col_cv, caption, label, fmt_med, fmt_ic, fmt_cv,
                           note_unit, total_df=None,
                           tot_col_med=None, tot_col_cv=None,
                           tot_col_q25=None, tot_col_q75=None, compact=False,
                           models_desc=""):
    """Génère un tableau LaTeX d'ensemble (médiane + IC50 + CV) pour M$ ou %.

    Si total_df est fourni avec tot_col_med, tot_col_q25, tot_col_q75 et
    tot_col_cv, une colonne 'Total 4 mines' avec 3 sous-colonnes (Médiane,
    IC50, CV%) est ajoutée à droite, symétrique aux colonnes par mine.

    Si compact=True, la mise en page est resserrée (taille de police plus
    petite, espace inter-colonnes réduit, en-tête « Médiane » abrégé en
    « Méd. ») pour un meilleur ajustement à la largeur de page.
    """
    _has_tot = (total_df is not None and tot_col_med is not None
                and tot_col_cv is not None
                and tot_col_q25 is not None and tot_col_q75 is not None)
    _nc   = 1 + 3 * len(mines_ord) + (3 if _has_tot else 0)
    _cfmt = 'l' + ('rrr' * len(mines_ord)) + ('rrr' if _has_tot else '')
    _med_lbl = "Méd." if compact else "Médiane"
    _size    = "\\scriptsize" if compact else "\\footnotesize"
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write(_size + "\n")
    if compact:
        f.write("\\setlength{\\tabcolsep}{3pt}\n")
    f.write(f"\\caption{{{caption}}}\n")
    f.write(f"\\label{{{label}}}\n")
    f.write(f"\\begin{{tabular}}{{{_cfmt}}}\n\\hline\\hline\n")
    # En-tête niveau 1
    f.write("\\multirow{2}{*}{Année}")
    for _m in mines_ord:
        f.write(f" & \\multicolumn{{3}}{{c}}{{{_m}}}")
    if _has_tot:
        f.write(" & \\multicolumn{3}{c}{Total}")
    f.write(" \\\\\n")
    # En-tête niveau 2
    f.write(" " * 22)
    for _ in mines_ord:
        f.write(f" & {_med_lbl} & IC$_{{50}}$ & CV\\,\\%")
    if _has_tot:
        f.write(f" & {_med_lbl} & IC$_{{50}}$ & CV\\,\\%")
    f.write(" \\\\\n\\hline\n")
    # Données
    for _yr in annees:
        _sub  = TE2[TE2['Annee'] == _yr]
        _line = str(int(_yr))
        for _m in mines_ord:
            _r = _sub[_sub['Mine_Label'] == _m]
            if _r.empty or _r['PIB_N'].values[0] < 2:
                _line += " & --- & --- & ---"
            else:
                _med = _r[col_med].values[0]
                _q25 = _r[col_q25].values[0]
                _q75 = _r[col_q75].values[0]
                _cv  = _r[col_cv ].values[0]
                _ic  = f"[{fmt_ic(_q25)}\\,;\\,{fmt_ic(_q75)}]"
                _line += f" & {fmt_med(_med)} & {_ic} & {fmt_cv(_cv)}"
        if _has_tot:
            _rt = total_df[total_df['Annee'] == _yr]
            if _rt.empty or pd.isna(_rt[tot_col_med].values[0]):
                _line += " & --- & --- & ---"
            else:
                _q25_t = _rt[tot_col_q25].values[0]
                _q75_t = _rt[tot_col_q75].values[0]
                _ic_t  = f"[{fmt_ic(_q25_t)}\\,;\\,{fmt_ic(_q75_t)}]"
                _line += (f" & {fmt_med(_rt[tot_col_med].values[0])}"
                          f" & {_ic_t}"
                          f" & {fmt_cv(_rt[tot_col_cv].values[0])}")
        _line += " \\\\\n"
        f.write(_line)
    f.write("\\hline\\hline\n")
    _note_size = "\\scriptsize" if compact else "\\footnotesize"
    f.write("\\multicolumn{" + str(_nc) + "}{l}{" + _note_size + " "
            f"\\textit{{Note :}} {models_desc}"
            "IC$_{50}$=[Q25\\,;\\,Q75]; "
            "CV\\,\\%=$\\sigma/\\mu\\times 100$ inter-modèles. "
            f"{note_unit}« --- » si $N<2$. "
            "Total 4 mines : recalcul d'ensemble sur la somme inter-mines."
            "} \\\\\n")
    f.write("\\end{tabular}\n\\end{table}\n\n")

# ── Fichier 1 : M$ CAD 2017 ───────────────────────────────────────────────
with open(_path_ens_tex_msc, 'w', encoding='utf-8') as _f:
    _write_ens_table_latex(
        _f, _TE2, _mines_ord_l, _annees_l,
        col_med='PIB_Mediane', col_q25='PIB_Q25', col_q75='PIB_Q75', col_cv='PIB_CV_pct',
        caption="Statistiques d'ensemble du PIB par mine --- Nunavut "
                "(M\\$ CAD, base 2017)",
        label="tab:ensemble_pib_msca",
        fmt_med=lambda v: _fmt_fr(v, 0),
        fmt_ic =lambda v: _fmt_fr(v, 0),
        fmt_cv =lambda v: _fmt_fr(v, 1),
        note_unit="Valeurs en M\\$ CAD 2017. ",
        total_df=ENSEMBLE_TOTAL,
        tot_col_med='PIB_Mediane', tot_col_cv='PIB_CV_pct',
        tot_col_q25='PIB_Q25', tot_col_q75='PIB_Q75',
        compact=True,
        models_desc=("Moyennage équipondéré entre $N$ modèles "
                     "(Emploi, OLS, IV2SLS, Chen, Score pondéré, StatCan). ")
    )
print(f"  ✔ Tableau LaTeX exporté (M$) : {_path_ens_tex_msc}")

# ── Fichier 2 : % du PIB Nunavut ──────────────────────────────────────────
with open(_path_ens_tex_pct, 'w', encoding='utf-8') as _f:
    _write_ens_table_latex(
        _f, _TE2, _mines_ord_l, _annees_l,
        col_med='TX_Mediane', col_q25='TX_Q25', col_q75='TX_Q75', col_cv='TX_CV_pct',
        caption="Statistiques d'ensemble de la part du PIB par mine --- Nunavut "
                "(\\% du PIB territorial)",
        label="tab:ensemble_pib_pct",
        fmt_med=lambda v: _fmt_fr(v, 1),
        fmt_ic =lambda v: _fmt_fr(v, 1),
        fmt_cv =lambda v: _fmt_fr(v, 1),
        note_unit="Valeurs en \\% du PIB Nunavut. ",
        total_df=ENSEMBLE_TOTAL,
        tot_col_med='TX_Mediane', tot_col_cv='TX_CV_pct',
        tot_col_q25='TX_Q25', tot_col_q75='TX_Q75',
        compact=True,
        models_desc=("Moyennage équipondéré entre $N$ modèles "
                     "(Emploi, OLS, IV2SLS, Chen, Score pondéré, StatCan). ")
    )
print(f"  ✔ Tableau LaTeX exporté (%) : {_path_ens_tex_pct}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FAN CHARTS — UN PAR MINE (style Banque du Canada / FMI)
#      Bandes d'incertitude : [Min ; Max] (clair) + [Q25 ; Q75] (foncé)
#      Ligne centrale : médiane (trait plein) + moyenne (tirets)
#      Points : estimations individuelles de chaque modèle
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        12,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'axes.linewidth':   0.8,
    'axes.edgecolor':   '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'grid.color':       '#CCCCCC',
    'grid.linewidth':   0.6,
})

_mines_fan  = [m for m in ['Meadowbank', 'Meliadine', 'Hope Bay', 'Baffinland']
               if m in TABLE_ENSEMBLE['Mine_Label'].unique()]
_col_pib_mod = dict(zip(_NOMS_MOD, _COLS_PIB))

fig_fan, axes_fan = plt.subplots(2, 2, figsize=(16, 11), facecolor='white')
#fig_fan.suptitle(
#    "Estimations du PIB par mine — Nunavut (M\\$ CAD, base 2017)\\n"
#    "Fan chart : bandes d'incertitude inter-modèles (moyennage équipondré)",
#    fontsize=14, fontweight='bold', y=1.01, fontfamily='serif'
#)

for _ax, _mine in zip(axes_fan.flatten(), _mines_fan):

    _c  = _C_MINE.get(_mine, '#444444')
    _dm = TABLE_ENSEMBLE[TABLE_ENSEMBLE['Mine_Label'] == _mine].sort_values('Annee')
    _dm = _dm[_dm['PIB_N'] >= 1].reset_index(drop=True)

    if _dm.empty:
        _ax.set_visible(False)
        continue

    _ann = _dm['Annee'].values.astype(int)

    # ── Bande [Min ; Max] ────────────────────────────────────────
    _ax.fill_between(_ann, _dm['PIB_Min'], _dm['PIB_Max'],
                     color=_c, alpha=0.12, label='[Min ; Max]', zorder=2)

    # ── Bande [Q25 ; Q75] ────────────────────────────────────────
    _ax.fill_between(_ann,
                     _dm['PIB_Q25'].where(_dm['PIB_N'] >= 2),
                     _dm['PIB_Q75'].where(_dm['PIB_N'] >= 2),
                     color=_c, alpha=0.28,
                     label='IC$_{50}$ [Q25 ; Q75]', zorder=3)

    # ── Ligne Médiane ─────────────────────────────────────────────
    _mask_med = _dm['PIB_N'] >= 1
    _ax.plot(_ann[_mask_med], _dm.loc[_mask_med, 'PIB_Mediane'],
             color=_c, linewidth=2.4, linestyle='-', zorder=5,
             marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.5,
             label='Médiane')

    # ── Ligne Moyenne ───────────────────────────────────────────
    _mask_moy = _dm['PIB_N'] >= 2
    _ax.plot(_ann[_mask_moy], _dm.loc[_mask_moy, 'PIB_Moyenne'],
             color=_c, linewidth=1.6, linestyle='--', zorder=4, alpha=0.85,
             label='Moyenne')

    # ── Points individuels — chaque modèle ───────────────────────────
    for _nom_m, _col_m in _col_pib_mod.items():
        _dm_mod = TABLE_ENSEMBLE[
            (TABLE_ENSEMBLE['Mine_Label'] == _mine) &
            TABLE_ENSEMBLE[_col_m].notna()
        ].sort_values('Annee')
        if _dm_mod.empty:
            continue
        _ax.scatter(
            _dm_mod['Annee'], _dm_mod[_col_m],
            color=_C_MOD.get(_nom_m, '#888888'),
            s=28, alpha=0.75, zorder=6,
            marker='D', linewidths=0.4, edgecolors='white',
            label=_nom_m
        )

    # ── Mise en forme ─────────────────────────────────────────────
    _ax.set_title(_mine, fontsize=12, fontweight='bold',
                  color=_c, pad=6, fontfamily='serif')
    _ax.set_xlabel('Année', fontsize=12, labelpad=5)
    _ax.set_ylabel('PIB estimé (M\\$ CAD, base 2017)', fontsize=12, labelpad=5)
    _ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}')
    )
    _ax.set_ylim(bottom=0)
    _ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
    _ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    _ax.tick_params(axis='x', labelrotation=45)
    _ax.spines['top'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    _ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
    _ax.set_axisbelow(True)
    _ax.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC',
               ncol=2, loc='upper left', borderpad=0.5, handlelength=1.4)

plt.tight_layout(h_pad=3.5, w_pad=3.0)
_fp_fan = os.path.join(fig_dir, "fan_chart_PIB_ensemble_mines.png")
plt.savefig(_fp_fan, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Fan chart M$ exporté : {_fp_fan}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CARTE THERMIQUE DU CV% (incertitude inter-modèles)
#     Chaque cellule = dispersion relative des N modèles disponibles
#     Vert = consensus fort, Rouge = divergence forte entre modèles
# ─────────────────────────────────────────────────────────────────────────────

_piv_cv_heat = TABLE_ENSEMBLE.pivot_table(
    index='Mine_Label', columns='Annee',
    values='PIB_CV_pct', aggfunc='mean'
).reindex(index=_mines_fan)

# Réindexer sur les mines disponibles
_mines_heat = [m for m in _mines_fan if m in _piv_cv_heat.index]
_piv_cv_heat = _piv_cv_heat.reindex(index=_mines_heat)

_piv_n_heat = TABLE_ENSEMBLE.pivot_table(
    index='Mine_Label', columns='Annee',
    values='PIB_N', aggfunc='mean'
).reindex(index=_mines_heat)

fig_heat, ax_heat = plt.subplots(
    figsize=(max(12, len(_piv_cv_heat.columns) * 0.65), 4.5),
    facecolor='white'
)

_cv_vals = _piv_cv_heat.values.astype(float)
_vmax    = np.nanpercentile(_cv_vals, 95)   # borne supérieure (robuste aux extrêmes)

_im = ax_heat.imshow(
    _cv_vals,
    aspect='auto',
    cmap='RdYlGn_r',        # vert = faible CV (accord), rouge = fort CV (désaccord)
    vmin=0, vmax=max(_vmax, 50),
    interpolation='nearest'
)

# Annotations dans chaque cellule
_n_rows, _n_cols = _cv_vals.shape
for _r in range(_n_rows):
    for _cl in range(_n_cols):
        _val_cv = _cv_vals[_r, _cl]
        _val_n  = _piv_n_heat.values[_r, _cl] if not _piv_n_heat.empty else np.nan
        if np.isnan(_val_cv):
            _txt = '—'
            _col_txt = '#AAAAAA'
        else:
            _txt = f'{_val_cv:.0f}%\n(N={int(_val_n)})'
            _col_txt = 'white' if _val_cv > _vmax * 0.55 else '#222222'
        ax_heat.text(_cl, _r, _txt, ha='center', va='center',
                     fontsize=7.5, color=_col_txt, fontfamily='serif')

# Axes
ax_heat.set_xticks(range(_n_cols))
ax_heat.set_xticklabels(
    [str(int(c)) for c in _piv_cv_heat.columns],
    rotation=55, ha='right', fontsize=9
)
ax_heat.set_yticks(range(_n_rows))
ax_heat.set_yticklabels(_mines_heat, fontsize=10, fontfamily='serif')

_cbar = plt.colorbar(_im, ax=ax_heat, pad=0.015, fraction=0.025)
_cbar.set_label("Coefficient de variation (%)", fontsize=10, fontfamily='serif')
_cbar.ax.tick_params(labelsize=9)

ax_heat.set_title(
#    "Incertitude inter-modèles — Coefficient de variation du PIB estimé par mine (%)\n"
    "Vert : fort consensus entre modèles  |  Rouge : forte divergence  |  N : nombre de modèles",
    fontsize=12, fontweight='bold', pad=8, fontfamily='serif'
)
ax_heat.set_xlabel('Année', fontsize=12, labelpad=6)

plt.tight_layout()
_fp_heat = os.path.join(fig_dir, "heatmap_CV_incertitude_modeles.png")
plt.savefig(_fp_heat, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Carte thermique CV exportée : {_fp_heat}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRAPHIQUE RÉCAPITULATIF — ESTIMATION CONSENSUS TOUTES MINES
#     Barres empilées (médiane) + barres d'erreur [Q25 ; Q75] + CV moyen
# ─────────────────────────────────────────────────────────────────────────────

_TE3 = TABLE_ENSEMBLE[
    (TABLE_ENSEMBLE['PIB_N'] >= 2) &
    (TABLE_ENSEMBLE['Mine_Label'].isin(_mines_fan))
].copy()

# PIB total toutes mines : somme des médianes + propagation de l'incertitude
_agg_tot = (
    _TE3.groupby('Annee')
    .agg(
        PIB_Mediane_tot   = ('PIB_Mediane', 'sum'),
        PIB_Q25_tot       = ('PIB_Q25',     'sum'),
        PIB_Q75_tot       = ('PIB_Q75',     'sum'),
        PIB_CV_moy        = ('PIB_CV_pct',  'mean'),
        N_mines           = ('Mine_Label',  'count'),
    )
    .reset_index()
    .sort_values('Annee')
)

fig_rec, (ax_rec, ax_cv) = plt.subplots(
    2, 1, figsize=(14, 9), facecolor='white',
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.35}
)

# ── Panneau supérieur : barres empilées par mine ───────────────────────────
_bottom = np.zeros(
    len(_TE3['Annee'].unique())
)
_annees_rec = sorted(_TE3['Annee'].unique().astype(int))
_x_rec      = np.arange(len(_annees_rec))

for _mine in _mines_fan:
    _dm_r = (
        _TE3[_TE3['Mine_Label'] == _mine]
        .set_index('Annee')
        .reindex(_annees_rec)
    )
    _vals = _dm_r['PIB_Mediane'].fillna(0).values
    _err_lo = (_dm_r['PIB_Mediane'] - _dm_r['PIB_Q25']).fillna(0).values
    _err_hi = (_dm_r['PIB_Q75']     - _dm_r['PIB_Mediane']).fillna(0).values
    _c_r    = _C_MINE.get(_mine, '#888888')

    _bars = ax_rec.bar(
        _x_rec, _vals,
        bottom=_bottom,
        color=_c_r, alpha=0.82, edgecolor='white', linewidth=0.3,
        width=0.70, zorder=3, label=_mine
    )
    # Barres d'erreur centrées sur le haut de chaque segment
    ax_rec.errorbar(
        _x_rec, _bottom + _vals,
        yerr=[_err_lo, _err_hi],
        fmt='none', ecolor=_c_r, elinewidth=1.2,
        capsize=3, capthick=1.0, alpha=0.70, zorder=4
    )
    _bottom = _bottom + _vals

# Ligne du total médian
ax_rec.plot(
    _x_rec, _agg_tot.set_index('Annee').reindex(_annees_rec)['PIB_Mediane_tot'].values,
    color='#222222', linewidth=1.8, linestyle='--',
    marker='D', markersize=4.5,
    markeredgecolor='white', markeredgewidth=0.5,
    zorder=5, label='Total (médiane)'
)

ax_rec.set_xticks(_x_rec)
ax_rec.set_xticklabels([str(a) for a in _annees_rec], rotation=50, ha='right')
ax_rec.set_xlim(-0.55, len(_annees_rec) - 0.45)
ax_rec.set_ylim(bottom=0)
ax_rec.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax_rec.set_ylabel('PIB estimé — médiane (M\$ CAD, base 2017)', fontsize=12, labelpad=5)
#ax_rec.set_title(
#    "Estimation consensus du PIB par mine — Nunavut\n"
#    "Médiane inter-modèles + barres d'erreur [Q25\,;\,Q75]",
#    fontsize=13, fontweight='bold', pad=8, fontfamily='serif'
#)
ax_rec.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC',
              ncol=2, loc='upper left')
ax_rec.spines['top'].set_visible(False)
ax_rec.spines['right'].set_visible(False)
ax_rec.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
ax_rec.set_axisbelow(True)

# ── Panneau inférieur : CV moyen (accord inter-modèles) ───────────────────
_cv_rec = _agg_tot.set_index('Annee').reindex(_annees_rec)['PIB_CV_moy'].values
_colors_cv = ['#27AE60' if v < 20 else ('#F39C12' if v < 40 else '#C0392B')
              for v in np.nan_to_num(_cv_rec, nan=0)]

ax_cv.bar(_x_rec, _cv_rec, color=_colors_cv, alpha=0.82,
          edgecolor='white', linewidth=0.3, width=0.70, zorder=3)
ax_cv.axhline(20, color='#27AE60', linewidth=1.0, linestyle='--',
              alpha=0.7, label='Seuil 20 % (accord fort)')
ax_cv.axhline(40, color='#C0392B', linewidth=1.0, linestyle='--',
              alpha=0.7, label='Seuil 40 % (divergence)')
ax_cv.set_xticks(_x_rec)
ax_cv.set_xticklabels([str(a) for a in _annees_rec], rotation=50, ha='right')
ax_cv.set_xlim(-0.55, len(_annees_rec) - 0.45)
ax_cv.set_ylim(bottom=0)
ax_cv.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}\%'))
ax_cv.set_ylabel('CV moyen (\%)', fontsize=12, labelpad=5)
ax_cv.set_title('Incertitude inter-modèles (CV moyen, toutes mines)', fontsize=11)
ax_cv.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC', loc='upper right')
ax_cv.spines['top'].set_visible(False)
ax_cv.spines['right'].set_visible(False)
ax_cv.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
ax_cv.set_axisbelow(True)

plt.tight_layout()
_fp_rec = os.path.join(fig_dir, "fig_consensus_PIB_mines_ensemble.png")
plt.savefig(_fp_rec, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Graphique consensus exporté : {_fp_rec}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  EXPORT LATEX — TABLEAU RÉSUMÉ PAR MINE (pour le corps du mémoire)
# ─────────────────────────────────────────────────────────────────────────────
_path_res_tex = os.path.join(tab_dir, "tableau_resume_ensemble_mine.tex")

_note_mod = ("Emploi, MCO Log-Log, IV2SLS, Chen, Score pondéré, StatCan. "
             "CV = écart-type / moyenne $\\times$ 100. "
             "$N_{\\text{moy}}$ = nombre moyen de modèles disponibles par année. "
             "« Total 4 mines » = recalcul d'ensemble sur la somme des 4 "
             "mines modèle par modèle (puis moyenne pluriannuelle).")

# _TOTAL_RES est déjà calculé plus haut (section 2 bis) avant l'export Excel.

with open(_path_res_tex, 'w', encoding='utf-8') as _f:

    # ── Table A : M$ CAD 2017 ──────────────────────────────────────────────
    _f.write("\\begin{table}[htbp]\n\\centering\n")
    _f.write("\\caption{Statistiques descriptives des estimations du PIB par mine "
             "--- Nunavut (M\\$ CAD, base 2017)\\newline"
             "\\footnotesize Moyennage équipondéré entre les modèles " + _note_mod + "}\n")
    _f.write("\\label{tab:resume_ensemble_mine_msc}\n")
    _f.write("\\begin{tabular}{lrrrrrrr}\n\\hline\\hline\n")
    _f.write("Mine & $N_{\\text{moy}}$ & Moyenne & Médiane & "
             "Écart-type & CV\\,\\% & Q25 & Q75 \\\\\n")
    _f.write(" & & \\multicolumn{6}{c}{(M\\$ CAD, base 2017)} \\\\\n")
    _f.write("\\hline\n")
    for _, _r in TABLE_RESUME_MINE.iterrows():
        _f.write(
            f"{_r['Mine_Label']} & {_fmt_fr(_r['PIB_N'], 1)} & "
            f"{_fmt_fr(_r['PIB_Moyenne'], 1)} & {_fmt_fr(_r['PIB_Mediane'], 1)} & "
            f"{_fmt_fr(_r['PIB_Ecart_type'], 1)} & {_fmt_fr(_r['PIB_CV_pct'], 1)} & "
            f"{_fmt_fr(_r['PIB_Q25'], 1)} & {_fmt_fr(_r['PIB_Q75'], 1)} \\\\\n"
        )
    # Ligne Total 4 mines
    _f.write("\\hline\n")
    _f.write(
        f"\\textbf{{Total 4 mines}} & {_fmt_fr(_TOTAL_RES['PIB_N'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES['PIB_Moyenne'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES['PIB_Mediane'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES['PIB_Ecart_type'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES['PIB_CV_pct'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES['PIB_Q25'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES['PIB_Q75'], 1)} \\\\\n"
    )
    _f.write("\\hline\\hline\n")
    _f.write("\\multicolumn{8}{l}{\\footnotesize \\textit{Note :} " + _note_mod + "} \\\\\n")
    _f.write("\\end{tabular}\n\\end{table}\n\n")

    # ── Table B : % du PIB Nunavut ─────────────────────────────────────────
    _f.write("\\begin{table}[htbp]\n\\centering\n")
    _f.write("\\caption{Statistiques descriptives de la part du PIB par mine "
             "--- Nunavut (\\% du PIB territorial)\\newline"
             "\\footnotesize Moyennage équipondéré entre les modèles " + _note_mod + "}\n")
    _f.write("\\label{tab:resume_ensemble_mine_pct}\n")
    _f.write("\\begin{tabular}{lrrrrrrr}\n\\hline\\hline\n")
    _f.write("Mine & $N_{\\text{moy}}$ & Moyenne & Médiane & "
             "Écart-type & CV\\,\\% & Q25 & Q75 \\\\\n")
    _f.write(" & & \\multicolumn{6}{c}{(\\% du PIB Nunavut)} \\\\\n")
    _f.write("\\hline\n")
    for _, _r in TABLE_RESUME_MINE.iterrows():
        _f.write(
            f"{_r['Mine_Label']} & {_fmt_fr(_r['PIB_N'], 1)} & "
            f"{_fmt_fr(_r['TX_Moyenne'], 2)} & {_fmt_fr(_r['TX_Mediane'], 2)} & "
            f"{_fmt_fr(_r['TX_Ecart_type'], 2)} & {_fmt_fr(_r['TX_CV_pct'], 1)} & "
            f"{_fmt_fr(_r['TX_Q25'], 2)} & {_fmt_fr(_r['TX_Q75'], 2)} \\\\\n"
        )
    # Ligne Total 4 mines
    _f.write("\\hline\n")
    _f.write(
        f"\\textbf{{Total 4 mines}} & {_fmt_fr(_TOTAL_RES['PIB_N'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES['TX_Moyenne'], 2)} & "
        f"{_fmt_fr(_TOTAL_RES['TX_Mediane'], 2)} & "
        f"{_fmt_fr(_TOTAL_RES['TX_Ecart_type'], 2)} & "
        f"{_fmt_fr(_TOTAL_RES['TX_CV_pct'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES['TX_Q25'], 2)} & "
        f"{_fmt_fr(_TOTAL_RES['TX_Q75'], 2)} \\\\\n"
    )
    _f.write("\\hline\\hline\n")
    _f.write("\\multicolumn{8}{l}{\\footnotesize \\textit{Note :} " + _note_mod + "} \\\\\n")
    _f.write("\\end{tabular}\n\\end{table}\n")

print(f"  ✔ Tableau LaTeX résumé par mine exporté : {_path_res_tex}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  RÉCAPITULATIF CONSOLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  ANALYSE D'INCERTITUDE MULTI-MODÈLES — FICHIERS GÉNÉRÉS")
print("=" * 80)
print(f"  Excel  : {_path_ens_xl}")
print(f"  LaTeX  : {_path_ens_tex_msc}")
print(f"  LaTeX  : {_path_ens_tex_pct}")
print(f"  LaTeX  : {_path_res_tex}")
print(f"  Figure : {_fp_fan}")
print(f"  Figure : {_fp_heat}")
print(f"  Figure : {_fp_rec}")
print("=" * 80)
print("\n  Statistiques d'ensemble — résumé global :")
print(f"  • Modèles inclus : {', '.join(_NOMS_MOD)}")
print(f"  • CV moyen (toutes mines, toutes années) : "
      f"{TABLE_ENSEMBLE['PIB_CV_pct'].mean():.1f}%")
_best_agree = TABLE_ENSEMBLE.loc[TABLE_ENSEMBLE['PIB_CV_pct'].idxmin()]
print(f"  • Meilleur accord : {_best_agree['Mine_Label']} "
      f"en {int(_best_agree['Annee'])} "
      f"(CV = {_best_agree['PIB_CV_pct']:.1f}%)")
_worst_agree = TABLE_ENSEMBLE.dropna(subset=['PIB_CV_pct']).loc[
    TABLE_ENSEMBLE.dropna(subset=['PIB_CV_pct'])['PIB_CV_pct'].idxmax()
]
print(f"  • Plus forte divergence : {_worst_agree['Mine_Label']} "
      f"en {int(_worst_agree['Annee'])} "
      f"(CV = {_worst_agree['PIB_CV_pct']:.1f}%)")
print("=" * 80)
plt.rcdefaults()
plt.show()


#%% ENSEMBLE 3-MODÈLES INDÉPENDANTS — RÉSULTAT PRINCIPAL
###############################################################################
# Restriction de l'analyse d'incertitude aux trois modèles dont l'identification
# repose sur des sources de données distinctes :
#   - Emploi    : marché du travail (salaires × main-d'œuvre minière)
#   - OLS (MCO) : régression log-log de la luminosité satellitaire (VIIRS)
#   - StatCan   : comptabilité nationale (VAF + SAMP allouée par VIIRS)
#
# IV2SLS, Chen et Score sont écartés du calcul principal car leur identification
# est redondante : IV2SLS partage le DN VIIRS avec OLS (estimateur alternatif
# de la même équation); Chen alloue via VIIRS; Score est un composite des
# autres inputs (Emploi + VIIRS + production).
#
# Sortie : tableaux Excel/LaTeX et figures avec le suffixe _3mod
###############################################################################

_COLS_PIB_3 = ['PIB_Emploi', 'PIB_OLS', 'PIB_SC']
_COLS_TX_3  = ['TX_Emploi',  'TX_OLS',  'TX_SC']
_NOMS_MOD_3 = ['Emploi', 'OLS (MCO)', 'StatCan']

# ── 1. Calcul des statistiques d'ensemble sur 3 modèles ─────────────────────
_TC2005_3 = TABLE_COMP[TABLE_COMP['Annee'] >= 2005].reset_index(drop=True)

_pib_rows_3, _tx_rows_3 = [], []
for _, _row in _TC2005_3.iterrows():
    _pib_rows_3.append(_ens_stats(_row[_COLS_PIB_3].values, 'PIB_'))
    _tx_rows_3.append( _ens_stats(_row[_COLS_TX_3].values,  'TX_'))

TABLE_ENSEMBLE_3 = pd.concat(
    [_TC2005_3[['Annee', 'Mine', 'Mine_Label'] + _COLS_PIB_3 + _COLS_TX_3]
     .reset_index(drop=True),
     pd.DataFrame(_pib_rows_3).reset_index(drop=True),
     pd.DataFrame(_tx_rows_3).reset_index(drop=True)],
    axis=1
).sort_values(['Mine', 'Annee']).reset_index(drop=True)

# Affichage console — M$
print("\n" + "=" * 105)
print("  ENSEMBLE 3 MODÈLES INDÉPENDANTS — PIB par mine, Nunavut (M$ CAD, base 2017)")
print("  Modèles : Emploi, OLS (MCO), StatCan  |  N = nombre de modèles disponibles par cellule")
print("=" * 105)
_aff_pib_3 = ['Annee', 'Mine_Label', 'PIB_N', 'PIB_Moyenne', 'PIB_Mediane',
              'PIB_Ecart_type', 'PIB_CV_pct', 'PIB_Q25', 'PIB_Q75', 'PIB_Min', 'PIB_Max']
print(TABLE_ENSEMBLE_3[_aff_pib_3].to_string(index=False,
      float_format=lambda x: f'{x:,.1f}'))

# Affichage console — % PIB
print("\n" + "=" * 105)
print("  ENSEMBLE 3 MODÈLES INDÉPENDANTS — Part du PIB par mine (% du PIB Nunavut)")
print("=" * 105)
_aff_tx_3 = ['Annee', 'Mine_Label', 'TX_N', 'TX_Moyenne', 'TX_Mediane',
             'TX_Ecart_type', 'TX_CV_pct', 'TX_Q25', 'TX_Q75', 'TX_Min', 'TX_Max']
print(TABLE_ENSEMBLE_3[_aff_tx_3].to_string(index=False,
      float_format=lambda x: f'{x:,.2f}'))

# ── 2. Résumé par mine ───────────────────────────────────────────────────────
_TE2_3 = TABLE_ENSEMBLE_3[TABLE_ENSEMBLE_3['PIB_N'] >= 2].copy()

TABLE_RESUME_MINE_3 = (
    _TE2_3.groupby('Mine_Label')
    [['PIB_N',
      'PIB_Moyenne', 'PIB_Mediane', 'PIB_Ecart_type', 'PIB_CV_pct', 'PIB_Q25', 'PIB_Q75',
      'TX_Moyenne',  'TX_Mediane',  'TX_Ecart_type',  'TX_CV_pct',  'TX_Q25',  'TX_Q75']]
    .mean().round(2).reset_index()
)

print("\n" + "=" * 95)
print("  ENSEMBLE 3 MODÈLES — RÉSUMÉ PAR MINE (M$ CAD 2017, moyenne 2005-2024, N >= 2)")
print("=" * 95)
_cols_resume_pib_3 = ['Mine_Label', 'PIB_N', 'PIB_Moyenne', 'PIB_Mediane',
                      'PIB_Ecart_type', 'PIB_CV_pct', 'PIB_Q25', 'PIB_Q75']
print(TABLE_RESUME_MINE_3[_cols_resume_pib_3].to_string(
    index=False, float_format=lambda x: f'{x:,.2f}'))

print("\n" + "=" * 95)
print("  ENSEMBLE 3 MODÈLES — RÉSUMÉ PAR MINE (% du PIB Nunavut, moyenne 2005-2024)")
print("=" * 95)
_cols_resume_tx_3 = ['Mine_Label', 'PIB_N', 'TX_Moyenne', 'TX_Mediane',
                     'TX_Ecart_type', 'TX_CV_pct', 'TX_Q25', 'TX_Q75']
print(TABLE_RESUME_MINE_3[_cols_resume_tx_3].to_string(
    index=False, float_format=lambda x: f'{x:,.2f}'))

# ── 2 bis-3mod. Calcul TOTAL 4 mines (avant Excel et LaTeX 3-mod) ──────────
_BASE_AGG_3 = _TC2005_3[_TC2005_3['Mine_Label'].isin(_MINES_AGG)]

_TOT_PIB_3 = _BASE_AGG_3.groupby('Annee')[_COLS_PIB_3].sum(min_count=1).reset_index()
_TOT_TX_3  = _BASE_AGG_3.groupby('Annee')[_COLS_TX_3 ].sum(min_count=1).reset_index()

_tot_rows_pib_3, _tot_rows_tx_3 = [], []
for _, _r in _TOT_PIB_3.iterrows():
    _s = _ens_stats(_r[_COLS_PIB_3].values, 'PIB_')
    _s['Annee'] = int(_r['Annee']); _tot_rows_pib_3.append(_s)
for _, _r in _TOT_TX_3.iterrows():
    _s = _ens_stats(_r[_COLS_TX_3].values, 'TX_')
    _s['Annee'] = int(_r['Annee']); _tot_rows_tx_3.append(_s)

ENSEMBLE_TOTAL_3 = (
    pd.DataFrame(_tot_rows_pib_3)
    .merge(pd.DataFrame(_tot_rows_tx_3), on='Annee')
    .sort_values('Annee').reset_index(drop=True)
)
_TOT_BY_YEAR_3 = ENSEMBLE_TOTAL_3.set_index('Annee')

# Stats pluriannuelles du Total 4 mines (3 modèles)
_ENS_TOT_VALIDES_3 = ENSEMBLE_TOTAL_3[ENSEMBLE_TOTAL_3['PIB_N'] >= 2]
_TOTAL_RES_3 = {
    _k: _ENS_TOT_VALIDES_3[_k].mean()
    for _k in ['PIB_N', 'PIB_Moyenne', 'PIB_Mediane', 'PIB_Ecart_type',
               'PIB_CV_pct', 'PIB_Q25', 'PIB_Q75',
               'TX_Moyenne', 'TX_Mediane', 'TX_Ecart_type',
               'TX_CV_pct', 'TX_Q25', 'TX_Q75']
}

# ── 3. Export Excel ──────────────────────────────────────────────────────────
_path_ens_xl_3 = os.path.join(tab_dir, "tableau_ensemble_incertitude_3mod.xlsx")

with pd.ExcelWriter(_path_ens_xl_3, engine='openpyxl') as _xl:
    # Feuille longue + lignes Total 4 mines
    _tot_long_3 = ENSEMBLE_TOTAL_3.copy()
    _tot_long_3.insert(0, 'Mine',       'Total_4_mines')
    _tot_long_3.insert(1, 'Mine_Label', 'Total 4 mines')
    _ens_long_3_aug = pd.concat([TABLE_ENSEMBLE_3, _tot_long_3],
                                ignore_index=True, sort=False)
    _ens_long_3_aug.to_excel(_xl, sheet_name='Ensemble_Long', index=False)

    # Pivots M$ + colonne Total
    for _stat, _lbl in [('PIB_Moyenne', 'Moyenne_M$'),
                        ('PIB_Mediane', 'Mediane_M$'),
                        ('PIB_CV_pct',  'CV_pct_M$'),
                        ('PIB_IQR',     'IQR_M$')]:
        _piv = _TE2_3.pivot_table(index='Annee', columns='Mine_Label',
                                  values=_stat, aggfunc='mean').round(2)
        _piv['Total 4 mines'] = _TOT_BY_YEAR_3[_stat].round(2)
        _piv.to_excel(_xl, sheet_name=_lbl)

    # Pivots % + colonne Total
    for _stat, _lbl in [('TX_Moyenne', 'Moyenne_Pct'),
                        ('TX_Mediane', 'Mediane_Pct'),
                        ('TX_CV_pct',  'CV_pct_Pct'),
                        ('TX_IQR',     'IQR_Pct'),
                        ('TX_Q25',     'Q25_Pct'),
                        ('TX_Q75',     'Q75_Pct')]:
        _piv = _TE2_3.pivot_table(index='Annee', columns='Mine_Label',
                                  values=_stat, aggfunc='mean').round(2)
        _piv['Total 4 mines'] = _TOT_BY_YEAR_3[_stat].round(2)
        _piv.to_excel(_xl, sheet_name=_lbl)

    # Résumé par mine + ligne Total 4 mines
    _tot_resume_row_3 = pd.DataFrame([{
        'Mine_Label': 'Total 4 mines',
        **{_k: _v for _k, _v in _TOTAL_RES_3.items()}
    }])
    _resume_aug_3 = pd.concat(
        [TABLE_RESUME_MINE_3, _tot_resume_row_3],
        ignore_index=True, sort=False
    )
    _resume_aug_3.to_excel(_xl, sheet_name='Resume_par_mine', index=False)

    # N modèles + colonne Total
    _piv_n_3 = TABLE_ENSEMBLE_3.pivot_table(index='Annee', columns='Mine_Label',
                                            values='PIB_N', aggfunc='mean').round(0)
    _piv_n_3['Total 4 mines'] = _TOT_BY_YEAR_3['PIB_N'].round(0)
    _piv_n_3.to_excel(_xl, sheet_name='N_modeles')

    # NOUVELLE FEUILLE — totaux annuels 4 mines (3 modèles indépendants)
    ENSEMBLE_TOTAL_3.round(2).to_excel(
        _xl, sheet_name='Total_4_mines_annuel', index=False
    )

print(f"\n  ✔ Excel 3-modèles exporté : {_path_ens_xl_3}")

# ── 4. LaTeX — tableau d'ensemble par année (2 fichiers : M$ et %) ──────────
_path_ens_tex_3_msc = os.path.join(tab_dir, "tableau_ensemble_incertitude_3mod_valeurs.tex")
_path_ens_tex_3_pct = os.path.join(tab_dir, "tableau_ensemble_incertitude_3mod_parts.tex")
_mines_ord_3 = [m for m in ['Meadowbank', 'Meliadine', 'Hope Bay', 'Baffinland']
                if m in _TE2_3['Mine_Label'].unique()]
_annees_3 = sorted(_TE2_3['Annee'].unique().astype(int))

# ENSEMBLE_TOTAL_3 est déjà calculé plus haut (section 3 Export Excel)
# avant les exports Excel et LaTeX qui l'utilisent.

# Fichier 1 : M$ CAD 2017
with open(_path_ens_tex_3_msc, 'w', encoding='utf-8') as _f:
    _write_ens_table_latex(
        _f, _TE2_3, _mines_ord_3, _annees_3,
        col_med='PIB_Mediane', col_q25='PIB_Q25', col_q75='PIB_Q75', col_cv='PIB_CV_pct',
        caption="Statistiques d'ensemble du PIB par mine --- Nunavut "
                "(M\\$ CAD, base 2017, 3 modèles indépendants)",
        label="tab:ensemble_pib_msca_3mod",
        fmt_med=lambda v: _fmt_fr(v, 0),
        fmt_ic =lambda v: _fmt_fr(v, 0),
        fmt_cv =lambda v: _fmt_fr(v, 1),
        note_unit="Valeurs en M\\$ CAD 2017. ",
        total_df=ENSEMBLE_TOTAL_3,
        tot_col_med='PIB_Mediane', tot_col_cv='PIB_CV_pct',
        tot_col_q25='PIB_Q25', tot_col_q75='PIB_Q75',
        compact=True,
        models_desc=("Moyennage équipondéré entre Emploi, OLS (MCO) et "
                     "StatCan. ")
    )
print(f"  ✔ LaTeX 3-modèles (M$) exporté : {_path_ens_tex_3_msc}")

# Fichier 2 : % du PIB Nunavut
with open(_path_ens_tex_3_pct, 'w', encoding='utf-8') as _f:
    _write_ens_table_latex(
        _f, _TE2_3, _mines_ord_3, _annees_3,
        col_med='TX_Mediane', col_q25='TX_Q25', col_q75='TX_Q75', col_cv='TX_CV_pct',
        caption="Statistiques d'ensemble de la part du PIB par mine --- Nunavut "
                "(\\% du PIB territorial, 3 modèles indépendants)",
        label="tab:ensemble_pib_pct_3mod",
        fmt_med=lambda v: _fmt_fr(v, 1),
        fmt_ic =lambda v: _fmt_fr(v, 1),
        fmt_cv =lambda v: _fmt_fr(v, 1),
        note_unit="Valeurs en \\% du PIB Nunavut. ",
        total_df=ENSEMBLE_TOTAL_3,
        tot_col_med='TX_Mediane', tot_col_cv='TX_CV_pct',
        tot_col_q25='TX_Q25', tot_col_q75='TX_Q75',
        compact=True,
        models_desc=("Moyennage équipondéré entre Emploi, OLS (MCO) et "
                     "StatCan. ")
    )
print(f"  ✔ LaTeX 3-modèles (%) exporté : {_path_ens_tex_3_pct}")

# ── 5. LaTeX — résumé par mine ──────────────────────────────────────────────
_path_res_tex_3 = os.path.join(tab_dir, "tableau_resume_ensemble_mine_3mod.tex")
_note_mod_3 = ("Trois estimateurs indépendants par construction~: "
               "Emploi (marché du travail), OLS (MCO --- luminosité satellitaire VIIRS), "
               "StatCan (comptabilité nationale). CV = écart-type / moyenne $\\times$ 100. "
               "$N_{\\text{moy}}$ = nombre moyen de modèles disponibles par année. "
               "« Total 4 mines » = recalcul d'ensemble sur la somme des 4 "
               "mines modèle par modèle.")

# _TOTAL_RES_3 est déjà calculé plus haut (section 2 bis-3mod) avant l'Excel.

with open(_path_res_tex_3, 'w', encoding='utf-8') as _f:
    # Table A — M$
    _f.write("\\begin{table}[htbp]\n\\centering\n")
    _f.write("\\caption{Statistiques descriptives des estimations du PIB par mine "
             "--- Nunavut (M\\$ CAD, base 2017) --- Ensemble 3 modèles indépendants"
             "\\newline\\footnotesize Moyennage équipondéré. " + _note_mod_3 + "}\n")
    _f.write("\\label{tab:resume_ensemble_mine_msc_3mod}\n")
    _f.write("\\begin{tabular}{lrrrrrrr}\n\\hline\\hline\n")
    _f.write("Mine & $N_{\\text{moy}}$ & Moyenne & Médiane & "
             "Écart-type & CV\\,\\% & Q25 & Q75 \\\\\n")
    _f.write(" & & \\multicolumn{6}{c}{(M\\$ CAD, base 2017)} \\\\\n")
    _f.write("\\hline\n")
    for _, _r in TABLE_RESUME_MINE_3.iterrows():
        _f.write(
            f"{_r['Mine_Label']} & {_fmt_fr(_r['PIB_N'], 1)} & "
            f"{_fmt_fr(_r['PIB_Moyenne'], 1)} & {_fmt_fr(_r['PIB_Mediane'], 1)} & "
            f"{_fmt_fr(_r['PIB_Ecart_type'], 1)} & {_fmt_fr(_r['PIB_CV_pct'], 1)} & "
            f"{_fmt_fr(_r['PIB_Q25'], 1)} & {_fmt_fr(_r['PIB_Q75'], 1)} \\\\\n"
        )
    # Ligne Total 4 mines
    _f.write("\\hline\n")
    _f.write(
        f"\\textbf{{Total 4 mines}} & {_fmt_fr(_TOTAL_RES_3['PIB_N'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES_3['PIB_Moyenne'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES_3['PIB_Mediane'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES_3['PIB_Ecart_type'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES_3['PIB_CV_pct'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES_3['PIB_Q25'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES_3['PIB_Q75'], 1)} \\\\\n"
    )
    _f.write("\\hline\\hline\n")
    _f.write("\\multicolumn{8}{l}{\\footnotesize \\textit{Note :} " + _note_mod_3 + "} \\\\\n")
    _f.write("\\end{tabular}\n\\end{table}\n\n")

    # Table B — % PIB
    _f.write("\\begin{table}[htbp]\n\\centering\n")
    _f.write("\\caption{Statistiques descriptives de la part du PIB par mine "
             "--- Nunavut (\\% du PIB territorial) --- Ensemble 3 modèles indépendants"
             "\\newline\\footnotesize Moyennage équipondéré. " + _note_mod_3 + "}\n")
    _f.write("\\label{tab:resume_ensemble_mine_pct_3mod}\n")
    _f.write("\\begin{tabular}{lrrrrrrr}\n\\hline\\hline\n")
    _f.write("Mine & $N_{\\text{moy}}$ & Moyenne & Médiane & "
             "Écart-type & CV\\,\\% & Q25 & Q75 \\\\\n")
    _f.write(" & & \\multicolumn{6}{c}{(\\% du PIB Nunavut)} \\\\\n")
    _f.write("\\hline\n")
    for _, _r in TABLE_RESUME_MINE_3.iterrows():
        _f.write(
            f"{_r['Mine_Label']} & {_fmt_fr(_r['PIB_N'], 1)} & "
            f"{_fmt_fr(_r['TX_Moyenne'], 2)} & {_fmt_fr(_r['TX_Mediane'], 2)} & "
            f"{_fmt_fr(_r['TX_Ecart_type'], 2)} & {_fmt_fr(_r['TX_CV_pct'], 1)} & "
            f"{_fmt_fr(_r['TX_Q25'], 2)} & {_fmt_fr(_r['TX_Q75'], 2)} \\\\\n"
        )
    # Ligne Total 4 mines
    _f.write("\\hline\n")
    _f.write(
        f"\\textbf{{Total 4 mines}} & {_fmt_fr(_TOTAL_RES_3['PIB_N'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES_3['TX_Moyenne'], 2)} & "
        f"{_fmt_fr(_TOTAL_RES_3['TX_Mediane'], 2)} & "
        f"{_fmt_fr(_TOTAL_RES_3['TX_Ecart_type'], 2)} & "
        f"{_fmt_fr(_TOTAL_RES_3['TX_CV_pct'], 1)} & "
        f"{_fmt_fr(_TOTAL_RES_3['TX_Q25'], 2)} & "
        f"{_fmt_fr(_TOTAL_RES_3['TX_Q75'], 2)} \\\\\n"
    )
    _f.write("\\hline\\hline\n")
    _f.write("\\multicolumn{8}{l}{\\footnotesize \\textit{Note :} " + _note_mod_3 + "} \\\\\n")
    _f.write("\\end{tabular}\n\\end{table}\n")
print(f"  ✔ LaTeX résumé par mine 3-modèles exporté : {_path_res_tex_3}")

# ── 6. Fan charts (3 modèles) ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        12,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'axes.linewidth':   0.8,
    'axes.edgecolor':   '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'grid.color':       '#CCCCCC',
    'grid.linewidth':   0.6,
})

_mines_fan_3 = [m for m in ['Meadowbank', 'Meliadine', 'Hope Bay', 'Baffinland']
                if m in TABLE_ENSEMBLE_3['Mine_Label'].unique()]
_col_pib_mod_3 = dict(zip(_NOMS_MOD_3, _COLS_PIB_3))

fig_fan_3, axes_fan_3 = plt.subplots(2, 2, figsize=(16, 11), facecolor='white')
#fig_fan_3.suptitle(
#    "Estimations du PIB par mine — Nunavut (M\\$ CAD, base 2017)\n"
#    "Ensemble 3 modèles indépendants : Emploi, OLS (MCO), StatCan",
#    fontsize=14, fontweight='bold', y=1.01, fontfamily='serif'
#)

for _ax, _mine in zip(axes_fan_3.flatten(), _mines_fan_3):
    _c  = _C_MINE.get(_mine, '#444444')
    _dm = TABLE_ENSEMBLE_3[TABLE_ENSEMBLE_3['Mine_Label'] == _mine].sort_values('Annee')
    _dm = _dm[_dm['PIB_N'] >= 1].reset_index(drop=True)

    if _dm.empty:
        _ax.set_visible(False); continue

    _ann = _dm['Annee'].values.astype(int)

    _ax.fill_between(_ann, _dm['PIB_Min'], _dm['PIB_Max'],
                     color=_c, alpha=0.12, label='[Min ; Max]', zorder=2)
    _ax.fill_between(_ann,
                     _dm['PIB_Q25'].where(_dm['PIB_N'] >= 2),
                     _dm['PIB_Q75'].where(_dm['PIB_N'] >= 2),
                     color=_c, alpha=0.28,
                     label='IC$_{50}$ [Q25 ; Q75]', zorder=3)
    _mask_med = _dm['PIB_N'] >= 1
    _ax.plot(_ann[_mask_med], _dm.loc[_mask_med, 'PIB_Mediane'],
             color=_c, linewidth=2.4, linestyle='-', zorder=5,
             marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.5,
             label='Médiane')
    _mask_moy = _dm['PIB_N'] >= 2
    _ax.plot(_ann[_mask_moy], _dm.loc[_mask_moy, 'PIB_Moyenne'],
             color=_c, linewidth=1.6, linestyle='--', zorder=4, alpha=0.85,
             label='Moyenne')

    for _nom_m, _col_m in _col_pib_mod_3.items():
        _dm_mod = TABLE_ENSEMBLE_3[
            (TABLE_ENSEMBLE_3['Mine_Label'] == _mine) &
            TABLE_ENSEMBLE_3[_col_m].notna()
        ].sort_values('Annee')
        if _dm_mod.empty: continue
        _ax.scatter(
            _dm_mod['Annee'], _dm_mod[_col_m],
            color=_C_MOD.get(_nom_m, '#888888'),
            s=36, alpha=0.85, zorder=6,
            marker='D', linewidths=0.4, edgecolors='white',
            label=_nom_m
        )

    _ax.set_title(_mine, fontsize=12, fontweight='bold',
                  color=_c, pad=6, fontfamily='serif')
    _ax.set_xlabel('Année', fontsize=12, labelpad=5)
    _ax.set_ylabel('PIB estimé (M\\$ CAD, base 2017)', fontsize=12, labelpad=5)
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    _ax.set_ylim(bottom=0)
    _ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
    _ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    _ax.tick_params(axis='x', labelrotation=45)
    _ax.spines['top'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    _ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
    _ax.set_axisbelow(True)
    _ax.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC',
               ncol=2, loc='upper left', borderpad=0.5, handlelength=1.4)

plt.tight_layout(h_pad=3.5, w_pad=3.0)
_fp_fan_3 = os.path.join(fig_dir, "fan_chart_PIB_ensemble_mines_3mod.png")
plt.savefig(_fp_fan_3, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Fan chart 3-modèles exporté : {_fp_fan_3}")

# ── 6 bis. Fan charts en % du PIB (3 modèles) ───────────────────────────────
_col_tx_mod_3 = dict(zip(_NOMS_MOD_3, _COLS_TX_3))

fig_fan_3p, axes_fan_3p = plt.subplots(2, 2, figsize=(16, 11), facecolor='white')
#fig_fan_3p.suptitle(
#    "Part du PIB par mine — Nunavut (\\% du PIB territorial)\n"
#    "Ensemble 3 modèles indépendants : Emploi, OLS (MCO), StatCan",
#    fontsize=14, fontweight='bold', y=1.01, fontfamily='serif'
#)

for _ax, _mine in zip(axes_fan_3p.flatten(), _mines_fan_3):
    _c  = _C_MINE.get(_mine, '#444444')
    _dm = TABLE_ENSEMBLE_3[TABLE_ENSEMBLE_3['Mine_Label'] == _mine].sort_values('Annee')
    _dm = _dm[_dm['TX_N'] >= 1].reset_index(drop=True)

    if _dm.empty:
        _ax.set_visible(False); continue

    _ann = _dm['Annee'].values.astype(int)

    _ax.fill_between(_ann, _dm['TX_Min'], _dm['TX_Max'],
                     color=_c, alpha=0.12, label='[Min ; Max]', zorder=2)
    _ax.fill_between(_ann,
                     _dm['TX_Q25'].where(_dm['TX_N'] >= 2),
                     _dm['TX_Q75'].where(_dm['TX_N'] >= 2),
                     color=_c, alpha=0.28,
                     label='IC$_{50}$ [Q25 ; Q75]', zorder=3)
    _mask_med = _dm['TX_N'] >= 1
    _ax.plot(_ann[_mask_med], _dm.loc[_mask_med, 'TX_Mediane'],
             color=_c, linewidth=2.4, linestyle='-', zorder=5,
             marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.5,
             label='Médiane')
    _mask_moy = _dm['TX_N'] >= 2
    _ax.plot(_ann[_mask_moy], _dm.loc[_mask_moy, 'TX_Moyenne'],
             color=_c, linewidth=1.6, linestyle='--', zorder=4, alpha=0.85,
             label='Moyenne')

    for _nom_m, _col_m in _col_tx_mod_3.items():
        _dm_mod = TABLE_ENSEMBLE_3[
            (TABLE_ENSEMBLE_3['Mine_Label'] == _mine) &
            TABLE_ENSEMBLE_3[_col_m].notna()
        ].sort_values('Annee')
        if _dm_mod.empty: continue
        _ax.scatter(
            _dm_mod['Annee'], _dm_mod[_col_m],
            color=_C_MOD.get(_nom_m, '#888888'),
            s=36, alpha=0.85, zorder=6,
            marker='D', linewidths=0.4, edgecolors='white',
            label=_nom_m
        )

    _ax.set_title(_mine, fontsize=12, fontweight='bold',
                  color=_c, pad=6, fontfamily='serif')
    _ax.set_xlabel('Année', fontsize=12, labelpad=5)
    _ax.set_ylabel('Part du PIB du Nunavut (\\%)', fontsize=12, labelpad=5)
    _ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}\\%'))
    _ax.set_ylim(bottom=0)
    _ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
    _ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    _ax.tick_params(axis='x', labelrotation=45)
    _ax.spines['top'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    _ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
    _ax.set_axisbelow(True)
    _ax.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC',
               ncol=2, loc='upper left', borderpad=0.5, handlelength=1.4)

plt.tight_layout(h_pad=3.5, w_pad=3.0)
_fp_fan_3p = os.path.join(fig_dir, "fan_chart_PIB_ensemble_mines_3mod_pct.png")
plt.savefig(_fp_fan_3p, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Fan chart 3-modèles (% PIB) exporté : {_fp_fan_3p}")

# ── 7. Carte thermique CV (3 modèles) ───────────────────────────────────────
_piv_cv_heat_3 = TABLE_ENSEMBLE_3.pivot_table(
    index='Mine_Label', columns='Annee',
    values='PIB_CV_pct', aggfunc='mean'
).reindex(index=_mines_fan_3)
_mines_heat_3 = [m for m in _mines_fan_3 if m in _piv_cv_heat_3.index]
_piv_cv_heat_3 = _piv_cv_heat_3.reindex(index=_mines_heat_3)
_piv_n_heat_3 = TABLE_ENSEMBLE_3.pivot_table(
    index='Mine_Label', columns='Annee',
    values='PIB_N', aggfunc='mean'
).reindex(index=_mines_heat_3)

fig_heat_3, ax_heat_3 = plt.subplots(
    figsize=(max(12, len(_piv_cv_heat_3.columns) * 0.65), 4.5),
    facecolor='white'
)
_cv_vals_3 = _piv_cv_heat_3.values.astype(float)
_vmax_3    = np.nanpercentile(_cv_vals_3, 95) if not np.all(np.isnan(_cv_vals_3)) else 50

_im_3 = ax_heat_3.imshow(
    _cv_vals_3, aspect='auto', cmap='RdYlGn_r',
    vmin=0, vmax=max(_vmax_3, 50), interpolation='nearest'
)

_n_rows_3, _n_cols_3 = _cv_vals_3.shape
for _r in range(_n_rows_3):
    for _cl in range(_n_cols_3):
        _val_cv = _cv_vals_3[_r, _cl]
        _val_n  = _piv_n_heat_3.values[_r, _cl] if not _piv_n_heat_3.empty else np.nan
        if np.isnan(_val_cv):
            _txt = '—'; _col_txt = '#AAAAAA'
        else:
            _txt = f'{_val_cv:.0f}%\n(N={int(_val_n)})'
            _col_txt = 'white' if _val_cv > _vmax_3 * 0.55 else '#222222'
        ax_heat_3.text(_cl, _r, _txt, ha='center', va='center',
                       fontsize=7.5, color=_col_txt, fontfamily='serif')

ax_heat_3.set_xticks(range(_n_cols_3))
ax_heat_3.set_xticklabels([str(int(c)) for c in _piv_cv_heat_3.columns],
                          rotation=55, ha='right', fontsize=9)
ax_heat_3.set_yticks(range(_n_rows_3))
ax_heat_3.set_yticklabels(_mines_heat_3, fontsize=10, fontfamily='serif')
_cbar_3 = plt.colorbar(_im_3, ax=ax_heat_3, pad=0.015, fraction=0.025)
_cbar_3.set_label("Coefficient de variation (%)", fontsize=10, fontfamily='serif')
_cbar_3.ax.tick_params(labelsize=9)
ax_heat_3.set_title(
#    "Incertitude inter-modèles — Ensemble 3 modèles indépendants (Emploi, OLS (MCO), StatCan)\n"
    "Vert : fort consensus  |  Rouge : forte divergence  |  N : nombre de modèles disponibles",
    fontsize=12, fontweight='bold', pad=8, fontfamily='serif'
)
ax_heat_3.set_xlabel('Année', fontsize=12, labelpad=6)
plt.tight_layout()
_fp_heat_3 = os.path.join(fig_dir, "heatmap_CV_incertitude_modeles_3mod.png")
plt.savefig(_fp_heat_3, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Carte thermique 3-modèles exportée : {_fp_heat_3}")

# ── 8. Graphique consensus (3 modèles) ──────────────────────────────────────
_TE3_3 = TABLE_ENSEMBLE_3[
    (TABLE_ENSEMBLE_3['PIB_N'] >= 2) &
    (TABLE_ENSEMBLE_3['Mine_Label'].isin(_mines_fan_3))
].copy()

_agg_tot_3 = (
    _TE3_3.groupby('Annee')
    .agg(
        PIB_Mediane_tot = ('PIB_Mediane', 'sum'),
        PIB_Q25_tot     = ('PIB_Q25',     'sum'),
        PIB_Q75_tot     = ('PIB_Q75',     'sum'),
        PIB_CV_moy      = ('PIB_CV_pct',  'mean'),
        N_mines         = ('Mine_Label',  'count'),
    ).reset_index().sort_values('Annee')
)

fig_rec_3, (ax_rec_3, ax_cv_3) = plt.subplots(
    2, 1, figsize=(14, 9), facecolor='white',
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.35}
)

_annees_rec_3 = sorted(_TE3_3['Annee'].unique().astype(int))
_x_rec_3      = np.arange(len(_annees_rec_3))
_bottom_3     = np.zeros(len(_annees_rec_3))

for _mine in _mines_fan_3:
    _dm_r = (_TE3_3[_TE3_3['Mine_Label'] == _mine]
             .set_index('Annee').reindex(_annees_rec_3))
    _vals  = _dm_r['PIB_Mediane'].fillna(0).values
    _err_lo = (_dm_r['PIB_Mediane'] - _dm_r['PIB_Q25']).fillna(0).values
    _err_hi = (_dm_r['PIB_Q75']     - _dm_r['PIB_Mediane']).fillna(0).values
    _c_r    = _C_MINE.get(_mine, '#888888')
    ax_rec_3.bar(_x_rec_3, _vals, bottom=_bottom_3,
                 color=_c_r, alpha=0.82, edgecolor='white', linewidth=0.3,
                 width=0.70, zorder=3, label=_mine)
    ax_rec_3.errorbar(_x_rec_3, _bottom_3 + _vals,
                      yerr=[_err_lo, _err_hi], fmt='none', ecolor=_c_r,
                      elinewidth=1.2, capsize=3, capthick=1.0, alpha=0.70, zorder=4)
    _bottom_3 = _bottom_3 + _vals

ax_rec_3.plot(_x_rec_3,
              _agg_tot_3.set_index('Annee').reindex(_annees_rec_3)['PIB_Mediane_tot'].values,
              color='#222222', linewidth=1.8, linestyle='--',
              marker='D', markersize=4.5, markeredgecolor='white', markeredgewidth=0.5,
              zorder=5, label='Total (médiane)')

ax_rec_3.set_xticks(_x_rec_3)
ax_rec_3.set_xticklabels([str(a) for a in _annees_rec_3], rotation=50, ha='right')
ax_rec_3.set_xlim(-0.55, len(_annees_rec_3) - 0.45)
ax_rec_3.set_ylim(bottom=0)
ax_rec_3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax_rec_3.set_ylabel('PIB estimé — médiane (M\\$ CAD, base 2017)', fontsize=12, labelpad=5)
ax_rec_3.set_title(
#    "Estimation consensus du PIB par mine — Nunavut (Ensemble 3 modèles indépendants)\n"
    "Médiane inter-modèles + barres d'erreur [Q25\\,;\\,Q75]",
    fontsize=13, fontweight='bold', pad=8, fontfamily='serif'
)
ax_rec_3.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC', ncol=2, loc='upper left')
ax_rec_3.spines['top'].set_visible(False)
ax_rec_3.spines['right'].set_visible(False)
ax_rec_3.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
ax_rec_3.set_axisbelow(True)

_cv_rec_3 = _agg_tot_3.set_index('Annee').reindex(_annees_rec_3)['PIB_CV_moy'].values
_colors_cv_3 = ['#27AE60' if v < 20 else ('#F39C12' if v < 40 else '#C0392B')
                for v in np.nan_to_num(_cv_rec_3, nan=0)]
ax_cv_3.bar(_x_rec_3, _cv_rec_3, color=_colors_cv_3, alpha=0.82,
            edgecolor='white', linewidth=0.3, width=0.70, zorder=3)
ax_cv_3.axhline(20, color='#27AE60', linewidth=1.0, linestyle='--', alpha=0.7,
                label='Seuil 20 % (accord fort)')
ax_cv_3.axhline(40, color='#C0392B', linewidth=1.0, linestyle='--', alpha=0.7,
                label='Seuil 40 % (divergence)')
ax_cv_3.set_xticks(_x_rec_3)
ax_cv_3.set_xticklabels([str(a) for a in _annees_rec_3], rotation=50, ha='right')
ax_cv_3.set_xlim(-0.55, len(_annees_rec_3) - 0.45)
ax_cv_3.set_ylim(bottom=0)
ax_cv_3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}\\%'))
ax_cv_3.set_ylabel('CV moyen (\\%)', fontsize=12, labelpad=5)
ax_cv_3.set_title('Incertitude inter-modèles (CV moyen)', fontsize=11)
ax_cv_3.legend(fontsize=10, framealpha=0.92, edgecolor='#CCCCCC', loc='upper right')
ax_cv_3.spines['top'].set_visible(False)
ax_cv_3.spines['right'].set_visible(False)
ax_cv_3.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#CCCCCC', zorder=0)
ax_cv_3.set_axisbelow(True)

plt.tight_layout()
_fp_rec_3 = os.path.join(fig_dir, "fig_consensus_PIB_mines_ensemble_3mod.png")
plt.savefig(_fp_rec_3, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Graphique consensus 3-modèles exporté : {_fp_rec_3}")

# ── 9. Récapitulatif console (3 modèles) ────────────────────────────────────
print("\n" + "=" * 80)
print("  ANALYSE D'INCERTITUDE — ENSEMBLE 3 MODÈLES INDÉPENDANTS — FICHIERS GÉNÉRÉS")
print("=" * 80)
print(f"  Excel  : {_path_ens_xl_3}")
print(f"  LaTeX  : {_path_ens_tex_3_msc}")
print(f"  LaTeX  : {_path_ens_tex_3_pct}")
print(f"  LaTeX  : {_path_res_tex_3}")
print(f"  Figure : {_fp_fan_3}")
print(f"  Figure : {_fp_heat_3}")
print(f"  Figure : {_fp_rec_3}")
print("=" * 80)
print(f"  • Modèles inclus : {', '.join(_NOMS_MOD_3)}")
print(f"  • CV moyen (toutes mines, toutes années) : "
      f"{TABLE_ENSEMBLE_3['PIB_CV_pct'].mean():.1f}%")
_best_agree_3 = TABLE_ENSEMBLE_3.dropna(subset=['PIB_CV_pct']).loc[
    TABLE_ENSEMBLE_3.dropna(subset=['PIB_CV_pct'])['PIB_CV_pct'].idxmin()
]
print(f"  • Meilleur accord : {_best_agree_3['Mine_Label']} "
      f"en {int(_best_agree_3['Annee'])} "
      f"(CV = {_best_agree_3['PIB_CV_pct']:.1f}%)")
_worst_agree_3 = TABLE_ENSEMBLE_3.dropna(subset=['PIB_CV_pct']).loc[
    TABLE_ENSEMBLE_3.dropna(subset=['PIB_CV_pct'])['PIB_CV_pct'].idxmax()
]
print(f"  • Plus forte divergence : {_worst_agree_3['Mine_Label']} "
      f"en {int(_worst_agree_3['Annee'])} "
      f"(CV = {_worst_agree_3['PIB_CV_pct']:.1f}%)")
print("=" * 80)
plt.rcdefaults()
plt.show()


#%% VALIDATION EXTERNE — COMPARAISON DES MODÈLES À STATISTIQUE CANADA
################################################################################
# SOUS-SECTION : VALIDATION EXTERNE CONTRE LE BENCHMARK STATISTIQUE CANADA
#
# Justification méthodologique :
#   Statistique Canada (PIB_SC) fournit une décomposition par mine fondée sur
#   les comptes nationaux (VAF + SAMP allouée). Bien que cette série ne
#   constitue pas une « vérité-terrain » au sens strict (StatCan ne publie
#   pas officiellement de PIB par mine), elle représente la meilleure
#   référence institutionnelle disponible et sert ici de benchmark externe.
#
# Approche : pour chaque modèle X ∈ {Emploi, OLS, IV2SLS, Chen, Score},
#   on évalue l'écart à StatCan sur les cellules (Mine × Année) où les deux
#   séries sont simultanément disponibles. On calcule :
#
#     • MAE       : erreur absolue moyenne (M$)
#     • RMSE      : racine de l'erreur quadratique moyenne (M$)
#     • MAPE      : erreur absolue en pourcentage de StatCan (%)
#     • Biais     : erreur moyenne (M$, positif = surestime)
#     • Corr. Pearson  : corrélation linéaire
#     • Corr. Spearman : corrélation de rang
#     • Theil U₂  : ratio RMSE / RMSE_naïf (1 = naïf, <1 = mieux)
#     • n         : nombre de cellules de comparaison
#
# Sortie : tableau Excel + table LaTeX classée + scatter plots vs StatCan +
#          diagrammes de Bland-Altman.
################################################################################

from scipy.stats import pearsonr, spearmanr

print("\n" + "=" * 78)
print("  VALIDATION EXTERNE — COMPARAISON DES 5 MODÈLES À STATISTIQUE CANADA")
print("=" * 78)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Définition des modèles à valider et benchmark
# ─────────────────────────────────────────────────────────────────────────────
_MODELES_VAL = {
    'Emploi':        'PIB_Emploi',
    'OLS (MCO)':     'PIB_OLS',
    'IV2SLS':        'PIB_IV',
    'Chen':          'PIB_CHEN',
    'Score pondéré': 'PIB_SCORE',
}
_BENCHMARK = 'PIB_SC'

_BASE_VAL = TABLE_COMP[
    (TABLE_COMP['Annee'] >= 2005) & (TABLE_COMP['Annee'] <= 2024)
].copy()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Calcul des métriques de validation (global + par mine)
# ─────────────────────────────────────────────────────────────────────────────
def _metrics_vs_sc(y_mod, y_sc):
    """Métriques de validation d'un modèle vs StatCan sur cellules communes.

    Utilise sMAPE (symmetric MAPE) au lieu du MAPE classique pour rester
    borné [0, 200] % même quand StatCan ≈ 0 :
        sMAPE = 100 × |mod-sc| / ((|mod|+|sc|)/2)
    """
    _df = pd.DataFrame({'mod': pd.to_numeric(y_mod, errors='coerce'),
                        'sc':  pd.to_numeric(y_sc,  errors='coerce')}).dropna()
    n = len(_df)
    if n < 3:
        return {'n': n, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan,
                'Biais': np.nan, 'Pearson': np.nan, 'Spearman': np.nan,
                'Theil_U2': np.nan}
    err = _df['mod'].values - _df['sc'].values
    mae   = float(np.mean(np.abs(err)))
    rmse  = float(np.sqrt(np.mean(err**2)))
    # sMAPE — borné [0, 200] %, robuste aux cellules StatCan ≈ 0
    _denom = (np.abs(_df['mod'].values) + np.abs(_df['sc'].values)) / 2.0
    _denom_safe = np.where(_denom == 0, np.nan, _denom)
    mape  = float(np.nanmean(np.abs(err) / _denom_safe) * 100)
    biais = float(np.mean(err))
    # Corrélations
    try:
        pear = float(pearsonr(_df['mod'], _df['sc'])[0])
    except Exception:
        pear = np.nan
    try:
        spear = float(spearmanr(_df['mod'], _df['sc'])[0])
    except Exception:
        spear = np.nan
    # Theil's U2 : RMSE_modele / RMSE_prevision_naive (sc_lag)
    _sc_sorted = _df['sc'].values
    if len(_sc_sorted) > 1:
        rmse_naif = float(np.sqrt(np.mean(np.diff(_sc_sorted)**2)))
        u2 = rmse / rmse_naif if rmse_naif > 0 else np.nan
    else:
        u2 = np.nan
    return {'n': n, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
            'Biais': biais, 'Pearson': pear, 'Spearman': spear,
            'Theil_U2': u2}

# Global (toutes mines confondues)
_VAL_GLOBAL_ROWS = []
for _nom_m, _col_m in _MODELES_VAL.items():
    _m = _metrics_vs_sc(_BASE_VAL[_col_m], _BASE_VAL[_BENCHMARK])
    _m['Modele'] = _nom_m
    _VAL_GLOBAL_ROWS.append(_m)
VALIDATION_GLOBAL = pd.DataFrame(_VAL_GLOBAL_ROWS)[
    ['Modele', 'n', 'MAE', 'RMSE', 'MAPE', 'Biais',
     'Pearson', 'Spearman', 'Theil_U2']
].sort_values('MAPE')

print("\n  --- Métriques de validation externe vs StatCan (toutes mines, 2005–2024) ---")
print(VALIDATION_GLOBAL.to_string(index=False,
                                  float_format=lambda x: f'{x:,.2f}'))

# Par mine
_VAL_MINE_ROWS = []
for _mine in _BASE_VAL['Mine'].dropna().unique():
    _sub = _BASE_VAL[_BASE_VAL['Mine'] == _mine]
    for _nom_m, _col_m in _MODELES_VAL.items():
        _m = _metrics_vs_sc(_sub[_col_m], _sub[_BENCHMARK])
        _m['Mine']   = _mine
        _m['Modele'] = _nom_m
        _VAL_MINE_ROWS.append(_m)
VALIDATION_PAR_MINE = pd.DataFrame(_VAL_MINE_ROWS)[
    ['Mine', 'Modele', 'n', 'MAE', 'RMSE', 'MAPE', 'Biais',
     'Pearson', 'Spearman', 'Theil_U2']
].sort_values(['Mine', 'MAPE'])

print("\n  --- Métriques de validation externe vs StatCan, par mine ---")
print(VALIDATION_PAR_MINE.to_string(index=False,
                                    float_format=lambda x: f'{x:,.2f}'))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Classement composite (rang moyen sur MAPE + RMSE + 1-Pearson)
# ─────────────────────────────────────────────────────────────────────────────
_R = VALIDATION_GLOBAL.copy()
_R['rang_MAPE']    = _R['MAPE'].rank(method='min')
_R['rang_RMSE']    = _R['RMSE'].rank(method='min')
_R['rang_Pearson'] = (-_R['Pearson']).rank(method='min')   # plus haut = mieux
_R['rang_TheilU2'] = _R['Theil_U2'].rank(method='min')
_R['Rang_moyen']   = _R[['rang_MAPE', 'rang_RMSE',
                          'rang_Pearson', 'rang_TheilU2']].mean(axis=1)
VALIDATION_CLASSEMENT = (_R[['Modele', 'MAPE', 'RMSE', 'Pearson',
                              'Theil_U2', 'Rang_moyen']]
                         .sort_values('Rang_moyen')
                         .reset_index(drop=True))
VALIDATION_CLASSEMENT.insert(0, 'Position',
                              range(1, len(VALIDATION_CLASSEMENT) + 1))

print("\n  --- Classement composite des modèles vs StatCan ---")
print(VALIDATION_CLASSEMENT.to_string(index=False,
                                       float_format=lambda x: f'{x:,.2f}'))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Export Excel multi-feuilles
# ─────────────────────────────────────────────────────────────────────────────
_path_val_xl = os.path.join(tab_dir, "tableau_validation_externe_StatCan.xlsx")
try:
    with pd.ExcelWriter(_path_val_xl, engine="openpyxl") as _xl:
        VALIDATION_GLOBAL.to_excel(_xl, sheet_name='Global',     index=False)
        VALIDATION_PAR_MINE.to_excel(_xl, sheet_name='Par_mine', index=False)
        VALIDATION_CLASSEMENT.to_excel(_xl, sheet_name='Classement', index=False)
        # Pivot MAPE par mine x modele
        _piv_mape = (VALIDATION_PAR_MINE
                     .pivot(index='Mine', columns='Modele', values='MAPE')
                     .round(2))
        _piv_mape.to_excel(_xl, sheet_name='MAPE_pivot')
    print(f"\n  ✔ Excel exporté : {_path_val_xl}")
except Exception as _e:
    print(f"  ⚠ Export Excel échoué : {_e}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Table LaTeX — Classement composite
# ─────────────────────────────────────────────────────────────────────────────
_path_val_tex = os.path.join(tab_dir, "tableau_validation_externe_StatCan.tex")
try:
    with open(_path_val_tex, 'w', encoding='utf-8') as _f:
        _f.write("\\begin{table}[htbp]\n\\centering\n")
        _f.write("\\caption{Validation externe des cinq modèles par rapport au "
                 "benchmark Statistique Canada (PIB minier décomposé via "
                 "VAF~+~SAMP). Métriques calculées sur les cellules "
                 "(mine~$\\times$~année) communes, 2005--2024.}\n")
        _f.write("\\label{tab:validation_externe_statcan}\n")
        _f.write("\\begin{tabular}{clrrrrr}\n\\hline\\hline\n")
        _f.write("Rang & Modèle & sMAPE\\,\\% & RMSE (M\\$) & "
                 "Corr.\\ Pearson & Theil U$_2$ & Rang moyen \\\\\n")
        _f.write("\\hline\n")
        for _, _r in VALIDATION_CLASSEMENT.iterrows():
            _f.write(
                f"{int(_r['Position'])} & {_r['Modele']} & "
                f"{_fmt_fr(_r['MAPE'], 1)} & {_fmt_fr(_r['RMSE'], 1)} & "
                f"{_fmt_fr(_r['Pearson'], 3)} & {_fmt_fr(_r['Theil_U2'], 3)} & "
                f"{_fmt_fr(_r['Rang_moyen'], 2)} \\\\\n"
            )
        _f.write("\\hline\\hline\n")
        _f.write("\\multicolumn{7}{l}{\\footnotesize \\textit{Note :} "
                 "sMAPE (symmetric MAPE) borné dans [0\\,;\\,200]\\,\\%, robuste "
                 "aux cellules StatCan proches de zéro. "
                 "Theil U$_2 < 1$ indique une meilleure prévision que le "
                 "modèle naïf (variation StatCan d'une période à l'autre). "
                 "Le rang moyen est calculé sur sMAPE, RMSE, $-$Pearson et "
                 "Theil U$_2$ (méthodologie Diebold-Mariano agrégée).} \\\\\n")
        _f.write("\\end{tabular}\n\\end{table}\n")
    print(f"  ✔ LaTeX exporté : {_path_val_tex}")
except Exception as _e:
    print(f"  ⚠ Export LaTeX échoué : {_e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Scatter plots — modèle vs StatCan (5 sous-graphiques + droite 1:1)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'axes.linewidth':   0.8,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

fig_val, axes_val = plt.subplots(2, 3, figsize=(15, 9.5), facecolor='white')
axes_val_flat = axes_val.flatten()

for _ax_v, (_nom_m, _col_m) in zip(axes_val_flat, _MODELES_VAL.items()):
    _df_v = _BASE_VAL[['Mine_Label', _col_m, _BENCHMARK]].dropna()
    if _df_v.empty:
        _ax_v.set_visible(False); continue
    # Scatter coloré par mine
    for _mine_lab, _grp in _df_v.groupby('Mine_Label'):
        _c_v = _C_MINE.get(_mine_lab, '#888888')
        _ax_v.scatter(_grp[_BENCHMARK], _grp[_col_m],
                      color=_c_v, alpha=0.75, s=36, edgecolor='white',
                      linewidth=0.5, label=_mine_lab, zorder=3)
    # Droite 1:1
    _lim = max(_df_v[[_col_m, _BENCHMARK]].max().max(), 1)
    _ax_v.plot([0, _lim], [0, _lim], color='#222222', linestyle='--',
               linewidth=1.2, zorder=2, label='1:1')
    # Annotation R, MAPE
    _met = next(d for d in _VAL_GLOBAL_ROWS if d['Modele'] == _nom_m)
    _txt = (f"$r$ = {_met['Pearson']:.3f}\n"
            f"sMAPE = {_met['MAPE']:,.1f}\\%\n"
            f"Theil $U_2$ = {_met['Theil_U2']:.2f}\n"
            f"$n$ = {_met['n']}")
    _ax_v.text(0.04, 0.96, _txt, transform=_ax_v.transAxes,
               va='top', fontsize=9, family='serif',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='#CCCCCC', alpha=0.92))
    _ax_v.set_xlabel("PIB StatCan (M\\$ CAD 2017)")
    _ax_v.set_ylabel(f"PIB {_nom_m} (M\\$ CAD 2017)")
    _ax_v.set_title(f"{_nom_m}  vs  StatCan", fontweight='bold')
    _ax_v.set_xlim(0, _lim * 1.05)
    _ax_v.set_ylim(0, _lim * 1.05)
    _ax_v.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    _ax_v.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    _ax_v.spines['top'].set_visible(False)
    _ax_v.spines['right'].set_visible(False)
    _ax_v.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC')
    _ax_v.set_axisbelow(True)
    _ax_v.legend(fontsize=7.5, framealpha=0.9, loc='lower right', ncol=1)

# Masquer la 6e case (5 modèles → 6 cases en 2x3)
if len(_MODELES_VAL) < len(axes_val_flat):
    axes_val_flat[-1].set_visible(False)

plt.suptitle("Validation externe — Comparaison des modèles à Statistique Canada\n"
             "Diagrammes de dispersion (droite 1:1 en pointillés)",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
_fp_val_scat = os.path.join(fig_dir, "validation_externe_StatCan_scatter.png")
plt.savefig(_fp_val_scat, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Scatter plots sauvegardés : {_fp_val_scat}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Bland-Altman — différence (modèle − StatCan) vs moyenne
# ─────────────────────────────────────────────────────────────────────────────
fig_ba, axes_ba = plt.subplots(2, 3, figsize=(15, 9.5), facecolor='white')
axes_ba_flat = axes_ba.flatten()

for _ax_b, (_nom_m, _col_m) in zip(axes_ba_flat, _MODELES_VAL.items()):
    _df_b = _BASE_VAL[['Mine_Label', _col_m, _BENCHMARK]].dropna()
    if _df_b.empty:
        _ax_b.set_visible(False); continue
    _moy  = (_df_b[_col_m] + _df_b[_BENCHMARK]) / 2
    _diff = _df_b[_col_m] - _df_b[_BENCHMARK]
    _mean_d = float(_diff.mean())
    _sd_d   = float(_diff.std(ddof=1))
    _loa_up = _mean_d + 1.96 * _sd_d
    _loa_lo = _mean_d - 1.96 * _sd_d
    for _mine_lab, _grp_idx in _df_b.groupby('Mine_Label').groups.items():
        _c_b = _C_MINE.get(_mine_lab, '#888888')
        _ax_b.scatter(_moy.loc[_grp_idx], _diff.loc[_grp_idx],
                      color=_c_b, alpha=0.75, s=34, edgecolor='white',
                      linewidth=0.5, label=_mine_lab, zorder=3)
    _ax_b.axhline(_mean_d, color='#222222', linewidth=1.5, linestyle='-',
                  label=f'Biais = {_mean_d:,.1f}')
    _ax_b.axhline(_loa_up, color='#C0392B', linewidth=1.0, linestyle='--',
                  label=f'±1,96·σ = ±{1.96*_sd_d:,.1f}')
    _ax_b.axhline(_loa_lo, color='#C0392B', linewidth=1.0, linestyle='--')
    _ax_b.axhline(0, color='#888888', linewidth=0.6, linestyle=':')
    _ax_b.set_xlabel("Moyenne (Modèle, StatCan) (M\\$ CAD)")
    _ax_b.set_ylabel(f"{_nom_m} $-$ StatCan (M\\$ CAD)")
    _ax_b.set_title(f"Bland-Altman : {_nom_m}", fontweight='bold')
    _ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    _ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    _ax_b.spines['top'].set_visible(False)
    _ax_b.spines['right'].set_visible(False)
    _ax_b.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC')
    _ax_b.set_axisbelow(True)
    _ax_b.legend(fontsize=7.5, framealpha=0.9, loc='best', ncol=1)

if len(_MODELES_VAL) < len(axes_ba_flat):
    axes_ba_flat[-1].set_visible(False)

plt.suptitle("Validation externe — Diagrammes de Bland-Altman\n"
             "Biais (trait plein) et limites d'accord ±1,96·σ (pointillés rouges)",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
_fp_val_ba = os.path.join(fig_dir, "validation_externe_StatCan_BlandAltman.png")
plt.savefig(_fp_val_ba, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Bland-Altman sauvegardés : {_fp_val_ba}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Bar chart — Classement composite (rang moyen)
# ─────────────────────────────────────────────────────────────────────────────
fig_rank, ax_rank = plt.subplots(figsize=(10, 5.5), facecolor='white')
_palette_rank = ['#27AE60', '#3498DB', '#F39C12', '#9B59B6', '#C0392B']
_bars = ax_rank.barh(
    VALIDATION_CLASSEMENT['Modele'][::-1],
    VALIDATION_CLASSEMENT['Rang_moyen'][::-1],
    color=_palette_rank[:len(VALIDATION_CLASSEMENT)][::-1],
    alpha=0.85, edgecolor='white', linewidth=1.0,
)
for _i, _r in enumerate(VALIDATION_CLASSEMENT[::-1].itertuples()):
    ax_rank.text(_r.Rang_moyen + 0.05, _i,
                 f"  sMAPE={_r.MAPE:,.1f}%  |  r={_r.Pearson:.2f}",
                 va='center', fontsize=9, family='serif')
ax_rank.set_xlabel("Rang moyen (1 = meilleur accord avec StatCan)", fontsize=11)
ax_rank.set_title("Classement composite des modèles vs Statistique Canada",
                  fontsize=12, fontweight='bold')
ax_rank.invert_yaxis()
ax_rank.spines['top'].set_visible(False)
ax_rank.spines['right'].set_visible(False)
ax_rank.xaxis.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC')
ax_rank.set_axisbelow(True)
plt.tight_layout()
_fp_val_rank = os.path.join(fig_dir, "validation_externe_StatCan_classement.png")
plt.savefig(_fp_val_rank, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Classement sauvegardé : {_fp_val_rank}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. VALIDATION COMPLÉMENTAIRE — PIB TOTAL ANNUEL (somme sur les 4 mines)
#
# Justification : le test cellule (Mine × Année) pénalise toute erreur
# d'allocation entre mines. Cette validation complémentaire compare la
# SOMME annuelle des PIB estimés aux 4 mines à la SOMME StatCan.
# Les erreurs d'allocation se compensent — c'est un test plus lenient
# qui répond uniquement à : « le modèle estime-t-il correctement le PIB
# minier total du Nunavut ? »
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 78)
print("  VALIDATION COMPLÉMENTAIRE — PIB MINIER TOTAL ANNUEL vs StatCan")
print("-" * 78)

# Agrégation par année : somme sur les 4 mines
_cols_a_sommer = list(_MODELES_VAL.values()) + [_BENCHMARK]
_BASE_VAL_TOT = (
    _BASE_VAL[['Annee'] + _cols_a_sommer]
    .groupby('Annee', as_index=False)
    .sum(min_count=1)            # min_count=1 → NaN si aucune valeur non-nulle
)

# Calcul des métriques sur les totaux annuels
_VAL_TOTAL_ROWS = []
for _nom_m, _col_m in _MODELES_VAL.items():
    _m = _metrics_vs_sc(_BASE_VAL_TOT[_col_m], _BASE_VAL_TOT[_BENCHMARK])
    _m['Modele'] = _nom_m
    _VAL_TOTAL_ROWS.append(_m)
VALIDATION_TOTAL = pd.DataFrame(_VAL_TOTAL_ROWS)[
    ['Modele', 'n', 'MAE', 'RMSE', 'MAPE', 'Biais',
     'Pearson', 'Spearman', 'Theil_U2']
].sort_values('MAPE')

print("\n  --- Métriques sur PIB minier TOTAL annuel (2005–2024) ---")
print(VALIDATION_TOTAL.to_string(index=False,
                                 float_format=lambda x: f'{x:,.2f}'))

# Classement composite sur totaux
_RT = VALIDATION_TOTAL.copy()
_RT['rang_MAPE']    = _RT['MAPE'].rank(method='min')
_RT['rang_RMSE']    = _RT['RMSE'].rank(method='min')
_RT['rang_Pearson'] = (-_RT['Pearson']).rank(method='min')
_RT['rang_TheilU2'] = _RT['Theil_U2'].rank(method='min')
_RT['Rang_moyen']   = _RT[['rang_MAPE', 'rang_RMSE',
                            'rang_Pearson', 'rang_TheilU2']].mean(axis=1)
VALIDATION_TOTAL_CLASSEMENT = (_RT[['Modele', 'MAPE', 'RMSE', 'Pearson',
                                     'Theil_U2', 'Rang_moyen']]
                               .sort_values('Rang_moyen')
                               .reset_index(drop=True))
VALIDATION_TOTAL_CLASSEMENT.insert(
    0, 'Position', range(1, len(VALIDATION_TOTAL_CLASSEMENT) + 1)
)
print("\n  --- Classement composite sur totaux annuels ---")
print(VALIDATION_TOTAL_CLASSEMENT.to_string(
    index=False, float_format=lambda x: f'{x:,.2f}'))

# Comparaison des deux classements (cellule vs total)
_CMP_CLASS = (VALIDATION_CLASSEMENT[['Modele', 'Position']]
              .rename(columns={'Position': 'Rang_cellule'})
              .merge(
                  VALIDATION_TOTAL_CLASSEMENT[['Modele', 'Position']]
                  .rename(columns={'Position': 'Rang_total'}),
                  on='Modele', how='outer'))
_CMP_CLASS['Ecart_rang'] = (_CMP_CLASS['Rang_total']
                            - _CMP_CLASS['Rang_cellule'])
print("\n  --- Comparaison rangs : cellule (Mine × Année) vs total annuel ---")
print(_CMP_CLASS.to_string(index=False))
print("  (Ecart_rang > 0 → le modèle gagne au test total ce qu'il perd au "
      "test d'allocation cellule par cellule)")

# Ajout aux exports Excel
try:
    with pd.ExcelWriter(_path_val_xl, engine="openpyxl",
                        mode='a', if_sheet_exists='replace') as _xl:
        VALIDATION_TOTAL.to_excel(_xl, sheet_name='Total_annuel', index=False)
        VALIDATION_TOTAL_CLASSEMENT.to_excel(
            _xl, sheet_name='Classement_total', index=False
        )
        _CMP_CLASS.to_excel(_xl, sheet_name='Comparaison_rangs', index=False)
        _BASE_VAL_TOT.to_excel(_xl, sheet_name='Series_totaux', index=False)
    print(f"  ✔ Feuilles 'Total_annuel' / 'Classement_total' ajoutées : "
          f"{_path_val_xl}")
except Exception as _e:
    print(f"  ⚠ Ajout Excel échoué : {_e}")

# Table LaTeX dédiée pour le test total
_path_val_tex_tot = os.path.join(
    tab_dir, "tableau_validation_externe_StatCan_total.tex"
)
try:
    with open(_path_val_tex_tot, 'w', encoding='utf-8') as _f:
        _f.write("\\begin{table}[htbp]\n\\centering\n")
        _f.write("\\caption{Validation externe sur le PIB minier TOTAL annuel "
                 "(somme des 4 mines) vs Statistique Canada, 2005--2024.}\n")
        _f.write("\\label{tab:validation_externe_statcan_total}\n")
        _f.write("\\begin{tabular}{clrrrrr}\n\\hline\\hline\n")
        _f.write("Rang & Modèle & sMAPE\\,\\% & RMSE (M\\$) & "
                 "Corr.\\ Pearson & Theil U$_2$ & Rang moyen \\\\\n")
        _f.write("\\hline\n")
        for _, _r in VALIDATION_TOTAL_CLASSEMENT.iterrows():
            _f.write(
                f"{int(_r['Position'])} & {_r['Modele']} & "
                f"{_fmt_fr(_r['MAPE'], 1)} & {_fmt_fr(_r['RMSE'], 1)} & "
                f"{_fmt_fr(_r['Pearson'], 3)} & {_fmt_fr(_r['Theil_U2'], 3)} & "
                f"{_fmt_fr(_r['Rang_moyen'], 2)} \\\\\n"
            )
        _f.write("\\hline\\hline\n")
        _f.write("\\multicolumn{7}{l}{\\footnotesize \\textit{Note :} "
                 "sMAPE (symmetric MAPE) borné dans [0\\,;\\,200]\\,\\%. "
                 "Métriques calculées sur la somme annuelle des PIB des 4 "
                 "mines. Ce test est complémentaire de la validation cellule "
                 "par cellule (tableau \\ref{tab:validation_externe_statcan}) "
                 ": il évalue le niveau global du secteur minier, les erreurs "
                 "d'allocation entre mines pouvant se compenser.} \\\\\n")
        _f.write("\\end{tabular}\n\\end{table}\n")
    print(f"  ✔ LaTeX (total) exporté : {_path_val_tex_tot}")
except Exception as _e:
    print(f"  ⚠ Export LaTeX (total) échoué : {_e}")

# Figure : série temporelle des totaux + scatter modèle vs StatCan (totaux)
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        11,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

fig_tot, (ax_tot_ts, ax_tot_sc) = plt.subplots(
    1, 2, figsize=(15, 5.5), facecolor='white'
)

# (a) Série temporelle des totaux
_palette_mod = {
    'Emploi':        '#2C5F8A',
    'OLS (MCO)':     '#C0392B',
    'IV2SLS':        '#2C7A4B',
    'Chen':          '#7D3C98',
    'Score pondéré': '#D68910',
}
ax_tot_ts.plot(_BASE_VAL_TOT['Annee'], _BASE_VAL_TOT[_BENCHMARK],
               color='#000000', linewidth=2.6, marker='D', markersize=6,
               label='StatCan (benchmark)', zorder=10)
for _nom_m, _col_m in _MODELES_VAL.items():
    ax_tot_ts.plot(_BASE_VAL_TOT['Annee'], _BASE_VAL_TOT[_col_m],
                   color=_palette_mod.get(_nom_m, '#888888'),
                   linewidth=1.6, marker='o', markersize=4,
                   alpha=0.85, label=_nom_m, zorder=5)
ax_tot_ts.set_xlabel("Année", fontsize=11)
ax_tot_ts.set_ylabel("PIB minier TOTAL (M\\$ CAD 2017)", fontsize=11)
ax_tot_ts.set_title("Série temporelle des PIB totaux par modèle",
                    fontsize=12, fontweight='bold')
ax_tot_ts.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v:,.0f}')
)
ax_tot_ts.legend(fontsize=9, framealpha=0.92, edgecolor='#CCCCCC', ncol=2)
ax_tot_ts.spines['top'].set_visible(False)
ax_tot_ts.spines['right'].set_visible(False)
ax_tot_ts.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC')
ax_tot_ts.set_axisbelow(True)

# (b) Scatter totaux vs StatCan (tous modèles superposés)
_lim_tot = max(_BASE_VAL_TOT[_cols_a_sommer].max().max(), 1)
ax_tot_sc.plot([0, _lim_tot], [0, _lim_tot], color='#222222',
               linestyle='--', linewidth=1.2, label='1:1', zorder=2)
for _nom_m, _col_m in _MODELES_VAL.items():
    _df_t = _BASE_VAL_TOT[[_BENCHMARK, _col_m]].dropna()
    if _df_t.empty:
        continue
    ax_tot_sc.scatter(_df_t[_BENCHMARK], _df_t[_col_m],
                      color=_palette_mod.get(_nom_m, '#888888'),
                      s=42, alpha=0.85, edgecolor='white', linewidth=0.5,
                      label=_nom_m, zorder=4)
ax_tot_sc.set_xlabel("PIB total StatCan (M\\$ CAD)", fontsize=11)
ax_tot_sc.set_ylabel("PIB total modèle (M\\$ CAD)", fontsize=11)
ax_tot_sc.set_title("Totaux annuels : modèle vs StatCan",
                    fontsize=12, fontweight='bold')
ax_tot_sc.set_xlim(0, _lim_tot * 1.05)
ax_tot_sc.set_ylim(0, _lim_tot * 1.05)
ax_tot_sc.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v:,.0f}')
)
ax_tot_sc.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v:,.0f}')
)
ax_tot_sc.legend(fontsize=9, framealpha=0.92, edgecolor='#CCCCCC',
                 loc='upper left')
ax_tot_sc.spines['top'].set_visible(False)
ax_tot_sc.spines['right'].set_visible(False)
ax_tot_sc.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC')
ax_tot_sc.set_axisbelow(True)

plt.suptitle("Validation externe sur PIB minier TOTAL annuel (vs StatCan)",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
_fp_val_tot = os.path.join(fig_dir, "validation_externe_StatCan_total.png")
plt.savefig(_fp_val_tot, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Figure totaux sauvegardée : {_fp_val_tot}")

plt.rcdefaults()
print("\n  ✔ Validation externe vs StatCan terminée.")
print("=" * 78 + "\n")


#%% GRAPHIQUE VALIDATION DES DONNÉES
################################################################################


# ── Style académique global ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        12,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'axes.linewidth':   0.8,
    'axes.edgecolor':   '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'grid.color':       '#CCCCCC',
    'grid.linewidth':   0.6,
})

# ── Palette par mine ─────────────────────────────────────────────────────────
_couleurs_val = [
    [0.00, 0.45, 0.74],  # GPMB - bleu foncé
    [0.85, 0.33, 0.10],  # IPB  - orange foncé
    [0.93, 0.69, 0.13],  # GPHB - jaune doré
    [0.49, 0.18, 0.56],  # GPM  - violet foncé
]
_labels_val = ['Mine Meadowbank', 'Mine Baffinland', 'Mine Hope Bay', 'Mine Meliadine']

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 : Production des entreprises minières
# ══════════════════════════════════════════════════════════════════════════════
DATA_PROD = DONNEES_NUNAVUT[['GPMB_CAD_EN', 'IPB_CAD_EN', 'GPHB_CAD_EN', 'GPM_CAD_EN']].fillna(0).values
DATA_GRAPH_PIB = DATA_PIB_DN_2009_2024[['Annee', 'PIB_REEL', 'VAEMP_EN', 'VAOA_EN', 'VAF_EN', 'VAEM_EN', 'SAMP_EN']].fillna(0).values
years = DONNEES_NUNAVUT['Annee'].values
DATA_PROD = np.array(DATA_PROD)

fig, ax = plt.subplots(figsize=(10, 6))

bottom = np.zeros(len(years))
for i in range(4):
    ax.fill_between(years, bottom, bottom + DATA_PROD[:, i],
                    color=_couleurs_val[i], label=_labels_val[i], alpha=0.8)
    bottom += DATA_PROD[:, i]

ax.set_xlabel('Année', fontweight='bold', fontsize=12, labelpad=5)
ax.set_ylabel('Production en million $ CAD', fontweight='bold', fontsize=12, labelpad=5)
#ax.set_title('Production des entreprises minières au Nunavut',
#             fontsize=13, fontweight='bold', pad=10, fontfamily='serif')

ax.set_xticks(years)
ax.set_xticklabels([str(int(y)) for y in years], rotation=45, ha='right')
ax.set_xlim([2010, 2024])

ax.legend(loc='best', fontsize=10, framealpha=0.92, edgecolor='#CCCCCC')
ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#AAAAAA', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.7)
ax.spines['bottom'].set_linewidth(0.7)

plt.tight_layout()
_fig_path = os.path.join(fig_dir, "validation_production_mines.png")
plt.savefig(_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Figure sauvegardée : {_fig_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2b : Emplois par mine (barres groupées)
# ══════════════════════════════════════════════════════════════════════════════
_mines_empl     = ['Mine_Meadowbank', 'Mine_Baffinland', 'Mine_Hope_Bay', 'Mine_Meliadine']
_labels_empl    = ['Mine Meadowbank', 'Mine Baffinland', 'Mine Hope Bay', 'Mine Meliadine']
_annees_empl    = TABLE_EMPLOI['Annee'].values
_data_empl      = TABLE_EMPLOI[_mines_empl].fillna(0).values   # shape : (n_annees, 4)

fig, ax = plt.subplots(figsize=(11, 6))

_n_mines   = len(_mines_empl)
_x         = np.arange(len(_annees_empl))
_bar_width = 0.8 / _n_mines

for i in range(_n_mines):
    ax.bar(_x + i * _bar_width - 0.4 + _bar_width / 2,
           _data_empl[:, i],
           width=_bar_width,
           color=_couleurs_val[i],
           edgecolor='white',
           linewidth=0.5,
           label=_labels_empl[i])

ax.set_xlabel('Année', fontweight='bold', fontsize=12, labelpad=5)
ax.set_ylabel("Nombre d'emplois", fontweight='bold', fontsize=12, labelpad=5)
#ax.set_title('Emplois par mine au Nunavut (2009–2022)',
#             fontsize=13, fontweight='bold', pad=10, fontfamily='serif')

ax.set_xticks(_x)
ax.set_xticklabels([str(int(a)) for a in _annees_empl], rotation=45, ha='right')

ax.legend(loc='best', fontsize=10, framealpha=0.92, edgecolor='#CCCCCC')
ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#AAAAAA', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.7)
ax.spines['bottom'].set_linewidth(0.7)

plt.tight_layout()
_fig_path = os.path.join(fig_dir, "validation_emploi_mines.png")
plt.savefig(_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Figure sauvegardée : {_fig_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 : Croissance de la Valeur Ajoutée du Secteur Minier
# ══════════════════════════════════════════════════════════════════════════════
_couleurs_recess = [
    [0.00, 0.45, 0.74],  # Bleu
    [0.85, 0.33, 0.10],  # Orange
    [0.93, 0.69, 0.13],  # Or
    [0.49, 0.18, 0.56],  # Violet
    [1.0,  1.0,  0.6],   # Jaune
]
_annees_recession = [2010, 2014, 2017, 2019, 2021]
_largeur = 0.5
_noms_recession = [
    'Ouverture Meadowbank',
    'Ouverture Baffinland',
    'Ouverture Hope Bay',
    'Ouverture Meliadine',
    'Fermeture Hope Bay',
]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(years, DONNEES_NUNAVUT['Diff_Log_PIB_EN'].values,
        'k', linewidth=1.5, label='PIB réel (Nunavut)')

ax.plot(years, DONNEES_NUNAVUT['Diff_Log_VAEMP_EN'].values,
        color='#C0392B', linewidth=1.5, linestyle='--',
        label='Valeur ajoutée minier')

# Bandes colorées aux années charnières (limites Y avant tracé des bandes)
yl = ax.get_ylim()
for i, annee in enumerate(_annees_recession):
    x1 = annee - _largeur / 2
    x2 = annee + _largeur / 2
    ax.fill_between([x1, x2], yl[0], yl[1],
                    color=_couleurs_recess[i], alpha=0.5, label=_noms_recession[i])

ax.set_xlim([2005, 2024])
ax.set_xticks(range(2005, 2024, 2))
ax.set_ylim([-0.5, 2.5])

ax.set_xlabel('Année', fontweight='bold', fontsize=12, labelpad=5)
ax.set_ylabel('Taux de croissance (%)', fontweight='bold', fontsize=12, labelpad=5)
#ax.set_title('Croissance de la Valeur Ajoutée du Secteur Minier',
#             fontsize=13, fontweight='bold', pad=10, fontfamily='serif')

# Légende sans doublons
_handles, _lbls = ax.get_legend_handles_labels()
_by_label = dict(zip(_lbls, _handles))
ax.legend(_by_label.values(), _by_label.keys(), loc='best', fontsize=10,
          framealpha=0.92, edgecolor='#CCCCCC')

ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#AAAAAA', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.7)
ax.spines['bottom'].set_linewidth(0.7)

plt.tight_layout()
_fig_path = os.path.join(fig_dir, "validation_croissance_VA_miniere.png")
plt.savefig(_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Figure sauvegardée : {_fig_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 : PIB en fonction de la Luminosité nocturne (DN) — nuage de points 2005-2024
# ══════════════════════════════════════════════════════════════════════════════
_df_piddn = (DN_DONNEES_NUNAVUT_2005_2024[['Annee', 'PIB_EN', 'VIIRS_SUM_RESAMPL']]
             .dropna().sort_values('Annee'))
_annees_pd = _df_piddn['Annee'].values
_pib_pd    = _df_piddn['PIB_EN'].values
_dn_pd     = _df_piddn['VIIRS_SUM_RESAMPL'].values

# Corrélation + régression linéaire simple (pour la droite d'ajustement)
_corr_pd = np.corrcoef(_dn_pd, _pib_pd)[0, 1]
_slope, _intercept = np.polyfit(_dn_pd, _pib_pd, 1)
_x_fit = np.linspace(_dn_pd.min(), _dn_pd.max(), 100)
_y_fit = _slope * _x_fit + _intercept

fig, ax = plt.subplots(figsize=(10, 6))

# Nuage de points coloré par année
_sc = ax.scatter(_dn_pd, _pib_pd, c=_annees_pd, cmap='viridis',
                 s=70, edgecolor='white', linewidth=0.8, zorder=3)

# Droite d'ajustement
ax.plot(_x_fit, _y_fit, color='#C0392B', linewidth=1.8, linestyle='--',
        label=f"Ajustement linéaire (pente = {_slope:.2f})", zorder=2)

# Étiquettes d'année pour chaque point
for _x, _y, _a in zip(_dn_pd, _pib_pd, _annees_pd):
    ax.annotate(str(int(_a)), (_x, _y),
                xytext=(5, 4), textcoords='offset points',
                fontsize=8, color='#555555')

# Colorbar pour l'année
_cbar = fig.colorbar(_sc, ax=ax, shrink=0.85, pad=0.02)
_cbar.set_label('Année', fontsize=10)
_cbar.ax.tick_params(labelsize=9)

ax.set_xlabel('Luminosité nocturne (VIIRS, somme)',
              fontweight='bold', fontsize=12, labelpad=5)
ax.set_ylabel('PIB réel (M$ enchaînés 2017)',
              fontweight='bold', fontsize=12, labelpad=5)
#ax.set_title('Figure 4 : PIB réel en fonction de la luminosité nocturne (2005–2024)',
#             fontsize=13, fontweight='bold', pad=10, fontfamily='serif')

# Annotation corrélation
ax.text(0.03, 0.97, f"Corrélation (Pearson) = {_corr_pd:.3f}\nN = {len(_pib_pd)}",
        transform=ax.transAxes, ha='left', va='top',
        fontsize=10, fontstyle='italic',
        bbox=dict(facecolor='white', edgecolor='#CCCCCC', alpha=0.85))

ax.legend(loc='lower right', fontsize=10, framealpha=0.92, edgecolor='#CCCCCC')
ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#AAAAAA', zorder=0)
ax.xaxis.grid(True, linestyle=':', linewidth=0.6, color='#AAAAAA', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.7)
ax.spines['bottom'].set_linewidth(0.7)

plt.tight_layout()
_fig_path = os.path.join(fig_dir, "validation_pib_dn.png")
plt.savefig(_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Figure sauvegardée : {_fig_path}")

# ── Réinitialiser les paramètres matplotlib par défaut ───────────────────────
plt.rcdefaults()
print("\n  ✔ Tous les graphiques de validation ont été générés (style mémoire).")


#%% TABLEAUX DE CORRÉLATION — NIVEAUX · LOGARITHMES · TAUX DE CROISSANCE
################################################################################
# Variables : PIB_EN, VAEMP_EN, SAM_EN, VAF_EN, VAOA_EN, PROD_MINE_CAD_EN,
#             Luminosité VIIRS (Nunavut), GPMB_CAD_EN, GPHB_CAD_EN,
#             GPM_CAD_EN, IPB_CAD_EN
# Trois matrices : niveaux (dollars enchaînés 2017), logarithmes, Δlog
################################################################################

# ── 2. Fusion données économiques + luminosité + emploi + couverture nuageuse ─
_lum = (
    DMSP_VIIRS_2000_2024_NUNAVUT[['Annee',
                                   'VIIRS_SUM_RESAMPL',
                                   'Log_VIIRS_SUM_RESAMPL',
                                   'Diff_Log_VIIRS_SUM_RESAMPL']]
    .copy()
    .rename(columns={
        'VIIRS_SUM_RESAMPL':          'LUM',
        'Log_VIIRS_SUM_RESAMPL':      'Log_LUM',
        'Diff_Log_VIIRS_SUM_RESAMPL': 'Diff_Log_LUM',
    })
)

# Emploi total Nunavut (POPULATION)
_pop = (
    POPULATION[['Annee', 'EMPL', 'Log_EMPL', 'Diff_Log_EMPL']]
    .copy()
)

# Couverture nuageuse annuelle (cover_total)
_cov = cover_total[['Annee', 'cloud_cover']].copy()
_cov['Log_cloud_cover']      = np.log(_cov['cloud_cover'].clip(lower=1e-10))
_cov['Diff_Log_cloud_cover'] = _cov['Log_cloud_cover'].diff()

_cols_eco_niv = ['Annee',
                 'PIB_EN', 'VAEMP_EN', 'SAMP_EN', 'VAF_EN', 'VAOA_EN',
                 'PROD_MINE_CAD_EN']

_cols_eco_log = ['Annee',
                 'Log_PIB_EN', 'Log_VAEMP_EN', 'Log_SAMP_EN', 'Log_VAF_EN',
                 'Log_VAOA_EN', 'Log_PROD_MINE_CAD_EN']

_cols_eco_tx  = ['Annee',
                 'Diff_Log_PIB_EN', 'Diff_Log_VAEMP_EN', 'Diff_Log_SAMP_EN',
                 'Diff_Log_VAF_EN', 'Diff_Log_VAOA_EN', 'Diff_Log_PROD_MINE_CAD_EN']

_eco_2005_2024 = DONNEES_NUNAVUT[
    (DONNEES_NUNAVUT['Annee'] >= 2005) & (DONNEES_NUNAVUT['Annee'] <= 2024)
]
_lum_2005_2024 = _lum[
    (_lum['Annee'] >= 2005) & (_lum['Annee'] <= 2024)
]
_pop_2005_2024 = _pop[
    (_pop['Annee'] >= 2005) & (_pop['Annee'] <= 2024)
]
_cov_2005_2024 = _cov[
    (_cov['Annee'] >= 2005) & (_cov['Annee'] <= 2024)
]

# Niveaux : éco + emploi + luminosité + couverture nuageuse
_df_niv = (
    pd.merge(_eco_2005_2024[_cols_eco_niv], _pop_2005_2024[['Annee', 'EMPL']],
             on='Annee', how='left')
    .merge(_lum_2005_2024[['Annee', 'LUM']], on='Annee', how='inner')
    .merge(_cov_2005_2024[['Annee', 'cloud_cover']], on='Annee', how='left')
    .drop(columns='Annee')
)

# Logarithmes : éco + emploi + luminosité + couverture nuageuse
_df_log = (
    pd.merge(_eco_2005_2024[_cols_eco_log], _pop_2005_2024[['Annee', 'Log_EMPL']],
             on='Annee', how='left')
    .merge(_lum_2005_2024[['Annee', 'Log_LUM']], on='Annee', how='inner')
    .merge(_cov_2005_2024[['Annee', 'Log_cloud_cover']], on='Annee', how='left')
    .drop(columns='Annee')
)

# Taux de croissance : éco + emploi + luminosité + couverture nuageuse
_df_tx = (
    pd.merge(_eco_2005_2024[_cols_eco_tx], _pop_2005_2024[['Annee', 'Diff_Log_EMPL']],
             on='Annee', how='left')
    .merge(_lum_2005_2024[['Annee', 'Diff_Log_LUM']], on='Annee', how='inner')
    .merge(_cov_2005_2024[['Annee', 'Diff_Log_cloud_cover']], on='Annee', how='left')
    .dropna()
    .drop(columns='Annee')
)

# ── 3. Étiquettes courtes ───────────────────────────────────────────────────
_labels_corr = ['PIB', 'VAEMP', 'SAM', 'VAF', 'VAOA',
                 'Prod.\nminière', 'EMPL', 'Lumin.', 'Couv.\nnuageuse']

# ── 4. Matrices de corrélation ──────────────────────────────────────────────
_corr_niv = _df_niv.corr()
_corr_log = _df_log.corr()
_corr_tx  = _df_tx.corr()

for _mat in (_corr_niv, _corr_log, _corr_tx):
    _mat.index   = _labels_corr
    _mat.columns = _labels_corr

# ── 4. Style académique ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'serif',
    'font.serif':     ['Times New Roman', 'DejaVu Serif'],
    'font.size':      10,
    'axes.titlesize': 12,
    'figure.dpi':     150,
})

def _heatmap_corr(corr_mat, titre, fichier):
    """Heatmap de corrélation style mémoire — triangle inférieur + diagonale."""
    n = len(corr_mat)
    _vals = corr_mat.values.copy()

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(_vals, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.025)
    cbar.set_label('Coefficient de corrélation de Pearson', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])

    # Ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(_labels_corr, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(_labels_corr, fontsize=9)
    ax.tick_params(length=0)

    # Valeurs dans chaque cellule
    for i in range(n):
        for j in range(n):
            val = _vals[i, j]
            txt_color = 'white' if abs(val) >= 0.75 else '#222222'
            ax.text(j, i, f'{val:.2f}',
                    ha='center', va='center', fontsize=7.5,
                    color=txt_color, fontfamily='serif')

    # Lignes de grille légères
    for k in np.arange(-0.5, n, 1):
        ax.axhline(k, color='white', linewidth=0.6)
        ax.axvline(k, color='white', linewidth=0.6)

#    ax.set_title(titre, fontsize=13, fontweight='bold',
#                 pad=12, fontfamily='serif')

    ax.spines[:].set_visible(False)

    plt.tight_layout()
    _p = os.path.join(fig_dir, fichier)
    plt.savefig(_p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✔ Figure sauvegardée : {_p}")

# ── 5. Génération des 3 figures ─────────────────────────────────────────────
_heatmap_corr(
    _corr_niv,
    'Tableau de corrélation — Niveaux (dollars enchaînés 2017)',
    'corr_niveaux.png'
)

_heatmap_corr(
    _corr_log,
    'Tableau de corrélation — Logarithmes',
    'corr_logarithmes.png'
)

_heatmap_corr(
    _corr_tx,
    'Tableau de corrélation — Taux de croissance (Δ log)',
    'corr_taux_croissance.png'
)

# ── 6. Export Excel (3 feuilles) ─────────────────────────────────────────────
_xl_path = os.path.join(tab_dir, 'tableau_correlations.xlsx')
with pd.ExcelWriter(_xl_path, engine='openpyxl') as _writer:
    _corr_niv.round(3).to_excel(_writer, sheet_name='Niveaux')
    _corr_log.round(3).to_excel(_writer, sheet_name='Logarithmes')
    _corr_tx.round(3).to_excel(_writer, sheet_name='Taux_de_croissance')
print(f"  ✔ Tableau Excel sauvegardé : {_xl_path}")

# ── 7. Export LaTeX (3 fichiers .tex) ────────────────────────────────────────
def _corr_to_latex(corr_mat, titre, caption, label, fichier):
    """
    Génère un fichier .tex avec un tableau de corrélation formaté
    pour un mémoire académique (booktabs, triangle inférieur).
    """
    n = len(corr_mat)
    cols = corr_mat.columns.tolist()

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{' + caption + r'}')
    lines.append(r'  \label{' + label + r'}')
    lines.append(r'  \small')
    # Colonnes : 1 col label + n cols numériques
    col_fmt = 'l' + 'r' * n
    lines.append(r'  \begin{tabular}{' + col_fmt + r'}')
    lines.append(r'    \toprule')

    # En-tête
    header_labels = [c.replace('\n', ' ') for c in cols]
    lines.append('    & ' + ' & '.join(header_labels) + r' \\')
    lines.append(r'    \midrule')

    # Lignes de données (triangle inférieur + diagonale)
    for i, row_label in enumerate(cols):
        row_label_clean = row_label.replace('\n', ' ')
        cells = []
        for j in range(n):
            if j > i:
                cells.append('')           # triangle supérieur vide
            else:
                val = corr_mat.values[i, j]
                # Mise en gras si |r| >= 0.70 (hors diagonale)
                _v_fmt = _fmt_fr(val, 3)
                if i != j and abs(val) >= 0.70:
                    cells.append(r'\textbf{' + _v_fmt + r'}')
                else:
                    cells.append(_v_fmt)
        lines.append('    ' + row_label_clean + ' & ' + ' & '.join(cells) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  \begin{minipage}{\linewidth}')
    lines.append(r'    \vspace{2pt}')
    lines.append(r'    \footnotesize \textit{Note :} Les coefficients de corrélation '
                 r'de Pearson sont calculés sur la période 2005--2024. '
                 r'Les valeurs en \textbf{gras} indiquent $|r| \geq 0{,}70$.')
    lines.append(r'  \end{minipage}')
    lines.append(r'\end{table}')

    _tex = '\n'.join(lines)
    _path = os.path.join(fig_dir, fichier)
    with open(_path, 'w', encoding='utf-8') as _f:
        _f.write(_tex)
    print(f"  ✔ LaTeX sauvegardé : {_path}")

_corr_to_latex(
    _corr_niv,
    titre='Corrélation — Niveaux',
    caption='Matrice de corrélation de Pearson --- Niveaux (dollars enchaînés 2017, 2005--2024)',
    label='tab:corr_niveaux',
    fichier='corr_niveaux.tex'
)

_corr_to_latex(
    _corr_log,
    titre='Corrélation — Logarithmes',
    caption='Matrice de corrélation de Pearson --- Logarithmes (2005--2024)',
    label='tab:corr_logarithmes',
    fichier='corr_logarithmes.tex'
)

_corr_to_latex(
    _corr_tx,
    titre='Corrélation — Taux de croissance',
    caption='Matrice de corrélation de Pearson --- Taux de croissance ($\Delta \log$, 2005--2024)',
    label='tab:corr_taux_croissance',
    fichier='corr_taux_croissance.tex'
)

plt.rcdefaults()
print("\n  ✔ Tableaux de corrélation générés (niveaux, log, taux de croissance).")


#%% ROBUSTESSE — ANALYSE D'INFLUENCE (Leave-one-out) + RÉGRESSION ROBUSTE (Huber)
###############################################################################
# 1) Influence par observation (Cook's distance, DFFITS, leverage)
# 2) Régression robuste M-estimator (Huber) pour comparaison avec MCO
###############################################################################

from statsmodels.stats.outliers_influence import OLSInfluence

# ─────────────────────────────────────────────────────────────────────────────
# 1) ANALYSE D'INFLUENCE LEAVE-ONE-OUT
# ─────────────────────────────────────────────────────────────────────────────
_inf = OLSInfluence(MODEL_LOG_PIB_DN_PH)

# Seuils usuels
_n_obs_inf = int(MODEL_LOG_PIB_DN_PH.nobs)
_k_inf     = MODEL_LOG_PIB_DN_PH.df_model + 1   # nb régresseurs incluant const
_thr_cook    = 4.0 / _n_obs_inf                 # seuil Cook's distance
_thr_dffits  = 2.0 * np.sqrt(_k_inf / _n_obs_inf)
_thr_lev     = 2.0 * _k_inf / _n_obs_inf        # leverage > 2k/n = point de levier

# Rattacher les années aux indices utilisés dans la régression
_annees_infl = DN_DONNEES_NUNAVUT_2005_2024.loc[
    DN_DONNEES_NUNAVUT_2005_2024.index.isin(data_model.index), "Annee"
].values

INFLUENCE_DIAG = pd.DataFrame({
    "Annee"   : _annees_infl,
    "Cook_d"  : _inf.cooks_distance[0],
    "DFFITS"  : _inf.dffits[0],
    "Leverage": _inf.hat_matrix_diag,
    "Residu_std": _inf.resid_studentized_internal,
})
INFLUENCE_DIAG["Flag_Cook"]    = INFLUENCE_DIAG["Cook_d"]    > _thr_cook
INFLUENCE_DIAG["Flag_DFFITS"]  = INFLUENCE_DIAG["DFFITS"].abs() > _thr_dffits
INFLUENCE_DIAG["Flag_Leverage"] = INFLUENCE_DIAG["Leverage"] > _thr_lev
INFLUENCE_DIAG["N_Flags"]      = (INFLUENCE_DIAG[["Flag_Cook", "Flag_DFFITS", "Flag_Leverage"]]
                                  .sum(axis=1))

print("\n" + "="*80)
print("ANALYSE D'INFLUENCE — MODEL_LOG_PIB_DN_PH (Leave-one-out)")
print("="*80)
print(f"Seuils : Cook_d > {_thr_cook:.4f}  |  |DFFITS| > {_thr_dffits:.4f}  "
      f"|  Leverage > {_thr_lev:.4f}")
print(INFLUENCE_DIAG.round(4).to_string(index=False))
print("\nAnnées flagguées (≥ 1 critère) :")
_flagged = INFLUENCE_DIAG[INFLUENCE_DIAG["N_Flags"] >= 1][["Annee", "N_Flags"]]
print(_flagged.to_string(index=False))

# Graphique Cook's distance par année
fig, ax = plt.subplots(figsize=(10, 5))
_bars = ax.bar(INFLUENCE_DIAG["Annee"].astype(int),
               INFLUENCE_DIAG["Cook_d"],
               color=["#C0392B" if _f else "#1f4e79"
                      for _f in INFLUENCE_DIAG["Flag_Cook"]],
               edgecolor='white', linewidth=0.5)
ax.axhline(_thr_cook, color='gray', linestyle='--', linewidth=1,
           label=f"Seuil 4/N = {_thr_cook:.3f}")
ax.set_xlabel("Année", fontweight='bold', fontsize=12)
ax.set_ylabel("Cook's distance", fontweight='bold', fontsize=12)
#ax.set_title("Figure — Analyse d'influence : Cook's distance par année",
#             fontsize=13, fontweight='bold', pad=10, fontfamily='serif')
ax.set_xticks(INFLUENCE_DIAG["Annee"].astype(int))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend(loc='best', fontsize=10)
ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#AAAAAA')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
_fig_inf = os.path.join(fig_dir, "influence_cook_distance.png")
plt.savefig(_fig_inf, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✔ Figure influence sauvegardée : {_fig_inf}")

# Export tableau influence
_path_inf_xlsx = os.path.join(tab_dir, "analyse_influence.xlsx")
INFLUENCE_DIAG.round(4).to_excel(_path_inf_xlsx, index=False)
print(f"Tableau influence enregistré : {_path_inf_xlsx}")

_path_inf_tex = os.path.join(tab_dir, "analyse_influence.tex")
_latex_inf = INFLUENCE_DIAG.round(4).to_latex(
    index=False, escape=True,
    caption="Analyse d'influence (Leave-one-out) -- MODEL\\_LOG\\_PIB\\_DN\\_PH",
    label="tab:influence",
    column_format="l" + "r" * (INFLUENCE_DIAG.shape[1] - 1),
)
_latex_inf = _latex_inf.replace(
    r'\begin{tabular}',
    r'\footnotesize' + '\n' + r'\begin{tabular}'
)
_latex_inf = _latex_inf.replace(
    r'\end{tabular}',
    r'\end{tabular}'
)
with open(_path_inf_tex, "w", encoding="utf-8") as _f:
    _f.write(_latex_inf)
print(f"LaTeX influence enregistré : {_path_inf_tex}")


# ─────────────────────────────────────────────────────────────────────────────
# 2) RÉGRESSION ROBUSTE M-ESTIMATOR (Huber)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_LOG_PIB_DN_PH_HUBER = sm.RLM(
    y_clean, X_clean,
    M=sm.robust.norms.HuberT()
).fit()

print("\n" + "="*80)
print("RÉGRESSION ROBUSTE (Huber M-estimator) — MODEL_LOG_PIB_DN_PH")
print("="*80)
print(MODEL_LOG_PIB_DN_PH_HUBER.summary())

# Tableau comparatif MCO vs Huber
stargazer_rob = Stargazer([MODEL_LOG_PIB_DN_PH, MODEL_LOG_PIB_DN_PH_HUBER])
stargazer_rob.title("Robustesse -- MCO (HAC) et régression robuste Huber")
stargazer_rob.custom_columns(["MCO (HAC)", "Huber robuste"], [1, 1])
stargazer_rob.rename_covariates({
    "const": "Const",
    "Log_VIIRS_SUM_RESAMPL": "Log(DN)",
    "n_mines_exploration": "Mines expl.",
    "n_mines_production":  "Mines prod.",
})
stargazer_rob.covariate_order([
    "const", "Log_VIIRS_SUM_RESAMPL",
    "n_mines_exploration", "n_mines_production",
])
stargazer_rob.dependent_variable_name("Log(PIB)")
stargazer_rob.add_custom_notes([
    "Colonne (1) : MCO avec erreurs-types HAC (Newey-West, 2 lags).",
    "Colonne (2) : Régression robuste M-estimator de Huber (moins sensible aux outliers).",
    "La régression Huber pondère automatiquement à la baisse les observations",
    "à fort résidu (notamment 2009, flaggée par l'analyse d'influence).",
    "Mines expl. / Mines prod. : nombre de mines en exploration / production.",
])

_path_rob_html = os.path.join(tab_dir, "robustesse_huber.html")
with open(_path_rob_html, "w", encoding="utf-8") as f:
    f.write(stargazer_rob.render_html())
print(f"HTML robustesse Huber : {_path_rob_html}")

_path_rob_tex = os.path.join(tab_dir, "robustesse_huber.tex")
_latex_rob = stargazer_rob.render_latex()
_latex_rob = _latex_rob.replace(
    r'\begin{tabular}',
    r'\footnotesize' + '\n' + r'\begin{tabular}'
)
_latex_rob = _latex_rob.replace(
    r'\end{tabular}',
    r'\end{tabular}'
)
# Injection du label pour permettre les renvois \ref{} dans Overleaf
_latex_rob = _latex_rob.replace(
    r'\end{table}',
    r'  \label{tab:robustesse_huber}' + '\n' + r'\end{table}'
)
with open(_path_rob_tex, "w", encoding="utf-8") as f:
    f.write(_latex_rob)
print(f"LaTeX robustesse Huber : {_path_rob_tex}")

#%% ROBUSTESSE — EXCLUSION DE L'OBSERVATION INFLUENTE (2009)
###############################################################################
# Re-estimation de MODEL_LOG_PIB_DN_PH sans 2009 (observation flaggée par
# l'analyse d'influence — Cook's distance et DFFITS au-dessus des seuils) pour :
#   - vérifier la stabilité des coefficients
#   - tester si le RESET passe sans cette observation atypique
###############################################################################

# Reconstruire y et X à partir du modèle original, puis filtrer sur l'index temporel
_annees_plein = DN_DONNEES_NUNAVUT_2005_2024.loc[
    DN_DONNEES_NUNAVUT_2005_2024.index.isin(data_model.index), "Annee"
]
_mask_no_outlier = ~_annees_plein.isin([2009])
_y_rob = y_clean.loc[_mask_no_outlier.index[_mask_no_outlier]]
_X_rob = X_clean.loc[_mask_no_outlier.index[_mask_no_outlier]]

MODEL_LOG_PIB_DN_PH_ROB = sm.OLS(_y_rob, _X_rob).fit(
    cov_type="HAC", cov_kwds={"maxlags": 2}
)

print("\n" + "="*80)
print("ROBUSTESSE — MCO (Log-Log, phases) SANS 2009")
print("="*80)
print(MODEL_LOG_PIB_DN_PH_ROB.summary())
print(f"N plein échantillon : {int(MODEL_LOG_PIB_DN_PH.nobs)}  |  "
      f"N sans 2009 : {int(MODEL_LOG_PIB_DN_PH_ROB.nobs)}")

# RESET de Ramsey sur le modèle sans 2009
try:
    _reset_rob = linear_reset(MODEL_LOG_PIB_DN_PH_ROB, power=2, use_f=True)
    print(f"\nRESET Ramsey (sans 2009) : F = {_reset_rob.statistic:.4f}  "
          f"p-valeur = {_reset_rob.pvalue:.4f}")
    _dec = "Rejet H0 (spécification douteuse)" if _reset_rob.pvalue < 0.05 \
           else "Non-rejet H0 (spécification OK)"
    print(f"Décision : {_dec}")
except Exception as _e:
    print(f"RESET (sans 2009) : échec ({_e})")

# Tableau stargazer — plein vs sans 2009
stargazer_rob = Stargazer([MODEL_LOG_PIB_DN_PH, MODEL_LOG_PIB_DN_PH_ROB])
stargazer_rob.title("Robustesse -- MCO avec et sans l'observation influente (2009)")
stargazer_rob.custom_columns(
    ["Plein échantillon", "Sans 2009"],
    [1, 1]
)
stargazer_rob.rename_covariates({
    "const": "Const",
    "Log_VIIRS_SUM_RESAMPL": "Log(DN)",
    "n_mines_exploration": "Mines expl.",
    "n_mines_production":  "Mines prod.",
})
stargazer_rob.covariate_order([
    "const", "Log_VIIRS_SUM_RESAMPL",
    "n_mines_exploration", "n_mines_production",
])
stargazer_rob.dependent_variable_name("Log(PIB)")
stargazer_rob.add_custom_notes([
    "Mines expl. : nombre de mines en exploration.",
    "Mines prod. : nombre de mines en production.",
    "DN : données de luminosité.",
    "Colonne (2) : estimation en excluant 2009.",
])

# Export HTML
_path_rob_html = os.path.join(tab_dir, "robustesse_modeles.html")
with open(_path_rob_html, "w", encoding="utf-8") as f:
    f.write(stargazer_rob.render_html())
print(f"HTML robustesse enregistré : {_path_rob_html}")

# Export LaTeX ajusté en \footnotesize
_path_rob_tex = os.path.join(tab_dir, "robustesse_modeles.tex")
_latex_rob = stargazer_rob.render_latex()
_latex_rob = _latex_rob.replace(
    r'\begin{tabular}',
    r'\footnotesize' + '\n' + r'\begin{tabular}'
)
_latex_rob = _latex_rob.replace(
    r'\end{tabular}',
    r'\end{tabular}'
)
# Injection du label pour permettre les renvois \ref{} dans Overleaf
_latex_rob = _latex_rob.replace(
    r'\end{table}',
    r'  \label{tab:robustesse_modeles}' + '\n' + r'\end{table}'
)
with open(_path_rob_tex, "w", encoding="utf-8") as f:
    f.write(_latex_rob)
print(f"LaTeX robustesse enregistré : {_path_rob_tex}")