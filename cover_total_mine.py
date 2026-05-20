# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:38:36 2026

@author: admin123
"""

# =============================================================================
# TÉLÉCHARGEMENT CLOUD COVER ANNUEL — NUNAVUT 2000–2025
# ERA5 Monthly Means via CDS API (Copernicus)
# =============================================================================

import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import os

# =============================================================================
# 0. Répertoire de travail
# =============================================================================

BASE_DIR = r"C:\Users\admin123\Desktop\Maitrise"
os.makedirs(BASE_DIR, exist_ok=True)
os.chdir(BASE_DIR)
print(f"Répertoire de travail : {os.getcwd()}")

# =============================================================================
# 0. Configuration API
# =============================================================================

# Créer le fichier ~/.cdsapirc automatiquement
cdsapirc_content = """url: https://cds.climate.copernicus.eu/api
key: 0123dbcd-65ea-4832-9046-9bb27849cdc9
"""
cdsapirc_path = Path.home() / ".cdsapirc"
cdsapirc_path.write_text(cdsapirc_content)
print(f"Fichier .cdsapirc écrit dans : {cdsapirc_path}")

# =============================================================================
# 1. Paramètres géographiques — Nunavut
# =============================================================================

# Bounding box : Nord, Ouest, Sud, Est
# Élargi à -110°W pour inclure Hope Bay (~106.6°W) — sinon Hope Bay tombe
# hors grille et sel(method="nearest") renvoie silencieusement la valeur du
# bord est, biaisant l'instrument composite (section 7).
BBOX_NUNAVUT = [75, -110, 60, -75]  # latitude max/min, longitude min/max

# Années disponibles (ERA5 a un délai ~3 mois → 2025 partiel)
ANNEES = list(range(2000, 2025))
MOIS   = [f"{m:02d}" for m in range(1, 13)]

os.makedirs(os.path.join(BASE_DIR, "donnees_meteo"), exist_ok=True)

OUTPUT_NC  = os.path.join(BASE_DIR, "donnees_meteo", "cloud_cover_nunavut_raw.nc")
OUTPUT_CSV = os.path.join(BASE_DIR, "donnees_meteo","cloud_cover_nunavut_annuel.csv")


# =============================================================================
# 2. Téléchargement ERA5 — total_cloud_cover mensuel
# =============================================================================

c = cdsapi.Client()

print("Téléchargement en cours (ERA5 monthly means — total_cloud_cover)...")
print("Cela peut prendre quelques minutes selon la taille de la requête.\n")

c.retrieve(
    "reanalysis-era5-single-levels-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable":     "total_cloud_cover",   # fraction 0 (ciel clair) à 1 (couvert)
        "year":         [str(a) for a in ANNEES],
        "month":        MOIS,
        "time":         "00:00",
        "area":         BBOX_NUNAVUT,
        "format":       "netcdf",
    },
    OUTPUT_NC
)

print(f"\nFichier NetCDF sauvegardé : {OUTPUT_NC}")

# =============================================================================
# 3. Traitement — moyenne spatiale Nunavut + agrégation annuelle
# =============================================================================

ds = xr.open_dataset(OUTPUT_NC)

# Identifier la variable (peut s'appeler "tcc" ou "TCC" selon la version)
var_name = [v for v in ds.data_vars if "cc" in v.lower() or "cloud" in v.lower()][0]
print(f"Variable détectée dans le fichier NetCDF : '{var_name}'")

cloud = ds[var_name]

# Moyenne spatiale pondérée par le cosinus de la latitude (pondération correcte)
weights    = np.cos(np.deg2rad(cloud.latitude))
cloud_wmean = cloud.weighted(weights).mean(dim=["latitude", "longitude"])

# Agrégation annuelle : moyenne des 12 mois
cloud_annuel = cloud_wmean.resample(valid_time="1YE").mean()

# Construire le DataFrame final
df_cloud = pd.DataFrame({
    "Annee":       cloud_annuel.valid_time.dt.year.values,
    "cloud_cover": cloud_annuel.values.round(4)
})

# Supprimer les années sans données complètes (ex : 2025 partiel → NaN possible)
df_cloud = df_cloud.dropna(subset=["cloud_cover"])

print("\n=== Aperçu des données ===")
print(df_cloud.to_string(index=False))

# =============================================================================
# 4. Sauvegarde CSV
# =============================================================================

df_cloud.to_csv(OUTPUT_CSV, index=False)
print(f"\nCSV sauvegardé : {OUTPUT_CSV}")
print(f"Période couverte : {df_cloud['Annee'].min()} – {df_cloud['Annee'].max()}")
print(f"Nombre d'années : {len(df_cloud)}")

# =============================================================================
# 5. Fusion rapide avec ta base principale (exemple)
# =============================================================================

# df = df.merge(df_cloud, on="Annee", how="left")
# → cloud_cover est désormais dispo comme instrument dans ton IV-2SLS

print("\nPour fusionner avec ta base principale :")
print("  df = df.merge(df_cloud, on='Annee', how='left')")


# =============================================================================
# 6. INSTRUMENT COMPOSITE — étape 1
#    Heures de nuit astronomique par mine et par année
#
#    Motivation : au Nunavut (lat 63–71 °N) le « soleil de minuit » réduit
#    fortement le nombre d'heures où VIIRS/DMSP peuvent observer les
#    lumières nocturnes. Cette quantité est purement géophysique
#    (fonction de la latitude et de la date) → instrument exogène
#    candidat, à combiner ensuite avec la couverture nuageuse (section 7).
#
#    Convention seuil :
#      -18° = nuit astronomique  → standard pour VIIRS-DNB monthly composites
#      -12° = crépuscule nautique
#       -6° = crépuscule civil (DMSP-OLS)
# =============================================================================

def heures_nuit_journaliere(lat_deg, day_of_year, seuil_elev_deg=-18.0):
    """
    Heures de nuit (élévation solaire <= seuil) pour une latitude et un
    jour julien donnés. Renvoie une valeur dans [0, 24].

    Gère explicitement les cas polaires :
      cos(H) >  1 → seuil au-dessus du max de sin(elev) (atteint à midi solaire)
                    le soleil ne dépasse JAMAIS le seuil → 24 h sous le seuil
      cos(H) < -1 → seuil sous le min de sin(elev) (atteint à minuit solaire)
                    le soleil reste TOUJOURS au-dessus du seuil → 0 h sous le seuil
                    (cas du soleil de minuit avec seuil = -18°)
    """
    # Déclinaison solaire (Cooper, 1969)
    delta = 23.45 * np.sin(np.deg2rad(360.0 * (284 + day_of_year) / 365.0))

    phi   = np.deg2rad(lat_deg)
    dlt   = np.deg2rad(delta)
    seuil = np.deg2rad(seuil_elev_deg)

    # sin(elev) = sin(φ)·sin(δ) + cos(φ)·cos(δ)·cos(H)
    # Angle horaire H où sin(elev) = sin(seuil) :
    cos_H = (np.sin(seuil) - np.sin(phi) * np.sin(dlt)) / (np.cos(phi) * np.cos(dlt))

    if cos_H > 1.0:
        # soleil toujours sous le seuil → 24 h de « nuit »
        return 24.0
    if cos_H < -1.0:
        # soleil toujours au-dessus du seuil → 0 h de nuit (soleil de minuit)
        return 0.0

    H_deg       = np.rad2deg(np.arccos(cos_H))
    heures_jour = 2.0 * H_deg / 15.0          # 15°/h
    return 24.0 - heures_jour


# Coordonnées WGS84 des 4 mines étudiées
# (les colonnes Lat/Lon des CSV VIIRS sont en projection Mollweide, inutilisables ici)
MINES_COORDS = {
    "Mine_Meadowbank": {"lat": 65.0200, "lon":  -96.0800},
    "Mine_Meliadine" : {"lat": 63.0300, "lon":  -92.1900},
    "Mine_Hope_Bay"  : {"lat": 68.1400, "lon": -106.6200},
    "Mine_Baffinland": {"lat": 71.3300, "lon":  -79.3500},  # Mary River
}

SEUILS = {
    "nuit_astro" : -18.0,   # standard VIIRS-DNB
    "nuit_naut"  : -12.0,
    "nuit_civile":  -6.0,   # standard DMSP-OLS
}

records = []
for annee in ANNEES:
    n_jours = 366 if pd.Timestamp(annee, 12, 31).dayofyear == 366 else 365
    jours   = np.arange(1, n_jours + 1)

    for mine, coords in MINES_COORDS.items():
        row = {"Mine": mine, "Annee": annee, "Latitude": coords["lat"], "n_jours": n_jours}
        for label, seuil in SEUILS.items():
            heures = np.array([
                heures_nuit_journaliere(coords["lat"], d, seuil)
                for d in jours
            ])
            row[f"{label}_h_total"] = round(float(heures.sum()), 2)
            row[f"{label}_part"]    = round(float(heures.sum() / (24.0 * n_jours)), 4)
        records.append(row)

df_nuit = pd.DataFrame(records)

print("\n=== Heures de nuit astronomique par mine — aperçu (3 années) ===")
print(df_nuit.query("Annee in [2000, 2012, 2024]").to_string(index=False))

OUTPUT_NUIT = os.path.join(BASE_DIR, "donnees_meteo", "heures_nuit_mine_annuel.csv")
df_nuit.to_csv(OUTPUT_NUIT, index=False)
print(f"\nCSV sauvegardé : {OUTPUT_NUIT}")

# Sanity check : Baffinland (71.3°N) doit avoir nettement moins d'heures
# de nuit astro que Meliadine (63.0°N) chaque année.
diff = (df_nuit.query("Mine == 'Mine_Meliadine'")["nuit_astro_h_total"].mean()
        - df_nuit.query("Mine == 'Mine_Baffinland'")["nuit_astro_h_total"].mean())
print(f"\nSanity check — écart annuel moyen Meliadine − Baffinland (nuit astro) : "
      f"{diff:+.0f} h  (attendu : > 0, car Baffinland est plus au nord)")


# =============================================================================
# 7. INSTRUMENT COMPOSITE — étape 2
#    Z_{m,t} = Σ_mois  h_nuit(lat_m, mois) × (1 − cloud_cover(lat_m, lon_m, mois))
#             ────────────────────────────────────────────────────────────────
#                                  heures totales de l'année t
#
#    Interprétation : fraction de l'année t pendant laquelle le ciel au-dessus
#    de la mine m est à la fois (i) astronomiquement nuit (sun ≤ -18°)
#    et (ii) suffisamment clair pour qu'un capteur DNB voie la surface.
#
#    Exogénéité : géophysique (orbite Terre/Soleil) × météo synoptique,
#    indépendant du niveau de production minière.
#
#    Pertinence (premier-stage) : les lumières VIIRS/DMSP mesurées par mine
#    sont mécaniquement plafonnées par cette borne d'observabilité.
# =============================================================================

# 7.1 — Ré-ouvrir le NetCDF (sans redéclencher l'API)
if not os.path.exists(OUTPUT_NC):
    raise FileNotFoundError(
        f"NetCDF manquant : {OUTPUT_NC}\n"
        f"→ lancer d'abord la section 2 pour le télécharger."
    )

ds_cloud   = xr.open_dataset(OUTPUT_NC)
var_cloud  = [v for v in ds_cloud.data_vars if "cc" in v.lower() or "cloud" in v.lower()][0]
cloud_grid = ds_cloud[var_cloud]                                 # dims : valid_time × lat × lon
time_dim   = "valid_time" if "valid_time" in cloud_grid.dims else "time"

# Garde-fou : chaque mine doit être DANS la grille (sinon nearest renvoie un bord)
lat_min, lat_max = float(cloud_grid.latitude.min()),  float(cloud_grid.latitude.max())
lon_min, lon_max = float(cloud_grid.longitude.min()), float(cloud_grid.longitude.max())
hors_grille = []
for mine, c in MINES_COORDS.items():
    lon_ref = c["lon"] + 360.0 if (lon_min >= 0 and c["lon"] < 0) else c["lon"]
    if not (lat_min <= c["lat"] <= lat_max and lon_min <= lon_ref <= lon_max):
        hors_grille.append((mine, c["lat"], c["lon"]))
if hors_grille:
    msg = "\n".join(f"  - {m}: lat={lat}, lon={lon}" for m, lat, lon in hors_grille)
    raise ValueError(
        "Mines hors de la grille ERA5 — les cloud_cover seraient biaisées :\n"
        f"{msg}\n"
        f"Grille actuelle : lat [{lat_min},{lat_max}], lon [{lon_min},{lon_max}]\n"
        f"→ élargir BBOX_NUNAVUT (section 1) et relancer la section 2 "
        f"pour re-télécharger le NetCDF."
    )


# 7.2 — Extraction nearest-neighbor au-dessus de chaque mine
def cloud_serie_point(da, lat, lon, time_dim_name):
    """Série mensuelle de cloud cover (fraction, clippée [0,1]) au point (lat, lon)."""
    # ERA5 peut indexer les longitudes en [-180,180] ou [0,360]
    if float(da.longitude.min()) >= 0 and lon < 0:
        lon = lon + 360.0
    serie = da.sel(latitude=lat, longitude=lon, method="nearest")
    vals  = np.clip(serie.values.astype(float), 0.0, 1.0)
    dates = pd.to_datetime(serie[time_dim_name].values)
    return pd.DataFrame({"date": dates, "cloud_cover": vals})


# 7.3 — Heures de nuit mensuelles (somme des h journalières du mois)
def heures_nuit_mensuelle(lat_deg, annee, mois, seuil=-18.0):
    debut = pd.Timestamp(int(annee), int(mois), 1)
    nb_j  = debut.daysinmonth
    return sum(
        heures_nuit_journaliere(lat_deg, (debut + pd.Timedelta(days=i)).dayofyear, seuil)
        for i in range(nb_j)
    )


SEUIL_INSTR = -18.0   # nuit astronomique = standard VIIRS-DNB

records_inst = []
for mine, coords in MINES_COORDS.items():
    df_m = cloud_serie_point(cloud_grid, coords["lat"], coords["lon"], time_dim)
    df_m["Annee"] = df_m["date"].dt.year
    df_m["mois"]  = df_m["date"].dt.month

    df_m["h_nuit_mois"]  = [
        heures_nuit_mensuelle(coords["lat"], a, m, SEUIL_INSTR)
        for a, m in zip(df_m["Annee"], df_m["mois"])
    ]
    df_m["h_total_mois"] = [
        24.0 * pd.Timestamp(int(a), int(m), 1).daysinmonth
        for a, m in zip(df_m["Annee"], df_m["mois"])
    ]
    df_m["h_obs_mois"]   = df_m["h_nuit_mois"] * (1.0 - df_m["cloud_cover"])

    agg = (
        df_m.groupby("Annee")
            .agg(cloud_cover_mine = ("cloud_cover",  "mean"),
                 h_nuit_annee     = ("h_nuit_mois",  "sum"),
                 h_obs_annee      = ("h_obs_mois",   "sum"),
                 h_total_annee    = ("h_total_mois", "sum"))
            .reset_index()
    )
    agg["Mine"]          = mine
    agg["Latitude"]      = coords["lat"]
    agg["part_nuit"]     = agg["h_nuit_annee"] / agg["h_total_annee"]
    agg["part_clair"]    = 1.0 - agg["cloud_cover_mine"]
    agg["observabilite"] = agg["h_obs_annee"]  / agg["h_total_annee"]
    records_inst.append(agg)

df_inst = pd.concat(records_inst, ignore_index=True)
df_inst = df_inst[[
    "Mine", "Annee", "Latitude",
    "cloud_cover_mine", "part_clair",
    "h_nuit_annee",     "part_nuit",
    "h_obs_annee",      "observabilite",
]]
df_inst[["cloud_cover_mine", "part_clair", "part_nuit", "observabilite"]] = (
    df_inst[["cloud_cover_mine", "part_clair", "part_nuit", "observabilite"]].round(4)
)
df_inst[["h_nuit_annee", "h_obs_annee"]] = df_inst[["h_nuit_annee", "h_obs_annee"]].round(1)

# 7.4 — Sauvegarde
OUTPUT_INSTR = os.path.join(BASE_DIR, "donnees_meteo", "instrument_composite_mine_annuel.csv")
df_inst.to_csv(OUTPUT_INSTR, index=False)

print("\n=== Instrument composite — aperçu (3 années × 4 mines) ===")
print(df_inst.query("Annee in [2010, 2017, 2024]").to_string(index=False))
print(f"\nCSV sauvegardé : {OUTPUT_INSTR}")

# 7.5 — Diagnostics
print("\n--- Moyennes 2000-2024 par mine ---")
moy = df_inst.groupby("Mine")[
    ["cloud_cover_mine", "part_clair", "part_nuit", "observabilite"]
].mean().round(3)
print(moy.to_string())

print("\n--- Décomposition de la variance de l'observabilité ---")
sd_total = df_inst["observabilite"].std()
sd_between = df_inst.groupby("Mine")["observabilite"].mean().std()
sd_within  = df_inst.groupby("Mine")["observabilite"].std().mean()
print(f"  σ totale       : {sd_total:.4f}")
print(f"  σ inter-mines  : {sd_between:.4f}   (variation latitudinale)")
print(f"  σ intra-mine   : {sd_within:.4f}    (variation annuelle des nuages)")
print(f"  → ratio within/total : {sd_within/sd_total:.2%}  "
      f"(part de la variance utilisable dans un modèle FE-mine)")

print("\n--- Pour fusion avec la base principale ---")
print('  df = df.merge(df_inst[["Mine","Annee","observabilite",')
print('                         "part_nuit","cloud_cover_mine"]],')
print('                on=["Mine","Annee"], how="left")')
