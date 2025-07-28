import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import astropy.units as u
import random
import pandas as pd
import copy
import json
import numba

from tqdm.auto import tqdm
from textwrap import fill
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.coordinates import search_around_sky
from matplotlib.patches import Rectangle
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from sklearn.model_selection import train_test_split
import matplotlib.colors as colors
from astropy.coordinates import angular_separation
import astropy.units as u
from astropy.table import join
from collections import Counter

from Constantes import LIM_FLUX_CLUSTER, LIM_FLUX_AGN
from Constantes import SEARCH_RADIUS_CLUSTER, SEARCH_RADIUS_AGN
from Constantes import WINDOW_SIZE_ARCMIN, NOMBRE_PHOTONS_MIN, MAX_Xamin_PAR_FENESTRON
from Constantes import VOCAB_SIZE, PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, SEP_AMAS, SEP_AGN, NOMBRE_TOKENS_SPECIAUX
from Constantes import EXT_LIKE_C1, EXT_LIKE_C2, EXT_C1_C2, PNT_DET_ML_SPURIOUS, EXT_LIKE_SPURIOUS
from Constantes import catalog_path_aftXamin, catalog_path_AGN, catalog_path_AMAS
from Constantes import SELECTED_COLUMNS_Xamin, SELECTED_COLUMNS_input_clusters, SELECTED_COLUMNS_input_AGN
from Constantes import Simulation1, Simulation2, name_dir

from Constantes import print_parameters

print_parameters()




#///////////////////////////////////////////////////////////////////////////////////////////////////////
#                              PREPARATION DES DONNEES - CORRELATION
#///////////////////////////////////////////////////////////////////////////////////////////////////////


titre = "PRÉPARATION DES DONNÉES - CORRÉLATION"
largeur = 80

print("#" * largeur)
print("#" + " " * (largeur-2) + "#")
print("#" + titre.center(largeur-2) + "#")
print("#" + " " * (largeur-2) + "#")
print("#" * largeur)


# //////////// Chargements des fichiers ////////////

print(f"\n Chargements des fichiers \n")

data_Xamin      = Table.read(catalog_path_aftXamin)
data_input_AMAS = Table.read(catalog_path_AMAS)
data_input_AGN  = Table.read(catalog_path_AGN)

data_Xamin['new_ID'] = np.arange(len(data_Xamin))
data_Xamin['Ntot'] = data_Xamin['INST0_EXP'] * data_Xamin['PNT_RATE_MOS'] + data_Xamin['INST1_EXP'] * data_Xamin['PNT_RATE_PN']
print(f"Nombre de sources Xamin:\nAvant la coupe sur le nombre de photons: {len(data_Xamin)}")
data_Xamin = data_Xamin[data_Xamin['Ntot']>=NOMBRE_PHOTONS_MIN]
print(f"Apres la coupe sur le nombre de photons: {len(data_Xamin)}")
print(f"\nRappel: NOMBRE_PHOTONS_MIN = {NOMBRE_PHOTONS_MIN} photons")


# //////////// Statistiques C1/C2 ////////////

print(f"\n Statistiques C1/C2 \n")

data_numeric = data_Xamin.to_pandas()

# Définition des classes C1 et C2
cond_C1 = np.logical_and((data_numeric['EXT'] > EXT_C1_C2) , (data_numeric['EXT_LIKE'] >= EXT_LIKE_C1))
cond_C2 = np.logical_and(np.logical_and((data_numeric['EXT'] > EXT_C1_C2) , (data_numeric['EXT_LIKE'] < EXT_LIKE_C1)),
                        (data_numeric['EXT_LIKE'] > EXT_LIKE_C2))

n_C1 = sum(cond_C1)
n_C2 = sum(cond_C2)
ni_C1_ni_C2 = len(data_numeric) - (n_C1+n_C2)

print("="*70)
print(f"Catalogue Xamin : {len(data_numeric)} dont Nph >= {NOMBRE_PHOTONS_MIN}")
print(f"\nNombre d'amas dans la classe C1 (EXT>{EXT_C1_C2} ET EXT_LIKE>={EXT_LIKE_C1}): {n_C1}")
print(f"Nombre d'amas dans la classe C2 (EXT>{EXT_C1_C2} ET {EXT_LIKE_C2}<EXT_LIKE<{EXT_LIKE_C1}): {n_C2}")
print(f"Nombre d'amas ni dans C1, ni dans C2: {ni_C1_ni_C2}")
print("="*70)


# Noms actuels des colonnes (pour vérification)
current_cols = data_input_AMAS.colnames
print("Colonnes actuelles:")
print(current_cols)

if(Simulation1):
    new_column_names = ['ID', 'R.A.', 'Dec', 'px', 'yx', 'm200', 'Tsl', 'Lx_soft', 'flux', 'flux_ABS', 'r500', 'z']

    # Renommer uniquement les colonnes 2 à 13
    for i in range(1, 13):
        data_input_AMAS.rename_column(current_cols[i], new_column_names[i-1])

    print("\nColonnes après renommage:")
    print(data_input_AMAS.colnames)

    #data_input_AGN.rename_column('id', 'object_id')
    data_input_AGN.rename_column('ra', 'ra_mag_gal')
    data_input_AGN.rename_column('dec', 'dec_mag_gal')

if Simulation2:
    for colname in data_input_AMAS.colnames:
        # Vérifie si la colonne est numérique (int, float, etc.)
        if data_input_AMAS[colname].dtype.kind in 'iufc':
            # Convertit en float64 (meilleure précision que float32)
            data_input_AMAS[colname] = data_input_AMAS[colname].astype(np.float64)
            
    # Afficher un aperçu de la data_input_AMAS pour vérification
    print(data_input_AMAS.colnames)

print(f"\n Filtrage du flux \n")

if (Simulation1):
    clefluxAGN = 'Fx_s_abs'
if (Simulation2):
    clefluxAGN = 'Fx_05_2'

print("STATISTIQUES DE FILTRAGE")
print(f"Nombre initial d'amas : {len(data_input_AMAS)}")
print(f"Nombre d'amas après masque (flux_ABS > {LIM_FLUX_CLUSTER}) : {len(data_input_AMAS[data_input_AMAS['flux_ABS'] > LIM_FLUX_CLUSTER])}")
print(f"\n{'='*50}")
print("STATISTIQUES DE FILTRAGE")
print(f"Nombre initial d'AGN : {len(data_input_AGN)}")
print(f"Nombre d'AGN après masque (flux_ABS > {LIM_FLUX_AGN}) : {len(data_input_AGN[data_input_AGN[clefluxAGN] > LIM_FLUX_AGN])}")
print(f"\n{'='*50}")


mask_flux_cluster = data_input_AMAS['flux_ABS'] > LIM_FLUX_CLUSTER
data_input_AMAS = data_input_AMAS[mask_flux_cluster]

mask_flux_agn = data_input_AGN[clefluxAGN] > LIM_FLUX_AGN
data_input_AGN = data_input_AGN[mask_flux_agn]

def filter_matched_sources(tableXAMIN, tableINPUT,
                           ra_xamin, dec_xamin,
                           ra_input, dec_input,
                           search_radius):

    search_radius_arcsec = (search_radius* u.deg).to(u.arcsec)  # en arcsec

    # Nettoyage des NaN
    mask_xamin = ~(np.isnan(tableXAMIN[ra_xamin])) & ~(np.isnan(tableXAMIN[dec_xamin]))
    mask_input = ~(np.isnan(tableINPUT[ra_input])) & ~(np.isnan(tableINPUT[dec_input]))
    clean_XAMIN = tableXAMIN[mask_xamin]
    clean_INPUT = tableINPUT[mask_input]

    # Conversion en SkyCoord
    coords_XAMIN = SkyCoord(ra=clean_XAMIN[ra_xamin]*u.deg, dec=clean_XAMIN[dec_xamin]*u.deg)
    coords_INPUT = SkyCoord(ra=clean_INPUT[ra_input]*u.deg, dec=clean_INPUT[dec_input]*u.deg)
    
    # Recherche des paires dans le rayon
    idx_input, idx_xamin, sep2d, _ = search_around_sky(coords_INPUT, coords_XAMIN, seplimit=search_radius_arcsec)
    
    # Récupérer les ID_Xamin correspondants
    matched_xamin_indices = np.unique(idx_xamin)
    matched_ids = clean_XAMIN['ID_Xamin'][matched_xamin_indices]

    # Créer un masque des sources INPUT ayant des correspondances
    matched_mask = np.isin(np.arange(len(tableINPUT)), np.unique(idx_input))
    
    return tableINPUT[matched_mask], np.array(matched_ids)


data_AMAS, list_ID_Xamin_AMAS = filter_matched_sources(data_Xamin, data_input_AMAS,
                                                       ra_xamin = 'EXT_RA', dec_xamin = 'EXT_DEC',
                                                       ra_input = 'R.A.', dec_input = 'Dec',
                                                       search_radius = SEARCH_RADIUS_CLUSTER)


data_AGN, list_ID_Xamin_AGN = filter_matched_sources(data_Xamin, data_input_AGN,
                                                     ra_xamin = 'PNT_RA', dec_xamin = 'PNT_DEC',
                                                     ra_input = 'ra_mag_gal', dec_input = 'dec_mag_gal',
                                                     search_radius = SEARCH_RADIUS_AGN)


separator = "═" * 60

print(f"""
{separator}
        RÉSULTATS DE LA CORRÉLATION DES CATALOGUES
{separator}

Sources Xamin (Nph ≥ {NOMBRE_PHOTONS_MIN}):
   • Total initial : {len(data_Xamin):>6d} sources

Correspondances amas/AGN:
{'-'*50}
Amas:
  • Catalogue d'entrée : {len(data_input_AMAS):>6d}
  • Corrélés avec Xamin :  {len(data_AMAS):>4d} ({len(data_AMAS)/len(data_input_AMAS):.1%})

AGN:
  • Catalogue d'entrée : {len(data_input_AGN):>6d}
  • Corrélés avec Xamin :  {len(data_AGN):>6d} ({len(data_AGN)/len(data_input_AGN):.1%})

{separator}
""")


print(f"\n Sauvegarde des fichiers \n")


# Sauvegarde fichiers d'entree AMAS

output_filename = f"AMAS_matches_r{SEARCH_RADIUS_CLUSTER*3600:.0f}arcsec_flux{LIM_FLUX_CLUSTER}_40ks.fits"
output_path = os.path.expanduser(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/{output_filename}')

data_AMAS.write(output_path, format='fits', overwrite=True)

with fits.open(output_path, mode='update') as hdul:
    hdr = hdul[1].header
    hdr['COMMENT'] = 'Input AMAS spatially matched source catalog'
    hdr['R_MATCH'] = (SEARCH_RADIUS_CLUSTER*3600, '[arcsec] Matching radius') 
    hdr['N_MATCH'] = (len(data_AMAS), 'Total matched sources')

print(f"\nCatalogue complet sauvegardé dans : {output_path}")
print(f"Dimensions : {len(data_AMAS)} lignes x {len(data_AMAS.columns)} colonnes")

# Sauvegarde fichiers des ID_Xamin des AMAS

output_filename = f"list_ID_Xamin_AMAS_matches_r{SEARCH_RADIUS_CLUSTER*3600:.0f}arcsec_flux{LIM_FLUX_CLUSTER}_40ks.fits"
output_path = os.path.expanduser(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/{output_filename}')
np.savetxt(output_path, list_ID_Xamin_AMAS, fmt='%d')

# Sauvegarde fichiers d'entree AGN

output_filename = f"AGN_matches_r{SEARCH_RADIUS_AGN*3600:.0f}arcsec_flux{LIM_FLUX_AGN}_40ks.fits"
output_path = os.path.expanduser(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/{output_filename}')

data_AGN.write(output_path, format='fits', overwrite=True)

with fits.open(output_path, mode='update') as hdul:
    hdr = hdul[1].header
    hdr['COMMENT'] = 'Input AGN spatially matched source catalog'
    hdr['R_MATCH'] = (SEARCH_RADIUS_AGN*3600, '[arcsec] Matching radius') 
    hdr['N_MATCH'] = (len(data_AGN), 'Total matched sources')

print(f"\nCatalogue complet sauvegardé dans : {output_path}")
print(f"Dimensions : {len(data_AGN)} lignes x {len(data_AGN.columns)} colonnes")

# Sauvegarde fichiers des ID_Xamin des AGN

output_filename = f"list_ID_Xamin_AGN_matches_r{SEARCH_RADIUS_AGN*3600:.0f}arcsec_flux{LIM_FLUX_AGN}_40ks.fits"
output_path = os.path.expanduser(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/{output_filename}')
np.savetxt(output_path, list_ID_Xamin_AGN, fmt='%d')

print(f""" \n
{separator}
                             THE END
{separator}
\n
""")

