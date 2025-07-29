

#///////////////////////////////////////////////////////////////////////////////////////////////////////
#                                       DEFINITIONS DES CONSTANTES
#///////////////////////////////////////////////////////////////////////////////////////////////////////



# /// CONSTANTES ///

LIM_FLUX_CLUSTER = 1e-15
LIM_FLUX_AGN = 1e-15
SEARCH_RADIUS_CLUSTER = 20.0 / 3600  # conversion arcsec en degrés
SEARCH_RADIUS_AGN = 10.0 / 3600  # conversion arcsec en degrés

# Definitions des classes C1 et C2
EXT_LIKE_C1 = 33
EXT_LIKE_C2 = 15
EXT_C1_C2 = 5

# Definitions des nouvelles classes C1 et C2
EXT_C1_C2_new = 13
EXT_LIKE_C1_new = 80
EXT_LIKE_C2_new = 35

# Taille de la fenetre carree
WINDOW_SIZE_ARCMIN  = 2 # arcmin
MAX_Xamin_PAR_FENESTRON = 2

# Fausses sources
PNT_DET_ML_SPURIOUS = 20
EXT_LIKE_SPURIOUS = 15

# Limite inferieur sur le nombre de photons des sources Xamin selectionnees
NOMBRE_PHOTONS_MIN = 100

# Nombre de rotations de la fenetre
Nbre2Rotations = 3200

# Dictionnaire pour la tokenisation

VOCAB_SIZE = 1029 # Tokens valides: 0-1023 = 1024 tokens + 3 tokens speciaux
PAD_TOKEN  = 1024  # Padding
SEP_TOKEN  = 1025  # Marque la fin de la sequence
CLS_TOKEN  = 1026  # Marque le debut de la sequence
SEP_AMAS   = 1027 # Marque le debut du catalogue AMAS
SEP_AGN    = 1028 # Marque le debut du catalogue AGN

NOMBRE_TOKENS_SPECIAUX = 5


# Dimensions du Transformer

BATCH_SIZE = 32
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 6


#/////////////////////////////////////////////////////////////////////////

def print_parameters():
    """Affiche les paramètres avec un alignement parfait."""
    params = [
        ('LIM_FLUX_CLUSTER', LIM_FLUX_CLUSTER, 'erg/cm²/s', '.2e'),
        ('LIM_FLUX_AGN', LIM_FLUX_AGN, 'erg/cm²/s', '.2e'),
        ('SEARCH_RADIUS_CLUSTER', SEARCH_RADIUS_CLUSTER * 3600, 'arcsec', '.2f'),
        ('SEARCH_RADIUS_AGN', SEARCH_RADIUS_AGN * 3600, 'arcsec', '.2f'),
        ('EXT_LIKE_C1', EXT_LIKE_C1, '', ''),
        ('EXT_LIKE_C2', EXT_LIKE_C2, '', ''),
        ('EXT_C1_C2', EXT_C1_C2, 'arcsec', ''),
        ('EXT_LIKE_C1_new', EXT_LIKE_C1_new, '', ''),
        ('EXT_LIKE_C2_new', EXT_LIKE_C2_new, '', ''),
        ('EXT_C1_C2_new', EXT_C1_C2_new, 'arcsec', ''),
        ('window_size', WINDOW_SIZE_ARCMIN, 'arcmin', '.1f'),
        ('MAX_Xamin_PAR_FENESTRON', MAX_Xamin_PAR_FENESTRON, '', ''),
        ('Nbre2Rotations', Nbre2Rotations, '', ''),
        ('PNT_DET_ML_SPURIOUS', PNT_DET_ML_SPURIOUS, '', ''),
        ('EXT_LIKE_SPURIOUS', EXT_LIKE_SPURIOUS, '', ''),
        ('NOMBRE_PHOTONS_MIN', NOMBRE_PHOTONS_MIN, 'photons', ''),
        ('VOCAB_SIZE', VOCAB_SIZE, '', ''),
        ('PAD_TOKEN', PAD_TOKEN, '', ''),
        ('SEP_TOKEN', SEP_TOKEN, '', ''),
        ('SEP_AMAS', SEP_AMAS, '', ''),
        ('SEP_AGN', SEP_AGN, '', ''),
        ('NOMBRE_TOKENS_SPECIAUX', NOMBRE_TOKENS_SPECIAUX, '', ''),
        ('BATCH_SIZE', BATCH_SIZE, '', ''),
        ('D_MODEL', D_MODEL, '', ''),
        ('NUM_HEADS', NUM_HEADS, '', ''),
        ('NUM_LAYERS', NUM_LAYERS, '', '')
    ]
    
    # Calcul des largeurs
    max_name_len = max(len(p[0]) for p in params)
    max_value_len = max(len(f"{p[1]:{p[3]}}" if p[3] else str(p[1])) for p in params)
    max_unit_len = max(len(p[2]) for p in params)
    
    # Largeur totale du cadre
    total_width = max_name_len + max_value_len + max_unit_len + 7  # 7 caractères fixes
    
    print(f"╭{'─' * (total_width)}╮")
    print(f"│{' PARAMÈTRES '.center(total_width)}│")
    print(f"├{'─' * (total_width)}┤")
    
    for name, value, unit, fmt in params:
        value_str = f"{value:{fmt}}" if fmt else str(value)
        line = f"│ {name:<{max_name_len}} : {value_str:<{max_value_len}}"
        if unit:
            line += f" {unit:<{max_unit_len}}"
        line += " "
        print(line)
    
    print(f"╰{'─' * (total_width)}╯")


#/////////////////////////////////////////////////////////////////////////

import os

Simulation1 = True
Simulation2 = False

if(Simulation1):
    name_dir = 'Simulation1'
    catalog_path_AMAS = os.path.expanduser('/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/data/Simulation1/XFSII_25_sx_p18_b05rc02_output.csv')
    catalog_path_aftXamin = os.path.expanduser('/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/data/Simulation1/merged_catalog_cleaned.fits')
    catalog_path_AGN      = os.path.expanduser('/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/data/Simulation1/FS2_MAMBO_AGN.fits')

if(Simulation2):
    name_dir = 'Simulation2'
    catalog_path_AMAS = os.path.expanduser('/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/data/Simulation2/fsII_25_lensed_AGN1/XFSII_25_p18_b05rc02_lensed_1e13Mo_output_cleaned.fits')
    catalog_path_aftXamin = os.path.expanduser('/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/data/Simulation2/fsII_25_lensed_AGN1/Xamin_onlyMOSPN/merged_catalog_cleaned.fits')
    catalog_path_AGN      = os.path.expanduser('/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/data/Simulation2/fsII_25_lensed_AGN1/FS2_MAMBO_AGN_1.fits')

path_list_ID_Xamin_AMAS = f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/list_ID_Xamin_AMAS_matches_r{SEARCH_RADIUS_CLUSTER*3600:.0f}arcsec_flux{LIM_FLUX_CLUSTER}_40ks.fits'
path_list_ID_Xamin_AGN  = f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/list_ID_Xamin_AGN_matches_r{SEARCH_RADIUS_AGN*3600:.0f}arcsec_flux{LIM_FLUX_AGN}_40ks.fits'

new_catalog_path_AMAS  = os.path.expanduser(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/AMAS_matches_r{SEARCH_RADIUS_CLUSTER*3600:.0f}arcsec_flux{LIM_FLUX_CLUSTER}_40ks.fits')
new_catalog_path_AGN  = os.path.expanduser(f'/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/AGN_matches_r{SEARCH_RADIUS_AGN*3600:.0f}arcsec_flux{LIM_FLUX_AGN}_40ks.fits')

#/////////////////////////////////////////////////////////////////////////

print(f" \n        *** Constantes utilisees *** ")

#SELECTED_COLUMNS_Xamin = ['EXT_LIKE', 'EXT', 'EXT_RA', 'EXT_DEC' ,'PNT_DET_ML', 'PNT_RA', 'PNT_DEC', 'PNT_RATE_MOS', 'PNT_RATE_PN']
SELECTED_COLUMNS_Xamin = ['EXT_LIKE', 'EXT', 'EXT_RA', 'EXT_DEC' ,'PNT_DET_ML']

SELECTED_COLUMNS_input_clusters = ['R.A.', 'Dec']
SELECTED_COLUMNS_input_AGN = ['ra_mag_gal', 'dec_mag_gal']

print(f'\nNombre de colonnes SELECTED_COLUMNS_Xamin: {len(SELECTED_COLUMNS_Xamin)}')
print(f'Nombre de colonnes SELECTED_COLUMNS_input_clusters: {len(SELECTED_COLUMNS_input_clusters)}')
print(f'Nombre de colonnes SELECTED_COLUMNS_input_AGN: {len(SELECTED_COLUMNS_input_AGN)}')

#use_log_scale_Xamin = [True, True, False, False, True, False, False, True, True]
use_log_scale_Xamin = [True, True, False, False, True]
use_log_scale_input_clusters = [False, False]
use_log_scale_input_AGN = [False, False]

print(f'\nNombre de colonnes use_log_scale_Xamin: {len(use_log_scale_Xamin)}')
print(f'Nombre de colonnes use_log_scale_input_clusters: {len(use_log_scale_input_clusters)}')
print(f'Nombre de colonnes use_log_scale_input_AGN: {len(use_log_scale_input_AGN)}')