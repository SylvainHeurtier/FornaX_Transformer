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
        ('PNT_DET_ML_SPURIOUS', PNT_DET_ML_SPURIOUS, '', ''),
        ('EXT_LIKE_SPURIOUS', EXT_LIKE_SPURIOUS, '', ''),
        ('NOMBRE_PHOTONS_MIN', NOMBRE_PHOTONS_MIN, 'photons', '')
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
