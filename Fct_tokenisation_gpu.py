
import cudf
import cupy as cp
from cuspatial import haversine_distance, lonlat_to_xy
from numba import cuda
from collections import Counter


from Constantes import LIM_FLUX_CLUSTER, LIM_FLUX_AGN
from Constantes import SEARCH_RADIUS_CLUSTER, SEARCH_RADIUS_AGN
from Constantes import WINDOW_SIZE_ARCMIN, NOMBRE_PHOTONS_MIN, MAX_Xamin_PAR_FENESTRON
from Constantes import VOCAB_SIZE, PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, SEP_AMAS, SEP_AGN, NOMBRE_TOKENS_SPECIAUX
from Constantes import catalog_path_aftXamin, catalog_path_AGN, catalog_path_AMAS
from Constantes import SELECTED_COLUMNS_Xamin, SELECTED_COLUMNS_input_clusters, SELECTED_COLUMNS_input_AGN
from Constantes import print_parameters

print_parameters()


#///////////////////////////////////////////////////////////////////////////////////////////////////////
#                                       DEFINITIONS FUNCTIONS
#///////////////////////////////////////////////////////////////////////////////////////////////////////


def CreateListID_Xamin(all_xamin_ids, list_id_xamin_amas, proportion=1):
    """
    Version GPU-optimisée avec CuPy.
    Requiert que les inputs soient déjà des arrays CuPy pour éviter les transferts CPU-GPU.
    """
    # Convertir les entrées en arrays CuPy si ce n'est pas déjà le cas
    all_ids_gpu = cp.asarray(all_xamin_ids)
    amas_ids_gpu = cp.asarray(list_id_xamin_amas)
    
    # Trouver les IDs communs (sur GPU)
    common_ids = cp.intersect1d(amas_ids_gpu, all_ids_gpu)
    
    # Trouver les IDs disponibles (sur GPU)
    available_ids = cp.setdiff1d(all_ids_gpu, amas_ids_gpu)
    
    # Calcul du nombre d'IDs à ajouter
    n_additional = int(proportion * len(common_ids))
    
    # Tirage aléatoire sur GPU
    if len(available_ids) >= n_additional:
        # Créer un générateur aléatoire CuPy
        rng = cp.random.default_rng()
        random_ids = rng.choice(available_ids, size=n_additional, replace=False)
    else:
        random_ids = available_ids
    
    # Concaténation finale sur GPU
    combined_ids = cp.concatenate([common_ids, random_ids])
    
    return combined_ids











def Batisseuse2Fenetres(data_Xamin, data_clusters, data_AGN, list_ID_Xamin,
                          cluster_columns=['window', 'R.A.', 'Dec', 'm200', 'z'],
                          AGN_columns=['window', 'ra_mag_gal', 'dec_mag_gal'],
                          window_size_arcmin=WINDOW_SIZE_ARCMIN):
    """
    Version GPU-optimisée utilisant RAPIDS (cuDF)
    Entrées et sorties en cuDF (pas de conversion Astropy)
    """
    # Conversion initiale en DataFrames GPU (si ce n'est pas déjà des cuDF)
    gdf_xamin = data_Xamin if isinstance(data_Xamin, cudf.DataFrame) else cudf.DataFrame.from_pandas(data_Xamin.to_pandas())
    gdf_clusters = data_clusters if isinstance(data_clusters, cudf.DataFrame) else cudf.DataFrame.from_pandas(data_clusters.to_pandas())
    gdf_agn = data_AGN if isinstance(data_AGN, cudf.DataFrame) else cudf.DataFrame.from_pandas(data_AGN.to_pandas())
    
    half_size_deg = (window_size_arcmin / 60) / 2
    selected_src = []
    selected_clusters = []
    selected_AGN = []

    # Pré-calcul des coordonnées en radians pour cuSpatial
    gdf_xamin['ra_rad'] = cp.radians(gdf_xamin['PNT_RA'])
    gdf_xamin['dec_rad'] = cp.radians(gdf_xamin['PNT_DEC'])

    for window_num, id in enumerate(list_ID_Xamin):
        # Centre de la fenêtre
        center = gdf_xamin[gdf_xamin['ID_Xamin'] == id][['PNT_RA', 'PNT_DEC']].iloc[0]
        center_ra, center_dec = center['PNT_RA'], center['PNT_DEC']
        
        # 1. Filtrage des sources Xamin
        delta_ra = cp.abs(gdf_xamin['PNT_RA'] - center_ra)
        delta_dec = cp.abs(gdf_xamin['PNT_DEC'] - center_dec)
        delta_ra = cp.minimum(delta_ra, 360 - delta_ra)
        
        mask = (delta_ra < half_size_deg) & (delta_dec < half_size_deg)
        win_sources = gdf_xamin[mask].copy()
        
        if len(win_sources) > 0:
            # Calcul des distances (méthode haversine GPU-optimisée)
            distances = haversine_distance(
                win_sources['ra_rad'], win_sources['dec_rad'],
                cp.radians(center_ra), cp.radians(center_dec))
            
            win_sources['separation_deg'] = cp.degrees(distances)
            win_sources['window'] = window_num
            
            # Recentrage des coordonnées
            for suffix in ['EXT', 'PNT', 'DBL', 'EPN']:
                ra_col = f"{suffix}_RA"
                dec_col = f"{suffix}_DEC"
                
                if ra_col in win_sources.columns:
                    win_sources[ra_col] -= center_ra
                    win_sources[dec_col] -= center_dec
            
            win_sources = win_sources.sort_values('separation_deg')
            selected_src.append(win_sources)

        # 2. Filtrage des clusters (même logique)
        delta_ra_cl = cp.abs(gdf_clusters['R.A.'] - center_ra)
        delta_dec_cl = cp.abs(gdf_clusters['Dec'] - center_dec)
        delta_ra_cl = cp.minimum(delta_ra_cl, 360 - delta_ra_cl)
        
        mask_cl = (delta_ra_cl < half_size_deg) & (delta_dec_cl < half_size_deg)
        win_clusters = gdf_clusters[mask_cl].copy()
        
        if len(win_clusters) > 0:
            distances_cl = haversine_distance(
                cp.radians(win_clusters['R.A.']), cp.radians(win_clusters['Dec']),
                cp.radians(center_ra), cp.radians(center_dec))
            
            win_clusters['separation_deg'] = cp.degrees(distances_cl)
            win_clusters['window'] = window_num
            win_clusters['R.A.'] -= center_ra
            win_clusters['Dec'] -= center_dec
            
            win_clusters = win_clusters.sort_values('separation_deg')[cluster_columns]
            selected_clusters.append(win_clusters)

        # 3. Filtrage des AGN
        delta_ra_agn = cp.abs(gdf_agn['ra_mag_gal'] - center_ra)
        delta_dec_agn = cp.abs(gdf_agn['dec_mag_gal'] - center_dec)
        delta_ra_agn = cp.minimum(delta_ra_agn, 360 - delta_ra_agn)
        
        mask_agn = (delta_ra_agn < half_size_deg) & (delta_dec_agn < half_size_deg)
        win_agn = gdf_agn[mask_agn].copy()
        
        if len(win_agn) > 0:
            distances_agn = haversine_distance(
                cp.radians(win_agn['ra_mag_gal']), cp.radians(win_agn['dec_mag_gal']),
                cp.radians(center_ra), cp.radians(center_dec))
            
            win_agn['separation_deg'] = cp.degrees(distances_agn)
            win_agn['window'] = window_num
            win_agn['ra_mag_gal'] -= center_ra
            win_agn['dec_mag_gal'] -= center_dec
            
            win_agn = win_agn.sort_values('separation_deg')[AGN_columns]
            selected_AGN.append(win_agn)

    # Concaténation des résultats (toujours en cuDF)
    list_windows = cudf.concat(selected_src) if selected_src else cudf.DataFrame()
    info_clusters = cudf.concat(selected_clusters) if selected_clusters else cudf.DataFrame()
    info_AGN = cudf.concat(selected_AGN) if selected_AGN else cudf.DataFrame()

    return list_windows, info_clusters, info_AGN













def GardeFenestronsSousPeuples(gdf_windows, gdf_clusters, gdf_AGN, max_Xamin_par_fenestron):
    """
    Version GPU-optimisée utilisant cuDF pour l'analyse des fenestrons.
    
    Args:
        gdf_windows: DataFrame cudf des sources
        gdf_clusters: DataFrame cudf des clusters (peut être None)
        gdf_AGN: DataFrame cudf des AGN (peut être None)
        max_Xamin_par_fenestron: int - nombre maximum de sources par fenestron
        
    Returns:
        Tuple de DataFrames cudf filtrés (windows, clusters, AGN)
    """
    # Vérification des entrées
    if not isinstance(gdf_windows, cudf.DataFrame):
        raise TypeError("gdf_windows doit être un DataFrame cudf")
    
    # Étape 1: Comptage des sources par fenestron (sur GPU)
    window_counts = gdf_windows['window'].value_counts().reset_index()
    window_counts.columns = ['window', 'count']
    
    # Étape 2: Identification des fenestrons valides
    valid_windows = window_counts[window_counts['count'] <= max_Xamin_par_fenestron]['window']
    valid_windows = valid_windows.to_cupy()  # Conversion en array CuPy pour performance

    # Étape 3: Filtrage des tables (toujours sur GPU)
    # Sources Xamin
    mask = gdf_windows['window'].isin(valid_windows)
    filtered_windows = gdf_windows[mask].copy()
    
    # Clusters
    if gdf_clusters is not None:
        mask_cl = gdf_clusters['window'].isin(valid_windows)
        filtered_clusters = gdf_clusters[mask_cl].copy()
    else:
        filtered_clusters = None
    
    # AGN
    if gdf_AGN is not None:
        mask_agn = gdf_AGN['window'].isin(valid_windows)
        filtered_agn = gdf_AGN[mask_agn].copy()
    else:
        filtered_agn = None

    return (filtered_windows,
            filtered_clusters if filtered_clusters is not None else cudf.DataFrame(),
            filtered_agn if filtered_agn is not None else cudf.DataFrame())





def CompteSourcesParFenetres(gdf_windows):
    """
    Version GPU-optimisée du comptage de sources par fenêtre.
    
    Args:
        gdf_windows: DataFrame cuDF avec colonne 'window' (doit déjà être un DataFrame cuDF)
        
    Returns:
        int: Nombre maximum de sources dans une fenêtre
    """
    # Vérification du type d'entrée
    if not isinstance(gdf_windows, cudf.DataFrame):
        raise TypeError("L'entrée doit être un DataFrame cuDF")
    
    # Comptage des sources par fenêtre (sur GPU)
    if len(gdf_windows) == 0:
        return 0
    
    window_counts = gdf_windows['window'].value_counts()
    
    # Récupération du maximum (reste sur GPU)
    max_count = window_counts.max()
    
    # Conversion en int Python (seul transfert GPU->CPU)
    return int(max_count)











def random_rotations_and_mirror(gdf_windows, gdf_clusters, gdf_agn, NumberOfRotations):
    """
    Version GPU-optimisée des rotations aléatoires et miroirs utilisant uniquement CuDF.
    
    Args:
        gdf_windows: DataFrame CuDF des sources
        gdf_clusters: DataFrame CuDF des clusters (peut être None)
        gdf_AGN: DataFrame CuDF des AGN (peut être None)
        NumberOfRotations: Nombre de rotations par fenêtre
        
    Returns:
        Tuple de DataFrames CuDF augmentés (windows, clusters, AGN)
    """
    # Vérification des entrées
    if not isinstance(gdf_windows, cudf.DataFrame):
        raise TypeError("gdf_windows doit être un DataFrame CuDF")

    # Préparation des colonnes de coordonnées
    coord_cols = []
    for suffix in ['EXT', 'PNT', 'DBL', 'EPN']:
        ra_col = f"{suffix}_RA"
        dec_col = f"{suffix}_DEC"
        if ra_col in gdf_windows.columns:
            coord_cols.append((ra_col, dec_col))

    # Noyau CUDA pour les rotations
    @cuda.jit
    def apply_rotations_kernel(ra, dec, angles, out_ra, out_dec):
        i = cuda.grid(1)
        if i < ra.shape[0]:
            angle = angles[i % angles.shape[0]]
            ra_rad = cp.radians(ra[i])
            dec_rad = cp.radians(dec[i])
            cos_t = cp.cos(angle)
            sin_t = cp.sin(angle)
            out_ra[i] = cp.degrees(ra_rad * cos_t - dec_rad * sin_t)
            out_dec[i] = cp.degrees(ra_rad * sin_t + dec_rad * cos_t)

    # Noyau CUDA pour le miroir
    @cuda.jit
    def apply_mirror_kernel(ra, out_ra):
        i = cuda.grid(1)
        if i < ra.shape[0]:
            out_ra[i] = -ra[i]

    unique_windows = gdf_windows['window'].unique().to_array()
    max_window_num = int(gdf_windows['window'].max()) if len(gdf_windows) > 0 else 0

    # Pré-allocation des résultats
    augmented_windows = [gdf_windows]
    augmented_clusters = [gdf_clusters] if gdf_clusters is not None else []
    augmented_agn = [gdf_agn] if gdf_agn is not None else []

    # Configuration CUDA
    threads_per_block = 256
    blocks_per_grid = (len(unique_windows) + threads_per_block - 1) // threads_per_block

    for window_id in unique_windows:
        win_mask = gdf_windows['window'] == window_id
        sub_src = gdf_windows[win_mask].copy()
        
        # Gestion des clusters et AGN
        sub_cluster = gdf_clusters[gdf_clusters['window'] == window_id].copy() if gdf_clusters is not None else None
        sub_agn = gdf_agn[gdf_agn['window'] == window_id].copy() if gdf_agn is not None else None

        # Génération des angles aléatoires
        angles = cp.random.uniform(0, 2*cp.pi, NumberOfRotations)

        # Rotation
        for angle in angles:
            # Rotation des sources
            rotated_src = sub_src.copy()
            for ra_col, dec_col in coord_cols:
                ra = rotated_src[ra_col].to_cupy()
                dec = rotated_src[dec_col].to_cupy()
                out_ra = cp.empty_like(ra)
                out_dec = cp.empty_like(dec)
                
                apply_rotations_kernel[blocks_per_grid, threads_per_block](ra, dec, angle, out_ra, out_dec)
                
                rotated_src[ra_col] = out_ra
                rotated_src[dec_col] = out_dec
            
            rotated_src['window'] = max_window_num + 1
            augmented_windows.append(rotated_src)

            # Rotation des clusters
            if sub_cluster is not None and len(sub_cluster) > 0:
                rotated_cluster = sub_cluster.copy()
                ra = rotated_cluster['R.A.'].to_cupy()
                dec = rotated_cluster['Dec'].to_cupy()
                out_ra = cp.empty_like(ra)
                out_dec = cp.empty_like(dec)
                
                apply_rotations_kernel[1, 1](ra, dec, angle, out_ra, out_dec)
                
                rotated_cluster['R.A.'] = out_ra
                rotated_cluster['Dec'] = out_dec
                rotated_cluster['window'] = max_window_num + 1
                augmented_clusters.append(rotated_cluster)

            # Rotation des AGN
            if sub_agn is not None and len(sub_agn) > 0:
                rotated_agn = sub_agn.copy()
                ra = rotated_agn['ra_mag_gal'].to_cupy()
                dec = rotated_agn['dec_mag_gal'].to_cupy()
                out_ra = cp.empty_like(ra)
                out_dec = cp.empty_like(dec)
                
                apply_rotations_kernel[1, 1](ra, dec, angle, out_ra, out_dec)
                
                rotated_agn['ra_mag_gal'] = out_ra
                rotated_agn['dec_mag_gal'] = out_dec
                rotated_agn['window'] = max_window_num + 1
                augmented_agn.append(rotated_agn)

            max_window_num += 1

        # Miroir
        mirrored_src = sub_src.copy()
        for ra_col, _ in coord_cols:
            ra = mirrored_src[ra_col].to_cupy()
            out_ra = cp.empty_like(ra)
            apply_mirror_kernel[blocks_per_grid, threads_per_block](ra, out_ra)
            mirrored_src[ra_col] = out_ra
        
        mirrored_src['window'] = max_window_num + 1
        augmented_windows.append(mirrored_src)

        if sub_cluster is not None and len(sub_cluster) > 0:
            mirrored_cluster = sub_cluster.copy()
            ra = mirrored_cluster['R.A.'].to_cupy()
            out_ra = cp.empty_like(ra)
            apply_mirror_kernel[1, 1](ra, out_ra)
            mirrored_cluster['R.A.'] = out_ra
            mirrored_cluster['window'] = max_window_num + 1
            augmented_clusters.append(mirrored_cluster)

        if sub_agn is not None and len(sub_agn) > 0:
            mirrored_agn = sub_agn.copy()
            ra = mirrored_agn['ra_mag_gal'].to_cupy()
            out_ra = cp.empty_like(ra)
            apply_mirror_kernel[1, 1](ra, out_ra)
            mirrored_agn['ra_mag_gal'] = out_ra
            mirrored_agn['window'] = max_window_num + 1
            augmented_agn.append(mirrored_agn)

        max_window_num += 1

    # Concaténation des résultats (reste sur GPU)
    def concat_gdfs(gdf_list):
        if not gdf_list:
            return cudf.DataFrame()
        return cudf.concat(gdf_list, ignore_index=True)

    return (concat_gdfs(augmented_windows),
            concat_gdfs(augmented_clusters) if augmented_clusters else cudf.DataFrame(),
            concat_gdfs(augmented_agn) if augmented_agn else cudf.DataFrame())













def compute_global_stats(data, selected_columns):
    """
    Version GPU-optimisée du calcul des statistiques globales.
    
    Args:
        data: Table Astropy ou DataFrame pandas/cuDF
        selected_columns: Liste des colonnes à analyser
        
    Returns:
        Dictionnaire des statistiques calculées
    """
    # Conversion en DataFrame cuDF si nécessaire
    if not isinstance(data, cudf.DataFrame):
        gdf = cudf.DataFrame.from_pandas(data.to_pandas() if hasattr(data, 'to_pandas') else data)
    else:
        gdf = data

    global_stats = {}

    for col in selected_columns:
        if col in gdf.columns:
            # Extraction de la colonne et suppression des NaN
            col_data = gdf[col].dropna()
            
            if len(col_data) > 0:
                # Conversion en array CuPy pour calculs accélérés
                values = col_data.to_cupy()
                
                # Calcul des statistiques de base
                col_min = float(cp.min(values))
                col_max = float(cp.max(values))
                
                # Calcul du log_min en évitant les valeurs <= 0
                positive_vals = values[values > 0]
                log_min = float(cp.log10(cp.min(positive_vals))) if len(positive_vals) > 0 else -10.0
                
                global_stats[col] = {
                    'min': col_min,
                    'max': col_max,
                    'log_min': log_min
                }

    return global_stats









def discretise_et_complete(data_ref, data, n_bins, global_stats, selected_columns, 
                             log_scale_flags, PAD_TOKEN, max_sources):
    """
    Version GPU-optimisée de la discrétisation et complétion.
    
    Args:
        data_ref: Table de référence (Astropy ou cuDF)
        data: Table à traiter (Astropy ou cuDF)
        n_bins: Nombre de bins pour discrétisation
        global_stats: Dictionnaire des statistiques précalculées
        selected_columns: Colonnes à inclure
        log_scale_flags: Indicateurs d'échelle logarithmique
        PAD_TOKEN: Valeur de padding
        max_sources: Nombre maximal de sources par fenêtre
        
    Returns:
        Array CuPy/numpy des fenêtres discrétisées
    """
    # Conversion en DataFrames GPU si nécessaire
    if not isinstance(data_ref, cudf.DataFrame):
        gdf_ref = cudf.DataFrame.from_pandas(data_ref.to_pandas())
    else:
        gdf_ref = data_ref
        
    if not isinstance(data, cudf.DataFrame):
        gdf_data = cudf.DataFrame.from_pandas(data.to_pandas())
    else:
        gdf_data = data

    # Pré-calculs initiaux sur GPU
    data_windows = set(gdf_data['window'].unique().to_array())
    ref_windows = gdf_ref['window'].unique().to_array()
    pad_length = len(selected_columns)
    n_bins_minus_1 = n_bins - 1
    
    # Préparation des paramètres de normalisation
    norm_params = []
    for col_idx, col in enumerate(selected_columns):
        stats = global_stats.get(col, {})
        use_log = log_scale_flags[col_idx] and col in global_stats
        if use_log:
            params = (cp.log10(stats['max']) - stats['log_min'] + 1e-10)
            norm_params.append(('log', stats['log_min'], params))
        elif col in global_stats:
            params = stats['max'] - stats['min'] + 1e-10
            norm_params.append(('linear', stats['min'], params))
        else:
            norm_params.append(('none',))
    
    # Noyau CUDA pour la discrétisation
    @cp.fuse()
    def discretize_value(val, norm_type, *params):
        if norm_type == 'log':
            safe_val = cp.maximum(val, 1e-10) if val <= 0 else val
            norm_val = (cp.log10(safe_val) - params[0]) / params[1]
        elif norm_type == 'linear':
            norm_val = (val - params[0]) / params[1]
        else:
            norm_val = val
        return int(cp.clip(norm_val * n_bins_minus_1, 0, n_bins_minus_1))
    
    # Traitement des fenêtres
    windows = []
    for window_id in ref_windows:
        if data_ref is not data and window_id not in data_windows:
            windows.append([[PAD_TOKEN]*pad_length]*max_sources)
            continue
            
        win_data = gdf_data[gdf_data['window'] == window_id]
        win_features = []
        
        # Traitement batch des sources
        if len(win_data) > 0:
            # Extraction des valeurs en une seule opération
            values = {col: win_data[col].to_cupy() for col in selected_columns 
                     if col in win_data.columns}
            
            # Discrétisation vectorisée
            for i in range(len(win_data)):
                src_features = []
                for col_idx, col in enumerate(selected_columns):
                    if col not in values:
                        src_features.append(PAD_TOKEN)
                        continue
                        
                    val = values[col][i]
                    if cp.isnan(val) or cp.isinf(val):
                        src_features.append(PAD_TOKEN)
                        continue
                        
                    norm_type, *params = norm_params[col_idx]
                    src_features.append(discretize_value(val, norm_type, *params))
                
                win_features.append(src_features)
        
        # Padding
        num_pad = max_sources - len(win_features)
        if num_pad > 0:
            current_len = len(win_features[0]) if win_features else pad_length
            win_features.extend([[PAD_TOKEN]*current_len for _ in range(num_pad)])
        
        windows.append(win_features)
    
    return cp.asarray(windows) if len(windows) > 0 else cp.empty((0, max_sources, pad_length))





def combine_and_flatten_with_special_tokens(windows_Xamin, windows_input_cluster, windows_input_AGN, 
                                          cls_token=CLS_TOKEN, sep_token=SEP_TOKEN, 
                                          sep_amas_token=SEP_AMAS, sep_agn_token=SEP_AGN):
    """
    Returns 2D cupy array of shape (n_windows, max_sources*n_features_Xamin + max_clusters*n_features_input_cluster + max_agn*n_features_input_agn + 2)
    Compatible with GPU using CuPy.
    """
    # Convert special tokens to cupy arrays and flatten them
    cls_token = cp.array(cls_token).flatten()
    sep_token = cp.array(sep_token).flatten()
    sep_amas_token = cp.array(sep_amas_token).flatten()
    sep_agn_token = cp.array(sep_agn_token).flatten()

    if len(windows_Xamin) != len(windows_input_cluster) or len(windows_input_AGN) != len(windows_input_cluster):
        raise ValueError("Les trois listes de fenêtres doivent avoir la même longueur.")

    result = []
    for win_xamin, win_input_cluster, win_input_AGN in zip(windows_Xamin, windows_input_cluster, windows_input_AGN):
        # Ensure inputs are cupy arrays (no conversion needed if they already are)
        win_xamin = cp.asarray(win_xamin)
        win_input_cluster = cp.asarray(win_input_cluster)
        win_input_AGN = cp.asarray(win_input_AGN)
        
        # Concatenate all components
        seq = cp.concatenate([
            cls_token,
            win_xamin.flatten(),
            sep_amas_token,
            win_input_cluster.flatten(),
            sep_agn_token,
            win_input_AGN.flatten(),
            sep_token
        ])
        
        result.append(seq)

    return result