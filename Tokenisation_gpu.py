import cudf
import cupy as cp
from cuml.preprocessing import MinMaxScaler
import json
from astropy.io import fits

from collections import Counter
import Fct_tokenisation_gpu
from Fct_tokenisation_gpu import CreateListID_Xamin, Batisseuse2Fenetres_gpu, GardeFenestronsSousPeuples_gpu, CompteSourcesParFenetres_gpu, random_rotations_and_mirror_gpu, compute_global_stats_gpu, discretise_et_complete_gpu, combine_and_flatten_with_special_tokens_gpu, convert_numpy_types

from Constantes import NOMBRE_PHOTONS_MIN, MAX_Xamin_PAR_FENESTRON
from Constantes import path_list_ID_Xamin_AMAS, path_list_ID_Xamin_AGN
from Constantes import SELECTED_COLUMNS_Xamin, SELECTED_COLUMNS_input_clusters, SELECTED_COLUMNS_input_AGN
from Constantes import VOCAB_SIZE, PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, SEP_AMAS, SEP_AGN, NOMBRE_TOKENS_SPECIAUX
from Constantes import WINDOW_SIZE_ARCMIN, NOMBRE_PHOTONS_MIN, MAX_Xamin_PAR_FENESTRON, name_dir
from Constantes import catalog_path_aftXamin, new_catalog_path_AGN, new_catalog_path_AMAS



#///////////////////////////////////////////////////////////////////////////////////////////////////////
#                                        TOKENISATION
#///////////////////////////////////////////////////////////////////////////////////////////////////////


titre = "SÉLECTION, ROTATION ET TOKÉNISATION DES FENETRES (GPU VERSION)"
largeur = 80

print("#" * largeur)
print("#" + " " * (largeur-2) + "#")
print("#" + titre.center(largeur-2) + "#")
print("#" + " " * (largeur-2) + "#")
print("#" * largeur)

# //////////// Chargements des fichiers ////////////
print(" ")
print(f"=== Chargement des fichiers ===")


def fits_to_cudf_optimized(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[1].data # Supposons que les données sont dans la première extension
        # Conversion directe des colonnes FITS en arrays cupy
        columns = {}
        for col in data.columns:
            col_data = data[col]
            # Conversion en array cupy en passant par numpy
            columns[col.name] = cp.asarray(col_data)
        return cudf.DataFrame(columns)

data_Xamin = fits_to_cudf_optimized(catalog_path_aftXamin)
data_AMAS = fits_to_cudf_optimized(new_catalog_path_AMAS)
data_AGN = fits_to_cudf_optimized(new_catalog_path_AGN)


data_Xamin['new_ID'] = cp.arange(len(data_Xamin))
data_Xamin['Ntot'] = data_Xamin['INST0_EXP'] * data_Xamin['PNT_RATE_MOS'] + data_Xamin['INST1_EXP'] * data_Xamin['PNT_RATE_PN']
print(f"\n Nombre de sources Xamin:\nAvant la coupe sur le nombre de photons: {len(data_Xamin)}")
data_Xamin = data_Xamin[data_Xamin['Ntot']>=NOMBRE_PHOTONS_MIN]
print(f"Apres la coupe sur le nombre de photons: {len(data_Xamin)}")
print(f"\nRappel: NOMBRE_PHOTONS_MIN = {NOMBRE_PHOTONS_MIN} photons")

list_ID_Xamin_AMAS = cp.loadtxt(path_list_ID_Xamin_AMAS, dtype=int)
list_ID_Xamin_AGN = cp.loadtxt(path_list_ID_Xamin_AGN, dtype=int)

# //////////// Separation des donnees d'entrainement et test ////////////
print(" ")
print(f"=== Séparation des données d'entraînement et test ===")

DEC_LIM_FOR_TRAINING = 2.15 # en degres 

# AMAS
mask_for_training = data_AMAS['Dec'] > DEC_LIM_FOR_TRAINING
data_AMAS_train = data_AMAS[mask_for_training]
data_AMAS_test = data_AMAS[~mask_for_training]

pourcentage_train = len(data_AMAS_train)*100/len(data_AMAS)
pourcentage_test = len(data_AMAS_test)*100/len(data_AMAS)

print(f"\nNombre total d'amas: {len(data_AMAS)}")
print(f"Zone train: {len(data_AMAS_train)} >> {pourcentage_train:.1f}%")
print(f"Zone test: {len(data_AMAS_test)} >> {pourcentage_test:.1f}%")

# AGN
mask_for_training = data_AGN['dec_mag_gal'] > DEC_LIM_FOR_TRAINING
data_AGN_train = data_AGN[mask_for_training]
data_AGN_test = data_AGN[~mask_for_training]

pourcentage_train = len(data_AGN_train)*100/len(data_AGN)
pourcentage_test = len(data_AGN_test)*100/len(data_AGN)

print(f"\nNombre total d'AGN: {len(data_AGN)}")
print(f"Zone train: {len(data_AGN_train)} >> {pourcentage_train:.1f}%")
print(f"Zone test: {len(data_AGN_test)} >> {pourcentage_test:.1f}%")

# Xamin
mask_for_training = data_Xamin['PNT_DEC'] > DEC_LIM_FOR_TRAINING
data_Xamin_train = data_Xamin[mask_for_training]
data_Xamin_test = data_Xamin[~mask_for_training]

print(f"\nNombre total Xamin: {len(data_Xamin)}")
print(f"Zone train: {len(data_Xamin_train)} >> {len(data_Xamin_train)*100/len(data_Xamin):.1f}%")
print(f"Zone test: {len(data_Xamin_test)} >> {len(data_Xamin_test)*100/len(data_Xamin):.1f}%")

# //////////// Selection des fenetres ////////////
AllXaminSources = False

if AllXaminSources:
    list_ID_Xamin_train = data_Xamin_train['ID_Xamin'].values
    list_ID_Xamin_test = data_Xamin_test['ID_Xamin'].values
else:
    list_ID_Xamin_train = CreateListID_Xamin(data_Xamin_train['ID_Xamin'].values, list_ID_Xamin_AMAS)
    list_ID_Xamin_test = CreateListID_Xamin(data_Xamin_test['ID_Xamin'].values, list_ID_Xamin_AMAS)

print(f"\nJeu train: {len(list_ID_Xamin_train)}")
print(f"Jeu test: {len(list_ID_Xamin_test)}")

# //////////// Construction des fenetres ////////////
print(" ")
print(f"=== Construction des fenetres ===")

list_windows_test, info_clusters_test, info_AGN_test = Batisseuse2Fenetres_gpu(data_Xamin, 
                                                                             data_AMAS_test, 
                                                                             data_AGN_test, 
                                                                             list_ID_Xamin_test)

print(f"\n✓ Fenêtres test construites")

list_windows_train, info_clusters_train, info_AGN_train = Batisseuse2Fenetres_gpu(data_Xamin, 
                                                                                data_AMAS_train, 
                                                                                data_AGN_train, 
                                                                                list_ID_Xamin_train)

print(f"\n✓ Fenêtres train construites")

# //////////// Selection des fenetres les moins peuplees ////////////
list_windows_test, info_clusters_test, info_AGN_test = GardeFenestronsSousPeuples_gpu(list_windows_test, 
                                                                                    info_clusters_test, 
                                                                                    info_AGN_test, 
                                                                                    MAX_Xamin_PAR_FENESTRON)

print(f"\n✓ Fenêtres test reduites")

list_windows_train, info_clusters_train, info_AGN_train = GardeFenestronsSousPeuples_gpu(list_windows_train, 
                                                                                       info_clusters_train, 
                                                                                       info_AGN_train, 
                                                                                       MAX_Xamin_PAR_FENESTRON)

print(f"\n✓ Fenêtres train reduites")

max_count_sources_train = CompteSourcesParFenetres_gpu(list_windows_train)
max_count_clusters_train = CompteSourcesParFenetres_gpu(info_clusters_train)
max_count_AGN_train = CompteSourcesParFenetres_gpu(info_AGN_train)

max_count_sources_test = CompteSourcesParFenetres_gpu(list_windows_test)
max_count_clusters_test = CompteSourcesParFenetres_gpu(info_clusters_test)
max_count_AGN_test = CompteSourcesParFenetres_gpu(info_AGN_test)

MAX_SOURCES = int(max(max_count_sources_train, max_count_sources_test))
MAX_CLUSTERS = int(max(max_count_clusters_train, max_count_clusters_test))
MAX_AGN = int(max(max_count_AGN_train, max_count_AGN_test))

print(f"\nMAX_SOURCES : {MAX_SOURCES}")
print(f"MAX_CLUSTERS : {MAX_CLUSTERS}")
print(f"MAX_AGN : {MAX_AGN}")

# //////////// Rotation des fenetres ////////////
print(" ")
print(f"=== Rotation des fenetres === \n")

Nbre2Rotations = 2

list_windows_test_augm, info_clusters_test_augm, info_AGN_test_augm = random_rotations_and_mirror_gpu(list_windows_test, 
                                                                                                    info_clusters_test, 
                                                                                                    info_AGN_test, 
                                                                                                    Nbre2Rotations)

list_windows_train_augm, info_clusters_train_augm, info_AGN_train_augm = random_rotations_and_mirror_gpu(list_windows_train, 
                                                                                                       info_clusters_train, 
                                                                                                       info_AGN_train, 
                                                                                                       Nbre2Rotations)

print("\nPour les donnees d'entrainement")
print("\n=== AVANT AUGMENTATION ===")
print(f"Nombre de fenêtres: {len(list_windows_train['window'].unique())}")
print(f"Nombre moyen de sources par fenêtre: ~ {len(list_windows_train)/len(list_windows_train['window'].unique()):.0f}")

print("\n=== APRÈS AUGMENTATION ===")
print(f"Nombre de fenêtres: {len(list_windows_train_augm['window'].unique())}")
print(f"\nFacteur d'augmentation: x{len(list_windows_train_augm)/len(list_windows_train)}")

# //////////// Statistiques globales ////////////
global_stats_Xamin = compute_global_stats_gpu(cudf.concat([list_windows_train_augm, list_windows_test_augm]), SELECTED_COLUMNS_Xamin)
global_stats_input_clusters = compute_global_stats_gpu(cudf.concat([info_clusters_train_augm, info_clusters_test_augm]), SELECTED_COLUMNS_input_clusters)
global_stats_input_AGN = compute_global_stats_gpu(cudf.concat([info_AGN_train_augm, info_AGN_test_augm]), SELECTED_COLUMNS_input_AGN)

# //////////// Discretisation des donnees ////////////
print(" ")
print(f"=== Discretisation des donnees ===")

windows_test = discretise_et_complete_gpu(list_windows_test_augm, list_windows_test_augm, int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX), global_stats_Xamin, SELECTED_COLUMNS_Xamin, use_log_scale_Xamin, PAD_TOKEN, MAX_SOURCES)
ClustersInWindows_test = discretise_et_complete_gpu(list_windows_test_augm, info_clusters_test_augm, int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX), global_stats_input_clusters, SELECTED_COLUMNS_input_clusters, use_log_scale_input_clusters, PAD_TOKEN, MAX_CLUSTERS)
AGNInWindows_test = discretise_et_complete_gpu(list_windows_test_augm, info_AGN_test_augm, int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX), global_stats_input_AGN, SELECTED_COLUMNS_input_AGN, use_log_scale_input_AGN, PAD_TOKEN, MAX_AGN)

windows_train = discretise_et_complete_gpu(list_windows_train_augm, list_windows_train_augm, int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX), global_stats_Xamin, SELECTED_COLUMNS_Xamin, use_log_scale_Xamin, PAD_TOKEN, MAX_SOURCES)
ClustersInWindows_train = discretise_et_complete_gpu(list_windows_train_augm, info_clusters_train_augm, int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX), global_stats_input_clusters, SELECTED_COLUMNS_input_clusters, use_log_scale_input_clusters, PAD_TOKEN, MAX_CLUSTERS)
AGNInWindows_train = discretise_et_complete_gpu(list_windows_train_augm, info_AGN_train_augm, int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX), global_stats_input_AGN, SELECTED_COLUMNS_input_AGN, use_log_scale_input_AGN, PAD_TOKEN, MAX_AGN)

print("\n=== TRAIN ===")
print(f"len(windows_train) = {len(windows_train)}")
print(f"len(ClustersInWindows_train) = {len(ClustersInWindows_train)}")
print(f"len(AGNInWindows_train) = {len(AGNInWindows_train)}")

print("\n=== TEST ===")
print(f"len(windows_test) = {len(windows_test)}")
print(f"len(ClustersInWindows_test) = {len(ClustersInWindows_test)}")
print(f"len(AGNInWindows_test) = {len(AGNInWindows_test)}")

# //////////// Concatenation des donnees ////////////
print(" ")
print(f"=== Concatenation des donnees ===")

X_train = combine_and_flatten_with_special_tokens_gpu(windows_train, ClustersInWindows_train, AGNInWindows_train)
X_test = combine_and_flatten_with_special_tokens_gpu(windows_test, ClustersInWindows_test, AGNInWindows_test)

print(f"\nDim de X_train: {X_train.shape}")
print(f"Dim de X_test: {X_test.shape}")

# //////////// Sauvegarde des donnees ////////////
print(" ")
print(f"=== Sauvegarde des donnees ===")

# Convertir en numpy avant sauvegarde si nécessaire
cp.savetxt(f'/lustre/fsstor/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_train.txt', X_train.get(), fmt='%d')
cp.savetxt(f'/lustre/fsstor/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_test.txt', X_test.get(), fmt='%d')

# Création du dictionnaire
constantes_du_modele = {
    "VOCAB_SIZE": VOCAB_SIZE,
    "PAD_TOKEN": PAD_TOKEN,
    "SEP_TOKEN": SEP_TOKEN,
    "CLS_TOKEN": CLS_TOKEN,
    "SEP_AMAS": SEP_AMAS,
    "SEP_AGN": SEP_AGN,
    "NOMBRE_TOKENS_SPECIAUX": NOMBRE_TOKENS_SPECIAUX,
    "MAX_SOURCES": MAX_SOURCES,
    "MAX_CLUSTERS": MAX_CLUSTERS,
    "MAX_AGN": MAX_AGN
}

# Sauvegarde en JSON
save_path = f"/lustre/fsstor/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/constantes_du_modele.json"
with open(save_path, 'w') as f:
    json.dump(constantes_du_modele, f, indent=4)

print(f"Dictionnaire sauvegardé dans {save_path}")

# Sauvegarde des statistiques globales
save_path = f"/lustre/fsstor/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/global_stats_Xamin.json"
with open(save_path, 'w') as f:
    json.dump(convert_numpy_types(global_stats_Xamin), f, indent=4)
print(f"Dictionnaire sauvegardé dans {save_path}")

save_path = f"/lustre/fsstor/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/global_stats_input_clusters.json"
with open(save_path, 'w') as f:
    json.dump(convert_numpy_types(global_stats_input_clusters), f, indent=4)
print(f"Dictionnaire sauvegardé dans {save_path}")

save_path = f"/lustre/fsstor/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/global_stats_input_AGN.json"
with open(save_path, 'w') as f:
    json.dump(convert_numpy_types(global_stats_input_AGN), f, indent=4, ensure_ascii=False)
print(f"Dictionnaire sauvegardé dans {save_path}")

print("\n   ***   THE END   ***   \n")