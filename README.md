# **README - Detecting Galaxy Clusters with Transformer Networks**

## **Overview**

This project implements a Transformer-based neural network to analyze astronomical catalogs and detect galaxy clusters and Active Galactic Nuclei (AGN) from X-ray source data (Xamin). The system processes spatial data by creating contextual windows around sources, applying data augmentation techniques, and transforming the data into tokenized sequences suitable for Transformer models.

## **Project Structure**

```text
TransformerProject/

├── src/
│   ├── Constantes.py            # Configuration parameters and constants
│   ├── Fct_tokenisation.py      # Core functions for data processing
│   ├── FiltrageDonneesEntree.py # Initial data filtering and correlation
│   ├── Tokenisation.py          # Data tokenization pipeline
│   ├── Model_XVI.py             # Transformer model implementation
│   ├── GeneratorSequence.py     # Sequence generation for predictions
│   └── submit_*.sh              # Job submission scripts for HPC
├── data/                        # Directory with the three base catalogs serving as the initial dataset
├── results/                     # Output directory for processed data and models
└── logs/                        # Log files from job executions
```
## **Key Features**

- Contextual Window Processing: Creates 3 arcmin² windows centered on X-ray sources

- Multi-catalog Integration: Combines data from X-ray sources (Xamin), galaxy clusters (AMAS), and AGN catalogs

- Data Augmentation: Applies random rotations and mirror transformations

- Custom Tokenization: Specialized sequence generation for astronomical data

- Transformer Architecture: Implements an autoregressive Transformer model

## **Installation**
Prerequisites: dependencies listed in requirements.txt


Install dependencies: 
```bash
pip install -r requirements.txt
```

## **Data Preparation Pipeline**

1. Initial Data Filtering

Execute the initial data correlation and filtering:

```bash
sbatch submit_tokenisation.sh
```

This runs:

- FiltrageDonneesEntree.py: Performs spatial matching between input catalogs and Xamin data

- Tokenisation.py: Processes the matched data into tokenized sequences

2. Model Training
Train the Transformer model:
```bash
sbatch submit_training_ModelXVI.sh
```
3. Sequence Generation
Generate predictions:

```bash
sbatch submit_GenerateSequence.sh
```

## **Configuration**

Key parameters are defined in Constantes.py:

Catalog Selection

- catalog_path_AMAS: Path to galaxy clusters catalog

- catalog_path_AGN: Path to AGN catalog

- catalog_path_aftXamin: Path to X-ray source catalog

Matching Parameters

- SEARCH_RADIUS_CLUSTER: Matching radius for clusters (20 arcsec)

- SEARCH_RADIUS_AGN: Matching radius for AGN (10 arcsec)

- LIM_FLUX_CLUSTER: Minimum flux for clusters (5×10⁻¹⁵ erg/cm²/s)

- LIM_FLUX_AGN: Minimum flux for AGN (5×10⁻¹⁵ erg/cm²/s)

Window Parameters

- WINDOW_SIZE_ARCMIN: Window size (2 arcmin)

- MAX_Xamin_PAR_FENESTRON: Maximum sources per window (2)

Transformer Configuration

- VOCAB_SIZE: Token vocabulary size (1029)

- D_MODEL: Model dimension (256)

- NUM_HEADS: Number of attention heads (8)

- NUM_LAYERS: Number of transformer layers (6)

- BATCH_SIZE: Training batch size (32)

## **Data Processing Workflow**

**Initial Filtering:**

- Matches input catalogs with Xamin sources within specified radii

- Applies flux cuts to select significant sources

- Splits data into training (Dec > 2.15°) and test sets: Splits data into training and test sets to keep approximately 80% of the data for training and 20% for testing the model.

**Window Creation:**

- For each Xamin source, creates a square window

- Includes all Xamin sources, clusters, and AGN within the window

- Recenters coordinates to (0,0) at window center

**Data Augmentation:**

- Applies random rotations (3200 per window)

- Adds mirror transformations

- Increases dataset size by ~3200x

**Tokenization:**

- Discretizes continuous parameters into integer tokens

- Normalizes values using log/linear scaling

- Pads sequences to fixed length

- Adds special tokens (CLS, SEP, etc.)

**Model Architecture**

The Transformer model (Model_XVI.py) features:

- Multi-head self-attention

- Layer normalization

- Positional embeddings

Autoregressive sequence generation

## **Output Files**

The pipeline generates:

- X_train.txt, X_test.txt: Tokenized sequences

- model_weights.h5: Trained model weights

- constantes_du_modele.json: Model configuration

- Global statistics files for each data type

- Training curves plot

## **Customization**
To modify the analysis:

Adjust parameters in Constantes.py

Update selected columns in:

- SELECTED_COLUMNS_Xamin

- SELECTED_COLUMNS_input_clusters

- SELECTED_COLUMNS_input_AGN

Correspondingly update the scaling flags:

- use_log_scale_Xamin

- use_log_scale_input_clusters

- use_log_scale_input_AGN

## Performance Notes

Training requires GPU acceleration (A100 recommended)

## Future Work
- Incorporate galaxy catalog data

- Add photometric redshift information

- Implement multi-task learning for simultaneous cluster/AGN detection

## Contact

**Author**: Sylvain HEURTIER
**Email**: sylvain.heurtier@iap.fr