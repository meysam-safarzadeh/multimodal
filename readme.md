# Attention-Based Bottleneck Fusion Model

This repository provides the implementation of an **Attention-Based Bottleneck Fusion** model, designed for multi-modal data fusion. The model is particularly tailored for predicting pain intensity levels using diverse modalities: **facial action units (FAU)**, **thermal data**, and **depth data**. By leveraging a transformer-based architecture, the model incorporates a bottleneck fusion layer to effectively combine and process multi-modal features.

## Project Overview

Pain intensity prediction is a critical task that requires robust feature extraction and fusion from diverse data sources. This model uses a **transformer-inspired architecture** to integrate multi-modal embeddings, ensuring attention-based feature prioritization and efficient fusion.

### Key Features
- **Multi-modal data support:** Integrates FAU, thermal, and depth data.
- **Transformer architecture:** Employs attention mechanisms for feature extraction and fusion.
- **Bottleneck fusion:** Optimized for combining features while reducing dimensionality.
- **Scalable design:** Capable of handling single or multiple modalities during training.

---

## Repository Structure

- **`train.py`**: Core script for training the multi-modal fusion model. Includes initialization, dataset handling, and training/validation loops.
- **`train_single_modality.py`**: A script for single-modality training. Specify the modality (FAU, thermal, or depth) using the `modality` argument.
- **`embeddings_fau/FAU_embeddings_with_labels.csv`**: Contains pre-extracted FAU embeddings with associated labels for training and evaluation.
- **`embeddings_thermal/Thermal_embeddings_and_filenames_new.npz`**: Thermal embeddings and corresponding filenames.
- **`embeddings_depth/Depth_embeddings_and_filenames_new.npz`**: Depth embeddings and corresponding filenames.
- **`cross_validation_split_2.csv`**: Provides cross-validation splits for robust evaluation.

---

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/meysam-safarzadeh/multimodal.git
   ```
2. Navigate to the project directory:
   ```bash
   cd multimodal
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Multi-Modality Training
To train the model with all modalities, run:
```bash
python train.py --config config.json
```
Example parameters in `config.json` include:
- `hidden_dim`: Size of hidden layers.
- `num_heads`: Number of attention heads.
- `num_layers`: Transformer layers.
- `learning_rate`: Learning rate for optimization.
- `dropout_rate`: Dropout to prevent overfitting.

### Single-Modality Training
For training on a single modality:
```bash
python train_single_modality.py
```

---

## Data Preparation
Ensure the embeddings are correctly placed in their respective directories:
- FAU embeddings: `embeddings_fau/`
- Thermal embeddings: `embeddings_thermal/`
- Depth embeddings: `embeddings_depth/`

Update the paths in the `config.json` file if necessary.

---

## Contribution
We welcome contributions! Please fork this repository and submit a pull request with detailed information about your changes.

---

## License
This project is licensed under the [MIT License](LICENSE).
```

You can copy and save this as `README.md` in your project repository. Let me know if further edits are needed!