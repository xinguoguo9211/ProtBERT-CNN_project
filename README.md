# ProtBERT-CNN

Frozen ProtBERT embedding layer coupled with a compact one-dimensional convolutional classifier for peptide-T-cell receptor (TCR) binding prediction.

This repository contains the official code for the manuscript **"ProtBERT embedding-layer representations support class I peptide-TCR recognition"**. The model treats the antigen peptide and the CDR3beta sequence as a joint sequence pair and predicts whether the pair binds. The ProtBERT encoder is kept fully frozen, and only the downstream CNN is optimized.

## Model architecture

1. A pretrained ProtBERT model encodes the joint peptide-CDR3beta input into 1024-dimensional residue representations.
2. No ProtBERT parameter is updated during training.
3. Two one-dimensional convolution blocks expand the channel width from 1024 to 2048 and then to 4096. Each block uses a kernel size of 3, ReLU activation, and max pooling.
4. Global average pooling feeds a fully connected layer of 64 units with ReLU and dropout of 0.5, followed by a 2-unit output layer.
5. The CNN contains 31,725,762 trainable parameters.

## Repository structure

```
ProtBERT-CNN_Project/
├── ProtBERT-CNN.py      # Training, validation, testing, and external recognition
├── requirements.txt     # Python dependencies
├── README.md
├── data/                # Place your CSV datasets here
└── output/              # Checkpoints, logs, and prediction files
```

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/xinguoguo9211/ProtBERT-CNN_Project.git
cd ProtBERT-CNN_Project
pip install -r requirements.txt
```

The script loads ProtBERT from `Rostlab/prot_bert` on the Hugging Face Hub by default. To use a local copy, pass its directory through `--plm_path`.

## Data format

Input files are CSV files with at least the following three columns.

| Column | Description |
|---|---|
| `peptide` | Antigen peptide amino acid sequence |
| `binding_TCR` | CDR3beta amino acid sequence |
| `label` | 1 for binding, 0 for non-binding |

Non-standard residues O, B, U, and Z are replaced with X automatically. Sequences are jointly tokenized with a maximum length of 512 tokens.

## Usage

### Train and evaluate on the internal balanced sets

```bash
python ProtBERT-CNN.py \
    --train_csv data/majority_model_dataset.csv \
    --test_csv data/majority_test_dataset.csv \
    --plm_path Rostlab/prot_bert \
    --output_dir output
```

The training file is split into 80 percent training and 20 percent validation rows. The checkpoint with the lowest validation loss is saved as `output/prot_bert_model.pth`.

### Run positive-only recognition on an external cohort

External sets contain verified binders only, so the reported quantity is the recognition rate rather than discrimination.

```bash
python ProtBERT-CNN.py \
    --checkpoint output/prot_bert_model.pth \
    --external_csv data/traitdb_positive.csv \
    --output_dir output
```

### Main arguments

| Argument | Default | Description |
|---|---|---|
| `--train_csv` | None | CSV for model building, split into train and validation |
| `--test_csv` | None | Balanced internal test CSV |
| `--external_csv` | None | Positive-only external CSV for recognition |
| `--checkpoint` | None | Existing CNN checkpoint for evaluation |
| `--plm_path` | Rostlab/prot_bert | Local ProtBERT directory or Hugging Face identifier |
| `--output_dir` | output | Directory for checkpoints and result files |
| `--max_length` | 512 | Maximum token length |
| `--batch_size` | 8 | Training and validation batch size |
| `--test_batch_size` | 32 | Test and external batch size |
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 0.0001 | Adam learning rate |
| `--gamma` | 0.9 | Exponential learning-rate decay per epoch |
| `--valid_size` | 0.2 | Validation fraction |
| `--seed` | 42 | Random seed |

## Training configuration

- Loss function, cross entropy
- Optimizer, Adam with learning rate 0.0001
- Learning-rate schedule, exponential decay with gamma 0.9 per epoch
- Batch size, 8
- Epochs, 10
- Decision threshold, 0.5 and never tuned on test rows
- Hardware, a single GPU without mixed precision

## Reported results

Internal balanced set.

| Metric | Validation | Test |
|---|---|---|
| AUC | 0.820 | 0.854 |
| Sensitivity | - | 0.733 |
| Specificity | - | 0.824 |
| Accuracy | - | 0.778 |
| F1 | - | 0.768 |
| MCC | - | 0.559 |

Independent positive-only external cohorts.

| Cohort | Recognition |
|---|---|
| Neoantigen set, 857 pairs | 723/857 (84.3%) |
| Clinical pulmonary nodule set, 14 pairs | 13/14 (92.9%) |
| TRAITdb set, 18,254 pairs | 15,454/18,254 (84.7%) |

Stratified recognition on the TRAITdb set reached 90.5% for MHC class I and 33.8% for MHC class II. The class II context was absent from both the model input and the training data.

## Data sources

The development corpus was curated from public databases including VDJdb, McPAS-TCR, and IEDB. External validation drew on a PanPep-derived neoantigen set, a pulmonary nodule clinical cohort mapped through TCRMatch, and TRAITdb. Please consult the manuscript for full curation and leakage audit details.

## Citation

Please cite the manuscript if you use this code.

Yang X, Sun X, Wu C, Zhou Y, Huang Z, Qiao H, Li J, Guo X, Huang J, Tao T. ProtBERT embedding-layer representations support class I peptide-TCR recognition. Citation details will be updated after publication.

## License

This project is released under the MIT License.

## Contact

Please open a GitHub issue for questions or contact the corresponding authors listed in the manuscript.
