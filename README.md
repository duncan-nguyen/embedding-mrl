# Matryoshka Representation Learning: ACL Conference Experiments

This repository contains experimental code for ACL conference submission, comparing three approaches to training Matryoshka embedding models with flexible-capacity representations.

## Overview

**Goal**: Train embedding models that produce nested representations where truncating to smaller dimensions ([16, 32, 64, 128, 256, 512, 1024] or [16, 32, 64, 128, 256, 512, 768]) still maintains quality, enabling adaptive computation based on task requirements.

## Project Structure

```
MIPIC/
├── ESE/           # EPRESSO baseline
├── MIPIC/         # Matryoshka Information Pipeline (main contribution)
└── MRL/           # Matryoshka Representation Learning Baseline
```

Each folder contains 4 Jupyter notebooks for different model architectures:
- BAAI/bge-m3
- google-bert/bert-base-uncased
- huawei-noah/TinyBERT_General_6L_768D
- Qwen/Qwen3-Embedding-0.6B

## Experimental Approaches

### 1. ESE (Baseline)
**Method**: Self-distillation with layer-wise and dimension-wise training

**Key Features**:
- EPRESSO (Efficient Progressive Representation Exploration) combined with SimCSE
- InfoNCE loss across nested dimensions [16, 32, 64, 128, 256, 512, 1024]
- Intermediate layer supervision
- Log-based dimension weighting for progressive training

**Notebooks**: [ESE/](ESE/)

### 2. MIPIC (Matryoshka Information Pipeline) - Our method
**Method**: Multi-component alignment system with horizontal (SIA) and vertical (PIC) information flow

**Loss Components**:
1. **Horizontal Attention Alignment**: Token importance ordering consistency across dimensions
2. **Submatrix CKA Loss**: Geometric structure preservation using top-k tokens
3. **Pipeline InfoNCE**: Vertical information chaining across layers
4. **Matryoshka InfoNCE**: Standard contrastive loss across nested dimensions

**Total Loss**:
```
L_MIPIC = α * L_MRL + (1-α) * (L_SIA + L_PIC)
```

**Pipeline Architecture**: Progressive depth/width chaining
- Example: (layer_3, dim_16) → (layer_7, dim_128) → (layer_11, dim_1024)

**Notebooks**: [MIPIC/](MIPIC/)

### 3. MRL (Baseline)
**Method**: Standard Matryoshka with InfoNCE contrastive loss

**Key Features**:
- Simpler baseline approach
- InfoNCE applied across nested dimensions
- No complex alignment mechanisms

**Notebooks**: [MRL/](MRL/)

## Training Configuration

- **Task**: Unsupervised pair classification (bi-encoder, SimCSE-style)
- **Max Sequence Length**: 256 tokens
- **Batch Size**: 16
- **Epochs**: 5-8
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW with cosine learning rate schedule
- **Temperature**: 0.07 (InfoNCE)
- **Mixed Precision**: FP16 via PyTorch autocast


## Data

**Training Data**: `final_data.csv` 

**Evaluation Tasks**:

### Classification (with logistic regression probe):
- Banking77 (banking intent classification)
- Emotion (emotion detection)
- Tweet (tweet classification)

### Semantic Textual Similarity (STS):
- SICK, STS12, STSB 
- SICK-R, STS13-16 

### Pair Classification:
- MRPC 
- SciTail 
- WiC 

All datasets are referenced via Kaggle paths in the notebooks.

## Evaluation Protocol

Each model is evaluated across **7 Matryoshka dimensions** [16, 32, 64, 128, 256, 512, 1024] or [16, 32, 64, 128, 256, 512, 768] :

1. **Classification**: Logistic regression on frozen embeddings
2. **STS**: Spearman correlation on cosine similarity scores
3. **Pair Tasks**: Accuracy, F1, precision, recall, Average Precision

## Dependencies

Install required packages:
```bash
pip install -U transformers huggingface_hub tokenizers
pip install torch torchvision torchaudio
pip install scikit-learn scipy pandas numpy
pip install peft  # For LoRA fine-tuning
pip install Levenshtein
```

## Usage

1. **Open a notebook** from ESE/, MIPIC/, or MRL/ folder
2. **Configure Kaggle paths** to your datasets
3. **Run all cells** to train and evaluate the model
4. **Results** are saved to `results.json` with performance across all dimensions

## Key Technical Details

- **Mean Pooling**: Attention-masked averaging over sequence length
- **CKA Loss** (MIPIC only): Centered Kernel Alignment for representation similarity
- **Multi-GPU Support**: Student/teacher on separate GPUs when available
- **Memory Optimization**: Gradient scaling, clipping, and periodic cache clearing

## Results Format

Each notebook outputs `results.json` containing:
```json
{
  "classification": {"Banking77": {...}, "Emotion": {...}, "Tweet": {...}},
  "sts": {"SICK": {...}, "STS12": {...}, "STSB": {...}, ...},
  "pair": {"MRPC": {...}, "SciTail": {...}, "WiC": {...}}
}
```

Performance metrics are reported for each of the 7 Matryoshka dimensions.

