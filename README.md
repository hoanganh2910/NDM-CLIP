# NDM-CLIP: Normalized Distribution Matching and Masked-Text Prediction for Text-Based Person Search

This repository contains the official implementation of the paper **"NDM-CLIP: Normalized Distribution Matching and Masked-Text Prediction for Text-Based Person Search"** (APSIPA ASC 2026), by Hoang-Anh Le, Thi-Lan Le, and Thi-Ngoc-Diep Do (Hanoi University of Science and Technology - HUST).

## Abstract

Text-based person search (TBPS) requires precise alignment between visual appearances and natural language descriptions. While models like CLIP provide strong global representations, they struggle with fine-grained local details and textual noise. To address this, we propose **NDM-CLIP**, a unified coarse-to-fine framework:
1. **Global Alignment**: Employs a Normalized Image-Text Contrastive (N-ITC) loss to stabilize the global feature space.
2. **Local Token Filtering**: Utilizes a Gradient-Attention mechanism (GASS) to dynamically identify and mask noisy tokens.
3. **Robust Fine-Grained Alignment**: Introduces a **reversed Similarity Distribution Matching (SDM)** loss that firmly anchors optimization to true positive labels, insulating the training process from mask-induced noise.
4. **Local Feature Reconstruction**: Combined with Masked-Text Prediction (MTP) to explicitly strengthen local visual-semantic associations.

Extensive experiments demonstrate the superiority of NDM-CLIP, particularly under constrained computational resources. Notably, NDM-CLIP achieves **69.76% Rank@1** on the full CUHK-PEDES dataset, and exhibits exceptional efficiency in data-scarce settings, attaining 51.81% Rank@1 when trained on just 10% of CUHK-PEDES and 58.81% on the challenging 3000Vn-V2E benchmark.

## Repository Structure

- `model.py`: Model definition including TBPSLightning for PyTorch Lightning integration.
- `dataset.py`: PyTorch Datasets and PyTorch Lightning DataModule for TBPS.
- `train.py`: Main entry point for training the model.
- `interactive.py`: Script for interactive inference and evaluation.
- `augmentations.py`: Image and text augmentation utilities.
- `visualize.py` / `visualize_gass.py`: Visualization scripts for GASS token importance and attention heatmaps.
- `utils.py`: Utility functions and global configurations (e.g., hyperparameters).
- `check_cuhk_data.py`, `preprocess.py`: Data preprocessing and format checking.
- `tbps/`: Core training losses (N-ITC, SDM, MTP) and modeling utilities.

## Installation

Ensure you have Python 3.8+ installed. Install the dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dependencies
The main libraries used in this project are:
- `torch` & `torchvision`
- `pytorch-lightning`
- `transformers`
- `numpy`
- `Pillow`
- `matplotlib`
- `wandb`

## Datasets

The model is evaluated on two TBPS benchmarks:
1. **CUHK-PEDES**: English, high-resource dataset.
2. **3000Vn-V2E**: English-translated version of the Vietnamese 3000VnPersonSearch dataset.

Ensure you prepare the datasets and update the root paths in `utils.py` before running the training script.

## Usage

### Training

NDM-CLIP is implemented in PyTorch using the CLIP ViT-B/16 backbone. The training configuration is optimized for constrained resources (e.g., NVIDIA RTX 3060).

To train the model, run:

```bash
python train.py
```
*Note: The training process uses Weights & Biases (`wandb`) for logging.*

### Interactive Inference

To test a trained checkpoint interactively with custom text queries and images, run:

```bash
python interactive.py
```

### Visualization

You can generate visualizations of token-wise weight scores and attention maps, demonstrating the precise localization of fine-grained visual attributes:

```bash
python visualize_gass.py
```

## Citation

If you find this code or our paper useful for your research, please cite our work:

```bibtex
@inproceedings{le2026ndmclip,
  title={NDM-CLIP: Normalized Distribution Matching and Masked-Text Prediction for Text-Based Person Search},
  author={Le, Hoang-Anh and Le, Thi-Lan and Do, Thi-Ngoc-Diep},
  booktitle={2026 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  year={2026}
}
```
