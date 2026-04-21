# MedSAM 2D Fine-Tuning for Meniscus Segmentation on PD Knee MRI

This repository contains a **2D MedSAM fine-tuning pipeline** for **meniscus segmentation** on **PD knee MRI scans** from the **IU Health dataset**. The project was developed as part of RA work in medical image segmentation and focuses on adapting a general medical foundation model to a small labeled clinical dataset.

The workflow starts from **DICOM MRI scans**, converts them to **NIfTI**, extracts labeled 2D slices, and fine-tunes MedSAM for meniscus segmentation. Training and evaluation were run on **Indiana University Big Red 200**, using **NVIDIA A100 GPUs** for faster experimentation on high-resolution medical images.

---

## Project Summary

The main goal of this project was to test whether **MedSAM (2D)** could be adapted to **meniscus segmentation** on a small PD knee MRI dataset.

### What was done
- Used **10 labeled PD MRI scans** from the **IU Health dataset**
- Started from **DICOM** images and converted them to **NIfTI**
- Extracted **2D positive slices** containing meniscus annotations
- Built box prompts from the ground-truth masks for MedSAM training
- Fine-tuned the model on the target dataset
- Evaluated segmentation quality using **Dice score**
- Ran training and debugging workflows on **Big Red 200**

---

## Why PD MRI?

**PD (Proton Density) MRI** was chosen because it is commonly used for **knee and meniscus assessment**. For this task, PD scans were a practical choice because they generally provide:

- good soft-tissue contrast
- clearer visibility of meniscus structure
- useful anatomical detail for segmentation experiments

This made PD MRI a better fit for meniscus-focused fine-tuning than using a more generic sequence without strong meniscus visibility.

---

## Dataset

- **Source:** IU Health dataset  
- **Imaging modality:** Knee MRI  
- **Sequence used:** PD MRI  
- **Original format:** DICOM  
- **Converted format:** NIfTI  
- **Training style:** 2D slice-based fine-tuning  

From the 10 labeled scans, positive meniscus slices were extracted and converted into training-ready samples.

### Data split used in the notebook
- **Total cases:** 10
- **Total positive slices:** 200
- **Training cases:** 8
- **Validation cases:** 2
- **Training slices:** 163
- **Validation slices:** 37

---

## Model and Training Setup

- **Base model:** MedSAM (2D)
- **Task:** Meniscus segmentation
- **Input:** 2D MRI slice with bounding-box prompt
- **Image size:** 1024 × 1024
- **Optimizer:** AdamW
- **Learning rate:** 1e-4
- **Weight decay:** 1e-4
- **Loss:** BCEWithLogitsLoss + Dice loss
- **Epochs:** 10

### Fine-tuning strategy
This implementation uses **decoder-only fine-tuning**:
- **Image encoder:** frozen
- **Prompt encoder:** frozen
- **Mask decoder:** trainable

This setup keeps training lighter while still adapting the segmentation head to the target meniscus data.

---

## Why Fine-Tuning Was Needed

The **pretrained MedSAM checkpoint did not perform well on this PD knee MRI meniscus dataset in its original form**. Initial predictions from the pretrained model were not strong enough for the target anatomy, so the model was fine-tuned on the IU Health meniscus data.

After fine-tuning, the segmentation quality improved substantially, showing that domain-specific adaptation was important for this task.

---

## Big Red 200

**Big Red 200** is Indiana University’s **high-performance computing (HPC)** system. In this project, it was used to run MedSAM training, debugging, and experiment workflows on GPU hardware.

### Why Big Red 200 was useful here
- supports GPU-based deep learning workflows
- makes training on 1024 × 1024 medical image slices much faster than a local machine
- is useful for repeated experiments, batch jobs, checkpointing, and debugging
- provides access to **NVIDIA A100 GPUs**, which are well suited for heavy deep learning workloads

### GPU used
- **NVIDIA A100**

The `bigred200/` folder contains the scripts used to run training and dataset debugging on the HPC cluster.

---

## Results

Fine-tuning led to a clear improvement over using the pretrained model directly.

### Validation performance of the fine-tuned model
- **Best validation Dice during training:** **0.7757**
- **Mean validation Dice (per slice):** **0.7894**
- **Mean validation Dice (per case):** **0.7882**

### Case-wise mean Dice on the validation set
- **AC0D5A4D78B628_SAG_PD_TSE_6_1:** **0.7735**
- **AC14D3737C0482_SAG_PD_6_1:** **0.8029**

These results show that even with a small dataset, MedSAM could be adapted to this meniscus segmentation task with reasonable performance after fine-tuning.

---

## Repository Structure

> Note: intermediate folders such as `medsam_meniscus_*` and `medsam_samples/` are not documented below because they were used for intermediate preprocessing and experimentation.

```bash
.
├── bigred200/
│   ├── checkpoints/
│   │   └── medsam_vit_b.pth
│   ├── debug_medsam.py
│   ├── medsam_run.sh
│   ├── remove_bad_npz.py
│   └── train_medsam.py
├── pretrained_models/
│   ├── medsam_vit_b.pth
│   ├── sam.ipynb
│   ├── sam2_b.pt
│   └── slice19.png
├── medsam.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

### Main files
- **`medsam.ipynb`**: notebook for preprocessing, dataset preparation, fine-tuning, visualization, and evaluation
- **`bigred200/train_medsam.py`**: main training script for HPC runs
- **`bigred200/medsam_run.sh`**: Slurm job submission script for Big Red 200
- **`bigred200/remove_bad_npz.py`**: utility script to remove problematic `.npz` samples during preprocessing
- **`bigred200/debug_medsam.py`**: debugging script added to handle data issues during HPC runs, including a corrupted `.npy` file case
- **`pretrained_models/medsam_vit_b.pth`**: pretrained MedSAM checkpoint used for initialization
- **`pretrained_models/sam2_b.pt`**: additional SAM checkpoint kept for reference/experimentation
- **`pretrained_models/sam.ipynb`**: notebook used for related model exploration
- **`pretrained_models/slice19.png`**: sample image used for quick testing/inspection

---

## Pipeline Overview

1. Load labeled PD knee MRI scans from the IU Health dataset  
2. Convert scans from **DICOM** to **NIfTI**  
3. Extract 2D slices containing meniscus annotations  
4. Resize slices and masks to the MedSAM input format  
5. Generate bounding-box prompts from the masks  
6. Fine-tune the MedSAM decoder on the training split  
7. Evaluate performance using Dice score on validation slices and cases  

---

## Notes and Limitations

- This is a **small-data experiment** based on **10 labeled scans**
- The pipeline is currently **2D**, so it does not use full 3D spatial context
- The project is intended as an implementation and adaptation study, not a final clinical system
- Broader testing on more scans would be needed for stronger generalization claims

---

## Future Improvements

Possible next steps include:
- training with more labeled IU Health scans
- comparing **2D MedSAM** with **3D segmentation models**
- improving prompt generation strategies
- adding more evaluation metrics beyond Dice
- testing robustness across additional MRI cases and protocols

---

## Acknowledgments

- **IU Health dataset** for the MRI data
- **MedSAM** for the base segmentation model
- **Indiana University Big Red 200** for HPC and GPU resources
