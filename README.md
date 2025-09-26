# 🧠 PathBench: Evaluating Vision and Pathology Foundation Models for Computational Pathology  
*A Comprehensive Benchmark Study*

## 👥 Authors

**Rohan Bareja**<sup>1</sup>, **Francisco Carrillo-Perez**<sup>1</sup>, **Yuanning Zheng**<sup>1</sup>, **Marija Pizurica**<sup>1</sup>  
**Tarak Nath Nandi**<sup>2</sup>, **Jeanne Shen**<sup>3</sup>, **Ravi Madduri**<sup>2</sup>, **Olivier Gevaert**<sup>1</sup>  

<sup>1</sup>Stanford Center for Biomedical Informatics Research (BMIR), Stanford University, School of Medicine  
<sup>2</sup>Data Science and Learning Division, Argonne National Laboratory  
<sup>3</sup>Department of Pathology, Stanford University, School of Medicine

---

## :memo:  Abstract

To advance precision medicine in pathology, robust AI-driven foundation models are increasingly needed to uncover complex patterns in large-scale pathology datasets, enabling more accurate disease detection, classification, and prognostic insights. However, despite substantial progress in deep learning and computer vision, the comparative performance and generalizability of these pathology foundation models across diverse histopathological datasets and tasks remain largely unexamined. In this study, we conduct a comprehensive benchmarking of 31 AI foundation models for computational pathology, including general vision models (VM), general vision-language models (VLM), pathology-specific vision models (Path-VM), and pathology-specific vision-language models (Path-VLM), evaluated over 41 tasks sourced from TCGA, CPTAC, external benchmarking datasets, and out-of-domain datasets. Across TCGA, CPTAC, and external benchmarks, Virchow2 consistently performed at the top, alongside Prov-GigaPath, H-optimus-0, and UNI all of which ranked among the leading models. Pairwise comparisons revealed no statistically significant differences among these top models, highlighting their comparable performance and robustness across diverse histopathological tasks. We also show that Path-VM outperformed both Path-VLM and VM, securing top rankings across tasks despite lacking a statistically significant edge over vision models. Our findings reveal that model size and data size did not consistently correlate with improved performance in pathology foundation models, challenging assumptions about scaling in histopathological applications. Lastly, our study demonstrates that a fusion model, integrating top-performing foundation models, achieved superior generalization across external tasks and diverse tissues in histopathological analysis. These findings emphasize the need for further research to understand the underlying factors influencing model performance and to develop strategies that enhance the generalizability and robustness of pathology-specific vision foundation models across different tissue types and datasets.


---

---

## 🔬 Overview

This repository accompanies our paper:

> **Evaluating Vision and Pathology Foundation Models for Computational Pathology: A Comprehensive Benchmark Study**  
> [medRxiv preprint, May 2025](https://www.medrxiv.org/content/10.1101/2025.05.08.25327250v1)

We benchmark **31 foundation models** across **41 computational pathology tasks**, including:

- General-purpose **Vision Models (VM)**
- **Vision-Language Models (VLM)**
- **Pathology-specific Vision Models (Path-VM)**
- **Pathology-specific Vision-Language Models (Path-VLM)**

We evaluate performance across data from **TCGA**, **CPTAC**, and several **external out-of-domain** datasets. Tasks include tumor classification, molecular subtyping, tumor stage, and pathway prediction.


<img width="855" height="439" alt="Screenshot 2025-09-03 at 10 28 29 AM" src="https://github.com/user-attachments/assets/6ad498f1-34ab-485b-9566-a33b54101e6e" />

---

## 📊 PathBench
You can explore the complete benchmark results interactively via our web portal:

[PathBench](https://pathbench.stanford.edu/)

## 📈 Key Findings

- **Virchow2** achieved the highest performance across TCGA, CPTAC, and external datasets.
- **Path-VM** models outperformed both VLMs and general-purpose VMs on average.
- **Model size and dataset size** were not reliably associated with better performance.
- A **fusion model** combining top-performing encoders generalized best across tissue types and institutions.

---

## 📁 Repository Structure

```bash
.
├── dashboard/              # Dashboard code (e.g., Streamlit app)
├── data/                   # Data used for the dashboard (summaries, plots, results)
├── environments/           # Conda environment YAML files
│   └── linear_eval.yml     # Recommended environment for model evaluation
├── models/                 # Vision transformer model code
├── scripts/                # Linear evaluation scripts for benchmarking
├── README.md               # Project overview and setup instructions

```
---

## 🚀 Getting Started

### ⚙️ System Requirements
- Operating system(s) tested: Linux (tested on SUSE Linux Enterprise Server 15 SP6; expected to run on other modern Linux distributions such as Ubuntu 22.04 or CentOS 7)
- Dependencies: fully specified in `environments/linear_eval.yml`
- Hardware: standard x86_64 CPU; GPU recommended for faster model evaluation
- Typical install time for conda environment on a "normal" desktop computer: ~10–15 minutes

1. Clone the repository

```bash

git clone https://github.com/gevaertlab/benchmarking-path-models.git
cd benchmarking-path-models
```
---

2. Set up the Conda environment
We recommend using the provided Conda environment for reproducibility:

```bash
conda env create -f environments/linear_eval.yml
conda activate linear_eval
```

3. Patch extraction
To extract patches from whole-slide images (WSIs), please use the script src/patch_gen_hdf5.py. An example script to run the patch extraction: src/submit_patch_gen_hdf5.sh

4. Example: Run evaluation script for UNI

```
python -m torch.distributed.launch \
  --master_port $RANDOM \
  --nproc_per_node=4 \
  /home/rbareja/dino/eval_linear_uni.py \
  --patch_data_path _Patches256x256_hdf5/ \
  --train_csv_path ../tcga_cancer_metadata/brain_meta/tcga_ref_brain_IDHmut_train_fold0.csv \
  --val_csv_path ../tcga_cancer_metadata/brain_meta/tcga_ref_brain_IDHmut_val_fold0.csv \
  --test_csv_path ../tcga_cancer_metadata/brain_meta/tcga_ref_brain_IDHmut_test.csv \
  --no_aug \
  --img_size=256 \
  --max_patches_total=500 \
  --bag_size=50 \
  --test_max_patches_total=500 \
  --test_bag_size=500 \
  --output_dir ../eval_brain/IDHmut_classification/"$out_dir"/ \
  --train_from_scratch no \
  --num_workers=2 \
  --batch_size_per_gpu 16 \
  --test_batch_size_per_gpu 2 \
  --num_labels 2 \
  --arch "$arch" \
  --patch_size="$p_size" \
  --epochs 30 \
  --evaluate \
  --pretrained_weights "$p_weights" \
  > ../eval_brain/IDHmut_classification/"$out_dir"/logtesdata.txt

```

## 📖 Citation
If you use this work in your research, please cite our preprint:

Bareja R, Carrillo-Perez F, Zheng Y, Pizurica M, Nandi TN, Shen J, Madduri R, Gevaert O.
Evaluating Vision and Pathology Foundation Models for Computational Pathology: A Comprehensive Benchmark Study.
medRxiv, 2025. https://doi.org/10.1101/2025.05.08.25327250

## 📄 License
This project is licensed under the [MIT License](https://opensource.org/license/MIT).
