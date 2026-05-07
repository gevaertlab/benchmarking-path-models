# 🧠 PathBench: Evaluating Vision and Pathology Foundation Models for Computational Pathology  
*A Comprehensive Benchmark Study*

## 👥 Authors

**Rohan Bareja**<sup>1</sup>, **Francisco Carrillo-Perez**<sup>1</sup>, **Yuanning Zheng**<sup>1</sup>, **Marija Pizurica**<sup>1</sup>  
**Tarak Nath Nandi**<sup>2</sup>, **Lu Tian**<sup>3</sup>,**Jeanne Shen**<sup>4</sup>, **Ravi Madduri**<sup>2</sup>, **Olivier Gevaert**<sup>1</sup>  

<sup>1</sup>Stanford Center for Biomedical Informatics Research (BMIR), Stanford University, School of Medicine  
<sup>2</sup>Data Science and Learning Division, Argonne National Laboratory  
<sup>3</sup>Department of Biomedical Data Science, Stanford University, School of Medicine, Stanford, CA, USA
<sup>4</sup>Department of Pathology, Stanford University, School of Medicine

---

## :memo:  Abstract

To advance precision medicine in pathology, robust AI-driven foundation models are increasingly needed to uncover complex patterns in large-scale pathology datasets, enabling more accurate disease detection, classification, and prognostic insights. However, despite substantial progress in deep learning and computer vision, the comparative performance and generalizability of these pathology foundation models across diverse histopathological datasets and tasks remain largely unexamined. In this study, we conduct a comprehensive benchmarking of 32 AI foundation models for computational pathology across 4 model categories - including general vision models (VM), general vision-language models (VLM), pathology-specific vision models (Path-VM), and pathology-specific vision-language models (Path-VLM), evaluated over 41 slide-level and patch-level tasks sourced from TCGA, CPTAC, external benchmarking datasets, and out-of-domain datasets. Across TCGA tasks, several pathology-specific vision models consistently ranked among the top performers, including Virchow2, Prov-GigaPath, H-optimus-0, and UNI. Extending evaluation to CPTAC and out-of-domain datasets revealed more nuanced generalization behavior, with relative model rankings exhibiting modest but consistent shifts across datasets and task categories. Pairwise statistical comparisons indicated that performance differences among the top models were often small and task dependent, highlighting broadly comparable performance rather than a single dominant model. We also show that Path-VM outperformed Path-VLM and showed competitive performance relative to VM, securing top rankings across tasks despite lacking a statistically significant edge over vision models (VM). Analyses of model and dataset scaling showed that increasing model size or pretraining dataset size did not consistently translate into improved downstream performance, particularly outside TCGA benchmarks. Finally, we demonstrate that model ensembling using a late fusion approach (fusion model), which combines predictions from multiple top-performing foundation models, yields improved aggregate performance across external datasets and tissue types, underscoring the complementary strengths learned by different models. Together, these results emphasize that generalization in computational pathology is heterogeneous and task dependent, and that factors beyond scale alone likely contribute to downstream performance, motivating further work to better understand and improve robustness across diverse tissues, datasets, and clinical settings.



---

---

## 🔬 Overview

This repository accompanies our paper:

> **Evaluating Vision and Pathology Foundation Models for Computational Pathology: A Comprehensive Benchmark Study**  
> [medRxiv preprint, May 2025](https://www.medrxiv.org/content/10.1101/2025.05.08.25327250v1)

We benchmark **32 foundation models** across **41 computational pathology tasks**, including:

- General-purpose **Vision Models (VM)**
- **Vision-Language Models (VLM)**
- **Pathology-specific Vision Models (Path-VM)**
- **Pathology-specific Vision-Language Models (Path-VLM)**

We evaluate performance across data from **TCGA**, **CPTAC**, and several **external out-of-domain** datasets. Tasks include tumor classification, molecular subtyping, tumor stage, and pathway prediction.




<img width="950" height="489" alt="Screenshot 2026-03-11 at 9 44 32 AM" src="https://github.com/user-attachments/assets/1640eb2f-a0d4-42cf-8ae5-f10c036bbbf2" />

---

## 📊 PathBench
You can explore the complete benchmark results interactively via our web portal:

[PathBench](https://pathbench.stanford.edu/)

## 📈 Key Findings

- Several pathology-specific vision foundation models consistently ranked among the top performers, including Virchow2, Prov-GigaPath, H-optimus-0, UNI, and UNI2, across a large benchmark of 41 pathology tasks spanning TCGA, CPTAC, external benchmarks, and out-of-domain datasets.

- Pathology-specific vision models (Path-VM) showed a clear advantage over pathology vision–language models (Path-VLM), while demonstrating performance comparable to general vision models (VM), with differences across model families varying by task and dataset.

- Model size and reported pretraining dataset size alone did not consistently explain downstream performance, suggesting that additional factors such as training data composition, tissue diversity, and architectural design likely influence model generalization.

- Explicit separation of TCGA and non-TCGA evaluations revealed heterogeneous generalization behavior across datasets, highlighting the importance of evaluating pathology foundation models beyond in-domain benchmarks.

- Model ensembling using a late fusion strategy improved aggregate performance across tasks and datasets, indicating that different foundation models capture complementary visual representations.

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

## Supported Models and Model Sources

| Category | Model Name | Weights / Source | SSL / Pretraining Method | Model Architecture | Parameters (M) | WSIs (M) | Patches / Image-Text Pairs (M) | Cancer / Tissue Types | Data Source | Notes |
|---|---|---|---|---|---:|---:|---:|---|---|---|
| Pathology Vision | cTransPath | https://github.com/Xiyue-Wang/TransPath | Contrastive learning / MoCo v3 | CNN + Swin Transformer | ~27 | 0.032 | 15 | 32 cancer types | TCGA, PAIP | Public weights available through repo |
| Pathology Vision | Kaiko | https://github.com/kaiko-ai/towards_large_pathology_fms | DINO | ViT-base | ~85 | 0.029 | NA | 32 cancer types | TCGA | Kaiko ViT-base used in this paper |
| Pathology Vision | HIPT | https://github.com/mahmoodlab/HIPT/tree/master/HIPT_4K/Checkpoints | DINO | ViT-small | ~21 | 0.010 | 104 | 33 cancer types | TCGA | Public checkpoint folder |
| Pathology Vision | Virchow | https://huggingface.co/paige-ai/Virchow | DINOv2 | ViT-huge | ~632 | 1.5 | 2000 | NA | Proprietary | Hugging Face model; access may be gated |
| Pathology Vision | Lunit / DinoSSLPath | https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights | DINO | ViT-small | ~21 | 0.036 | 32.5 | NA | TCGA, TULIP | Public pretrained weights |
| Pathology Vision | UNI | https://huggingface.co/MahmoodLab/UNI | DINOv2 | ViT-large | ~307 | 0.100 | 100 | 20 tissue types | Proprietary, GTEx | Hugging Face model; access may be gated |
| Pathology Vision | Hibou-B | https://huggingface.co/histai/hibou-b | DINOv2 | ViT-base | ~85 | 1.0 | 1200 | NA | Proprietary | Hugging Face model used in this paper |
| Pathology Vision | Phikon | https://huggingface.co/owkin/phikon | iBOT | ViT-base | ~85 | 0.006 | 40 | 16 cancer types | TCGA | Public Hugging Face model |
| Pathology Vision | GPFM | https://huggingface.co/majiabo/GPFM | DINOv2 | ViT-large | ~307 | 0.086 | 190 | 34 tissue types | TCGA, PAIP, 49 public datasets | Also see GitHub: https://github.com/birkhoffkiki/GPFM |
| Pathology Vision | H-optimus-0 | https://huggingface.co/bioptimus/H-optimus-0 | DINOv2 | ViT-giant | ~1100 | 0.5 | NA | NA | Proprietary | Hugging Face model |
| Pathology Vision | UNI2 | https://huggingface.co/MahmoodLab/UNI2-h | DINOv2 | ViT-huge | ~632 | 0.350 | 200 | NA | Proprietary | Hugging Face model; access may be gated |
| Pathology Vision | Phikon-v2 | https://huggingface.co/owkin/phikon-v2 | DINOv2 | ViT-large | ~307 | 0.058 | 460 | 30 cancer sites | TCGA, GTEx, CPTAC, TCIA | Public Hugging Face model |
| Pathology Vision | Virchow2 | https://huggingface.co/paige-ai/Virchow2 | DINOv2 | ViT-huge | ~632 | 3.1 | NA | 17 tissue types | Proprietary | Hugging Face model; access may be gated |
| Pathology Vision | Prov-GigaPath | https://huggingface.co/prov-gigapath/prov-gigapath | DINOv2 | ViT-giant | ~1100 | 0.171 | 1300 | 31 tissue types | Proprietary | Also see GitHub: https://github.com/prov-gigapath/prov-gigapath |
| Pathology Vision | H-optimus-mini / H0-mini | https://huggingface.co/bioptimus/H0-mini | DINOv2 distilled | ViT-base | ~85 | 0.006 | 43 | 16 tissue types | TCGA | Lightweight distilled H-optimus model |
| Pathology Vision | EXAONEPath | https://huggingface.co/LGAI-EXAONE/EXAONEPath | DINOv1 | ViT-base | ~85 | 0.030 | 285 | - | Public datasets | Public Hugging Face model |
| Pathology Vision-Language | PLIP | https://github.com/PathologyFoundation/plip | CLIP | ViT-base | ~85 | NA | 0.208 image-text pairs | Pathology | Pathology image-text pairs from Twitter, PathLAION | Public GitHub |
| Pathology Vision-Language | QuiltNet-B16 | https://huggingface.co/wisdomik/QuiltNet-B-16 | CLIP | ViT-base | ~85 | NA | 1.0 image-text pairs | Pathology | Quilt-1M / histopathology videos | Public Hugging Face model |
| Pathology Vision-Language | BiomedCLIP | https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 | CLIP | ViT-base | ~85 | NA | 14-15 image-text pairs | Biomedical | Biomedical image-text pairs | Public Hugging Face model |
| Pathology Vision-Language | MI-Zero PubMedBERT | https://github.com/mahmoodlab/MI-Zero | Contrastive learning | ViT-small | ~21 | NA | 0.033 image-text pairs | Pathology | Histopathology image-caption pairs | Public GitHub |
| Pathology Vision-Language | MI-Zero ClinicalBERT | https://github.com/mahmoodlab/MI-Zero | Contrastive learning | ViT-small | ~21 | NA | 0.033 image-text pairs | Pathology | Histopathology image-caption pairs | Public GitHub |
| Pathology Vision-Language | CONCH | https://github.com/mahmoodlab/CONCH | CoCa | ViT-base | ~85 | NA | 1.17 image-text pairs | Pathology | Histopathology caption pairs | Public GitHub; access may be gated |
| Pathology Vision-Language | TITAN | https://huggingface.co/MahmoodLab/TITAN | iBOT + CoCa | ViT-base | ~85 | 0.336 | 0.423 image-text pairs | Pathology | WSIs and image captions | Hugging Face model |
| General Vision | DINO-S16 | https://github.com/facebookresearch/dino | DINO | ViT-small / 16 | ~21 | NA | 14 ImageNet images | General images | ImageNet | Public GitHub |
| General Vision | DINO-B16 | https://github.com/facebookresearch/dino | DINO | ViT-base / 16 | ~85 | NA | 14 ImageNet images | General images | ImageNet | Public GitHub |
| General Vision | DINOv2-L | https://github.com/facebookresearch/dinov2 | DINOv2 | ViT-large | ~300 | NA | 142 LVD images | General images | LVD-142M | Public GitHub |
| General Vision | iBOT-B16 | https://github.com/bytedance/ibot#pre-trained-models | iBOT | ViT-base / 16 | ~85 | NA | ImageNet-22K | General images | ImageNet-22K | Public GitHub |
| General Vision | iBOT-L16 | https://github.com/bytedance/ibot#pre-trained-models | iBOT | ViT-large / 16 | ~307 | NA | ImageNet-22K | General images | ImageNet-22K | Public GitHub |
| General Vision-Language | CLIP-B16 | https://huggingface.co/openai/clip-vit-base-patch16 | CLIP / contrastive learning | ViT-base / 16 | ~115 | NA | 400 image-text pairs | General images + text | Web image-text pairs | Also see GitHub: https://github.com/openai/CLIP |
| General Vision-Language | BLIP-B16-14M | https://github.com/salesforce/BLIP | BLIP | ViT-base | ~85 | NA | 14 ImageNet images | General images + text | ImageNet | Public GitHub |
| General Vision-Language | ALIGN-base | https://huggingface.co/kakaobrain/align-base/tree/main | Contrastive learning | EfficientNet + BERT | ~746 | NA | 700 image-text pairs | General images + text | Web image-text pairs | Hugging Face checkpoint |
| General Vision-Language | BEiT-3-L16 | https://github.com/microsoft/unilm/tree/master/beit3 | Multimodal pretraining | ViT-large / BEiT-3 | ~307 | NA | ImageNet-21K + text | General images + text | ImageNet-21K, text corpus | Public GitHub |
|



## 💻 Computational Requirements

- Model inference time depends on the model, task, and dataset size.  
  - Typically, evaluation of a single model takes **a couple of hours** on a standard desktop with 4 GPUs or a moderately-sized CPU cluster.  
- Scripts support **parallel evaluation** via PyTorch Distributed for multi-GPU setups.  
- Exact runtime may vary depending on hardware, batch size, and data preprocessing.
  
## 📖 Citation
If you use this work in your research, please cite our preprint:

Bareja R, Carrillo-Perez F, Zheng Y, Pizurica M, Nandi TN, Shen J, Madduri R, Gevaert O.
Evaluating Vision and Pathology Foundation Models for Computational Pathology: A Comprehensive Benchmark Study.
medRxiv, 2025. https://doi.org/10.1101/2025.05.08.25327250

## 📄 License
This project is licensed under the [MIT License](https://opensource.org/license/MIT).
