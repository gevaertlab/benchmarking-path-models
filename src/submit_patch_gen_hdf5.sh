#!/usr/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH -c 4
#SBATCH --output=/projects/HE/logs/patch_extract_%j.log
#SBATCH --job-name=patch_ext
#SBATCH --array=0-204:20

export src_path="/projects/HE"
export wsi_path="$src_path/WSI"
export patch_path="$src_path/Patches256x256_n4000"

echo $wsi_path
echo $patch_path

python3 $src_path/code/patch_gen_hdf5.py \
        --wsi_path $wsi_path \
        --patch_path $patch_path \
        --mask_path $patch_path \
        --patch_size 256 \
        --max_patches_per_slide 4000 \
        --start ${SLURM_ARRAY_TASK_ID} \
        --end $((SLURM_ARRAY_TASK_ID+20)) \
