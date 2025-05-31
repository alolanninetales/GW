#!/bin/bash
#SBATCH --job-name=flowjax_test
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/hguo/gw_project/flowMC/logs/flowjax_output.txt
# module load 2023
# module load Python/3.10.4-GCCcore-11.3.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flowmc
cd /home/hguo/gw_project/flowMC
# python -m src.flowMC.resource.nf_model.test_flowjax
python -m test.unit.test_multi_flowjax
# cd ~/gw_project/flowMC/src/flowMC/resource/nf_model
# python test_flowjax.py
