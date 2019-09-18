#!/bin/bash
#SBATCH -J DeepSAGPool
#SBATCH -o %x.%3a.%A.out
#SBATCH -e %x.%3a.%A.err
#SBATCH --time=9-0:00:00
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --qos=ivul
#SBATCH --mail-user=abdulellah.abualshour@kaust.edu.sa
#SBATCH --mail-type=ALL

# activate your conda env
echo "Loading anaconda..."

module purge
module load gcc
module load cuda/10.1.105
module load anaconda3
#source ~/.bashrc
source activate
source activate deepgcn

echo "...Anaconda env loaded"
python main.py
echo "...training function Done"