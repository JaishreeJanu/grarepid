#!/usr/bin/env bash                                     
#SBATCH --job-name=grarepid
#SBATCH --output=LID.log                       
#SBATCH --error=LID.err                        
#SBATCH --mail-user=janu@ismll.de                  
#SBATCH --partition=STUD   
#SBATCH --gres=gpu:1                                                                            
srun python3 nclideval.py
