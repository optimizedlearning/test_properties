#!/bin/bash -l

# Request 4 CPUs
#$ -pe omp 4

# Request 1 GPU
#$ -l gpus=1
#$ -l gpu_c=7.0


#specify a project (probably not necessary, so currently off)
#$ -P aclab

#merge the error and output
#$ -j y

#send email at the end
#$ -m e

# assuming we are in runs/ here:

source /projectnb/aclab/tranhp/venvs/mynewenv/bin/activate
python main.py  --e 200 
