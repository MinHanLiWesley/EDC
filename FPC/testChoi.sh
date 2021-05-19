#!/usr/bin/bash
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8
#PBS -q gpu
#PBS -j oe
source ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wesley/anaconda3/lib/
cd /home/wesley/EDC/FPC/ML
echo "=========================================================="
echo "Starting on : $(date)"
echo "Running on node : $(hostname)"
echo "Current directory : $(pwd)"
echo "Current job ID : $PBS_JOBID"
echo "=========================================================="

START_TIME=$SECONDS

# NAME='0321_lowparam_bigmass2'
NAME='0323_choi_V2'
# NAME='0323_choi_V3'
# DATA='training_dataV6_mimicFPC_bigmass.csv'
# DATA='training_data_cracking_V7_3m_choi_rev.csv'
DATA='training_data_cracking_V8_3m_choi_rev_big.csv'

MODEL='ML_model_V2'

mkdir /home/wesley/EDC/FPC/ML/results/$NAME
# python -u -m $MODEL --no_gene --no_predict --name $NAME --data $DATA | tee /home/wesley/EDC/FPC/ML/results/$NAME/$NAME.log
python -u -m $MODEL --no_train --name $NAME --no_tubes --no_mass --no_pressure --no_ccl4 --no_predict  --data $DATA | tee /home/wesley/EDC/FPC/ML/results/$NAME/predict.log
# python  -u -m $MODEL --no_train --no_gene --name $NAME   --data $DATA |tee /home/wesley/EDC/FPC/ML/results/$NAME/predict2.log

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
echo "Job Ended at $(date)"
echo '======================================================='

conda deactivate
