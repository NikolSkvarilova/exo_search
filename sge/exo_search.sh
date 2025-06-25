#!/bin/bash
#
#$ -N exoplanet1
#$ -o sge_out.txt
#$ -e sge_err.txt
####$ -m e
#$ -q all.q@@blade,all.q@@servers
###$ -q long.q@@blade,long.q@@servers
# step: 3->4G, 2->16G, 1->?
#$ -l ram_free=10G,tmp_free=10M
####$ -q all.q@@gpu
####$ -l gpu=1,gpu_ram=16G,ram_free=10G,tmp_free=10M
#$ -p -512
#$ -tc 400

#3*3600
ulimit -t 10800

#8*3600
#ulimit -t 28800

ulimit -v unlimited

#export SGE_TASK_ID=1

WORK_DIR=/pub/users/xskvar11/bp_code

cd $WORK_DIR || exit

hostname >../sge_logs_0-10_000/sge_out_${SGE_TASK_ID}.log

export CUDA_DIR=/usr/local/share/cuda-12.2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_DIR
export LD_LIBRARY_PATH=${CUDA_DIR}/lib64/

# general format:
# time python3.10 run.py command .... --n=$SGE_TASK_ID >>../sge_out_${SGE_TASK_ID}.log 2>../sge_err_${SGE_TASK_ID}.log

# how to run:
# cd $WORK_DIR; qsub -t1:10 exo_search.sh
