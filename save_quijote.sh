#!/bin/bash
#SBATCH -p RCIF
#SBATCH -N1
#SBATCH -n1
#SBATCH --mail-user=prabh.bhambra.12@ucl.ac.uk
#SBATCH --mail-type=ALL

source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate /home/pbhambra/pbhambra/nbodykit_env

# export GLOBUS_PROFILE=profile1 #profile1 to profile8
# globus whoami
# echo $GLOBUS_PROFILE

ep1=f4863854-3819-11eb-b171-0ee0d5d9299f # Quijote San Diego
ep2=6b6f24e6-3486-11ef-962c-453c3ae125a5 # pbhambra_hypatia
succ='SUCCEEDED'

mkdir /share/lustre/pbhambra/n_body_sims/latin_hypercube/raw
mkdir /home/pbhambra/pbhambra/projects/n_body_sims/unet_625/out
wget -P /home/pbhambra/pbhambra/projects/n_body_sims/unet_625/out https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master/latin_hypercube/latin_hypercube_params.txt

for N_ID in {1990..1999} # This will take ages to run, so split it up in batches and run multiple processess. Globus allows users to have three transfer processess per profile running at a time, so splitting this into three is a good idea.
do
  echo "Transfer starting $N_ID"
  run_task=$(globus transfer --notify failed $ep1:/Snapshots/latin_hypercube/$N_ID/ICs $ep2:/share/lustre/pbhambra/n_body_sims/latin_hypercube/raw/$N_ID/ICs --recursive)
  extract_id=${run_task:102:36}
  extract_id_corr=$(echo $extract_id)
  output=$(globus task show $extract_id_corr)
  task_state=${output:250:31}
  task_state_corr=$(echo $task_state)

  while [[ "$task_state_corr" != "$succ" ]]
  do
    sleep 10
    output=$(globus task show $extract_id_corr)
    task_state=${output:250:32}
    task_state_corr=$(echo $task_state)
  done

  for SNAP_ID in 000 001 002 003 004
  do
      run_task=$(globus transfer --notify failed $ep1:/Snapshots/latin_hypercube/$N_ID/snapdir_$SNAP_ID $ep2:/share/lustre/pbhambra/n_body_sims/latin_hypercube/raw/$N_ID/snapdir_$SNAP_ID --recursive)
      extract_id=${run_task:102:36}
      extract_id_corr=$(echo $extract_id)
      output=$(globus task show $extract_id_corr)
      task_state=${output:250:30}
      task_state_corr=$(echo $task_state)

      while [[ "$task_state_corr" != "$succ" ]]
      do
      sleep 10
      output=$(globus task show $extract_id_corr)
      task_state=${output:250:30}
      task_state_corr=$(echo $task_state)
      done
  done

  python /home/pbhambra/pbhambra/projects/n_body_sims/unet_625/save_quijote.py ${N_ID}
  rm -rf /share/lustre/pbhambra/n_body_sims/latin_hypercube/raw/$N_ID/ICs
  rm -rf /share/lustre/pbhambra/n_body_sims/latin_hypercube/raw/$N_ID/snapdir*
  echo "Transfer done $N_ID"
done