#!/bin/bash

#SBATCH --job-name=correctors-only-elliptic-128
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=u3013241@connect.hku.hk
#SBATCH --partition=intel
#SBATCH --qos=normal
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x.out.%j
#SBATCH --error=logs/%x.err.%j

set -e

cd "${SLURM_SUBMIT_DIR}"

echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"

module load mpich/gcc/4.1.2-gcc12.3
module load singularity

unset CC CXX FC F77 F90 MPICC MPICXX MPIFC

SANDBOX_DIR="${HOME}/Desktop/singularity-images/fenicsx-0.11.0-sdbx"
PROJECT_DIR="${HOME}/Desktop/code-projects/multiscale-methods"

mpirun -n "${SLURM_NTASKS}" \
    singularity exec -w "${SANDBOX_DIR}" \
    bash -c "
        cd '${PROJECT_DIR}'
        python run_experiment.py configs/correctors/only-elliptic-128.yaml
    "

echo "Job finish: $(date)"