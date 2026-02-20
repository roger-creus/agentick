#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
{gres_line}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --requeue
#SBATCH -o {log_dir}/slurm-{job_name}-%j.out
#SBATCH -e {log_dir}/slurm-{job_name}-%j.err

module load {modules}
conda activate {conda_env}

{env_exports}

cd {project_root}

echo "=== SLURM Job: $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "Config: {config_path} | Profile: {profile_name} | Started: $(date)"

uv run {runner_command}

echo "=== Finished: $(date), exit=$? ==="
