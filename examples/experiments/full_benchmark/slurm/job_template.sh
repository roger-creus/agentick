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
echo "Config: {config_display} | Profile: {profile_name} | Started: $(date)"

# Write inline config to a temporary file (self-contained, no external deps)
{inline_config_block}

uv run {runner_command}
_exit=$?

{cleanup_block}

echo "=== Finished: $(date), exit=$_exit ==="
exit $_exit
