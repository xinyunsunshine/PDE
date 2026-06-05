To submit a job to normal queue with a certain config:

# For normal queue
bash ../cluster_scripts/submit-job.sh /PROJECT_ROOT/scripts/run_embodiment.sh maniskill_openvla_grpo

# For preemptable queue
bash ../cluster_scripts/submit-job-preemptable.sh /PROJECT_ROOT/scripts/run_embodiment.sh {config name}

(the config is read from config/)
