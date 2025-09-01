#!/bin/bash

# peugeot: 129.104.252.78
# fiat: 129.104.252.70
# niva: 129.104.252.77

MASTER_IP="129.104.252.70" # IP of Machine A (Master)
MACHINE_B_IP="129.104.252.77" # Worker 1
MACHINE_C_IP="129.104.252.78" # Worker 2
MASTER_PORT="29501" # Ensure this port is open/free on the master
NPROC_PER_NODE=1    # Number of GPUs to use on each machine
NNODES=3            # Total number of machines participating

# --- IMPORTANT: Set your actual password here ---
# --- THIS IS INSECURE. Use SSH keys if possible. ---
PASSWORD="Sm1lee_bee@2026"

# --- Ensure sshpass is installed on the machine running this script ---
# sudo apt-get install sshpass  (Debian/Ubuntu)
# sudo yum install sshpass      (CentOS/RHEL)

SCRIPT_PATH="/users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/FoodCNN/train_parallel_caching_full.py" # Your DDP Python script
OUTPUT_DIR="/users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/results_full" # Changed output dir name
BASE_DIR="/Data/nutrition5k"

# Arguments for the Python script
# Added --include_side_angles and --num_side_angles_per_dish
MODEL_ARGS="--model_name DeepConvNet --batch_size 128 --epochs 50 --save_plots --num_workers 0 --lr 1e-4 --enable_gpu_caching --include_side_angles --num_side_angles_per_dish 5"

# SSH options to avoid host key prompts (use with caution on untrusted networks)
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# Environment variables for distributed training
DIST_ENV="export NCCL_DEBUG=INFO && export NCCL_DEBUG_SUBSYS=ALL && export PYTHONUNBUFFERED=1 && "
# Using PYTHONUNBUFFERED=1 can help with seeing print outputs sooner in logs

# Activate virtual environment
ACTIVATE_VENV="source /users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/venv/bin/activate && "

# Base torch.distributed.run command
LAUNCHER="python3 -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --master_port=${MASTER_PORT}"

# Check if PASSWORD is set
if [ "${PASSWORD}" == "your_shared_password" ]; then
    echo "ERROR: Please set your actual password in the PASSWORD variable within the script."
    exit 1
fi
if ! command -v sshpass &> /dev/null
then
    echo "ERROR: sshpass could not be found. Please install it or use SSH keys."
    exit 1
fi


# Launch on Master (Machine A, rank 0)
echo "Launching on Master (Machine A: ${MASTER_IP}, Rank 0)..."
COMMAND_MASTER="${DIST_ENV} ${ACTIVATE_VENV} ${LAUNCHER} --node_rank=0 --master_addr=\"${MASTER_IP}\" ${SCRIPT_PATH} --output_dir ${OUTPUT_DIR} --base_dir ${BASE_DIR} ${MODEL_ARGS}"
sshpass -p "${PASSWORD}" ssh ${SSH_OPTS} georgii.kuznetsov@${MASTER_IP} "nohup bash -c '${COMMAND_MASTER}' > /users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/master_train_sides.log 2>&1 &"

# Give master a moment to start
sleep 10 # Increased sleep slightly just in case
                            
# Launch on Worker 1 (Machine B, rank 1)
echo "Launching on Worker (Machine B: ${MACHINE_B_IP}, Rank 1)..."
COMMAND_WORKER1="${DIST_ENV} ${ACTIVATE_VENV} ${LAUNCHER} --node_rank=1 --master_addr=\"${MASTER_IP}\" ${SCRIPT_PATH} --output_dir ${OUTPUT_DIR} --base_dir ${BASE_DIR} ${MODEL_ARGS}"
sshpass -p "${PASSWORD}" ssh ${SSH_OPTS} georgii.kuznetsov@${MACHINE_B_IP} "nohup bash -c '${COMMAND_WORKER1}' > /users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/worker1_train_sides.log 2>&1 &"

# Launch on Worker 2 (Machine C, rank 2)
echo "Launching on Worker (Machine C: ${MACHINE_C_IP}, Rank 2)..."
COMMAND_WORKER2="${DIST_ENV} ${ACTIVATE_VENV} ${LAUNCHER} --node_rank=2 --master_addr=\"${MASTER_IP}\" ${SCRIPT_PATH} --output_dir ${OUTPUT_DIR} --base_dir ${BASE_DIR} ${MODEL_ARGS}"
sshpass -p "${PASSWORD}" ssh ${SSH_OPTS} georgii.kuznetsov@${MACHINE_C_IP} "nohup bash -c '${COMMAND_WORKER2}' > /users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/worker2_train_sides.log 2>&1 &"


echo "Training jobs launched. Monitor logs on respective machines:"
echo "Master log (Rank 0): ssh georgii.kuznetsov@${MASTER_IP} tail -f /users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/master_train_sides.log"
echo "Worker 1 log (Rank 1): ssh georgii.kuznetsov@${MACHINE_B_IP} tail -f /users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/worker1_train_sides.log"
echo "Worker 2 log (Rank 2): ssh georgii.kuznetsov@${MACHINE_C_IP} tail -f /users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/worker2_train_sides.log"
echo "REMINDER: Using hardcoded passwords with sshpass is insecure. Use SSH keys for better security."