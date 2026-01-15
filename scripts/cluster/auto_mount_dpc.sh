#!/bin/bash

# Define nodes array
NODES=(
    "10.16.201.75"
)

user="jianzhnie"
# Prompt for sudo password
read -s -p "Enter sudo password for remote nodes: " sudo_password
echo

# Source and target paths
SOURCE="/llmtuner"
TARGET="/home/${user}/llmtuner"

# Function to mount on a single node
mount_on_node() {
    local node=$1
    echo "Processing node: $node"
    
    # SSH command to mount
    ssh ${user}@$node "
        if [ ! -d '$TARGET' ]; then
            echo '$sudo_password' | sudo -S mkdir -p '$TARGET'
            echo 'Created directory $TARGET on $node'
        fi
        if mountpoint -q '$TARGET'; then
            echo '$TARGET is already mounted on $node'
        else
            echo '$sudo_password' | sudo -S mount -t dpc '$SOURCE' '$TARGET'
            if [ $? -eq 0 ]; then
                echo 'Successfully mounted $SOURCE to $TARGET on $node'
            else
                echo 'Failed to mount on $node' >&2
            fi
        fi
    "
}

# Main execution
for node in "${NODES[@]}"; do
    mount_on_node "$node" &
done

# Wait for all background processes
wait
echo "All mount operations completed"