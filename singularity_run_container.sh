#!/bin/bash
# Run me inside this directory after you've built the container.
#
# Run container with NFS and possibly GPU access.
# Assumes the container is already built and is in the current directory.
# This opens an interactive container shell on one of the SLURM worker nodes.
# Make the call to singularity exec; submit to the sbatch queue.
# singularity shell -B "$MOUNT_HOST:$MOUNT_CONTAINER" $CONTAINER

# Adding the --nv option allows for the CUDA drivers potentially installed
# within a singularity container to work with the CUDA drivers on a machine
singularity shell --nv -B "$MOUNT_HOST:$MOUNT_CONTAINER" $CONTAINER
