# HPC Usage Guide

## Basic Usage

Refer to the official doc for more info https://ucthpc.uct.ac.za/index.php/hpc-cluster/

There are 3 ways to query a job.

1. sintx : interactive, probably the easiest. Simply type `sintx` and you'd be able to use git and run scripts etc.
2. For actual jobs, you should probably use `sbatch yourscript.sh`. The script should be a slurm script. The GPU that we get allocated to is l40s. You can only use 2 GPUs max. This is the config I used for my project last year. Simply add these to the start of your script. Note that the shorter you set the time, the less you have to wait.
   ```
   #!/bin/bash
   #SBATCH --account=l40sfree
   #SBATCH --partition=l40s
   #SBATCH --job-name="ssmt-train"
   #SBATCH --nodes=1
   #SBATCH --ntasks=2
   #SBATCH --gres=gpu:2
   #SBATCH --time=04:00:00
   #SBATCH --mail-user=user@myuct.ac.za
   #SBATCH --mail-type=ALL
   #SBATCH --output=slurm-%j.out
   #SBATCH --mem-per-cpu=8G
   ```

But essentially, you get allocated a `/scratch` space and a `/home` and a home directory. Use scratch to store your data and large files. But just note that it's not backed up. So store your important stuff in home

## Sending Files

For files:

```
scp /Volumes/MAC_KIOXIA/Data/image1.png  /Volumes/MAC_KIOXIA/Data/image2.png yhxjin001@hpc.uct.ac.za:../../scratch/yhxjin001/InstaRoad
```

For entire folder:

```
scp -r /Volumes/MAC_KIOXIA/Data/imagery yhxjin001@hpc.uct.ac.za:../../scratch/yhxjin001/InstaRoad
```

## Apptainer

This should avoid all dependency issues we might face. It was painful. But now it should work. The docker container is already built and stored on Github using Github Action. So all you need to do is convert the docker container into an apptainer. Simply run the following command line:

Run this first, it's a workaround for a specific process that gets killed because it consumes too much memory.

```
APPTAINER_MKSQUASHFS_PROCS=1
export APPTAINER_MKSQUASHFS_MEM=2G
```

And then the build command

```


SINGULARITY_DOCKER_USERNAME=username SINGULARITY_DOCKER_PASSWORD=PAT_TOKEN singularity build instaroad.sif docker://ghcr.io/instaroad/instaroadprototype:latest
```

You might run out of disk space at home. If you do, run these two and then run the build command again:

```
export APPTAINER_TMPDIR=/scratch/$USER/tmp_build
export APPTAINER_CACHEDIR=/scratch/$USER/tmp_build
```
