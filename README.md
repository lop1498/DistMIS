<!--<br/><br/><br/>-->
<!--<h1 align="center">
  Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation
  <br/><br/>
  <img src="./images/3d_unet.svg" alt="Distributing Deep Learning for 3D Medical Image Segmentation" width="300">
  <img src="./images/pipeline.png" alt="Distributing Deep Learning for 3D Medical Image Segmentation" width="350">
  <br/><br/>
  
</h1>-->

<h1 align="center">
  DistMIS
  <br>
  <small>Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation.</small>
  <!--<br>
  <img src="#" alt="XXXXXX" width="600">-->
</h1>

<p align="left">
  <a href='https://opensource.org/licenses/Apache-2.0'>
    <img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg' alt='License'/>
  </a>
  <!--<a href="#">
    <img src="https://zenodo.org/badge/XXXXXXX.svg" alt="DOI">
  </a>-->
</p>

<p align="center">
    <b>DistriMIS</b> Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation.
</p>

<p align="center">
  <a href="#data-preparation">Data Preparation</a> •
  <a href="#parallelism-modes">Parallelism Modes</a>
</p>

## Parallelism Modes

## Parallelism Modes

Additionally to the Tensorflow based open implementation of the Data and Experiment Parallelism design for Supercomputing environments, focused on Deep Learning for Image Segmentation, we have also provided an extension using Pytorch. In this addendum we provide the scripts and configurations needed to replicate the experimentation that has been done, comparing via 24 experiments the scalability of data and experiment parallelism with Pytorch.

### Data Parallelism

Similarly to the Tensorflow proposal, we have provided a *data parallel* script that replicates the model on each GPU and divides the data in chunks, each sent to one device. To do so, we have used Ray.SGD and the corresponding TorchTrainer object. The TorchTrainer object is a wrapper around torch.distributed.launch that automatically replicates the training components across different machines so that training can be executed in parallel. One of the main benefits of Ray.SGD is that we can scale up the number of workers seamlessly across multiple nodes without making any change to the code.

##### Usage:
First, a configuration JSON file must be defined to execute the script. This configuration file requires the following parameters:

- lr
  > (int) Hyperparameter that defines the step size at each iteration while moving toward a minimum of a loss function.
- epochs
  > (int) Number of epochs each model will train for.
- verbose
  >(int) Number of epochs each model will train for.
- nodes
  >  (int) Number of nodes.
- gpus
  > (int) Number of GPUs per node.
- batch_size
  > (int) Batch size handled by each replica, i.e. GPU.
- num_workers
  > (int) Number of nodes * number of GPUs per node.
- use_gpu
  > (bool) Boolean that indicates if the train is going to be done using GPU resources.

Once the configuration file is ready, we can replicate the experimentation with one node as follows. First, we initialize the ray cluster, specifying the number of gpus and cpus per gpu.

```console
foo@bar:~$ ray start --head --num-cpus=20 --num-gpus=2
```
Under the hood, TorchTrainer will create replicas of your model, each of which is managed by a Ray actor connected to the Ray cluster. Thus, we can now execute the training script, defining with flag g the number of gpus.

```console
foo@bar:~$ python multiexperiment.py -g 2
```

If we are using **multi node**, we first need to initialize a ray cluster and ray for each node with a more complex bash script. Please refer to the section [Multi-node Ray Cluster](#multi-node-ray-cluster).

### Experiment Parallelism

Similarly to the Experiment Parallelism approach presented in Tensorflow, we used the ray.tune library for experiment execution and hyperparameter tuning at any scale. Given an object *trainable* and the number of samples (experiments) that we want to make, the ray function tune.run executes the hyperparameter tuning. This function manages the experiment and provides many features such as logging, checkpointing, and early stopping. As happens in Data Parallelism, in a multinode environment we have to deal with ray initializations for each node and the Ray cluster, so the bash script gets more complex (Multi-node Ray Cluster](#multi-node-ray-cluster)). However, Experiment Parallelism with Ray.tune in one node is even easier than Data Parallelism with Ray.SGD.

##### Usage:
First, a configuration JSON file must be defined to execute the script. This configuration file requires the following parameters:

- lr
  > (int) Hyperparameter that defines the step size at each iteration while moving toward a minimum of a loss function.

- epochs
  > (int) Number of epochs each model will train for.

- verbose
  >(int) Verbose.

- nodes
  >  (int) Number of nodes.


- gpus
  > (int) Number of GPUs per node.

- batch_size
  > (int) Batch size handled by each replica, i.e. GPU.

- num_workers
  > (int) Number of nodes * number of GPUs per node.

- use_gpu
  > (bool) Boolean that indicates if the train is going to be done using GPU resources.

- multinode
  > (bool) Boolean that indicates if the train is going to be done using GPU resources.

- use_gpu
  > (bool) Boolean that indicates if the train is going to be done using GPU resources.

- cpu_per_trial
  > (int) Number of CPUs assigned to each trial.

- gpu_per_trial
  > (int)  Number of GPUs assigned to each trial.

- num_samples
  > (int) Number of experiments.

In order to execute the experiment parallelism script we first need to start a ray.cluster with the required resources. If we are using a **single node** then we can type the following command with the given cpus and gpus.

.```console
ray start --head --num-cpus=20 --num-gpus=2
```
Once the ray cluster is started, we can call our script with our configuration json file.

```console
python train.py -c ../config/2gpu/config.json
```
It is a good practice to shutdown the ray cluster when the work is done.

```console
foo@bar:~$ ray stop
```

### Multi-node Ray Cluster
In our case we are using a cluster with 4 GPUs per node, so given n GPUs for n >= 4, we are using multi-node.
If you are using **multi-node**, you need to start a ray cluster in a different way from what is shown in the previous sections. Once the cluster is initialized you can run the script as usual.
Here we show an example, a bash script to start a ray cluster using **SLURM**.

```bash
#!/bin/bash
# shellcheck disable=SC2206
# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE RUNNABLE!

#SBATCH --job-name 16_gpus
#SBATCH -D .
#SBATCH --output=./%j.out
#SBATCH --error=./%j.err

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=4
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=160
#SBATCH --gres='gpu:4'
#SBATCH --time 48:00:00
#SBATCH --exclusive

# Load modules or your own conda environment here
module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 \
openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 \
python/3.7.4_ML ray/1.4.1

# ===== DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING =====
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
  ray start --head --num-cpus 40 --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --block &
sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" ray start --num-cpus 40 --address "$ip_head" --redis-password="$redis_password" --block &
  sleep 5
done
# ===== Call your code below =====
```
With the first part of the script we have requested the resources via slurm and we have started a ray cluster. The script continues with the following lines where you can call the script you wish.
```bash
export PYTHONUNBUFFERED=1
python data_parallelism.py --config ./config.json
```
