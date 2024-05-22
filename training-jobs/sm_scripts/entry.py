import os
import json
import socket
import yaml

if __name__ == "__main__":
   
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))
    
    num_hosts = len(hosts)
    
    #Parse the IP address of the master node in the multiple nodes cluster of SageMaker training.
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)
    
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    # backend env config
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['FI_PROVIDER'] = 'efa'
    os.environ['NCCL_PROTO'] = 'simple'
    # os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1' # only support P4d
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['HCCL_OVER_OFI'] = '1'
    
    os.system("wandb disabled")
    # use different xxx.sh to train different model
    os.system("chmod +x ./train.sh")
    os.system("chmod +x ./s5cmd")
    os.system("/bin/bash -c ./train.sh")
