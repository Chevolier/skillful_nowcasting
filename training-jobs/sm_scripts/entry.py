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
        
    # file_name = './accelerate_config.yaml'
    # with open(file_name) as f:
    #     doc = yaml.safe_load(f)
    # doc['machine_rank'] = host_rank
    # doc['main_process_ip'] = str(master_addr)
    # doc['num_machines'] = int(num_hosts)
    # doc['num_processes'] = int(num_hosts*int(os.environ['SM_NUM_GPUS']))
    
    # print('------DDD-------- yaml doc:',doc)
    
#     with open('./as_local_config.yaml', 'w') as f:
#         yaml.safe_dump(doc, f)
    
    os.system("wandb disabled")
    # use different xxx.sh to train different model
    os.system("chmod +x ./train.sh")
    os.system("chmod +x ./s5cmd")
    os.system("/bin/bash -c ./train.sh")
