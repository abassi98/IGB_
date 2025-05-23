import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import src.utils as ut
import multiprocessing
import time
from mpi4py import MPI

dtype = np.float32


class MLP(nn.Module):
    def __init__(self, 
                hidd_layer_size,
                num_layers,
                act,
                sigma_w,
                sigma_b):
        super(MLP, self).__init__()
        self.hidd_layer_size = hidd_layer_size
        self.num_layers = num_layers
        self.act = act
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b

        # Hidden layers
        layers=[]
        for _ in range(self.num_layers):
            layers.append(nn.Linear(self.hidd_layer_size, self.hidd_layer_size))
        self.layers = nn.Sequential(*layers)

        # initialize
        self.initialize()

        self.pre_activations = []


    def initialize(self):
        
        for l in self.layers:
            nn.init.normal_(l.weight, mean=0.0, std=self.sigma_w/np.sqrt(self.hidd_layer_size))
            nn.init.normal_(l.bias, mean=0.0, std=self.sigma_b)

    def forward(self, x):
        self.pre_activations.clear()
        for l in self.layers[:-1]:
            x = l(x)
            x.retain_grad()
            self.pre_activations.append(x) 
            x = self.act(x)

        return self.layers[-1](x)

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=100, help="Number MLP hidden layers, i.e. depth of the network.")
    parser.add_argument('--width', type=int, default=1000, help="Number MLP neurons per layer, i.e. width.")
    parser.add_argument('--act_func', type=str, default="Linear", choices=["Linear", "ReLU", "TanhLike", "Tanh"], help="Activation function.")
    parser.add_argument('--Vw_max', type=float, default=1.5)
    parser.add_argument('--Vw_min', type=float, default=0.5)
    parser.add_argument('--Vb_max', type=float, default=1.0)
    parser.add_argument('--Vb_min', type=float, default=0.0)
    parser.add_argument('--n_w', type=int, default=1)
    parser.add_argument('--n_b', type=int, default=1)
    parser.add_argument('--n_net_samples', type=int, default=100, help="Number of samples of the net ensemble.")
    parser.add_argument('--n_net_processes', type=int, default=1, help="Number processes over which net samples computation is splitted.")
    parser.add_argument('--n_data_samples', type=int, default=100, help="Number of samples of the dataset.")
    parser.add_argument('--seed', type=int, default=42)

    cfg = vars(parser.parse_args())
    return cfg


def process(args):
    # get args 
    Vb, Vw, n_net_samples_per_process, cfg = args
    width = cfg["width"]
    max_depth = cfg["max_depth"]
    act_func = cfg["act_func"]
    n_data_samples = cfg["n_data_samples"]
    buf = np.zeros((max_depth, n_net_samples_per_process), dtype=dtype) # V, V_tilde the first dimension
    data = torch.randn(n_data_samples,width, requires_grad=True)
   
    for kk in range(n_net_samples_per_process):
        model = MLP(hidd_layer_size=width,
                    num_layers=max_depth+1,
                    act=getattr(nn, act_func)(),
                    sigma_w=np.sqrt(Vw),
                    sigma_b=np.sqrt(Vb))

        # compute output
        out = model(data[0,:])
        out_label = torch.zeros_like(out)
        loss = nn.MSELoss()(out, out_label)
        loss.backward()

        grads = []
        for pa in model.pre_activations:
            grads.append(pa.grad.numpy())
        buf[:,kk] = np.mean(np.array(grads).astype(dtype)**2, axis=1, keepdims=False)[::-1]
       
    return buf

# def common world
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
my_size = comm.Get_size()

# program specifics
cfg = get_args()
max_depth = cfg["max_depth"]
width = cfg["width"]
act_func = cfg["act_func"]
n_net_samples = cfg["n_net_samples"]



n_w, n_b, n_net_processes = cfg["n_w"], cfg["n_b"], cfg["n_net_processes"]
Vw_min, Vw_max = cfg["Vw_min"], cfg["Vw_max"]
Vb_min, Vb_max = cfg["Vb_min"], cfg["Vb_max"]
Vw_vec = np.linspace(Vw_min,Vw_max, n_w)
Vb_vec = np.linspace(Vb_min,Vb_max,n_b)

# define mpi variables
my_colour = my_rank // n_net_processes # every contigous number of n_net_processes (according to my_rank), i.e. colour, processes same input
sub_comm = comm.Split(my_colour, 0)
my_sub_size = sub_comm.Get_size()
my_sub_rank = sub_comm.Get_rank()


# seed
seed = cfg["seed"]
np.random.seed(seed)
seed = int(np.random.uniform(low=0, high=1e2)) * my_rank +1
random.seed(seed)
np.random.seed(seed)

# check number of cpus
if my_size < n_b * n_w * n_net_processes:
    raise ValueError(f"Number of MPI processes ({multiprocessing.cpu_count()}) is too low. At least {n_b * n_w * n_net_processes} necessary")


# check the number of net processes fed to each process
if n_net_samples%n_net_processes != 0:
    raise ValueError("The number of net sample processes must be divisor of the total number of net samples.")

# define number of net samples per process
n_net_samples_per_process = n_net_samples // n_net_processes


if my_colour == 0:
    # get ids corresponding to the rank
    id_vb = my_colour%n_b
    id_vw = (my_colour - id_vb)//n_b
    Vb = Vb_vec[id_vb]
    Vw = Vw_vec[id_vw]
    
    print(f"Hello from rank {my_rank}, colour {my_colour} with sub-rank {my_sub_rank} on {MPI.Get_processor_name()} with seed {seed}. I am computing Vb={Vb}, Vw={Vw} and width = {width}")

    # run computation
    buf = process((Vb, Vw, n_net_samples_per_process, cfg)) # buf.shape = (2, max_depth, width, n_net_samples_per_process)
    # compute local sums
    local_sum = np.sum(buf, axis=(1), keepdims=False)
    #local_sum = np.ascontiguousarray(local_sum)
    # aggregate
    global_sum = np.zeros((max_depth), dtype=dtype)
    sub_comm.Reduce(local_sum, (global_sum, max_depth, MPI.FLOAT), op=MPI.SUM, root=0)
    
    # aggregation of the result for each sub communicator
    if my_sub_rank==0: # colour 0 process 0 (i.e. global 0)
        # save 0th buffer
        save_buf = np.zeros((max_depth, n_b,n_w), dtype=dtype)
        save_buf[:, 0, 0] =  global_sum / (n_net_samples) 
        # save receive other buffers
        for sub_colour in range(1,n_b * n_w):
            id_vb = sub_colour%n_b
            id_vw = (sub_colour - id_vb)//n_b
            rcv_buf = np.zeros((max_depth), dtype=dtype)
            comm.Recv([rcv_buf, max_depth, MPI.FLOAT], source=sub_colour*n_net_processes, tag=sub_colour) 
            save_buf[:,id_vb, id_vw] = rcv_buf

        # save data
        ut.save_grad(save_buf, f"data/grad_{act_func}_D{cfg['max_depth']}_W{cfg['width']}.h5", cfg)

elif my_colour >0 and my_colour < n_b*n_w:
    # get ids corresponding to the rank
    id_vb = my_colour%n_b
    id_vw = (my_colour - id_vb)//n_b
    Vb = Vb_vec[id_vb]
    Vw = Vw_vec[id_vw]
    
    print(f"Hello from rank {my_rank}, colour {my_colour} with sub-rank {my_sub_rank} on {MPI.Get_processor_name()} with seed {seed}. I am computing Vb={Vb}, Vw={Vw} and width = {width}")

    # run computation
    buf = process((Vb, Vw, n_net_samples_per_process, cfg)) # buf.shape = (2, max_depth, width, n_net_samples_per_process)
    # compute local sums
    local_sum = np.sum(buf, axis=(2), keepdims=False)
    # aggregate
    global_sum = np.zeros((max_depth), dtype=dtype)
    sub_comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)
    global_mean = global_sum / (n_net_samples) 

    # aggregation of the result for each sub communicator
    if my_sub_rank==0:
        comm.Send([global_mean,max_depth, MPI.FLOAT], dest=0, tag=my_colour)

else:
    print(f"Hello from rank {my_rank}, colour {my_colour} with sub-rank {my_sub_rank} on {MPI.Get_processor_name()} with seed {seed}. I should not even exist.")
