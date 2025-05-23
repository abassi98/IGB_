import numpy as np
import random
import argparse
import src.utils as ut
import multiprocessing
import time
from mpi4py import MPI
from scipy.integrate import quad, dblquad

dtype = np.float32

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=100, help="Number MLP hidden layers, i.e. depth of the network.")
    parser.add_argument('--act_func', type=str, default="Linear", choices=["Linear", "ReLU", "TanhLike", "Tanh"], help="Activation function.")
    parser.add_argument('--Vw_max', type=float, default=2.2)
    parser.add_argument('--Vw_min', type=float, default=1.8)
    parser.add_argument('--Vb_max', type=float, default=0.1)
    parser.add_argument('--Vb_min', type=float, default=0.0)
    parser.add_argument('--n_w', type=int, default=3)
    parser.add_argument('--n_b', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=100, help="Number of samples of the net ensemble.")
    parser.add_argument('--seed', type=int, default=42)

    cfg = vars(parser.parse_args())
    return cfg

def update_variance(q, Vw, Vb, act_func):
    f = lambda z: 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)*act_func(np.sqrt(q)*z)**2
    I, err = quad(f, -np.inf, np.inf)
    return Vw * I + Vb

def update_covariance(qab, qaa, qbb, Vw, Vb, act_func):
    cab = qab/np.sqrt(qaa*qbb)
    f = lambda z1,z2: 1/(2*np.pi) * np.exp(-z1**2/2-z2**2/2)*act_func(np.sqrt(qaa)*z1) * act_func(np.sqrt(qbb)*(cab*z1 + np.sqrt(1.0-cab**2)*z2))
    I, err = dblquad(f, -np.inf, np.inf, -np.inf, np.inf)
    return Vw*I + Vb

def process(args):
    # get args 
    Vb, Vw, cfg = args
    max_depth = cfg["max_depth"]
    act_func = getattr(ut,cfg["act_func"])()
    n_samples = cfg["n_samples"]
    width = 10000
    buf = np.zeros((3, max_depth, n_samples), dtype=dtype) # qaa, qbb, qab
    qaa_init = 1.0 + np.random.randn(n_samples)*np.sqrt(2.0/width)
    qbb_init = 1.0 + np.random.randn(n_samples)*np.sqrt(2.0/width)
    qab_init = np.random.randn(n_samples)/np.sqrt(width)

    for ss in range(n_samples):
        qaa = qaa_init[ss]
        qbb = qbb_init[ss]
        qab = qab_init[ss]
        for ll in range(max_depth):
            cab = qab/np.sqrt(qaa*qbb)
            assert np.abs(cab) <= 1.0 
            qab = update_covariance(qab, qaa, qbb, Vw, Vb, act_func)
            qaa = update_variance(qaa, Vw, Vb, act_func)
            qbb = update_variance(qbb, Vw, Vb, act_func)
            buf[0, ll, ss] = qaa
            buf[1, ll, ss] = qbb
            buf[2, ll, ss] = qab
            
    return buf

# def common world
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
my_size = comm.Get_size()

# program specifics
cfg = get_args()
max_depth = cfg["max_depth"]
act_func = cfg["act_func"]
n_samples = cfg["n_samples"]


n_w, n_b = cfg["n_w"], cfg["n_b"]
Vw_min, Vw_max = cfg["Vw_min"], cfg["Vw_max"]
Vb_min, Vb_max = cfg["Vb_min"], cfg["Vb_max"]
Vw_vec = np.linspace(Vw_min,Vw_max, n_w)
Vb_vec = np.linspace(Vb_min,Vb_max,n_b)


# seed
seed = cfg["seed"]
np.random.seed(seed)
seed = int(np.random.uniform(low=0, high=1e2)) * my_rank +1
random.seed(seed)
np.random.seed(seed)

# check number of cpus
if my_size < n_b * n_w:
    raise ValueError(f"Number of MPI processes ({multiprocessing.cpu_count()}) is too low. At least {n_b * n_w} necessary")


if my_rank == 0:
    # get ids corresponding to the rank
    id_vb = my_rank%n_b
    id_vw = (my_rank - id_vb)//n_b
    Vb = Vb_vec[id_vb]
    Vw = Vw_vec[id_vw]
    
    print(f"Hello from rank {my_rank} on {MPI.Get_processor_name()} with seed {seed}. I am computing Vb={Vb}, Vw={Vw}")

    # run computation
    buf = process((Vb, Vw, cfg)) # buf.shape = (2, max_depth, width, n_net_samples_per_process)
    # save 0th buffer
    save_buf = np.zeros((3, max_depth, n_samples, n_b,n_w), dtype=dtype)
    save_buf[:,:,:, 0, 0] =  buf
    # save receive other buffers
    for sub_rank in range(1,n_b * n_w):
        id_vb = sub_rank%n_b
        id_vw = (sub_rank - id_vb)//n_b
        rcv_buf = np.zeros((3, max_depth, n_samples), dtype=dtype)
        comm.Recv([rcv_buf, 3*max_depth*n_samples, MPI.FLOAT], source=sub_rank, tag=sub_rank) 
        save_buf[:,:,:,id_vb, id_vw] = rcv_buf

    # save data
    ut.save_data_dip(save_buf, f"data/{act_func}_D{cfg['max_depth']}_DIP.h5", cfg)

elif my_rank > 0 and my_rank < n_b*n_w:
    # get ids corresponding to the rank
    id_vb = my_rank%n_b
    id_vw = (my_rank - id_vb)//n_b
    Vb = Vb_vec[id_vb]
    Vw = Vw_vec[id_vw]
    
    print(f"Hello from rank {my_rank} on {MPI.Get_processor_name()} with seed {seed}. I am computing Vb={Vb}, Vw={Vw}.")

    # run computation
    buf = process((Vb, Vw, cfg)) # buf.shape = (3, max_depth, n_samples, n_net_samples_per_process)
   
    # send results to 0
    comm.Send([buf,3*max_depth*n_samples, MPI.FLOAT], dest=0, tag=my_rank)

else:
    print(f"Hello from rank {my_rank} on {MPI.Get_processor_name()} with seed {seed}. I should not even exist.")
