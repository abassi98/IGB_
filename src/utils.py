import numpy as np
import h5py

dtype = np.float32

class Linear:
    def __init__(self):
        pass
    def __call__(self, x):
        return x

class ReLU:
    def __init__(self):
        pass
    def __call__(self, x):
        return np.maximum(0.0,x)

class Tanh:
    def __init__(self):
        pass
    def __call__(self, x):
        return np.tanh(x)

class TanhLike:
    def __init__(self):
        pass
    def __call__(self, x):
        return np.sign(x)*(1-np.exp(-np.abs(x)))
    

class MLP:
    """
    Generic MLP 
    """
    def __init__(self,
                 hidd_layer_size,
                 num_layers,
                 act,
                 sigma_w,
                 sigma_b):
    
        self.hidd_layer_size = hidd_layer_size
        self.num_layers = num_layers
        self.act = act
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b

        ### network architecture
        self.weights = np.empty([self.hidd_layer_size,self.hidd_layer_size], dtype=dtype)
        self.biases = np.empty([self.hidd_layer_size,1], dtype=dtype)
        
    def __call__(self, x):
        self.V = np.empty([self.num_layers], dtype=dtype)
        self.Vtilde = np.empty([self.num_layers], dtype=dtype)
        self.c_mean = np.empty([self.num_layers], dtype=dtype) #  track total correlation (not averaged over dataset, only over network ensemble)
        self.c_var = np.empty([self.num_layers], dtype=dtype) #  track total correlation (not averaged over dataset, only over network ensemble)
        # compute attention scores with a MLP 
        for ii in range(self.num_layers):
            # x.shape = (width, num_data_samples)
            self.weights[:] = np.random.normal(loc=0.0, scale=self.sigma_w / np.sqrt(self.hidd_layer_size), size=(self.hidd_layer_size,self.hidd_layer_size)) # initialize weights
            self.biases[:] = np.random.normal(loc=0.0, scale=self.sigma_b, size=(self.hidd_layer_size,1)) # initialize biases
            x = np.matmul(self.weights, x) + self.biases # propagate the signal and compute pre-activation
            var_data = np.var(x, axis=1, keepdims=False)
            mean_data = np.mean(x, axis=1,keepdims=False)
            print(np.var(var_data)/np.mean(var_data)**2) # self-averaging of V
            self.V[ii] = np.mean(var_data, axis=0, keepdims=False) # compute V
            self.Vtilde[ii] = np.var(mean_data, axis=0, keepdims=False) # compute Vtilde
            # check correlations
            corr = np.triu(np.corrcoef(x, rowvar=False), k=1) # take upper triangula correlations, excluding the main diagional, putting everything else to zero
            corr = corr[corr != 0] # filter out zeros
            self.c_mean[ii] = np.mean(corr)
            self.c_var[ii] = np.var(corr)
            x = self.act(x) # compute post activation values

        return None


def save_data(buf_: np.ndarray,
                    file_path: str,
                    cfg: dict,
                    ):


    with h5py.File(file_path, 'w') as out_f:
        buf = out_f.create_dataset(
            'buf',
            shape=buf_.shape,
            chunks=True,
            dtype=np.float64,
            compression='gzip')

        # assign values
        buf[:] = buf_
      
        # buf[0] = V, buf[1] = Vtilde
        # save cfg
        for key, value in cfg.items():
            out_f.attrs[key] = value

        #out_f.flush()


def save_data_dip(buf_: np.ndarray,
                    file_path: str,
                    cfg: dict,
                    ):
    
   
    with h5py.File(file_path, 'w') as out_f:
        buf = out_f.create_dataset(
            'buf',
            shape=buf_.shape,
            chunks=True,
            dtype=np.float64,
            compression='gzip')

        # assign values
        buf[:] = buf_
      
        # buf[0] = V, buf[1] = Vtilde
        # save cfg
        for key, value in cfg.items():
            out_f.attrs[key] = value

        #out_f.flush()
        
        
def save_grads(buf_: np.ndarray,
                    file_path: str,
                    cfg: dict,
                    ):
    

    with h5py.File(file_path, 'w') as out_f:
        buf = out_f.create_dataset(
            'buf',
            shape=buf_.shape,
            chunks=True,
            dtype=np.float64,
            compression='gzip')

        # assign values
        buf[:] = buf_
      
        # buf[0] = V, buf[1] = Vtilde
        # save cfg
        for key, value in cfg.items():
            out_f.attrs[key] = value

        #out_f.flush()

def read_data(file_path):
    with h5py.File(file_path, 'r') as f:
        buf = np.array(f["buf"])
        V = buf[0,:,:,:]
        Vtilde = buf[1,:,:,:]
        c_mean = buf[2,:,:,:]
        c_var = buf[3,:,:,:]

        cfg = {}
        for key, value in f.attrs.items():  # Reading attributes
            cfg[key] = value
        
    return cfg, V, Vtilde, c_mean, c_var

def read_data_dip(file_path):
    with h5py.File(file_path, 'r') as f:
        buf = np.array(f["buf"])
        qaa = buf[0,:,:,:,:]
        qbb = buf[1,:,:,:,:]
        qab = buf[2,:,:,:,:]
       

        cfg = {}
        for key, value in f.attrs.items():  # Reading attributes
            cfg[key] = value
        
    return cfg, qaa, qbb, qab

def read_grads(file_path):
    with h5py.File(file_path, 'r') as f:
        buf = np.array(f["buf"])
        grads_mm_data = buf[0,:,:,:]
        grads_varm_data = buf[1,:,:,:]
        grads_msqm_wandb = buf[2,:,:,:]
        grads_varsqm_wandb = buf[3,:,:,:]
        q_mean = buf[4,:,:,:]
        q_var = buf[5,:,:,:]
        c_mean = buf[6,:,:,:]
        c_var = buf[7,:,:,:]

        cfg = {}
        for key, value in f.attrs.items():  # Reading attributes
            cfg[key] = value
        
    return cfg, grads_mm_data, grads_varm_data, grads_msqm_wandb, grads_varsqm_wandb, q_mean, q_var, c_mean, c_var