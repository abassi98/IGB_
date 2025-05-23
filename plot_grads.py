import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, ListedColormap
import math

import src.utils as ut
plt.subplots_adjust(right=0.8)
seed = 42
np.seterr(under="warn")

def parse_list(arg):
    # Strip brackets and split by commas
    return [int(x) for x in arg.strip('[]').split(',')]

def generate_colour_shades(length, colour):
    # Ensure the length is valid
    if length <= 0:
        return []

    shades = []
    step = 255 // max(1, length )  # Step for variation in shades

    for i in range(length):
        value = (i+1) * step / 256.0  # Incrementally adjust the blue intensity
        if colour=="blue":
            shades.append((0, 0, 1.0, value))  # RGBA with full opacity (255)
        elif colour=="lightblue":
            shades.append((0.67, 0.84, 0.902, value)) 
        elif colour=="green":
            shades.append((0.0, 1.0, 0.0, value))  # RGBA with full opacity (255)
        elif colour=="red":
            shades.append((1.0, 0, 0, value))  # RGBA with full opacity (255)
        elif colour=="orange":
            shades.append((1.0, 0.5, 0, value))  # RGBA with full opacity (255)

    return shades

def g_relu(x):
    if x <=np.inf:
        a = 2.0
        val =  a/np.pi *x* np.arctan(np.sqrt(a*x+1.0))   + np.sqrt(a*x+1.0)/np.pi #- (1)/np.pi# + np.exp(x/10000.0) #- 1.0/(3.0*np.pi*np.sqrt(2*x))
        return val
    else:
        return x

def f_relu(x):
    if x.any() >=0:
        val = np.ones_like(x)  +x -g_relu(x)
        return  val
    
    else:
        raise ValueError("gamma cannot be negative")

def g_frac_f(x):
    return g_relu(x)/f_relu(x) 

def I(x):
    den = np.sqrt(np.pi*x)
    a = x * np.arctan(np.sqrt(2*x+1))
    b = x*x / ((x+1)*np.sqrt(2*x+1))
    return den * (2/np.pi*(a+b)-x/2)
#gradient = grad(g_frac_f)

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=100, help="Number MLP hidden layers, i.e. depth of the network.")
    parser.add_argument('--widths', type=parse_list, default=[1000], help="Widths to plot")
    parser.add_argument('--act_func', type=str, default="Linear", choices=["Linear", "ReLU", "TanhLike", "Tanh"], help="Activation function.")
    parser.add_argument('--plot_mode', type=str, default="plot", help="Plot type from pyplot, e.g. plot, semilog etc")
    cfg = vars(parser.parse_args())
    return cfg

def func_gamma_ther(X, Y, l):
    out = 0.0
    for k in range(l+1):
        out += 1. / X**(k+1)
    return out*Y

if __name__=="__main__":
    cfg = get_args()
    max_depth = cfg["max_depth"]
    widths = cfg["widths"]
    print(widths)
    n_widths = len(widths)
    act_func = cfg["act_func"]
    plot_mode = cfg["plot_mode"]
    random.seed(seed)
    np.random.seed(seed)
    
    
    # load experimental values
    run_cfg, grads_mm_data_prime, grads_varm_data_prime, grads_msqm_wandb_prime, grads_varsqm_wandb_prime, q_mean_prime, q_var_prime, c_mean_prime, c_var_prime = ut.read_grads(f"data/grads_{act_func}_D{max_depth}_W{widths[0]}.h5")
    n_w, n_b, Vw_range, Vb_range = run_cfg["n_w"], run_cfg["n_b"], [run_cfg["Vw_min"], run_cfg["Vw_max"]], [run_cfg["Vb_min"], run_cfg["Vb_max"]]
    grads_mm_data = np.zeros((max_depth, n_widths, n_b, n_w))
    grads_varm_data = np.zeros((max_depth, n_widths, n_b, n_w))
    grads_msqm_wandb = np.zeros((max_depth, n_widths, n_b, n_w))
    grads_varsqm_wandb =  np.zeros((max_depth, n_widths, n_b, n_w))
    q_mean = np.zeros((max_depth, n_widths, n_b, n_w))
    q_var = np.zeros((max_depth, n_widths, n_b, n_w))
    c_mean = np.zeros((max_depth, n_widths, n_b, n_w))
    c_var = np.zeros((max_depth, n_widths, n_b, n_w))
    grads_mm_data[:,0,:,:] = grads_mm_data_prime
    grads_varm_data[:,0,:,:] = grads_varm_data_prime
    grads_msqm_wandb[:,0,:,:] = grads_msqm_wandb_prime
    grads_varsqm_wandb[:,0,:,:] = grads_varsqm_wandb_prime
    q_mean[:,0,:,:] = q_mean_prime
    q_var[:,0,:,:] = q_var_prime
    c_mean[:,0,:,:] = c_mean_prime
    c_var[:,0,:,:] = c_var_prime
    
    for ww, width in enumerate(widths[1:]):
        _, grads_mm_data_prime, grads_varm_data_prime, grads_msqm_wandb_prime, grads_varsqm_wandb_prime, q_mean_prime, q_var_prime, c_mean_prime, c_var_prime  = ut.read_grads(f"data/grads_{act_func}_D{max_depth}_W{width}.h5")
        grads_mm_data[:,ww+1,:,:] = grads_mm_data_prime
        grads_varm_data[:,ww+1,:,:] = grads_varm_data_prime
        grads_msqm_wandb[:,ww+1,:,:] = grads_msqm_wandb_prime
        grads_varsqm_wandb[:,ww+1,:,:] = grads_varsqm_wandb_prime
        q_mean[:,ww+1,:,:] = q_mean_prime
        q_var[:,ww+1,:,:] = q_var_prime
        c_mean[:,ww+1,:,:] = c_mean_prime
        c_var[:,ww+1,:,:] = c_var_prime
        

    Vw_vec = np.linspace(Vw_range[0],Vw_range[1],n_w)
    Vb_vec = np.linspace(Vb_range[0],Vb_range[1],n_b)
   

    # set up figures
    fig, ax = plt.subplots(n_b,n_w,figsize=(16,9), sharex=True)
   
    blue_shades = generate_colour_shades(n_widths, colour="blue")
    lightblue_shades = generate_colour_shades(n_widths, colour="lightblue")
    red_shades = generate_colour_shades(n_widths, colour="red")
    green_shades = generate_colour_shades(n_widths, colour="green")
    plot_depth = max_depth
    depths = np.arange(1,101)
    for ii, Vb in enumerate(Vb_vec):
        for jj, Vw in enumerate(Vw_vec):
            if ii==0:
                ax[ii,jj].set_title(f"{np.round(Vw,2)}", fontsize=15)
            if jj==0:
                ax[ii,jj].set_ylabel(f"{np.round(Vb,2)}",  fontsize=15)
            # Plot experimental
            for oo, width in enumerate(widths):
                lw = 1.0 +  5* (oo +1)/ (len(widths)+1)
                # plot only last width
                if oo == len(widths)-1:
                    # mean over data
                    getattr(ax[ii,jj], plot_mode)(depths, grads_mm_data[:plot_depth,oo, ii,jj], label="Grads (mean data)", ls="dotted", c=blue_shades[oo])
                    ax[ii,jj].fill_between(depths, grads_mm_data[:plot_depth,oo, ii,jj]-np.sqrt(grads_varm_data[:plot_depth,oo, ii,jj]), grads_mm_data[:plot_depth,oo, ii,jj]+ np.sqrt(grads_varm_data[:plot_depth,oo, ii,jj]), color=blue_shades[oo], alpha=0.5)
                    # mean squared over wandb
                    getattr(ax[ii,jj], plot_mode)(depths, grads_msqm_wandb[:plot_depth,oo, ii,jj], label="Grads (squared mean wandb)", ls="dotted", c=lightblue_shades[oo])
                    ax[ii,jj].fill_between(depths, grads_msqm_wandb[:plot_depth,oo, ii,jj]-np.sqrt(grads_varsqm_wandb[:plot_depth,oo, ii,jj]), grads_msqm_wandb[:plot_depth,oo, ii,jj]+ np.sqrt(grads_varsqm_wandb[:plot_depth,oo, ii,jj]), color=lightblue_shades[oo], alpha=0.5)
                    getattr(ax[ii,jj], plot_mode)(depths, q_mean[:plot_depth,oo, ii,jj], label="q", ls="dotted", c=red_shades[oo])
                    ax[ii,jj].fill_between(depths, q_mean[:plot_depth,oo, ii,jj]-np.sqrt(q_var[:plot_depth,oo, ii,jj]), q_mean[:plot_depth,oo, ii,jj]+ np.sqrt(q_var[:plot_depth,oo, ii,jj]), color=red_shades[oo], alpha=0.5)
                    getattr(ax[ii,jj], plot_mode)(c_mean[:plot_depth,oo, ii,jj], label="c", ls="dotted", c=green_shades[oo])
                    ax[ii,jj].fill_between(depths, c_mean[:plot_depth,oo, ii,jj]- np.sqrt(c_var[:plot_depth,oo, ii,jj]), c_mean[:plot_depth,oo, ii,jj]+ np.sqrt(c_var[:plot_depth,oo, ii,jj]), color=green_shades[oo], alpha=0.5)
                    ax[ii,jj].axhline(y=1.0, color='g', linestyle='-')
            

    # save main figure
    handles, labels = ax[00,00].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    fig.legend(handles, labels, loc='center right', handletextpad=0.05, bbox_to_anchor=(0.0, 0.0, 0.95, 1.0), ncol=1,frameon=True, fontsize=20)
    fig.supxlabel("# of layers", fontsize=20)
    fig.supylabel("$V_B$", fontsize=20)
    fig.suptitle("$V_W$",fontsize=20)
    fig.tight_layout(rect=[0, 0, 0.85, 1]) 
    fig.savefig(f"figures/{plot_mode}_{act_func}_GRADS_D{max_depth}_W{max(widths)}.png", dpi=300, bbox_inches='tight')

 