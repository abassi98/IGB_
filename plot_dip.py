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
        elif colour=="green":
            shades.append((0.0, 1.0, 0.0, value))  # RGBA with full opacity (255)
        elif colour=="red":
            shades.append((1.0, 0, 0, value))  # RGBA with full opacity (255)
        elif colour=="orange":
            shades.append((1.0, 0.5, 0, value))  # RGBA with full opacity (255)

    return shades

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
    parser.add_argument('--plot_mode', type=str, default="plot", help="Plot type from pyplot, e.g. plot, semilog etc")
    cfg = vars(parser.parse_args())
    return cfg

if __name__=="__main__":
    cfg = get_args()
    max_depth = cfg["max_depth"]
    act_func = cfg["act_func"]
    plot_mode = cfg["plot_mode"]
    random.seed(seed)
    np.random.seed(seed)
    
    # load theoretical values (DIP)
    run_cfg, qaa, qbb, qab = ut.read_data_dip(f"data/{act_func}_D{cfg['max_depth']}_DIP.h5")
    cab = qab / np.sqrt(qaa*qbb)
    n_samples = run_cfg["n_samples"]

    n_w, n_b = run_cfg["n_w"], run_cfg["n_b"]
    Vw_max, Vw_min = run_cfg["Vw_max"], run_cfg["Vw_min"]
    Vb_max, Vb_min = run_cfg["Vb_max"], run_cfg["Vb_min"]
    Vw_vec = np.linspace(Vw_min,Vw_max,n_w)
    Vb_vec = np.linspace(Vb_min,Vb_max,n_b)
   

    # set up figures
    fig, ax = plt.subplots(n_b,n_w,figsize=(16,9), sharex=True, sharey=True)
   
    plot_depth = max_depth
    depths = np.arange(1,max_depth+1)
    for ii, Vb in enumerate(Vb_vec):
        for jj, Vw in enumerate(Vw_vec):
            if ii==0:
                ax[ii,jj].set_title(f"{np.round(Vw,2)}", fontsize=15)
            if jj==0:
                ax[ii,jj].set_ylabel(f"{np.round(Vb,2)}",  fontsize=15)
            # Plot theoretical
            mean_cab = np.mean(cab, axis=1, keepdims=False)
            var_cab = np.var(cab, axis=1, keepdims=False)
            #for ss in range(n_samples):
            #getattr(ax[ii,jj], plot_mode)(depths, qaa[:plot_depth,ss,ii,jj], label="qaa", ls="dotted", c="blue")
            #ax[ii,jj].fill_between(depths, np.mean(qa[:plot_depth,ss, ii,jj] -np.std(qaa[:plot_depth,ss, ii,jj], keepdims=True), grads_mean[:plot_depth,oo, ii,jj]+ np.sqrt(grads_var[:plot_depth,oo, ii,jj]), color=blue_shades[oo], alpha=0.5)
            #getattr(ax[ii,jj], plot_mode)(depths, q_mean[:plot_depth,oo, ii,jj], label="q", ls="dotted", c=red_shades[oo])
            #ax[ii,jj].fill_between(depths, q_mean[:plot_depth,oo, ii,jj]-np.sqrt(q_var[:plot_depth,oo, ii,jj]), q_mean[:plot_depth,oo, ii,jj]+ np.sqrt(q_var[:plot_depth,oo, ii,jj]), color=red_shades[oo], alpha=0.5)
            getattr(ax[ii,jj], plot_mode)(depths, mean_cab[:plot_depth,ii,jj], ls="dotted", c="blue")
            ax[ii,jj].fill_between(depths, mean_cab[:plot_depth,ii,jj]-np.sqrt(var_cab[:plot_depth,ii,jj]), mean_cab[:plot_depth,ii,jj]+ np.sqrt(var_cab[:plot_depth,ii,jj]), color="red", alpha=0.5)
            #ax[ii,jj].fill_between(depths, c_mean[:plot_depth,oo, ii,jj]- np.sqrt(c_var[:plot_depth,oo, ii,jj]), c_mean[:plot_depth,oo, ii,jj]+ np.sqrt(c_var[:plot_depth,oo, ii,jj]), color=green_shades[oo], alpha=0.5)
            #ax[ii,jj].axhline(y=1.0, color='g', linestyle='-')
        

    # save main figure
    handles, labels = ax[00,00].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    fig.legend(handles, labels, loc='center right', handletextpad=0.05, bbox_to_anchor=(0.0, 0.0, 0.95, 1.0), ncol=1,frameon=True, fontsize=20)
    fig.supxlabel("# of layers", fontsize=20)
    fig.supylabel("$V_B$", fontsize=20)
    fig.suptitle("$V_W$",fontsize=20)
    fig.tight_layout(rect=[0, 0, 0.85, 1]) 
    fig.savefig(f"figures/{plot_mode}_{act_func}_DIP_D{max_depth}.png", dpi=300, bbox_inches='tight')

 