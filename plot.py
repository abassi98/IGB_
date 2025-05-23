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
    parser.add_argument('--widths', type=parse_list, default=[10000], help="Widths to plot")
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
    run_cfg, Vexp_prime, Vtildeexp_prime, cmeanprime, cvarprime = ut.read_data(f"data/{act_func}_D{max_depth}_W{widths[0]}.h5")
    n_w, n_b, Vw_range, Vb_range = run_cfg["n_w"], run_cfg["n_b"], [run_cfg["Vw_min"], run_cfg["Vw_max"]], [run_cfg["Vb_min"], run_cfg["Vb_max"]]
    Vexp = np.zeros((max_depth, n_widths, n_b, n_w))
    Vtildeexp = np.zeros((max_depth, n_widths, n_b, n_w))
    cmeanexp = np.zeros((max_depth, n_widths, n_b, n_w))
    cvarexp = np.zeros((max_depth, n_widths, n_b, n_w))
    Vexp[:,0,:,:] = Vexp_prime
    Vtildeexp[:,0,:,:] = Vtildeexp_prime
    cmeanexp[:,0,:,:] = cmeanprime
    cvarexp[:,0,:,:] = cvarprime

    for ww, width in enumerate(widths[1:]):
        _, Vexp_prime, Vtildeexp_prime, cmeanprime, cvarprime = ut.read_data(f"data/{act_func}_D{max_depth}_W{width}.h5")
        Vexp[:,ww+1,:,:] = Vexp_prime
        Vtildeexp[:,ww+1,:,:] = Vtildeexp_prime
        cmeanexp[:,ww+1,:,:] = cmeanprime
        cvarexp[:,ww+1,:,:] = cvarprime

    Gexp = Vtildeexp / Vexp # experimental gamma
    Cigb = Gexp/(Gexp+1) # c from IGB
    Cresidual = Cigb - cmeanexp
    Vw_vec = np.linspace(Vw_range[0],Vw_range[1],n_w)
    Vb_vec = np.linspace(Vb_range[0],Vb_range[1],n_b)
   
    # # plot ReLU interesting functions
    # if act_func == "ReLU":
    #     fig1, ax1 = plt.subplots(1,1,figsize=(5,5))
    #     max_gamma, n_gamma = 1000, 100000
    #     gamma = np.linspace(0.0,max_gamma,n_gamma)
    #     dx = max_gamma/n_gamma
    #     fgamma = f_relu(gamma)
    #     ggamma = g_relu(gamma)
    #     delta = g_frac_f(gamma)
    #     getattr(ax1, plot_mode)(gamma, fgamma, label="f(x)")
    #     #getattr(ax1, plot_mode)(gamma, delta, label="g(x)/f(x)-x")
    #     getattr(ax1, plot_mode)(gamma, ggamma, label="g(x)")
    #     #grad = [gradient(g) for g in gamma]
    #     #getattr(ax1, plot_mode)(gamma, grad, label="gradient")
    #     #ax1.semilogy(gamma, gamma, label="x")
    #     #ax1.loglog(gamma, np.gradient(ggamma/fgamma), label="d(g/f)/dx")
    #     getattr(ax1, plot_mode)(gamma, ggamma/fgamma, label="g(x)/f(x)")
    #     #ax1.loglog(gamma, gamma, label="x")
    #     #ax1.hlines(y=1.0, xmin=gamma[0], xmax=gamma[-1])
    #     fig1.legend(loc="center")
    #     fig1.savefig(f"figures/Functions_{plot_mode}_{act_func}.png", dpi=500)
    
    # set up figures
    fig, ax = plt.subplots(n_b,n_w,figsize=(16,9), sharex=True)
    fig_qc, ax_qc = plt.subplots(n_b,n_w,figsize=(16,9), sharex=True)
    fig_v, ax_v = plt.subplots(n_b,n_w,figsize=(16,9), sharex=True)
    fig_vtilde, ax_vtilde = plt.subplots(n_b,n_w,figsize=(16,9), sharex=True)
    fig_g, ax_g = plt.subplots(n_b,n_w,figsize=(16,9), sharex=True)
    fig_res, ax_res = plt.subplots(n_b,n_w,figsize=(16,9), sharex=True, sharey=True)

    # compute theoretical values
    Vtildeth = np.zeros((max_depth, n_b, n_w))
    Vth = np.zeros((max_depth, n_b, n_w))
    Gth = np.zeros((max_depth, n_b, n_w))
    Qth = np.zeros((max_depth, n_b, n_w))
    blue_shades = generate_colour_shades(n_widths, colour="blue")
    green_shades = generate_colour_shades(n_widths, colour="green")
    red_shades = generate_colour_shades(n_widths, colour="red")
    orange_shades = generate_colour_shades(n_widths, colour="orange")
    plot_depth = max_depth
    depths = np.arange(1,101)
    for ii, Vb in enumerate(Vb_vec):
        for jj, Vw in enumerate(Vw_vec):
            # initialize first values
            Vth[0,ii,jj] = Vw
            Vtildeth[0,ii,jj] = Vb + Vw*0.01
            Gth[0,ii,jj] = Vtildeth[0,ii,jj]/Vth[0,ii,jj]
            for kk in range(max_depth-1): # iterate corresponding to number of layers
                if act_func == "ReLU":
                    Vth[kk+1,ii,jj] = Vw*Vth[kk,ii,jj]/2.0 * f_relu(Gth[kk,ii,jj])
                    Vtildeth[kk+1, ii,jj] = Vw*Vth[kk,ii,jj]/2.0 * g_relu(Gth[kk,ii,jj]) + Vb 
                elif act_func == "Linear":
                    Vtildeth[kk+1, ii,jj] = Vw*Vtildeth[kk,ii,jj]+Vb
                    Vth[kk+1,ii,jj] = Vw*Vth[kk,ii,jj]
                else:
                    pass
                
                Gth[kk+1,ii,jj] = Vtildeth[kk+1,ii,jj] / (Vth[kk+1,ii,jj])
            
            Qth = Vtildeth +  Vth
            # Plot theoretical
            if ii==0:
                ax[ii,jj].set_title(f"{np.round(Vw,2)}", fontsize=15)
                ax_qc[ii,jj].set_title(f"{np.round(Vw,2)}", fontsize=15)
                ax_v[ii,jj].set_title(f"{np.round(Vw,2)}", fontsize=15)
                ax_vtilde[ii,jj].set_title(f"{np.round(Vw,2)}", fontsize=15)
                ax_g[ii,jj].set_title(f"{np.round(Vw,2)}", fontsize=15)
                ax_res[ii,jj].set_title(f"{np.round(Vw,2)}", fontsize=15)
            if jj==0:
                ax[ii,jj].set_ylabel(f"{np.round(Vb,2)}",  fontsize=15)
                ax_qc[ii,jj].set_ylabel(f"{np.round(Vb,2)}",  fontsize=15)
                ax_v[ii,jj].set_ylabel(f"{np.round(Vb,2)}",  fontsize=15)
                ax_vtilde[ii,jj].set_ylabel(f"{np.round(Vb,2)}",  fontsize=15)
                ax_g[ii,jj].set_ylabel(f"{np.round(Vb,2)}",  fontsize=15)
                ax_res[ii,jj].set_ylabel(f"{np.round(Vb,2)}",  fontsize=15)
            
            ax[ii,jj].grid()
            ax_qc[ii,jj].grid()
            getattr(ax[ii,jj], plot_mode)(Vth[:plot_depth,ii,jj], label="V th.",ls=(0,(5,10)), c="green")
            getattr(ax[ii,jj], plot_mode)(Vtildeth[:plot_depth,ii,jj], label="$\\tilde{V}$ th.", ls=(0,(5,10)),c="red")
            getattr(ax[ii,jj], plot_mode)(Gth[:plot_depth,ii,jj], label="$\\gamma$ th.", ls=(0,(5,10)), c="blue")
           
            # Plot experimental
            for oo, width in enumerate(widths):
                lw = 1.0 +  5* (oo +1)/ (len(widths)+1)
                # plot error
                #getattr(ax_v[ii,jj], plot_mode)(100*np.abs((Vth[:,ii,jj]-Vexp[:plot_depth,oo, ii,jj])/Vth[:,ii,jj]), lw=lw,label="V th.", c=green_shades[oo])
                #getattr(ax_vtilde[ii,jj], plot_mode)(100*np.abs((Vtildeth[:,ii,jj]-Vtildeexp[:plot_depth,oo, ii,jj])/Vtildeth[:,ii,jj]), lw=lw,label="$\\tilde{V}$ th.", c=red_shades[oo])
                #getattr(ax_g[ii,jj], plot_mode)(100*np.abs((Gth[:,ii,jj]-Gexp[:plot_depth,oo, ii,jj])/Gth[:,ii,jj]), label="$\\gamma$ th.", lw=lw, c=blue_shades[oo])
                # plot only last width
                if oo == len(widths)-1:
                    getattr(ax[ii,jj], plot_mode)(Vexp[:plot_depth,oo, ii,jj], label="V ex.", ls="dotted", c=green_shades[oo])
                    getattr(ax[ii,jj], plot_mode)(Vtildeexp[:plot_depth,oo, ii,jj], label="$\\tilde{V}$ ex.", ls="dotted", c=red_shades[oo])
                    getattr(ax[ii,jj], plot_mode)(Gexp[:plot_depth,oo, ii,jj], label="$\\gamma$ ex.", ls="dotted", c=blue_shades[oo])
                    # getattr(ax_qc[ii,jj], plot_mode)(cmeanexp[:plot_depth,oo, ii,jj], label="c_mean", ls="dotted", c=red_shades[oo])
                    # getattr(ax_qc[ii,jj], plot_mode)(cvarexp[:plot_depth,oo, ii,jj], label="c_var", ls="dotted", c=green_shades[oo])
                    # getattr(ax_qc[ii,jj], plot_mode)(cmeanexp[:plot_depth,oo, ii,jj]/(1.0-cmeanexp[:plot_depth,oo, ii,jj]), label="$\\gamma$ ex.", ls="dotted", c=blue_shades[oo])
                    getattr(ax_res[ii,jj], plot_mode)(Cigb[:plot_depth,oo, ii,jj], label="$c$ igb.", ls="dotted", c=blue_shades[oo],  lw=lw)
                    getattr(ax_res[ii,jj], plot_mode)(cmeanexp[:plot_depth,oo, ii,jj], label="$c$ dip.", ls="dotted", c=red_shades[oo],  lw=lw)
                    #getattr(ax_res[ii,jj], plot_mode)(100*Cresidual[:plot_depth,oo, ii,jj]/Cigb[:plot_depth,oo, ii,jj], label="residual", ls="dotted", c=orange_shades[oo],  lw=lw)
                    ax_res[ii,jj].fill_between(depths, cmeanexp[:plot_depth,oo, ii,jj]-np.sqrt(cvarexp[:plot_depth,oo, ii,jj]), cmeanexp[:plot_depth,oo, ii,jj]+ np.sqrt(cvarexp[:plot_depth,oo, ii,jj]), color=red_shades[oo], alpha=0.5)

    # save main figure
    handles, labels = ax[00,00].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    fig.legend(handles, labels, loc='center right', handletextpad=0.05, bbox_to_anchor=(0.0, 0.0, 0.95, 1.0), ncol=1,frameon=True, fontsize=20)
    fig.supxlabel("# of layers", fontsize=20)
    fig.supylabel("$V_B$", fontsize=20)
    fig.suptitle("$V_W$",fontsize=20)
    fig.tight_layout(rect=[0, 0, 0.85, 1]) 
    fig.savefig(f"figures/{plot_mode}_{act_func}_EXPTH_D{max_depth}_W{max(widths)}.png", dpi=300, bbox_inches='tight')

    # # save qc figure
    # handles, labels = ax_qc[00,00].get_legend_handles_labels()
    # #fig.legend(handles, labels, loc='upper center')
    # fig_qc.legend(handles, labels, loc='center right', handletextpad=0.05, bbox_to_anchor=(0.0, 0.0, 0.95, 1.0), ncol=1,frameon=True, fontsize=20)
    # fig_qc.supxlabel("# of layers", fontsize=20)
    # fig_qc.supylabel("$V_B$", fontsize=20)
    # fig_qc.suptitle("$V_W$",fontsize=20)
    # fig_qc.tight_layout(rect=[0, 0, 0.85, 1]) 
    # fig_qc.savefig(f"figures/{plot_mode}_{act_func}_Qcvarexp_D{max_depth}_W{max(widths)}.png", dpi=300, bbox_inches='tight')

    # save error V
    handles, labels = ax_v[00,00].get_legend_handles_labels()
    fig_v.supxlabel("# of layers", fontsize=20)
    fig_v.supylabel("$V_B$", fontsize=20)
    fig_v.suptitle("$V_W$",fontsize=20)
    fig_v.legend(handles, widths,title="Width", loc='center right', frameon=True,  bbox_to_anchor=(0.0, 0.0, 0.98, 1.0), ncol=1,fontsize=20, title_fontsize=20)
    fig_v.tight_layout(rect=[0, 0, 0.85, 1]) 
    fig_v.savefig(f"figures/{plot_mode}_{act_func}_VERR_D{max_depth}_W{max(widths)}.png", dpi=300)
    
    # save error Vtilde
    handles, labels = ax_vtilde[00,00].get_legend_handles_labels()
    fig_vtilde.supxlabel("# of layers", fontsize=20)
    fig_vtilde.supylabel("$V_B$", fontsize=20)
    fig_vtilde.suptitle("$V_W$",fontsize=20)
    fig_vtilde.legend(handles, widths, title="Width",loc='center right', frameon=True,  bbox_to_anchor=(0.0, 0.0, 0.98, 1.0), ncol=1,fontsize=20, title_fontsize=20)
    fig_vtilde.tight_layout(rect=[0, 0, 0.85, 1]) 
    fig_vtilde.savefig(f"figures/{plot_mode}_{act_func}_VtildeERR_D{max_depth}_W{max(widths)}.png", dpi=300)
    
    # save error g
    handles, labels = ax_g[00,00].get_legend_handles_labels()
    fig_g.supxlabel("# of layers", fontsize=20)
    fig_g.supylabel("$V_B$", fontsize=20)
    fig_g.suptitle("$V_W$",fontsize=20)
    fig_g.legend(handles, widths, title="Width",loc='center right', frameon=True,  bbox_to_anchor=(0.0, 0.0, 0.98, 1.0), ncol=1,fontsize=20, title_fontsize=20)
    fig_g.tight_layout(rect=[0, 0, 0.85, 1]) 
    fig_g.savefig(f"figures/{plot_mode}_{act_func}_GERR_D{max_depth}_MaxW{max(widths)}.png", dpi=300)
    

    # save comparison figure for residuals
    handles, labels = ax_res[00,00].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    fig_res.legend(handles, labels, loc='center right', handletextpad=0.05, bbox_to_anchor=(0.0, 0.0, 0.95, 1.0), ncol=1,frameon=True, fontsize=20)
    fig_res.supxlabel("# of layers", fontsize=20)
    fig_res.supylabel("$V_B$", fontsize=20)
    fig_res.suptitle("$V_W$",fontsize=20)
    fig_res.tight_layout(rect=[0, 0, 0.85, 1]) 
    fig_res.savefig(f"figures/{plot_mode}_{act_func}_Cresidual_D{max_depth}_W{max(widths)}.png", dpi=300, bbox_inches='tight')

    # # plot in function of the width
    # fig_V, ax_V = plt.subplots(n_b,n_w,figsize=(10,4), sharex=True)
    # fig_G, ax_G = plt.subplots(n_b,n_w,figsize=(10,4), sharex=True)
    # fig_Vtilde, ax_Vtilde = plt.subplots(n_b,n_w,figsize=(10,4), sharex=True)
    # cmap = plt.cm.Blues
    # blue_shades = generate_blue_shades(n_widths)
    # widths = np.logspace(np.log10(10), np.log10(max_width), num=n_widths).astype(np.int16)
    # for ii, Vb in enumerate(Vb_vec):
    #     for jj, Vw in enumerate(Vw_vec):
    #         for oo, width in enumerate(widths): 
    #             # Plot experimental
    #             getattr(ax_V[ii,jj], plot_mode)(Vexp[:,oo, ii,jj], label=f"{width}", c=blue_shades[oo],  ls="dotted")
    #             getattr(ax_Vtilde[ii,jj], plot_mode)(Vtildeexp[:,oo, ii,jj], c=blue_shades[oo], label=f"{width}", ls="dotted")
    #             getattr(ax_G[ii,jj], plot_mode)(Gexp[:, oo, ii,jj], c=blue_shades[oo], label=f"{width}", ls="dotted")
            
    # handles, labels = ax_V[00,00].get_legend_handles_labels()
    
    # # savefigs
    # fig_V.supxlabel("Width")
    # fig_V.supylabel("$V_B$")
    # fig_V.suptitle("$V_W$")
    # fig_V.legend(handles, labels, loc='center right', bbox_to_anchor=(0.95, 0.5), frameon=True)
    # fig_V.tight_layout(rect=[0, 0, 0.85, 1])
    # fig_V.savefig(f"figures/V_{plot_mode}_{act_func}_WIDTHEXP_D{max_depth}_W{max_width}.png", dpi=300)

    # fig_Vtilde.supxlabel("Width")
    # fig_Vtilde.supylabel("$V_B$")
    # fig_Vtilde.suptitle("$V_W$")
    # fig_Vtilde.legend(handles, labels, loc='center right', bbox_to_anchor=(0.95, 0.5), frameon=True)
    # fig_Vtilde.tight_layout(rect=[0, 0, 0.85, 1])
    # fig_Vtilde.savefig(f"figures/Vtilde_{plot_mode}_{act_func}_WIDTHEXP_D{max_depth}_W{max_width}.png", dpi=300)

    # fig_G.supxlabel("Width")
    # fig_G.supylabel("$V_B$")
    # fig_Vtilde.suptitle("$V_W$")
    # fig_G.legend(handles, labels, loc='center right', bbox_to_anchor=(0.95, 0.5), frameon=True)
    # fig_G.tight_layout(rect=[0, 0, 0.85, 1])
    # fig_G.savefig(f"figures/Gamma_{plot_mode}_{act_func}_WIDTHEXP_D{max_depth}_W{max_width}.png", dpi=300)
    