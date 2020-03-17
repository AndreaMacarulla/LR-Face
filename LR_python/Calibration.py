# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 22:05:52 2020

@author: andrea
"""

from tkinter import filedialog as FileDialog
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats



def weibull(u,shape,scale):
    '''Weibull distribution with shape parameter k and scale parameter A'''
    return (shape / scale) * (u / scale)**(shape-1) * np.exp(-(u/scale)**shape)

#%% ask for data file
file = FileDialog.askopenfilename(
    initialdir=".", 
    filetypes=(
        ("Numpy data files","*.npy"),
        ("Ficheros de texto", "*.txt"),
        
    ), 
    title = "Open data file"
)

ll = np.load(file).tolist()

# we separate BSW and WSW
dif = [val[1] for i, val in enumerate(ll) if val[0]==0]
same = [val[1] for i, val in enumerate(ll) if val[0]==1]

##

plt.title('BSW')
plt.hist(dif, bins=60, alpha=1, edgecolor = 'black',  linewidth=1)
plt.grid(True)
plt.show()
plt.clf()


plt.title('WSW')
plt.hist(same, bins=60, alpha=1, edgecolor = 'black',  linewidth=1)
plt.grid(True)
plt.show()
plt.clf()



#%% We create a Weibull approximation for calibration


dif_W0 = stats.weibull_min.fit(dif,floc=0)
#dif_W = stats.exponweib.fit(dif)
shape = dif_W0[0]
scale = dif_W0[2]
dif_W = stats.weibull_min(*dif_W0)

dif_K = stats.gaussian_kde(dif)
#print ('shape:',shape)
#print ('scale:',scale)

#### Plotting
# Histogram first
values,bins,hist = plt.hist(dif,bins=51,range=(0,1.8),density=True)
center = (bins[:-1] + bins[1:]) / 2.

# Using all params and the stats function
plt.plot(center,dif_W.pdf(center),lw=4,label='Weibull')
plt.plot(center,dif_K.pdf(center),lw=4,label='KDE')
# Using my own Weibull function as a check
plt.plot(center,weibull(center,shape,scale),label='Wind analysis',lw=2)
plt.legend()
plt.title('Different person')
plt.xlabel('Facenet distance')
plt.ylabel('Probability')
plt.show()
plt.clf()

#%% Same person
#same_W = stats.exponweib.fit(same, floc=0, f0=1)
same_W0 = stats.weibull_min.fit(same,floc=0)

shape = same_W0[0]
loc = same_W0[1]
scale = same_W0[2]
same_W = stats.weibull_min(*same_W0)

#print ('shape:',shape)
#print ('scale:',scale)

same_K = stats.gaussian_kde(same)

#### Plotting
# Histogram first
values,bins,hist = plt.hist(same,bins=51,range=(0,1.8),density=True)
center = (bins[:-1] + bins[1:]) / 2.

# Using all params and the stats function
plt.plot(center,same_W.pdf(center),lw=4,label='Weibull')
plt.plot(center,same_K.pdf(center),lw=4,label='KDE')
# Using my own Weibull function as a check
plt.plot(center,weibull(center,shape,scale),label='Wind analysis',lw=2)
plt.legend()
plt.title('Same person')
plt.xlabel('Facenet distance')
plt.ylabel('Probability')
plt.show()
plt.clf()


#%% DET plots
#cumulative cdf = cumsum * bin_width
FAR_K = np.cumsum(dif_K.pdf(center)*np.diff(bins))  
FRR_K = 1.-np.cumsum(same_K.pdf(center)*np.diff(bins))


keep = (FAR_K > 1e-4) & (FRR_K > 1e-4);
FAR_K = FAR_K[keep]
FRR_K = FRR_K[keep]

#FAR_W = np.cumsum(dif_W.pdf(center)*np.diff(bins))  
#FRR_W = 1.-np.cumsum(same_W.pdf(center)*np.diff(bins))

#%%
FAR_W = dif_W.cdf(center)
FRR_W = 1.0-same_W.cdf(center)

keep = (FAR_W > 1e-4) & (FRR_W > 1e-4);
FAR_W = FAR_W[keep]
FRR_W = FRR_W[keep]

#%%
plt.plot(100*FAR_W,100*FRR_W,lw=4,label='Weibull')
plt.plot(100*FAR_K,100*FRR_K,lw=4,label='KDE')
plt.xscale('log')
plt.yscale('log')
plt.gca().set_aspect(1,adjustable='box')
plt.legend()
plt.xlabel('False Acceptance Rate')
plt.ylabel('False Rejection Rate')
plt.show()
plt.clf()

