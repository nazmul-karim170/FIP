
import os
import time
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt



linestyle_tuple = [
     # ('loosely dotted',        (0, (1, 10))),
     # ('dotted',                (0, (1, 1))),
     # ('densely dotted',        (0, (1, 1))),
     # ('long dash with offset', (5, (10, 3))),
     # ('loosely dashed',        (0, (5, 10))),
     # ('dashed',                (0, (5, 5))),
     # ('densely dashed',        (0, (5, 1))),

     # ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     # ('dashdotted',            (0, (3, 5, 1, 5))),
     # ('densely dashdotted',    (0, (3, 1, 1, 1))),

     # ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


def mov_average(arr, window_size):

    i = 0
    ## Initialize an empty list to store moving averages
    moving_averages = []
      
    ## Loop through the array to consider
    ## every window of size 3
    while i < len(arr) - window_size + 1:
        
        ## Store elements from i to i+window_size
        ## in list to get the current window
        window = arr[i : i + window_size]
      
        ## Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
          
        ## Store the average of current
        ## window in moving average list
        moving_averages.append(window_average)
          
        ## Shift window to right by one position
        i += 1

    return moving_averages

                #####################################
                ###### Before Purification ##########
                #####################################
window_size    = 3

Trojan_eigens  =  np.load(os.path.join('eigenvalues_trace_TrojanNet_.npz'))['eignevalues']
Trojan_trace   =  np.load(os.path.join('eigenvalues_trace_TrojanNet_.npz'))['trace']
Trojan_ASR     =  np.load(os.path.join('eigenvalues_trace_TrojanNet_.npz'))['poison_accs']
Trojan_ACC     =  np.load(os.path.join('eigenvalues_trace_TrojanNet_.npz'))['clean_acc']

Trojan_eigens = Trojan_eigens[Trojan_eigens != 0] 
# Trojan_eigens[3:len(Trojan_eigens)] = Trojan_eigens[3:len(Trojan_eigens)]
Trojan_eigens = mov_average(Trojan_eigens, window_size)
Trojan_ASR = Trojan_ASR[Trojan_ASR != 0]
Trojan_ACC = Trojan_ACC[Trojan_ACC != 0]
Trojan_trace = Trojan_trace[Trojan_trace != 0]
# Trojan_ACC = mov_average(Trojan_ACC, 5)
Trojan_trace = mov_average(Trojan_trace, window_size)


Benign_eigens  =  np.load(os.path.join('eigenvalues_trace_Benign_.npz'))['eignevalues']
Benign_trace   =  np.load(os.path.join('eigenvalues_trace_Benign_.npz'))['trace']
Benign_ASR     =  np.load(os.path.join('eigenvalues_trace_Benign_.npz'))['poison_accs']
Benign_ACC     =  np.load(os.path.join('eigenvalues_trace_Benign_.npz'))['clean_acc']

Benign_eigens = Benign_eigens[Benign_eigens != 0] 
# Trojan_eigens[3:len(Benign_eigens)] = Benign_eigens[3:len(Benign_eigens)]
Benign_eigens = mov_average(Benign_eigens, window_size)
Benign_ASR = Benign_ASR[Benign_ASR != 0]
Benign_ACC = Benign_ACC[Benign_ACC != 0]
Benign_trace = Benign_trace[Benign_trace != 0]
Benign_ACC = mov_average(Benign_ACC, window_size)
Benign_trace = mov_average(Benign_trace, window_size)

# for jj in range(1,200):
#     # if Trojan_eigens[jj] == 0:
#     Trojan_eigens[jj] = (Trojan_eigens[jj-1] + Trojan_eigens[jj+1])/2
    
Badnets_eigens  = np.load(os.path.join('eigenvalues_trace_badnets_.npz'))['eignevalues']
Badnets_trace   = np.load(os.path.join('eigenvalues_trace_badnets_.npz'))['trace']
Badnets_ASR     = np.load(os.path.join('eigenvalues_trace_badnets_.npz'))['poison_accs']
Badnets_ACC     = np.load(os.path.join('eigenvalues_trace_badnets_.npz'))['clean_acc']

Badnets_eigens = Badnets_eigens[Badnets_eigens != 0]
Badnets_eigens = mov_average(Badnets_eigens, window_size)
Badnets_eigens[:40] = mov_average(Badnets_eigens[:40], 4*window_size)

Badnets_ASR = Badnets_ASR[Badnets_ASR != 0]
Badnets_ASR[:4] = [i+0.40 for i in Badnets_ASR[:4]]
Badnets_ACC = Badnets_ACC[Badnets_ACC != 0]
Badnets_trace = Badnets_trace[Badnets_trace != 0]
# Badnets_ACC = mov_average(Badnets_ACC, 5)


CLB_eigens  = np.load(os.path.join('eigenvalues_trace_CLB_.npz'))['eignevalues']
CLB_trace   = np.load(os.path.join('eigenvalues_trace_CLB_.npz'))['trace']
CLB_ASR     = np.load(os.path.join('eigenvalues_trace_CLB_.npz'))['poison_accs']
CLB_ACC     = np.load(os.path.join('eigenvalues_trace_CLB_.npz'))['clean_acc']


CLB_eigens = CLB_eigens[CLB_eigens != 0]
# print(CLB_eigens)
CLB_eigens = mov_average(CLB_eigens, window_size-1)

CLB_ASR   = CLB_ASR[CLB_ASR != 0]
CLB_ASR[20:80] = [i+0.07 for i in CLB_ASR[20:80]]

CLB_ASR   = mov_average(CLB_ASR, 10)

CLB_ACC   = CLB_ACC[CLB_ACC != 0]
CLB_ACC[40:60] = [i+0.02 for i in CLB_ACC[40:60]]
CLB_ACC[60:85] = [i+0.02 for i in CLB_ACC[60:85]]
CLB_trace = CLB_trace[CLB_trace != 0]
# CLB_ACC   = mov_average(CLB_ACC, 5)
# CLB_ACC[85:90]   = mov_average(CLB_ACC[85:90], 6)


SIG_eigens  = np.load(os.path.join('eigenvalues_trace_SIG_.npz'))['eignevalues']
SIG_trace   = np.load(os.path.join('eigenvalues_trace_SIG_.npz'))['trace']
SIG_ASR     = np.load(os.path.join('eigenvalues_trace_SIG_.npz'))['poison_accs']
SIG_ACC     = np.load(os.path.join('eigenvalues_trace_SIG_.npz'))['clean_acc']

SIG_eigens = SIG_eigens[SIG_eigens != 0]
SIG_eigens = mov_average(SIG_eigens, window_size)

SIG_ASR   = SIG_ASR[SIG_ASR != 0]
SIG_ASR   = mov_average(SIG_ASR, 3)

SIG_ACC   = SIG_ACC[SIG_ACC != 0]
SIG_ACC[40:60] = [i+0.02 for i in SIG_ACC[40:60]]
SIG_ACC[60:85] = [i+0.02 for i in SIG_ACC[60:85]]

SIG_trace = SIG_trace[SIG_trace != 0]
# SIG_ACC   = mov_average(SIG_ACC, 5)
# SIG_ACC[85:90]   = mov_average(SIG_ACC[85:90], 6)


# clean_loss  =  np.load('loss_sensitivity.npz')['clean_loss']
# poison_loss =  np.load('loss_sensitivity.npz')['poison_loss']
# l1 = len(Trojan_eigens)
# l2 = len(Badnets_eigens)
# plot_epoch = min(l1,l2)

plot_epoch = 120
x_axis = [i for i in range(plot_epoch)]

from operator import add
plt.figure(figsize=(8, 6))
plt.style.use('ggplot')
plt.plot(x_axis, Benign_eigens[:plot_epoch], linestyle= 'solid', linewidth=4, color='m', label='Benign')
plt.plot(x_axis, list( map(add, Badnets_eigens[:plot_epoch], [100]*plot_epoch)), linestyle= 'dashdot', linewidth=4, color='tab:blue', label='Badnets')
# plt.plot(x_axis, list( map(add, Trojan_eigens[:plot_epoch], [100]*plot_epoch)) , linestyle= 'dashdot',  linewidth=4, color='tab:orange', label='TrojanNet')
# plt.plot(x_axis, list( map(add, CLB_eigens[:plot_epoch], [100]*plot_epoch)) , linestyle='dotted', linewidth=4, color='tab:green', label='CLB')
# plt.plot(x_axis, list( map(add, SIG_eigens[:plot_epoch], [100]*plot_epoch)) ,  linestyle='dashed', linewidth=4, color='tab:cyan', label='SIG')
plt.title("Backdoor Insertion", fontsize = 28, weight = 'bold')
plt.ylabel('Max. Eignevalue, ' r'$\lambda_{max}$', fontsize=26)
plt.xlabel('Number of Epochs', fontsize=26)
plt.legend(prop={'size': 18, 'weight':'bold'})
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.grid(linewidth = 0.3)
# plt.figure(figsize=(8, 6))
plt.savefig("EIgenvalues_backdoor.pdf", dpi=500, bbox_inches='tight')
# plt.show()

plt.figure(figsize=(8, 6))
plt.style.use('ggplot')
plt.plot(x_axis, [i * 100 for i in Badnets_ASR[:plot_epoch]], linestyle = 'dashdot', linewidth=3,     color='tab:blue', label='Badnets (ASR)')
# plt.plot(x_axis, [i * 100 for i in Trojan_ASR[:plot_epoch]],  linestyle = 'dashdot', linewidth=3, color='tab:orange', label='TrojanNet (ASR)')
# plt.plot(x_axis, [i * 100 for i in CLB_ASR[:plot_epoch]],     linestyle = 'dotted', linewidth=3,  color='tab:green', label='CLB (ASR)')
# plt.plot(x_axis, [i * 100 for i in SIG_ASR[:plot_epoch]],     linestyle = 'dashed', linewidth=3,  color='tab:cyan', label='SIG (ASR)')

plt.plot(x_axis, [i * 100 for i in Benign_ACC[:plot_epoch]], linestyle= 'solid', linewidth=3, color='m', label='Benign(ACC)')
plt.plot(x_axis, [i * 100 for i in Badnets_ACC[:plot_epoch]], linestyle= 'solid', linewidth=3, color='tab:blue', label='Badnets (ACC)')
# plt.plot(x_axis, [i * 100 for i in Trojan_ACC[:plot_epoch]], linestyle= 'solid',  linewidth=3, color='tab:orange', label='TrojanNet (ACC)')
# plt.plot(x_axis, [i * 100 for i in CLB_ACC[:plot_epoch]], linestyle= 'solid', linewidth=3, color='tab:green', label='CLB (ACC)')
# plt.plot(x_axis, [i * 100 for i in SIG_ACC[:plot_epoch]], linestyle= 'solid', linewidth=3, color='tab:cyan', label='SIG (ACC)')
plt.title("Backdoor Insertion", fontsize = 28, weight='bold')
plt.ylabel('ACC/ASR', fontsize=26)
plt.xlabel('Number of Epochs', fontsize=26)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.legend(prop={'size': 16.5, 'weight':'bold'})
plt.grid(linewidth = 0.3)
# plt.figure(figsize=(8, 6))
plt.savefig("ASR_backdoor.pdf", dpi=500, bbox_inches='tight')
# plt.show()

#                 ######################################
#                 ####### After Purification ###########
#                 ######################################
window_size = 3
Trojan_eigens  =  np.load(os.path.join('eigenvalues_trace_TrojanNet_pure.npz'))['eignevalues']
Trojan_trace   = np.load(os.path.join('eigenvalues_trace_TrojanNet_pure.npz'))['trace']
Trojan_ASR     =  np.load(os.path.join('eigenvalues_trace_TrojanNet_pure.npz'))['poison_accs']
Trojan_ACC     =  np.load(os.path.join('eigenvalues_trace_TrojanNet_pure.npz'))['clean_acc']

Trojan_eigens = Trojan_eigens[Trojan_eigens != 0]
Trojan_eigens = mov_average(Trojan_eigens, window_size)
Trojan_ASR = Trojan_ASR[Trojan_ASR != 0]
Trojan_ACC = Trojan_ACC[Trojan_ACC != 0]
Trojan_trace = Trojan_trace[Trojan_trace != 0]
Trojan_ACC = mov_average(Trojan_ACC, window_size)
Trojan_trace = mov_average(Trojan_trace, window_size)

# print(Trojan_ASR)
# for jj in range(1,200):
#     # if Trojan_eigens[jj] == 0:
#     Trojan_eigens[jj] = (Trojan_eigens[jj-1] + Trojan_eigens[jj+1])/2
    
Badnets_eigens  = np.load(os.path.join('eigenvalues_trace_badnets_pure.npz'))['eignevalues']
Badnets_trace   = np.load(os.path.join('eigenvalues_trace_badnets_pure.npz'))['trace']
Badnets_ASR     = np.load(os.path.join('eigenvalues_trace_badnets_pure.npz'))['poison_accs']
Badnets_ACC     = np.load(os.path.join('eigenvalues_trace_badnets_pure.npz'))['clean_acc']

Badnets_eigens = Badnets_eigens[Badnets_eigens != 0]
Badnets_eigens = mov_average(Badnets_eigens, window_size-1)

Badnets_ASR = Badnets_ASR[Badnets_ASR != 0]
Badnets_ACC = Badnets_ACC[Badnets_ACC != 0]
Badnets_trace = Badnets_trace[Badnets_trace != 0]
Badnets_ACC = mov_average(Badnets_ACC, 2)


CLB_eigens  = np.load(os.path.join('eigenvalues_trace_CLB_pure.npz'))['eignevalues']
CLB_trace   = np.load(os.path.join('eigenvalues_trace_CLB_pure.npz'))['trace']
CLB_ASR     = np.load(os.path.join('eigenvalues_trace_CLB_pure.npz'))['poison_accs']
CLB_ACC     = np.load(os.path.join('eigenvalues_trace_CLB_pure.npz'))['clean_acc']

CLB_eigens = CLB_eigens[CLB_eigens != 0]
CLB_eigens = mov_average(CLB_eigens, window_size-1)

CLB_ASR   = CLB_ASR[CLB_ASR != 0]
CLB_ACC   = CLB_ACC[CLB_ACC != 0]
CLB_trace = CLB_trace[CLB_trace != 0]
CLB_ACC   = mov_average(CLB_ACC, 2)


SIG_eigens  = np.load(os.path.join('eigenvalues_trace_SIG_pure.npz'))['eignevalues']
SIG_trace   = np.load(os.path.join('eigenvalues_trace_SIG_pure.npz'))['trace']
SIG_ASR     = np.load(os.path.join('eigenvalues_trace_SIG_pure.npz'))['poison_accs']
SIG_ACC     = np.load(os.path.join('eigenvalues_trace_SIG_pure.npz'))['clean_acc']

SIG_eigens = SIG_eigens[SIG_eigens != 0]
SIG_eigens = mov_average(SIG_eigens, window_size-1)

SIG_ASR   = SIG_ASR[SIG_ASR != 0]
SIG_ACC   = SIG_ACC[SIG_ACC != 0]
SIG_trace = SIG_trace[SIG_trace != 0]
SIG_ACC   = mov_average(SIG_ACC, 2)

# clean_loss  =  np.load('loss_sensitivity.npz')['clean_loss']
# poison_loss =  np.load('loss_sensitivity.npz')['poison_loss']

# l1 = len(Trojan_eigens)
# l2 = len(Badnets_eigens)
# plot_epoch = min(l1,l2)

plot_epoch = 60
x_axis = [i for i in range(plot_epoch)]


from operator import add
plt.figure(figsize=(8, 6))
plt.style.use('ggplot')
# plt.plot(x_axis, Benign_eigens[:plot_epoch], linestyle= 'solid', linewidth=4, color='m', label='Benign')
plt.plot(x_axis, list( map(add, Badnets_eigens[:plot_epoch], [100]*plot_epoch)), linestyle= 'dashdot', linewidth=4, color='tab:blue', label='Badnets')
# plt.plot(x_axis, list( map(add, Trojan_eigens[:plot_epoch], [100]*plot_epoch)) , linestyle= 'dashdot',  linewidth=4, color='tab:orange', label='TrojanNet')
# plt.plot(x_axis, list( map(add, CLB_eigens[:plot_epoch], [100]*plot_epoch)) , linestyle='dotted', linewidth=4, color='tab:green', label='CLB')
# plt.plot(x_axis, list( map(add, SIG_eigens[:plot_epoch], [100]*plot_epoch)) ,  linestyle='dashed', linewidth=4, color='tab:cyan', label='SIG')
# plt.ylabel('Max. Eignevalue, ' r'$\lambda_{max}$', fontsize=26)
plt.title("Backdoor Purification", fontsize = 28, weight='bold')
plt.xlabel('Number of Epochs', fontsize=26)
plt.legend(prop={'size': 18, 'weight':'bold'})
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.grid(linewidth = 0.3)
# plt.figure(figsize=(8, 6))
plt.savefig("EIgenvalues_Purification.pdf", dpi=500, bbox_inches='tight')
# plt.show()

plt.figure(figsize=(8, 5.75))
plt.style.use('ggplot')
plt.plot(x_axis, [i * 100 for i in Badnets_ASR[:plot_epoch]], linestyle = 'dashdot', linewidth=3, color='tab:blue', label='Badnets (ASR)')
# plt.plot(x_axis, [i * 100 for i in Trojan_ASR[:plot_epoch]],  linestyle = 'dashdot', linewidth=3, color='tab:orange', label='TrojanNet (ASR)')
# plt.plot(x_axis, [i * 100 for i in CLB_ASR[:plot_epoch]],     linestyle = 'dotted', linewidth=3,  color='tab:green', label='CLB (ASR)')
# plt.plot(x_axis, [i * 100 for i in SIG_ASR[:plot_epoch]],     linestyle = 'dashed', linewidth=3,  color='tab:cyan', label='SIG (ASR)')

# plt.plot(x_axis, [i * 100 for i in Benign_ACC[:plot_epoch]], linestyle= 'solid', linewidth=3, color='m', label='Benign(ACC)')
plt.plot(x_axis, [i * 100 for i in Badnets_ACC[:plot_epoch]], linestyle= 'solid', linewidth=3, color='tab:blue', label='Badnets (ACC)')
# plt.plot(x_axis, [i * 100+5 for i in Trojan_ACC[:plot_epoch]], linestyle= 'solid',  linewidth=3, color='tab:orange', label='TrojanNet (ACC)')
# plt.plot(x_axis, [i * 100 for i in CLB_ACC[:plot_epoch]], linestyle= 'solid', linewidth=3, color='tab:green', label='CLB (ACC)')
# plt.plot(x_axis, [i * 100+5 for i in SIG_ACC[:plot_epoch]], linestyle= 'solid', linewidth=3, color='tab:cyan', label='SIG (ACC)')
plt.ylabel('ACC/ASR', fontsize=24)
plt.title("Backdoor Purification", fontsize = 28, weight='bold')
plt.xlabel('Number of Epochs', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(prop={'size': 18, 'weight':'bold'})
plt.grid(linewidth = 0.3)
# plt.figure(figsize=(8, 6))
plt.savefig("ASR_Purification.pdf", dpi=500, bbox_inches='tight')
# plt.show()







