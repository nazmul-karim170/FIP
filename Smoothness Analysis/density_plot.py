#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os


def get_esd_plot(eigenvalues, weights, top_eigen, Trace, ACC, ASR, mode= 'clean', save_dir='/logs/models/',  name=''):
    density, grids = density_generate(eigenvalues, weights)
    plt.figure(figsize=(8,6))
    plt.style.use('ggplot')     ## For different style of plots

    ## For rendering to Latex form 
    # plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    plt.rcParams["font.family"] = "Times New Roman"
    # params = {'text.usetex' : True,
    #             'font.family' : 'Times New Roman',
    #             'text.latex.unicode': True}
    # plt.rcParams.update(params)

    # plt.semilogy(grids, density + 1.0e-8)
    density = density + 1.0e-8
    # density[density<1e-9] = 0
    plt.xscale('symlog')
    plt.semilogy(grids, density, color='tab:blue')
    # grids_overlap = np.linspace(-1000, 1000, num=100000)
    # density_overlap = np.zeros(100000)

    plt.ylabel('Density (Log Scale)', fontsize=26, labelpad=10)
    plt.xlabel('Eigenvalue', fontsize=26, labelpad=10)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])


    ## For showing an even plot 
    # plt.title()
    plt.axis([-1.5*np.max(eigenvalues), 1.5*np.max(eigenvalues), np.min(density), 1.5*np.max(density)])
    plt.tight_layout()

    print("The Statistics:", top_eigen, Trace, ACC, ASR)
    den_max = np.max(density)

    # ## trojan Model
    # y_min = 1
    # y_max = 0.25
    
    # ## Purified Model
    # y_min = 11.5
    # y_max = 3.5

    ## Benign Model

    if mode == 'clean':
        # x_min = np.max(eigenvalues)/20
        x_min = 1
        y_min = 4.5
        y_max = 0.9
        plt.text(-.65*np.max(eigenvalues), y_min, r'$\lambda_{max}: %.2f$'%(top_eigen), fontsize=24, weight = 'bold')
        plt.text(-.65*np.max(eigenvalues), y_max, r'$Tr(H): %.2f$'%(Trace), fontsize=24, weight = 'bold')
    
    elif mode == 'Trojan':
        x_min = 3
        y_min = 1
        y_max = 0.2
        plt.text(-0.5*np.max(eigenvalues), y_min, r'$\lambda_{max}: %.2f$'%(top_eigen), fontsize=24, weight = 'bold')
        plt.text(-0.5*np.max(eigenvalues), y_max, r'$Tr(H): %.2f$'%(Trace), fontsize=24, weight = 'bold')     
    
    elif mode == 'Purified_ngf':
        x_min = 0.85  
        y_min = 11.5
        y_max = 2    
        plt.text(-0.9*np.max(eigenvalues), y_min, r'$\lambda_{max}: %.2f$'%(top_eigen), fontsize=24, weight = 'bold')
        plt.text(-0.9*np.max(eigenvalues), y_max, r'$Tr(H): %.2f$'%(Trace), fontsize=24, weight = 'bold')        
    
    else:
        x_min = 3  
        y_min = 1
        y_max = 0.2   
        plt.text(-0.5*np.max(eigenvalues), y_min, r'$\lambda_{max}: %.2f$'%(top_eigen), fontsize=24, weight = 'bold')
        plt.text(-0.5*np.max(eigenvalues), y_max, r'$Tr(H): %.2f$'%(Trace), fontsize=24, weight = 'bold')        
    
    plt.text(x_min, y_min, r'$ACC: %.2f$'%(ACC), fontsize=24, weight = 'bold')
    plt.text(x_min, y_max, r'$ASR: %.2f$'%(ASR), fontsize=24, weight = 'bold')

    #         # Benign
    # plt.text(-.65*np.max(eigenvalues), y_min, r'$\lambda_{max}: %.1f$'%(top_eigen), fontsize=22)
    # plt.text(-.65*np.max(eigenvalues), y_max, r'$Tr(H): %.1f$'%(Trace), fontsize=22)

            ## Purified  
    # plt.text(-0.9*np.max(eigenvalues), y_min, r'$\lambda_{max}: %.1f$'%(top_eigen), fontsize=22)
    # plt.text(-0.9*np.max(eigenvalues), y_max, r'$Tr(H): %.1f$'%(Trace), fontsize=22)

    #         ## Trojan
    # plt.text(-0.5*np.max(eigenvalues), y_min, r'$\lambda_{max}: %.1f$'%(top_eigen), fontsize=24)
    # plt.text(-0.5*np.max(eigenvalues), y_max, r'$Tr(H): %.1f$'%(Trace), fontsize=24)
        
    plt.grid(linewidth = 0.3)    
    # plt.legend(prop={'size': 18})
    # mpl.rcParams.update({'font.size': 18})
    # plt.show()
    plt.savefig(os.path.join(save_dir, name+'density.pdf'), dpi=500, bbox_inches='tight')


def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.1):

    eigenvalues = np.array(eigenvalues)
    weights     = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    # grids = np.linspace(lambda_max, 1.5*lambda_max, num=num_bins)
    # sigma = sigma_squared * max(1, (lambda_max-lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)
