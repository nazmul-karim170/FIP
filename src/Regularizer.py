import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader
import torch.autograd as AG


class CDA_Regularizer:

    def __init__(self, args, device, model, crit, lr=0.002, reg_F = 0.01,  weight = 5):
        self.model = model
        self.device = device
        self.weight = weight
        self.reg_F = args.reg_F
        self.crit = crit
        self.args = args
        self.iter_gap = 5
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.95)    
        print("Regularizer Coffieicients:", self.iter_gap, self.reg_F, self.weight)

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params_initial(self, current_ds, batch_size, num_batch):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_likelihoods = []
        for i, (inputs, target) in enumerate(dl):
            if i > num_batch:
                break
            inputs, target = inputs.cuda(), target.cuda()
            output = F.log_softmax(self.model(inputs), dim=1)
            log_likelihoods.append(output[:, target])
        
        log_likelihood = torch.cat(log_likelihoods).mean()
        # print("log likelihood:", log_likelihood)

        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [name for name, param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            _buff_param_name = _buff_param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)


    ## Resgister using clean validation images
    def register_ewc_params(self, dataset, batch_size, num_batches):
        self._update_fisher_params_initial(dataset, batch_size, num_batches)
        self._update_mean_params()


    ## Regularization Loss
    def _compute_reg_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean   = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))

                loss_consolidation = (estimated_fisher * (param - estimated_mean) ** 2).sum()
                losses.append(loss_consolidation)

            return (weight / 2) * sum(losses)
        
        except AttributeError:
            return 0

    def get_trace_loss(self, outputs, target, hi=20):

        output = F.log_softmax(outputs, dim=1)
        log_liklihoods = output[:, target]
        log_likelihood = log_liklihoods.mean()
        Fv = AG.grad(log_likelihood, self.model.parameters(), create_graph=True)

        # for V_i in V:
        #     # Hv = AG.grad(Fv, params, V_i, create_graph=True)

        niters = hi
        V = list()
        for _ in range(niters):
            # V_i = [torch.randint_like(p, high=2, device=device) for p in model.parameters()]
            # for V_ij in V_i:
            #     V_ij[V_ij == 0] = -1
            V_i = [torch.randn_like(p, device=self.device) for p in self.model.parameters()]
            V.append(V_i)

        trace = list()
        for V_i in V:
            this_trace = 0.0
            for Hv_, V_i_ in zip(Fv, V_i):
                this_trace = this_trace + torch.sum(Hv_ * V_i_)
                trace.append(this_trace)

        return sum(trace) / niters



    def forward_backward_update(self, input_s, target, iteration):
        self.optimizer.zero_grad()

        outputs = self.model(input_s)
        ce_loss     = self.crit(outputs, target)
        # reg_loss    = self._compute_reg_loss(self.weight)

        ### Backpropagate the loss 
        if iteration%self.iter_gap==0:
            trace_loss = self.get_trace_loss(outputs, target)
        else:
            trace_loss = 0

        ## Total Loss 
        loss = ce_loss + self.reg_F*trace_loss #+ reg_loss
        loss.backward()
        self.optimizer.step()

        return loss, outputs

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)