import torch
from torch.optim import Optimizer
import numpy as np
import copy
class sgd(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, clip=False):
        if lr and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.clip = clip
        self.iteration = 0
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(sgd, self).__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            for p in group['params']:
                state = self.state[p]
                # print("initial", p.device)
                state['displacement'] = torch.zeros_like(p)
                state['max_grad'] = torch.zeros(1, device = p.device)
                state['prev_param'] = torch.zeros_like(p)
                state['prev_grad']= torch.zeros_like(p)
                state['prev_prev_param'] = torch.zeros_like(p)
                state['prev_prev_grad']= torch.zeros_like(p)
    def get_params(self):
        param = []
        grads = []
        #print("get_param is called")
        for group in self.param_groups:
            for p in group['params']:
                param.append(p.clone().detach())
                grads.append(p.grad.data.clone().detach())
        return param,grads
    def get_param_groups(self):
        return self.param_groups
    def check_convexity(self):
        sum = 0
        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    prev_param=  state['prev_param']
                    d_p = p.grad
                    param = p
                    sum +=torch.dot(torch.flatten(d_p), torch.flatten(param.add(-prev_param))).item()
        return sum
    def check_smoothness(self):
        sum_num= 0
        sum_denom = 0
        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    prev_param, prev_grad =  state['prev_param'], state['prev_grad']
                    d_p = p.grad
                    param = p
                    sum_num +=torch.norm(d_p -prev_grad)
                    sum_denom += torch.norm(p - prev_param)
                    # L = max(L, torch.div(torch.norm(d_p -prev_grad),torch.norm(p - prev_param) ))
        return sum_num/sum_denom
    def save_param(self, param_dict = None):
        if param_dict is None:
            for group in self.param_groups:
                with torch.no_grad():
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        prev_param, prev_grad=  state['prev_param'], state['prev_grad']
                        prev_param.copy_(p)
                        prev_grad.copy_(p.grad)
        else:
            state_dict = copy.deepcopy(param_dict)
            saved_state = state_dict['state']
            i = 0
            for group in self.param_groups:
                with torch.no_grad():
                    for p in group['params']:
                        state = self.state[p]
                        prev_param, prev_grad=  state['prev_param'], state['prev_grad'] 
                        prev_param.copy_(saved_state[i]['prev_param'])
                        prev_grad.copy_(saved_state[i]['prev_grad'])
                        i+=1
    def save_prev_param(self):
        for group in self.param_groups:
                with torch.no_grad():
                    for p in group['params']:
                        state = self.state[p]
                        prev_param, prev_grad, prev_prev_param, prev_prev_grad=  state['prev_param'], state['prev_grad'], \
                                                                state['prev_prev_param'], state['prev_prev_grad']
                        prev_prev_param.copy_(prev_param)
                        prev_prev_grad.copy_(prev_grad)
    
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        #print("Step is called")
        loss = None
        # i = 0
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.iteration += 1
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            # vector = []
            # grads = []
            # param = []
            # for p in group['params']:
            #     if p.grad is None:
            #         continue
            #     vector.append(self.state[p]['displacement'])
            #     grads.append(p.grad)
            #     param.append(p)

            with torch.no_grad():
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    # i+=1
                    displacement, max_grad, prev_param= state['displacement'], state['max_grad'], state['prev_param']
                    # print("displacement", displacement.device)
                    # print("max_grad", max_grad.device)
                    # print("prev_grad", prev_grad.device)
                    # print("p", p.device)
                    with torch.no_grad():
                        d_p = p.grad
                        # print("displacement", displacement.shape)
                        # print("p.grad", d_p.shape)
                        # print("sum", sum)
                        # sum +=torch.dot(torch.flatten(d_p), torch.flatten(displacement)).item()
                        if weight_decay != 0:
                            d_p = d_p.add(p, alpha=weight_decay)
                        if momentum != 0:
                            if 'momentum_buffer' not in state:
                                buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                            else:
                                buf = state['momentum_buffer']
                                buf.add_(0).add_(displacement, alpha = weight_decay).mul_(momentum).add_(d_p, alpha=1 - dampening)
                                if self.clip:
                                    torch.nn.utils.clip_grad_norm_(buf, max_grad)
                                    max_grad.copy_(torch.maximum((1-dampening)/(1-momentum)*torch.norm(d_p), max_grad))
                            if nesterov:
                                d_p = d_p.add(buf, alpha=momentum)
                            else:
                                d_p = buf
                        # prev_grad.copy_(d_p)
                        # num = np.random.uniform(0,1)
                        # num = max(np.random.exponential(1),3)
                        # num = np.random.exponential(1)
                        # prev_param.copy_(p)
                        displacement.copy_(d_p).mul_(-group['lr'])
                        p.add_(displacement)
                        
        # print("the final tally is", i)          
        return loss
        
    
            
                
class sgd_Snapshot(Optimizer):
    def __init__(self, params):
        defaults = dict()
        super(sgd_Snapshot, self).__init__(params, defaults)
      
    def get_param_groups(self):
        return self.param_groups
    
    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                p.data[:] = q.data[:]
                  
    def get_params(self):
        param = []
        grads = []
        print("get_param is called")
        for group in self.param_groups:
            for p in group['params']:
                param.append(p.clone().detach())
                grads.append(p.grad.data.clone().detach())
        return param,grads       
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
