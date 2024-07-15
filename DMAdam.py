import math
import torch
from torch.optim.optimizer import Optimizer


class DMAdam(Optimizer):
    r"""Implements DMAdam algorithm.
It has been proposed in `DMAdam:Dual Averaging Enhanced Adaptive Gradient Method for Deep Neural Networks`_.
Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (α) (float, optional): learning rate (default: 1e-3)
    ct: (η) momentum  (default: 0.9)
    beta (float, optional): coefficients used for computing
        running averages of the seconde momentum (default: 0.999)
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    amsgrad (boolean, optional): whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)
"""
    def __init__(self, params, lr=1e-3, ct=0.9, beta=0.999, eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if beta < 0 or beta >= 1:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        if ct < 0:
            raise ValueError("Invalid momentum parameter: {}".format(ct))

        defaults = dict(lr=lr, ct=ct, beta=beta, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(DMAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DMAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()


        if 'k' not in self.state:
            self.state['k'] = torch.tensor([0], dtype=torch.long)
        self.state['k'] += 1
        k = self.state['k'].item()

        for group in self.param_groups:

            lr, decay = group["lr"], group["weight_decay"]
            ct, beta, eps = group["ct"], group["beta"], group["eps"]

            lamb = lr * math.sqrt(k)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:

                    state["exp_avg"] = torch.zeros_like(p.data).detach()
                    state["grad_sum_sq"] = torch.zeros_like(p.data).detach()

                    if amsgrad:
                        state['max_grad_sum_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                grad_sum_sq = state["grad_sum_sq"]
                exp_avg = state["exp_avg"]

                if amsgrad:
                    max_grad_sum_sq = state['max_grad_sum_sq']


                if decay != 0:
                    grad.add_(p.data, alpha=decay)

                # Decay the first and second moment running average coefficient
                exp_avg.data.add_(grad, alpha=lamb)  # m = m + λ × grad
                exp_avg.data.div_(math.sqrt(k + 1))  # m = m / sqrt(k + 1)

                grad_sum_sq.mul_(beta).addcmul_(grad, grad, value=1 - beta).add_(eps)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_grad_sum_sq, grad_sum_sq, out=max_grad_sum_sq)
                    # Use the max. for normalizing running avg. of gradient
                    rms = (max_grad_sum_sq.sqrt() / math.sqrt(1 - beta ** (k + 1))).add_(eps)
                else:
                    rms = (grad_sum_sq.sqrt() / math.sqrt(1 - beta ** (k + 1))).add_(eps)


                p.data.addcdiv_(-ct, exp_avg, rms)  # p = p - ct * m / rms


        return loss