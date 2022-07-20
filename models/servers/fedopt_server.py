import copy
import torch
import torch.optim as optim
from collections import OrderedDict

from .fedavg_server import Server


class FedOptServer(Server):
    def __init__(self, client_model, server_opt, server_lr, momentum=0, opt_ckpt=None):
        super().__init__(client_model)
        print("Server optimizer:", server_opt, "with lr", server_lr, "and momentum", momentum)
        self.server_lr = server_lr
        self.server_momentum = momentum
        self.server_opt = self._get_optimizer(server_opt)
        if opt_ckpt is not None:
            self.load_optimizer_checkpoint(opt_ckpt)

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, analysis=False):
        self.server_opt.zero_grad()
        sys_metrics = super(FedOptServer, self).train_model(num_epochs, batch_size, minibatch, clients, analysis)
        self._save_updates_as_pseudogradients()
        return sys_metrics

    def update_model(self):
        """FedAvg on the clients' updates for the current round.
        Saves the new central model in self.client_model and its state dictionary in self.model
        """
        self.client_model.load_state_dict(self.model)
        # Average deltas and obtain global pseudo gradient (fedavg)
        pseudo_gradient = self._average_updates()
        # Update global model according to chosen optimizer
        self._update_global_model_gradient(pseudo_gradient)
        self.model = copy.deepcopy(self.client_model.state_dict())
        self.total_grad = self._get_model_total_grad()
        self.updates = []
        return

    def save_model(self, round, ckpt_path, swa_n=None):
        """Saves the servers model and optimizer on checkpoints/dataset/model.ckpt."""
        # Save servers model
        save_info = {'model_state_dict': self.model,
                     'opt_state_dict': self.server_opt.state_dict(),
                     'round': round}
        if self.swa_model is not None:
            save_info['swa_model'] = self.swa_model.state_dict()
        if swa_n is not None:
            save_info['swa_n'] = swa_n
        torch.save(save_info, ckpt_path)
        return ckpt_path

    def _save_updates_as_pseudogradients(self):
        clients_models = copy.deepcopy(self.updates)
        self.updates = []
        for i, (num_samples, update) in enumerate(clients_models):
            delta = self._compute_client_delta(update)
            self.updates.append((num_samples, delta))

    def _compute_client_delta(self, cmodel):
        """Args:
            cmodel: client update, i.e. state dict of client's update.
        Returns:
            delta: delta between client update and global model. """
        delta = OrderedDict.fromkeys(cmodel.keys()) # (delta x_i)^t
        for k, x, y in zip(self.model.keys(), self.model.values(), cmodel.values()):
            delta[k] = y - x if "running" not in k and "num_batches_tracked" not in k else y
        return delta

    def _update_global_model_gradient(self, pseudo_gradient):
        """Args:
            pseudo_gradient: global pseudo gradient, i.e. weighted average of the trained clients' deltas.

        Updates the global model gradient as -1.0 * pseudo_gradient
        """
        for n, p in self.client_model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]

        self.server_opt.step()

        bn_layers = OrderedDict(
            {k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.client_model.load_state_dict(bn_layers, strict=False)

    def _get_model_total_grad(self):
        """Returns:
            total_grad: sum of the L2-norm of the gradient of each trainable parameter"""
        total_norm = 0
        for name, p in self.client_model.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except Exception:
                    # this param had no grad
                    pass
        total_grad = total_norm ** 0.5
        # print("total grad norm:", total_grad)
        return total_grad

    def _get_optimizer(self, server_opt):
        """Returns optimizer given its name. If not allowed, NotImplementedError expcetion is raised."""
        if server_opt == 'sgd':
            return optim.SGD(params=self.client_model.parameters(), lr=self.server_lr, momentum=self.server_momentum)
        elif server_opt == 'adam':
            return optim.Adam(params=self.client_model.parameters(), lr=self.server_lr, betas=(0.9, 0.99), eps=10**(-1))
        elif server_opt == 'adagrad':
            return optim.Adagrad(params=self.client_model.parameters(), lr=self.server_lr, eps=10**(-2))
        raise NotImplementedError

    def load_optimizer_checkpoint(self, optimizer_ckpt):
        """Load optimizer state from checkpoint"""
        self.server_opt.load_state_dict(optimizer_ckpt)