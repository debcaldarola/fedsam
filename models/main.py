"""Script to run the baselines."""
import importlib
import inspect
import json
import numpy as np
import os
import pandas as pd
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import random
import torch
import torch.nn as nn
import wandb
from datetime import datetime

import metrics.writer as metrics_writer
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, ACCURACY_KEY, CLIENT_PARAMS_KEY, CLIENT_GRAD_KEY, \
    CLIENT_TASK_KEY
from utils.args import parse_args, check_args
from utils.cutout import Cutout
from utils.main_utils import *
from utils.model_utils import read_data

os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

def main():
    args = parse_args()
    check_args(args)

    # Set the random seed if provided (affects client sampling and batching)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # CIFAR: obtain info on parameter alpha (Dirichlet's distribution)
    alpha = args.alpha
    if alpha is not None:
        alpha = 'alpha_{:.2f}'.format(alpha)
        print("Alpha:", alpha)

    # Setup GPU
    device = torch.device(args.device if torch.cuda.is_available else 'cpu')
    print("Using device:", torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu')

    run, job_name = init_wandb(args, alpha, run_id=args.wandb_run_id)

    # Obtain the path to client's model (e.g. cifar10/cnn.py), client class and servers class
    model_path = '%s/%s.py' % (args.dataset, args.model)
    dataset_path = '%s/%s.py' % (args.dataset, 'dataloader')
    server_path = 'servers/%s.py' % (args.algorithm + '_server')
    client_path = 'clients/%s.py' % (args.client_algorithm + '_client' if args.client_algorithm is not None else 'client')
    check_init_paths([model_path, dataset_path, server_path, client_path])

    model_path = '%s.%s' % (args.dataset, args.model)
    dataset_path = '%s.%s' % (args.dataset, 'dataloader')
    server_path = 'servers.%s' % (args.algorithm + '_server')
    client_path = 'clients.%s' % (args.client_algorithm + '_client' if args.client_algorithm is not None else 'client')

    # Load model and dataset
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')
    dataset = importlib.import_module(dataset_path)
    ClientDataset = getattr(dataset, 'ClientDataset')

    # Load client and server
    print("Running experiment with server", server_path, "and client", client_path)
    Client, Server = get_client_and_server(server_path, client_path)
    print("Verify client and server:", Client, Server)

    # Experiment parameters (e.g. num rounds, clients per round, lr, etc)
    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with servers model
    client_model = ClientModel(*model_params, device)
    if args.load and wandb.run.resumed:  # load model from checkpoint
        client_model, checkpoint, ckpt_path_resumed = resume_run(client_model, args, wandb.run)
        if args.restart:    # start new wandb run
            wandb.finish()
            print("Starting new run...")
            run = init_wandb(args, device, alpha, run_id=None)

    client_model = client_model.to(device)

    #### Create server ####
    server_params = define_server_params(args, client_model, args.algorithm,
                                         opt_ckpt=checkpoint['opt_state_dict'] if args.load and 'opt_state_dict' in checkpoint else None)
    server = Server(**server_params)

    start_round = 0 if not args.load else checkpoint['round']
    print("Start round:", start_round)

    #### Create and set up clients ####
    train_clients, test_clients = setup_clients(args, client_model, Client, ClientDataset, run, device)
    train_client_ids, train_client_num_samples = server.get_clients_info(train_clients)
    test_client_ids, test_client_num_samples = server.get_clients_info(test_clients)
    if set(train_client_ids) == set(test_client_ids):
        print('Clients in Total: %d' % len(train_clients))
    else:
        print(f'Clients in Total: {len(train_clients)} training clients and {len(test_clients)} test clients')

    server.set_num_clients(len(train_clients))

    # Initial status
    print('--- Random Initialization ---')

    start_time = datetime.now()
    current_time = start_time.strftime("%m%d%y_%H:%M:%S")

    ckpt_path, res_path, file, ckpt_name = create_paths(args, current_time, alpha=alpha, resume=wandb.run.resumed)
    ckpt_name = job_name + '_' + current_time + '.ckpt'
    if args.load:
        ckpt_name = ckpt_path_resumed
        if 'round' in ckpt_name:
            ckpt_name = ckpt_name.partition("_")[2]
        print("Checkpoint name:", ckpt_name)

    fp = open(file, "w")
    last_accuracies = []

    print_stats(start_round, server, train_clients, train_client_num_samples, test_clients, test_client_num_samples,
                args, fp)
    wandb.log({'round': start_round}, commit=True)

    ## Setup SWA
    swa_n = 0
    swa_start = args.swa_start
    if args.swa:
        if args.swa_start is None:
            swa_start = int(.75 * num_rounds)
        if wandb.run.resumed and start_round > swa_start:
            print("Loading SWA model...")
            server.setup_swa_model(swa_ckpt=checkpoint['swa_model'])
            swa_n = checkpoint['swa_n']
            print("SWA n:", swa_n)
        print("SWA starts @ round:", swa_start)

    # Start training
    for i in range(start_round, num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))
        fp.write('--- Round %d of %d: Training %d Clients ---\n' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train during this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)
        print("Selected clients:", c_ids)

        if args.swa and i >= swa_start:
            if i == swa_start:
                print("Setting up SWA...")
                server.setup_swa_model()
            # Update lr according to https://arxiv.org/pdf/1803.05407.pdf
            if args.swa_c > 1:
                lr = schedule_cycling_lr(i, args.swa_c, args.lr, args.swa_lr)
                server.update_clients_lr(lr)

        ##### Simulate servers model training on selected clients' data #####
        sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size,
                                         minibatch=args.minibatch)

        ##### Update server model (FedAvg) #####
        print("--- Updating central model ---")
        server.update_model()

        if args.swa and i > swa_start and (i - swa_start) % args.swa_c == 0:  # end of cycle
            print("Number of models:", swa_n)
            server.update_swa_model(1.0 / (swa_n + 1))
            swa_n += 1

        ##### Test model #####
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds or (i+1) > num_rounds - 100:  # eval every round in last 100 rounds
            _, test_metrics = print_stats(i + 1, server, train_clients, train_client_num_samples, test_clients, test_client_num_samples,
                                                args, fp)
            if (i+1) > num_rounds - 100:
                last_accuracies.append(test_metrics[0])

        ### Gradients information ###
        model_grad_norm = server.get_model_grad()
        grad_by_param = server.get_model_grad_by_param()
        for param, grad in grad_by_param.items():
            name = 'params_grad/' + param
            wandb.log({name: grad}, commit=False)
        model_params_norm = server.get_model_params_norm()

        wandb.log({'model total norm': model_grad_norm, 'global model parameters norm': model_params_norm, 'round': i+1}, commit=True)

        # Save round global model checkpoint
        if (i + 1) == num_rounds * 0.05 or (i + 1) == num_rounds * 0.25 or (i + 1) == num_rounds * 0.5 or (i + 1) == num_rounds * 0.75:
            where_saved = server.save_model(i+1, os.path.join(ckpt_path, 'round:' + str(i+1) + '_' + job_name + '_' + current_time + '.ckpt'),
                                            swa_n if args.swa else None)
        else:
            where_saved = server.save_model(i + 1, os.path.join(ckpt_path, ckpt_name), swa_n if args.swa else None)
        wandb.save(where_saved)
        print('Checkpoint saved in path: %s' % where_saved)
        wandb.save(file)

    ## FINAL ANALYSIS ##
    where_saved = server.save_model(num_rounds, os.path.join(ckpt_path, 'round:' + str(num_rounds) + '_' + job_name + '_' + current_time + '.ckpt'))
    wandb.save(where_saved)
    print('Checkpoint saved in path: %s' % where_saved)

    if last_accuracies:
        avg_acc = sum(last_accuracies) / len(last_accuracies)
        print("Last {:d} rounds accuracy: {:.3f}".format(len(last_accuracies), avg_acc))
        wandb.log({'Averaged final accuracy': avg_acc, 'round': num_rounds}, commit=True)

    # Save results
    fp.close()
    wandb.save(file)
    print("File saved in path: %s" % res_path)
    wandb.finish()


def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, train_data, test_data, model, args, ClientDataset, Client, run=None, device=None):
    clients = []
    client_params = define_client_params(args.client_algorithm, args)
    client_params['model'] = model
    client_params['run'] = run
    client_params['device'] = device
    for u in users:
        c_traindata = ClientDataset(train_data[u], train=True, loading=args.where_loading, cutout=Cutout if args.cutout else None)
        c_testdata = ClientDataset(test_data[u], train=False, loading=args.where_loading, cutout=None)
        client_params['client_id'] = u
        client_params['train_data'] = c_traindata
        client_params['eval_data'] = c_testdata
        clients.append(Client(**client_params))
    return clients


def setup_clients(args, model, Client, ClientDataset, run=None, device=None,):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    train_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'test')

    train_users, train_groups, test_users, test_groups, train_data, test_data = read_data(train_data_dir, test_data_dir, args.alpha)

    train_clients = create_clients(train_users, train_data, test_data, model, args, ClientDataset, Client, run, device)
    test_clients = create_clients(test_users, train_data, test_data, model, args, ClientDataset, Client, run, device)

    return train_clients, test_clients

def get_client_and_server(server_path, client_path):
    mod = importlib.import_module(server_path)
    cls = inspect.getmembers(mod, inspect.isclass)
    server_name = server_path.split('.')[1].split('_server')[0]
    server_name = list(map(lambda x: x[0], filter(lambda x: 'Server' in x[0] and server_name in x[0].lower(), cls)))[0]
    Server = getattr(mod, server_name)
    mod = importlib.import_module(client_path)
    cls = inspect.getmembers(mod, inspect.isclass)
    client_name = max(list(map(lambda x: x[0], filter(lambda x: 'Client' in x[0], cls))), key=len)
    Client = getattr(mod, client_name)
    return Client, Server

def init_wandb(args, alpha=None, run_id=None):
    group_name = args.algorithm
    if args.algorithm == 'fedopt':
        group_name = group_name + '_' + args.server_opt

    configuration = args
    if alpha is not None:
        alpha = float(alpha.split('_')[1])
        if alpha not in [0.05, 0.1, 0.2, 0.5]:
            alpha = int(alpha)
        configuration.alpha = alpha

    job_name = 'K' + str(args.clients_per_round) + '_N' + str(args.num_rounds) + '_' + args.model + '_E' + \
               str(args.num_epochs) + '_clr' + str(args.lr) + '_' + args.algorithm
    if alpha is not None:
        job_name = 'alpha' + str(alpha) + '_' + job_name

    if args.server_opt is not None:
        job_name += '_' + args.server_opt + '_slr' + str(args.server_lr)

    if args.server_momentum > 0:
        job_name = job_name + '_b' + str(args.server_momentum)

    if args.client_algorithm is not None:
        job_name = job_name + '_' + args.client_algorithm
        if args.client_algorithm == 'asam' or args.client_algorithm == 'sam':
            job_name += '_rho' + str(args.rho)
            if args.client_algorithm == 'asam':
                job_name += '_eta' + str(args.eta)

    if args.mixup:
        job_name += '_mixup' + str(args.mixup_alpha)

    if args.cutout:
        job_name += '_cutout'

    if args.swa:
        job_name += '_swa' + (str(args.swa_start) if args.swa_start is not None else '') \
                    + '_c' + str(args.swa_c) + '_swalr' + str(args.swa_lr)

    if run_id is None:
        id = wandb.util.generate_id()
    else:
        id = run_id
    run = wandb.init(
                id = id,
                # Set entity to specify your username or team name
                entity="federated-learning",
                # Set the project where this run will be logged
                project='fl_' + args.dataset,
                group=group_name,
                # Track hyperparameters and run metadata
                config=configuration,
                resume="allow")

    if os.environ["WANDB_MODE"] != "offline" and not wandb.run.resumed:
        random_number = wandb.run.name.split('-')[-1]
        wandb.run.name = job_name + '-' + random_number
        wandb.run.save()

    return run, job_name

def print_stats(num_round, server, train_clients, train_num_samples, test_clients, test_num_samples, args, fp):
    train_stat_metrics = server.test_model(train_clients, args.batch_size, set_to_use='train')
    val_metrics = print_metrics(train_stat_metrics, train_num_samples, fp, prefix='train_')

    test_stat_metrics = server.test_model(test_clients, args.batch_size, set_to_use='test' )
    test_metrics = print_metrics(test_stat_metrics, test_num_samples, fp, prefix='{}_'.format('test'))

    wandb.log({'Validation accuracy': val_metrics[0], 'Validation loss': val_metrics[1],
               'Test accuracy': test_metrics[0], 'Test loss': test_metrics[1], 'round': num_round}, commit=False)

    return val_metrics, test_metrics

def print_metrics(metrics, weights, fp, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    metrics_values = []
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))
        fp.write('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g\n' \
                 % (prefix + metric,
                    np.average(ordered_metric, weights=ordered_weights),
                    np.percentile(ordered_metric, 10),
                    np.percentile(ordered_metric, 50),
                    np.percentile(ordered_metric, 90)))
        # fp.write("Clients losses:", ordered_metric)
        metrics_values.append(np.average(ordered_metric, weights=ordered_weights))
    return metrics_values


if __name__ == '__main__':
    main()
