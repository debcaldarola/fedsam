import argparse

DATASETS = ['cifar10', 'cifar100']
SERVER_ALGORITHMS = ['fedavg', 'fedopt']
SERVER_OPTS = ['sgd', 'adam', 'adagrad', 'fedavgm']
CLIENT_ALGORITHMS = ['asam', 'sam']
MINIMIZERS = ['sam', 'asam']
SIM_TIMES = ['small', 'medium', 'large']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    ## FEDERATED SETTING ##
    parser.add_argument('--num-rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval-every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients-per-round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('-model',
                        help='name of model;',
                        type=str,
                        required=True)
    parser.add_argument('-algorithm',
                        help='algorithm used for server aggregation;',
                        choices=SERVER_ALGORITHMS,
                        default='fedavg')
    parser.add_argument('--client-algorithm',
                        help='algorithm used on the client-side for regularization',
                        choices=CLIENT_ALGORITHMS,
                        default=None)
    parser.add_argument('-alpha',
                        help='alpha value to retrieve corresponding file',
                        type=float,
                        default=None)
    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting;',
                        type=int,
                        default=0)

    ## SERVER OPTMIZER ##
    parser.add_argument('--server-opt',
                        help='server optimizer;',
                        choices=SERVER_OPTS,
                        required=False)
    parser.add_argument('--server-lr',
                        help='learning rate for server optimizers;',
                        type=float,
                        required=False)
    parser.add_argument('--server-momentum',
                        help='momentum for server optimizers;',
                        type=float,
                        default=0,
                        required=False)

    ## CLIENT TRAINING ##
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                                        help='None for FedAvg, else fraction;',
                                        type=float,
                                        default=None)
    epoch_capability_group.add_argument('--num-epochs',
                                        help='number of epochs when clients train on data;',
                                        type=int,
                                        default=1)
    parser.add_argument('-lr',
                        help='learning rate for local optimizers;',
                        type=float,
                        default=-1,
                        required=False)
    parser.add_argument('--weight-decay',
                        help='weight decay for local optimizers;',
                        type=float,
                        default=0,
                        required=False)
    parser.add_argument('-momentum',
                        help='Client momentum for optimizer',
                        type=float,
                        default=0)
    parser.add_argument('-mixup',
                        help='True if use mixup data augmentation at training time',
                        action='store_true',
                        default=False)
    parser.add_argument('--mixup-alpha',
                        help='Parameter alpha in mixup',
                        type=float,
                        default=1.0)
    parser.add_argument('-cutout',
                        help='apply cutout',
                        action='store_true',
                        default=False)

    ## GPU ##
    parser.add_argument('-device',
                        type=str,
                        default='cuda:0')

    ## DATALOADER ##
    parser.add_argument('--num-workers',
                        help='dataloader num workers',
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument('--where-loading',
                        help='location for loading data in ClientDataset',
                        type=str,
                        choices=['init', 'training_time'],
                        default='training_time',
                        required=False)

    ## LOAD CHECKPOINT AND RESTART OPTIONS ##
    parser.add_argument('-load',
                        action='store_true',
                        default=False)
    parser.add_argument('--wandb-run-id',
                        help='wandb run id for resuming run',
                        type=str,
                        default=None)
    parser.add_argument('-restart',
                        help='True if download model from wandb run but restart experiment with new wandb id',
                        action='store_true',
                        default=False)
    parser.add_argument('--restart-round',
                        help='Round to be restarted (default: last executed round)',
                        type=int,
                        default=None)

    ## FedSAM, FedASAM, SWA hyperparams
    parser.add_argument('-rho',
                        help='Rho parameter for SAM and ASAM minimizers',
                        type=float,
                        default=None)
    parser.add_argument('-eta',
                        help='Eta parameter for SAM and ASAM minimizers',
                        type=float,
                        default=None)
    parser.add_argument('-swa',
                        help='Server-side SWA usage flag (default: off)',
                        action='store_true',
                        default=False)
    parser.add_argument('--swa-start',
                        help='SWA start round number (if SWA, default: 75% round budget)',
                        type=float,
                        default=None)
    parser.add_argument('--swa-c',
                        help='SWA model collection frequency/cycle length in rounds (default: 1)',
                        type=int,
                        default=1)
    parser.add_argument('--swa-lr',
                        help='SWA learning rate (alpha2)',
                        type=float,
                        default=1e-4)


    ## ANALYSIS OPTIONS ##
    parser.add_argument('--metrics-name',
                        help='name for metrics file;',
                        type=str,
                        default='metrics',
                        required=False)
    parser.add_argument('--metrics-dir',
                        help='dir for metrics file;',
                        type=str,
                        default='metrics',
                        required=False)
    parser.add_argument('-t',
                        help='simulation time: small, medium, or large;',
                        type=str,
                        choices=SIM_TIMES,
                        default='large')
    return parser.parse_args()

def check_args(args):
    if (args.client_algorithm == 'sam' or args.client_algorithm == 'asam') and (args.rho is None or args.eta is None):
        print("Specificy values for rho, eta and minimizer for running SAM or ASAM algorithm")
        exit(-1)