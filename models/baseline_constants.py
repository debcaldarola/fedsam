"""Configuration file for common models/experiments"""

MAIN_PARAMS = {
    'cifar100': {
        'small': (1000, 100, 10),
        'medium': (10000, 100, 10),
        'large': (20000, 100, 10)
        },
    'cifar10': {
        'small': (1000, 100, 10),
        'medium': (10000, 100, 10),
        'large': (20000, 100, 10)
        },
}
"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_PARAMS = {
    'cifar100.cnn': (0.01, 100),
    'cifar10.cnn': (0.01, 10),
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
CLIENT_PARAMS_KEY = 'client_params_norm'
CLIENT_GRAD_KEY = 'client_grad_norm'
CLIENT_TASK_KEY = 'client_task'
