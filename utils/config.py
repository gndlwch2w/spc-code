import os
import os.path as osp
import argparse
import inspect

def get_config(**kwargs):
    """Get the configuration for the experiment."""
    if kwargs.get('exp', None) is None:
        fn = inspect.currentframe().f_back.f_code.co_filename
        kwargs['exp'] = osp.splitext(osp.basename(fn))[0].split('train_')[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='name of the dataset')
    parser.add_argument('--root_path', type=str, default=None, help='Name of Experiment')
    parser.add_argument('--exp', type=str, default=None, help='experiment_name')
    parser.add_argument('--model', type=str, default='unet', help='model_name')
    parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--num_classes', type=int, default=None, help='output channel of network')
    parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
    parser.add_argument('--labelnum', type=int, default=10, help='labeled data')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
    parser.add_argument('--gpu', type=str, default='0', help='gpu to use')
    parser.add_argument('--round', type=str, default='0', help='round to run the experiment')

    for k, v in kwargs.items():
        if f'--{k}' not in parser._option_string_actions:
            parser.add_argument(f'--{k}', type=type(v), default=v)
        else:
            parser.set_defaults(**{k: v})

    args = parser.parse_args()
    if args.name == "ACDC":
        args.root_path = osp.join(os.getenv('DATASET_HOME'), 'ACDC_SSL4MIS')
        args.num_classes = 4
    elif args.name == "Synapse":
        args.root_path = osp.join(os.getenv('DATASET_HOME'), 'Synapse_TransUNet')
        args.num_classes = 9
    return args
