import numpy as np
import torch as tc


def fuzzy(para=None):
    if para is None:
        para = dict()
    para['classifier_type'] = 'GTNC'
    para['save_data_path'] = '../data/'
    return para


def gtn(para=None):
    if para is None:
        para = dict()
    para.update(ml())
    para.update(program())
    para.update(mps())
    para.update(feature_map())
    para.update(training())
    # Machine Learning Parameter
    para['classifier_type'] = 'GTN'  # classifier name (don not change)
    para['dataset'] = 'mnist'  # data set name ('mnist' or 'fashion')
    para['data_deal_method'] = ['normalization']  # normalizing images berfor training
    para['resize_size'] = (14, 14)
    # Program Parameter
    para['dtype'] = tc.float32
    # MPS Parameter
    para['physical_bond'] = 2
    para['virtual_bond_limit'] = 30  # the \chi of the MPS
    para['mps_cutoff'] = -1
    para['mps_rand_seed'] = 1
    # Feature Map Parameter
    para['theta'] = (np.pi/2) 
    para['mapped_dimension'] = 2
    # Training Parameter
    # para['training_label'] = [3]  # para['training_label'] should be a list
    para['training_label'] = list(range(10))  # para['training_label'] should be a list
    para['n_training'] = 6000  # an int or 'all', which is the size of training date st
    para['rand_index_seed'] = 1
    para['tensor_acc'] = 1e-7
    return para


def gtn_net(para=None):
    if para is None:
        para = dict()
    para.update(ml())
    para.update(program())
    para.update(mps())
    para.update(feature_map())
    # Machine Learning Parameter
    para['classifier_type'] = 'GTN_Net'
    para['dataset'] = 'mnist'
    para['data_deal_method'] = ['normalization']
    para['resize_size'] = (14, 14)
    # Program Parameter
    para['dtype'] = tc.float32
    # MPS Parameter
    para['physical_bond'] = 2
    para['virtual_bond_limit'] = 30
    para['mps_cutoff'] = -1
    para['mps_rand_seed'] = 1
    # Feature Map Parameter
    para['theta'] = (np.pi/2)
    para['mapped_dimension'] = 2
    # Training Parameter
    para['batch_size'] = 1000
    para['training_label'] = list(range(10))  # para['training_label'] should be a list
    para['n_training'] = 6000  # an int or 'all'
    para['rand_index_seed'] = 1
    para['init_step'] = 1e-4
    para['opt_type'] = 'Adam'
    return para


def ml(para=None):
    if para is None:
        para = dict()
    para['dataset'] = 'mnist'
    para['path_dataset'] = '../../data_python/'
    para['data_type'] = ['train', 'test']
    para['classifier_type'] = None
    para['sort_module'] = 'rand'
    para['divide_module'] = 'label'
    para['save_data_path'] = '../data/'
    para['rand_index_seed'] = 1
    para['data_deal_method'] = ['normalization']
    para['resize_size'] = (5, 5)
    para['split_shapes'] = (8, 12)
    para['split_shapeb'] = (20, 20)
    para['map_module'] = 'many_body_Hilbert_space'
    para['theta'] = np.pi/2
    return para


def program(para=None):
    if para is None:
        para = dict()
    para['dtype'] = tc.float64
    para['device'] = 'cuda'
    return para


def mps(para=None):
    if para is None:
        para = dict()
    para['physical_bond'] = 2
    para['virtual_bond_limit'] = 8
    para['tensor_network_type'] = 'MPS'
    para['mps_cutoff'] = 1e-2
    para['mps_normalization_mode'] = True
    para['tensor_initialize_type'] = 'rand'
    para['tensor_initialize_bond'] = 'max'
    para['mps_rand_seed'] = 1
    return para


def feature_map(para=None):
    if para is None:
        para = dict()
    para['map_module'] = 'many_body_Hilbert_space'
    para['theta'] = (np.pi/2) 
    para['mapped_dimension'] = 2
    return para


def training(para=None):
    if para is None:
        para = dict()
    para['training_label'] = [[3], [8]]
    para['n_training'] = 40  # an int or 'all'
    para['update_step'] = np.tan(np.pi/18)
    para['step_decay_rate'] = 3
    para['step_accuracy'] = 1e-2
    para['converge_type'] = 'cost function'
    para['converge_accuracy'] = 1e-2
    para['rand_index_seed'] = 1
    para['tensor_acc'] = 0
    return para





