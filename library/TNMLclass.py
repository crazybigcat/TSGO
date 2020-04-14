import numpy as np
import os
import torch as tc
from library import BasicFunctions_szz
from library import Parameters
from library import MPSclass
from library import MLclass
from library import Programclass

class GTN(MPSclass.MPS, MLclass.MachineLearning, Programclass.Program):
    def __init__(self, para=Parameters.gtn(), debug_mode=False, device='cuda'):
        # Initialize Parameters
        Programclass.Program.__init__(self, device=device, dtype=para['dtype'], debug_mode=debug_mode)
        MLclass.MachineLearning.__init__(self, para=para, debug_mode=debug_mode)
        MPSclass.MPS.__init__(self)
        self.initialize_parameters_gtn()
        self.name_md5_generate()
        self.debug_mode = debug_mode
        # Initialize MPS and update info
        if not self.debug_mode:
            self.load_gtn()
        if len(self.tensor_data) == 0:
            self.initialize_dataset()
            # Initialize info
            self.generate_tensor_info()
            self.generate_update_info()
            self.initialize_mps_gtn()
        if not self.tensor_data[0].device == self.device:
            for ii in range(len(self.tensor_data)):
                self.tensor_data[ii] = tc.tensor(self.tensor_data[ii], device=self.device)
            tc.cuda.empty_cache()
        # Environment Preparation
        self.tensor_input = None
        self.environment_left = tuple()
        self.environment_right = tuple()
        self.environment_zoom = dict()

    def prepare_start_learning(self):
        if 'dealt_input' not in self.images_data.keys():
            self.initialize_dataset()
        if self.tensor_input is None:
            self.tensor_input = self.feature_map(self.images_data['dealt_input'])
        self.environment_left = list(range(self.tensor_info['n_length']))
        self.environment_right = list(range(self.tensor_info['n_length']))
        self.environment_zoom = dict()
        self.initialize_environment()
        if 'update_direction' not in self.update_info.keys():
            self.update_info['update_direction'] = +1
        if 'update_position' not in self.update_info.keys():
            self.update_info['update_position'] = self.tensor_info['regular_center']
        if self.update_info['loops_learned'] != 0:
            print('load mps trained ' + str(self.update_info['loops_learned']) + ' loops')

    def initialize_mps_gtn(self):
        if self.para['tensor_initialize_type'] == 'rand':
            tc.manual_seed(self.para['mps_rand_seed'])
        if self.para['tensor_initialize_type'] == 'rand':
            for ii in range(self.tensor_info['n_length']):
                self.tensor_data.append(tc.rand(
                    self.tensor_info['tensor_initialize_bond'],
                    self.tensor_info['physical_bond'],
                    self.tensor_info['tensor_initialize_bond'],
                    device=self.device, dtype=self.dtype))
            ii = 0
            self.tensor_data[ii] = tc.rand(
                1, self.tensor_info['physical_bond'],
                self.tensor_info['tensor_initialize_bond'],
                device=self.device, dtype=self.dtype)
            ii = -1
            self.tensor_data[ii] = tc.rand(
                self.tensor_info['tensor_initialize_bond'],
                self.tensor_info['physical_bond'], 1,
                device=self.device, dtype=self.dtype)
        elif self.para['tensor_initialize_type'] == 'ones':
            for ii in range(self.tensor_info['n_length']):
                self.tensor_data.append(tc.ones((
                    self.tensor_info['tensor_initialize_bond'],
                    self.tensor_info['physical_bond'], self.tensor_info['tensor_initialize_bond']),
                    device=self.device, dtype=self.dtype))
            ii = 0
            self.tensor_data[ii] = tc.ones((
                1, self.tensor_info['physical_bond'], self.tensor_info['tensor_initialize_bond']),
                device=self.device, dtype=self.dtype)
            ii = -1
            self.tensor_data[ii] = tc.ones((
                self.tensor_info['tensor_initialize_bond'], self.tensor_info['physical_bond'], 1),
                device=self.device, dtype=self.dtype)
        # Regularization
        self.mps_regularization(-1)
        self.mps_regularization(0)
        self.update_info['update_position'] = self.tensor_info['regular_center']

    def generate_tensor_info(self):
        self.tensor_info['regular_center'] = None
        self.tensor_info['normalization_mode'] = self.para['mps_normalization_mode']
        self.tensor_info['cutoff'] = self.para['mps_cutoff']
        self.tensor_info['regular_bond_dimension'] = self.para['virtual_bond_limit']
        self.tensor_info['n_length'] = self.data_info['n_feature']
        self.tensor_info['physical_bond'] = self.para['physical_bond']

    def update_mps_once(self):
        self.calculate_gradient()
        # Update MPS
        tmp_id = self.update_info['update_position']
        self.tensor_data[tmp_id] -= self.tensor_data[tmp_id].norm() * self.update_info[
            'step'] * self.tmp['gradient'] / ((self.tmp['gradient']).norm() + self.para['tensor_acc'])

    def calculate_gradient(self):
        # Calculate gradient
        tmp_index1 = self.update_info['update_position']
        tmp_tensor_current = self.tensor_data[tmp_index1]
        tmp_tensor1 = tc.einsum(
            'ni,nv,nj->nivj',
            self.environment_left[tmp_index1],
            self.tensor_input[:, tmp_index1, :],
            self.environment_right[tmp_index1]).reshape(self.data_info['n_training'], -1)
        tmp_inner_product = (tmp_tensor1.mm(tmp_tensor_current.view(-1, 1))).t()
        tmp_tensor1 = ((1 / tmp_inner_product).mm(tmp_tensor1)).reshape(tmp_tensor_current.shape)
        self.tmp['gradient'] = 2 * (
                (tmp_tensor_current / (tmp_tensor_current.norm() ** 2))
                - tmp_tensor1 / self.data_info['n_training'])
        # # detect difrection
        # if self.tmp['index_com'] >=0:
        #     inn = tc.einsum('abc,cde,gde,')
        # self.tmp['gradient_com'] = self.tmp['gradient'].copy()
        # self.tmp['index_com'] = tmp_index1

    def update_mps(self):
        self.tensor_data[self.update_info['update_position']] -= self.update_info['step'] * self.tmp['gradient'] / (
                (self.tmp['gradient']).norm() + self.para['tensor_acc'])

    def update_one_loop(self):
        self.calculate_running_time(mode='start')
        if self.tensor_info['regular_center'] != self.update_info['update_position']:
            self.mps_regularization(self.update_info['update_position'])
        if self.update_info['update_direction'] > 0:
            while self.update_info['update_position'] < self.tensor_info['n_length'] - 1:
                self.update_mps_once()
                self.mps_regularization(self.update_info['update_position'] + self.update_info['update_direction'])
                self.update_info['update_position'] = self.tensor_info['regular_center']
                self.calculate_environment_next(self.update_info['update_position'])
            self.update_mps_once()
        elif self.update_info['update_direction'] < 0:
            while self.update_info['update_position'] > 0:
                self.update_mps_once()
                self.mps_regularization(self.update_info['update_position'] + self.update_info['update_direction'])
                self.update_info['update_position'] = self.tensor_info['regular_center']
                self.calculate_environment_forward(self.update_info['update_position'])
            self.update_mps_once()
        self.calculate_running_time(mode='end')
        self.update_info['update_direction'] = -self.update_info['update_direction']
        self.tensor_data[self.tensor_info['regular_center']] /= (
            self.tensor_data[self.tensor_info['regular_center']]).norm()
        self.calculate_cost_function()
        print('cost function = ' + str(self.update_info['cost_function'])
              + ' at ' + str(self.update_info['loops_learned'] + 1) + ' loops.')
        self.print_running_time()
        self.update_info['cost_function_loops'].append(self.update_info['cost_function'])
        self.update_info['loops_learned'] += 1

    def initialize_environment(self):
        self.environment_zoom['left'] = tc.zeros(
            (self.tensor_info['n_length'], self.data_info['n_training']),
            device=self.device, dtype=self.dtype)
        self.environment_zoom['right'] = tc.zeros(
            (self.tensor_info['n_length'], self.data_info['n_training']),
            device=self.device, dtype=self.dtype)
        ii = 0
        self.environment_left[ii] = tc.ones(self.data_info['n_training'], device=self.device, dtype=self.dtype)
        self.environment_left[ii].resize_(self.environment_left[ii].shape + (1,))
        ii = self.tensor_info['n_length'] - 1
        self.environment_right[ii] = tc.ones(self.data_info['n_training'], device=self.device, dtype=self.dtype)
        self.environment_right[ii].resize_(self.environment_right[ii].shape + (1,))
        # for ii in range(self.tensor_info['n_length'] - 1):
        #     self.calculate_environment_next(ii + 1)
        for ii in range(self.tensor_info['n_length']-1, 0, -1):
            self.calculate_environment_forward(ii - 1)

    def calculate_cost_function(self):
        tmp_index = self.update_info['update_position']
        # tmp_matrix = tc.einsum('nv,kvj->nkj', [self.tensor_input[:, tmp_index, :], self.tensor_data[tmp_index]])
        tmp_inner_product = tc.einsum('nk,kvj,nj,nv->n', [self.environment_left[tmp_index],
                                                          self.tensor_data[tmp_index],
                                                          self.environment_right[tmp_index],
                                                          self.tensor_input[:, tmp_index, :]]).cpu()
        # tmp_inner_product = ((tmp_matrix.mul(self.environment_right[0])).sum(1)).cpu()
        Z = self.tensor_data[self.tensor_info['regular_center']].norm() ** 2
        self.update_info['cost_function'] = np.log(Z.cpu()) - np.log(self.data_info['n_training']) - 2 * sum(
            self.environment_zoom['right'][tmp_index, :].cpu() + np.log(
                abs(tmp_inner_product)) + self.environment_zoom['left'][tmp_index, :].cpu()) / self.data_info[
            'n_training']
        # self.update_info['cost_function'] = 2 * np.log((
        #     self.tensor_data[0]).norm().cpu()) - np.log(self.data_info['n_training']) - 2 * sum(
        #     self.environment_zoom['right'][0, :].cpu() + np.log(abs(tmp_inner_product))) / self.data_info['n_training']

    def calculate_cost_function2(self):
        Z = self.tensor_data[self.tensor_info['regular_center']].norm() ** 2
        tmp_inner = self.calculate_inner_product(self.tensor_input)
        self.update_info['cost_function'] = np.log(Z.cpu()) - np.log(self.data_info['n_training']) - sum(
            2 * np.log(abs(tmp_inner.cpu())))/self.data_info['n_training']

    def calculate_cost_function_with_input(self, images_data):
        img_mapped = self.feature_map(images_data)
        n_imgs = img_mapped.shape[0]
        Z = self.tensor_data[self.tensor_info['regular_center']].norm() ** 2
        tmp_inner = self.calculate_inner_product2(img_mapped)
        cost_function = np.log(Z.cpu()) - np.log(n_imgs) - 2 * sum(tmp_inner.cpu())/n_imgs
        return cost_function

    def calculate_environment_forward(self, environment_index):
        self.environment_right[environment_index] = tc.einsum(
            'nj,ivj,nv->ni',
            self.environment_right[environment_index + 1],
            self.tensor_data[environment_index + 1],
            self.tensor_input[:, environment_index + 1, :])
        tmp_norm = (self.environment_right[environment_index]).norm(dim=1)
        self.environment_zoom['right'][environment_index, :] = \
            self.environment_zoom['right'][environment_index + 1, :] + tc.log(tmp_norm)
        self.environment_right[environment_index] = tc.einsum(
            'ij,i->ij', [self.environment_right[environment_index], 1/tmp_norm])

    def calculate_environment_next(self, environment_index):
        self.environment_left[environment_index] = tc.einsum(
            'ni,ivj,nv->nj',
            self.environment_left[environment_index - 1],
            self.tensor_data[environment_index - 1],
            self.tensor_input[:, environment_index - 1, :])
        tmp_norm = self.environment_left[environment_index].norm(dim=1)
        self.environment_zoom['left'][environment_index, :] = \
            self.environment_zoom['left'][environment_index - 1, :] + tc.log(tmp_norm)
        self.environment_left[environment_index] = tc.einsum(
            'ij,i->ij', self.environment_left[environment_index], 1/tmp_norm)

    def calculate_inner_product(self, images_mapped):
        n_images = images_mapped.shape[0]
        tmp_inner_product = tc.ones((n_images, 1), device=self.device, dtype=self.dtype)
        for ii in range(self.tensor_info['n_length']):
            tmp_inner_product = tc.einsum(
                'ni,ivj,nv->nj', tmp_inner_product, self.tensor_data[ii], images_mapped[:, ii, :])
        tmp_inner_product = tmp_inner_product.reshape(-1)
        return tmp_inner_product

    def calculate_inner_product2(self, images_mapped):
        n_images = images_mapped.shape[0]
        tmp_norm = tc.zeros((n_images, ), device=self.device, dtype=self.dtype)
        tmp_inner_product = tc.ones((n_images, 1), device=self.device, dtype=self.dtype)
        for ii in range(self.tensor_info['n_length']):
            tmp_inner_product = tc.einsum(
                'ni,ivj,nv->nj', tmp_inner_product, self.tensor_data[ii], images_mapped[:, ii, :])
            tmp_norm = tmp_norm + tc.log(tmp_inner_product.norm(dim=1))
            tmp_inner_product = tc.einsum('nj,n->nj', [tmp_inner_product, 1/tmp_inner_product.norm(dim=1)])
        return tmp_norm

    def calculate_inner_product_test(self, images_mapped, acc=1e-12):
        n_images = images_mapped.shape[0]
        acc = tc.tensor([acc]).to(self.device).to(self.dtype)
        tmp_norm = tc.zeros((n_images, ), device=self.device, dtype=self.dtype)
        tmp_inner_product = tc.ones((n_images, 1), device=self.device, dtype=self.dtype)
        for ii in range(self.tensor_info['n_length']):
            tmp_inner_product = tc.einsum(
                'ni,ivj,nv->nj', tmp_inner_product, self.tensor_data[ii], images_mapped[:, ii, :])
            tmp_norm = tmp_norm + tc.log(tmp_inner_product.norm(dim=1).max(acc))
            tmp_inner_product = tc.einsum('nj,n->nj', [tmp_inner_product, 1/tmp_inner_product.norm(dim=1).max(acc)])
        return tmp_norm

    def initialize_parameters_gtn(self):
        self.program_info['program_name'] = self.para['classifier_type']
        self.program_info['path_save'] = \
            self.para['save_data_path'] + self.program_info['program_name'] + '/'
        if self.para['tensor_initialize_bond'] == 'max':
            self.tensor_info['tensor_initialize_bond'] = self.para['virtual_bond_limit']
        else:
            self.tensor_info['tensor_initialize_bond'] = self.para['tensor_initialize_bond']

    def load_gtn(self):
        load_path = self.program_info['path_save'] + self.program_info['save_name']
        if os.path.isfile(load_path):
            self.tensor_data, self.tensor_info, self.update_info = \
                BasicFunctions_szz.load_pr(load_path, ['tensor_data', 'tensor_info', 'update_info'])

    def save_data(self):
        if not self.debug_mode:
            BasicFunctions_szz.save_pr(
                self.program_info['path_save'],
                self.program_info['save_name'],
                [self.tensor_data, self.tensor_info, self.update_info, self.para],
                ['tensor_data', 'tensor_info', 'update_info', 'para'])


class GTN_Net(tc.nn.Module, MLclass.MachineLearning, MPSclass.MPS, Programclass.Program):
    def __init__(self, para=Parameters.gtn_net(), debug_mode=False, device='cuda'):
        # Initialize Parameters
        self.para = para
        Programclass.Program.__init__(self, device=device, dtype=para['dtype'], debug_mode=debug_mode)
        MPSclass.MPS.__init__(self)
        MLclass.MachineLearning.__init__(self, self.para, debug_mode=debug_mode)
        tc.nn.Module.__init__(self)
        self.initialize_parameters_gtn_net()
        self.name_md5_generate()
        self.debug_mode = debug_mode
        self.update_info = dict()
        self.data_info = dict()
        # Initialize MPS and update info
        self.weight = None
        self.tensor_shape = None
        if not debug_mode:
            self.load_weight()
        if self.weight is None:
            self.initialize_dataset()
            self.generate_tensor_info()
            self.generate_update_info()
            self.initialize_mps_gtn_net()
            self.initialize_tensor_shape()
            self.weight = tc.nn.Parameter(tc.empty((
                 self.tensor_info['n_length'],
                 self.tensor_info['tensor_initialize_bond'],
                 self.tensor_info['physical_bond'],
                 self.tensor_info['tensor_initialize_bond']), device=self.device, dtype=self.dtype))
            self.initialize_weight()
        self.tensor_input = None
        self.opt = None
        self.initialize_opt()
        self.fun = tc.nn.LogSoftmax(dim=0)

    def initialize_parameters_gtn_net(self):
        self.program_info['program_name'] = self.para['classifier_type']
        self.program_info['path_save'] = \
            self.para['save_data_path'] + self.program_info['program_name'] + '/'
        if self.para['tensor_initialize_bond'] == 'max':
            self.tensor_info['tensor_initialize_bond'] = self.para['virtual_bond_limit']
        else:
            self.tensor_info['tensor_initialize_bond'] = self.para['tensor_initialize_bond']

    def initialize_mps_gtn_net(self):
        if self.para['tensor_initialize_type'] == 'rand':
            tc.manual_seed(self.para['mps_rand_seed'])
        if self.para['tensor_initialize_type'] == 'rand':
            for ii in range(self.tensor_info['n_length']):
                self.tensor_data.append(tc.rand(
                    self.tensor_info['tensor_initialize_bond'],
                    self.tensor_info['physical_bond'],
                    self.tensor_info['tensor_initialize_bond'],
                    device=self.device, dtype=self.dtype))
            ii = 0
            self.tensor_data[ii] = tc.rand(
                1, self.tensor_info['physical_bond'],
                self.tensor_info['tensor_initialize_bond'],
                device=self.device, dtype=self.dtype)
            ii = -1
            self.tensor_data[ii] = tc.rand(
                self.tensor_info['tensor_initialize_bond'],
                self.tensor_info['physical_bond'], 1,
                device=self.device, dtype=self.dtype)
        elif self.para['tensor_initialize_type'] == 'ones':
            for ii in range(self.tensor_info['n_length']):
                self.tensor_data.append(tc.ones((
                    self.tensor_info['tensor_initialize_bond'],
                    self.tensor_info['physical_bond'], self.tensor_info['tensor_initialize_bond']),
                    device=self.device, dtype=self.dtype))
            ii = 0
            self.tensor_data[ii] = tc.ones((
                1, self.tensor_info['physical_bond'], self.tensor_info['tensor_initialize_bond']),
                device=self.device, dtype=self.dtype)
            ii = -1
            self.tensor_data[ii] = tc.ones((
                self.tensor_info['tensor_initialize_bond'], self.tensor_info['physical_bond'], 1),
                device=self.device, dtype=self.dtype)
        # Regularization
        self.mps_regularization(-1)
        self.mps_regularization(0)
        self.update_info['update_position'] = self.tensor_info['regular_center']

    def initialize_weight(self):
        for ii in range(self.tensor_info['n_length']):
            d1, d2, d3 = self.tensor_shape[ii]
            self.weight.data[ii, :d1, :d2, :d3] = self.tensor_data[ii]

    def initialize_opt(self):
        if self.para['opt_type'] == 'Adam':
            self.opt = tc.optim.Adam(self.parameters(), lr=self.para['init_step'])
        if self.para['opt_type'] == 'SGD':
            self.opt = tc.optim.SGD(self.parameters(), lr=self.para['init_step'])

    def initialize_tensor_shape(self):
        self.tensor_shape = list()
        for ii in range(len(self.tensor_data)):
            self.tensor_shape.append(self.tensor_data[ii].shape)

    def generate_tensor_info(self):
        self.tensor_info['regular_center'] = None
        self.tensor_info['normalization_mode'] = self.para['mps_normalization_mode']
        self.tensor_info['cutoff'] = self.para['mps_cutoff']
        self.tensor_info['regular_bond_dimension'] = self.para['virtual_bond_limit']
        self.tensor_info['n_length'] = self.data_info['n_feature']
        self.tensor_info['physical_bond'] = self.para['physical_bond']

    def generate_update_info(self):
        self.update_info['loops_learned'] = 0
        self.update_info['cost_function_loops'] = list()
        self.update_info['cost_time_cpu'] = list()
        self.update_info['cost_time_wall'] = list()
        self.update_info['is_converged'] = 'untrained'

    def prepare_start_learning(self):
        if 'dealt_input' not in self.images_data.keys():
            self.initialize_dataset()
        if self.tensor_input is None:
            self.tensor_input = self.feature_map(self.images_data['dealt_input'])
        if self.update_info['loops_learned'] != 0:
            print('load net trained ' + str(self.update_info['loops_learned']) + ' loops')
        # self.tensor_input = self.tensor_input.to(tc.device('cpu'))

    def start_learning(self, learning_loops=30):
        self.print_program_info(mode='start')
        if self.update_info['is_converged'] is not True:
            self.prepare_start_learning()
            if self.update_info['is_converged'] == 'untrained':
                self.update_info['is_converged'] = False
            if self.update_info['loops_learned'] >= learning_loops:
                print('you have learnt too many loops')
                # learning_loops = int(input("learning_loops = "))
            if self.update_info['loops_learned'] == 0:
                self.calculate_cost_function()
                self.update_info['cost_function_loops'].append(self.update_info['cost_function'])
                print('Initializing ... cost function = ' + str(self.update_info['cost_function']))
            if not self.update_info['is_converged']:
                print('start to learn to ' + str(learning_loops) + ' loops')
            # self.mod = tc.nn.DataParallel(self)
            while (self.update_info['loops_learned'] < learning_loops) and not(self.update_info['is_converged']):
                self.update_one_loop()
                self.save_data()
        else:
            print('load converged mps, do not need training.')
        if self.update_info['is_converged']:
            self.print_converge_info()
        else:
            print('Training end, cost function = ' + str(self.update_info['cost_function']) + ', do not converge.')
        self.calculate_program_info_time(mode='end')
        self.print_program_info(mode='end')
        self.update_info['cost_function_loops'].append(self.update_info['cost_function'])
        self.update_info['loops_learned'] += 1

    def update_one_loop(self):
        self.calculate_running_time(mode='start')
        batch_size = self.para['batch_size']
        batch_num = self.data_info['n_training'] // batch_size
        rand_index = np.random.permutation(self.data_info['n_training'])
        for batch_idx in range(batch_num):
            if (batch_idx + 1) * batch_size <= self.data_info['n_training']:
                tmp_index = rand_index[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                # loss = self.mod.forward(self.tensor_input[tmp_index, :, :]).mean()
                loss = self.forward(self.tensor_input[tmp_index, :, :])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            else:
                loss = self.forward(self.tensor_input[batch_idx * batch_size:, :, :])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        self.calculate_running_time(mode='end')
        self.calculate_cost_function()
        print('cost function = ' + str(self.update_info['cost_function'].data)
              + ' at ' + str(self.update_info['loops_learned'] + 1) + ' loops.')
        self.print_running_time()
        self.update_info['cost_function_loops'].append(self.update_info['cost_function'])
        self.update_info['loops_learned'] += 1

    def forward(self, tmp_input):
        ii = 0
        tmp_inner_product = tc.einsum('ivj,nv->nj', [self.get_tensor(ii), tmp_input[:, ii, :]])
        for ii in range(1, self.tensor_info['n_length']):
            tmp_inner_product = tc.einsum(
                'ni,ivj,nv->nj', tmp_inner_product, self.get_tensor(ii), tmp_input[:, ii, :])
        tmp_inner_product = tmp_inner_product.reshape(-1)
        # print(tmp_inner_product ** 2/self.calculate_Z())
        # print(tmp_inner_product)
        # print(self.calculate_Z())
        # print(self.data_info['n_training'])
        # time.sleep(100)
        # pro = tc.log(tmp_inner_product ** 2/self.calculate_Z())
        loss = - tc.log(tmp_inner_product ** 2/self.calculate_Z()).sum()/tmp_inner_product.shape[0] - np.log(
            tmp_inner_product.shape[0])
        # loss = -self.fun(pro).sum()/tmp_inner_product.shape[0] - np.log(tmp_inner_product.shape[0])
        # loss = -tc.log(pro/pro.sum()).sum()/tmp_inner_product.shape[0] - np.log(tmp_inner_product.shape[0])
        return loss

    def get_tensor(self, index):
        d1, d2, d3 = self.tensor_shape[index]
        return self.weight[index, :d1, :d2, :d3].reshape(d1, d2, d3)

    def calculate_cost_function(self):
        # calculate inner_product
        tmp_inner_product = self.calculate_inner_product(self.tensor_input)
        # calculate cost function
        self.update_info['cost_function'] = (- tc.log(
            tmp_inner_product ** 2/self.calculate_Z()).sum()/tmp_inner_product.shape[0] - np.log(
            tmp_inner_product.shape[0]))

    def calculate_cost_function_with_input(self, images_data):
        img_mapped = self.feature_map(images_data)
        n_imgs = img_mapped.shape[0]
        Z = self.calculate_Z()
        tmp_inner = self.calculate_inner_product2(img_mapped)
        cost_function = tc.log(Z.cpu()) - np.log(n_imgs) - 2 * sum(tmp_inner.cpu())/n_imgs
        return cost_function

    def calculate_inner_product(self, images_mapped):
        n_images = images_mapped.shape[0]
        tmp_inner_product = tc.ones((n_images, 1), device=self.device, dtype=self.dtype)
        for ii in range(self.tensor_info['n_length']):
            tmp_inner_product = tc.einsum(
                'ni,ivj,nv->nj', tmp_inner_product, self.get_tensor(ii), images_mapped[:, ii, :])
        tmp_inner_product = tmp_inner_product.reshape(-1)
        return tmp_inner_product
    
    def calculate_inner_product2(self, images_mapped):
        n_images = images_mapped.shape[0]
        tmp_norm = tc.zeros((n_images, ), device=self.device, dtype=self.dtype)
        tmp_inner_product = tc.ones((n_images, 1), device=self.device, dtype=self.dtype)
        for ii in range(self.tensor_info['n_length']):
            tmp_inner_product = tc.einsum(
                'ni,ivj,nv->nj', tmp_inner_product, self.get_tensor(ii), images_mapped[:, ii, :])
            tmp_norm = tmp_norm + tc.log(tmp_inner_product.norm(dim=1))
            tmp_inner_product = tc.einsum('nj,n->nj', [tmp_inner_product, 1/tmp_inner_product.norm(dim=1)])
        return tmp_norm

    def calculate_Z(self):
        ii = 0
        tmp_matrix = tc.einsum('idj,idm->jm', [self.get_tensor(ii), self.get_tensor(ii)])
        for ii in range(1, self.tensor_info['n_length']):
            tmp_matrix = tc.einsum('ij,idk,jdm->km', [tmp_matrix, self.get_tensor(ii), self.get_tensor(ii)])
        return tmp_matrix.reshape(-1)

    def load_weight(self):
        load_path = self.program_info['path_save'] + self.program_info['save_name']
        if os.path.isfile(load_path):
            self.weight, self.tensor_info, self.update_info, self.tensor_shape = \
                BasicFunctions_szz.load_pr(load_path, ['weight', 'tensor_info', 'update_info', 'tensor_shape'])
            self.weight.data = self.weight.data.to(self.device)

    def save_data(self):
        if not self.debug_mode:
            BasicFunctions_szz.save_pr(
                self.program_info['path_save'],
                self.program_info['save_name'],
                [self.weight, self.tensor_info, self.update_info, self.tensor_shape, self.para],
                ['weight', 'tensor_info', 'update_info', 'tensor_shape', 'para'])
