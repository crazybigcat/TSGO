import numpy
import torch as tc
import operator
import library.wheel_functions as wf
import copy


class TensorNetwork:

    def __init__(self):
        self.tensor_data = list()
        self.tensor_info = dict()
        self.tmp = dict()


class MPS(TensorNetwork):
    def __init__(self):
        # Prepare parameters
        TensorNetwork.__init__(self)

    def mps_regularization(self, regular_center):
        if regular_center == -1:
            regular_center = self.tensor_info['n_length']-1
        if self.tensor_info['regular_center'] is None:
            self.tensor_info['regular_center'] = 0
            while self.tensor_info['regular_center'] < self.tensor_info['n_length']-1:
                self.move_regular_center2next()
        while self.tensor_info['regular_center'] < regular_center:
            self.move_regular_center2next()
        while self.tensor_info['regular_center'] > regular_center:
            self.move_regular_center2forward()

    def move_regular_center2next(self):
        tensor_index = self.tensor_info['regular_center']
        d1, dv, d2 = self.tensor_data[tensor_index].shape
        q, r = tc.qr(self.tensor_data[tensor_index].to('cpu').reshape(d1 * dv, d2))
        # q, r = tc.qr(self.tensor_data[tensor_index].reshape(d1 * dv, d2))
        # u, s, v = tc.svd(self.tensor_data[tensor_index].to('cpu').reshape(d1 * dv, d2))
        # q = u
        # r = tc.einsum('i,ij->ij', [s, v.t()])
        if self.tensor_info['normalization_mode']:
            r /= r.norm()
        self.tensor_data[tensor_index].data = q.to(self.device).reshape(d1, dv, -1)
        self.tensor_data[tensor_index + 1].data = tc.einsum(
            'ij,jkl->ikl', [r.to(self.device), self.tensor_data[tensor_index+1]])
        self.tensor_info['regular_center'] += 1

    def move_regular_center2forward(self):
        tensor_index = self.tensor_info['regular_center']
        d1, dv, d2 = self.tensor_data[tensor_index].shape
        q, r = tc.qr(self.tensor_data[tensor_index].to('cpu').reshape(d1, dv * d2).t())
        # q, r = tc.qr(self.tensor_data[tensor_index].reshape(d1, dv * d2).t())
        r = r.t()
        if self.tensor_info['normalization_mode']:
            r /= r.norm()
        self.tensor_data[tensor_index].data = q.t().to(self.device).reshape(-1, dv, d2)
        self.tensor_data[tensor_index - 1].data = tc.einsum(
            'ijk,kl->ijl', [self.tensor_data[tensor_index - 1], r.to(self.device)])
        self.tensor_info['regular_center'] -= 1

    # def move_regular_center2forward(self):
    #     self.reverse_mps()
    #     self.move_regular_center2next()
    #     self.reverse_mps()

    # def move_regular_center2next(self):

    def measure_mps(self, operator=numpy.diag([1, -1])):
        # testing code
        measure_data = numpy.zeros(self.tensor_info['n_length'])
        if self.tensor_info['regular_center'] != 0:
            self.mps_regularization(0)
        self.tensor_data[0] /= self.tensor_data[0].norm()
        for ii in range(self.tensor_info['n_length']):
            measure_data[ii] = wf.tensor_contract(
                wf.tensor_contract(self.tensor_data[ii], self.tensor_data[ii], [[0, -1], [0, -1]]),
                operator, [[0, 1], [1, 0]])
            if ii < self.tensor_info['n_length'] - 1:
                self.move_regular_center2next()
        return measure_data

    def measure_images_from_mps(self):
        # testing code
        probability = tc.empty((self.tensor_info['n_length'], 2))
        state = tc.empty((self.tensor_info['n_length'], 2, 2))
        if self.tensor_info['regular_center'] != 0:
            self.mps_regularization(0)
        self.tensor_data[0] /= (self.tensor_data[0]).norm()
        for ii in range(self.tensor_info['n_length']):
            probability[ii], state[ii] = tc.symeig(
                tc.einsum('ivj,iwj->vw', self.tensor_data[ii], self.tensor_data[ii]),
                eigenvectors=True)
            if ii < self.tensor_info['n_length'] - 1:
                self.move_regular_center2next()
        return probability, state

    def calculate_single_entropy(self, dot_index='all'):
        if operator.eq(dot_index, 'all'):
            dot_index = list(range(self.tensor_info['n_length']))
        elif isinstance(dot_index, int):
            dot_index = [dot_index]
        entropy = dict()
        for ii in dot_index:
            self.mps_regularization(ii)
            tmp_tensor = copy.deepcopy(self.tensor_data[ii])
            tmp_tensor = tmp_tensor / numpy.linalg.norm(tmp_tensor)
            tmp_tensor = wf.tensor_contract(tmp_tensor, tmp_tensor, [[0, -1], [0, -1]])
            tmp_spectrum = numpy.linalg.eigh(tmp_tensor)[0]
            tmp_spectrum /= numpy.sum(tmp_spectrum)
            tmp_spectrum[tmp_spectrum <= 0] = 1
            entropy[ii] = abs((tmp_spectrum * numpy.log(tmp_spectrum)).sum())
        return entropy

    def calculate_divided_entropy(self, dot_index='all'):
        if operator.eq(dot_index, 'all'):
            dot_index = list(range(self.tensor_info['n_length']))
        elif isinstance(dot_index, int):
            dot_index = [dot_index]
        dot_index = tc.tensor(dot_index).reshape(-1)
        dot_index_sorted = dot_index.sort()
        entropy = dict()
        for ii in dot_index_sorted[0]:
            self.mps_regularization(ii)
            u, s, v = wf.tensor_svd(self.tensor_data[ii], index_right=2)
            s = s[s > 0]
            s /= s.norm()
            s_tmp = s ** 2
            entropy[ii] = -tc.einsum('a,a->', s_tmp, tc.log(s_tmp))
        return entropy

    def calculateZwithop(self, op):
        op = op.to(self.device).to(self.dtype)
        n_length = len(self.tensor_data)
        tmp_matrix = tc.ones((1, 1), device=self.device, dtype=self.dtype)
        for ii in range(n_length):
            tmp_matrix = tc.einsum('ae,acb,cd,edf->bf', [tmp_matrix, self.tensor_data[ii], op, self.tensor_data[ii]])
        return tmp_matrix.norm()


class LMPS(TensorNetwork):
    def __init__(self):
        # Prepare parameters
        TensorNetwork.__init__(self)

    def mps_regularization(self, regular_center):
        if regular_center == -1:
            regular_center = self.tensor_info['n_length']-1
        if self.tensor_info['regular_center'] is None:
            self.tensor_info['regular_center'] = 0
            while self.tensor_info['regular_center'] < self.tensor_info['n_length']-1:
                self.move_regular_center2next()
        while self.tensor_info['regular_center'] < regular_center:
            self.move_regular_center2next()
        while self.tensor_info['regular_center'] > regular_center:
            self.move_regular_center2forward()

    def move_regular_center2next(self):
        tensor_index = self.tensor_info['regular_center']
        d1, dv, d2, da = self.tensor_data[tensor_index].shape
        u, s, v = tc.svd(self.tensor_data[tensor_index].to('cpu').reshape(d1 * dv, d2 * da))
        s /= s.max()
        s = s[s > self.tensor_info['cutoff']]
        s = s[:self.tensor_info['regular_bond_dimension']]
        dm = len(s)
        u = u[:, :dm]
        v = v[:, :dm]
        if self.tensor_info['normalization_mode']:
            s /= s.norm()
        v = tc.einsum('i,ji->ij', [s, v]).to(self.device).reshape(dm, d2, da)
        self.tensor_data[tensor_index].data = u.to(self.device).reshape(d1, dv, dm)
        self.tensor_data[tensor_index + 1].data = tc.einsum(
            'mja,jkn->mkna', [v, self.tensor_data[tensor_index+1]])
        self.tensor_info['regular_center'] += 1

    def move_regular_center2forward(self):
        tensor_index = self.tensor_info['regular_center']
        d1, dv, d2, da = self.tensor_data[tensor_index].shape
        u, s, v = tc.svd(self.tensor_data[tensor_index].to('cpu').permute(0, 3, 1, 2).reshape(d1 * da, dv * d2))
        s /= s.max()
        s = s[s > self.tensor_info['cutoff']]
        s = s[:self.tensor_info['regular_bond_dimension']]
        dm = len(s)
        u = u[:, :dm]
        v = v[:, :dm]
        if self.tensor_info['normalization_mode']:
            s /= s.norm()
        u = tc.einsum('ij,j->ij', [u, s]).to(self.device).reshape(d1, da, dm)
        self.tensor_data[tensor_index].data = v.t().to(self.device).reshape(dm, dv, d2)
        self.tensor_data[tensor_index - 1].data = tc.einsum(
            'ijk,kam->ijma', [self.tensor_data[tensor_index - 1], u])
        self.tensor_info['regular_center'] -= 1

    # def move_regular_center2forward(self):
    #     self.reverse_mps()
    #     self.move_regular_center2next()
    #     self.reverse_mps()

    # def move_regular_center2next(self):

    def measure_mps(self, operator=numpy.diag([1, -1])):
        # testing code
        measure_data = numpy.zeros(self.tensor_info['n_length'])
        if self.tensor_info['regular_center'] != 0:
            self.mps_regularization(0)
        self.tensor_data[0] /= self.tensor_data[0].norm()
        for ii in range(self.tensor_info['n_length']):
            measure_data[ii] = wf.tensor_contract(
                wf.tensor_contract(self.tensor_data[ii], self.tensor_data[ii], [[0, -1], [0, -1]]),
                operator, [[0, 1], [1, 0]])
            if ii < self.tensor_info['n_length'] - 1:
                self.move_regular_center2next()
        return measure_data

    def measure_images_from_mps(self):
        # testing code
        probability = tc.empty((self.tensor_info['n_length'], 2))
        state = tc.empty((self.tensor_info['n_length'], 2, 2))
        if self.tensor_info['regular_center'] != 0:
            self.mps_regularization(0)
        self.tensor_data[0] /= (self.tensor_data[0]).norm()
        for ii in range(self.tensor_info['n_length']):
            probability[ii], state[ii] = tc.symeig(
                tc.einsum('ivj,iwj->vw', self.tensor_data[ii], self.tensor_data[ii]),
                eigenvectors=True)
            if ii < self.tensor_info['n_length'] - 1:
                self.move_regular_center2next()
        return probability, state

    def calculate_single_entropy(self, dot_index='all'):
        if operator.eq(dot_index, 'all'):
            dot_index = list(range(self.tensor_info['n_length']))
        elif isinstance(dot_index, int):
            dot_index = [dot_index]
        entropy = dict()
        for ii in dot_index:
            self.mps_regularization(ii)
            tmp_tensor = copy.deepcopy(self.tensor_data[ii])
            tmp_tensor = tmp_tensor / numpy.linalg.norm(tmp_tensor)
            tmp_tensor = wf.tensor_contract(tmp_tensor, tmp_tensor, [[0, -1], [0, -1]])
            tmp_spectrum = numpy.linalg.eigh(tmp_tensor)[0]
            tmp_spectrum /= numpy.sum(tmp_spectrum)
            tmp_spectrum[tmp_spectrum <= 0] = 1
            entropy[ii] = abs((tmp_spectrum * numpy.log(tmp_spectrum)).sum())
        return entropy

    def calculate_divided_entropy(self, dot_index='all'):
        if operator.eq(dot_index, 'all'):
            dot_index = list(range(self.tensor_info['n_length']))
        elif isinstance(dot_index, int):
            dot_index = [dot_index]
        dot_index = tc.tensor(dot_index).reshape(-1)
        dot_index_sorted = dot_index.sort()
        entropy = dict()
        for ii in dot_index_sorted[0]:
            self.mps_regularization(ii)
            u, s, v = wf.tensor_svd(self.tensor_data[ii], index_right=2)
            s = s[s > 0]
            s /= s.norm()
            s_tmp = s ** 2
            entropy[ii] = -tc.einsum('a,a->', s_tmp, tc.log(s_tmp))
        return entropy

    def calculate_label_entropy(self):
        entropy = list()
        for ii in range(self.tensor_info['n_length']):
            self.mps_regularization(ii)
            # tmp_tensor = tc.einsum('ijkm,ijkn->mn', [self.tensor_data[ii], self.tensor_data[ii]]).to('cpu')
            tmp_tensor = tc.einsum('ijkm,inkm->jn', [self.tensor_data[ii], self.tensor_data[ii]]).to('cpu')
            s = tc.symeig(tmp_tensor)[0]
            s[s <= 0] = 1
            entropy.append(-s.mul(tc.log(s)).sum())
            if tc.sum(tc.isnan(tc.tensor(entropy))) >= 1:
                print(entropy)
        return entropy

    def calculate_label_entropy_test(self, label_vector):
        entropy = list()
        for ii in range(self.tensor_info['n_length']):
            self.mps_regularization(ii)
            tmp_tensor = tc.einsum('ijkm,m->ijk', [self.tensor_data[ii], label_vector.to(self.device).to(self.dtype)])
            # tmp_tensor = tc.einsum('ijkm,ijkn->mn', [self.tensor_data[ii], self.tensor_data[ii]]).to('cpu')
            tmp_tensor = tc.einsum('ijk,ink->jn', [tmp_tensor, tmp_tensor]).to('cpu')
            s = tc.symeig(tmp_tensor)[0]
            s[s <= 0] = 1
            entropy.append(-s.mul(tc.log(s)).sum())
        return entropy
