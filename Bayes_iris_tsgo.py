from sklearn import datasets
import torch as tc
import numpy as np



def normlize_data(data_input):
    data_min = np.min(data_input, axis=0)
    data_out = (data_input - data_min)
    data_max = np.max(data_out, axis=0)
    data_out = data_out / data_max
    return data_out


def discretize_data(data_input, cc):
    tmp_input = tc.tensor(data_input)
    n_feature = tmp_input.shape[1]
    for nn in range(n_feature):
        tmp_col = tmp_input[:, nn]
        ts = tc.sort(tmp_col)
        tmp_indice = ts[1].chunk(cc)
        for xx in range(cc):
            tmp_col[tmp_indice[xx]] = xx
        tmp_input[:, nn] = tmp_col
    return tmp_input.to(tc.int64)


def split_data(train_input, label_input):
    nt = round(train_input.shape[0] * 0.8)
    out_train = [train_input[:nt, :], train_input[nt:, :]]
    out_label = [label_input[:nt], label_input[nt:]]
    return out_train, out_label


class BayesNet():
    def __init__(self, device='cuda:0', dtype=tc.float64, lr1=1e-1, theta=np.pi/18, class_num=3, theta_acc=0,
                 test_data=None, test_label=None):
        tc.nn.Module.__init__(self)
        self.device = device
        self.dtype = dtype
        self.w_ma = None
        self.m_ma = None
        self.theta = theta
        self.class_num = class_num
        self.train_data = None
        self.label_data = None
        self.acc_loops = list()
        self.test_data = test_data
        self.test_label = test_label
        self.theta_acc = theta_acc
        self.loos_loops = list()
        self.initialize_net()
        self.loss = None

    def initialize_net(self):
        tc.manual_seed(1)
        self.w_ma = list()
        self.m_ma = list()
        # cl
        self.w_ma.append(tc.rand(5 * (self.class_num, ), dtype=self.dtype, device=self.device))
        self.m_ma.append(tc.empty(5 * (self.class_num, ), dtype=self.dtype, device=self.device, requires_grad=True))
        # sl
        self.w_ma.append(tc.rand(4 * (self.class_num, ), dtype=self.dtype, device=self.device))
        self.m_ma.append(tc.empty(4 * (self.class_num, ), dtype=self.dtype, device=self.device, requires_grad=True))
        tmp_m = self.w_ma[0].data.reshape(self.class_num, -1)
        self.w_ma[0].data = tc.einsum('ij,j->ij', [tmp_m, 1 / tmp_m.sum(dim=0)]).reshape(5 * (self.class_num,))
        self.w_ma[0] = tc.nn.Parameter(self.w_ma[0])
        tmp_m = self.w_ma[1].data.reshape(self.class_num, -1)
        self.w_ma[1].data = tc.einsum('ij,j->ij', [tmp_m, 1 / tmp_m.sum(dim=0)]).reshape(4 * (self.class_num,))
        self.w_ma[1] = tc.nn.Parameter(self.w_ma[1])
        self.m_ma[0].data = self.w_ma[0].data ** 0.5
        self.m_ma[1].data = self.w_ma[1].data ** 0.5

    def forward(self, x_train, y_train):
        tmp_m = self.m_ma[0].reshape(self.class_num, -1)
        tmp_norm = tmp_m.norm(dim=0)
        self.w_ma[0] = tc.einsum('ij,j->ij', [tmp_m ** 2, 1/tmp_norm**2]).reshape(5 * (self.class_num,))
        tmp_m = self.m_ma[1].reshape(self.class_num, -1)
        tmp_norm = tmp_m.norm(dim=0)
        self.w_ma[1] = tc.einsum('ij,j->ij', [tmp_m ** 2, 1 / tmp_norm ** 2]).reshape(4 * (self.class_num,))
        w1 = self.w_ma[0][(y_train.reshape(-1, 1),) + x_train.chunk(4, dim=1)]
        w2 = self.w_ma[1][x_train.chunk(4, dim=1)]
        pd = w1.mul(w2)
        nll = -tc.log(pd).sum()
        # nll = -pd.sum()
        return nll

    def get_label(self, x_test):
        tmp_m = self.m_ma[0].reshape(self.class_num, -1)
        tmp_norm = tmp_m.norm(dim=0)
        self.w_ma[0] = tc.einsum('ij,j->ij', [tmp_m ** 2, 1 / tmp_norm ** 2]).reshape(5 * (self.class_num,))
        tmp_m = self.m_ma[1].reshape(self.class_num, -1)
        tmp_norm = tmp_m.norm(dim=0)
        self.w_ma[1] = tc.einsum('ij,j->ij', [tmp_m ** 2, 1 / tmp_norm ** 2]).reshape(4 * (self.class_num,))
        nt = x_test.shape[0]
        w_test = tc.zeros((nt, self.class_num), dtype=self.dtype, device=self.device)
        for yy in range(self.class_num):
            test_label = yy * tc.ones((nt, )).to(tc.int64)
            # test_label = self.label_data
            w1 = self.w_ma[0].data[(test_label.reshape(-1, 1),) + x_test.chunk(4, dim=1)]
            w2 = self.w_ma[1].data[x_test.chunk(4, dim=1)]
            w_test[:, yy] = w1.mul(w2).reshape(-1)
        label = tc.argmax(w_test, dim=1).to('cpu')
        return label

    def calculate_acc(self, x_test, y_test):
        label_ca = self.get_label(x_test)
        acc = np.sum(y_test.cpu().numpy() == label_ca.cpu().numpy())/len(y_test)
        return acc

    def update_one_loop(self):
        # self.acc_loops.append(self.calculate_acc(self.test_data, self.test_label))
        self.loss.backward()

        grad_data0 = self.m_ma[0].grad.data.reshape(self.class_num, -1)
        tmp_norm = grad_data0.norm(dim=0) + 1e-100
        grad_data0 = tc.einsum('ij,j->ij', [grad_data0, 1/tmp_norm]).reshape(5 * (self.class_num,))
        self.m_ma[0].data = self.m_ma[0].data - np.tan(self.theta) * grad_data0
        self.m_ma[0].grad.data = tc.zeros(self.m_ma[0].grad.shape, device=self.device, dtype=self.dtype)

        grad_data1 = self.m_ma[1].grad.data.reshape(self.class_num, -1)
        tmp_norm = grad_data1.norm(dim=0) + 1e-100
        grad_data1 = tc.einsum('ij,j->ij', [grad_data1, 1 / tmp_norm]).reshape(4 * (self.class_num,))
        self.m_ma[1].data = self.m_ma[1].data - np.tan(self.theta) * grad_data1
        self.m_ma[1].grad.data = tc.zeros(self.m_ma[1].grad.shape, device=self.device, dtype=self.dtype)

        self.loss = self.forward(self.train_data, self.label_data)
        self.loos_loops.append(self.loss.detach())
        print('loss = ' + str(self.loss.detach()))

    def start_learning(self, train_data, label_data, loops=30, acc=1e-3):
        self.train_data = train_data
        self.label_data = label_data
        self.loss = self.forward(self.train_data, label_data)
        print('start with loss = ' + str(self.loss.data))
        self.loos_loops.append(self.loss.detach())
        for loop in range(loops):
            print('start ' + str(loop) + ' loops')
            self.update_one_loop()
            is_converge = self.is_converge(acc)
            if is_converge:
                break

    def is_converge(self, acc):
        if (self.loos_loops[-2] - self.loos_loops[-1]) / abs(self.loos_loops[-2]) <= 1e-3:
            if self.theta > self.theta_acc:
                self.theta = self.theta * 0.5
                return False
            else:
                return False
        else:
            return False


x_data = tc.tensor(datasets.load_iris().data).to(tc.float64).cuda()
y_data = tc.tensor(datasets.load_iris().target).to(tc.int64).cuda()
x_data = discretize_data(x_data, 3)
dd_out, aa_out = split_data(x_data, y_data)


result = list()
acc_list0 = list()
acc_list1 = list()
for theta in [18, 6, 3]:
    A = BayesNet(theta=np.pi/theta, class_num=3, device='cpu')
    A.start_learning(dd_out[0], aa_out[0], loops=100)
    result.append(A.loos_loops)
    ac0 = A.calculate_acc(dd_out[0], aa_out[0])
    ac1 = A.calculate_acc(dd_out[1], aa_out[1])
    acc_list0.append(ac0)
    acc_list1.append(ac1)
# result.append(A.acc_loops)
print(acc_list0)
print(acc_list1)
np.savetxt('tsgo.txt', np.array(result).T)
# np.savetxt('tmp.txt', A.w_ma.data.cpu().numpy().reshape(-1))
