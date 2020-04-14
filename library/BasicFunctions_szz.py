import numpy
import pynvml
import os
import pickle
import hashlib
import torch
import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# next line is NOT useless
from library import Parameters


def info_contact():
    """Return the information of the contact"""
    info = dict()
    info['name'] = 'Sun Zhengzhi'
    info['email'] = 'sunzhengzhi16@mails.ucas.ac.cn'
    info['affiliation'] = 'University of Chinese Academy of Sciences'
    return info
# These are from the original functions of Sun Zheng-Zhi


def get_best_gpu(device='cuda'):
    if isinstance(device, torch.device):
        return device
    elif device == 'cuda':
        pynvml.nvmlInit()
        num_gpu = pynvml.nvmlDeviceGetCount()
        memory_gpu = torch.zeros(num_gpu)
        for index in range(num_gpu):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gpu[index] = memory_info.free
        max_gpu = int(torch.sort(memory_gpu, )[1][-1])
        return torch.device('cuda:' + str(max_gpu))
    elif device == 'cpu':
        return torch.device('cpu')
    else:
        return torch.device(device)


def sort_dict(a):
    b = dict()
    dict_index = sorted(a.keys())
    for index in dict_index:
        b[index] = a[index]
    return b


def issubdict(a, b):
    pointer = True
    # check keys
    for key in a.keys():
        if not a.get(key) == b.get(key):
            pointer = False
            break
    return pointer


def save_pr_add_data(path, file, data, names):
    mkdir(path)
    if os.path.isfile(path+file):
        tmp = load_pr(path+file)
    else:
        tmp = {}
    s = open(path + file, 'wb')
    for ii in range(0, len(names)):
        tmp[names[ii]] = data[ii]
    pickle.dump(tmp, s)
    s.close()


def save_pr_del_data(path, file, names):
    mkdir(path)
    if os.path.isfile(path+file):
        tmp = load_pr(path+file)
    else:
        tmp = {}
    s = open(path + file, 'wb')
    for ii in range(0, len(names)):
        tmp.pop(names[ii])
    pickle.dump(tmp, s)
    s.close()


def name_generator_md5(path, file, input_parameter):
    file_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    file_save = file + '_' + file_time
    file_path = path + 'code_book/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    tmp_save = dict()
    input_parameter = sort_dict(input_parameter)
    number_md5 = hashlib.md5(str(input_parameter).encode(encoding='utf-8')).hexdigest()
    tmp_save[number_md5] = input_parameter
    save_pr(file_path, file_save, [input_parameter], [number_md5])
    # integrate_codebook(path, file)
    return number_md5


def fuzzy_load(para, force_mode=True, old_data=True):
    check_set = {'save_data_path', 'classifier_type'}
    outdated_keys = ('mps_normalization_mode',
                     'tensor_initialize_bond')
    if old_data:
        for key in outdated_keys:
            try:
                para.pop(key)
            except KeyError:
                pass
    if check_set.issubset(para.keys()):
        path = para['save_data_path'] + para['classifier_type'] + '/code_book/'
        all_filename = os.listdir(path)
        if len(all_filename) == 0:
            if force_mode:
                print('no parameters loaded, use default parameters')
                instruct = 'Parameters.' + para['classifier_type'].lower() + '()'
                right_para = eval(instruct)
                right_para.update(para)
                return right_para
        elif len(all_filename) >= 1:
            # print('warning, there may be alone code books. Give them a home!!')
            path_tmp = para['save_data_path'] + para['classifier_type'] + '/'
            file_tmp = para['classifier_type']
            integrate_codebook(path_tmp, file_tmp)
            all_filename = os.listdir(path)
            tmp_save = load_pr(path + all_filename[0])
            right_para = dict()
            for key in tmp_save.keys():
                if issubdict(para, tmp_save[key]):
                    right_para[key] = tmp_save[key]
            if len(right_para.keys()) == 1:
                print('load parameters')
                # somebody save me
                return list(right_para.values())[0]
            elif len(right_para.keys()) == 0:
                if force_mode:
                    print('no parameters loaded, use default parameters')
                    instruct = 'Parameters.' + para['classifier_type'].lower() + '()'
                    right_para = eval(instruct)
                    right_para.update(para)
                    return right_para
                else:
                    print('no parameters loaded')
                    return None
            else:
                # need optimization
                keys_all = set()
                for para_tmp in right_para.values():
                    keys_all = keys_all.union(set(para_tmp.keys()))
                keys_com = keys_all.difference(set(para.keys()))
                keys_wait = dict()
                for key1 in keys_com:
                    pointer, value = compare_dict_1key(list(right_para.values()), key1)
                    if not pointer:
                        keys_wait[key1] = value
                print('these keys should be specified')
                for item in keys_wait.items():
                    print(item)
                return None
    else:
        print('save_data_path and classifier_type is needed')
        return None


def integrate_codebook(path, file):
    path = path + 'code_book/'
    if not os.path.exists(path):
        os.makedirs(path)
    all_filename = os.listdir(path)
    tmp_save = dict()
    for filename in all_filename:
        if file in filename:
            tmp_load = load_pr(path + filename)
            tmp_save.update(tmp_load)
    for filename in all_filename:
        if 'code_book' not in filename:
            os.remove(path + filename)
    save_pr(path, (file + '_codebook'), list(tmp_save.values()), list(tmp_save.keys()))


# These are from BasicFunctions of Ran Shi_ju with some changes


def save_pr(path, file, data, names):
    """
    Save the data as a dict in a file located in the path
    :param path: the location of the saved file
    :param file: the name of the file
    :param data: the data to be saved
    :param names: the names of the data
    Notes: 1. Conventionally, use the suffix \'.pr\'. 2. If the folder does not exist, system will
    automatically create one. 3. use \'load_pr\' to load a .pr file
    Example:
    >>> x = 1
    >>> y = 'good'
    >>> save_pr('/test', 'ok.pr', [x, y], ['name1', 'name2'])
      You have a file '/test/ok.pr'
    >>> z = load_pr('/test/ok.pr')
      z = {'name1': 1, 'name2': 'good'}
      type(z) is dict
    """
    mkdir(path)
    # print(os.path.join(path, file))
    s = open(path+file, 'wb')
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    pickle.dump(tmp, s)
    s.close()


def load_pr(path_file, names=None):
    """
    Load the file saved by save_pr as a dict from path
    :param path_file: the path and name of the file
    :param names: the specific names of the data you want to load
    :return  the file you loaded
    Notes: the file you load should be a  \'.pr\' file.
    Example:
        >>> x = 1
        >>> y = 'good'
        >>> z = [1, 2, 3]
        >>> save_pr('.\\test', 'ok.pr', [x, y, z], ['name1', 'name2', 'name3'])
        >>> A = load_pr('.\\test\\ok.pr')
          A = {'name1': 1, 'name2': 'good'}
        >>> y, z = load_pr('\\test\\ok.pr', ['y', 'z'])
          y = 'good'
          z = [1, 2, 3]
    """
    if os.path.isfile(path_file):
        s = open(path_file, 'rb')
        if names is None:
            data = pickle.load(s)
            s.close()
            return data
        else:
            tmp = pickle.load(s)
            if type(names) is str:
                data = tmp[names]
                s.close()
                return data
            elif (type(names) is list) or (type(names) is tuple):
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                s.close()
                return tuple(data)
    else:
        return False


def mkdir(path):
    """
       Create a folder at your path
       :param path: the path of the folder you wish to create
       :return: the path of folder being created
       Notes: if the folder already exist, it will not create a new one.
    """
    path = path.strip()
    path = path.rstrip("\\")
    path_flag = os.path.exists(path)
    if not path_flag:
        os.makedirs(path)
    return path_flag


def easy_plot(x, y):

    plt.figure(figsize=(8, 4))
    plt.plot(numpy.array(x), numpy.array(y), 'b*')
    plt.plot(numpy.array(x), numpy.array(y), 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim()
    plt.title('default')
    plt.show()


def easy_plot_3d(x, y, z):

    x, y = numpy.meshgrid(numpy.array(x), numpy.array(y))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, numpy.array(z), rstride=1, cstride=1, cmap='rainbow')
    plt.show()


def compare_dict_1key(dict_list, key_com):
    pointer = True
    value = list()
    try:
        value.append(dict_list[0][key_com])
    except KeyError:
        pass
    for dict_one in dict_list:
        try:
            if dict_one[key_com] not in value:
                value.append(dict_one[key_com])
                pointer = False
        except KeyError:
            pointer = False
    return pointer, value


def seek_unique_value(a_list):
    check = []
    for xx in a_list:
        if xx not in a_list:
            check.append(xx)
    return check


if __name__ == '__main__':
    print(info_contact())
