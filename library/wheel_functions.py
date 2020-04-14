import numpy as np
import os
import torch as tc


def outer_parallel(a, *matrix):
    # need optimization
    for b in matrix:
        a = (a.repeat(b.shape[1], 1).reshape(a.shape + (-1,))
             * b.repeat(a.shape[1], 0).reshape(a.shape + (-1,))).reshape(a.shape[0], -1)
    return a


def outer(a, *matrix):
    for b in matrix:
        a = np.outer(a, b).flatten()
    return a


def generate_tensor(tensor_size, initial_type='rand', device='cuda', dtype=tc.float64):
    if initial_type == 'rand':
        tensor_tmp = tc.rand(tensor_size, device=device, dtype=dtype)
    elif initial_type == 'ones':
        tensor_tmp = tc.ones(tensor_size, device=device, dtype=dtype)
    else:
        tensor_tmp = None
    return tensor_tmp


def split_2d(input_size=(4, 4), output_size=(2, 2)):
    input_num = np.prod(input_size)
    in_sq = tc.arange(0, input_num).reshape(input_size)
    out_sq = list()
    split0 = in_sq.chunk(output_size[0], dim=0)
    for ii in range(output_size[0]):
        tmp_split = split0[ii]
        out_sq.append(tmp_split.chunk(output_size[1], dim=1))
    return out_sq


def tensor_contract(a, b, index):
    ndim_a = np.array(a.shape)
    ndim_b = np.array(b.shape)
    order_a = np.arange(len(ndim_a))
    order_b = np.arange(len(ndim_b))
    order_a_contract = np.array(order_a[index[0]]).flatten()
    order_b_contract = np.array(order_b[index[1]]).flatten()
    order_a_hold = np.setdiff1d(order_a, order_a_contract)
    order_b_hold = np.setdiff1d(order_b, order_b_contract)
    hold_shape_a = ndim_a[order_a_hold].flatten()
    hold_shape_b = ndim_b[order_b_hold].flatten()
    return np.dot(
        a.transpose(np.concatenate([order_a_hold, order_a_contract])).reshape(hold_shape_a.prod(), -1),
        b.transpose(np.concatenate([order_b_contract, order_b_hold])).reshape(-1, hold_shape_b.prod()))\
        .reshape(np.concatenate([hold_shape_a, hold_shape_b]))


def tensor_svd(tmp_tensor, index_left='none', index_right='none'):
    tmp_shape = np.array(tmp_tensor.shape)
    tmp_index = np.arange(len(tmp_tensor.shape))
    if index_left == 'none':
        index_right = tmp_index[index_right].flatten()
        index_left = np.setdiff1d(tmp_index, index_right)
    elif index_right == 'none':
        index_left = tmp_index[index_left].flatten()
        index_right = np.setdiff1d(tmp_index, index_left)
    index_right = np.array(index_right).flatten()
    index_left = np.array(index_left).flatten()
    tmp_tensor = tmp_tensor.permute(tuple(np.concatenate([index_left, index_right])))
    tmp_tensor = tmp_tensor.reshape(tmp_shape[index_left].prod(), tmp_shape[index_right].prod())
    u, l, v = tc.svd(tmp_tensor)
    return u, l, v


def tensor_qr(tmp_tensor, index_left='none', index_right='none'):
    tmp_shape = np.array(tmp_tensor.shape)
    tmp_index = np.arange(len(tmp_tensor.shape))
    if index_left == 'none':
        index_right = tmp_index[index_right].flatten()
        index_left = np.setdiff1d(tmp_index, index_right)
    elif index_right == 'none':
        index_left = tmp_index[index_left].flatten()
        index_right = np.setdiff1d(tmp_index, index_left)
    index_right = np.array(index_right).flatten()
    index_left = np.array(index_left).flatten()
    tmp_tensor = tmp_tensor.permute(tuple(np.concatenate([index_left, index_right])))
    tmp_tensor = tmp_tensor.reshape(tmp_shape[index_left].prod(), tmp_shape[index_right].prod())
    q, r = tc.qr(tmp_tensor)
    return q, r


def print_shape(*xx):
    for x in xx:
        print(x.shape)


def calculate_mse(image_origin, image_noised):
    return ((image_origin - image_noised) ** 2) / np.prod(image_origin.shape)


def calculate_psnr(image_origin, image_noised):
    # not very correct
    image_origin = image_origin.flatten()
    image_noised = image_noised.flatten()
    return 20 * np.log10(
        np.concatenate((image_origin, image_noised)).max()) - 10 * np.log10(
        ((image_origin - image_noised) ** 2).sum() / np.prod(image_origin.shape))


def calculate_ssim(im1, im2):
    # testing code
    if im1.max() <= 1:
        im1 = 255 * im1.copy()
    if im2.max() <= 1:
        im2 = 255 * im2.copy()
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma12 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    ssim = l12 * c12
    return ssim


def noise_image(image_origin, noise_type='Gaussian', means=0, sigma=1):
    pixel_max = image_origin.max()
    pixel_min = image_origin.min()


def read_data_tmp(data_path=None, file_path=None,
                  right_head='EB 90 B7 13',
                  right_end='EE EE DD DD',
                  len_head=4,
                  len_end=4):
    if data_path is None:
        data_path = '../../data_python/laser_data/'
    if file_path is None:
        file_path = 'T3-10s/'
    full_path = data_path + file_path
    all_filename = os.listdir(full_path)
    data = bytes()
    for filename in sorted(all_filename):
        with open(full_path + filename, mode='rb') as tmp:
            data += tmp.read()
    if not (data[:len_head] == bytes.fromhex(right_head)):
        print('head wrong')
        return None
    elif not (data[-len_end:] == bytes.fromhex(right_end)):
        print('end wrong')
        return None
    elif not ((len(data) % 4) == 0):
        print('length wrong')
        return None
    else:
        return np.array(bytearray(data[len_head: -len_end]), dtype=np.uint16).reshape(-1, 4).T