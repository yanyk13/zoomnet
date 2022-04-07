import torch
import logging
import re
import os
import numpy as np
import random

from os.path import dirname, realpath
from os.path import join
from os.path import exists
from pprint import pprint
from seqeval.metrics import accuracy_score, classification_report


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def ctext(src, color='white'):
    """generate text with color
    :param src: text
    :param color: color
    :return: colored text
    """
    colors = {'white': '\033[0;m',
              'cyan': '\033[0;36m',
              'grey': '\033[0;37m',
              'red': '\033[0;31m',
              'green': '\033[0;32m',
              'yellow': '\033[0;33m',
              'blue': '\033[0;34m',
              'purple': '\033[0;35m',
              }

    if color not in colors.keys():
        raise ImportError('wrong color')
    else:
        return '{}{}{}'.format(colors[color], src, colors['white'])


def confirm_directory(directory):
    if not exists(directory):
        os.system('mkdir {}'.format(directory))


def relative_path(path):
    """
    :param path: relative path to current folder
    :return: real path
    """
    folder_real_path = dirname(realpath(__file__))
    return join(folder_real_path, path)


def idx_padding(batch, padding_value=0):
    """
    :param batch: tensor list
    :param padding_value:
    :return:
    """
    sample_lengths = [sample.size(0) for sample in batch]
    max_length = max(sample_lengths)
    batch_padded = torch.stack([torch.cat([sample, padding_value * torch.ones(max_length - sample_length).long()])
                                for sample, sample_length in zip(batch, sample_lengths)])
    mask = torch.stack([torch.cat([torch.ones_like(sample), 0 * torch.ones(max_length - sample_length).long()])
                        for sample, sample_length in zip(batch, sample_lengths)])
    return batch_padded, mask


def tensor_padding(batch, sort=True):
    """
    pad a list of tensor into a matrix with zero as padding value
    :param batch: a batch of encoded samples [[length, dim_encoded] ...]
    :param sort: whether sort the samples via length
    :return:
    """
    dim_encoded = batch[0].size(1)
    sample_lengths = [sample.size(0) for sample in batch]
    max_length = max(sample_lengths)

    if sort:
        sort_index, sort_batch = zip(*sorted(list(enumerate(batch)),
                                             key=lambda x: x[1].size(0),
                                             reverse=True))
        sort_lengths = torch.tensor([sample.size(0) for sample in sort_batch]).int()
    else:
        sort_batch = batch
        sort_index = check_gpu_device(torch.tensor(range(0, len(sort_batch))))
        sort_lengths = torch.tensor([sample.size(0) for sample in sort_batch]).int()

    sort_batch = torch.stack([torch.cat([sample, check_gpu_device(torch.zeros(max_length -
                                                                              sample.size(0),
                                                                              dim_encoded))], dim=0)
                              for sample in sort_batch])
    return sort_batch, sort_lengths, sort_index


def check_gpu_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def clog(key, value, color, lf='', rt=''):
    key_text = ctext(key, color)
    value_text = ctext('{}'.format(value), 'white')
    text = ' '.join([ctext(lf, 'grey'), key_text, value_text, ctext(rt, 'grey')])
    print(text)


def convert_label(label, BIO=True):
    """convert event mark label to BIEO label"""
    convert_dict = {'O': 0, 'B': 1, 'I': 2}
    new_label = label[:]
    max_mark = max(new_label)
    for mark in range(1, max_mark + 1):
        event_length = new_label.count(mark)
        count = 0
        for i, l in enumerate(new_label):
            if l == mark and count == 0:
                new_label[i] = 'B'
                count += 1
            elif l == mark:
                new_label[i] = 'I'
                count += 1
    new_label = ['O' if x == -1 or x == 0 else x for x in new_label]
    if not BIO:
        new_label = list(map(lambda x: convert_dict[x], new_label))
    return new_label


# def convert_performance(performance):
#     info = list(filter(lambda x: len(x) != 0, re.split(r'\s', performance)))
#     performance_dict = {
#         'micro_precision': float(info[11]),
#         'micro_recall': float(info[12]),
#         'micro_f1': float(info[13]),
#         'macro_precision': float(info[17]),
#         'macro_recall': float(info[18]),
#         'macro_f1': float(info[19])
#     }
#     return performance_dict


def compute_performance(ground_labels, pred_labels, loss, wlar):
    performance = classification_report(y_true=ground_labels,
                                        y_pred=pred_labels,
                                        digits=5)
    accuracy = accuracy_score(y_true=ground_labels,
                              y_pred=pred_labels)
    info = list(filter(lambda x: len(x) != 0, re.split(r'\s', performance)))
    performance_dict = {
        'accuracy': accuracy,
        'macro_precision': float(info[17]),
        'macro_recall': float(info[18]),
        'macro_f1': float(info[19]),
        'loss': loss,
        'word_level_action_ratio': wlar
    }
    return performance_dict


def compute_bilstm_performance(ground_labels, pred_labels, loss):
    # print(ground_labels[0])
    # print(pred_labels[0])
    performance = classification_report(y_true=ground_labels,
                                        y_pred=pred_labels,
                                        digits=5)
    accuracy = accuracy_score(y_true=ground_labels,
                              y_pred=pred_labels)
    info = list(filter(lambda x: len(x) != 0, re.split(r'\s', performance)))
    performance_dict = {
        'accuracy': accuracy,
        'macro_precision': float(info[17]),
        'macro_recall': float(info[18]),
        'macro_f1': float(info[19]),
        'loss': loss,
    }
    return performance_dict


def show_performance(performance_dict, tag, color):
    for key in performance_dict.keys():
        clog(key='{} {}'.format(tag, key).ljust(25),
             value='{}'.format(performance_dict[key])[:9],
             color=color)


def record_performance(performance_dict, writer, tag, global_step):
    for key in performance_dict.keys():
        writer.add_scalar(tag='{} {}'.format(tag, key),
                          scalar_value=performance_dict[key],
                          global_step=global_step)


def get_value(tensor):
    if torch.get_device(tensor) == -1:
        # print('???')
        return tensor.data.numpy()
    else:
        # print('wtf')
        return tensor.cpu().data.numpy()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



# if __name__ == '__main__':
#     from seqeval.metrics import accuracy_score, f1_score, classification_report
#
#     # test_label = [['O', 'O', 'B', 'I', 'O', 'I', 'I', 'O']]
#     # pred_label = [['O', 'O', 'B', 'I', 'O', 'I', 'B', 'O']]
#     # print(classification_report(test_label, pred_label))
#
#     test_label = [convert_label([-1, -1, 1, 1, -1, 1]), convert_label([-1, -1, 1, 1, -1, 1])]
#     pred_label = [convert_label([-1, -1, 1, 1, -1, -1]), convert_label([-1, -1, 1, 1, -1, 1])]
#     performance = classification_report(test_label, pred_label)
#     performance_dict = convert_performance(performance=performance)
#     show_performance(performance_dict, tag='test', color='blue')

