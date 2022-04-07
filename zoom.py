import torch
import time
import numpy as np
import json
from torch import nn
from torch.nn import functional
from torch import optim
from torch.utils.checkpoint import checkpoint
from functools import reduce
from transformers import BertModel, XLNetModel
from os.path import join
from math import ceil
from prettytable import PrettyTable
from pprint import pprint

from data import TOFRData, DataLoader, collate
from utils import set_global_logging_level
from utils import check_gpu_device
from utils import idx_padding, tensor_padding, ctext, clog
from utils import relative_path, confirm_directory

from utils import show_performance, record_performance
from seqeval.metrics import accuracy_score, classification_report


class Configure(object):
    def __init__(self, path, default=False):
        if default:
            """folders and paths"""
            self.hyper = 'sentence_only'
            self.data = relative_path('./data/criminal')

            """model setting"""
            self.vocab_size = 2977
            self.dim_unit = 3
            self.dim_word = 128
            self.dim_sentence = 256
            self.dim_paragraph = 512
            self.dim_location = 3
            self.dim_state = 512
            self.dim_scorer_hid = 256
            self.dim_sym = 9
            self.num_action_type = 3
            self.score_mode = 'softmax'
            self.seed = 1135

            """training setting"""
            self.rl_lambda = 1
            self.random_sampling = 1
            self.learning_rate = 1e-5
            self.weight_decay = 1e-4
            self.gpu_id = 4
            self.early_stop_delay_margin = 200

            """data setting"""
            self.document_batch_size = 1
            self.max_epoch = 20000
        else:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            for key in config_dict.keys():
                if isinstance(config_dict[key], str):
                    exec("self.{}='{}'".format(key, config_dict[key]))
                else:
                    exec("self.{}={}".format(key, config_dict[key]))

        name = "ZoomNet"
        time_tag = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.tag = '{}_{}_{}'.format(name, self.hyper, time_tag)

    def inform(self):
        config_table = PrettyTable(['hyper-parameters',
                                    'value',
                                    ])
        for key in self.__dict__.keys():
            config_table.add_row([key, self.__dict__[key]])
        print(config_table)

    def mark_down(self):
        lines = ['|hyper-parameters|value|\n',
                 '|----------------|-----|\n']
        for key in self.__dict__.keys():
            lines.append('|{}|{}|\n'.format(key, self.__dict__[key]))
        return ''.join(lines)


class BERTEncoder(nn.Module):

    def __init__(self, pretrained_path):
        super(BERTEncoder, self).__init__()
        self.pretrained_path = pretrained_path
        self._init_components()
        if torch.cuda.is_available():
            self.cuda()

    def _init_components(self):
        self.bert = BertModel.from_pretrained(self.pretrained_path)

    def forward(self, batch, drop_out=None, segment_length=512):
        batch_idx = batch['bert_idx']
        batch_mask = batch['mask']
        batch_idx = check_gpu_device(batch_idx)
        batch_mask = check_gpu_device(batch_mask)
        batch_length = batch_idx.size(1)
        num_seg = ceil(batch_length/segment_length)
        encoded = []
        for i in range(num_seg):
            init_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, batch_length)
            seg_idx = batch_idx[:, init_idx: end_idx]
            seg_mask = batch_mask[:, init_idx: end_idx]
            seg_encoded, _ = self.bert.forward(input_ids=seg_idx,
                                               attention_mask=seg_mask,
                                               return_dict=False)
            encoded.append(seg_encoded)
        encoded = torch.cat(encoded, dim=1)
        if drop_out != None:
            encoded = drop_out(encoded)
        return encoded

    def sample_forward(self, sample, segment_length=512):
        sample_idx = sample['bert_idx']
        sample_idx = check_gpu_device(sample_idx).unsquezze(0)
        sample_length = sample_idx.size(1)
        num_seg = ceil(sample_length / segment_length)
        encoded = []
        for i in range(num_seg):
            init_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, sample_length)
            seg_idx = sample_idx[:, init_idx: end_idx]
            seg_encoded, _ = self.bert.forward(input_ids=seg_idx)
            encoded.append(seg_encoded)
        encoded = torch.cat(encoded, dim=1)
        return encoded


class Mlp(nn.Module):

    def __init__(self, dim_in, dim_hid, dim_out):
        super(Mlp, self).__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        self._init_components()

    def _init_components(self):
        self.liner1 = nn.Linear(in_features=self.dim_in,
                                out_features=self.dim_hid)
        self.liner2 = nn.Linear(in_features=self.dim_hid,
                                out_features=self.dim_out)

    def forward(self, feature):
        """
        :param feature: [batch size, length, dim_in] torch.FloatTensor
        :return: [batch size, length, dim_out] torch.FloatTensor with no activation function
        """
        hidden = torch.relu(self.liner1(feature))
        out = self.liner2(hidden)
        return out


class ScorerGroup(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, mode):
        """
        :param dim_in:
        :param dim_hid:
        :param dim_out:
        :param mode: use sigmoid / softmax
        """
        super(ScorerGroup, self).__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        self.mode = mode
        self._init_params()

    def _init_params(self):
        self.mlps = nn.ModuleList([Mlp(dim_in=self.dim_in,
                                       dim_hid=self.dim_hid,
                                       dim_out=1) for _ in range(self.dim_out)])

    def forward(self, in_feature):
        if self.mode == 'sigmoid':
            scores = torch.sigmoid(torch.cat([mlp.forward(in_feature) for mlp in self.mlps]))
        elif self.mode == 'softmax':
            scores = torch.softmax(torch.cat([mlp.forward(in_feature) for mlp in self.mlps]), dim=0)
        else:
            raise ImportError('Not a legal score mode')
        return scores


class UpperEncoder(nn.Module):
    def __init__(self, dim_basic, dim_upper):
        super(UpperEncoder, self).__init__()
        self.dim_basic = dim_basic
        self.dim_upper = dim_upper
        self._init_params()

    def _init_params(self):
        self.bilstm = nn.LSTM(input_size=self.dim_basic,
                              hidden_size=self.dim_upper // 2,
                              bidirectional=True,
                              batch_first=True)

    @staticmethod
    def get_basic_sections(sample_basic_memory, single_upper_boundary):
        """
        divide the sample memory of one sample at basic level into several groups corresponding to the
        boundaryments spans
        :param sample_basic_memory: tensor | length * dim_basic
        :param single_upper_boundary: int-list | spans_list
        :return:
        """
        single_upper_boundary_ss = single_upper_boundary[:-1]
        single_upper_boundary_es = single_upper_boundary[1:]
        single_unit_memory = [sample_basic_memory[boundary_s: boundary_e] for boundary_s, boundary_e in
                              zip(single_upper_boundary_ss, single_upper_boundary_es)]
        return single_unit_memory

    @staticmethod
    def reorganize_memory(unpack_memory, sorted_index, num_sample_upper):
        """
        :param unpack_memory: upper memory
        :param sorted_index:
        :param num_sample_upper:
        :return:
        """
        new_unpack_memory = check_gpu_device(torch.zeros_like(unpack_memory))
        for i, ix in enumerate(sorted_index):
            new_unpack_memory[ix, :, :] = unpack_memory[i, :, :]
        num_sample_upper.insert(0, 0)
        seg_idx = [sum(num_sample_upper[:i]) for i
                   in range(1, len(num_sample_upper) + 1)]
        memory_list = [unpack_memory[idx_s: idx_e] for idx_s, idx_e
                       in zip(seg_idx[:-1], seg_idx[1:])]
        level_memory = [torch.max(memory, dim=1)[0] for memory in memory_list]
        return level_memory

    def forward(self, basic_memory, upper_boundary):
        """
        :param basic_memory: encoded basic memory | tensor | batch_size * length * dim_basic
        :param upper_boundary: spans of upper unit | list of int-list | [spans_list] * batch_size
        :return:
        """
        batch_size = basic_memory.size(0)
        num_uppers = [len(x) - 1 for x in upper_boundary]
        # num_uppers: the number of upper level language units in each sample of one batch
        batch_upper = [self.get_basic_sections(single_upper_boundary=upper_boundary[i],
                                               sample_basic_memory=basic_memory[i]) for i in range(batch_size)]
        # batch_upper: list of list of basic memory of one single upper unit
        upper = reduce(lambda x, y: x + y, batch_upper)
        # concatenate elements of batch_upper to wrap out sample bounds
        sorted_upper, sorted_length, sorted_index = tensor_padding(upper)
        # can not use 'input=' in pack_padded_sequence dual to the arg[]. should report
        upper_pack = nn.utils.rnn.pack_padded_sequence(sorted_upper,
                                                       lengths=sorted_length,
                                                       batch_first=True)
        upper_encoded, _ = self.bilstm(upper_pack)
        upper_unpack, _ = nn.utils.rnn.pad_packed_sequence(upper_encoded,
                                                           batch_first=True)
        upper_memory = self.reorganize_memory(upper_unpack.contiguous(), sorted_index, num_uppers)
        # contiguous: make sure using a single block of memory
        upper_memory, _, _ = tensor_padding(upper_memory, sort=False)  # DO NOT SORT BY LENGTH
        return upper_memory


class ZoomEncoder(nn.Module):

    def __init__(self, config):
        super(ZoomEncoder, self).__init__()
        self.config = config
        self.dim_word = config.dim_word
        self.num_vocab = config.vocab_size
        self.dim_sentence = config.dim_sentence
        self.dim_paragraph = config.dim_paragraph
        self.document_batch_size = config.document_batch_size
        self.bert_path = config.bert
        self.mode = config.mode
        self._init_components()
        if torch.cuda.is_available():
            self.cuda()

    def _init_components(self):
        self.embedding = nn.Embedding(num_embeddings=self.config.vocab_size,
                                      embedding_dim=self.config.dim_word)
        self.bert_encoder = BERTEncoder(pretrained_path=self.bert_path)
        self.sentence_encoder = UpperEncoder(dim_basic=self.dim_word,
                                             dim_upper=self.dim_sentence)
        self.paragraph_encoder = UpperEncoder(dim_basic=self.dim_sentence,
                                              dim_upper=self.dim_paragraph)

    # def form_sentence_batches(self, sentences_padded, sentences_att_mask):
    #     num_sentences = sentences_padded.size(0)
    #     sentences_batch_num = ceil(num_sentences / self.sentence_batch_size)
    #     for i in range(sentences_batch_num):
    #         init_idx = i * self.sentence_batch_size
    #         end_idx = min((i + 1) * self.sentence_batch_size, num_sentences)
    #         yield sentences_padded[init_idx:end_idx, :], sentences_att_mask[init_idx: end_idx, :]

    @staticmethod
    def reform_content_encoded(batch_encoded, num_sentences, sentences_len):
        shift = 0
        samples_content_encoded = []
        for sample_idx, num in enumerate(num_sentences):
            sample_encoded = []
            for length in sentences_len[sample_idx]:
                sentence_encoded = batch_encoded[shift, :length, :]
                sample_encoded.append(sentence_encoded)
                shift += 1
            sample_encoded = torch.cat(sample_encoded, dim=0)
            samples_content_encoded.append(sample_encoded)
        batch_encoded, _, _ = tensor_padding(samples_content_encoded, sort=False)
        return batch_encoded

    def forward(self, batch):
        """
        :param batch:
        :return:
        """
        sentence_boundary = batch['sentence_boundaries']
        paragraph_boundary = batch['paragraph_boundaries']
        simple_idx_padded = check_gpu_device(batch['simple_idx'])
        bert_idx_padded = check_gpu_device(batch['bert_idx'])
        if self.config.mode == 'simple':
            batch_encoded = self.embedding(simple_idx_padded)
        else:
            batch_encoded = self.bert_encoder.forward(batch)
        batch_sentence_encoded = self.sentence_encoder.forward(basic_memory=batch_encoded,
                                                               upper_boundary=sentence_boundary)
        batch_paragraph_encoded = self.paragraph_encoder.forward(basic_memory=batch_sentence_encoded,
                                                                 upper_boundary=paragraph_boundary)

        return batch_encoded, batch_sentence_encoded, batch_paragraph_encoded

    def save(self, hyper, idx=0):
        tag = self.config.tag
        confirm_directory('./check_points')
        folder = relative_path('./check_points/{}_{}'.format(tag.split(' ')[0], hyper))
        confirm_directory(folder)
        path = join(folder, 'encoder_{}'.format(idx) + '.pth')
        confirm_directory(folder)
        torch.save(self.state_dict(), path)
        print('{} saved'.format(path))

    def load(self, hyper, idx=0, special=False):
        tag = self.config.tag
        folder = relative_path('./check_points/{}_{}'.format(tag.split(' ')[0], hyper))
        if special:
            path = hyper
        else:
            path = join(folder, 'encoder_{}'.format(idx) + '.pth')
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def show_configure(self):
        self.config.inform()


class ZoomAgent(nn.Module):
    def __init__(self, config):
        super(ZoomAgent, self).__init__()
        self.config = config
        self.dim_word = config.dim_word
        self.dim_sentence = config.dim_sentence
        self.dim_paragraph = config.dim_paragraph
        self.dim_state = config.dim_state
        self.dim_location = 3
        self.dim_symbolic = config.dim_sym
        self.num_action_type = config.num_action_type
        self.dim_actions = config.dim_unit * config.num_action_type
        self.dim_in = 2 * (self.dim_word +
                           self.dim_sentence +
                           self.dim_paragraph) + self.dim_actions + self.dim_symbolic
        self.dim_scorer_hid = config.dim_scorer_hid
        self.score_mode = config.score_mode
        self.random_sampling = config.random_sampling
        self._init_components()
        if torch.cuda.is_available():
            self.cuda()
        """initialize state"""
        self.location = [0] * config.dim_unit
        self.previous_action = 0
        self.current_mark = 0
        self.action_count = np.zeros([self.num_action_type, 3])
        self.loss = check_gpu_device(torch.tensor(0.0))
        self.policy_pies = []
        self.pred_label = 0
        self.predict_label = []
        self.ht, self.ct = self.init_rnn_state(dim_hidden=self.config.dim_state,
                                               bidirectional=False)
        self.continue_flag = True

    def _init_components(self):
        self.state_updater = nn.LSTM(input_size=self.dim_in,
                                     hidden_size=self.dim_state)
        self.scorer_group = ScorerGroup(dim_in=self.dim_state,
                                        dim_hid=self.dim_scorer_hid,
                                        dim_out=self.config.dim_unit * self.num_action_type,
                                        mode=self.score_mode)
        self.layer_norm = nn.LayerNorm(normalized_shape=[self.dim_state])

    def _init_status(self):
        self.location = [0] * 3
        self.previous_action = 0
        self.current_mark = 0
        self.action_count = np.zeros([self.num_action_type, 3])
        self.loss = check_gpu_device(torch.tensor(0.0))
        self.policy_pies = []
        self.predict_label = []
        self.ht, self.ct = self.init_rnn_state(dim_hidden=self.config.dim_state,
                                               bidirectional=False)
        self.continue_flag = True

    @staticmethod
    def init_rnn_state(dim_hidden, bidirectional):
        dim_direction = 2 if bidirectional else 1
        ht = check_gpu_device(torch.zeros(dim_direction, 1, int(dim_hidden/dim_direction)))
        ct = check_gpu_device(torch.zeros(dim_direction, 1, int(dim_hidden/dim_direction)))
        return ht, ct

    def _compute_step_loss(self, prob, action_label, location, next_location):
        if self.score_mode == 'sigmoid':
            loss = sum([torch.log(prob[i]) if action_label[i] == 1 else torch.log(1 - prob[i])
                        for i in range(len(action_label))]) / (3 * self.config.dim_unit)

        elif self.score_mode == 'softmax':
            num_choice = 0
            loss = check_gpu_device(torch.tensor(0.0))
            prob_1 = check_gpu_device(torch.tensor(0.0))
            for i, label in enumerate(action_label):
                if label == 1:
                    prob_1 += prob[i]
                else:
                    num_choice += 1
                    loss += torch.log(1 - prob[i])
            if 1 in action_label:
                loss = torch.log(prob_1)
            else:
                loss = check_gpu_device(torch.tensor(0.0))
            loss = loss * (next_location[0] - location[0])
        else:
            raise ImportError('illegal score mode')
        return - loss

    def _compute_sl_loss(self):
        count_action = check_gpu_device(torch.tensor(np.sum(np.sum(self.action_count)))).float()
        loss = torch.sum(torch.sum(self.loss)) / count_action
        return loss

    def _compute_sum_log_pie(self):
        count_action_level = np.sum(self.action_count, axis=1)
        reward = np.sum(count_action_level * np.array([-1, 0, 0])) / np.sum(count_action_level)
        word_action_ratio = count_action_level[0] / np.sum(count_action_level)
        count_action = np.sum(count_action_level)
        policy_pies = torch.cat(self.policy_pies)
        log_sum_pie = -(torch.sum(policy_pies)) / float(count_action)
        return reward, log_sum_pie, word_action_ratio

    def _compute_next_location(self, num_word, sentence_boundaries, paragraph_boundaries, action):
        num_action_type = self.num_action_type
        """generate new location based on action selected
               pay attention for skip path"""

        def ex_st_range(sentence_span, l):
            return list(range(sentence_span[l[1]], sentence_span[l[1] + 1]))

        def ex_prg_range(paragraph_span, l):
            return list(range(paragraph_span[l[2]], paragraph_span[l[2] + 1]))

        max_wix = num_word
        max_stix = len(sentence_boundaries) - 1
        max_prgix = len(paragraph_boundaries) - 1
        new_location = self.location[:]
        if action in range(0, num_action_type):
            # print('word action')
            new_location[0] = self.location[0] + 1
            if new_location[0] not in ex_st_range(sentence_boundaries, self.location):
                new_location[1] = self.location[1] + 1

                if new_location[1] not in ex_prg_range(paragraph_boundaries, self.location):
                    new_location[2] = self.location[2] + 1

        elif action in range(num_action_type, 2 * num_action_type):
            # print('sentence action')
            new_location[1] = self.location[1] + 1
            if new_location[1] < max_stix:
                new_location[0] = sentence_boundaries[new_location[1]]
                if new_location[1] not in ex_prg_range(paragraph_boundaries, self.location):
                    new_location[2] = self.location[2] + 1
            else:
                new_location = [max_wix, max_stix - 1, max_prgix - 1]

        elif action in range(2 * num_action_type, 3 * num_action_type):
            # print('paragraph action')
            new_location[2] = self.location[2] + 1
            if new_location[2] < max_prgix:
                new_location[1] = paragraph_boundaries[new_location[2]]
                new_location[0] = sentence_boundaries[new_location[1]]
            else:
                new_location = [max_wix, max_stix - 1, max_prgix - 1]

        return new_location

    def _gen_pred_label(self, action, next_location):
        """generate identify label based on the chosen action"""
        if action % 3 == 0:
            pred_label = [-1] * (next_location[0] - self.location[0])
        elif action % 3 == 1:
            pred_label = [self.current_mark] * (next_location[0] - self.location[0])
        else:
            pred_label = [self.current_mark + 1] * (next_location[0] - self.location[0])
        return pred_label

    def _ex_fragment_label(self, label, next_location):
        return label[self.location[0]:next_location[0]]

    def _gen_action_label(self, num_word, sentence_boundaries, paragraph_boundaries, label):
        """generate action label for this time step based on the consequence after corresponding action"""
        action_labels = []
        for action in range(3 * self.config.dim_unit):
            next_location = self._compute_next_location(num_word, sentence_boundaries, paragraph_boundaries, action)
            pred_label = self._gen_pred_label(action, next_location)
            ground_label = self._ex_fragment_label(label, next_location)
            if pred_label == ground_label:
                action_labels.append(1)
            else:
                action_labels.append(0)

        return action_labels

    def _gen_skip_action(self, prob, mask):
        """mask: which actions are acceptable at current time step
           action: the action chosen to skip"""
        random_flag = self.random_sampling
        mask = check_gpu_device(torch.tensor(mask, requires_grad=False)).float()
        prob = prob * mask
        if random_flag:
            """action random selection"""
            action = torch.multinomial(prob, num_samples=1)
        else:
            """action deterministic selection"""
            _, action = torch.max(prob, dim=0)
        if torch.cuda.is_available():
            action = action.data.cpu().numpy()
        else:
            action = action.data.numpy()
        return action

    @staticmethod
    def _convert_label(label, bmeo=False):
        """convert event mark label to BMEO label"""
        convert_dict = {'O': 0, 'B': 1, 'M': 2, 'E': 3}
        new_label = label[:]
        max_mark = max(new_label)
        for mark in range(1, max_mark + 1):
            event_length = new_label.count(mark)
            count = 0
            for i, simple_label in enumerate(new_label):
                if simple_label == mark and count == 0:
                    new_label[i] = 'B'
                    count += 1
                elif simple_label == mark and count == event_length - 1:
                    new_label[i] = 'E'
                elif simple_label == mark:
                    new_label[i] = 'M'
                    count += 1
        new_label = ['O' if x == -1 or x == 0 else x for x in new_label]
        if not bmeo:
            new_label = list(map(lambda x: convert_dict[x], new_label))

        return new_label

    @staticmethod
    def _gen_pred_action(prob):
        _, action = torch.max(prob, dim=0)
        if torch.cuda.is_available():
            action = action.data.cpu().numpy()
        else:
            action = action.data.numpy()
        return action

    @staticmethod
    def _show_result(sample, label):
        print(len(sample), len(label))
        # assert len(sample) == len(label)
        text = ''
        for word, simple_label in zip(sample, label):
            if simple_label == 2:
                text += ctext(word, 'gray')
            elif simple_label == 1 or simple_label == 3 or simple_label == 4:
                text += ctext(word, 'blue')
            else:
                text += ctext(word, 'green')
        print(text)

    def forward(self, memory, location, htm1, ctm1, previous_action, sym_mark='n'):
        action_vector = check_gpu_device(torch.tensor([0] * self.dim_actions, requires_grad=False)).float()
        action_vector[previous_action] = 1

        """convert symbolic memory"""
        sym_vector = check_gpu_device(torch.tensor([0.] * 3 * self.num_action_type, requires_grad=False))
        if sym_mark != 'n':
            sym_vector[int(sym_mark)] = 1
        input_controller = torch.cat([memory[0][location[0]],
                                      memory[1][location[1]],
                                      memory[2][location[2]],
                                      action_vector
                                      ], dim=0)
        memory_next_unit = torch.cat([memory[i][location[i] + 1] if location[i] + 1 < memory[i].size(0) else
                                      check_gpu_device(torch.zeros(dim)) for i, dim in
                                      enumerate([self.dim_word, self.dim_sentence, self.dim_paragraph])], dim=0)

        in_feature = torch.cat([input_controller, memory_next_unit, sym_vector], dim=0)
        in_feature = in_feature.view(1, 1, -1)
        state_controller, (ht, ct) = self.state_updater.forward(in_feature, (htm1, ctm1))
        state_controller = state_controller.view(-1)

        prob = self.scorer_group.forward(state_controller)
        # prob = torch.clamp(prob, 1e-6, 1 - 1e-6)
        pie = torch.log(prob)
        return pie, prob, ht, ct

    def sample_forward(
            self,
            content,
            label,
            memory,
            sentence_boundaries,
            paragraph_boundaries,
            mode='train',
            show_result=False,
            show_action_level=False
    ):
        num_action = 0  # tmp
        content = ''.join(content)
        num_word = len(content)
        self._init_status()

        action_sequence = ''

        while self.continue_flag:
            # print(self.location)
            # print('sentences', memory[1].size())
            # print('paragraphs', memory[2].size())
            pie, prob, self.ht, self.ct = self.forward(memory=memory,
                                                       location=self.location,
                                                       htm1=self.ht,
                                                       ctm1=self.ct,
                                                       previous_action=self.previous_action)
            """check which actions are acceptable"""
            action_label = self._gen_action_label(num_word=num_word,
                                                  sentence_boundaries=sentence_boundaries,
                                                  paragraph_boundaries=paragraph_boundaries,
                                                  label=label)
            if mode == 'train':
                assert sum(action_label) != 0
            """skip_action: the action we use to update location [does note exist while validation and testing]"""
            if mode == 'train':
                chosen_action = self._gen_skip_action(prob, mask=action_label)
            elif mode == 'eval':
                chosen_action = self._gen_pred_action(prob)
            else:
                raise ImportError('Wrong forward mode')

            """save log(pie(action|station))"""
            self.policy_pies.append(pie[chosen_action].view(1, -1))

            """generate new location based on skip action"""
            next_location = self._compute_next_location(action=chosen_action,
                                                        num_word=num_word,
                                                        sentence_boundaries=sentence_boundaries,
                                                        paragraph_boundaries=paragraph_boundaries)
            section_text = content[self.location[0]: next_location[0]]

            """generate predicted label based on pred_action"""
            step_label = self._gen_pred_label(action=chosen_action,
                                              next_location=next_location)

            """add predicted label to previous labels"""
            self.predict_label.extend(step_label)

            """compute current loss"""
            step_loss = self._compute_step_loss(prob, action_label=action_label,
                                                location=self.location,
                                                next_location=next_location)
            self.action_count[int(chosen_action / 3), chosen_action % 3] += 1
            num_action += 1
            # self.loss[int(chosen_action/3), chosen_action % 3] += step_loss

            self.loss += step_loss

            if chosen_action in range(3):
                action_sequence += ctext(section_text, 'grey')
            elif chosen_action in range(3, 6):
                action_sequence += ctext(section_text, 'blue')
            else:
                action_sequence += ctext(section_text, 'yellow')

            """update location & check whether reach the bottom"""
            self.location = next_location[:]

            if self.location[0] == num_word:
                self.continue_flag = False

            """update previous action"""
            self.previous_action = chosen_action

            """update current mark"""
            if mode == 'train':
                self.current_mark = max(max(label[:self.location[0]]), 0)
            elif mode == 'validate' or 'test':
                self.current_mark = max(max(self.predict_label), 0)
            else:
                raise ImportError('Wrong forward mode')

        # sl_loss = self._compute_sl_loss()/(self.location[0])
        sl_loss = self.loss / num_word
        reward, log_sum_pie, word_action_ratio = self._compute_sum_log_pie()
        self.predict_label = self._convert_label(self.predict_label)
        label = self._convert_label(label)

        result = list(zip(label, self.predict_label, content))
        coma_list = ['：', '，', '。', '；', '\n']
        self.pred_label = [l_p if w not in coma_list else l for l, l_p, w in result]

        if show_result:
            print(ctext('Ground Truth', 'green'))
            self._show_result(content, label=label)
            print(ctext('Predicted', 'green'))
            self._show_result(content, label=self.predict_label)
        if show_action_level:
            print(action_sequence)
        return sl_loss, reward, log_sum_pie, word_action_ratio, self.predict_label, label

    def save(self, id_):
        tag = self.config.tag
        confirm_directory('./check_points')
        folder = relative_path('./check_points/{}'.format(tag.split(' ')[0]))
        confirm_directory(folder)
        path = join(folder, 'agent_{}'.format(id_) + '.pth')
        confirm_directory(folder)
        torch.save(self.state_dict(), path)
        return 'agent_{}_{}'.format(tag, id_)

    def load(self, id_):
        tag = self.config.tag
        folder = relative_path('./check_points/{}'.format(tag.split(' ')[0]))
        path = join(folder, 'agent_{}'.format(id_) + '.pth')
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def show_configure(self):
        self.config.inform()


class ZoomClassifier(nn.Module):
    def __init__(self, config):
        super(ZoomClassifier, self).__init__()
        self.config = config
        self.dim_word = config.dim_word
        self.dim_sentence = config.dim_sentence
        self.dim_paragraph = config.dim_paragraph
        self.dim_state = config.dim_state
        self.dim_location = 3
        self.dim_symbolic = config.dim_sym
        self.num_action_type = config.num_action_type
        self.dim_actions = config.dim_unit * config.num_action_type
        self.dim_in = 2 * (self.dim_word +
                           self.dim_sentence +
                           self.dim_paragraph) + self.dim_actions + self.dim_symbolic
        self.dim_scorer_hid = config.dim_scorer_hid
        self.score_mode = config.score_mode
        self.random_sampling = config.random_sampling
        self._init_components()
        if torch.cuda.is_available():
            self.cuda()
        """initialize state"""
        self.location = [0] * config.dim_unit
        self.previous_action = 0
        self.current_mark = 0
        self.action_count = np.zeros([self.num_action_type, 3])
        self.loss = check_gpu_device(torch.tensor(0.0))
        self.policy_pies = []
        self.pred_label = 0
        self.predict_label = []
        self.ht, self.ct = self.init_rnn_state(dim_hidden=self.config.dim_state,
                                               bidirectional=False)
        self.continue_flag = True

    def _init_components(self):
        self.state_updater = nn.LSTM(input_size=self.dim_in,
                                     hidden_size=self.dim_state)
        self.scorer_group = ScorerGroup(dim_in=self.dim_state,
                                        dim_hid=self.dim_scorer_hid,
                                        dim_out=self.config.dim_unit * self.num_action_type,
                                        mode=self.score_mode)
        self.layer_norm = nn.LayerNorm(normalized_shape=[self.dim_state])

    def _init_status(self):
        self.location = [0] * 3
        self.previous_action = 0
        self.current_mark = 0
        self.action_count = np.zeros([self.num_action_type, 3])
        self.loss = check_gpu_device(torch.tensor(0.0))
        self.policy_pies = []
        self.predict_label = []
        self.ht, self.ct = self.init_rnn_state(dim_hidden=self.config.dim_state,
                                               bidirectional=False)
        self.continue_flag = True

    @staticmethod
    def init_rnn_state(dim_hidden, bidirectional):
        dim_direction = 2 if bidirectional else 1
        ht = check_gpu_device(torch.zeros(dim_direction, 1, int(dim_hidden/dim_direction)))
        ct = check_gpu_device(torch.zeros(dim_direction, 1, int(dim_hidden/dim_direction)))
        return ht, ct

    def _compute_step_loss(self, prob, action_label, location, next_location):
        if self.score_mode == 'sigmoid':
            loss = sum([torch.log(prob[i]) if action_label[i] == 1 else torch.log(1 - prob[i])
                        for i in range(len(action_label))]) / (3 * self.config.dim_unit)

        elif self.score_mode == 'softmax':
            num_choice = 0
            loss = check_gpu_device(torch.tensor(0.0))
            prob_1 = check_gpu_device(torch.tensor(0.0))
            for i, label in enumerate(action_label):
                if label == 1:
                    prob_1 += prob[i]
                else:
                    num_choice += 1
                    loss += torch.log(1 - prob[i])
            if 1 in action_label:
                loss = torch.log(prob_1)
            else:
                loss = check_gpu_device(torch.tensor(0.0))
            loss = loss * (next_location[0] - location[0])
        else:
            raise ImportError('illegal score mode')
        return -loss

    def _compute_sl_loss(self):
        count_action = check_gpu_device(torch.tensor(np.sum(np.sum(self.action_count)))).float()
        loss = torch.sum(torch.sum(self.loss)) / count_action
        return loss

    def _compute_sum_log_pie(self):
        count_action_level = np.sum(self.action_count, axis=1)
        reward = np.sum(count_action_level * np.array([-1, 1, 1])) / np.sum(count_action_level)
        word_action_ratio = count_action_level[0] / np.sum(count_action_level)
        count_action = np.sum(count_action_level)
        policy_pies = torch.cat(self.policy_pies)
        log_sum_pie = -(torch.sum(policy_pies)) / float(count_action)
        return reward, log_sum_pie, word_action_ratio

    def _compute_next_location(self, num_word, sentence_boundaries, paragraph_boundaries, action):
        num_action_type = self.num_action_type
        """generate new location based on action selected
               pay attention for skip path"""

        def ex_st_range(sentence_span, l):
            return list(range(sentence_span[l[1]], sentence_span[l[1] + 1]))

        def ex_prg_range(paragraph_span, l):
            return list(range(paragraph_span[l[2]], paragraph_span[l[2] + 1]))

        max_wix = num_word
        max_stix = len(sentence_boundaries) - 1
        max_prgix = len(paragraph_boundaries) - 1
        new_location = self.location[:]
        if action in range(0, num_action_type):
            # print('word action')
            new_location[0] = self.location[0] + 1
            if new_location[0] not in ex_st_range(sentence_boundaries, self.location):
                new_location[1] = self.location[1] + 1

                if new_location[1] not in ex_prg_range(paragraph_boundaries, self.location):
                    new_location[2] = self.location[2] + 1

        elif action in range(num_action_type, 2 * num_action_type):
            # print('sentence action')
            new_location[1] = self.location[1] + 1
            if new_location[1] < max_stix:
                new_location[0] = sentence_boundaries[new_location[1]]
                if new_location[1] not in ex_prg_range(paragraph_boundaries, self.location):
                    new_location[2] = self.location[2] + 1
            else:
                new_location = [max_wix, max_stix - 1, max_prgix - 1]

        elif action in range(2 * num_action_type, 3 * num_action_type):
            # print('paragraph action')
            new_location[2] = self.location[2] + 1
            if new_location[2] < max_prgix:
                new_location[1] = paragraph_boundaries[new_location[2]]
                new_location[0] = sentence_boundaries[new_location[1]]
            else:
                new_location = [max_wix, max_stix - 1, max_prgix - 1]

        return new_location

    def _gen_pred_label(self, action, next_location):
        """generate identify label based on the chosen action"""
        if action % 3 == 0:
            pred_label = [-1] * (next_location[0] - self.location[0])
        elif action % 3 == 1:
            pred_label = [self.current_mark] * (next_location[0] - self.location[0])
        else:
            pred_label = [self.current_mark + 1] * (next_location[0] - self.location[0])
        return pred_label

    def _ex_fragment_label(self, label, next_location):
        return label[self.location[0]:next_location[0]]

    def _gen_action_label(self, num_word, sentence_boundaries, paragraph_boundaries, label):
        """generate action label for this time step based on the consequence after corresponding action"""
        action_labels = []
        for action in range(3 * self.config.dim_unit):
            next_location = self._compute_next_location(num_word, sentence_boundaries, paragraph_boundaries, action)
            pred_label = self._gen_pred_label(action, next_location)
            ground_label = self._ex_fragment_label(label, next_location)
            if pred_label == ground_label:
                action_labels.append(1)
            else:
                action_labels.append(0)

        return action_labels

    def _gen_skip_action(self, prob, mask):
        """mask: which actions are acceptable at current time step
           action: the action chosen to skip"""
        random_flag = self.random_sampling
        mask = check_gpu_device(torch.tensor(mask, requires_grad=False)).float()
        prob = prob * mask
        if random_flag:
            """action random selection"""
            action = torch.multinomial(prob, num_samples=1)
        else:
            """action deterministic selection"""
            _, action = torch.max(prob, dim=0)
        if torch.cuda.is_available():
            action = action.data.cpu().numpy()
        else:
            action = action.data.numpy()
        return action

    @staticmethod
    def _convert_label(label, bmeo=False):
        """convert event mark label to BMEO label"""
        convert_dict = {'O': 0, 'B': 1, 'M': 2, 'E': 3}
        new_label = label[:]
        max_mark = max(new_label)
        for mark in range(1, max_mark + 1):
            event_length = new_label.count(mark)
            count = 0
            for i, simple_label in enumerate(new_label):
                if simple_label == mark and count == 0:
                    new_label[i] = 'B'
                    count += 1
                elif simple_label == mark and count == event_length - 1:
                    new_label[i] = 'E'
                elif simple_label == mark:
                    new_label[i] = 'M'
                    count += 1
        new_label = ['O' if x == -1 or x == 0 else x for x in new_label]
        if not bmeo:
            new_label = list(map(lambda x: convert_dict[x], new_label))

        return new_label

    @staticmethod
    def _gen_pred_action(prob):
        _, action = torch.max(prob, dim=0)
        if torch.cuda.is_available():
            action = action.data.cpu().numpy()
        else:
            action = action.data.numpy()
        return action

    @staticmethod
    def _show_result(sample, label):
        print(len(sample), len(label))
        # assert len(sample) == len(label)
        text = ''
        for word, simple_label in zip(sample, label):
            if simple_label == 2:
                text += ctext(word, 'gray')
            elif simple_label == 1 or simple_label == 3 or simple_label == 4:
                text += ctext(word, 'blue')
            else:
                text += ctext(word, 'green')
        print(text)

    def forward(self, memory, location, htm1, ctm1, previous_action, sym_mark='n'):
        action_vector = check_gpu_device(torch.tensor([0] * self.dim_actions, requires_grad=False)).float()
        action_vector[previous_action] = 1

        """convert symbolic memory"""
        sym_vector = check_gpu_device(torch.tensor([0.] * 3 * self.num_action_type, requires_grad=False))
        if sym_mark != 'n':
            sym_vector[int(sym_mark)] = 1
        input_controller = torch.cat([memory[0][location[0]],
                                      memory[1][location[1]],
                                      memory[2][location[2]],
                                      action_vector
                                      ], dim=0)
        memory_next_unit = torch.cat([memory[i][location[i] + 1] if location[i] + 1 < memory[i].size(0) else
                                      check_gpu_device(torch.zeros(dim)) for i, dim in
                                      enumerate([self.dim_word, self.dim_sentence, self.dim_paragraph])], dim=0)

        in_feature = torch.cat([input_controller, memory_next_unit, sym_vector], dim=0)
        in_feature = in_feature.view(1, 1, -1)
        state_controller, (ht, ct) = self.state_updater.forward(in_feature, (htm1, ctm1))
        state_controller = state_controller.view(-1)

        prob = self.scorer_group.forward(state_controller)
        # prob = torch.clamp(prob, 1e-6, 1 - 1e-6)
        pie = torch.log(prob)
        return pie, prob, ht, ct

    def sample_forward(
            self,
            content,
            label,
            memory,
            sentence_boundaries,
            paragraph_boundaries,
            mode='train',
            show_result=False,
            show_action_level=False
    ):
        num_action = 0  # tmp
        content = ''.join(content)
        num_word = len(content)
        self._init_status()

        action_sequence = ''

        while self.continue_flag:
            # print(self.location)
            # print('sentences', memory[1].size())
            # print('paragraphs', memory[2].size())
            pie, prob, self.ht, self.ct = self.forward(memory=memory,
                                                       location=self.location,
                                                       htm1=self.ht,
                                                       ctm1=self.ct,
                                                       previous_action=self.previous_action)
            """check which actions are acceptable"""
            action_label = self._gen_action_label(num_word=num_word,
                                                  sentence_boundaries=sentence_boundaries,
                                                  paragraph_boundaries=paragraph_boundaries,
                                                  label=label)
            if mode == 'train':
                assert sum(action_label) != 0
            """skip_action: the action we use to update location [does note exist while validation and testing]"""
            if mode == 'train':
                chosen_action = self._gen_skip_action(prob, mask=action_label)
            elif mode == 'eval':
                chosen_action = self._gen_pred_action(prob)
            else:
                raise ImportError('Wrong forward mode')

            """save log(pie(action|station))"""
            self.policy_pies.append(pie[chosen_action].view(1, -1))

            """generate new location based on skip action"""
            next_location = self._compute_next_location(action=chosen_action,
                                                        num_word=num_word,
                                                        sentence_boundaries=sentence_boundaries,
                                                        paragraph_boundaries=paragraph_boundaries)
            section_text = content[self.location[0]: next_location[0]]

            """generate predicted label based on pred_action"""
            step_label = self._gen_pred_label(action=chosen_action,
                                              next_location=next_location)

            """add predicted label to previous labels"""
            self.predict_label.extend(step_label)

            """compute current loss"""
            step_loss = self._compute_step_loss(prob, action_label=action_label,
                                                location=self.location,
                                                next_location=next_location)
            self.action_count[int(chosen_action / 3), chosen_action % 3] += 1
            num_action += 1
            # self.loss[int(chosen_action/3), chosen_action % 3] += step_loss

            self.loss += step_loss

            if chosen_action in range(3):
                action_sequence += ctext(section_text, 'grey')
            elif chosen_action in range(3, 6):
                action_sequence += ctext(section_text, 'blue')
            else:
                action_sequence += ctext(section_text, 'orange')

            """update location & check whether reach the bottom"""
            self.location = next_location[:]

            if self.location[0] == num_word:
                self.continue_flag = False

            """update previous action"""
            self.previous_action = chosen_action

            """update current mark"""
            if mode == 'train':
                self.current_mark = max(max(label[:self.location[0]]), 0)
            elif mode == 'validate' or 'test':
                self.current_mark = max(max(self.predict_label), 0)
            else:
                raise ImportError('Wrong forward mode')

        # sl_loss = self._compute_sl_loss()/(self.location[0])
        sl_loss = self.loss / num_word
        reward, log_sum_pie, word_action_ratio = self._compute_sum_log_pie()
        self.predict_label = self._convert_label(self.predict_label)
        label = self._convert_label(label)

        result = list(zip(label, self.predict_label, content))
        coma_list = ['：', '，', '。', '；', '\n']
        self.pred_label = [l_p if w not in coma_list else l for l, l_p, w in result]

        if show_result:
            print(ctext('Ground Truth', 'green'))
            self._show_result(content, label=label)
            print(ctext('Predicted', 'green'))
            self._show_result(content, label=self.predict_label)
        if show_action_level:
            print(action_sequence)
        return sl_loss, reward, log_sum_pie, word_action_ratio, self.predict_label, label

    def save(self, id_):
        tag = self.config.tag
        confirm_directory('./check_points')
        folder = relative_path('./check_points/{}'.format(tag.split(' ')[0]))
        confirm_directory(folder)
        path = join(folder, 'agent_{}'.format(id_) + '.pth')
        confirm_directory(folder)
        torch.save(self.state_dict(), path)
        return 'agent_{}_{}'.format(tag, id_)

    def load(self, id_):
        tag = self.config.tag
        folder = relative_path('./check_points/{}'.format(tag.split(' ')[0]))
        path = join(folder, 'agent_{}'.format(id_) + '.pth')
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def show_configure(self):
        self.config.inform()


if __name__ == '__main__':
    set_global_logging_level()
    config = Configure()
    torch.cuda.set_device(device=config.gpu_id)
    dataset = TOFRData(data_path='./data/criminal/dev.pkl')
    zoom_encoder = ZoomEncoder(config=config)
    zoom_agent = ZoomAgent(config=config)

    for epoch in range(1):
        data = DataLoader(dataset=dataset,
                          batch_size=2,
                          collate_fn=collate)
        for batch in data:
            word_encoded, sentence_encoded, paragraph_encoded = zoom_encoder.forward(batch)
            for word, sentence, paragraph in zip(
                    word_encoded,
                    sentence_encoded,
                    paragraph_encoded):
                memory = [word, sentence, paragraph]
            break






