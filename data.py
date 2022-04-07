import _pickle as pkl
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, ElectraTokenizer, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch

from utils import relative_path, ctext
from random import shuffle
from math import ceil
from pprint import pprint


class TOFRData(Dataset):

    def __init__(self, data_path, max_length=-1):
        self.data_path = data_path
        self.tokenizer1 = BertTokenizer.from_pretrained(relative_path('./bert-base-chinese'))
        self.tokenizer2 = XLNetTokenizer.from_pretrained(relative_path('./chinese-xlnet-base'))
        self.tokenizer3 = AutoTokenizer.from_pretrained(relative_path('./chinese-roberta-wwm-ext'))
        self.tokenizer4 = ElectraTokenizer.from_pretrained(relative_path('./chinese-electra-base-discriminator'))
        self.max_length = max_length
        data = self.load_data()
        if max_length != -1:
            self.data = list(filter(lambda x: len(x['content']) < max_length, data))
        else:
            self.data = data

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            data = pkl.load(f)
        word2idx, _ = self.load_dict()
        samples = []
        for sample_id, sample in enumerate(data):
            samples.append(
                {
                    "id": sample_id,
                    "content": sample["content"],
                    "bio": sample["bio"],
                    "bmes": sample["bmes"],
                    "zoom": sample["zoom"],
                    "bert_idx": torch.tensor(self.tokenizer1.convert_tokens_to_ids(sample["content"])),
                    "xlnet_idx": torch.tensor(self.tokenizer2.convert_tokens_to_ids(sample["content"])),
                    "roberta_idx": torch.tensor(self.tokenizer3.convert_tokens_to_ids(sample["content"])),
                    "electra_idx": torch.tensor(self.tokenizer4.convert_tokens_to_ids(sample["content"])),
                    "simple_idx": torch.tensor(self.convert_token2idx(word2idx=word2idx,
                                                                      content=sample["content"])),
                    "sentence_boundaries": self.get_boundary_info_zh(content=sample["content"])[0],
                    "paragraph_boundaries": self.get_boundary_info_zh(content=sample["content"])[1]
                }
            )

        if self.max_length != -1:
            samples = list(filter(lambda x: len(x['content']) < self.max_length, samples))

        samples = [self.get_sentences(sample) for sample in samples]
        return samples

    @staticmethod
    def load_dict(dict_path=relative_path('./dict/dict.pkl')):
        with open(dict_path, 'rb') as f:
            word2idx, idx2word = pkl.load(f)
        return word2idx, idx2word

    @staticmethod
    def convert_token2idx(word2idx, content):
        token_list = list(content)
        idx_list = [word2idx[token] if token in word2idx.keys() else word2idx['unknown']
                    for token in token_list]
        return idx_list

    @staticmethod
    def get_boundary_info_zh(content):
        content = list(content)
        sentence_boundary_idx = [0]
        paragraph_boundary_idx = [0]
        sentence_boundary_token = ['，', '；', '。', '？', '：', '\n']
        paragraph_boundary_token = ['\n']
        for idx, token in enumerate(content):
            if token in sentence_boundary_token and idx != 0:
                sentence_boundary_idx.append(idx + 1)
            if token in paragraph_boundary_token and idx != 0:
                paragraph_boundary_idx.append(len(sentence_boundary_idx) - 1)
        last_idx = len(content)
        # print(sentence_boundary_idx[-1])
        # print(last_idx)
        if last_idx not in sentence_boundary_idx:
            print('no end')
            sentence_boundary_idx.append(last_idx)
            paragraph_boundary_idx.append(len(sentence_boundary_idx) - 1)  # add -1
        return sentence_boundary_idx, paragraph_boundary_idx

    @staticmethod
    def get_sentences(sample):
        sentence_boundary = sample['sentence_boundaries']
        idx = sample['simple_idx']
        num_sentence = len(sentence_boundary) - 1
        sentences = [idx[sentence_boundary[i]: sentence_boundary[i + 1]] for i in range(num_sentence)]
        sentences_len = [sentence_boundary[i + 1] - sentence_boundary[i] for i in range(num_sentence)]
        sample['sentences'] = sentences
        sample['sentences_len'] = sentences_len
        return sample

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collate(batch):
    data = dict()
    non_padding_keys = ['id', 'content', 'bio', 'bmes', 'zoom', 'sentence_boundaries', 'paragraph_boundaries']
    padding_keys = ['simple_idx', 'bert_idx', 'xlnet_idx', 'roberta_idx', 'electra_idx']
    for key in non_padding_keys:
        data[key] = [sample[key] for sample in batch]
    for key in padding_keys:
        data[key] = pad_sequence(
            sequences=[sample[key] for sample in batch],
            batch_first=True,
            padding_value=0.
        )
    data['mask'] = pad_sequence(
        sequences=[torch.ones_like(sample['bert_idx']) for sample in batch],
        batch_first=True,
        padding_value=0
    )
    return data


def show(content, ibo_label):
    content = list(content)
    text = ''
    for t, l in zip(content, ibo_label):
        if l == 0:
            text += ctext(t, 'grey')
        elif l == 1:
            text += ctext(t, 'blue')
        else:
            text += ctext(t)
    print(text)


if __name__ == '__main__':
    # event_all = DetectionDataset(data_path='./data/pkls/event/train.pkl',
    #                              max_length=3000)
    # batches = form_batch_generator(event_all, batch_size=1)
    # samples = next(batches)
    # print(samples['IBO_label_padded'])
    # event_all = list(event_all)
    # del_list = []
    # for idx, sample in enumerate(event_all):
    #     show(sample['content'], sample['IBO_label'])
    #     flag = input('del?')
    #     if flag == '1':
    #         del_list.append(idx)
    # print(del_list)
    # batches = form_batch_generator(iirdataset, 2)
    # batch = next(batches)
    # print(batch['idx'])
    # print(batch['att_mask'])

    # del_list = [2, 16, 48, 58, 76, 84, 133, 159, 170, 177, 190, 218, 236, 239, 349,
    #             388, 552, 586, 653, 710, 711, 751, 756, 760, 762, 789, 820, 828, 833,
    #             841, 853, 871, 900, 944, 1039, 1045, 1061, 1076, 1091, 1093, 1107, 1139,
    #             1142, 1146, 1158, 1160, 1174, 1186, 1194, 1231, 1235, 1279, 1299, 1309,
    #             1318, 1404, 1467, 1495, 1543, 1546, 1565, 1608, 1609, 1616, 1634, 1652,
    #             1735, 1748, 1798, 1803, 1831, 1894, 1895, 1909, 1940, 1959, 1996, 2068,
    #             2167, 2182, 2249, 2280, 2294, 2325, 2461, 2463, 2479, 2611, 2612, 2656,
    #             2724, 2761, 2825, 2918, 2921, 3007, 3038, 3095, 3174, 3262, 3323, 3333,
    #             3354, 3398, 3423, 3428, 3473, 3518, 3521, 3522, 3574, 3588]
    # event_3000_train = DetectionDataset(data_path='./data/bmes/event/train.bmes')
    # with open('./data/processed_pkls/event/train.pkl', 'wb') as f:
    #     pkl.dump(event_3000_train, f)
    #

    data_set = TOFRData(data_path='./data/criminal/train.pkl')
    # for sample in tqdm(data_set):
    #     content = ''.join(sample['content'])
    #     sentence_boundaries = sample['sentence_boundaries']
    #     paragraph_boundaries = sample['paragraph_boundaries']
    #     sentence_spans = [[init_idx, end_idx] for init_idx, end_idx in zip(sentence_boundaries[:-1],
    #                                                                        sentence_boundaries[1:])]
    #
    #     paragraph_spans = [[sentence_boundaries[init_idx], sentence_boundaries[end_idx]]
    #                        for init_idx, end_idx in zip(paragraph_boundaries[:-1],
    #                                                     paragraph_boundaries[1:])]


