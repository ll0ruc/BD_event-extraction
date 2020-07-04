import numpy as np
from torch.utils import data
import json
from BD_consts import NONE, PAD, CLS, SEP, UNK, TRIGGERS, ARGUMENTS
from BD_utils import build_vocab
from pytorch_pretrained_bert import BertTokenizer

# init vocab
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS)
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese-vocab.txt', do_lower_case=False, never_split=(PAD, CLS, SEP, UNK))


class TrainDataset(data.Dataset):
    def __init__(self, fpath):
        self.sent_li, self.id_li, self.triggers_li, self.arguments_li = [], [], [], []

        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                id = data['id']
                sentence = data['text'].replace(' ', '-')
                sentence = sentence.replace('\n', ',')
                sentence = sentence.replace('\u3000', '-')
                sentence = sentence.replace('\xa0', ',')
                sentence = sentence.replace('\ue627', ',')
                words = [word for word in sentence]
                if len(words) > 500:
                    continue
                triggers = [NONE] * len(words)
                arguments = {
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }

                for event_mention in data['event_list']:
                    id_s = event_mention['trigger_start_index']
                    trigger_text = event_mention['trigger']
                    id_e = id_s + len(trigger_text)
                    trigger_type = event_mention['event_type'].split('-')[-1]
                    for i in range(id_s, id_e):
                        if i == id_s:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)

                    event_key = (id_s, id_e, trigger_type)
                    arguments['events'][event_key] = []
                    for argument in event_mention['arguments']:
                        role = argument['role']
                        a_id_s = argument['argument_start_index']
                        argument_text = argument['argument']
                        a_id_e = a_id_s + len(argument_text)
                        arguments['events'][event_key].append((a_id_s, a_id_e, role))

                self.sent_li.append([CLS] + words + [SEP])
                self.id_li.append(id)
                self.triggers_li.append(triggers)
                self.arguments_li.append(arguments)

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, id, triggers, arguments = self.sent_li[idx], self.id_li[idx], self.triggers_li[idx], self.arguments_li[idx]

        tokens_x, is_heads = [], []
        for w in words:
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            if len(tokens) != 1:
                print(words)
                print("This is not a single chinese tokens!")
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)
            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            tokens_x.extend(tokens_xx), is_heads.extend(is_head)

        triggers_y = [trigger2idx[t] for t in triggers]
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)
        seqlen = len(tokens_x)
        mask = [1] * seqlen

        return tokens_x, id, triggers_y, arguments, seqlen, head_indexes, mask, words, triggers

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


def Trainpad(batch):
    tokens_x_2d, id, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, mask, words_2d, triggers_2d = list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        mask[i] = mask[i] + [0] * (maxlen - len(mask[i]))

    return tokens_x_2d, id, \
           triggers_y_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, mask, \
           words_2d, triggers_2d



class TestDataset(data.Dataset):
    def __init__(self, fpath):
        self.sent_li, self.id_li = [], []

        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                id = data['id']
                sentence = data['text'].replace(' ', '-')
                sentence = sentence.replace('\n', ',')
                sentence = sentence.replace('\u3000', '-')
                sentence = sentence.replace('\xa0', ',')
                sentence = sentence.replace('\ue627', ',')
                words = [word for word in sentence]
                if len(words) > 500:
                    continue

                self.sent_li.append([CLS] + words + [SEP])
                self.id_li.append(id)

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, id = self.sent_li[idx], self.id_li[idx]

        tokens_x, is_heads = [], []
        for w in words:
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)
            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            tokens_x.extend(tokens_xx), is_heads.extend(is_head)

        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)
        seqlen = len(tokens_x)
        mask = [1] * seqlen

        return tokens_x, id, seqlen, head_indexes, mask, words

def Testpad(batch):
    tokens_x_2d, id, seqlens_1d, head_indexes_2d, mask, words_2d = list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        mask[i] = mask[i] + [0] * (maxlen - len(mask[i]))

    return tokens_x_2d, id, \
           seqlens_1d, head_indexes_2d, mask, \
           words_2d