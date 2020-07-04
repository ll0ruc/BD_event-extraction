import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from BD_data_load import idx2trigger, argument2idx, idx2argument
from BD_consts import NONE
from BD_utils import find_triggers


class Net(nn.Module):
    def __init__(self, trigger_size=None, argument_size=None, device=torch.device("cpu")):
        super().__init__()
        self.bert = BertModel.from_pretrained('./bert-base-chinese')
        hidden_size = 768
        self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=768, hidden_size=768 // 2, batch_first=True)
        self.linear_l = nn.Linear(hidden_size, hidden_size//2)
        self.linear_r = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size),
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size, argument_size),
        )
        self.device = device

    def predict_triggers(self, tokens_x_2d, mask, head_indexes_2d, arguments_2d=None, Test=False):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        mask = torch.LongTensor(mask).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(input_ids=tokens_x_2d, attention_mask=mask)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(input_ids=tokens_x_2d, attention_mask=mask)
                enc = encoded_layers[-1]
        x = enc
        batch_size = tokens_x_2d.shape[0]
        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])

        trigger_logits = self.fc_trigger(x)
        trigger_hat_2d = trigger_logits.argmax(-1)

        x_rnn, h0, argument_candidate = [], [], []
        for i in range(batch_size):
            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor_l = self.linear_l(x[i, t_start, :])
                event_tensor_r = self.linear_r(x[i, t_end-1, :])
                event_tensor = torch.stack([event_tensor_l, event_tensor_r])
                h0.append(event_tensor)
                x_rnn.append(x[i])
                argument_candidate.append((i, t_start, t_end, t_type_str))

        argument_logits, arguments_y_1d = [0], [0]
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        if len(argument_candidate) > 0:
            h0 = torch.stack(h0, dim=1)
            c0 = torch.zeros(h0.shape[:], dtype=torch.float)
            c0 = c0.to(self.device)
            x_rnn = torch.stack(x_rnn)
            rnn_out, (hn, cn) = self.rnn(x_rnn, (h0,c0))
            argument_logits = self.fc_argument(rnn_out)
            argument_hat = argument_logits.argmax(-1)

            argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
            for i in range(len(argument_hat)):
                ba, st, ed, event_type_str = argument_candidate[i]
                if (st, ed, event_type_str) not in argument_hat_2d[ba]['events']:
                    argument_hat_2d[ba]['events'][(st, ed, event_type_str)] = []
                predicted_arguments = find_triggers([idx2argument[argument] for argument in argument_hat[i].tolist()])
                for predicted_argument in predicted_arguments:
                    e_start, e_end, e_type_str = predicted_argument
                    argument_hat_2d[ba]['events'][(st, ed, event_type_str)].append((e_start, e_end, e_type_str))

            arguments_y_1d = []
            if Test == False:
                for i, t_start, t_end, t_type_str in argument_candidate:
                    a_label = [NONE] * x.shape[1]
                    if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                        for (a_start, a_end, a_role) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                            for j in range(a_start, a_end):
                                if j == a_start:
                                    a_label[j] = 'B-{}'.format(a_role)
                                else:
                                    a_label[j] = 'I-{}'.format(a_role)
                        # for e_start, e_end, e_role in argument_hat_2d[i]['events'][(t_start, t_end, t_type_str)]:
                        #     for (a_start, a_end, a_role) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                        #         if e_start == a_start and e_end == a_end:
                        #             for j in range(e_start, e_end):
                        #                 if j == e_start:
                        #                     a_label[j] = 'B-{}'.format(a_role)
                        #                 else:
                        #                     a_label[j] = 'I-{}'.format(a_role)
                        #             break
                    a_label = [argument2idx[t] for t in a_label]
                    arguments_y_1d.append(a_label)

            arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)


        return trigger_logits, trigger_hat_2d, argument_logits, arguments_y_1d, argument_hat_2d
