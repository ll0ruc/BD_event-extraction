import os
import torch
from BD_data_load import idx2trigger

def test(model, iterator, fname):
    model.eval()

    words_all, triggers_hat_all, arguments_hat_all = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, id, seqlens_1d, head_indexes_2d, mask, words_2d = batch

            trigger_logits, trigger_hat_2d, argument_logits, arguments_y_1d, argument_hat_2d = model.predict_triggers(
                tokens_x_2d=tokens_x_2d, mask=mask,
                head_indexes_2d=head_indexes_2d,Test=True)

            words_all.extend(words_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_hat_all.extend(argument_hat_2d)

    with open('temp', 'w') as fout:

        for i, (words, triggers_hat, arguments_hat) in enumerate(zip(words_all, triggers_hat_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)-2]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]
            
            for w, t_h in zip(words[1:-1], triggers_hat):
                fout.write('{}\t{}\n'.format(w, t_h))
            fout.write('#arguments#{}\n'.format(arguments_hat['events']))
            fout.write("\n")

    with open(fname, 'w') as fout:
        result = open("temp", "r").read()
        fout.write("{}\n".format(result))
    os.remove("temp")
