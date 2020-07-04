import os
import torch
from BD_data_load import idx2trigger
from BD_utils import calc_metric, find_triggers


def eval(model, iterator, fname=None):
    model.eval()

    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, id, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, mask, words_2d, triggers_2d = batch

            trigger_logits, trigger_hat_2d, argument_logits, arguments_y_1d, argument_hat_2d = model.predict_triggers(tokens_x_2d=tokens_x_2d,
                                                                                                             mask=mask,head_indexes_2d=head_indexes_2d,
                                                                                                         arguments_2d=arguments_2d)

            words_all.extend(words_2d)
            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)
            arguments_hat_all.extend(argument_hat_2d)

    triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
    with open('temp', 'w') as fout:
        for i, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(
                zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)-2]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]
    
            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])
    
            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_role = argument
                    arguments_true.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_role))
    
            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_role = argument
                    arguments_pred.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_role))
    
        print('[trigger classification]')
        trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
        print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))
        print('[argument classification]')
        argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
        print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))
    
        print('[trigger identification]')
        triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
        triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
        trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred)
        print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))
        print('[argument identification]')
        arguments_true = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in arguments_true]
        arguments_pred = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in arguments_pred]
        argument_p_, argument_r_, argument_f1_ = calc_metric(arguments_true, arguments_pred)
        print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p_, argument_r_, argument_f1_))
        
        if fname:
            metric = '[trigger classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p, trigger_r, trigger_f1)
            metric += '[argument classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p, argument_r, argument_f1)
            metric += '[trigger identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p_, trigger_r_, trigger_f1_)
            metric += '[argument identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p_, argument_r_, argument_f1_)

            with open(fname, 'w') as fout:
                result = open("temp", "r").read()
                fout.write("{}\n".format(result))
                fout.write(metric)
            os.remove("temp")

    return trigger_f1, argument_f1
