import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from BD_model import Net
from BD_data_load import TrainDataset, Trainpad, TestDataset, Testpad, all_triggers, all_arguments
from BD_eval import eval
from BD_test import test


def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        tokens_x_2d, id, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, mask, words_2d, triggers_2d = batch
        optimizer.zero_grad()
        trigger_logits, trigger_hat_2d, argument_logits, arguments_y_1d, argument_hat_2d = model.predict_triggers(tokens_x_2d=tokens_x_2d,
                                                                                    mask=mask,head_indexes_2d=head_indexes_2d,
                                                                                            arguments_2d=arguments_2d)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(model.device)
        triggers_y_2d = triggers_y_2d.view(-1)
        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = criterion(trigger_logits, triggers_y_2d)

        if len(argument_logits) != 1:
            argument_logits = argument_logits.view(-1, argument_logits.shape[-1])
            argument_loss = criterion(argument_logits, arguments_y_1d.view(-1))
            loss = trigger_loss + 2 * argument_loss
        else:
            loss = trigger_loss

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        if i % 40 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="output")
    parser.add_argument("--trainset", type=str, default="./data/train.json")
    parser.add_argument("--devset", type=str, default="./data/dev.json")
    parser.add_argument("--testset", type=str, default="./data/test1.json")

    hp = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(
        device=device,
        trigger_size=len(all_triggers),
        argument_size=len(all_arguments)
    )
    if device == 'cuda':
        model = model.cuda()

    train_dataset = TrainDataset(hp.trainset)
    dev_dataset = TrainDataset(hp.devset)
    test_dataset = TestDataset(hp.testset)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=Trainpad)
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=Trainpad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=Testpad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    early_stop = 15
    stop = 0
    best_scores = 0.0
    for epoch in range(1, hp.n_epochs + 1):
        stop += 1
        print("=========train at epoch={}=========".format(epoch))
        train(model, train_iter, optimizer, criterion)

        fname = os.path.join(hp.logdir, str(epoch))
        print("=========dev at epoch={}=========".format(epoch))
        trigger_f1, argument_f1 = eval(model, dev_iter, fname + '_dev')

        print("=========test at epoch={}=========".format(epoch))
        test(model, test_iter, fname + '_test')
        if stop >= early_stop:
            print("The best result in epoch={}".format(epoch-early_stop-1))
            break
        if trigger_f1 + argument_f1 > best_scores:
            best_scores = trigger_f1 + argument_f1
            stop = 0
            print("The new best in epoch={}".format(epoch))
            # torch.save(model, "best_model.pt")

