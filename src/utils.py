import pickle
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import os
import numpy as np
from sklearn.metrics import roc_auc_score


def load_pickle_dump(path, filename):
    with open(path + filename, 'rb') as infile:
        indata = pickle.load(infile)
        train_dataloader = indata['train_set']
        val_dataloader = indata['val_set']

        print('# Successfully loaded dump file')
    return train_dataloader, val_dataloader


def plot_loss_sketch(title, losses, path):
    plt.figure(figsize=(15, 10))
    _label = ['training', 'validation']
    for i, loss in enumerate(losses):
        _x = [i + 1 for i in range(len(loss))]
        plt.plot(_x, loss, label=i)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend(_label)
    # plt.show()
    plt.savefig(path + '/' + title + '.png')


def save_checkpoint(state, model, save_path):
    TIMESTAMP = datetime.now().strftime('%d%m_%H:%M')
    # model.save_pretrained(save_path)
    torch.save(state, os.path.join(save_path, 'model_checkpoint_{}.pth.tar'.format(TIMESTAMP)))


def load_checkpoint(load_path, pretrained_path="None"):
    checkpoint = torch.load(load_path)
    return checkpoint


# AUC-ROC, Confusion Matrix. Implementation of metrics functions and their visualizations
def plot_aucroc(pred, tar):
    from sklearn.metrics import roc_curve
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        fpr_rt, tpr_rt, _ = roc_curve(tar[i], pred[i])
        plt.plot(fpr_rt, tpr_rt, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


# AUC-ROC
def get_aucroc(pred, tar):
    label = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    avg_micro = roc_auc_score(tar, pred, average="micro")
    # avg_macro = roc_auc_score(tar, pred, average="macro")
    per_label_rocauc = []
    for i in range(pred.shape[0]):
        per_label_rocauc.append(roc_auc_score(tar[i], pred[i], average=None))
    for i, elem in enumerate(per_label_rocauc):
        print(f'Label {label[i]}: {per_label_rocauc[i]}')

    print(f'\nMicro AUC-ROC: {avg_micro}')
    print(f'Macro AUC-ROC: {sum(per_label_rocauc)/len(per_label_rocauc)}')

    # print(f'\nMacro AUC-ROC: {avg_macrp}')
    return per_label_rocauc, avg_micro


# CONFUSION MATRIX PLOT (takes in input ONE target label class eg for toxic label: plot_cm(pred_toxic, tar_toxic))
# predictions in input should be set to 0 or 1 given a chosen threshold
def plot_cm(pred, tar):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(tar, pred)
    # plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap='RdBu')
    classNames = ['Non Toxic', 'Toxic']
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]),
                     horizontalalignment='center', color='White')

    plt.show()
