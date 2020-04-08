import os
# import pickle
import numpy as np
import torch
from pytorch_transformers import BertForSequenceClassification, BertTokenizer, BertModel
from pytorch_transformers.optimization import AdamW, WarmupCosineSchedule
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm, trange
# import matplotlib.pyplot as plt
# from datetime import datetime
# from sklearn.metrics import confusion_matrix
# import copy
import argparse
from utils import *
# from datetime import datetime


def get_model(w=None):
    if w == 'original':
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    else:
        model = BertForHLT.from_pretrained('bert-base-uncased', num_out_labels=6)
    return model


class BertForHLT(BertModel):
    def __init__(self, config, num_out_labels):
        super(BertForHLT, self).__init__(config)
        self.num_labels = num_out_labels

        # pretrained BERT
        self.bert = BertModel(config)
        # setting last layer as linear FC classifier
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(config.hidden_size, num_out_labels)   
        self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None,
                head_mask=None, labels=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            # @ https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
            loss_fn = BCEWithLogitsLoss()   
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss, logits
        else:
            return logits
        
        
    def set_fine_tuning(self, fine_tuning=True):
        for name, child in self.named_children():
            if name == 'classifier':               
                print(f'{name} is unfrozen')
                for param in child.parameters():
                    param.requires_grad = True
            else:
                print(f'{name} is '+('frozen' if fine_tuning else 'unfrozen'))
                for param in child.parameters():
                    param.requires_grad = not fine_tuning


def train_model(model, optimizer, train_dataloader, val_dataloader, epochs, finetune, evaluate, filename, history_loss=([],[]), save_path=None, save_best=False):
    TIMESTAMP = datetime.now().strftime('%d%m_%H:%M')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    print('\n# Model Parameters Status:')
    model.set_fine_tuning(finetune)
    if not finetune:
        print('All parameters unfrozen')
    
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=3, t_total=7, cycles=.5, last_epoch=-1)

    num_epoch = epochs
    start_epoch = 0
    best_acc = 1000
    best_model_wts = None

    train_loss, val_loss = history_loss
    train_pred = []
    train_label = []
    val_pred = []
    val_label = []

    print('\n# Start training BERT')
    
    for _i in range(start_epoch, num_epoch):
        for phase in ['Training','Validation']:
            print(phase)
            dataloader = train_dataloader if phase is 'train' else val_dataloader
            epoch_loss = 0
            epoch_predictions = None
            epoch_labels = None

            for step, batch in enumerate(tqdm(dataloader, desc=f'Epoch {_i+1}/{opt.nepochs} - Batches')):
                
                if phase is 'Training':
                    model.train()
                    
                else:
                    model.eval()

                input_x, mask_x, target_y = batch
                input_x = input_x.to(device)
                mask_x = mask_x.to(device)
                target_y = target_y.float().to(device)
                output, pred = model(input_x, token_type_ids=None, attention_mask=mask_x, labels=target_y)

                if phase is 'Training':
                    output.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                 

                pred = np.array(pred.cpu().detach())
                target_y = np.array(target_y.cpu().detach())        

                epoch_predictions = pred if epoch_predictions is None else np.concatenate((epoch_predictions, pred), axis=0)
                epoch_labels = target_y if epoch_labels is None else np.concatenate((epoch_labels, target_y), axis=0)

                loss = output.cpu().detach().item()
                epoch_loss += loss
                    
            if phase is 'Training':
                train_loss.append(epoch_loss/step)
                train_pred.append(epoch_predictions)
                train_label.append(epoch_labels)
            else:
                val_loss.append(epoch_loss/step)
                val_pred.append(epoch_predictions)
                val_label.append(epoch_labels)
      
        print(f'\nEpoch {_i+1}:\nTraining loss: {train_loss[-1]}\nValidation loss: {val_loss[-1]}\n')
        _,_ = get_aucroc(val_pred[-1].T, val_label[-1].T)
        print('-'*50)
        
        save_checkpoint({
            'epoch': _i,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'history_loss': (train_loss, val_loss),
        }, model, save_path=save_path)

        # deep copy the best model
        if save_best and val_loss[-1] <= best_acc:
            best_acc = val_loss[-1]
            best_model_wts = copy.deepcopy(model.state_dict())                
                        
    plot_loss_sketch('History Loss', [train_loss, val_loss], save_path)
    
#    if save_best: 
#        model.load_state_dict( best_model_wts )
#    if save_path:
#        save_checkpoint({
#            'epoch': epochs,
#            'state_dict': model.state_dict(),
#            'optimizer' : optimizer.state_dict(),
#            'history_loss': (train_loss, val_loss),
#        }, model, save_path=save_path)

    return train_pred, train_label, val_pred, val_label


def eval_model(model, test_dataloader):
    predictions = None
    labels = None
    
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            input_x, mask_x, target_y = batch
            input_x = input_x.to(device)
            mask_x = mask_x.to(device)
            pred_batch = model(input_x, token_type_ids=None, attention_mask=mask_x)
            pred_batch = np.array(pred_batch.cpu().detach())
            target_y = np.array(target_y.cpu().detach())

            predictions = pred_batch if predictions is None else np.concatenate((predictions, pred_batch), axis=0)
            labels = target_y if labels is None else np.concatenate((labels, target_y), axis=0)  
            
    return predictions, labels


def main():
    TIMESTAMP = datetime.now().strftime('%d%m_%H:%M')
    filename = opt.dataset
    train_dataloader, val_dataloader = load_pickle_dump('dataset/', filename)

    # setting hyperparameters
    lr = opt.lr
    epochs = opt.nepochs
    evaluate = opt.set_evaluate
    finetune_last_layer = opt.fine_tune
    history_loss = ([],[])
    save_path = 'models/Bert_' + TIMESTAMP
    if opt.from_checkpoint is not None:
        load_path = 'models/Bert_' + opt.from_checkpoint

    model = get_model().to(device)

    # Linear decay on AdamW
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # Loading model from checkpoint
    if opt.from_checkpoint is not None:
        print("Starting from  checkpoint: {}".format(load_path))
        checkpoint = load_checkpoint(load_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'history_loss' in checkpoint:  
            history_loss = checkpoint["history_loss"]

    train_pred, train_label, val_pred, val_label = train_model(model, optimizer, train_dataloader, val_dataloader,
                                                               epochs, finetune_last_layer, evaluate, filename,
                                                               history_loss, save_path, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HLT Project 2019')

    parser.add_argument('--mode', type=str, default='predict', help='Either train, evaluate or predict')

    parser.add_argument('--dataset', type=str, default='training_local_128_8.pickle', help=f'dataset to train or evaluate model')
    parser.add_argument('--lr', type=int, default=1e-5, help='training learning rate')
    parser.add_argument('--nepochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--set_evaluate', type=bool, default=True, help='whether to evaluate model performance or not in training phase')
    parser.add_argument('--fine_tune', type=bool, default=False, help='freeze all the layer but output classifier and train it')
    parser.add_argument('--from_checkpoint', type=str, default=None, help='load model from checkpoint and resume training')

    opt = parser.parse_args()

    print(f'Torch Version: {torch.__version__}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print('Runnig accellerator: ', device)

    main()

