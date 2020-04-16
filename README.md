# hlt_2019
University of Pisa Human Language Technologies class 

Human Language Technologies final project
Toxic comment classification with Bert transformer

Run it with _-h_ or _--help_ to show this help.

```
Usage:main.py [options]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           Either train, evaluate or predict
  --dataset DATASET     dataset to train or evaluate model
  --lr LR               training learning rate
  --nepochs NEPOCHS     Number of epochs
  --set_evaluate SET_EVALUATE
                        whether to evaluate model performance or not in training phase
  --fine_tune FINE_TUNE
                        freeze all the layer but output classifier and train it
  --from_checkpoint FROM_CHECKPOINT
                        load model from checkpoint and resume training
```
