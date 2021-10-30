This folder contains various experiments to have a DNN model for MNIST dataset with the following constraints
Contsraints: < 10K params, 99.4% accuracy within 15 Epochs

Steps:
1) Started with a base model with ~99.07% accuracy
2) Reduced parameters by 1/4th with similar accuracy
3) Reduced parameters furhter with better accuracy than first two models
4) Added batch normalization to improve efficiency (Train Accuracy: 99.75%, Test Accurcy: 99.15%)
5) Tried dropout, GAP at the end, however couldnt reach 99.4% on test accuracy
