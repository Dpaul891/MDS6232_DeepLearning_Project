import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

plt.style.use('ggplot')

# For the only image -- EfficientNet-B3 model training
image_only_train_f1 = np.load('./results/image_only_train_f1.npy')
image_only_val_f1 = np.load('./results/image_only_val_f1.npy')
plt.figure(1)
plt.plot(image_only_train_f1, label='training')
plt.plot(image_only_val_f1, label='validation')
plt.title('Training EfficientNet-B3')
plt.xlabel('epoch')
plt.ylabel('F1 score')
plt.legend(edgecolor='black', facecolor='white')
plt.savefig('./figure/image_only_f1.pdf')

image_only_train_loss = np.load('./results/image_only_train_loss.npy')
plt.figure(2)
plt.plot(image_only_train_loss)
plt.title('Training EfficientNet-B3')
plt.xlabel('epoch')
plt.ylabel('Training loss')
plt.savefig('./figure/image_only_train_loss.pdf')

# For the BERT model training
bert_train_loss = np.load('./results/bert_train_loss_list.npy')
bert_val_loss = np.load('./results/bert_val_loss_list.npy')
plt.figure(3)
plt.plot(bert_train_loss, label='training')
plt.plot(bert_val_loss, label='validation')
plt.title('Training BERT')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(edgecolor='black', facecolor='white')
plt.savefig('./figure/BERT_loss.pdf')


bert_train_acc = np.load('./results/bert_train_acc_list.npy')
bert_val_acc = np.load('./results/bert_val_acc_list.npy')
plt.figure(4)
plt.plot(bert_train_acc, label='training')
plt.plot(bert_val_acc, label='validation')
plt.title('Training BERT')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(edgecolor='black', facecolor='white')
plt.savefig('./figure/BERT_acc.pdf')

#For the fusion training

fusion_train_loss = np.load('./results/fusion_train_loss.npy')
plt.figure(5)
plt.plot(fusion_train_loss)
plt.title('Training EfficientNet-B3+BERT')
plt.xlabel('epoch')
plt.ylabel('Training loss')
plt.savefig('./figure/fusion_train_loss.pdf')

fusion_train_f1 = np.load('./results/fusion_train_f1.npy')
fusion_val_f1 = np.load('./results/fusion_val_f1.npy')
plt.figure(6)
plt.plot(fusion_train_f1, label='training')
plt.plot(fusion_val_f1, label='validation')
plt.title('Training EfficientNet-B3+BERT')
plt.xlabel('epoch')
plt.ylabel('F1 score')
plt.legend(edgecolor='black', facecolor='white')
plt.savefig('./figure/fusion_f1.pdf')






