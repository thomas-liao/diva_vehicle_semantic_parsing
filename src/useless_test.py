# #!/usr/bin/env python
# # coding: utf-8
#
# # In[1]:
#
#
# # Install pytorch and tqdm (if necessary)
#
#
# # In[2]:
#
#
# # Mount your google drive as the data drive
# # This will require google authorization
# # from google.colab import drive  ## doing it on local machine.. google GPU sucks
# # drive.mount('/content/drive')
#
#
# # In[3]:
#
#
# # Handle imports
#
# import math
# import os
# import datetime
# import csv
#
# import matplotlib.pyplot as plt
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# from torchvision import datasets
# from torchvision import transforms
# from torch.autograd import Variable
# import numpy as np
# import tqdm
#
# from IPython import display
#
#
# # In[4]:
#
#
# # The Args object will contain all of our parameters
# # If you want to run with different arguments, create another Args object
#
# class Args(object):
#     def __init__(self, name='mnist', batch_size=64, test_batch_size=1000,
#                  epochs=10, lr=0.01, optimizer='sgd', momentum=0.5,
#                  seed=1, log_interval=100, dataset='mnist',
#                  #             data_dir='/content/drive/My Drive/cs482/data', model='default',
#                  data_dir='~//home/tliao4/Desktop/DeepLearningHw/data', model='default',
#                  cuda=True):
#         self.name = name  # name for this training run. Don't use spaces.
#         self.batch_size = batch_size
#         self.test_batch_size = test_batch_size  # Input batch size for testing
#         self.epochs = epochs  # Number of epochs to train
#         self.lr = lr  # Learning rate
#         self.optimizer = optimizer  # sgd/p1sgd/adam/rms_prop
#         self.momentum = momentum  # SGD Momentum
#         self.seed = seed  # Random seed
#         self.log_interval = log_interval  # Batches to wait before logging
#         # detailed status. 0 = never
#         self.dataset = dataset  # mnist/fashion_mnist
#         self.data_dir = data_dir
#         self.model = model  # default/P2Q7DoubleChannelsNet/P2Q7HalfChannelsNet/
#         # P2Q8BatchNormNet/P2Q9DropoutNet/P2Q10DropoutBatchnormNet/
#         # P2Q11ExtraConvNet/P2Q12RemoveLayerNet/P2Q13UltimateNet
#         self.cuda = cuda and torch.cuda.is_available()
#
#
# print(torch.cuda.is_available())
#
#
# # In[5]:
#
#
# # Define the neural network classes
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         # F is just a functional wrapper for modules from the nn package
#         # see http://pytorch.org/docs/_modules/torch/nn/functional.html
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
#
# class P2Q7HalfChannelsNet(nn.Module):
#     def __init__(self):
#         super(P2Q7HalfChannelsNet, self).__init__()
#         # TODO Implement me
#         self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
#         self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
#         self.fc1 = nn.Linear()
#
#     def forward(self, x):
#         # TODO Implement me
#         raise NotImplementedError
#
#
# class P2Q7DoubleChannelsNet(nn.Module):
#     def __init__(self):
#         super(P2Q7DoubleChannelsNet, self).__init__()
#         # TODO Implement me
#         raise NotImplementedError
#
#     def forward(self, x):
#         # TODO Implement me
#         raise NotImplementedError
#
#
# class P2Q8BatchNormNet(nn.Module):
#     def __init__(self):
#         super(P2Q8BatchNormNet, self).__init__()
#         # TODO Implement me
#         raise NotImplementedError
#
#     def forward(self, x):
#         # TODO Implement me
#         raise NotImplementedError
#
#
# class P2Q9DropoutNet(nn.Module):
#     def __init__(self):
#         super(P2Q9DropoutNet, self).__init__()
#         # TODO Implement me
#         raise NotImplementedError
#
#     def forward(self, x):
#         # TODO Implement me
#         raise NotImplementedError
#
#
# class P2Q10DropoutBatchnormNet(nn.Module):
#     def __init__(self):
#         super(P2Q10DropoutBatchnormNet, self).__init__()
#         # TODO Implement me
#         raise NotImplementedError
#
#     def forward(self, x):
#         # TODO Implement me
#         raise NotImplementedError
#
#
# class P2Q11ExtraConvNet(nn.Module):
#     def __init__(self):
#         super(P2Q11ExtraConvNet, self).__init__()
#         # TODO Implement me
#         raise NotImplementedError
#
#     def forward(self, x):
#         # TODO Implement me
#         raise NotImplementedError
#
#
# class P2Q12RemoveLayerNet(nn.Module):
#     def __init__(self):
#         super(P2Q12RemoveLayerNet, self).__init__()
#         # TODO Implement me
#         raise NotImplementedError
#
#     def forward(self, x):
#         # TODO Implement me
#         raise NotImplementedError
#
#
# class P2Q13UltimateNet(nn.Module):
#     def __init__(self):
#         super(P2Q13UltimateNet, self).__init__()
#         # TODO Implement me
#         raise NotImplementedError
#
#     def forward(self, x):
#         # TODO Implement me
#         raise NotImplementedError
#
#
# # In[6]:
#
#
# def prepare_dataset(args):
#     # choose the dataset
#     if args.dataset == 'mnist':
#         DatasetClass = datasets.MNIST
#     elif args.dataset == 'fashion_mnist':
#         DatasetClass = datasets.FashionMNIST
#     else:
#         raise ValueError('unknown dataset: ' + args.dataset +
#                          ' try mnist or fashion_mnist')
#
#     def time_stamp(fname, fmt='%m-%d-%H-%M_{fname}'):
#         return datetime.datetime.now().strftime(fmt).format(fname=fname)
#
#     training_run_name = time_stamp(args.dataset + '_' + args.name)
#
#     kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#
#     # Create the dataset: mnist or fasion_mnist
#     dataset_dir = os.path.join(args.data_dir, args.dataset)
#     training_run_dir = os.path.join(args.data_dir, training_run_name)
#     train_dataset = DatasetClass(
#         dataset_dir, train=True, download=True,
#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ]))
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
#     test_dataset = DatasetClass(
#         dataset_dir, train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ]))
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
#
#     if not os.path.exists(training_run_dir):
#         os.makedirs(training_run_dir)
#
#     return train_loader, test_loader, train_dataset, test_dataset, training_run_dir
#
#
# # In[7]:
#
#
# # visualize some images
#
# args = Args()
# _, _, _, test_dataset, _ = prepare_dataset(args)
# images = test_dataset.test_data[:6]
# labels = test_dataset.test_labels[:6]
# fig, axes = plt.subplots(1, 6)
# for axis, img, lbl in zip(axes, images, labels):
#     axis.imshow(img)
#     axis.set_title(lbl.data.numpy())
#     axis.set_yticklabels([])
#     axis.set_xticklabels([])
# plt.show()
#
#
# # In[8]:
#
#
# def train(model, optimizer, train_loader, epoch, total_minibatch_count,
#           train_losses, train_accs):
#     # Training for a full epoch
#
#     model.train()
#     correct_count, total_loss, total_acc = 0., 0., 0.
#     progress_bar = tqdm.tqdm(train_loader, desc='Training')
#
#     for batch_idx, (data, target) in enumerate(progress_bar):
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#
#         optimizer.zero_grad()
#
#         # Forward prediction step
#         output = model(data)
#         loss = F.nll_loss(output, target)
#
#         # Backpropagation step
#         loss.backward()
#         optimizer.step()
#
#         # The batch has ended, determine the accuracy of the predicted outputs
#         pred = output.data.max(1)[1]
#
#         # target labels and predictions are categorical values from 0 to 9.
#         matches = target == pred
#         accuracy = matches.float().mean()
#         correct_count += matches.sum()
#
#         if args.log_interval != 0 and total_minibatch_count % args.log_interval == 0:
#             train_losses.append(loss.data[0])
#             train_accs.append(accuracy.data[0])
#
#         total_loss += loss.data
#         total_acc += accuracy.data
#
#         progress_bar.set_description(
#             'Epoch: {} loss: {:.4f}, acc: {:.2f}'.format(
#                 epoch, total_loss / (batch_idx + 1), total_acc / (batch_idx + 1)))
#         # progress_bar.refresh()
#
#         total_minibatch_count += 1
#
#     return total_minibatch_count
#
#
# # In[9]:
#
#
# def test(model, test_loader, epoch, total_minibatch_count,
#          val_losses, val_accs):
#     # Validation Testing
#     model.eval()
#     test_loss, correct = 0., 0.
#     progress_bar = tqdm.tqdm(test_loader, desc='Validation')
#     with torch.no_grad():
#         for data, target in progress_bar:
#             if args.cuda:
#                 data, target = data.cuda(), target.cuda()
#             data, target = Variable(data), Variable(target)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').data  # sum up batch loss
#             pred = output.data.max(1)[1]  # get the index of the max L23d_pmc-probability
#             correct += (target == pred).float().sum()
#
#     test_loss /= len(test_loader.dataset)
#
#     acc = correct / len(test_loader.dataset)
#
#     val_losses.append(test_loss)
#     val_accs.append(acc)
#
#     progress_bar.clear()
#     progress_bar.write(
#         '\nEpoch: {} validation test results - Average val_loss: {:.4f}, val_acc: {}/{} ({:.2f}%)'.format(
#             epoch, test_loss, correct, len(test_loader.dataset),
#             100. * correct / len(test_loader.dataset)))
#
#     return acc
#
#
# # In[10]:
#
#
# # Run the experiment
# def run_experiment(args):
#     total_minibatch_count = 0
#
#     torch.manual_seed(args.seed)
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
#
#     train_loader, test_loader, _, _, run_path = prepare_dataset(args)
#
#     epochs_to_run = args.epochs
#
#     # Choose model
#     # TODO add all the other models here if their parameter is specified
#     if args.model == 'default' or args.model == 'P2Q7DefaultChannelsNet':
#         model = Net()
#     elif args.model in globals():
#         model = globals()[args.model]()
#     else:
#         raise ValueError('Unknown model type: ' + args.model)
#
#     if args.cuda:
#         model.cuda()
#
#     # Choose optimizer
#     if args.optimizer == 'sgd':
#         optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#     elif args.optimizer == 'adam':
#         optimizer = optim.Adam(model.parameters())
#     elif args.optimizer == 'rmsprop':
#         optimizer = optim.RMSprop(model.parameters())
#     else:
#         raise ValueError('Unsupported optimizer: ' + args.optimizer)
#
#     # Run the primary training loop, starting with validation accuracy of 0
#     val_acc = 0
#     train_losses, train_accs = [], []
#     val_losses, val_accs = [], []
#
#     for epoch in range(1, epochs_to_run + 1):
#         # train for 1 epoch
#         total_minibatch_count = train(model, optimizer, train_loader,
#                                       epoch, total_minibatch_count,
#                                       train_losses, train_accs)
#         # validate progress on test dataset
#         val_acc = test(model, test_loader, epoch, total_minibatch_count,
#                        val_losses, val_accs)
#
#     fig, axes = plt.subplots(1, 4, figsize=(13, 4))
#     # plot the losses and acc
#     plt.title(args.name)
#     axes[0].plot(train_losses)
#     axes[0].set_title("Loss")
#     axes[1].plot(train_accs)
#     axes[1].set_title("Acc")
#     axes[2].plot(val_losses)
#     axes[2].set_title("Val loss")
#     axes[3].plot(val_accs)
#     axes[3].set_title("Val Acc")
#
#     # Write to csv file
#     with open(os.path.join(run_path + 'train.csv'), 'w') as f:
#         csvw = csv.writer(f, delimiter=',')
#         for loss, acc in zip(train_losses, train_accs):
#             csvw.writerow((loss, acc))
#
#     # Predict and Test
#     images, labels = next(iter(test_loader))
#     if args.cuda:
#         images, labels = images.cuda(), labels.cuda()
#     output = model(images)
#     predicted = torch.max(output, 1)[1]
#     fig, axes = plt.subplots(1, 6)
#     for i, (axis, img, lbl) in enumerate(zip(axes, images, predicted)):
#         if i > 5:
#             break
#         img = img.permute(1, 2, 0).squeeze()
#         axis.imshow(img)
#         axis.set_title(lbl.data.numpy())
#         axis.set_yticklabels([])
#         axis.set_xticklabels([])
#
#     if args.dataset == 'fashion_mnist' and val_acc > 0.92 and val_acc <= 1.0:
#         print("Congratulations, you beat the Question 13 minimum of 92"
#               "with ({:.2f}%) validation accuracy!".format(val_acc))
#
#
# # In[11]:
#
#
# run_experiment(Args(dataset='mnist', epochs=10))
#
# # In[ ]:
#
#
# run_experiment(Args(dataset='fashion_mnist', epochs=10))
#
# # In[ ]:
#
#
# # mnist 20 epochs
# run_experiment(Args(dataset='mnist', epochs=20))
#
# # In[ ]:
#
#
# # fashion_mnist 20 epochs
# run_experiment(Args(dataset='fashion_mnist', epochs=20))
#
# # In[ ]:
#
#
# # Q3 Change the SGD Learning Rate by a factor of
# #    - [0.1x, 1x, 10x]
# factor = [0.1, 1, 10]
# base_lr = 0.01
#
# # Q3 - 0.1
# run_experiment(Args(lr=base_lr * factor[0]))
#
# # In[ ]:
#
#
# # Q3 - 1x
# run_experiment(Args(lr=base_lr * factor[1]))
#
# # In[ ]:
#
#
# # Q3 - 10x
# run_experiment(Args(lr=base_lr * factor[2]))
#
# # In[ ]:
#
#
# # Q4 Compare Optimizers
# #   - [SGD, Adam, Rmsprop]
#
# # sgd
# optimizers = ['sgd', 'p1sgd', 'adam', 'rms_prop']
# run_experiment(Args(optimizer='optimizers[0]'))
#
# # In[ ]:
#
#
# # Q4 - Adam
# run_experiment(Args(optimizer='optimizers[2]'))
#
# # In[ ]:
#
#
# # Q4 - rms_prop
# run_experiment(Args(optimizer='optimizers[3]'))
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
# # Q6 - batch size
# batch_sizes_factors = [0.125, 1, 8]
# base_batch_size = 64
#
# batch_sizes = [int(base_batch_size * temp) for temp in batch_sizes_factors]
#
# # Q6 - 1/8
# run_experiment(Args(batch_size=batch_sizes[0]))
#
# # In[ ]:
#
#
# # Q6 - 1
#
# run_experiment(Args(batch_size=batch_sizes[1]))
#
# # In[ ]:
#
#
# run_experiment(Args(batch_size=batch_sizes[2]))
#
#
# # In[ ]:
#
#
#






# hierarchical level - 3
