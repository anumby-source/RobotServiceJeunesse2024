import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision  #to get the MNIST dataset


import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append('../fidle')
sys.path.append('.')
import fidle.pwk as ooo

# from fidle_pwk_additional import convergence_history_CrossEntropyLoss


class convergence_history_CrossEntropyLoss:
    def __init__(self):
        """
        Class to save the training converge properties
        """
        self.loss = nn.CrossEntropyLoss()
        self.history = {}  # Save convergence measures in the end of each epoch
        self.history['loss'] = []  # value of the cost function on training data
        self.history[
            'accuracy'] = []  # percentage of  correctly classified instances on training data (if classification)
        self.history['val_loss'] = []  # value of the cost function on validation data
        self.history[
            'val_accuracy'] = []  # percentage of  correctly classified instances on validation data (if classification)

    def update(self, current_model, xtrain, ytrain, xtest, ytest):
        # convergence information on the training data
        nb_training_obs = xtrain.shape[0]
        if nb_training_obs > xtest.shape[0]:
            nb_training_obs = xtest.shape[0]

        epoch_shuffler = np.arange(xtrain.shape[0])
        np.random.shuffle(epoch_shuffler)
        mini_batch_observations = epoch_shuffler[:nb_training_obs]
        var_X_batch = Variable(xtrain[mini_batch_observations, :]).float()
        var_y_batch = Variable(ytrain[mini_batch_observations])
        y_pred_batch = current_model(var_X_batch)
        curr_loss = self.loss(y_pred_batch, var_y_batch)

        self.history['loss'].append(curr_loss.item())
        self.history['accuracy'].append(float((torch.argmax(y_pred_batch, dim=1) == var_y_batch).float().mean()))

        # convergence information on the test data
        var_X_batch = Variable(xtest[:, :]).float()
        var_y_batch = Variable(ytest[:])
        y_pred_batch = current_model(var_X_batch)
        curr_loss = self.loss(y_pred_batch, var_y_batch)

        self.history['val_loss'].append(curr_loss.item())
        self.history['val_accuracy'].append(float((torch.argmax(y_pred_batch, dim=1) == var_y_batch).float().mean()))


class convergence_history_MSELoss:
    def __init__(self):
        """
        Class to save the training converge properties
        """
        self.loss = nn.MSELoss()
        self.MAE_loss = nn.L1Loss()
        self.history = {}  # Save convergence measures in the end of each epoch
        self.history['loss'] = []  # value of the cost function on training data
        self.history['mae'] = []  # mean absolute error on training data
        self.history['val_loss'] = []  # value of the cost function on validation data
        self.history['val_mae'] = []  # mean absolute error on validation data

    def update(self, current_model, xtrain, ytrain, xtest, ytest):
        # convergence information on the training data
        nb_training_obs = xtrain.shape[0]
        if nb_training_obs > xtest.shape[0]:
            nb_training_obs = xtest.shape[0]

        epoch_shuffler = np.arange(xtrain.shape[0])
        np.random.shuffle(epoch_shuffler)
        mini_batch_observations = epoch_shuffler[:nb_training_obs]
        var_X_batch = Variable(xtrain[mini_batch_observations, :]).float()
        var_y_batch = Variable(ytrain[mini_batch_observations]).float()
        y_pred_batch = current_model(var_X_batch)
        curr_loss = self.loss(y_pred_batch.view(-1), var_y_batch.view(-1))

        self.history['loss'].append(curr_loss.item())
        self.history['mae'].append(self.MAE_loss(y_pred_batch.view(-1), var_y_batch.view(-1)).item())

        # convergence information on the test data
        var_X_batch = Variable(xtest[:, :]).float()
        var_y_batch = Variable(ytest[:]).float()
        y_pred_batch = current_model(var_X_batch)
        curr_loss = self.loss(y_pred_batch.view(-1), var_y_batch.view(-1))

        self.history['val_loss'].append(curr_loss.item())
        self.history['val_mae'].append(self.MAE_loss(y_pred_batch.view(-1), var_y_batch.view(-1)).item())


#get and format the training set
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
x_train=mnist_trainset.data.type(torch.DoubleTensor)
y_train=mnist_trainset.targets


#get and format the test set
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
x_test=mnist_testset.data.type(torch.DoubleTensor)
y_test=mnist_testset.targets

#check data shape and format
print("Size of the train and test observations")
print(" -> x_train : ",x_train.shape)
print(" -> y_train : ",y_train.shape)
print(" -> x_test  : ",x_test.shape)
print(" -> y_test  : ",y_test.shape)

print("\nRemark that we work with torch tensors and not numpy arrays:")
print(" -> x_train.dtype = ",x_train.dtype)
print(" -> y_train.dtype = ",y_train.dtype)

print('Before normalization : Min={}, max={}'.format(x_train.min(),x_train.max()))

xmax=x_train.max()
x_train = x_train / xmax
x_test  = x_test  / xmax

print('After normalization  : Min={}, max={}'.format(x_train.min(),x_train.max()))

np_x_train=x_train.numpy().astype(np.float64)
np_y_train=y_train.numpy().astype(np.uint8)

# display some images from the train set
# ooo.plot_images(np_x_train,np_y_train , [27],  x_size=5,y_size=5, colorbar=True)
# ooo.plot_images(np_x_train,np_y_train, range(5,41), columns=12)


class MyModel(nn.Module):
    """
    Basic fully connected neural-network
    """

    def __init__(self):
        hidden1 = 100
        hidden2 = 100
        super(MyModel, self).__init__()
        self.hidden1 = nn.Linear(28 * 28, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.hidden3 = nn.Linear(hidden2, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # flatten the images before using fully-connected layers
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.softmax(x, dim=0)
        return x


model = MyModel()

def fit(model, X_train, Y_train, X_test, Y_test, EPOCHS=5, BATCH_SIZE=32):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # lr is the learning rate
    model.train()

    history = convergence_history_CrossEntropyLoss()

    history.update(model, X_train, Y_train, X_test, Y_test)

    n = X_train.shape[0]  # number of observations in the training data

    step = int(EPOCHS / 10)

    # stochastic gradient descent
    for epoch in range(EPOCHS):

        if epoch % step == 0:
            print("epoch=", epoch)

        batch_start = 0
        epoch_shuffler = np.arange(n)
        np.random.shuffle(epoch_shuffler)  # remark that 'utilsData.DataLoader' could be used instead

        while batch_start + BATCH_SIZE < n:
            # get mini-batch observation
            mini_batch_observations = epoch_shuffler[batch_start:batch_start + BATCH_SIZE]
            var_X_batch = Variable(X_train[mini_batch_observations, :, :]).float()  # the input image is flattened
            var_Y_batch = Variable(Y_train[mini_batch_observations])

            # gradient descent step
            optimizer.zero_grad()  # set the parameters gradients to 0
            Y_pred_batch = model(var_X_batch)  # predict y with the current NN parameters

            curr_loss = loss(Y_pred_batch, var_Y_batch)  # compute the current loss
            curr_loss.backward()  # compute the loss gradient w.r.t. all NN parameters
            optimizer.step()  # update the NN parameters

            # prepare the next mini-batch of the epoch
            batch_start += BATCH_SIZE

        history.update(model, X_train, Y_train, X_test, Y_test)

    return history

batch_size  = 512
epochs      = 256

# Train the model
history = fit(model, x_train, y_train, x_test, y_test, EPOCHS=epochs, BATCH_SIZE = batch_size)

# Make predictions using the test data
var_x_test = Variable(x_test[:,:,:]).float()
var_y_test = Variable(y_test[:])
y_pred = model(var_x_test)

loss = nn.CrossEntropyLoss()
curr_loss = loss(y_pred, var_y_test)

val_loss = curr_loss.item()
val_accuracy  = float( (torch.argmax(y_pred, dim=1) == var_y_test).float().mean() )

print('Test loss     :', val_loss)
print('Test accuracy :', val_accuracy)

ooo.plot_history(history, figsize=(6,4))

y_pred = model(var_x_test)
np_y_pred_label = torch.argmax(y_pred, dim= 1).numpy().astype(np.uint8)

np_x_test=x_test.numpy().astype(np.float64)
np_y_test=y_test.numpy().astype(np.uint8)

ooo.plot_images(np_x_test, np_y_test, range(0,60), columns=12, x_size=1, y_size=1, y_pred=np_y_pred_label)

errors=[ i for i in range(len(np_y_test)) if np_y_pred_label[i]!=np_y_test[i] ]
errors=errors[:min(24,len(errors))]
ooo.plot_images(np_x_test, np_y_test, errors[:15], columns=6, x_size=2, y_size=2, y_pred=np_y_pred_label)

ooo.display_confusion_matrix(np_y_test,np_y_pred_label, range(10))

