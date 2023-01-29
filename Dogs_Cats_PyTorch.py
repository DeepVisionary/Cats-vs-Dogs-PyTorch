#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Dogs vs Cats, not cleaned yet


# In[5]:


import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from PIL import Image
from torch import optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Currently operating on {device}')
import cv2, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from torchsummary import summary


# In[4]:


# If operating in Colab, uncomment and run the cell for extraction of data.
#!pip install -q kaggle
#from google.colab import files
#files.upload() #Use your Kaggle Token API from the downloads

#!mkdir -p ~/.kaggle
#!cp kaggle.json ~/.kaggle/
#!ls ~/.kaggle
#!chmod 600 /root/.kaggle/kaggle.json

#!kaggle datasets download -d tongpython/cat-and-dog

#!unzip cat-and-dog.zip


# In[19]:


import os
os.path.exists('C:/Users/tommy/git/Cats-vs-Dogs-PyTorch/archive(1)/training_set/training_set/') #Check for the path


# In[37]:


train_data_dir = 'C:/Users/tommy/git/Cats-vs-Dogs-PyTorch/archive(1)/training_set/training_set/'
test_data_dir = 'C:/Users/tommy/git/Cats-vs-Dogs-PyTorch/archive(1)/test_set/test_set/'


# In[38]:


from torch.utils.data import DataLoader, Dataset
from random import shuffle, seed 
seed(10)
class cats_dogs(Dataset):
    def __init__(self, folder):
        cats = glob(folder+'/cats/*.jpg')
        dogs = glob(folder+'/dogs/*.jpg')
        self.fpaths = cats + dogs
        shuffle(self.fpaths)
        self.targets = [fpath.split('/')[-1].startswith('dog') for fpath in self.fpaths]
    def __len__(self):
        return len(self.fpaths)
    def __getitem__ (self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im = (cv2.imread(f)[:,:,::-1])
        im = cv2.resize(im, (224,224))
        return torch.tensor(im/255).permute(2,0,1).to(device).float(), torch.tensor([target]).float().to(device)

    


# In[39]:


data = cats_dogs(train_data_dir)


# In[40]:


im, label = data[200]
plt.imshow(im.permute(1,2,0).cpu())
print(label)


# In[41]:


def conv_layer(ni,no, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(ni,no, kernel_size, stride),
        nn.ReLU(),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(2))
  
def get_model():
    model = nn.Sequential(
        conv_layer(3,64,3),
        conv_layer(64,512,3),
        conv_layer(512,512,3),
        conv_layer(512,512,3),
        conv_layer(512,512,3),
        conv_layer(512,512,3),
        nn.Flatten(),
        nn.Linear(512,1),
        nn.Sigmoid(),
        ).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


# In[42]:


model, loss_fn, optimizer = get_model()
summary(model, input_size=(3,224,224))


# In[43]:


def get_data():
    
    train = cats_dogs(train_data_dir)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)

    val = cats_dogs(test_data_dir)
    val_dl = DataLoader(val, batch_size=32, shuffle=True, drop_last = True)

    return trn_dl, val_dl

def train_batch(x,y,model,opt,loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x,y,model):
    prediction = model(x)
    is_correct = (prediction>0.5) == y
    return is_correct.cpu().numpy().tolist()
@torch.no_grad()
def val_loss(x,y,model):
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()


# In[44]:


trn_dl, val_dl = get_data()
model, loss_fn, optimizer = get_model()


# In[45]:


train_losses, train_accuracies = [],[]
val_losses, val_accuracies = [],[]

for epoch in range(10):
    print(f'Completing Epoch: {epoch}')
    train_epoch_losses, train_epoch_accuracies = [],[]
    val_epoch_accuracies = []
    

    for ix, batch in enumerate(iter(trn_dl)):
        
        x,y = batch
        batch_loss = train_batch(x,y,model, optimizer, loss_fn)
        train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()
  
    for ix, batch in enumerate(iter(trn_dl)):
        x,y = batch
        is_correct = accuracy(x,y,model)
        train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

    for ix, batch in enumerate(iter(val_dl)):
        x,y = batch
        val_is_correct = accuracy(x,y,model)
        val_epoch_accuracies.extend(val_is_correct)
        val_epoch_accuracy = np.mean(val_epoch_accuracies)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_accuracies.append(val_epoch_accuracy)


# In[46]:


epochs = np.arange(10)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.plot(epochs, train_accuracies, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_accuracies, 'r', label = 'Validation Accuracy')
plt.gca().xaxis.set_major_locator(mtick.MultipleLocator(1))
plt.title('Training and validation accuracy with 4k data points used for training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()
plt.grid('off')
plt.show()


# In[47]:


torch.save(model.state_dict(), 'C:/Users/tommy/git/Cats-vs-Dogs-PyTorch/Cats_Dogs_Trained.pth')


# In[ ]:


# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()


# In[50]:


def predict_custom_img(PATH):
    img = cv2.imread(PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img, (224,224))
    img1 = torch.tensor(img1/255).permute(2,0,1).to(device).float()
    img1 = torch.unsqueeze(img1,0)
    img1 = img1.permute(0,1,2,3)
    print(img1.shape)
    prediction = model(img1)
    if prediction < 0.5:
        a = str(f'This is a cat with a score: {prediction}')
    else:
        a = str(f'This is a dog with a score: {prediction}')
    plt.imshow(img)
    plt.title(a)
    return img1


# In[51]:


'C:/Users/tommy/git/Cats-vs-Dogs-PyTorch/Custom Images Pred/husky.jpg'


# In[52]:


def predict_custom_img_features(PATH):
    img = cv2.imread(PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img, (224,224))
    img1 = torch.tensor(img1/255).permute(2,0,1).to(device).float()
    img1 = torch.squeeze(img1,0)
    print(f'The tensor shape is {img1.shape}')
    return img1


# In[54]:


im = predict_custom_img_features('C:/Users/tommy/git/Cats-vs-Dogs-PyTorch/Custom Images Pred/husky.jpg')
first_layer = nn.Sequential(*list(model.children())[:1])
intermediate_output = first_layer(im[None])[0].detach()


# In[55]:


fig, ax = plt.subplots(7,1, figsize = (20,20))
for ix, axis in enumerate(ax.flat):
  axis.set_title('Filter:'+str(ix))
  axis.imshow(intermediate_output[ix].cpu())
plt.tight_layout()
plt.show()


# In[118]:


im = predict_custom_img_features('/content/husky.jpg')
first_layer = nn.Sequential(*list(model.children())[:1])
intermediate_output = first_layer(im[None])[0].detach()

fig, ax = plt.subplots(7,1, figsize = (20,20))
for ix, axis in enumerate(ax.flat):
  axis.set_title('Filter:'+str(ix))
  axis.imshow(intermediate_output[ix].cpu())
plt.tight_layout()
plt.show()


# In[114]:


#Cat
im = predict_custom_img_features('/content/phoebe.jpg')
first_layer = nn.Sequential(*list(model.children())[:1])
intermediate_output = first_layer(im[None])[0].detach()
fig, ax = plt.subplots(7,1, figsize = (20,20))
for ix, axis in enumerate(ax.flat):
  axis.set_title('Filter:'+str(ix))
  axis.imshow(intermediate_output[ix].cpu())
  #axis.imshow(cv2.imread('/content/phoebe.jpg'))
plt.tight_layout()
plt.show()


# In[ ]



