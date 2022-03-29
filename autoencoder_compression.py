import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.nn import Conv2d
from torchvision import datasets,transforms
from torch.optim import Adam as adam
from celeba_dataloader import CelebA256x256

from_scratch=False

class Cnn_Block(nn.Module):
    def __init__(self,layers_num=3,kernel_size = (3,3)):
        super().__init__()
        self.block=nn.ModuleList()
        for i in range(layers_num):
            self.block.append(Conv2d(in_channels=3,out_channels=3,kernel_size=(3,3),stride=(1,1),padding = (1,1)))

    def forward(self,img):
        inputs = img
        for i in range(len(self.block)):
            inputs = nn.BatchNorm2d(3,device='cuda')(F.relu(self.block[i](inputs)))
        return inputs

class Encoder(nn.Module):
    
    def __init__(self,num_layers,pooling = 'no'):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.pooling = pooling
        for i in range(num_layers):
            self.conv_layers.append(Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), stride=(2,2),padding =(1,1)))

    def forward(self,imgs):

        inputs = imgs
        for i in range(len(self.conv_layers)):
            if self.pooling == 'max':
                inputs =  nn.MaxPool2d(2,stride=(2,2))(nn.BatchNorm2d(3,device='cuda')(F.relu(self.conv_layers[i](inputs))))
            elif self.pooling == 'avg':
                inputs = nn.AvgPool2d(2,stride=(2,2))(nn.BatchNorm2d(3,device='cuda')(F.relu(self.conv_layers[i](inputs))))
            else:
                inputs = nn.BatchNorm2d(3,device='cuda')(F.relu(self.conv_layers[i](inputs)))
            
            
        return inputs

class Decoder(nn.Module):
    def __init__(self,num_layers):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(2,2), stride=(2,2)))
    def forward(self,imbeddings):

        inputs = imbeddings
        for i in range(len(self.conv_layers)):
            inputs = nn.BatchNorm2d(3,device = 'cuda')(F.relu(self.conv_layers[i](inputs)))
        return inputs
class AutoEncoder(nn.Module):
    def __init__(self,num_layers):
        super().__init__()
        self.encoder = Encoder(num_layers).to('cuda')
        self.decoder = Decoder(num_layers).to('cuda')
        self.pre_block = Cnn_Block().to('cuda')
        self.post_block = Cnn_Block().to('cuda')

    def forward(self,imgs):
        embeds = self.encoder(self.pre_block(imgs))
        predicted_imgs = self.post_block(self.decoder(embeds))
        return predicted_imgs


from  torchvision.datasets import CIFAR10
# train = datasets.SVHN('.',split='train',download=True,transform = transforms.Compose([transforms.ToTensor()]))
# test = datasets.SVHN('.',split='valid',download=True,transform = transforms.Compose([transforms.ToTensor()]))
celeba_dl = CelebA256x256()
# train_set = torch.utils.data.DataLoader(train,batch_size=32,shuffle=True)
# test_set = torch.utils.data.DataLoader(test,batch_size = 32,shuffle=False)

auto_encoder = AutoEncoder(3).to('cuda')
if from_scratch==False:
    auto_encoder.load_state_dict(torch.load('model_checkpoints/model_100'))
batch_size = celeba_dl.get_batch_size()
print(list(auto_encoder.parameters()))
optimizer = adam(list(auto_encoder.parameters()),lr=0.001)

loss = nn.MSELoss()

number_of_epochs = 4

for i in range(number_of_epochs):
    for i in range(celeba_dl.get_num_of_batches()):
        X = celeba_dl.next_batch()
        auto_encoder.zero_grad()
        X=torch.tensor(X).to('cuda')
        predicted_imgs = auto_encoder(X)
        loss_output = loss(predicted_imgs,X)
        loss_output.backward()
        optimizer.step()
        if i%300==0:
            print(loss_output)
            torch.save(auto_encoder.state_dict(),'model_checkpoints/model_'+str(i))
    celeba_dl.on_epoch_ends()
from einops import rearrange
x  = X.to('cpu').detach().numpy()
img_dim = 8
x = rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=img_dim, b2=int(batch_size/img_dim))
x*=255
x = x.astype(np.uint8)
cv2.imwrite("org.jpg", x)

x  = predicted_imgs.to('cpu').detach().numpy()
x = rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=img_dim, b2=int(batch_size/img_dim))
x*=255
x = x.astype(np.uint8)
cv2.imwrite("predicted.jpg", x)
