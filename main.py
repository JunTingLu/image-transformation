import numpy as np
import pandas as pd
import os, math, sys
import time, datetime
import glob, itertools
import argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
import plotly
from scipy import signal
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from keras.callbacks import TensorBoard
from tensorboard import notebook

#%%
""" input parameters"""
# 查詢gpu數
torch.cuda.device_count()

random.seed(42)
import warnings
warnings.filterwarnings("ignore")
torch.__version__
torch.cuda.is_available()
### Settings 
# path to pre-trained models
pretrained_model_path = "/content/gdrive/MyDrive/ColabNotebooks/cycleGAN_datasets/vangogh2photo/vangogh2photo/"
# epoch to start training from
epoch_start = 1
# number of epochs of training
n_epochs = 3
# name of the dataset
dataset_path = "/content/gdrive/MyDrive/ColabNotebooks/cycleGAN_datasets/vangogh2photo/vangogh2photo/"
# size of the batches"
batch_size = 1
# adam: learning rate
lr = 0.00012
# adam: decay of first order momentum of gradient
b1 = 0.2
# adam: decay of first order momentum of gradient
b2 = 0.999
# epoch from which to start lr decay
decay_epoch = 1
# number of cpu threads to use during batch generation
n_workers = 0
# size of image height
img_height = 256
# size of image width
img_width = 256
# number of image channels
channels = 3
# interval between saving generator outputs
sample_interval = 100
# interval between saving model checkpoints
checkpoint_interval = -1
# number of residual blocks in generator
n_residual_blocks = 9
# cycle loss weight
lambda_cyc = 10.0
# identity loss weight
lambda_id = 5.0
# Development / Debug Mode
debug_mode = False

#%%
""" define image buffer and load data function """
### Define Utilities
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

# save generated images in buffer
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
                
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
                    
        #notice the Variable() type
        return Variable(torch.cat(to_return)) 

### Define Dataset Class
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train", debug_mode=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B") + "/*.*"))

        self.files_A = self.files_A[:100] if debug_mode else self.files_A
        self.files_B = self.files_B[:100] if debug_mode else self.files_B

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        """
        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
        """
        # [revised] More efficient way
        if self.unaligned:
            index_B = random.randint(0, len(self.files_B) - 1)
        else:
            index_B = index % len(self.files_B)
        image_B = Image.open(self.files_B[index_B])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        return {"A": self.transform(image_A), 
                "B": self.transform(image_B)}

    def __len__(self):
        Maximum = max(len(self.files_A), len(self.files_B))
        
        return Maximum
#%%
""" image preproccessing """
def transformGenerator():
    res = [
        # 將圖片先擴大
        transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
        # 隨機裁減圖片來取樣本
        transforms.RandomCrop((img_height, img_width)),
        # 隨機將圖片左右翻轉
        transforms.RandomHorizontalFlip(),
        # 將PIL Image轉換為Tensor，並除255使數值界於0~1之間
        transforms.ToTensor(),
        # 標準化處理，使模型更容易收斂
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    return res

### Get Train/Test Dataloaders
# Image transformations
transforms_ = transformGenerator()
test_transforms_ = transformGenerator()

#%%
""" define models """
# 權重初始化
def weights_init_normal(m):
    # [revised] for readability
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    #elif classname.find("BatchNorm2d") != -1:

    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, expansion=1, num_blocks=2,use_dropout=False):
        super(ResidualBlock, self).__init__()

        layers = []

        # in_featuresCopy = in_features* expansion
        #print(in_features)

        for _ in range(num_blocks):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features * expansion, in_features, 3),
                nn.InstanceNorm2d(in_features),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features * expansion, in_features, 3),
                nn.InstanceNorm2d(in_features),
            ]

            if use_dropout:
                layers += [nn.Dropout(0.2)]

        self.block = nn.Sequential(*layers)

        #print(in_features)

    def forward(self, x):
        #print(x.shape)
        #print(self.block(x).shape)
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), 
                  nn.Conv2d(out_features, channels, 7), 
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers


        """ 
        *discriminator_block
        the '*' here is for disclose the list()
        """
        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            ResidualBlock(64),
            *discriminator_block(64, 128),
            ResidualBlock(128),
            *discriminator_block(128, 256),
            ResidualBlock(256, use_dropout=True),
            *discriminator_block(256, 512),
            ResidualBlock(512, use_dropout=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
    
#%%
""" define the discriminator/generator and setup gpu """
### Train CycleGAN
torch.cuda.is_available()
# Losses criterion
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()
input_shape = (channels, img_height, img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)


# 若調用gpu
if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# run with gpu




#%%
""" function of  loss"""
def lossGenerator(real_A, real_B, valid):
    # Identity loss
    loss_id_A = criterion_identity(G_BA(real_A), real_A)
    loss_id_B = criterion_identity(G_AB(real_B), real_B)
    loss_identity = (loss_id_A + loss_id_B) / 2

    # GAN loss
    fake_B = G_AB(real_A)
    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
    fake_A = G_BA(real_B)
    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

    # Cycle loss
    recov_A = G_BA(fake_B)
    loss_cycle_A = criterion_cycle(recov_A, real_A)
    recov_B = G_AB(fake_A)
    loss_cycle_B = criterion_cycle(recov_B, real_B)
    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
    
    return (fake_A, fake_B, loss_identity, loss_GAN, loss_cycle)
#%%
""" function of generator """
def train_G(real_A, real_B, valid):
    ### Train Generators
    optimizer_G.zero_grad()

    fake_A, fake_B, loss_identity, loss_GAN, loss_cycle = lossGenerator(real_A, real_B, valid)
    # Total loss
    loss_G = lambda_id * loss_identity + loss_GAN + lambda_cyc * loss_cycle
    loss_G.backward()
    optimizer_G.step()
    return fake_A, fake_B, loss_G, loss_identity, loss_GAN, loss_cycle

def train_D(D_type, real_A, fake_A, real_B, fake_B, fake, valid, fake_A_buffer, fake_B_buffer):
  if (D_type == 0):
    ### Train Discriminator-A
    D_A.train()
    optimizer_D_A.zero_grad()
    # Real loss
    loss_real = criterion_GAN(D_A(real_A), valid)
    # Fake loss (on batch of previously generated samples)
    fake_A_ = fake_A_buffer.push_and_pop(fake_A)
    loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
    # Total loss
    loss_D_A = (loss_real + loss_fake) / 2
    loss_D_A.backward()
    optimizer_D_A.step()
    return loss_D_A
  else:
    ### Train Discriminator-B
    D_B.train()
    optimizer_D_B.zero_grad()
    # Real loss
    loss_real = criterion_GAN(D_B(real_B), valid)
    # Fake loss (on batch of previously generated samples)
    fake_B_ = fake_B_buffer.push_and_pop(fake_B)
    loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
    # Total loss
    loss_D_B = (loss_real + loss_fake) / 2
    loss_D_B.backward()
    optimizer_D_B.step()
    return loss_D_B
  
  #%%
""" training...."""
def train(train_dataloader,log):  

    # 紀錄訓練曲線
    # writer = SummaryWriter()
    # # write in tensorboard 
    writer = SummaryWriter(log)

    # define the tensor
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    train_counter = []
    train_losses_gen, train_losses_id, train_losses_gan, train_losses_cyc = [], [], [], []
    train_losses_disc, train_losses_disc_a, train_losses_disc_b = [], [], []

    # test_counter = [2*idx*len(train_dataloader.dataset) for idx in range(epoch_start+1, n_epochs+1)]
    # test_losses_gen, test_losses_disc = [], []

    for epoch in range(epoch_start, n_epochs):
        
        #### Training
        loss_gen = loss_id = loss_gan = loss_cyc = 0.0
        loss_disc = loss_disc_a = loss_disc_b = 0.0
        tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))

        for batch_idx, batch in enumerate(tqdm_bar):
            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            
            # Adversarial ground truths (label 1/0 in pics)
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            ### Train Generators
            G_AB.train()
            G_BA.train()
            fake_A, fake_B, loss_G, loss_identity, loss_GAN, loss_cycle = train_G(real_A, real_B, valid)
            
            loss_D_A = train_D(0, real_A, fake_A, real_B, fake_B, fake, valid, fake_A_buffer, fake_B_buffer)
            loss_D_B = train_D(1, real_A, fake_A, real_B, fake_B, fake, valid, fake_A_buffer, fake_B_buffer)
            loss_D = (loss_D_A + loss_D_B) / 2

            ### Log Progress --> item()檢索張量
            loss_gen += loss_G.item()
            loss_id += loss_identity.item()
            loss_gan += loss_GAN.item()
            loss_cyc += loss_cycle.item()
            loss_disc += loss_D.item()
            loss_disc_a += loss_D_A.item()
            loss_disc_b += loss_D_B.item()
            # 儲存當前訓練樣本數(batch_idx:批次索引;batch_size:批次大小;real_A.size(0):當前樣本數)
            # 乘以2代表每個epoch會進行兩次訓練(一次生成器，一次判別器)
            # train_counter.append(2*(batch_idx*batch_size + real_A.size(0) + epoch*len(train_dataloader.dataset)))
            train_losses_gen.append(loss_G.item())
            train_losses_id.append(loss_identity.item())
            train_losses_gan.append(loss_GAN.item())
            train_losses_cyc.append(loss_cycle.item())
            train_losses_disc.append(loss_D.item())
            train_losses_disc_a.append(loss_D_A.item())
            train_losses_disc_b.append(loss_D_B.item())
            tqdm_bar.set_postfix(
                    Gen_loss=loss_gen/(batch_idx+1), 
                    identity=loss_id/(batch_idx+1),
                    adv=loss_gan/(batch_idx+1), 
                    cycle=loss_cyc/(batch_idx+1),
                    Disc_loss=loss_disc/(batch_idx+1), 
                    disc_a=loss_disc_a/(batch_idx+1), 
                    disc_b=loss_disc_b/(batch_idx+1))
            

            # # write in tensorboard 
            # writer = SummaryWriter(log)
            writer.add_scalar('Loss/Generator', loss_gen, epoch )
            writer.add_scalar('Loss/Identity', loss_id, epoch )
            writer.add_scalar('Loss/Adversarial', loss_gan, epoch )
            writer.add_scalar('Loss/Cycle-Consistency', loss_cyc, epoch)
            writer.add_scalar('Loss/Discriminator', loss_disc, epoch)
            
    return train_dataloader, fake_A_buffer, fake_B_buffer, loss_gen,loss_disc

#%%
""" define test model """
def test_D(D_type, real_A, fake_A, real_B, fake_B, fake, valid, fake_A_buffer, fake_B_buffer):
  if D_type == 0:
    ### Test Discriminator-A
    D_A.eval()
    # Real loss
    loss_real = criterion_GAN(D_A(real_A), valid)
    # Fake loss (on batch of previously generated samples)
    fake_A_ = fake_A_buffer.push_and_pop(fake_A)
    loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
    # Total loss
    loss_D_A = (loss_real + loss_fake) / 2
    return loss_D_A
  else:
    ### Test Discriminator-B
    D_B.eval()
    # Real loss
    loss_real = criterion_GAN(D_B(real_B), valid)
    # Fake loss (on batch of previously generated samples)
    fake_B_ = fake_B_buffer.push_and_pop(fake_B)
    loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
    # Total loss
    loss_D_B = (loss_real + loss_fake) / 2
    return loss_D_B
  
""" testing.... """  
def test(save_path,test_dataloader,train_dataloader,fake_A_buffer,fake_B_buffer): 
    
  #### Testing (model reload and show results)
  # define the tensorpip
  Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
  # test_counter = [2*idx*len(train_dataloader.dataset) for idx in range(epoch_start+1, n_epochs+1)]
  test_losses_gen, test_losses_disc = [], []
  # save_path='/content/gdrive/MyDrive/ColabNotebooks/cycleGAN_datasets/vangogh2photo/vangogh2photo/'

  for epoch in range(epoch_start, n_epochs):
    loss_gen = loss_id = loss_gan = loss_cyc = 0.0
    loss_disc = loss_disc_a = loss_disc_b = 0.0
    tqdm_bar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch} ', total=int(len(test_dataloader)))

    for batch_idx, batch in enumerate(tqdm_bar):
        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths (label)
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        ### Test Generators
        G_AB.eval()
        G_BA.eval()
        
        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2
        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        # Total loss
        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

        loss_D_A = test_D(0, real_A, fake_A, real_B, fake_B, fake, valid)
        loss_D_B = test_D(1, real_A, fake_A, real_B, fake_B, fake, valid)
        loss_D = (loss_D_A + loss_D_B) / 2
        
        ### Log Progress
        loss_gen += loss_G.item(); 
        loss_id += loss_identity.item(); 
        loss_gan += loss_GAN.item(); 
        loss_cyc += loss_cycle.item()
        loss_disc += loss_D.item(); 
        loss_disc_a += loss_D_A.item(); 
        loss_disc_b += loss_D_B.item()
        tqdm_bar.set_postfix(Gen_loss=loss_gen/(batch_idx+1), identity=loss_id/(batch_idx+1), adv=loss_gan/(batch_idx+1), cycle=loss_cyc/(batch_idx+1),
                            Disc_loss=loss_disc/(batch_idx+1), disc_a=loss_disc_a/(batch_idx+1), disc_b=loss_disc_b/(batch_idx+1))
        
    # If at sample interval save image
    # if random.uniform(0,1)<1:
    if epoch % n_epochs==0:
        # Arrange images along x-axis
        real_A = make_grid(real_A, nrow=1, normalize=True)
        real_B = make_grid(real_B, nrow=1, normalize=True)
        fake_A = make_grid(fake_A, nrow=1, normalize=True)
        fake_B = make_grid(fake_B, nrow=1, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B), -1)
        save_image(image_grid, f"save_path{epoch}_{batch_idx}.png", normalize=False)

    test_losses_gen.append(loss_gen/len(test_dataloader))
    test_losses_disc.append(loss_disc/len(test_dataloader))

    # Save model checkpoints
    # len(test_losses_gen)-1為測試集生成損失列表的最後一個元素的索引(n從0開始)
    if np.argmin(test_losses_gen) == len(test_losses_gen)-1:
      # Save model checkpoints
      torch.save(G_AB.state_dict(), create_folder(save_path)+"G_AB.pth")
      torch.save(G_BA.state_dict(), create_folder(save_path)+"G_BA.pth")
      torch.save(D_A.state_dict(), create_folder(save_path)+"D_A.pth")
      torch.save(D_B.state_dict(), create_folder(save_path)+"D_B.pth")  

#%%
""" function of create new folder """
# 儲存新創建的資料夾中
def create_folder(dataset_path):
  folder_path = os.path.dirname(dataset_path)
  new_folder_path = os.path.join(folder_path, 'pth-files/')

  if not os.path.isdir(new_folder_path):
    os.mkdir(new_folder_path)
  
  return  new_folder_path 
#%%
""" multiple-data training/testing """
def DataLoad(dataset_path,batch_size,transforms_):
  train_dataloader = DataLoader(
      ImageDataset(f"{dataset_path}", transforms_=transforms_, unaligned=True),
      batch_size=batch_size,
      shuffle=True,
      num_workers=0)

  test_dataloader = DataLoader(
      ImageDataset(f"{dataset_path}", transforms_=test_transforms_, unaligned=True),
      batch_size=batch_size,
      shuffle=True,
      num_workers=0)
  return train_dataloader,test_dataloader


# optimizer
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))
# log with process
log_dir='logs_result/'

# set training parameters
learning_rate = 0.001
batch_size = 1
num_epochs = 10
# define dataset class
Dataset_Monet='/content/gdrive/MyDrive/ColabNotebooks/cycleGAN_datasets/monet2photo/monet2photo/'
Dataset_Vango='/content/gdrive/MyDrive/ColabNotebooks/cycleGAN_datasets/vangogh2photo/vangogh2photo/'

dataset_path=[Dataset_Monet,Dataset_Vango]

for i in range(len(dataset_path)):

  train_loader,test_loader = DataLoad(dataset_path[i],batch_size,transforms_)
  print('start training')
  # log process
  log = dataset_path[i]+log_dir
  # training 
  train_dataloader,fake_A_buffer,fake_B_buffer = train(train_loader,log)
  
  # testing
  test(dataset_path[i],test_loader,train_dataloader,fake_A_buffer,fake_B_buffer)
  print('Finished!')


#%%
# 啟動tensorboard查看訓練曲線
""" inspect log """
# terminal 輸入
# %load_ext tensorboard
# %tensorboard --logdir=/cycleGAN_datasets/monet2photo/monet2photo/logs_result/events.out.tfevents.1680885225
# %reload_ext tensorboard