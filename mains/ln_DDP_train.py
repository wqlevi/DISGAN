# TODO
# -[x] ViT, SwinIR backbone
# -[x] YAML file for config experiment
# -[x] residual connection in Unet, improving PSNR
# -[ ] Logging experiments with logging libarary
# -[x] frequency guided Dnet
import os, sys, argparse
#sys.path.append("../")
import wandb
import glob
import nibabel as nb
try:
    from lightning.pytorch.loggers import WandbLogger
    import lightning as L
    from L.pytorch.callbacks import ModelCheckpoint
except:
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    import pytorch_lightning as L
    from pytorch_lightning.strategies import DDPStrategy
from torchmetrics.functional import peak_signal_noise_ratio,structural_similarity_index_measure
#from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import DataLoader, random_split
from importlib import import_module
import pathlib
from utils import utils
from model.model_base import freeze_model

class DataModule(L.LightningDataModule):
    def __init__(
            self,
            hr_shape: int,
            data_dir:str,
            batch_size :int,
            num_workers :int=2
            ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.shape = hr_shape
        self.lr_shape = int(hr_shape/2)
        self.norm = transforms.Lambda(lambda x : (x - mean)/std)
        self.files = sorted(glob.glob(self.data_dir+'/*nii*'))
    def __getitem__(self,index):
        img = nb.load(self.files[index%len(self.files)]).get_fdata()
        img = torch.Tensor(img).unsqueeze(0).unsqueeze(0) #make it 5D
        img_hr = img.squeeze(0)
        img_lr = torch.nn.functional.interpolate(img,32).squeeze(0)
        return {"lr":img_lr, "hr":img_hr}
    def __len__(self):
        return len(self.files)
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            trainset = DataModule(self.shape, self.data_dir, self.batch_size, self.num_workers)
            self.trainset,self.validset = random_split(trainset, [len(trainset)//4, len(trainset)-len(trainset)//4])
    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            drop_last = True
                )
    def val_dataloader(self):
        return DataLoader(
                self.validset,
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                drop_last = True
                )
class GAN(L.LightningModule):
    def __init__(
            self,
            config,
            channels:int=1,
            #hr_shape:tuple,
            lr:float=.0001,
            batch_size:int=32,
            update_FE:bool=False,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.batch_size = batch_size
        self.standarize = lambda x: (x-x.min())/(x.max()-x.min())
        # updating attr needed
        self.fe_sv_list = []
        self.d_sv_list = []
        self.g_sv_list = []

        self.update_FE = update_FE
        self.log_images_interval = 1000
        self.psnr_cal = utils.psnr
        self.ssim_cal = utils.ssim
        #self.lpips_cal = LearnedPerceptualImagePatchSimilarity(net='vgg', reduction='mean')

        mod = import_module("model.model_DISGAN")
        self.generator = getattr(mod, config.G_type)()
        self.discriminator = getattr(mod, config.D_type)()
        self.FE = getattr(mod, config.FE_type)()

        self.noise_mean = torch.zeros((self.batch_size, *self.discriminator.input_shape))

        if not self.update_FE:
            freeze_model(self.FE)
    def forward(self,x):
        return self.generator(x)
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)
    def l1_loss(self, y_hat, y):
        return F.l1_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs_hr = batch['hr']
        imgs_lr = batch['lr']
        optimizer_g, optimizer_d = self.optimizers()
        self.step_sigma = 1/self.trainer.max_epochs

        valid = torch.ones((self.batch_size, *self.discriminator.output_shape),dtype=torch.float32)
        fake = torch.zeros((self.batch_size, *self.discriminator.output_shape),dtype=torch.float32)

        sigma_numerics = 1 - self.current_epoch * self.step_sigma
        sigma_numerics = max(sigma_numerics, 0)
        sigma = torch.full((self.batch_size, *self.discriminator.input_shape), sigma_numerics)

        instancenoise = torch.normal(mean = self.noise_mean, std=sigma).type_as(imgs_lr)
        valid = valid.type_as(imgs_lr)
        fake = fake.type_as(imgs_lr)

        #---Update G---#
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(imgs_lr)

        pred_real = self.discriminator(imgs_hr + instancenoise).detach()
        pred_fake = self.discriminator(self.generated_imgs + instancenoise)
        g_adv_loss = self.adversarial_loss( pred_fake - pred_real.mean(0,keepdim=True), valid)
        g_pixel_loss = self.l1_loss(self.FE(self.generated_imgs),self.FE(imgs_hr))
        g_content_loss = self.l1_loss(self.generated_imgs, imgs_hr)
        g_loss = g_content_loss + 5e-3*g_adv_loss + 1e-2*g_pixel_loss
        self.log("g_loss", g_loss, prog_bar = True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        #---Update D---#
        self.toggle_optimizer(optimizer_d)
        pred_real = self.discriminator(imgs_hr + instancenoise)
        pred_fake = self.discriminator(self.generated_imgs.detach() + instancenoise)
        loss_real = self.adversarial_loss(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.adversarial_loss(pred_fake - pred_real.mean(0, keepdim=True), fake)

        d_loss = (loss_real + loss_fake)/2
        self.log("d_loss", d_loss, prog_bar = True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        grid_sr = make_grid(self.generated_imgs[:,:,imgs_hr.size(2)//2])
        grid_hr = make_grid(imgs_hr[:,:,imgs_hr.size(2)//2])

        fe_sv_list, g_sv_list = [],[]
        with torch.no_grad():
            psnr = self.psnr_cal(self.generated_imgs.squeeze().cpu().numpy(), imgs_hr.squeeze().cpu().numpy())
            ssim = self.ssim_cal(self.generated_imgs.squeeze().cpu().numpy(), imgs_hr.squeeze().cpu().numpy())
        #-----Logging-----#
        self.log("PNSR", psnr)
        self.log("SSIm", ssim)
        if batch_idx % self.log_images_interval == 0:
            self.logger.log_image("Results", [grid_sr, grid_hr], caption=["SR", "GT"])

        #print(f'{(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0))/(1024**2)}MB')

    def configure_optimizers(self):
        if self.update_FE:
            opt_g = torch.optim.Adam(list(self.FE.parameters())+list(self.generator.parameters()), lr=self.hparams.lr)
        else:
            opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return [opt_g, opt_d], []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str, default='test')
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--plot_per_iter',type=int, default=1000)
    parser.add_argument('--image_size',type=int, default=64)
    parser.add_argument('--batch_size',type=int, default=4)
    parser.add_argument('--num_epochs',type=int, default=10)
    parser.add_argument('--input_channel',type=int, default=1)
    parser.add_argument('--lr',type=float, default=1e-4)
    parser.add_argument('--arch_type',type=str, default='VGG16')
    parser.add_argument('--n_gpus',type=int, default=1)
    parser.add_argument('--n_nodes',type=int, default=1)
    parser.add_argument('--update_FE',action='store_true', help='update only when such flag is added to execute the script') # 
    parser.add_argument('--name_ckp',type=str, default="no_name")
    parser.add_argument('--D_type',type=str, default="no_name")
    parser.add_argument('--G_type',type=str, default="no_name")
    parser.add_argument('--FE_type',type=str, default="VGG16")
    parser.add_argument('--activation_layer', type=str, default='relu')
    parser.add_argument('--G_BN', type=int, default=0)

    opt = parser.parse_args()
    pwd_path = pathlib.Path().resolve()
    dict_yaml = utils.load_yaml(f"config/{opt.model_name}")
    update_opt_dict = vars(opt)
    update_opt_dict.update(dict_yaml)
    opt = argparse.Namespace(**update_opt_dict)

    wandb_logger = WandbLogger(project = opt.model_name,
            log_model = False,
            group=opt.name_ckp)

    dm = DataModule(hr_shape = opt.image_size,
            data_dir= opt.data_path,
            batch_size = opt.batch_size,
            num_workers=32)

    dp = DDPStrategy(process_group_backend='gloo',
                     find_unused_parameters=True)
    ckp_path = os.getcwd()+ "/saved_models/" + opt.model_name + "/" + opt.name_ckp + "/checkpoints/"
    checkpoint_callback = ModelCheckpoint(dirpath = ckp_path,
            save_last = True,
            save_top_k = -1
            )
    trainer = L.Trainer(
            accelerator = "auto",
            devices = opt.n_gpus,
            num_nodes = opt.n_nodes,
            max_epochs = opt.num_epochs,
            strategy='ddp_find_unused_parameters_true',
            #strategy=dp,
            logger = wandb_logger,
            callbacks = [checkpoint_callback],
            fast_dev_run=True
            ) # strategy flag when one model has not updating parameters
    if trainer.global_rank == 0:
        print('\033[93m WARNING: New working dir is %s \033[0m'%pwd_path)
        if checkpoint_callback.file_exists(ckp_path+'last.ckpt', trainer):
            print('\033[93m WARNING: Checkpoint loading... \033[0m')
    model = GAN(
            opt,
            channels=opt.input_channel,
            hr_shape=opt.image_size,
            batch_size = opt.batch_size,
            update_FE = opt.update_FE,
            arch_type = opt.arch_type
            )
    trainer.fit(model,
            dm, 
            ckpt_path = 'last'
            )
