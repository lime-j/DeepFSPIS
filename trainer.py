import torch
import torch.nn as nn
from data.datasets import *
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import save_image
import torch.nn.functional as F
from typing import Tuple
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter

def dir_edge_calc(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Calculate image gradients horizontally and vertically, then add them up.
    Args :
        images : torch.Tensor, input image in [N, C, H, W] shape
    Return :
        Edge maps, torch.Tensor
    """

    return F.pad((image[:, :, :, :-1] - image[:, :, :, 1:]), (1, 0, 0, 0)), \
           F.pad((image[:, :, :-1, :] - image[:, :, 1:, :]), (0, 0, 1, 0))


def edge_calc(image: torch.Tensor) -> torch.Tensor:
    r"""
    Calculate image gradients horizontally and vertically, then add them up.
    Args :
        images : torch.Tensor, input image in [N, C, H, W] shape
    Return :
        Edge maps, torch.Tensor
    """

    edg_x, edg_y = F.pad(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), (1, 0, 0, 0)), \
                   F.pad(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), (0, 0, 1, 0))
    return (edg_x + edg_y) / 2



class Trainer:
    def __init__(
        self, 
        model,
        optim,
        scheduler,
        folder,
        train_loader,
        val_loader,
        train_batch_size,
        checkpoint_interval,
        sample_interval=500
        ):
    
        self.model = model
        self.train_batch_size = train_batch_size
        self.optim = optim
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.current_epoch = 0
        self.checkpoint_interval = checkpoint_interval
        self.folder = folder 
        self.sample_interval = sample_interval
        os.makedirs(self.folder, exist_ok=True)
        self.writer = SummaryWriter(self.folder)
        
    def load_checkpoint(self, path) :
        param = torch.load(path)
        if 'model' in param.keys(): 
            self.model.load_state_dict(param['model'])
        elif 'icnn' in param.keys():
            self.model.load_state_dict(param['icnn'])
        self.optim.load_state_dict(param['optimizer'])
        if 'scheduler' in param.keys(): 
            self.scheduler.load_state_dict(param['scheduler'])
        if 'epoch' in param.keys():
            self.current_epoch = param['epoch']

    def save_checkpoint(self, key):
        checkpoint = {"epoch" : self.current_epoch, 
                        "model" : self.model.state_dict(),
                        "optimizer" : self.optim.state_dict(),
                        "scheduler" : self.scheduler.state_dict()}#, "scaler" : scaler.state_dict()}
        torch.save(checkpoint, f'{self.folder}/train_{key}_{self.current_epoch}.pth')

    def show_single_image(self, path, name, image):
        ndarr = image[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(f"{self.folder}/{path}/{name}.png")

    def show_images(self, itr, dic) :
        for key, value in dic : 
            self.show_single_image(f'{self.current_epoch}_{iter}', key, value)
    
    def write(self, itr, dic, typ='image') :
        for key, value in dic :
            if typ == 'image' :
                self.writer.add_images(key, value, itr)
            elif typ == 'metric' :
                self.writer.add_scalar(key, value, itr)

    def train(self) : pass
    
    def infer(self) : pass

class SmootherTrainer(Trainer):
    def __init__(self,
        model,
        adjuster_model,
        adjuster_path,
        loss,
        optim,
        scheduler,
        train_dataset_path,
        val_dataset_path,
        train_batch_size = 1,
        num_threads = 8,
        max_epoch = 20,
        checkpoint_interval = 1,
        result_dir = "./results"
        ):
        super().__init__(
            model, optim, scheduler, result_dir,
            DataLoader(ImageDataset(train_dataset_path, 
                        transforms_=[#transforms.RandomResizedCrop((512, 512)),
                                     transforms.ToTensor()],
                        unaligned=True, size=10000),
                        batch_size=train_batch_size,
                        shuffle=True, 
                        num_workers=num_threads),
            DataLoader(ImageDataset(val_dataset_path, 
                        transforms_=[transforms.ToTensor()],
                        unaligned=True, size=10), 
                        batch_size=1,
                        shuffle=False, 
                        num_workers=num_threads),
            train_batch_size, checkpoint_interval)
        self.max_epoch = max_epoch
        self.adjuster = adjuster_model
        self.adjuster.load_state_dict(torch.load(adjuster_path)["icnn"])
        self.loss = loss



    def train(self):
        self.model.train()
        for epoch in list(range(self.current_epoch, self.max_epoch)):
            print(f"Epoch : {epoch}")
            self.current_epoch = epoch
            for i, batch in enumerate(tqdm(self.train_loader)):
                zero = False
                lamb = random.uniform(0, 1)
                lamb = lamb ** (epoch // 4 + 1)
                input_image = batch['img'].to("cuda:0")

                with torch.no_grad():
                    mask_images = self.adjuster(input_image, lamb, inference=True)
                
                generated_images0, generated_images1, generated_images2 = self.model(input_image, mask_images, inference=False)


                loss0 = self.loss(generated_images0,input_image,mask_images, zero, lamb, 0)
                loss1 = self.loss(generated_images1,input_image,mask_images, zero, lamb, 1)
                loss2 = self.loss(generated_images2,input_image,mask_images, zero, lamb, 2)
                loss = loss0 + loss1 + loss2

                loss.backward()
                self.optim.step()

                self.optim.zero_grad()
                if i % self.sample_interval == 0:
                    mask = torch.cat([mask_images.data, mask_images.data, mask_images.data], dim=1)

                    img_sample = torch.cat((input_image.data, mask, generated_images0.data, generated_images1.data, generated_images2.data), 0)
                    #save_image(img_sample4, 'result/%s_4.png' % batches_done, nrow=5, normalize=True)
                    save_image(img_sample, f'{self.folder}/{i}_{lamb}.png', nrow=4, normalize=False)

            # Update learning rates
            self.scheduler.step()


            if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
                self.save_checkpoint("smoother")

class AdjusterTrainer(Trainer) :
    def __init__(self,
        model,
        loss,
        optim,
        scheduler,
        train_dir,
        edge_dir,
        train_batch_size = 1,
        num_threads = 8,
        max_epoch = 20,
        checkpoint_interval = 5000,
        result_dir = "./results"
        ):
        super().__init__(
            model, optim, scheduler, result_dir,
            DataLoader(COCOHIPeDataset(origin_dir=train_dir,
                        edge_dir=edge_dir, syn_dir=None, 
                        data_len=10000 - 4),
                        batch_size=train_batch_size,
                        shuffle=True, num_workers=num_threads,
                        pin_memory=True),
            None, # we don't perform test on adjuster, 
            train_batch_size, checkpoint_interval)
        self.max_epoch = max_epoch
        self.loss = loss
        self.calib = lambda x : torch.sqrt(x)
    
    def forward(self, input, lam):

        output_edge = self.model(input, lam=lam)

        return output_edge    

    def backward(self, 
        output, masks, edges, gradient,
        gam=0, lap=False, one=False, zero=False, id=None, ref=False) :

        loss_edge_smooth = None
        loss_dice = None
        loss_edge_fidelity = 0
        edge_target = None

     
        if not ref:
            if one:
                loss_edge_fidelity = self.loss['t_fidelity'](output, gradient) * 4 * torch.exp(1 - gam)
                loss_dice = self.loss['t_dice'](output, gradient)
                edge_target = gradient
            elif zero:
                loss_edge_fidelity = self.loss['t_fidelity'](output, masks[-1]) * 4 * torch.exp(1 - gam)
                loss_dice = self.loss['t_dice'](output, masks[-1])
                edge_target = edges[-1]            
            elif id == 1:
                loss_edge_fidelity = self.loss['t_fidelity'](output, masks[-1]) * 4 * torch.exp(1 - gam)
            else : self.loss_edge_fidelity = 0
        
        else :
            loss_edge_fidelity = self.loss['t_fidelity'](output, masks[id]) * 4 * torch.exp(1 - gam)
            loss_dice = self.loss['t_dice'](output, masks[id])
            edge_target = edges[id]
        
        if edge_target is None: edge_target = gradient

        loss_edge_reduction = self.loss['t_reduction'](output, gradient, gam, self.calib)\
                                                     * (0.1 if not ref and id >= 2 else 0)
        loss_edge_consistency = self.loss['t_consistency'](output, edge_target,
                                                            gam=gam, lonly=not ref) * 0.4

        loss_G = loss_edge_consistency + loss_edge_reduction + loss_edge_fidelity +\
                 (0 if loss_dice is None else loss_dice * 0.002)
        loss_G = loss_G.mean() #.view(-1)
        ret = loss_G.item()
        #if self.loss_render_smooth is not None:
        #    self.loss_G += self.loss_render_smooth
        loss_G.backward()
        return ret

    def train(self):
        self.model.train()
        iter_cnt = -1
        for epoch in list(range(self.current_epoch, self.max_epoch)):
            self.current_epoch = epoch
            for i, data in enumerate(tqdm(self.train_loader)):
                iter_cnt += 1
                input, data_name = data['org_input'], data['fn']
                edge, mask = data["edge"], data["mask"]
        
                input = input.to("cuda")
                edges = [item.to("cuda") for item in edge]
                mask = [item.to("cuda") for item in mask]

                grad = torch.clip(torch.max(edge_calc(input), dim=-3, keepdim=True)[0], 0, 1)
                #vanilla_grad = grad  # torch.clip(torch.max(edge_calc(input), dim=-3, keepdim=True)[0], 0, 1) #grad.clone().detach()#.requires_grad_(True)
                #grad_h, grad_w = dir_edge_calc(input)
                zero = torch.zeros(1, 1, input.shape[-2], input.shape[-1], device='cuda')
                gradient = grad  # .clone().detach()
                gradient[gradient > 0.005] = 1  # TODO: fine-tune this threshold
                gradient[gradient <= 0.005] = 0
                
                lam_list = list(map(lambda x: torch.tensor(x).to("cuda"),
                            [0, 0.2, 0.8, 1])) 
                
                calib = lambda x: torch.sqrt(x)
                ref_lams = [calib(torch.sum(item, dim=(-3, -2, -1)) / torch.sum(gradient, dim=(-3, -2, -1))) for item in edges]
                ref_outs, outs =  [[] for i in range(len(ref_lams))], [[] for i in range(len(lam_list))]
                total_loss = 0
                for idx, lam in enumerate(lam_list):
                    output = self.forward(input, lam.view(-1, 1, 1, 1))
                    loss = self.backward(output, mask, edges, gradient, 
                                        lam.view(-1, 1, 1, 1), ref=False, 
                                        one=True if idx == len(lam_list) - 1 else False,
                                        zero=True if idx == 0 else False, id=idx)
                    total_loss += loss 
                for idx, ref_lam in enumerate(ref_lams):
                    output = self.forward(input, ref_lam.view(-1, 1, 1, 1))
                    loss = self.backward(output, mask, edges, gradient,
                                         ref_lam.view(-1, 1, 1, 1), ref=True, id=idx)
                    total_loss += loss
                self.optim.step()
                self.optim.zero_grad()
                self.write(iter_cnt, {"train_loss" : total_loss}, typ='metric')
                lam_ratios = [torch.mean(item).item() for item in ref_lams]
            
            if self.current_epoch % self.checkpoint_interval == 0:
                self.save_checkpoint('adjuster')
            self.scheduler.step()
