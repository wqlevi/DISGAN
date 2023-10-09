import torch
import numpy as np
import nibabel as nb
import argparse
import glob
import os



def img2tensor(filename,device,hr_shape,scale = 1):
    img = np.array(nb.load(filename).dataobj)
    img = np.expand_dims(img,axis=0)
    img = np.expand_dims(img,axis=0)
    tensor = torch.from_numpy(img).float()
    tensor_lr = torch.nn.functional.interpolate(tensor,int(scale*hr_shape/2)).to(device)
    return tensor,tensor_lr

#profile
def img2lr(img,device,hr_shape,scale = 1):
    tensor = torch.from_numpy(img[None,None]).float().pin_memory()
    #tensor = torch.from_numpy(img[None,None]).float()
    tensor_lr = torch.nn.functional.interpolate(tensor,int(hr_shape*scale/2)).to(device)
    return tensor,tensor_lr

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad=False
        
def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad=True
        
        
def load_pretrained_ln(model, pretrain_path, key_dict:str='state_dict'):
    ckp = torch.load(pretrain_path)
    ckp_dict = ckp['state_dict']
    model_dict = model.state_dict()
    ckp_dict_new = {k.replace("generator.",""):v for k,v in ckp_dict.items() if "generator." in k}
    ckp_dict_new = {k.replace("module.",""):v for k,v in ckp_dict_new.items() if k.replace("module.","") in model_dict.keys()}
    assert ckp_dict_new == model_dict, "Loaded state_dict still unmatch!"
    model_dict.update(ckp_dict_new)
    model.load(model_dict)
    return model

def load_pretrained(model,pretrain_path,key_dict:str='state_dict', **kwargs):
    # [BUG] Gnet not correctly loaded('module.' already exist in Gnet.state_dict())
    '''
    Parameters
    ----------
        model: model of network
        pretrain_path : str | path to store checkpoint
        key_dict : str | key string of the model(e.g. FE_state_dict, etc.)
    '''
    use_ln=False
    device = "cuda:0"
    model_name = 'generator'
    if 'use_ln' in kwargs:
        use_ln=kwargs['use_ln']
    if 'device' in kwargs:
        device = kwargs['device']
    if 'model_name' in kwargs:
        model_name = kwargs['model_name']
    pretrains = torch.load(pretrain_path, map_location=torch.device(device))
    net_dict = model.state_dict()
    if not use_ln:
        pretrain_dict = {k.replace("module.",""): v for k, v in pretrains['%s'%key_dict].items() if k.replace("module.","") in net_dict.keys()}
        #pretrain_dict = pretrains # legacy for C11 model
    else:
        pretrain_dict = {k.replace(f"{model_name}.",""):v for k,v in pretrains['%s'%key_dict].items() if f"{model_name}." in k}
        pretrain_dict = {k.replace("module.",""):v for k,v in pretrain_dict.items() if k.replace("module.","") in net_dict.keys()}
    assert pretrain_dict.keys() == net_dict.keys(), "Loaded state_dict keys still unmatch!"

    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    return model
