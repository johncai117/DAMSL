import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from methods import damsl_v1
from methods import damsl_v1_proto
from methods import damsl_v2
from methods import damsl_v2_ss ##change back later
from methods import damsl_v2_ss_lab ##change back later
from methods import damsl_v2_gnn
from methods import damsl_v2_proto
from methods import gnnnet
from methods import gnn

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
from io_utils import model_dict, parse_args, get_resume_file, get_assigned_file
from datasets import miniImageNet_few_shot, DTD_few_shot, cifar_few_shot, caltech256_few_shot, CUB_few_shot
from utils import device

def train(base_loader, model, optimization, start_epoch, stop_epoch, params):  
    for _, param in model.named_parameters():
            param.requires_grad = True
    
    if "sbmtl" in params.method and params.start_epoch >= 401:
        for _, param in model.feature_baseline.named_parameters():
            param.requires_grad = False  

    if params.optimization == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.1, momentum = 0.9 )
    elif params.optimization == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    else:
       raise ValueError('Unknown optimization, please define by yourself')     
    model.train()

    if not params.fine_tune:
      for epoch in range(start_epoch,stop_epoch):
          if params.method == "gnnnet":
            model.train_loop2(epoch, base_loader,  optimizer ) 
          else:
            model.train_loop(epoch, base_loader,  optimizer )
          if not os.path.isdir(params.checkpoint_dir):
              os.makedirs(params.checkpoint_dir)

          if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
              outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
              torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    else:
      for epoch in range(start_epoch,stop_epoch):
          model.train()
          model.train_loop_finetune(epoch, base_loader,  optimizer ) 

          if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
              outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
              torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
         
            
     
    return model

if __name__=='__main__':
    print("HELLO")
    
    params = parse_args('train')
    print(params.method)
    if not params.start_epoch > 0:
      np.random.seed(10) #original was 10
    
    image_size = 224

    optimization = params.optimization

    if params.method in ['baseline'] :

        if params.dataset == "miniImageNet":
            #print('hi')
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)
            #print("bye")
            base_loader = datamgr.get_data_loader(aug = params.train_aug )
            params.num_classes = 64
            #print("loaded")
        elif params.dataset == "CUB":

            #base_file = configs.data_dir['CUB'] + 'base.json' 
            base_datamgr    = CUB_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader     = base_datamgr.get_data_loader(aug = True )
       
            params.num_classes = 200
        elif params.dataset == "CIFAR":
            base_datamgr    = cifar_few_shot.SimpleDataManager("CIFAR100", image_size, batch_size = 16)
            base_loader    = base_datamgr.get_data_loader( "base" , aug = True )
                
            params.num_classes = 100

        elif params.dataset == 'Caltech':
            base_datamgr  = caltech256_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader = base_datamgr.get_data_loader(aug = False )
            params.num_classes = 257

        elif params.dataset == "DTD":
            base_datamgr    = DTD_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader     = base_datamgr.get_data_loader( aug = True )
            params.num_classes = 47

        else:
           raise ValueError('Unknown dataset')
        
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)
        model           = BaselineTrain( model_dict[params.model], params.num_classes)

    elif params.method in ['sbmtl','maml','relationnet','protonet', 'gnnnet', 'metaoptnet', "damsl_v2_gnn", "sbmtl_proto"] or "damsl" in params.method:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        if params.method == "damsl_v2_ss":
            n_query = 15
        
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
    
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 

        if params.dataset == "miniImageNet":
            print("loading")
            datamgr            = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
            base_loader        = datamgr.get_data_loader(aug = params.train_aug)
            #datamgr         = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 64)
            #data_loader     = datamgr.get_data_loader(aug = False )
            
            print("BYE")

        else:
           raise ValueError('Unknown dataset')

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'protonet_damp':
            model           = protonet_damp.ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'relationnet':
            feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
            model           = RelationNet(feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method == 'maml':
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model           = MAML( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'metaoptnet':
            model           = MetaOptNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'gnnnet':
            model           = GnnNet( model_dict[params.model], **train_few_shot_params)
        elif params.method == 'damsl_v1':
            model           = damsl_v1.GnnNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'damsl_v1_proto':
            model           = damsl_v1_proto.GnnNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'damsl_v2':
            model           = damsl_v2.GnnNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'damsl_v2_ss':
            model           = damsl_v2_ss.GnnNet( model_dict[params.model], **train_few_shot_params )
            model.n_query = 15
        elif params.method == 'damsl_v2_ss_lab':
            model           = damsl_v2_ss_lab.GnnNet( model_dict[params.model], **train_few_shot_params )
            model.n_query = 15
        elif params.method == 'damsl_v2_gnn':
            model           = damsl_v2_gnn.GnnNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'damsl_v2_proto': ##remember to rename this
            model           = damsl_v2_proto.GnnNet( model_dict[params.model], **train_few_shot_params )
       
       
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    save_dir =  configs.save_dir
    print("WORKING")


    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, params.dataset, params.model, params.method)
    
    
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    
    if params.optimization != "SGD":
        params.checkpoint_dir += "_" + params.optimization

    print("Optimizer: ",params.optimization)

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.method == "sbmtl" and params.start_epoch == 400:
        params.checkpoint_dir = "logs_original_original/logs/checkpoints/miniImageNet/ResNet10_gnnnet_aug_5way_5shot/400.tar"

    print(params.checkpoint_dir)
  
    if params.start_epoch > 401:
    
        resume_file = get_assigned_file(params.checkpoint_dir, params.start_epoch -1)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            
            state = tmp['state']
            state_keys = list(state.keys())
            for _, key in enumerate(state_keys):
                if "feature2." in key:
                    state.pop(key)
                if "feature3." in key:
                    state.pop(key)
          
      
      
    if params.start_epoch == 401 and "damsl" in params.method:
        #model.load_state_dict(state)
        model.instantiate_baseline(params)
    elif params.start_epoch > 401 and "damsl" in params.method:

        model.instantiate_baseline(params)
        model.load_state_dict(state)
    elif params.start_epoch > 0:
        model.load_state_dict(state)



    model.cuda()


    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params)
