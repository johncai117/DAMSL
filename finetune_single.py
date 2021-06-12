import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations
import copy
import backbone as backbone
import configs
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.gnnnet import GnnNet
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods import damsl_v1
from methods import damsl_single
from methods import damsl_v1_proto
from methods import damsl_v2
from methods import damsl_v2_gnn
from methods import damsl_v2_proto
from methods import damsl_v2_ss_lab
from methods import damsl_v2_ss
from methods.protonet import euclidean_dist
from self_supervised_label import *

#configs.save_dir = 'logs_final_train' ##override
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, miniImageNet_few_shot, DTD_few_shot, CUB_few_shot, cifar_few_shot, caltech256_few_shot, cars_few_shot, plantae_few_shot, places_few_shot

#### re-factor code below and re-implement it







######################
if __name__=='__main__':
  np.random.seed(10)
  params = parse_args('train')

  ##################################################################
  image_size = 224
  iter_num = 600
  n_way  = 5
  pretrained_dataset = "miniImageNet"
  ds = False

  n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
  few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 



  if params.method == "gnnnet": ## damsl_v1_gnn (this is the version)
      model           = GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v1':
        model           = damsl_v1.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v1_proto':
        model           = damsl_v1_proto.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v2':
        model           = damsl_v2.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v2_gnn':
        model           = damsl_v2_gnn.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v2_proto': ## proper name damsl_v2_proto
        model           = damsl_v2_proto.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v2_ss': ## proper name damsl_v2_proto
        model           = damsl_v2_ss.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'damsl_v2_ss_lab': ## proper name damsl_v2_proto
        model           = damsl_v2_ss_lab.GnnNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
  elif params.method == 'relationnet':
        feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
  elif params.method == "baseline":
        checkpoint_dir_b = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, pretrained_dataset, params.model, "baseline")
        if params.train_aug:
            checkpoint_dir_b += '_aug'

        if params.save_iter != -1:
            modelfile_b   = get_assigned_file(checkpoint_dir_b, 400)
        elif params.method in ['baseline', 'baseline++'] :
            modelfile_b   = get_resume_file(checkpoint_dir_b)
        else:
            modelfile_b  = get_best_file(checkpoint_dir_b)
        
        tmp_b = torch.load(modelfile_b)
        state_b = tmp_b['state']
  
  elif params.method == "all":
        
        model_2           = GnnNet( model_dict[params.model], **few_shot_params )
        checkpoint_dir2 = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, 'miniImageNet', params.model, "gnnnet")
        
        

        checkpoint_dir_b = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, pretrained_dataset, params.model, "baseline")
        if params.train_aug:
            checkpoint_dir_b += '_aug'

        if params.save_iter != -1:
            modelfile_b   = get_assigned_file(checkpoint_dir_b, 400)
        elif params.method in ['baseline', 'baseline++'] :
            modelfile_b   = get_resume_file(checkpoint_dir_b)
        else:
            modelfile_b  = get_best_file(checkpoint_dir_b)
        
        tmp_b = torch.load(modelfile_b)
        state_b = tmp_b['state']

  


  if params.method != "all" and params.method != "baseline":
      checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, 'miniImageNet', params.model, params.method)
      if params.train_aug:
          checkpoint_dir += '_aug'

      if not params.method in ['baseline'] :
          if params.optimization != "Adam":
            checkpoint_dir += "_" + params.optimization
          checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
      print(checkpoint_dir)
      if not params.method in ['baseline'] : 
          if params.save_iter != -1:
              modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
          else:
              modelfile   = get_best_file(checkpoint_dir)
          classifier_found = False
          if modelfile is not None:
              if torch.cuda.device_count() == 1:
                map_loc = torch.device('cuda:0')
              else:
                map_loc = device
              tmp = torch.load(modelfile, map_location = map_loc)
              state = tmp['state']
              state_keys = list(state.keys())
              for _, key in enumerate(state_keys):
                  if "feature2." in key:
                      state.pop(key)
                  if "feature3." in key:
                      state.pop(key)
                  if "classifier2." in key:
                      classifier_found = True
                      state.pop(key)
                  if "classifier3." in key:
                      classifier_found = True
                      state.pop(key)
              model.classifier = Classifier(model.feat_dim, n_way)
              model.batchnorm = nn.BatchNorm1d(5, track_running_stats=False)
              if "damsl" in params.method:
                  model.instantiate_baseline(params)
              model.load_state_dict(state)
              model.to(device)
     
  elif params.method == "all":
        
                
        if params.method == "all":
            checkpoint_dir2 += '_aug'
            if not params.method in ['baseline'] :
              checkpoint_dir2 += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
            if not params.method in ['baseline'] : 
              modelfile2   = get_assigned_file(checkpoint_dir2,600)
              
              modelfile2_o   = get_assigned_file(checkpoint_dir2,600)
                
                
                
              

              if modelfile2 is not None:
                tmp2 = torch.load(modelfile2)
                tmp2_o  = torch.load(modelfile2_o)
                state2 = tmp2['state']
                state2_o = tmp2_o['state']
                state_keys = list(state2.keys())
                for _, key in enumerate(state_keys):
                    if "feature2." in key:
                        state2.pop(key)
                    if "feature3." in key:
                        state2.pop(key)
                model_2.load_state_dict(state2)
                
                
                
                
                del tmp2
                del modelfile2

              
                
  
              

  freeze_backbone = params.freeze_backbone
  ##################################################################
  pretrained_dataset = "miniImageNet"

  ### Loading datasets below
  
  if params.test_dataset == "ISIC":
    print ("Loading ISIC")
    datamgr             =  ISIC_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)

  
  elif params.test_dataset == "EuroSAT":
    print ("Loading EuroSAT")
    datamgr             =  EuroSAT_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)
  
  
  elif params.test_dataset == "CropDisease":
    print ("Loading CropDisease")
    datamgr             =  CropDisease_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)


  elif params.test_dataset == "ChestX":
    print ("Loading ChestX")
    datamgr             =  Chest_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???


  elif params.test_dataset == "DTD":
    print ("Loading DTD")
    datamgr             =  DTD_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???


  elif params.test_dataset == "CUB":
    print ("Loading CUB")
    datamgr             =  CUB_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???
  

  elif params.test_dataset == "Caltech":
    print ("Loading Caltech")
    datamgr             =  caltech256_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???


  elif params.test_dataset == "Cifar":
    print ("Loading Cifar")
    datamgr             =  cifar_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???
  

  elif params.test_dataset == "Cars":
    print ("Loading Cars")
    datamgr             =  cars_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???
  

  elif params.test_dataset == "Places":
    print ("Loading Places")
    datamgr             =  places_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???


  elif params.test_dataset == "Plantae":
    print ("Loading Plantae")
    datamgr             =  plantae_few_shot.SetDataManager2(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(num_aug = params.gen_examples)  ### what if aug is true???
  ## uncomment code below to see if code is same across loaders
  

  #########################################################################
  
  acc_all = []
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  print (freeze_backbone)
  


  # replace finetine() with your own method
  iter_num = iter_num
  print(params.n_shot)
  #print(iter_num)
  #print(len(novel_loader))
  print(params.ablation)
  
  for idx, (elem) in enumerate(novel_loader):
      leng = len(elem)
      
      
      ## uncomment below assertion to check that same images are shown - the same image is selected despite using two different loaders
      ## as we set the seed to be fixed
      ## for more checking, you can save the images and print them to see if they seem like crops / data augmentation of the underlying data
      assert(torch.all(torch.eq(elem[0][0] , elem[1][0])) )
      _, y = elem[0]
      
      liz_x = [x for (x,y) in elem]
   
      #for i in range(leng):
        #if i >= 1:
          #assert(torch.all(torch.eq(elem[i][1] , elem[i-1][1])) ) ##assertion check
      
  
      if params.method == "relationnet":
        scores = nofinetune(liz_x[0], y, model, state, flatten = False, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
      elif params.method == "baseline":
        scores = finetune_linear(liz_x, y, state_in = state_b, linear = True, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
      elif params.method in ["gnnnet"] or "damsl" in params.method:
        scores = finetune_classify(liz_x,y, model, state, ds = ds, save_it = params.save_iter, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)

      n_way = 5
      n_query = 15

      y_query = np.repeat(range(n_way ), n_query )

      topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
      topk_ind = topk_labels.cpu().numpy()
      
      top1_correct = np.sum(topk_ind[:,0] == y_query)
      correct_this, count_this = float(top1_correct), len(y_query)
      acc_all.append((correct_this/ count_this *100))
      if idx % 1 == 0:
        print(idx)
        print(correct_this/ count_this *100)
        #print(sum(acc_all) / len(acc_all))
  
      ###############################################################################################

  acc_all  = np.asarray(acc_all)
  acc_mean = np.mean(acc_all)
  acc_std  = np.std(acc_all)
  print(params.test_dataset)
  
  print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
  print(params.model)
