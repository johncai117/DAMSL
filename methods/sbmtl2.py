import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
from methods.gnn import GNN_nl
from torch.autograd import Variable
import backbone
import copy
import math
import random
import torch.nn.functional as F
import configs

from methods.baselinetrain import BaselineTrain
from io_utils import model_dict, parse_args, get_resume_file, get_assigned_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

class GnnNet(MetaTemplate):
  maml=False
  def __init__(self, model_func,  n_way, n_support):
    super(GnnNet, self).__init__(model_func, n_way, n_support)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()
    self.first = True

    # metric function
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, 64), nn.BatchNorm1d(64, track_running_stats=False)) 
    self.fc2 = nn.Sequential(nn.Linear(n_way, 32), nn.BatchNorm1d(32, track_running_stats=False)) 
    self.fc3 = nn.Sequential(nn.Linear(n_way, 32), nn.BatchNorm1d(32, track_running_stats=False)) 
    self.gnn = GNN_nl(64 + self.n_way, 32, self.n_way)
    self.method = 'GnnNet'

    ## batchnorm and classifier
    self.classifier = Classifier(self.feat_dim, self.n_way)
    self.batchnorm = nn.BatchNorm1d(5, track_running_stats=False)
        
    
    
    # number of layers to allow to adapt during fine-tuning
    self.num_FT_block = 2 ##default
    self.ft_epoch = 3

    self.num_FT_layers = -9

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way)

  def cuda(self):
    self.feature.to(device)
    self.fc.to(device)
    self.fc2.to(device)
    self.fc3.to(device)
    self.gnn.to(device)
    self.batchnorm.to(device)
    self.classifier.to(device)
    
    self.support_label = self.support_label.to(device)
    return self

  def instantiate_baseline(self, params):
    baseline_model  = BaselineTrain( backbone.ResNet10, 64)
    save_dir =  configs.save_dir
    self.params = copy.deepcopy(params)
    self.params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, "miniImageNet", "ResNet10", "baseline")
    if self.params.train_aug:
        self.params.checkpoint_dir += '_aug'
    
    resume_file2 = get_assigned_file(self.params.checkpoint_dir, 400)
    if resume_file2 is not None:
      tmp = torch.load(resume_file2)
      
      state = tmp['state']
      state_keys = list(state.keys())
      for _, key in enumerate(state_keys):
          if "feature2." in key:
              state.pop(key)
          if "feature3." in key:
              state.pop(key)
          elif "feature" in key:
              newkey = key.replace("feature.", "")
              state[newkey] = state.pop(key)
          else:
              state.pop(key)
          
            
    baseline_model.feature.load_state_dict(state)  
    self.feature_baseline = copy.deepcopy(baseline_model.feature)
    self.batchnorm2 = nn.BatchNorm1d(5, track_running_stats=False)
    self.fc_new = nn.Sequential(nn.Linear(10, 64), nn.BatchNorm1d(64, track_running_stats=False)) 
    self.fc_deep = nn.Sequential(nn.Linear(n_way, 32), nn.BatchNorm1d(32, track_running_stats=False), nn.Linear(32,32), nn.BatchNorm1d(32, track_running_stats=False), nn.Linear(32,32), nn.BatchNorm1d(32, track_running_stats=False)) ## deep NN
    del baseline_model
    self.batchnorm2.to(device)
    self.feature_baseline.to(device)
    self.fc_new.to(device)
    self.fc_deep.to(device)

  def instantiate_baseline2(self, params):
    baseline_model  = BaselineTrain( backbone.ResNet10, 64)
    save_dir =  configs.save_dir
    self.params = copy.deepcopy(params)
    self.params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, "miniImageNet", "ResNet10", "baseline")
    if self.params.train_aug:
        self.params.checkpoint_dir += '_aug'
    
    resume_file2 = get_assigned_file(self.params.checkpoint_dir, 400)
    if resume_file2 is not None:
      tmp = torch.load(resume_file2)
      
      state = tmp['state']
      state_keys = list(state.keys())
      for _, key in enumerate(state_keys):
          if "feature2." in key:
              state.pop(key)
          if "feature3." in key:
              state.pop(key)
          elif "feature" in key:
              newkey = key.replace("feature.", "")
              state[newkey] = state.pop(key)
          else:
              state.pop(key)
          
            
    baseline_model.feature.load_state_dict(state)  
    self.feature_baseline = copy.deepcopy(baseline_model.feature)
    self.batchnorm2 = nn.BatchNorm1d(5, track_running_stats=False)
    del baseline_model
    self.batchnorm2.to(device)

  def set_forward(self,x,is_feature=False):
    x = x.to(device)

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + 15)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      x = x.view(-1, *x.size()[2:])
      z = self.fc(self.feature(x))
      z = z.view(self.n_way, -1, z.size(1))

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn(z_stack)
    return scores

  def train_loop_finetune(self, epoch, train_loader, optimizer ):
        ### load baseline model

        print_freq = 10
        avg_loss=0
        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss_finetune( x )
            loss.backward()
            optimizer.step()
            self.MAML_update()
            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
  
  def MAML_update(self):
    names = []
    for name, param in self.feature.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    names_sub = names[:self.num_FT_layers] ### last Resnet block can adapt
    if not self.first:
      for (name, param), (name1, param1), (name2, param2) in zip(self.feature.named_parameters(), self.feature2.named_parameters(), self.feature3.named_parameters()):
        if name not in names_sub:
          dat_change = param2.data - param1.data ### Y - X
          new_dat = param.data - dat_change ### (Y- V) - (Y-X) = X-V
          param.data.copy_(new_dat)
      for (name, param), (name1, param1), (name2, param2) in zip(self.classifier.named_parameters(), self.classifier2.named_parameters(), self.classifier3.named_parameters()):
        if name not in names_sub:
          dat_change = param2.data - param1.data ### Y - X
          new_dat = param.data - dat_change ### (Y- V) - (Y-X) = X-V
          param.data.copy_(new_dat)

  def set_forward_finetune(self,x,is_feature=False, linear = False):
    x = x.to(device)
    # get feature using encoder
    support_size = self.n_way * self.n_support 
    batch_size = 8

    for name, param  in self.feature.named_parameters():
      param.requires_grad = True

    x_var = Variable(x)
      
    y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).to(device) # (25,)
    
    x_b_i = x_var[:, self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) 
    x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:])
    x_inn = x_var.view(self.n_way* (self.n_support + self.n_query), *x.size()[2:]).to(device)
    ### copy over feature and classifier of the main model
    feat_network = copy.deepcopy(self.feature)
    classifier = copy.deepcopy(self.classifier)


    #### copy baseline feature and instantiate classifer
    baseline_feat = copy.deepcopy(self.feature_baseline)
    classifier_baseline = Classifier(self.feature_baseline.final_feat_dim, self.n_way) ##instantiate classifier
    classifier_baseline.to(device)

    ### select layers to freeze
    names = []
    for name, param in feat_network.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    names_sub = names[:-9] ### last Resnet block can adapt

    for name, param in feat_network.named_parameters():
      if name in names_sub:
        param.requires_grad = False   

    ### loss function and oiptimizer
    delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, feat_network.parameters()), lr = 0.01)
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr = 0.01)

    ### freeze layers of baseline feat
    names_b = []

    for _, param in baseline_feat.named_parameters():
            param.requires_grad = True

    for name, param in baseline_feat.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    names_sub_b = names_b[:-9] ### last Resnet block can adapt

    for name, param in baseline_feat.named_parameters():
      if name in names_sub_b:
        param.requires_grad = False   

    delta_opt_b = torch.optim.Adam(filter(lambda p: p.requires_grad, baseline_feat.parameters()), lr = 0.01)
    classifier_opt_b = torch.optim.Adam(classifier_baseline.parameters(), lr = 0.01)


    loss_fn = nn.CrossEntropyLoss().to(device) 
    if self.params.n_shot <= 20:
      total_epoch = 15
    else:
      total_epoch = 5

    classifier.train()
    feat_network.train()

    classifier.to(device)
    feat_network.to(device)

    for epoch in range(total_epoch):
          rand_id = np.random.permutation(support_size)

          for j in range(0, support_size, batch_size):
              classifier_opt.zero_grad()
              
              delta_opt.zero_grad()

              #####################################
              selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).to(device)
              
              z_batch = x_a_i[selected_id]
              y_batch = y_a_i[selected_id] 
              #####################################

              output = feat_network(z_batch)
              score  = classifier(output)
              loss = loss_fn(score, y_batch)
              #grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)

              #####################################
              loss.backward() ### think about how to compute gradients and achieve a good initialization

              classifier_opt.step()
              delta_opt.step()


    for epoch in range(total_epoch):
          rand_id = np.random.permutation(support_size)

          for j in range(0, support_size, batch_size):
              classifier_opt_b.zero_grad()
              
              delta_opt_b.zero_grad()

              #####################################
              selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).to(device)
              
              z_batch = x_a_i[selected_id]
              y_batch = y_a_i[selected_id] 
              #####################################

              output = baseline_feat(z_batch)
              score  = classifier_baseline(output)
              loss_b = loss_fn(score, y_batch)
              #grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)

              #####################################
              loss_b.backward() ### think about how to compute gradients and achieve a good initialization

              classifier_opt_b.step()
              delta_opt_b.step()
    
    if self.first == True:
      self.first = False
    self.feature2 = copy.deepcopy(self.feature)
    self.feature3 = copy.deepcopy(feat_network) ## before the new state_dict is copied over
    self.feature.load_state_dict(feat_network.state_dict()) 

    self.classifier2 = copy.deepcopy(self.classifier)
    self.classifier3 = copy.deepcopy(classifier) ## before the new state_dict is copied over
    self.classifier.load_state_dict(classifier.state_dict()) 
    
    for name, param  in self.feature.named_parameters():
        param.requires_grad = True    

    output_all = self.feature(x_inn.to(device)).view(self.n_way, self.n_support + self.n_query, -1).detach()
    final = self.classifier(output_all)
    final = torch.transpose(self.batchnorm(torch.transpose(final, 1,2)),1,2).contiguous()

    ### load baseline feature
    output_all_b = baseline_feat(x_inn.to(device)).view(self.n_way, self.n_support + self.n_query, -1).detach()
    final_b = classifier_baseline(output_all_b).detach()
    final_b = torch.transpose(self.batchnorm2(torch.transpose(final_b, 1,2)),1,2).contiguous()
    
    ### feed into fc and gnn

    assert(final.size(1) == self.n_support + 16) ##16 query samples in each batch

    z = self.fc_deep(final.view(-1, *final.size()[2:])) ## use fc deep for deep embedding network
    z = z.view(self.n_way, -1, z.size(1))

    z_b = self.fc_deep(final_b.view(-1, *final_b.size()[2:]))
    z_b = z_b.view(self.n_way, -1, z_b.size(1))

    z = torch.cat([z, z_b], dim = 2)
    
    z_support = z[:,:self.n_support,:].contiguous()
    z_query = z[:,self.n_support:,:].contiguous()
    
    z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
    z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

    dists = euclidean_dist(z_query, z_proto)
    scores = -dists

    #z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    
    #assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    
    #scores = self.forward_gnn(z_stack)
    return scores

  def forward_gnn(self, zs):
    # gnn inp: n_q * n_way(n_s + 1) * f
    nodes = torch.cat([torch.cat([z, self.support_label.to(device)], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes)
    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  def set_forward_loss(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.to(device)
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return loss

  def set_forward_loss_finetune(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.to(device)
    scores = self.set_forward_finetune(x)
    loss = self.loss_fn(scores, y_query)
    return loss
