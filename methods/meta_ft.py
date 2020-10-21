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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Meta_FT(MetaTemplate):
  def __init__(self, model_func,  n_way, n_support):
    super(Meta_FT, self).__init__(model_func, n_way, n_support)

    # loss function
    self.loss_fn = nn.CrossEntropyLoss()
    self.first = True

    # metric function
    self.classifier = nn.Linear(self.feat_dim, self.n_way)
    self.method = 'Meta_FT'
    
    # number of layers to allow to adapt during fine-tuning
    self.num_FT_block = 2 ##default
    self.ft_epoch = 3

    if self.num_FT_block % 2 == 0:
      self.num_FT_layers = (-9 * math.floor(self.num_FT_block / 2))
    else:
      self.num_FT_layers = (-9 * math.floor(self.num_FT_block / 2)) - 6

  def cuda(self):
    self.feature.cuda()
    self.classifier.cuda()
    return self

  def set_forward(self,x,is_feature=False):
    x    = Variable(x.cuda())
    out  = self.feature.forward(x)
    scores  = self.classifier.forward(out)
    return scores
  
  def train_loop_finetune(self, epoch, train_loader, optimizer ):
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
  def train_loop_finetune_ep(self, epoch, train_loader, optimizer ):
      print_freq = 10

      avg_loss=0
      for i, elem in enumerate(train_loader):
          liz_x = [x for (x,_) in elem]    
          optimizer.zero_grad()
          loss = self.set_forward_loss_finetune_ep( liz_x)
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
        dat_change = param2.data - param1.data ### Y - X
        new_dat = param.data - dat_change ### (Y- V) - (Y-X) = X-V
        param.data.copy_(new_dat)


  def set_forward_finetune(self,x,is_feature=False, linear = False):
    x = x.to(device)
    # get feature using encoder
    batch_size = 32
    support_size = self.n_way * self.n_support 

    for name, param  in self.feature.named_parameters():
      param.requires_grad = True

    x_var = Variable(x)
      
    y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() # (25,)
    
    x_b_i = x_var[:, self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) 
    x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) # (25, 3, 224, 224)
    feat_network = copy.deepcopy(self.feature)
    classifier = copy.deepcopy(self.classifier)
    delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, feat_network.parameters()), lr = 0.01)
    loss_fn = nn.CrossEntropyLoss().cuda() ##change this code up ## dorop n way
    classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 0.01) ##try it with weight_decay
    
    names = []
    for name, param in feat_network.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    assert self.num_FT_block <= 9, "cannot have more than 9 blocks unfrozen during training"
    
    names_sub = names[:self.num_FT_layers] ### last Resnet block can adapt

    for name, param in feat_network.named_parameters():
      if name in names_sub:
        #print(name)
        param.requires_grad = False    
  
    total_epoch = 15

    classifier.train()
    feat_network.train()

    classifier.cuda()
    feat_network.cuda()

    for epoch in range(total_epoch):
          rand_id = np.random.permutation(support_size)

          for j in range(0, support_size, batch_size):
              classifier_opt.zero_grad()
              
              delta_opt.zero_grad()

              #####################################
              selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
              
              z_batch = x_a_i[selected_id]
              y_batch = y_a_i[selected_id] 
              #####################################

              output = feat_network(z_batch)
              scores  = classifier(output)
              loss = loss_fn(scores, y_batch)
              #grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)

              #####################################
              loss.backward() ### think about how to compute gradients and achieve a good initialization

              classifier_opt.step()
              delta_opt.step()
    
    if self.first == True:
      self.first = False
    self.feature2 = copy.deepcopy(self.feature)
    self.feature3 = copy.deepcopy(feat_network) ## before the new state_dict is copied over
    self.feature.load_state_dict(feat_network.state_dict())

    self.classifier2 = copy.deepcopy(self.classifier)
    self.classifier3 = copy.deepcopy(classifier)
    self.classifier.load_state_dict(classifier.state_dict())

    
    for name, param  in self.feature.named_parameters():
        param.requires_grad = True    

    #output_support = self.feature(x_a_i.cuda()).view(self.n_way, self.n_support, -1)
    output_query = self.feature(x_b_i.cuda())
    scores = self.classifier(output_query)
    
    return scores

  def set_forward_finetune_ep(self,liz_x,is_feature=False, n_query = 16):
    ### introduce a random int generator here to randomly take an augmented sample. This is to allow for better regularization in the fine tuning process.
    random_int = random.randint(0, len(liz_x)-1) 
    x = liz_x[random_int] ### non-changed one
    self.n_query = n_query
    x = x.to(device)
    # get feature using encoder
    batch_size = 16
    support_size = self.n_way * self.n_support 

    for name, param  in self.feature.named_parameters():
      param.requires_grad = True

    x_var = Variable(x)
      
    y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() # (25,)
    x_b_i = x_var[:, self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) 
    x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) # (25, 3, 224, 224)
    x_inn = x_var.view(self.n_way* (self.n_support + self.n_query), *x.size()[2:])
    
    ### to load all the changed examples

    x_a_i_new = torch.cat((x_a_i, x_a_i), dim = 0) ##oversample the first one
    y_a_i = torch.cat((y_a_i, y_a_i), dim = 0)
    for i, x_aug in enumerate(liz_x):
      if not i == random_int:
        x_aug = x_aug.to(device)
        x_aug = Variable(x_aug)
        x_a_aug = x_aug[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:])
        y_a_aug = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).to(device)
        x_a_i_new = torch.cat((x_a_i_new, x_a_aug), dim = 0)
        y_a_i = torch.cat((y_a_i, y_a_aug.to(device)), dim = 0)
    

    feat_network = copy.deepcopy(self.feature)
    classifier = copy.deepcopy(self.classifier)
    
    
    names = []
    for name, param in feat_network.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    assert self.num_FT_block <= 9, "cannot have more than 9 blocks unfrozen during training"
    
    names_sub = names[:self.num_FT_layers] ### last Resnet block can adapt

    for name, param in feat_network.named_parameters():
      if name in names_sub:
        #print(name)
        param.requires_grad = False    

    delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, feat_network.parameters()), lr = 0.01)
    loss_fn = nn.CrossEntropyLoss().cuda() ##change this code up ## dorop n way
    classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 0.01) ##try it with weight_decay
  
    total_epoch = self.ft_epoch 

    classifier.train()
    feat_network.train()

    classifier.cuda()
    feat_network.cuda()

    lengt = len(liz_x) +1
    for epoch in range(total_epoch):
        rand_id = np.random.permutation(support_size * lengt)
        for j in range(0, support_size * lengt, batch_size):
            classifier_opt.zero_grad()
            delta_opt.zero_grad()
            #####################################
            selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size * lengt)]).to(device)
            
            z_batch = x_a_i_new[selected_id].to(device)

            y_batch = y_a_i[selected_id]

            #####################################

            output = feat_network(z_batch)

            scores  = classifier(output)

            loss = loss_fn(scores, y_batch)

            #####################################
            loss.backward()
            classifier_opt.step()
            delta_opt.step()
    
    if self.first == True:
      self.first = False
    self.feature2 = copy.deepcopy(self.feature)
    self.feature3 = copy.deepcopy(feat_network) ## before the new state_dict is copied over
    self.feature.load_state_dict(feat_network.state_dict())    
    self.classifier2 = copy.deepcopy(self.classifier)
    self.classifier3 = copy.deepcopy(classifier)
    self.classifier.load_state_dict(classifier.state_dict())

    
    for name, param  in self.feature.named_parameters():
        param.requires_grad = True    

    #output_support = self.feature(x_a_i.cuda()).view(self.n_way, self.n_support, -1)
    output_query = self.feature(x_b_i.cuda())
    scores = self.classifier(output_query)

    return scores

  def set_forward_finetune_ep_gnn(self,liz_x,is_feature=False, n_query = 16):
    ### introduce a random int generator here to randomly take an augmented sample. This is to allow for better regularization in the fine tuning process.
    x = liz_x[0] ### non-changed one
    self.n_query = n_query
    x = x.to(device)
    # get feature using encoder
    batch_size = 6
    support_size = self.n_way * self.n_support 

    for name, param  in self.feature.named_parameters():
      param.requires_grad = True

    x_var = Variable(x)
      
    y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() # (25,)

    self.MAML_update_gnn() ## call MAML update
    x_b_i = x_var[:, self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) 
    x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) # (25, 3, 224, 224)
    x_inn = x_var.view(self.n_way* (self.n_support + self.n_query), *x.size()[2:])
    
    ### to load all the changed examples

    x_a_i_new = x_a_i
    for i, x_aug in enumerate(liz_x):
      if i > 1: ## past the second one
        x_aug = x_aug.to(device)
        x_aug = Variable(x_aug)
        x_a_aug = x_aug[:,:self.n_support,:,:,:].contiguous()
        x_a_aug = x_a_aug.view( self.n_way* self.n_support, *x.size()[2:])
        y_a_aug = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).to(device)
        x_a_i_new = torch.cat((x_a_i_new, x_a_aug), dim = 0)
        y_a_i = torch.cat((y_a_i, y_a_aug.to(device)), dim = 0)
    

    feat_network = copy.deepcopy(self.feature)
    fc_network = copy.deepcopy(self.fc)
    gnn_network = copy.deepcopy(self.gnn)
    #classifier = Classifier(self.feat_dim, self.n_way)
    
    
    names = []
    for name, param in feat_network.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    assert self.num_FT_block <= 9, "cannot have more than 9 blocks unfrozen during training"
    
    names_sub = names[:self.num_FT_layers] ### last Resnet block can adapt. default is -9

    for name, param in feat_network.named_parameters():
      if name in names_sub:
        #print(name)
        param.requires_grad = False    

    delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, feat_network.parameters()), lr = 0.01)
    loss_fn = nn.CrossEntropyLoss().cuda() ##change this code up ## dorop n way
    gnn_opt = torch.optim.Adam(gnn_network.parameters(), lr = 0.01) ##try it with weight_decay
    fc_opt = torch.optim.Adam(fc_network.parameters(), lr = 0.01) ##try it with weight_decay
  
    total_epoch = self.ft_epoch ###change this

    gnn_network.train()
    fc_network.train()
    feat_network.train()

    gnn_network.cuda()
    fc_network.cuda()
    feat_network.cuda()

    lengt = len(liz_x) +1
    for epoch in range(total_epoch):
        rand_id = np.random.permutation(support_size * lengt)
        for j in range(0, support_size * lengt, batch_size):
            delta_opt.zero_grad()
            gnn_opt.zero_grad()
            fc_opt.zero_grad()

            #####################################
            selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size * lengt)]).to(device)
            
            z_batch = x_a_i_new[selected_id].to(device)
            y_batch = y_a_i[selected_id] 
            #####################################

            output = feat_network(z_batch)
            scores  = classifier(output)
            loss = loss_fn(output, y_batch)

            #####################################
            loss.backward()
            classifier_opt.step()
            delta_opt.step()
    
    if self.first == True:
      self.first = False
    self.feature2 = copy.deepcopy(self.feature)
    self.feature3 = copy.deepcopy(feat_network) ## before the new state_dict is copied over
    self.feature.load_state_dict(feat_network.state_dict())
    
    for name, param  in self.feature.named_parameters():
        param.requires_grad = True
    
    output_support = self.feature(x_a_i.cuda()).view(self.n_way, self.n_support, -1)
    output_query = self.feature(x_b_i.cuda()).view(self.n_way,self.n_query,-1)

    final = torch.cat((output_support, output_query), dim =1).cuda()

    assert(final.size(1) == self.n_support + 16) ##16 query samples in each batch
    z = self.fc(final.view(-1, *final.size()[2:]))
    z = z.view(self.n_way, -1, z.size(1))

    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    
    scores = self.forward_gnn(z_stack)   

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
    y_query = y_query.cuda()
    scores = self.set_forward(x)
    loss = self.loss_fn(scores, y_query)
    return loss

  def set_forward_loss_finetune(self, x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    scores = self.set_forward_finetune(x)
    loss = self.loss_fn(scores, y_query)
    return loss

  def set_forward_loss_finetune_ep(self, liz_x):
    y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
    y_query = y_query.cuda()
    
    scores = self.set_forward_finetune_ep(liz_x)
    loss = self.loss_fn(scores, y_query)
    return loss
