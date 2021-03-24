# DAMSL: Domain Agnostic Meta Score-based Learning for Cross-Domain Few-Shot Learning

## Introduction

Submission for ICCV 2021.

### Abstract

  While many deep learning methods have successfully tackled domain adaptation and few-shot learning separately, far fewer methods are able to tackle Cross-Domain Few-Shot Learning (CD-FSL).  We identify key problems in previous meta-learning methods over-fitting to the source domain, and previous transfer-learning methods overly relying on a fine-tuning process that does not utilize the structure of the support set. In this paper, we propose Domain Agnostic Meta Score-based Learning (DAMSL), a novel, versatile and highly effective solution that addresses the above problems to deliver significant out-performance over state-of-the-art methods. The core idea is that instead of directly using the scores from a fine-tuned feature encoder, we use these scores to create input coordinates for a domain independent metric space. A graph neural network is applied to learn an embedding and relation function over these coordinates, in order to process all information contained in the score distribution of the support set. Our proposed module is versatile and can be attached to refine any initial set of input scores over an episode. We test our model on both established CD-FSL benchmarks and new domains. From our extensive experiments across 5-shot, 20-shot and 50-shot, we show that our method overcomes the limitations of previous meta-learning and transfer-learning methods to deliver substantial improvements in accuracy across both smaller and larger domain shifts.


## Results

* **Average accuracy across all trials: 74.99\% 
* This is a 6.86\% improvement over the best-performing fine-tuning model (Transductive Fine-Tuning) and a 15.21\% improvement over the best-performing meta-learning model (Prototypical Networks).

## Key Contributions

* Achives state-of-the-art performance compared to previous methods.
* First method to propose using pre-softmax classification scores as coordinates for a metric space. Unlocks a new direction for score-based performance boosting.
* Provides a flexible framework to combine transfer-based and metric-based meta-learning methods.

## Datasets
The following datasets are used for this paper.

### Source domain: 

* miniImageNet.

    Downsampled for faster training: https://www.dropbox.com/s/sbttsmb1cca0y0k/miniImagenet3.zip?dl=0

### Target domains of BSCD-FSL: 

* **EuroSAT**:

    Home: http://madm.dfki.de/downloads

    Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.kitware.com/#phase/5abcbc6f56357d0139260e66

* **Plant Disease**:

    Home: https://www.kaggle.com/saroz014/plant-disease/

    Direct: command line `kaggle datasets download -d plant-disease/data`

* **ChestX-Ray8**:

    Home: https://www.kaggle.com/nih-chest-xrays/data

    Direct: command line `kaggle datasets download -d nih-chest-xrays/data`

    Note that some rearrangement of the files is required in order to fit the format of the dataloader.


## Additional Target Domains used in this paper

* CIFAR100
* Caltech256
* CUB
* DTD (https://www.robots.ox.ac.uk/~vgg/data/dtd/)
* Places
* Cars
* Plantae

### Codebase
The codebase is built on previous work by https://github.com/IBM/cdfsl-benchmark [1] and https://github.com/hytseng0509/CrossDomainFewShot. [2]


## Steps for Loading Data   

1. Download the datasets for evaluation (EuroSAT, ISIC2018, Plant Disease, ChestX-Ray8) using the above links for BSCD-FSL. Download the other relevant datasets if testing on other domains. 

2. Download miniImageNet using:

    ```bash
     wget https://www.dropbox.com/s/sbttsmb1cca0y0k/miniImagenet3.zip?dl=1
    ```

    These are the downsampled images of the original dataset that were used in this study. Downsampled for faster training.

3. Change configuration file `./configs.py` to reflect the correct paths to each dataset. Please see the existing example paths for information on which subfolders these paths should point to.


## Steps for Testing using Pre-trained Models

1. Download the pre-trained models from a link that you can find here: https://www.dropbox.com/s/f0hj68z2s5evo8b/logs_final.zip?dl=0

    ```bash
     wget https://www.dropbox.com/s/f0hj68z2s5evo8b/logs_final.zip?dl=1
    ```
 
    Unzip the file and place it in the main directory of the project
 
2. Run the main experiments in this paper for 5-shot, 20-shot and 50-shot

    • *5-shot*

    ```bash
     python finetune.py --model ResNet10 --method damsl_v2  --train_aug --n_shot 5 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```

    • *20-shot*

    ```bash
     python finetune.py --model ResNet10 --method damsl_v2 --train_aug --n_shot 20 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```

    • *50-shot*
    ```bash
     python finetune.py --model ResNet10 --method damsl_v2  --train_aug --n_shot 50 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
     ```
 
  • *Example output:* 600 Test Acc = 98.78% +- 0.19%
 
 Replace the test_dataset argument with {CropDisease, EuroSat, ISIC, ChestX}.

 Replace the method argument with {damsl_v1, damsl_v2}.

3. If there is an error in data loading in the next few steps below, it is most likely because of the num_workers argument - multi-threading large files may not work, especially at larger shots. 
 
   If error is encountered, do the following:
   Configure the num_workers=0 in the data_loader_params in the functions of SetDataset2.get_data_loader in:
  
    CropDisease_few_shot.py,
    EuroSAT_few_shot.py,
    ISIC_few_shot.py,
    Chest_few_shot.py
   
   Another edit you can do is to if you run out of RAM is to change the data_loading process to read images on the fly (this would reduce the memory load but take longer to run). 
 
## Steps for Re-training and Testing


1. Train supervised feature encoders on miniImageNet for 400 epochs

    • *Standard supervised learning on miniImageNet using SGD*
    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method baseline --train_aug --start_epoch 0 --stop_epoch 401
    ```
    • *Standard supervised learning on miniImageNet using Adam*
    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method baseline --train_aug --start_epoch 0 --stop_epoch 401 --optimizer Adam
    ```

2. Episodic Training of DAMSL_v2 module on MiniImageNet for another 200 epochs

    • *GNN on miniImageNet for 5 shot*

    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method damsl_v2 --n_shot 5 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
    ```
   • *GNN on miniImageNet for 20 shot*

    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method damsl_v2 --n_shot 20 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
    ```
 
    • *GNN on miniImageNet for 50 shot*

    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method damsl_v2 --n_shot 50 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
    ```
    
    Note: if we are using damsl_v1 instead, we would need to train the GNN feature encoder as well.

3. Test

    Follow step 2 and 3 in the "Testing using Pre-trained Models" section.
    
## Steps for Other Results and Ablation Studies

1. No Data Augmentation

    To remove data augmentation, change the argument for --gen-examples from "17" to "0".

2. Ablation Study: Linear Meta Transfer-Learning

    Add argument "--ablation linear"

    ```bash
     python finetune.py --model ResNet10 --method damsl_v1  --train_aug --n_shot 5 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 --ablation linear
    ```

    Replace the test_dataset argument with {CropDisease, EuroSat, ISIC, ChestX}.

3. Study of the Confusion Matrix: Asymmetric Confusion

   ```bash
     python finetune_confusion.py --model ResNet10 --method damsl_v1  --train_aug --n_shot 5 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```

    Replace the test_dataset argument with {CropDisease, EuroSat, ISIC, ChestX}.

4. Score-based Prototypical Networks

    Will need to retrain the model between 401 and 601 epochs and then follow the same steps.

    Models incldue: {damsl_v1_proto, damsl_v2_proto}


## References

[1] Yunhui  Guo,  Noel  CF  Codella,  Leonid  Karlinsky,  John  RSmith,  Tajana  Rosing,  and  Rogerio  Feris. A  new  bench-mark for evaluation of cross-domain few-shot learning.arXivpreprint arXiv:1912.07200, 2019

[2] Tseng, H. Y., Lee, H. Y., Huang, J. B., & Yang, M. H. Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation. arXiv preprint arXiv:2001.08735, 2020.

