# SB-MTL: Score-based Meta Transfer-Learning for Cross-Domain Few-Shot Learning

## Introduction

Submission for the CVPR 2021 Main Conference.

### Abstract

  While many deep learning methods have seen significant success in tackling the problem of domain adaptation and few-shot learning separately, far fewer methods are able to jointly tackle both problems in Cross-Domain Few-Shot Learning (CD-FSL). This problem is exacerbated under sharp domain shifts that typify common computer vision applications. In this paper, we present a novel, flexible and effective method to address the CD-FSL problem. Our method, called Score-based Meta Transfer-Learning (SB-MTL), combines transfer-learning and meta-learning by using a MAML-optimized feature encoder and a score-based Graph Neural Network. First, we have a feature encoder that has specific layers designed to be fine-tuned. To do so, we apply a first-order MAML algorithm to find good initializations. Second, instead of directly taking the classification scores after fine-tuning, we interpret the scores as coordinates by mapping the pre-softmax classification scores onto a metric space. Subsequently, we apply a Graph Neural Network to propagate label information from the support set to the query set in our score-based metric space. We test our model on the Broader Study of Cross-Domain Few-Shot Learning (BSCD-FSL) benchmark, which includes a range of target domains with highly varying dissimilarity to the miniImagenet source domain. We observe significant improvements in accuracy across 5-shot, 20-shot and 50-shot, and on the four target domains of the BSCD-FSL benchmark. In terms of average accuracy, our model outperforms previous transfer-learning methods by 5.93% and outperforms previous meta-learning methods by 14.28%.


## Results

* **Average accuracy across all trials: 74.06\% 
* This is a 5.93\% improvement over the best-performing fine-tuning model (Transductive Fine-Tuning) and a 14.28\% improvement over the best-performing meta-learning model (Prototypical Networks).

## Key Contributions

* Achives state-of-the-art performance compared to previous methods.
* First method to propose using pre-softmax classification scores as coordinates for a metric space.
* Provides a flexible framework to combine transfer-based and metric-based meta-learning methods.

## Datasets
The following datasets are used for this paper.

### Source domain: 

* miniImageNet.

    Downsampled for faster training: https://www.dropbox.com/s/sbttsmb1cca0y0k/miniImagenet3.zip?dl=0

### Target domains: 

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

### Codebase
The codebase is built on previous work by https://github.com/IBM/cdfsl-benchmark [1] and https://github.com/hytseng0509/CrossDomainFewShot. [2]


## Steps for Loading Data   

1. Download the datasets for evaluation (EuroSAT, ISIC2018, Plant Disease, ChestX-Ray8) using the above links. 

2. Download miniImageNet using:

    ```bash
     wget https://www.dropbox.com/s/sbttsmb1cca0y0k/miniImagenet3.zip?dl=1>
    ```

    These are the downsampled images of the original dataset that were used in this study. Trains faster.

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
     python finetune.py --model ResNet10 --method sbmtl  --train_aug --n_shot 5 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```

    • *20-shot*

    ```bash
     python finetune.py --model ResNet10 --method sbmtl  --train_aug --n_shot 20 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```

    • *50-shot*
    ```bash
     python finetune.py --model ResNet10 --method sbmtl  --train_aug --n_shot 50 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
     ```
 
  • *Example output:* 600 Test Acc = 98.78% +- 0.19%
 
 Replace the test_dataset argument with {CropDisease, EuroSat, ISIC, ChestX}.
 
3. If there is an error in data loading in the next few steps below, it is most likely because of the num_workers argument - multi-threading large files may not work, especially at larger shots. 
 
   If error is encountered, do the following:
   Configure the num_workers=0 in the data_loader_params in the functions of SetDataset2.get_data_loader in:
  
    CropDisease_few_shot.py,
    EuroSAT_few_shot.py,
    ISIC_few_shot.py,
    Chest_few_shot.py
   
   Another edit you can do is to if you run out of RAM is to change the data_loading process to read images on the fly (this would reduce the memory load but take longer to run). 
 
## Steps for Re-training and Testing


1. Train supervised feature encoder on miniImageNet for 400 epochs

    • *Standard supervised learning on miniImageNet*
    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method baseline --train_aug --start_epoch 0 --stop_epoch 401
    ```
2. Train GNN feature encoder on MiniImagenet for 5 and 20 shots for 400 epochs

    • *GNN on miniImageNet for 5 shot*

    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method sbmtl --n_shot 5 --train_aug --start_epoch 0 --stop_epoch 401
    ```
    
    • *GNN on miniImageNet for 20 shot*

    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method sbmtl --n_shot 20 --train_aug --start_epoch 0 --stop_epoch 401
    ```

    • *GNN on miniImageNet for 50 shot*

    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method sbmtl --n_shot 50 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
    ```

3. Episodic Training of Score-based Meta Transfer-Learning on MiniImageNet for another 200 epochs

    • *GNN on miniImageNet for 5 shot*

    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method sbmtl --n_shot 5 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
    ```
   • *GNN on miniImageNet for 20 shot*

    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method sbmtl --n_shot 20 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
    ```
 
    • *GNN on miniImageNet for 50 shot*

    ```bash
     python train.py --dataset miniImageNet --model ResNet10  --method sbmtl --n_shot 50 --train_aug --start_epoch 401 --stop_epoch 601 --fine_tune
    ```
    
6. Test

    Follow step 2 and 3 in the "Testing using Pre-trained Models" section.
    
## Steps for Other Results

1. No Data Augmentation

    To remove data augmentation, change the argument for --gen-examples from "17" to "0".

2. Ablation Study: Linear Meta Transfer-Learning

    Same arguments, but run the finetune_ablation.py file instead.

    ```bash
     python finetune_ablation.py --model ResNet10 --method sbmtl  --train_aug --n_shot 5 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```

    Replace the test_dataset argument with {CropDisease, EuroSat, ISIC, ChestX}.

3. Study of the Confusion Matrix: Asymmetric Confusion

   ```bash
     python finetune_confusion.py --model ResNet10 --method sbmtl  --train_aug --n_shot 5 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 
    ```

    Replace the test_dataset argument with {CropDisease, EuroSat, ISIC, ChestX}.


## References

[1] Yunhui  Guo,  Noel  CF  Codella,  Leonid  Karlinsky,  John  RSmith,  Tajana  Rosing,  and  Rogerio  Feris.A  new  bench-mark for evaluation of cross-domain few-shot learning.arXivpreprint arXiv:1912.07200, 2019

[2] Tseng, H. Y., Lee, H. Y., Huang, J. B., & Yang, M. H. Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation. arXiv preprint arXiv:2001.08735, 2020.

