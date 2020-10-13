import os

os.system( python finetune_50.py --model ResNet10 --method all  --train_aug --n_shot 50 --save_iter 600 --fine_tune_epoch 5 --test_dataset CropDisease --gen_examples 17 &&  python finetune_50.py --model ResNet10 --method all  --train_aug --n_shot 50 --save_iter 600 --fine_tune_epoch 5 --test_dataset EuroSAT --gen_examples 17 
&&  python finetune_50.py --model ResNet10 --method all  --train_aug --n_shot 50 --save_iter 600 --fine_tune_epoch 5 --test_dataset ISIC --gen_examples 17 &&  python finetune_50.py --model ResNet10 --method all  --train_aug --n_shot 50 --save_iter 600 --fine_tune_epoch 5 --test_dataset ChestX --gen_examples 17 
)