######### train MCTformer V1 ##########
CUDA_VISIBLE_DEVICES=6 python main.py --model deit_small_MCTformerV1_patch16_224 --batch-size 64 --data-set VOC12 --img-list voc12 --data-path Datasets/VOC2012 --layer-index 12 --output_dir MCTformer_results/voc/MCTformer_v1 --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
