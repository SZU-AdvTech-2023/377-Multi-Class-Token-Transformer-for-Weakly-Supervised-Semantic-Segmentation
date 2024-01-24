 python main.py  --model deit_small_MCTformerV2_patch16_224 \
                --batch-size 64 \
                --data-set COCO \
                --img-list coco \
                --data-path /data0/zhongxiang/code/MCTformer/Datasets/MSCOCO \
                --layer-index 12 \
                --output_dir MCTformer_results/coco/MCTformer_v2 \
                --label-file-path COCO_cls_labels.npy \
                --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
