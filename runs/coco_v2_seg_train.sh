
CUDA_VISIBLE_DEVICES=7 python seg/train_seg.py --network resnet38_seg \
                    --num_epochs 30 \
                    --seg_pgt_path MCTformer_results/coco/MCTformer_v2/fused-patchrefine  \
                    --init_weights weights/res38_cls.pth\
                    --save_path  MCTformer_results/coco/MCTformer_v2/seg_log\
                    --list_path coco/train_id.txt \
                    --img_path Datasets/MSCOCO/train2014  \
                    --num_classes 80 \
                    --batch_size 8