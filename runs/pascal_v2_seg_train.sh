python seg/train_seg.py --network resnet38_seg \
                    --num_epochs 30 \
                    --seg_pgt_path MCTformer_results/voc/MCTformer_v2/pgt-psa-rw \
                    --init_weights weights/res38_cls.pth \
                    --save_path  MCTformer_results/voc/MCTformer_v2/seg_log \
                    --list_path voc12/train_aug_id.txt \
                    --img_path Datasets/VOC2012/JPEGImages \
                    --num_classes 21 \
                    --batch_size 16
