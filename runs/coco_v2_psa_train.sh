CUDA_VISIBLE_DEVICES=7 python psa/train_aff.py --weights weights/res38_cls.pth \
                        --voc12_root Datasets/MSCOCO/train2014 \
                        --train_list coco/train_id.txt \
                        --session_name coco_resnet38_aff \
                        --batch_size 16 \
                        --la_crf_dir /MCTformer_results/coco/MCTformer_v2/fused-patchrefine-npy-crf_1 \
                        --ha_crf_dir /MCTformer_results/coco/MCTformer_v2/fused-patchrefine-npy-crf_12 \

