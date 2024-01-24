CUDA_VISIBLE_DEVICES=7 python psa/infer_aff.py --weights weights/resnet38_aff.pth \
                    --infer_list coco/train_aug_id.txt \
                    --cam_dir /MCTformer/MCTformer_results/coco/MCTformer_v2/fused-patchrefine-npy \
                    --voc12_root Datasets/MSCOCO/train2014 \
                    --out_rw MCTformer_results/coco/MCTformer_v2/pgt-psa-rw \

