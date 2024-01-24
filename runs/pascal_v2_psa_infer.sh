 python psa/infer_aff.py --weights weights/resnet38_aff.pth \
                    --infer_list voc12/train_id.txt \
                    --cam_dir MCTformer_results/voc/MCTformer_v2/attn-patchrefine-npy \
                    --voc12_root Datasets/VOC2012 \
                    --out_rw MCTformer_results/voc/MCTformer_v2/pgt-psa-rw \
