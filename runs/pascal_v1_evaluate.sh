######### Evaluating the generated class-specific localization maps ##########
CUDA_VISIBLE_DEVICES=6 python evaluation.py --list Datasets/VOC2012/ImageSets/Segmentation/train_id.txt \
                     --gt_dir Datasets/VOC2012/SegmentationClassAug \
                     --logfile MCTformer_results/voc/MCTformer_v1/attn-patchrefine-npy/evallog.txt \
                     --type npy \
                     --curve True \
                     --predict_dir MCTformer_results/voc/MCTformer_v1/attn-patchrefine-npy \
                     --comment "train1464"
