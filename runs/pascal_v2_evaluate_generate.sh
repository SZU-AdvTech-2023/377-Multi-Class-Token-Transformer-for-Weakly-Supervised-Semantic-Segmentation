
######### Evaluating the generated class-specific localization maps ##########
python evaluation.py --list voc12/train_id.txt \
                     --gt_dir Datasets/VOC2012/SegmentationClassAug \
                     --logfile psa/voc12/evallog.txt \
                     --type npy \
                     --curve True \
                     --predict_dir MCTformer_results/voc/MCTformer_v2/attn-patchrefine-npy  \
                     --comment "train1464"

