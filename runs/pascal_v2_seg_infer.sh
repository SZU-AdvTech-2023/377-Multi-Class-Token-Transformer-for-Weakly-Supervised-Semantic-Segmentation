

CUDA_VISIBLE_DEVICES=7 python seg/infer_seg.py --weights MCTformer_results/voc/MCTformer_v2/seg_log/model_29.pth  \
                      --network resnet38_seg \
                      --list_path voc12/val_id.txt \
                      --gt_path Datasets/VOC2012/SegmentationClassAug  \
                      --img_path Datasets/VOC2012/JPEGImages  \
                      --save_path val_ms_crf \
                      --save_path_c val_ms_crf_c \
                      --scales 0.5 0.75 1.0 1.25 1.5 \
                      --use_crf True
