######### Generating class-specific localization maps ##########
CUDA_VISIBLE_DEVICES=6 python main.py --model deit_small_MCTformerV1_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list Datasets/VOC2012/ImageSets/Segmentation \
                --data-path Datasets/VOC2012 \
                --output_dir MCTformer_results/voc/MCTformer_v1 \
                --resume MCTformer_results/voc/MCTformer_v1/checkpoint.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --attention-dir MCTformer_results/voc/MCTformer_v1/attn-patchrefine \
                --cam-npy-dir MCTformer_results/voc/MCTformer_v1/attn-patchrefine-npy \
