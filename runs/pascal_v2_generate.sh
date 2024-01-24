######### Generating class-specific localization maps ##########
CUDA_VISIBLE_DEVICES=7 python main.py --model deit_small_MCTformerV2_patch16_224 \
                --data-set VOC12MS \
                --scales 1.0 \
                --img-list voc12 \
                --data-path Datasets/VOC2012 \
                --resume MCTformer_results/voc/MCTformer_v2/checkpoint_best.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --attention-dir MCTformer_results/voc/MCTformer_v2/attn-patchrefine \
                --cam-npy-dir MCTformer_results/voc/MCTformer_v2/attn-patchrefine-npy \
                --out-crf MCTformer_results/voc/MCTformer_v2/attn-patchrefine-npy-crf \
