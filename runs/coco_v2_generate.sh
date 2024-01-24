######### Generating class-specific localization maps ##########
 python main.py --model deit_small_MCTformerV2_patch16_224 \
                --data-set COCOMS \
                --scales 1.0 \
                --img-list coco \
                --data-path Datasets/MSCOCO \
                --resume weights/MCTformerV2_coco.pth \
                --gen_attention_maps \
                --attention-type fused \
                --layer-index 3 \
                --visualize-cls-attn \
                --patch-attn-refine \
                --label-file-path COCO_cls_labels.npy \
                --attention-dir MCTformer_results/MCTformer_v2/coco/fused-patchrefine \
                --cam-npy-dir MCTformer_results/MCTformer_v2/coco/fused-patchrefine-npy \
                --output_dir MCTformer_results/coco/MCTformer_v2 \

