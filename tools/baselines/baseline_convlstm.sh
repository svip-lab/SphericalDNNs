CUDA_VISIBLE_DEVICES=0,1 python train_ablation_convlstm.py \
                         --batch-size 24\
                         -lr 3e-3\
                         -epochs 200\
                         --save-interval 10\
                         --sequence-len 6\
                         --exp-name 'ablation_convlstm-seqlen6-FTTT-node25'\
                         --snapshot-fname-dir 'ablation_convlstm_seqlen_6_FTTT_node25_3e2_b32_05_20'\
                         --snapshot-fname-prefix 'ablation_convlstm_seqlen_6_FTTT_node25_3e2_b32_05_20'\
                         
