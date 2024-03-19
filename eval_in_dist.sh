export CUDA_VISIBLE_DEVICES=0

eval_tasks=("stack-block-pyramid-seq-seen-colors"
            "packing-seen-google-objects-seq"
            "packing-seen-google-objects-group"
            "assembling-kits-seq-seen-colors")

for eval_task in "${eval_tasks[@]}"
do
    python3 cliport/calib_test.py eval_task=$eval_task \
                        agent=cliport \
                        mode=test \
                        n_demos=1000 \
                        train_demos=1000 \
                        checkpoint_type=test_best \
                        exp_folder=exps \
                        action_selection.enabled=False\
                        action_selection.attn_uaa=True\
                        action_selection.attn_tau=9\
                        exp_name=1000eps_base

    python3 cliport/calib_test.py eval_task=$eval_task \
                        agent=cliport \
                        mode=test \
                        n_demos=1000 \
                        train_demos=1000 \
                        checkpoint_type=test_best \
                        exp_folder=exps \
                        action_selection.enabled=True\
                        action_selection.attn_uaa=True\
                        action_selection.attn_tau=11\
                        exp_name=1000eps_attn