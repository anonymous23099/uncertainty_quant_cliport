eval_tasks=("packing-seen-google-objects-seq"
            "packing-seen-google-objects-group")
eval_tasks=("packing-seen-google-objects-group")

eval_tasks=("assembling-kits-seq-seen-colors")

export CUDA_VISIBLE_DEVICES=2

for eval_task in "${eval_tasks[@]}"
do
  for trans_tau in 5 11 15 21
  do
    python3 cliport/calib_test.py eval_task=$eval_task \
                        agent=cliport \
                        mode=test \
                        n_demos=100 \
                        train_demos=1000 \
                        checkpoint_type=test_best \
                        exp_folder=exps \
                        action_selection.enabled=True\
                        action_selection.trans_uaa=True\
                        action_selection.trans_tau=$trans_tau
  done
done
# packing-seen-google-objects-seq