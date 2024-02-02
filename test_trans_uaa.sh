for trans_tau in 15 20 30
do
  export CUDA_VISIBLE_DEVICES=1
  python3 cliport/calib_test.py eval_task=stack-block-pyramid-seq-seen-colors \
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