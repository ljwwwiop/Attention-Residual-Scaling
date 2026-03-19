cd /lpai/volumes/mind-vla-ali-sh-mix/lianjiawei/foundation_model/AttnRes_Scaled

export CUDA_VISIBLE_DEVICES=7,6,1,2,3,5,4,0

# export CUDA_VISIBLE_DEVICES=5,6

torchrun --nproc_per_node=8 train.py --config configs/medium-large.yaml --model_type attnres_block


