#!/usr/bin/env

data=/path-to-train-dataset/train.csv,/path-to-validation-dataset/val.csv
selected_cols=0,4,2,3

log_dir=/log-path/logs
save_dir=/save-path/checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=/path-to-GenIE/utils/BPE
user_dir=/path-to-GenIE/ofa_module

task=vqa_gen
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
batch_size=8
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=8192
max_tgt_length=4096
num_bins=1000

val_inference_type=beamsearch

unconstrained_training_flag="--unconstrained-training"

for max_epoch in {50,}; do
  echo "max_epoch "${max_epoch}
  for warmup_ratio in {0.04,}; do
    echo "warmup_updates "${warmup_updates}  
    for lr in {5e-5,}; do
      echo "lr "${lr}
      for patch_image_size in {480,}; do
        echo "patch_image_size "${patch_image_size}

        log_file=${log_dir}/genie_base_sroie_fullshot.log
        save_path=${save_dir}/genie_base_sroie_fullshot
        mkdir -p $save_path

        python /path-to-GenIE/train.py \
            ${data} \
            --selected-cols=${selected_cols} \
            --bpe-dir=${bpe_dir} \
            --user-dir=${user_dir} \
            --restore-file=${restore_file} \
            --reset-optimizer --reset-dataloader --reset-meters \
            --save-dir=${save_path} \
            --task=${task} \
            --arch=${arch} \
            --criterion=${criterion} \
            --label-smoothing=${label_smoothing} \
            --batch-size=${batch_size} \
            --update-freq=${update_freq} \
            --encoder-normalize-before \
            --decoder-normalize-before \
            --share-decoder-input-output-embed \
            --share-all-embeddings \
            --layernorm-embedding \
            --patch-layernorm-embedding \
            --code-layernorm-embedding \
            --resnet-drop-path-rate=${resnet_drop_path_rate} \
            --encoder-drop-path-rate=${encoder_drop_path_rate} \
            --decoder-drop-path-rate=${decoder_drop_path_rate} \
            --dropout=${dropout} \
            --attention-dropout=${attention_dropout} \
            --weight-decay=0.01 \
            --optimizer=adam \
            --adam-betas="(0.9,0.999)" \
            --adam-eps=1e-08 \
            --clip-norm=1.0 \
            --lr-scheduler=polynomial_decay \
            --lr=${lr} \
            --max-epoch=${max_epoch} \
            --warmup-ratio=${warmup_ratio} \
            --log-format=simple \
            --log-interval=10 \
            --fixed-validation-seed=7 \
            --keep-last-epochs=15 \
            --save-interval=1 --validate-interval=1 \
            --best-checkpoint-metric=ner_accuracy --maximize-best-checkpoint-metric \
            --max-src-length=${max_src_length} \
            --max-tgt-length=${max_tgt_length} \
            --find-unused-parameters \
            --freeze-encoder-embedding \
            --freeze-decoder-embedding \
            ${unconstrained_training_flag} \
            --add-type-embedding \
            --scale-attn \
            --scale-fc \
            --scale-heads \
            --disable-entangle \
            --num-bins=${num_bins} \
            --patch-image-size=${patch_image_size} \
            --prompt-type=none \
            --fp16 \
            --fp16-scale-window=512 \
            --val-inference-type=${val_inference_type} \
            --num-workers=0 > ${log_file} 2>&1
      done
    done
  done
done
