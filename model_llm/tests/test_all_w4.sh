export CUDA_VISIBLE_DEVICES=1
export TRANSFORMERS_OFFLINE=1

tasks=("arc_easy" "arc_challenge" "piqa" "winogrande" "hellaswag")

models=("pretrained=/data01/home/xuzk/datasets/lm_mamba_weight/mamba-1.4b-hf" \
        "pretrained=/data01/home/xuzk/datasets/lm_mamba_weight/mamba-790m-hf" \
        "pretrained=/data01/home/xuzk/datasets/lm_mamba_weight/mamba-370m-hf" \
        "pretrained=/data01/home/xuzk/datasets/lm_mamba_weight/mamba-130m-hf" \
        "pretrained=/data01/home/xuzk/datasets/lm_mamba_weight/mamba-2.8b-hf")

for model in "${models[@]}"; do
    # for task in "${tasks[@]}"; do
    #     echo "model: $model, task: $task, w4a8_method_0.5"
    #     /data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tests/quant_naive.py \
    #         --model mamba \
    #         --model_args "$model" \
    #         --tasks "$task" \
    #         --device cuda \
    #         --batch_size 64 \
    #         --quant_weight \
    #         --quant_act \
    #         --w_bit 4 \
    #         --a_bit 8 \
    #         --w_perchannel \
    #         --use_hadmard \
    #         --use_hadmard_R1 \
    #         --use_hadmard_R5 

    # done
    
    # for task in "${tasks[@]}"; do
    #     echo "model: $model, task: $task, w4a8_method_1"
    #     /data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tests/quant_naive.py \
    #         --model mamba \
    #         --model_args "$model" \
    #         --tasks "$task" \
    #         --device cuda \
    #         --batch_size 64 \
    #         --quant_weight \
    #         --quant_act \
    #         --w_bit 4 \
    #         --a_bit 8 \
    #         --w_perchannel \
    #         --observe percentile \
    #         --use_hadmard \
    #         --use_S5 \
    #         --use_S7 \
    #         --use_hadmard_R1 \
    #         --use_hadmard_R5 \
    #         --use_pertoken \
    #         --use_klt
    # done
        for task in "${tasks[@]}"; do
        echo "model: $model, task: $task, w4a8_method_1.5"
        /data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tests/quant_naive.py \
            --model mamba \
            --model_args "$model" \
            --tasks "$task" \
            --device cuda \
            --batch_size 64 \
            --quant_weight \
            --quant_act \
            --w_bit 4 \
            --a_bit 8 \
            --w_perchannel \
            --observe percentile \
            --use_hadmard \
            --use_S2 \
            --use_S4 \
            --use_S5 \
            --use_S7 \
            --use_hadmard_R1 \
            --use_hadmard_R5 \
            --use_pertoken \
            --use_klt
    done
    
    for task in "${tasks[@]}"; do
        echo "model: $model, task: $task, w4a8_method_2"
        /data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tests/quant_naive.py \
            --model mamba \
            --model_args "$model" \
            --tasks "$task" \
            --device cuda \
            --batch_size 64 \
            --quant_weight \
            --quant_act \
            --w_bit 4 \
            --a_bit 8 \
            --w_perchannel \
            --observe percentile \
            --use_hadmard \
            --use_S2 \
            --use_S4 \
            --use_S5 \
            --use_S7 \
            --use_hadmard_R1 \
            --use_hadmard_R2 \
            --use_hadmard_R4 \
            --use_hadmard_R5 \
            --use_pertoken \
            --use_klt
    done
done