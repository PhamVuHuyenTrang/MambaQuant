export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_OFFLINE=1

RESUME="./ckpt/mamba2d_s.pth"
CONFIG="./config/mamba2d_s.py"

echo "$RESUME w4a8 method_1A **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tools/quant_naive.py \
    --checkpoint "$RESUME"\
    --config "$CONFIG" \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 4 \
    --a_bit 8 \
    --use_hadmard \
    --use_S2 \
    --fake_online_hadamard 

echo "$RESUME w4a8 method_1b **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tools/quant_naive.py \
    --checkpoint "$RESUME"\
    --config "$CONFIG" \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 4 \
    --a_bit 8 \
    --use_hadmard \
    --use_S2 \
    --use_hadmard_R1 \
    --use_hadmard_R5 \
    --fake_online_hadamard 
    
echo "$RESUME w4a8 method_1c **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tools/quant_naive.py \
    --checkpoint "$RESUME"\
    --config "$CONFIG" \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 4 \
    --a_bit 8 \
    --use_hadmard \
    --use_S2 \
    --use_hadmard_R1 \
    --use_hadmard_R3 \
    --use_hadmard_R5 \
    --fake_online_hadamard 

echo "$RESUME w4a8 method_2 **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tools/quant_naive.py \
    --checkpoint "$RESUME"\
    --config "$CONFIG" \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 4 \
    --a_bit 8 \
    --use_hadmard \
    --use_S2 \
    --use_hadmard_R1 \
    --use_hadmard_R3 \
    --use_hadmard_R5 \
    --observe percentile

echo "$RESUME w4a8 method_2 **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tools/quant_naive.py \
    --checkpoint "$RESUME"\
    --config "$CONFIG" \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 4 \
    --a_bit 8 \
    --use_hadmard \
    --use_S2 \
    --use_hadmard_R1 \
    --use_hadmard_R2 \
    --use_hadmard_R3 \
    --use_hadmard_R5 \
    --observe percentile

RESUME="./ckpt/mamba2d_b.pth"
CONFIG="./config/mamba2d_b.py"

echo "$RESUME w4a8 method_1A **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tools/quant_naive.py \
    --checkpoint "$RESUME"\
    --config "$CONFIG" \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 4 \
    --a_bit 8 \
    --use_hadmard \
    --use_S2 \
    --fake_online_hadamard 

echo "$RESUME w4a8 method_1b **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tools/quant_naive.py \
    --checkpoint "$RESUME"\
    --config "$CONFIG" \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 4 \
    --a_bit 8 \
    --use_hadmard \
    --use_S2 \
    --use_hadmard_R1 \
    --use_hadmard_R5 \
    --fake_online_hadamard 
    
echo "$RESUME w4a8 method_1c **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tools/quant_naive.py \
    --checkpoint "$RESUME"\
    --config "$CONFIG" \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 4 \
    --a_bit 8 \
    --use_hadmard \
    --use_S2 \
    --use_hadmard_R1 \
    --use_hadmard_R3 \
    --use_hadmard_R5 \
    --fake_online_hadamard 

echo "$RESUME w4a8 method_2 **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tools/quant_naive.py \
    --checkpoint "$RESUME"\
    --config "$CONFIG" \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 4 \
    --a_bit 8 \
    --use_hadmard \
    --use_S2 \
    --use_hadmard_R1 \
    --use_hadmard_R3 \
    --use_hadmard_R5 \
    --observe percentile

echo "$RESUME w4a8 method_2 **********************************************************************"
/data01/home/xuzk/anaconda3/envs/mamba-nd/bin/python tools/quant_naive.py \
    --checkpoint "$RESUME"\
    --config "$CONFIG" \
    --static_quant \
    --quant_weight \
    --quant_act \
    --w_bit 4 \
    --a_bit 8 \
    --use_hadmard \
    --use_S2 \
    --use_hadmard_R1 \
    --use_hadmard_R2 \
    --use_hadmard_R3 \
    --use_hadmard_R5 \
    --observe percentile