SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo $SCRIPT_DIR


rm $SCRIPT_DIR/model_vim_quant/saved_checkpoint
rm $SCRIPT_DIR/model_image_classification/ckpt
rm $SCRIPT_DIR/model_video_classification/ckpt
rm $SCRIPT_DIR/model_segmentation/ckpt
rm $SCRIPT_DIR/model_llm/states-spaces

ln -s $SCRIPT_DIR/saved_checkpoint  $SCRIPT_DIR/model_vim_quant/saved_checkpoint
ln -s $SCRIPT_DIR/saved_checkpoint  $SCRIPT_DIR/model_image_classification/ckpt
ln -s $SCRIPT_DIR/saved_checkpoint  $SCRIPT_DIR/model_video_classification/ckpt
ln -s $SCRIPT_DIR/saved_checkpoint  $SCRIPT_DIR/model_segmentation/ckpt
ln -s "/data01/home/xuzk/datasets/lm_mamba_weight/"  $SCRIPT_DIR/model_llm/states-spaces



rm $SCRIPT_DIR/model_vim_quant/quantize
rm $SCRIPT_DIR/model_image_classification/quantize
rm $SCRIPT_DIR/model_video_classification/quantize
rm $SCRIPT_DIR/model_segmentation/quantize
rm $SCRIPT_DIR/model_llm/quantize

ln -s $SCRIPT_DIR/quantize  $SCRIPT_DIR/model_vim_quant/quantize
ln -s $SCRIPT_DIR/quantize  $SCRIPT_DIR/model_image_classification/quantize
ln -s $SCRIPT_DIR/quantize  $SCRIPT_DIR/model_video_classification/quantize
ln -s $SCRIPT_DIR/quantize  $SCRIPT_DIR/model_segmentation/quantize
ln -s $SCRIPT_DIR/quantize  $SCRIPT_DIR/model_llm/quantize
