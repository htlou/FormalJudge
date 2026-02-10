DIRS=(
    # PATH TO AGENT OUTPUTS
)

BASE_MODEL_NAME=(
    # BASE MODEL NAMES
)

for i in $(seq 0 $(( ${#DIRS[@]} - 1 ))); do
    DIR=${DIRS[i]}
    MODEL_NAME=${BASE_MODEL_NAME[i]}
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_with_shield.py --model_path ~/models/ShieldAgent --filepath $DIR --filename gen_res.json --label_type "" --batch_size 16 --target_model_name $MODEL_NAME
done