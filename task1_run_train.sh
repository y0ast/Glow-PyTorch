LOG=$(date +%m%d_%H%M_logs_task1)
echo $LOG
mkdir $LOG
python3 train.py --y_condition --output_dir $LOG \
                  --batch_size 8 \
                  --epochs 500 \
                  --dataroot "/home/yellow/deep-learning-and-practice/hw7/dataset/task_1/" \
                  --K 16 \
                  --L 3 \
                  --dataset "task1" \
                  --classifier_weight "/home/yellow/deep-learning-and-practice/hw7/classifier_weight.pth" \
                  --y_weight 0.1
