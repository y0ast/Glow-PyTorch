LOG=$(date +%m%d_%H%M_logs)
echo $LOG
python3 train.py --y_condition --output_dir $LOG \
                  --batch_size 16 \
                  --epochs 25 \
                  --dataroot "/home/yellow/deep-learning-and-practice/hw7/dataset/task_2/" \
                  --K 5 \
                  --L 3
