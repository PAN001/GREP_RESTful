for i in 1
do
python extract.py --net_number $i --test_set train --loss softmax --model_path "weights/"$i"nets_model.npy" --data_path /Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/HappeiDetectedFaces/train/
done
