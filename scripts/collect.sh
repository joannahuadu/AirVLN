
conda activate AirVLN

cd ./AirVLN
echo $PWD


# nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1,4,5,6,7,8 &

python -u ./src/vlnce_src/train.py \
--run_type collect \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize 16


