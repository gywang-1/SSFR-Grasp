dataset_name="refcoco" # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name="bridge_v16.yaml"
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python3 -u train.py \
      --config config/$dataset_name/$config_name \
      #--opts dist_url tcp://127.0.0.1:29501