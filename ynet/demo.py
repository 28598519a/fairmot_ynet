import os
import pandas as pd
import yaml
import argparse
import torch
from model import YNet

# Some hyperparameters and settings

CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
DATASET_NAME = 'sdd'

# TEST_DATA_PATH = 'data/SDD/test_trajnet.pkl'
# update pandas ikut ke fairmot jadi versi 1.5.1
TEST_DATA_PATH = 'data/FairYnet/fair-mot_test.pkl'
TEST_IMAGE_PATH = 'data/FairYnet/test'  # only needed for YNet, PECNet ignores this value
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 2  # K_e (To spilt and cluster goal point)
NUM_TRAJ = 1  # K_a

ROUNDS = 1  # Y-net is stochastic. How often to evaluate the whole dataset
BATCH_SIZE = 8

# Load config file and print hyperparameters

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
params

# Load preprocessed Data

print(TEST_DATA_PATH)
df_test = pd.read_pickle(TEST_DATA_PATH)
# df_test = df_test[:20]
# print(df_test['frame'], df_test['trackId'])
# print(df_test['x'], df_test['y'])
# print(df_test['sceneId'], df_test['metaId'])
# df_test['y'][:8] = df_test['y'][:8] - 600

df_test.head()
# isi dftest
# frame is int64
# trackId is float64 from numpy
# x float64
# y float64
# sceneId is object
# metaId is int64
#   frame  trackId       x      y  sceneId  metaId
# 0       0     28.0  1539.5  578.0  coupa_0       0
# 1      12     28.0  1484.5  576.0  coupa_0       0
# 2      24     28.0  1484.5  576.0  coupa_0       0
# 3      36     28.0  1459.5  571.0  coupa_0       0
# 4      48     28.0  1432.5  569.0  coupa_0       0
# 5      60     28.0  1407.5  564.0  coupa_0       0
# 6      72     28.0  1382.5  562.0  coupa_0       0
# 7      84     28.0  1355.5  560.0  coupa_0       0
# 8      96     28.0  1330.5  555.0  coupa_0       0
# 9     108     28.0  1303.5  553.0  coupa_0       0
# 10    120     28.0  1278.5  551.0  coupa_0       0
# 11    132     28.0  1253.5  546.0  coupa_0       0
# 12    144     28.0  1226.5  544.0  coupa_0       0
# 13    156     28.0  1201.5  539.0  coupa_0       0
# 14    168     28.0  1173.5  537.0  coupa_0       0
# 15    180     28.0  1149.5  535.0  coupa_0       0
# 16    192     28.0  1127.5  530.0  coupa_0       0
# 17    204     28.0  1105.5  528.0  coupa_0       0
# 18    216     28.0  1083.5  524.0  coupa_0       0
# 19    228     28.0  1061.5  521.0  coupa_0       0
# mencoba dummy data tracking dari fairmot, berhasil, harus buat folder baru di sdd dengan
# nama folder coba dan setiap image diganti dengan nama reference.jpg
# selanjutnya kita coba print setiap prediksi yang dia punya
# df_test['sceneId'] = 'coba'
# df_test['x'][8:] = 0
# df_test['y'][8:] = 0
# print(df_test)

# Initiate model and load pretrained weights

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
model.load(f'pretrained_models/{experiment_name}_weights.pt')

# Evaluate model

# model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
#                batch_size=BATCH_SIZE, rounds=ROUNDS,
#                num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME)

for j in range(0, len(df_test), OBS_LEN + PRED_LEN):
    df_split = df_test[j: OBS_LEN + PRED_LEN + j]
    df_split.head()
    model.evaluate(df_split, params, image_path=TEST_IMAGE_PATH,
                   batch_size=BATCH_SIZE, rounds=ROUNDS,
                   num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME)

# -r 取決於原始影片fps (final video用-preset placebo)
os.chdir('../FairMOT/demo')
# use \\ not / to call
cmd_str = '"..\\..\\dependency\\ffmpeg" -start_number 0 -i {}/%05d.jpg -c:v libx264 -preset veryslow -r 24 -crf 22 -vf scale=trunc(iw/2)*2:trunc(ih/2)*2 -pix_fmt yuv420p {}'.format('frame_ynet', 'ynet_result.mp4')
os.system(cmd_str)
