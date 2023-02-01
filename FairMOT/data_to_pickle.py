import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

def convertion_csv():
    txt = open('results.txt', 'r')
    t = txt.read()
    tind = 'frame_n,id,x,y,w,h,s1,s2,s3,s4\n' + t
    csv = open('results.csv', 'w')
    csv.write(tind)
    txt.close()
    csv.close()

def process_data(csv_or):
    data = pd.read_csv(csv_or)
    all_id_frame = []
    obs_len = 8
    obj_div = 5  #default is 1 but will have more trajectories process (ex: 30fps / 5 = 6fps)
    #print('id1\n', data.loc[data['id'] == 1].iloc[:,:4])
    # frame_n from fairmot should be add by negatif one more because there is error
    data['frame_n'] = data['frame_n'] - 1
    # the fix at point x, y has to be in the middle so it looks like this
    '''
    data['x'] = data['x'] + (data['w'] / 2)
    data['y'] = data['y'] + (data['h'] / 2)
    '''
    # Use Buttom
    data['x'] = data['x'] + (data['w'] / 2)
    data['y'] = data['y'] + (data['h'])
    uniq_obj_id = np.array(data['id'].tolist())
    uniq_obj_id = np.unique(uniq_obj_id)
    for un in uniq_obj_id:
        obj_max = len(data.loc[data['id'] == un].iloc[:, :4])
        div, div_frameid = [], []
        arranges = np.arange(0, obj_max, obj_div)
        start = 0
        while start + obs_len <= len(arranges):
            div.append(list(arranges[start : start + obs_len]))
            start += 1
        data_ids = data.loc[data['id'] == un].iloc[:,:4]
        data_ids = data_ids.reset_index(drop=True)
        for g in div:
            data_buffer = data_ids.iloc[g, :4]
            data_buffer = data_buffer.reset_index(drop=True)
            if data_buffer.empty == False:
                div_frameid.append(data_buffer)
        if div:
            all_id_frame.append(div_frameid)
        # print(len(div))
    uniq_frame = []
    for g in all_id_frame:
        for h in g:
            uniq_frame.append(h.iloc[-1, 0])
    uniq_frame_np = np.asarray(uniq_frame)
    uniq_frame_np = np.unique(uniq_frame_np)
        # print("ini apa sih", un, obj_max , len(div))
    # object divider
    return all_id_frame, uniq_frame_np

'''
# frame is int64
# trackId is float64 from numpy
# x float64
# y float64
# sceneId is object
# metaId is int64
# frame  trackId       x      y  sceneId  metaId
new_df = pd.DataFrame({
            'frame' : int,
            'trackId' : float,
            'x' : float,
            'y' : float,
            'sceneId' : object,
            'metaId' : int
        }, ignore_index=True)
print( all_id[0].info())
'''
def all_id_converter_to_ynet(all_id):
    all_id = [j for g in all_id for j in g]
    
    with tqdm(total=len(all_id), desc='Progress') as pbar:
        for ind, g in enumerate(all_id):
            last = g.iloc[-1, 0]
            id = g.iloc[-1, 1]
            forms = "00000"
            scenes = [(forms[:-1 * len(str(last))] + str(last)) for g in range(8)]
            all_id[ind]['sceneId'] = scenes
            
            ind_list = []
            for j in range(1,13):
                ind_list.append({'frame_n': last + j,
                                'id' : id,
                                'x' : 0,
                                'y' : 0,
                                'sceneId': (forms[:-1 * len(str(last))] + str(last))
                                })
                '''
                all_id[ind] = all_id[ind].append({'frame_n': last + j,
                                  'id' : id,
                                  'x' : 0,
                                  'y' : 0,
                                  'sceneId': (forms[:-1 * len(str(last))] + str(last))
                                  }, ignore_index=True)
                '''
            all_id[ind] = pd.concat([all_id[ind], pd.DataFrame.from_records(ind_list)], ignore_index=True)
            
            merge_ = all_id[0]
            for ind in range(1, len(all_id)):
                merge_ = pd.concat([merge_, all_id[ind]], ignore_index=True)
            merge_['frame'] = merge_['frame_n']
            merge_['metaId'] = merge_['id']
            merge_['metaId'] = 0
            merge_['trackId'] = merge_['id']
            merge_['trackId'] = merge_['trackId'].astype(float)
            merge_.pop('frame_n')
            merge_.pop('id')
            merge_ = merge_.iloc[:, [3, 5, 0, 1, 2, 4]]
            FairYnet_test_path = '../../ynet/data/FairYnet/'
            merge_.to_pickle(f'{FairYnet_test_path}fair-mot_test.pkl')
            
            pbar.update(1)
    
    print('success (2/2)')

def move_frame_FairYnet_ynet(uniq_frame):
    # insert FairYnet test path on ynet
    FairYnet_test_path = '../../ynet/data/FairYnet/test/'
    frame_path = './frame/'
    forms = "00000"
    exten = ".jpg"
    
    if os.path.exists(FairYnet_test_path):
        shutil.rmtree(FairYnet_test_path, ignore_errors=True)
    os.mkdir(FairYnet_test_path)
    
    str_uniq = [str(g) for g in uniq_frame]
    str_uniq = [(forms[:-1 * len(g)] + g) for g in str_uniq]
    for g in str_uniq:
        os.mkdir('{}{}'.format(FairYnet_test_path, g))
        shutil.copy2("{}{}.jpg".format(frame_path,g), '{}{}/reference.jpg'.format(FairYnet_test_path, g))
    
    print('success (1/2)')

if __name__ == "__main__":
    os.chdir("demo")
    if os.path.exists("frame_ynet"):
        shutil.rmtree("frame_ynet", ignore_errors=True)
    shutil.copytree("frame", "frame_ynet", symlinks=False, ignore=None, copy_function=shutil.copy2, ignore_dangling_symlinks=False, dirs_exist_ok=False)
    
    convertion_csv()
    all_id, uniq_frame = process_data("results.csv")
    move_frame_FairYnet_ynet(uniq_frame)
    all_id_converter_to_ynet(all_id)
