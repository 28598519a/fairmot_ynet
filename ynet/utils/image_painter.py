import cv2
import os
import torch

class painter():
    def __init__(self, data_pred, image, data_test):
        # data -> tensor future_samples
        # image -> image  or image_path
        # name_directory -> data.sceneId.unique()
        self.image = image
        self.data_pred = torch.div(data_pred, 0.25).tolist()
        self.data_test = data_test
        self.data_gt = data_test[:8]
        #self.data_gt_future = data_test[8:]
        self.image_painter()

    def image_painter(self):
        # straight doodle
        scene_int = int(self.data_test.sceneId.unique()[0])
        forms = "00000"
        path_frame_fair = "../FairMOT/demo/frame_ynet/"
        
        # range為obj_div (data_to_pickle.py)
        for ind in range(scene_int, scene_int + 5):
            im_path = f'{path_frame_fair}/{forms[:-1 * len(str(ind))] + str(ind)}.jpg'
            
            if os.path.exists(im_path):
                im = cv2.imread(im_path)
                # cv2.namedWindow('cek')
                gt_points = self.data_gt.iloc[:, 2:4].values.tolist()
                '''
                gt_future_points = self.data_gt_future.iloc[:, 2:4].values.tolist()
                '''
                
                g_past = None
                for g in gt_points:
                    #im = cv2.circle(im, (int(g[0]), int(g[1])), radius=3, color=(0, 0, 255), thickness=-1)
                    if g_past is not None:
                        im = cv2.line(im, (int(g_past[0]), int(g_past[1])), (int(g[0]), int(g[1])), color=(0, 0, 255), thickness=2)
                    g_past = g
                '''
                g_past = None
                for g in gt_future_points:
                    #im = cv2.circle(im, (int(g[0]), int(g[1])), radius=3, color=(0, 0, 255), thickness=-1)
                    if g_past is not None:
                        im = cv2.line(im, (int(g_past[0]), int(g_past[1])), (int(g[0]), int(g[1])), color=(255, 0, 0), thickness=2)
                    g_past = g
                '''
                for g in self.data_pred:
                    s_past = None
                    for j in g:
                        for s in j:
                            #im = cv2.circle(im, (int(s[0]), int(s[1])), radius=3, color=(0, 255, 0), thickness=-1)
                            if s_past is not None:
                                im = cv2.line(im, (int(s_past[0]), int(s_past[1])), (int(s[0]), int(s[1])), color=(0, 255, 0), thickness=2)
                            s_past = s
                    # only  plot first
                    break
                cv2.imwrite(im_path, im)
