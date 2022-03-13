import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import random
import time
import tables as pytables
from typing import Optional
from utils.config import cfg
import os


class SurrealTrain(Dataset):
    def __init__(self, size: Optional[int] = 5000):
        self.ratio_list = [0.02, 0.04, 0.06, 0.08, 0.10]
        self.file_name = '/home/hot/PycharmProjects/non-rigid/dataset/trainset.h5'
        if not os.path.exists(self.file_name):
            self.file_name = '/data/home/scv0495/run/nonrigid/data/trainset.h5'

        self.soft_label = False

        self.size = size

        self.data =self.load_data()
        self.count = self.size

        self.len = len(self.data)
        self.pair_len = self.len//2
        self.L = list(range(0, self.len))

    def update_L(self):
        #if len(self.L)==0:
        if self.count == 0 :
            self.count = self.size
            self.L = list(range(0, self.len))

    def load_data(self):
        f = h5py.File(self.file_name, mode='r')
        return f['xyz2']

    def get_index_pairs(self,index):
        random.seed(index)
        t1 = [self.L.pop(random.randrange(len(self.L))) for _ in range(2)]
        assert (t1[0] != t1[1])
        return t1

    def permuted_transfrom(self, xyz1, index, random_fix=True):
        npoint = xyz1.shape[0]
        I = np.eye(npoint)  # N2xN1
        p = I.copy()
        while (np.array_equal(p, I)):
            if random_fix == True:
                np.random.seed(index)
                np.random.shuffle(p)  # N2xN1

        permuted_xyz1 = np.dot(p, xyz1)  # N2xN1 N1x3 = N2x3

        label = p  # N1xN2
        return label, permuted_xyz1

    def full_permute(self, pointcloud1, pointcloud2, item):
        now = int(time.time())
        np.random.seed(now)
        pointcloud1 = np.random.permutation(pointcloud1)
        np.random.seed(now)
        pointcloud2 = np.random.permutation(pointcloud2)
        corr_matrix_label, permuted_pointcloud = self.permuted_transfrom(pointcloud1, now)
        return corr_matrix_label, permuted_pointcloud, pointcloud2

    def __getitem__(self, item):

        self.update_L()
        t = self.get_index_pairs(item)
        index1 = t[0]
        index2 = t[1]
        assert (index1 != index2)
        input1_ori = np.array(self.data[index1])
        input2_ori = np.array(self.data[index2])
        corr_matrix_label, permuted_input1, input2 = \
            self.full_permute(input1_ori, input2_ori, item)

        input1 = torch.from_numpy(permuted_input1).float()
        corr_matrix_label = torch.from_numpy(corr_matrix_label).float()
        input2 = torch.from_numpy(input2).float()
        self.count = self.count - 1

        return corr_matrix_label, input1, input2

    def __len__(self):
        return self.size


class SurrealTest(Dataset):
    def __init__(self):
        # self.ratio_list = [0.02, 0.04, 0.06, 0.08, 0.10]
        self.file_name = 'dataset/testset.h5'
        self.soft_label = False
        self.data = self.load_data()
        self.pair_len = len(self.data)

    def load_data(self):
        f = h5py.File(self.file_name)
        return f['data']

    def __getitem__(self, item):

        point1 = self.data[item]['src_flat'].reshape(-1, 3)
        point2 = self.data[item]['tgt_flat'].reshape(-1, 3)
        per = self.data[item]['label_flat'].reshape(-1, 1024)

        return per, point1, point2 #, A1_gt, A2_gt

    def __len__(self):
        return self.pair_len



class SHRECTest(Dataset):
    def __init__(self):
        self.file_path = '/home/hot/PycharmProjects/non-rigid/dataset/SHREC/'
        self.ratio_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        self.src = np.load(self.file_path + 'source_shrec.npy')
        self.tgt = np.load(self.file_path + 'target_shrec.npy')
        self.label = np.load(self.file_path + 'label_shrec.npy')
        self.sl_002 = np.load(self.file_path + 'sl_002.npy')
        self.sl_004 = np.load(self.file_path + 'sl_004.npy')
        self.sl_006 = np.load(self.file_path + 'sl_006.npy')
        self.sl_008 = np.load(self.file_path + 'sl_008.npy')
        self.sl_01 = np.load(self.file_path + 'sl_01.npy')
        self.sl_012 = np.load(self.file_path + 'sl_012.npy')
        self.sl_014 = np.load(self.file_path + 'sl_014.npy')
        self.sl_016 = np.load(self.file_path + 'sl_016.npy')
        self.sl_018 = np.load(self.file_path + 'sl_018.npy')
        self.sl_02 = np.load(self.file_path + 'sl_02.npy')
        self.pair_len = len(self.src)

    def __getitem__(self, item):

        point1 = self.src[item].astype(np.float32)
        point2 = self.tgt[item].astype(np.float32)
        per = self.label[item].astype(np.float32)
        sl_002 = self.sl_002[item].astype(np.float32)
        sl_004 = self.sl_004[item].astype(np.float32)
        sl_006 = self.sl_006[item].astype(np.float32)
        sl_008 = self.sl_008[item].astype(np.float32)
        sl_01 = self.sl_01[item].astype(np.float32)
        sl_012 = self.sl_012[item].astype(np.float32)
        sl_014 = self.sl_014[item].astype(np.float32)
        sl_016 = self.sl_016[item].astype(np.float32)
        sl_018 = self.sl_018[item].astype(np.float32)
        sl_02 = self.sl_02[item].astype(np.float32)

        sl = [sl_002,sl_004,sl_006,sl_008,sl_01,sl_012,sl_014,sl_016,sl_018,sl_02]

        return per, point1, point2, sl

    def __len__(self):
        return self.pair_len


class SHRECTest_witout(Dataset):
    def __init__(self):
        self.file_path = '/home/hot/PycharmProjects/non-rigid/dataset/SHREC/'    # our for paint
        if not os.path.exists(self.file_path):
            self.file_path = '/data/home/scv0495/run/nonrigid/data/SHREC/'
        self.src = np.load(self.file_path + 'source_shrec.npy')
        self.tgt = np.load(self.file_path + 'target_shrec.npy')
        self.label = np.load(self.file_path + 'label_shrec.npy')
        self.pair_len = len(self.src)

    def __getitem__(self, item):
        point1 = self.src[item].astype(np.float32)
        point2 = self.tgt[item].astype(np.float32)
        per = self.label[item].astype(np.float32)

        return per, point1, point2

    def __len__(self):
        return self.pair_len


if __name__ == '__main__':
    train = SurrealTrain()
    for data in train:
        break
    per = np.array(data[0])
    point1 = np.array(data[1])
    point2 = np.array(data[2]) + 0.5
    point1 = np.matmul(per.T, point1)
    import open3d as o3d

    lines = []
    k = 100
    point = []
    for i in range(k):
        lines.append([2 * i, 2 * i + 1])
        point.append(point1[i].tolist())
        point.append(point2[i].tolist())

    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(point),
        lines=o3d.utility.Vector2iVector(lines),
    )
    # line_set.colors = o3d.utility.Vector3dVector(colors)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point2)
    o3d.visualization.draw_geometries([line_set, pcd1, pcd2])

