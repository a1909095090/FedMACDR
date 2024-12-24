# -*- coding: utf-8 -*-
"""Local Graph class.
"""
import numpy as np
import scipy.sparse as sp
import torch
import os
import pickle

# -*- coding: utf-8 -*-
"""Customized dataset.
"""
import os
import pickle
import queue
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

class UserGraph(Dataset):
    """A customized dataset reading and preprocessing data of a certain domain
    from ".txt" files.
    """
    data_dir = "data"        # 数据目录在data里
    prep_dir = "prep_data"   # 预训练数据目录在prep_data里

    def __init__(self, args, domain):
        # assert model =="FedMACDR"
        self.gamma=0.5
        self.file_name=domain+"_users_graph.npy"
       # 参数δ通常被称为“带宽”（bandwidth）或“尺度参数”（scaleparameter）。这个参数控制了高斯函数的扩散程度，即它决定了函数曲线的宽度。
       # 当δ较小时，函数值在 x = y附近迅速减小，形成一个尖锐的峰；当δ较大时，函数值在x=y附近的变化较为平缓，形成一个较宽的峰。
        self.args = args
        self.mode="train"
        self.domain = domain
        # self.model = model
        # self.mode = mode
        self.max_hop_count=99999999
        self.W = None
        self.D = None
        # 数据集路径
        self.dataset_dir = os.path.join(self.data_dir, self.domain + "_"
                                        + "FKCB")
        self.num_users,self.num_items, self.items_users= self.get_items_users(self.dataset_dir)

        self.users_matrix=None
        self.get_users_adjacency_matrix()



    def get_items_users(self, dataset_dir):
        """
        """
        items_users = defaultdict(list)
        users_items = defaultdict(list)
        num_users=int(0)
        num_itemss = int(0)
        with open(os.path.join(dataset_dir, "num_users.txt"),
                  "rt", encoding="utf-8") as infile:
            num_users = int(infile.readline()) # 项目数。
        with open(os.path.join(dataset_dir, "num_items.txt"),
                  "rt", encoding="utf-8") as infile:
            num_items = int(infile.readline()) # 项目数。
        with open(os.path.join(self.dataset_dir,
                               "%s_data.txt" % "train"), "rt",
                  encoding="utf-8") as infile:

            for line in infile.readlines():
                user, item = line.strip().split("\t")
                user, item = int(user), int(item)
                items_users[item].append(user)
                users_items[user].append(item)

        print(" Successfully load UserGraph  %s %s data!" % (self.domain, self.mode))

        return  num_users,num_items,items_users

    def get_users_adjacency_matrix(self):
        if self.users_matrix is not None:
            return self.users_matrix

        if os.path.exists(self.file_name):
            self.users_matrix= np.load(self.file_name)

            return self.users_matrix


        self.users_matrix=np.full((self.num_users,self.num_users),self.max_hop_count)
        users_adjtable=defaultdict(list)
        for item in self.items_users:
            for user1 in self.items_users[item]:
                for user2 in self.items_users[item]:
                    if user1==user2:
                        self.users_matrix[user1][user1]=0
                        continue
                    users_adjtable[user1].append(user2)
                    users_adjtable[user2].append(user1)
        for user in range(self.num_users):
            self.bfs(user,users_adjtable)
        np.save(self.file_name,self.users_matrix)
        return  self.users_matrix
    def bfs(self,node,users_adjtable):
        q = queue.Queue()
        dist=0
        q.put(node)
        while q.empty() == False and dist<self.max_hop_count:
            size=q.qsize()
            dist =dist+2
            while size>0:
                top=q.get()
                for user in users_adjtable[top]:
                    if user == top:
                        continue
                    if dist<self.users_matrix[node][user] or self.users_matrix[node][user]==0:
                        q.put(user)
                        self.users_matrix[node][user]=dist
                        self.users_matrix[user][node]=dist
                size=size-1
    def get_W(self):
        if self.W is None:
            self.W=np.zeros((self.num_users,self.num_users))
            n =self.num_users
            for row in range(n):
                for col in range(n):
                    if row != col and self.users_matrix[row][col]!=self.max_hop_count:
                        self.W[row][col]=np.exp(-self.users_matrix[row][col]/(self.gamma**2))

        return self.W
    def get_D(self):
        self.get_W()
        if self.D is None:
            self.D=np.zeros((self.num_users,self.num_users))
            for user in range(self.num_users):
                self.D[user][user]=sum(self.W[user])
        return self.D
    def get_Laplacian(self):

        return self.D-self.W




