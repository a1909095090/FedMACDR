# -*- coding: utf-8 -*-
"""Local Graph class.
"""
import numpy as np
import scipy.sparse as sp
import torch
import os
import pickle


def normalize(mx):
    """Row-normalize sparse matrix.
    """
    rowsum = np.array(mx.sum(1)) # 矩阵按行求和得到度向量。
    r_inv = np.power(rowsum, -1).flatten() # 求倒数（-1次幂）。
    r_inv[np.isinf(r_inv)] = 0.           #将inf值的数据置为0
    r_mat_inv = sp.diags(r_inv)           #生成对角矩阵，以稀疏矩阵 coo（值，行，列）的形式存储。
    mx = r_mat_inv.dot(mx)                #D-1M
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32) #以稀疏矩阵 coo（值，行，列）的形式存储，并将元素类型转换为float32
    indices = torch.from_numpy(  #在NumPy中，np.vstack 函数用于垂直堆叠（按行）两个或多个数组。
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    #row = np.array([0, 1, 2]),col = np.array([2, 0, 1]) np.vstack((srow,col))的结果为[[0 1 2]，[2 0 1]]
    #torch.from_numpy 将numpy转化为pytorch
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class LocalGraph(object):
    """A local graph data structure class reading training data of a certain
    domain from ".txt" files, and preprocess it into a local graph.
    一个局部图数据结构类，从“.txt”文件中读取某个领域的训练数据，并将其预处理成局部图。
    """
    data_dir = "data"
    prep_dir = "prep_data"

    def __init__(self, args, domain, num_users, num_items, model="GNN",
                 load_prep=True):
        assert (model in ["HL_HF", "HF", "DHCF", "GNN", "PPDM"])
        self.args = args
        self.domain = domain
        self.model = model
        # 数据集路径
        self.dataset_dir = os.path.join(self.data_dir, self.domain + "_"
                                        + "".join([domain[0] for domain
                                                   in self.args.domains]))
        self.raw_data = self.read_train_data(self.dataset_dir)
        self.num_users = num_users
        self.num_items = num_items
        ##########################################################################
        if self.model == "HL_HF":
            self.UU_adj, self.VV_adj, self.M, self.perturb_UU_adj \
                = self.preprocess(self.raw_data, load_prep)
        elif self.model in ["HF", "DHCF"]:
            self.UU_adj, self.VV_adj = self.preprocess(
                self.raw_data, load_prep)
        else:
            self.all_adj = self.preprocess(self.raw_data, load_prep)

    def read_train_data(self, dataset_dir):
        """读取训练集"""
        with open(os.path.join(dataset_dir, "train_data.txt"),
                  "rt", encoding="utf-8") as infile:
            train_data = []
            for line in infile.readlines():
                user, item = line.strip().split("\t")
                train_data.append((int(user), int(item)))
        return train_data

    def preprocess(self, data, load_prep):
        if not os.path.exists(os.path.join(self.dataset_dir, self.prep_dir)):
            os.makedirs(os.path.join(self.dataset_dir, self.prep_dir))

        self.prep_graph_data_path = os.path.join(
            self.dataset_dir, self.prep_dir, "%s_graph_data.pkl" % self.model)

        if os.path.exists(self.prep_graph_data_path) and load_prep:
            with open(os.path.join(self.prep_graph_data_path), "rb") as infile:
                graph_data = pickle.load(infile)
            print("Successfully load preprocessed %s graph data!"
                  % self.domain)
        else:
            UV_edges = []
            VU_edges = []
            all_edges = []
            for (user, item) in data:
                UV_edges.append([user, item])
                VU_edges.append([item, user])

                all_edges.append([user, item + self.num_users]) # 将物品id加上用户数
                all_edges.append([item + self.num_users, user])

            UV_edges = np.array(UV_edges)
            VU_edges = np.array(VU_edges)
            all_edges = np.array(all_edges)
            UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]),
                                    (UV_edges[:, 0], UV_edges[:, 1])),
                                   shape=(self.num_users, self.num_items),
                                   dtype=np.float32)
            # 存储稀疏矩阵sp.coo_matrix 是 SciPy 库中的一个函数，用于创建稀疏矩阵的一种表示形式：坐标格式（COO，Coordinate Format）。
            # 在坐标格式中，每个非零元素的位置和值都单独存储。这种表示形式对于构造大型稀疏矩阵非常有用，因为它可以节省内存并提高计算效率。
            VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]),
                                    (VU_edges[:, 0], VU_edges[:, 1])),
                                   shape=(self.num_items, self.num_users),
                                   dtype=np.float32)
            all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]),
                                     (all_edges[:, 0], all_edges[:, 1])),
                                    shape=(self.num_users + self.num_items,
                                           self.num_users + self.num_items),
                                    dtype=np.float32)

            if self.model == "HL_HF":
                UU_adj, M = self.fedhcdr_construct_hyper_A_from_H(
                    UV_adj, n_rw=self.args.n_rw)
                VV_adj = self.fedhcdr_construct_hyper_A_from_H(VU_adj)
            elif self.model in ["HF", "DHCF"]:
                UU_adj = self.hf_construct_hyper_A_from_H(UV_adj)
                VV_adj = self.hf_construct_hyper_A_from_H(VU_adj)
            else:
                all_adj = normalize(all_adj)

            if self.model in ["HL_HF", "HF", "DHCF"]:
                UU_adj = sparse_mx_to_torch_sparse_tensor(UU_adj)
                VV_adj = sparse_mx_to_torch_sparse_tensor(VV_adj)
            else:
                all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)

            if self.model == "HL_HF":
                # False is masked, True is not masked
               # # torch.bernoulli从伯努利分布中抽样。伯努利分布是一种离散概率分布，随机变量取值为1的概率为p，取值为0的概率为1 - p。
                mask = torch.bernoulli(torch.full(
                    UU_adj.shape, 1 - self.args.drop_edge_rate)).to(torch.bool)

                # # For edge (u, v), the degree of v is smaller, the weight of
                # # edge (u, v) is larger, and it is more likely to be dropped
                # UU_adj_dense = UU_adj.to_dense()
                # edge_weights = UU_adj_dense.where(
                #     UU_adj_dense < 0.7, torch.ones_like(UU_adj_dense) * 0.7)
                # # mask = torch.bernoulli(edge_weights
                # #                        * self.args.drop_edge_rate)\
                # #     .to(torch.bool)
                # mask = torch.bernoulli(edge_weights).to(torch.bool)

                # For `torch.masked_fill`, True is masked, False is not masked, so we need to use `~mask`
                # torch.masked_fill()用于根据给定的掩码（mask）将张量（tensor）中的某些元素替换为指定的值。掩码张量中为
                # True的位置对应的元素将被替换，而为False 的位置保持不变。
                perturb_UU_adj = torch.masked_fill(
                    input=UU_adj.to_dense(), mask=~mask, value=0).to_sparse()

            with open(self.prep_graph_data_path, "wb") as infile:
                if self.model == "HL_HF":
                    graph_data = [UU_adj, VV_adj, M, perturb_UU_adj]
                elif self.model in ["HF", "DHCF"]:
                    graph_data = [UU_adj, VV_adj]
                else:
                    graph_data = all_adj
                pickle.dump(graph_data, infile)
            print("Successfully preprocess %s graph data!" %
                  (self.domain))
        return graph_data

    def fedhcdr_construct_hyper_A_from_H(self, H, n_rw=-1):
        """
        H ：表示点是否属于超图
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        def m_power(mx, n):
            """Row-normalize sparse matrix.
            """
            r_inv = np.power(mx, n)
            r_inv[np.isinf(r_inv)] = 0.
            return r_inv

        D_u = H.sum(axis=1).reshape(1, -1) #reshape(1, -1) 中的 1 表示新的张量应该有一个维度为1。
        D_v = H.sum(axis=0).reshape(1, -1) #-1 是一个特殊的值，表示该维度的大小应该自动计算，以便保持张量中元素的总数不变。
        # The vertex has smaller degree has larger weights
        D_v_sum = D_v.sum()
        P = 1 - D_v / D_v_sum
        # D_v = np.log(D_v)
        # # The vertex has smaller degree has larger weights
        # P = (D_v.max() - D_v) / (D_v.max() - D_v.mean())

        # Note that multiply is point-wise multiplication
        # tilde_D_u用于保证用户超图邻接矩阵Auk的对角线被0填充，即用户超图中不存在自环 原论文6页公式5
        tilde_D_u = H.multiply(P).dot(m_power(D_v.reshape(-1, 1), -1))
        tilde_D_u = sp.diags(np.array(tilde_D_u).flatten()) #生成tilde_D_u

        temp1 = (H.transpose().multiply(m_power(D_u, -0.5))).transpose()
        temp2 = temp1.transpose()
        temp3 = tilde_D_u.multiply(m_power(D_u, -1)).transpose()
        A = temp1.multiply(P).multiply(m_power(D_v, -1)).dot(temp2) \
            - temp3.multiply(m_power(D_u, -1))

        if n_rw != -1:
            M = H.multiply(m_power(D_v, -1)).multiply(P)\
                .dot(H.transpose()).multiply(m_power(D_u, -1)).transpose()\
                - tilde_D_u.multiply(m_power(D_u, -1)).transpose()
            SE = [torch.from_numpy(M.diagonal()).float()]
            M_power = M
            for _ in range(n_rw - 1):
                M_power = M_power.dot(M)
                SE.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw = torch.stack(SE, dim=-1)
            return A, SE_rw
        else:
            return A

    def hf_construct_hyper_A_from_H(self, H):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        def m_power(mx, n):
            """Row-normalize sparse matrix.
            """
            r_inv = np.power(mx, n)
            r_inv[np.isinf(r_inv)] = 0.
            return r_inv

        D_u = H.sum(axis=1).reshape(1, -1)
        D_v = H.sum(axis=0).reshape(1, -1)
        temp1 = (H.transpose().multiply(m_power(D_u, -0.5))).transpose()
        temp2 = temp1.transpose()
        A = temp1.multiply(m_power(D_v, -1)).dot(temp2)

        return A
