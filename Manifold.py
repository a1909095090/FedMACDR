import torch
import torch.nn as nn
import torch.nn.functional as F
from userGraph import *
import  os
from models.gnn.gnn_model import GNN
from utils import train_utils
num_mlp_layers=3
emb_size = 128  # Network embedding dimension
hidden_size = 64  # Lantent dimension
dropout_rate = 0.3  # Dropout rate p
leakey = 0.1  # Hyperparameter of LeakyReLU
comment_dim=64
import numpy as np

torch.autograd.set_detect_anomaly(True)
class Discriminator(nn.Module):
    def __init__(self, emb_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(emb_size * 2, 1)

    def forward(self, input1, input2):
        if input1.dim() <= input2.dim():
            smaller, larger = input1, input2
        else:
            smaller, larger = input2, input1
        if input1.dim() != input2.dim():
            if smaller.dim() == 2:
                # (batch_size, 1, emb_size)
                smaller = smaller.view(smaller.size()[0], 1, -1) #将smaller 的维度从（smaller.size()[0], smaller.size()[1] ）转化为： smaller.size()[0],1, smaller.size()[1]
                # (batch_size, num_neg, emb_size) in training mode,
                # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
                smaller = smaller.repeat(1, larger.size()[1], 1) #repeat() 方法用于重复张量中的元素以形成一个新的张量。这个方法允许你指定每个维度的重复次数
                #将smaller的维度从（smaller.size()[0], 1,smaller.size()[1] ）转化为： smaller.size()[0], larger.size()[1], smaller.size()[1]

            # if `smaller` is graph representation `z_s`
            elif smaller.dim() == 1:
                smaller = smaller.view(1, -1)  # (1, emb_size)
                # (batch_size, emb_size)
                smaller = smaller.repeat(larger.size()[0], 1)

        input = torch.cat([smaller, larger], dim=-1)
        output = self.fc(input)
        return output

def get_score(input1,input2,mode="train"):

    if input1.dim() != input2.dim():
        #
        #     # (batch_size, 1, emb_size)
        # input1 = input1.view(input1.size()[0], 1,
        #                            -1)  # 将smaller 的维度从（smaller.size()[0], smaller.size()[1] ）转化为： smaller.size()[0],1, smaller.size()[1]
        #     # (batch_size, num_neg, emb_size) in training mode,
        #     # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
        # input1 = input1.repeat(1, input2.size()[1], 1)  # repeat() 方法用于重复张量中的元素以形成一个新的张量。这个方法允许你指定每个维度的重复次数
        #     # 将smaller的维度从（smaller.size()[0], 1,smaller.size()[1] ）转化为： smaller.size()[0], larger.size()[1], smaller.size()[1]
        input1 = input1.unsqueeze(1)  # 形状变为[1024, 1, 128]
        output = torch.bmm(input1, input2.transpose(2, 1))  # 转置negv的最后两个维度

        # 结果的形状将是[1024, 1, 99]
        # 如果你想要移除中间的维度，可以使用squeeze
        output = output.squeeze(1)  # 形状变为[1024, 99]
    else:
        output=torch.mul(input1,input2)
    # gamma=torch.sum(output,dim=1)

    return output


# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x=self.fc1(x)
        x = self.relu(self.fc2(x))
        x=self.fc3(x)
        return x
class Manifold(nn.Module):
    def __init__(self,args,domain):
        super(Manifold, self).__init__()
        self.layer_number = num_mlp_layers
        self.uG=UserGraph(args, domain)
        # self.user_emb=users_emb
        # self.user_emb.requires_grad = False
        self.model = GNN(self.uG.num_users, self.uG.num_items,args)
        # 冻结参数


        # self.user_emb=self.model.user_item_emb
        # self.users_emb=None
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.D=self.uG.get_D()
        self.L=self.uG.get_Laplacian()
        self.D=torch.Tensor(self.D)
        self.D.requires_grad=False
        self.L=torch.Tensor(self.L)
        self.L.requires_grad=False
        self.lamda=1
        self.num_users=self.uG.num_users
        self.users =np.arange(  self.num_users)
        self.emb_size=emb_size
        self.comment_dim=comment_dim
        # self.encoder = torch.rand(self.emb_size, self.comment_dim)
        self.encoder=MLP(emb_size,hidden_size,comment_dim)
        # self.decoder = torch.rand(self.comment_dim, self.emb_size)
        self.decoder = MLP(comment_dim,hidden_size,emb_size)
        self.discri = Discriminator(emb_size).to(self.device)

        self.params = list(self.encoder.parameters()) +list(self.decoder.parameters()) + list(self.discri.parameters())

        self.optimizer = train_utils.get_optimizer(
            args.optimizer, self.params, args.malr)
        self.global_emb=nn.Embedding(self.uG.num_users,comment_dim)
        self.step = 0
        # self.decoder = nn.Sequential(
        #     nn.Linear(comment_dim, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),  # Dropout 层，丢弃率 50%
        #     nn.Linear(hidden_size, emb_size)
        # )
        # for i in range(self.layer_number):
        #     num_features = (emb_size if i == 0 else hidden_size)
        #     num_hidden = (comment_dim if i == self.layer_number - 1
        #                   else hidden_size)
        #     self.encoder.append(nn.Linear(num_features,num_hidden))

        self.dropout = dropout_rate

        # print(model.user_item_emb.weight)
        # Food_uemb=model.user_item_emb.weight[:1898]

        # self.opt = optim.Adam(self.Mamodel.parameters(), lr=0.1,weight_decay=0.001)

    # def init_users_emb(self):
    #     if self.users_emb is None:
    #         self.users_emb=self.model.user_item_emb
    #         self.users_emb.requires_grad=False

    def forward(self):
        # self.init_users_emb()
        u=self.get_index_select_umb(self.users)
        output=self.encoder(u)
        return output
    def get_index_select_umb(self,users):
        users = torch.LongTensor(users).to(self.device)

        u = self.model.my_index_select_embedding(self.model.user_item_emb, users)
        return u
    def get_index_select_items_mb(self, items):
        # v=torch.LongTensor(items).to(self.device)
        v= self.model.my_index_select_embedding(self.model.user_item_emb, self.model.num_users+items)
        return v
    def get_users_maemb(self,users):
        # self.init_users_emb()
        u=self.get_index_select_umb(users)
        output = self.encoder(u)
        return output
    def get_target_domain_ue(self,users):

        users_maemb=self.get_users_maemb(users)
        target_domain_ue=self.decoder(users_maemb)
        return target_domain_ue

    @ staticmethod
    def flatten(source):
        return torch.cat([value.flatten() for value in source])

    def prox_reg(self, params1, params2, mu):
        params1_values, params2_values = [], []
        # Record the model parameter aggregation results of each branch
        # separately
        for branch_params1, branch_params2 in zip(params1, params2):
            branch_params2 = [branch_params2[key]
                              for key in branch_params1.keys()]
            params1_values.extend(branch_params1.values())
            params2_values.extend(branch_params2)

        # Multidimensional parameters should be compressed into one dimension
        # using the flatten function
        s1 = self.flatten(params1_values)
        s2 = self.flatten(params2_values)
        return mu/2 * torch.norm(s1 - s2)



    def loss(self):
        # print(self.user_emb)

        users=np.arange(self.num_users)
        # users = torch.LongTensor(users).to(self.device)
        # u_emb=self.get_index_select_umb(users)
        users = torch.LongTensor(users).to(self.device)
        u = self.model.my_index_select_emb(self.model.U_and_V, users)
        # u_emb=self.model.my_index_select_embedding(self.model.U_and_V,users)
        F = self.encoder(u)
        # F=torch.mul(self.encoder)
        Ft=F.T
        # print(Ft.shape,self.L.shape,self.D.shape)
        criterion = nn.L1Loss()
        tmp1=torch.matmul(Ft,self.L)
        tmp2=torch.matmul(tmp1,F)
        tmp3=torch.matmul(Ft,self.D)
        tmp4=torch.matmul(tmp3,F)
        # print(tmp4.shape)
        I = torch.eye(self.comment_dim, dtype=torch.float32,requires_grad=False)  # 创建一个 nxn 的单位阵，数据类型为
        la= tmp2+self.lamda*(I-tmp4)
        zero_tensor=torch.zeros_like(la,requires_grad=False)
        # user_emb=self.users_emb(self.users)
        encoder_loss= torch.norm(la, p=2)
        decoder_loss=torch.norm(self.decoder(F)-u, p=2)
        # encoder_loss = criterion(la, zero_tensor)
        # encoder_loss_sum=torch.norm()

        # print(user_emb)
        # decoder_loss=(self.decoder(F)-user_emb)**2
        # declosssum=torch.sum(decoder_loss)
        loss_sum = encoder_loss+decoder_loss
        return loss_sum,encoder_loss,decoder_loss
    def get_Maloss(self,users,interactions):
        interactions = [torch.LongTensor(x).to(
            self.device) for x in interactions]
        _,_,loss=self.loss()
        users = torch.LongTensor(users).to(self.device)
        items, neg_items = interactions
        # u: (batch_size, emb_size)
        # v: (batch_size, emb_size)
        # neg_v: (batch_size, num_neg, emb_size)
        _, v, neg_v = self.model(users, items, neg_items)
        u=self.get_shared_param(users)



        return
    def get_shared_param(self):
        # users = torch.LongTensor(users).to(self.device)

        shared_param = self.forward()
        return shared_param

    def train_batch(self, users, interactions,
                    global_params=None):
        """Trains the model for one batch.

        Args:
            users: Input user IDs.
        """

        self.optimizer.zero_grad()
        u_emb = self.get_index_select_umb(users)
        users = torch.LongTensor(users).to(self.device)


        interactions = [torch.LongTensor(x).to(
            self.device) for x in interactions]
        items, neg_items = interactions

        # _, v, neg_v = self.model(users, items, neg_items)
        # u, v, neg_v = self.model(users, items, neg_items)
        v=self.get_index_select_items_mb(items)

        neg_v=self.get_index_select_items_mb(neg_items)
        # print("v:", v.shape, "negv:", neg_v.shape)
        # F = self.encoder(u_emb)
        F=self.forward()
        # F = self.encoder(u)

        # u =self.decoder(self.encoder(u_emb))

        loss,encoder_loss,_  = self.loss()
        if global_params  is not None:
            self.global_emb.weight= nn.Parameter(global_params)
            self.global_emb.requires_grad = False
            encoder_loss = encoder_loss+ torch.sum((F-self.global_emb.weight)**2)
            u = self.decoder(self.global_emb(users))
            loss1 = self.gnn_loss_fn(u, v, neg_v)

            encoder_loss = encoder_loss+ loss1

        self.optimizer.zero_grad()
        encoder_loss.backward()
        # encoder_loss.backward()
        self.optimizer.step()


        self.step += 1
        return encoder_loss
    def test_batch(self, users, interactions):
        """Tests the model for one batch.

        Args:
            users: Input user IDs.
            interactions: Input user interactions.
        """



        users = torch.LongTensor(users).to(self.device)
        interactions = [torch.LongTensor(x).to(
            self.device) for x in interactions]

        # items: (batch_size, )
        # neg_items: (batch_size, num_test_neg)
        items, neg_items = interactions
        # all_items: (batch_size, num_test_neg + 1)
        # Note that the elements in the first column are the positive samples.
        # all_items = torch.hstack([items.reshape(-1, 1), neg_items])
        all_items = torch.hstack([items.reshape(-1, 1), neg_items])


        u, v = self.model(users, all_items)
        # u = self.get_index_select_umb(users)
        # u =self.decoder(self.encoder(u))
        u=self.global_emb(users)
        u = self.decoder(u)
        # u=self.decoder(u)
        # print("u:",u)
            # if "Ma" in self.method:
            #     u=

        # result = self.discri(u, v)
        result=get_score(u,v)
        print(v.shape)
        # (batch_size, num_test_neg + 1)
        # result = result.view(result.size()[0],
        #                      result.size()[1])

        pred = []
        for score in result:
            # score:  (num_test_neg + 1)
            # Note that the first one is the positive sample.
            # `(-score).argsort().argsort()` indicates where the elements at
            # each position are ranked in the list of logits in descending
            # order (since `argsort()` defaults to ascending order, we use
            # `-score` here). Since the first one is the positive sample,
            # then `...[0].item()` indicates the ranking of the positive
            # sample.
            rank = (-score).argsort().argsort()[0].item()
            pred.append(rank + 1)  # `+1` makes the ranking start from 1

        return pred

    def gnn_loss_fn(self, u, v, neg_v):
        # pos_score = self.discri(u, v)  # (batch_size, )
        # neg_score = self.discri(u, neg_v)

        pos_score = get_score(u, v)  # (batch_size, )
        neg_score = get_score(u, neg_v)

        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
            # - F.logsigmoid(-neg_score).mean(dim=1).mean()

        return loss

    def update_lr(self, new_lr):
        train_utils.change_lr(self.optimizer, new_lr)





