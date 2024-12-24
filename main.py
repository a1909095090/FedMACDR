# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import random
import argparse
import torch
from trainer import ModelTrainer
import logging
from client import Client
from server import Server
from utils.data_utils import load_ratings_dataset, load_graph_dataset, \
    init_clients_weight
from utils.io_utils import save_config, ensure_dir
from fl import run_fl
from userGraph import *
from Manifold  import *
import torch.optim as optim

def arg_parse():
    parser = argparse.ArgumentParser()

    # Dataset part
    parser.add_argument(dest="domains", metavar="domains", nargs="*",

                        help="`Food Kitchen Clothing Beauty` or "
                        "`Movies Books Games` or `Sports Garden Home`")
    parser.add_argument("--load_prep", dest="load_prep", action="store_true",
                        default=False,
                        help="Whether need to load preprocessed the data. If "
                        "you want to load preprocessed data, add it")

    # Training part
    parser.add_argument("--method", type=str, default="FedHCDR",
                        help="method, possible are `FedHCDR` (ours), `FedHF`, "
                        "`LocalHF`, `FedDHCF`, `LocalDHCF`, `FedMF`, "
                        "`LocalMF`, `FedGNN`, `LocalGNN`, `LocalP2FCDR`, "
                        "`FedP2FCDR`, `FedPPDM`, `LocalPPDM`")
    parser.add_argument("--log_dir", type=str,
                        default="log", help="directory of logs")
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--num_round", type=int, default=40,
                        help="Number of total training rounds.")
    parser.add_argument("--local_epoch", type=int, default=3,
                        help="Number of local training epochs.")
    parser.add_argument("--optimizer", choices=["sgd", "adagrad", "adam",
                                                "adamax"], default="adam",
                        help="Optimizer: sgd, adagrad, adam or adamax.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Applies to sgd and adagrad.")  # 0.001
    parser.add_argument("--lr_decay", type=float, default=0.9,
                        help="Learning rate decay rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--decay_epoch", type=int, default=10,
                        help="Decay learning rate after this epoch.")
    parser.add_argument("--batch_size", type=int,
                        default=1024, help="Training batch size.")
    # parser.add_argument("--seed", type=int, default=42) #
    parser.add_argument("--seed", type=int, default=43)  #
    parser.add_argument("--eval_interval", type=int,
                        default=1, help="Interval of evalution")
    parser.add_argument("--frac", type=float, default=1,
                        help="Fraction of participating clients")
    parser.add_argument("--mu", type=float, default=0,
                        help="hyper parameter for FedProx")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoint", help="Checkpoint Dir")
    parser.add_argument("--model_id", type=str, default=str(int(time.time())),
                        help="Model ID under which to save models.")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--es_patience", type=int,
                        default=5, help="Early stop patience.")
    parser.add_argument("--ld_patience", type=int, default=1,
                        help="Learning rate decay patience.")

    # Hypergraph signal decoupling arguments for FedHCDR method (ours)
    # 2.0 for FKCB, 3.0 for SCEC, and 1.0 for SGHT is the best
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Coefficient of local-global bi-directional "
                        "knowledge transfer loss (MI) for FedHCDR method "
                        "(ours). 2.0 for FKCB, 3.0 for SCEC, and 1.0 for SGHT "
                        "is the best")
    parser.add_argument("--n_rw", type=int, default=128,
                        help="Steps of hypergraph random walk for FedHCDR "
                        "method (ours). 128 is the best")
    parser.add_argument("--wait_round", type=int, default=0,
                        help="Rounds to wait before performing local-global "
                        "bi-directional knowledge transfer")

    # Contrastive arguments for FedHCDR method (ours)
    # 2.0 for FKCB, 1.0 for SCEC, and 3.0 for SGHT is the best
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Coefficient of hypergraph contrastive loss for "
                        "FedHCDR method (ours). 2.0 for FKCB, 1.0 for SCEC, "
                        "and 3.0 for SGHT is the best")
    parser.add_argument("--drop_edge_rate", type=float, default=0.05,
                        help="Rate of edges being dropped in graph "
                        "perturbation for FedHCDR method (ours). 0.05 is the "
                        "best")

    args = parser.parse_args()
    assert (args.method in ["FedHCDR", "FedHF", "LocalHF",
                            "FedDHCF", "LocalDHCF", "FedMF",
                            "LocalMF", "FedGNN", "LocalGNN",
                            "FedPriCDR", "LocalPriCDR",
                            "FedP2FCDR", "LocalP2FCDR",
                            "FedPPDM", "LocalPPDM"])
    return args


def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed) #为cpu设置随机数
    # 为CPU中设置种子，生成随机数：
    # torch.manual_seed(number)
    # 为特定GPU设置种子，生成随机数：
    # torch.cuda.manual_seed(number)
    # 为所有GPU设置种子，生成随机数：
    # torch.cuda.manual_seed_all(number)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def init_logger(args):
    """Init a file logger that opens the file periodically and write to it.
    """
    log_path = os.path.join(args.log_dir,
                            "domain_" + "".join([domain[0] for domain
                                                 in args.domains]))

    ensure_dir(log_path, verbose=True) #确保文件存在，如果不存在则创建。

    model_id = args.model_id if len(args.model_id) > 1 else "0" + args.model_id
    log_file = os.path.join(log_path, args.method + "_" + model_id + ".log")  # 日志文件保存在log_path目录下。日志文件名格式为 方法_时间整数.log

    logging.basicConfig(
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode="w+"
    )

    console = logging.StreamHandler() # 创建一个StreamHandler对象，它是一个日志处理器，用于将日志信息输出到流，通常是控制台。
    console.setLevel(logging.INFO)    # 设置console处理器的日志级别为INFO。
    formatter = logging.Formatter("%(asctime)s | %(message)s") #创建一个Formatter对象，定义了控制台输出的日志格式。
    console.setFormatter(formatter)                            #将创建的格式化器设置到console处理器上。
    logging.getLogger("").addHandler(console)                  #获取一个名为""（空字符串）的日志记录器，并添加console处理器到这个记录器上。
                                                               # 这意味着所有发送到这个记录器的日志消息都会通过console处理器输出到控制台。


def main():
    # 加载参数
    args = arg_parse()
    # 设置cpu，gpu ，numpy 随机数种子
    seed_everything(args)
    # 设置日志文件
    init_logger(args)

    # 加载数据集 里边有所有域的数据集 每个域的的数据集包括 user_ids, items, user_items, num_users, num_items
    train_datasets, valid_datasets, test_datasets = load_ratings_dataset(args)

    if args.method == "FedHCDR":
        UU_adjs, VV_adjs, Ms, perturb_UU_adjs = load_graph_dataset(
            args, train_datasets)
    elif ("HF" in args.method) or ("DHCF" in args.method):
        UU_adjs, VV_adjs = load_graph_dataset(args, train_datasets)
    elif ("GNN" in args.method) or ("PPDM" in args.method):
        all_adjs = load_graph_dataset(args, train_datasets)

    n_clients = len(args.domains)
    if args.method == "FedHCDR":
        clients = [Client(ModelTrainer, c_id, args,
                          train_datasets[c_id], valid_datasets[c_id],
                          test_datasets[c_id],
                          UU_adj=UU_adjs[c_id], VV_adj=VV_adjs[c_id],
                          M=Ms[c_id], perturb_UU_adj=perturb_UU_adjs[c_id])
                   for c_id in range(n_clients)]
    elif ("HF" in args.method) or ("DHCF" in args.method):
        clients = [Client(ModelTrainer, c_id, args,
                          train_datasets[c_id], valid_datasets[c_id],
                          test_datasets[c_id],
                          UU_adj=UU_adjs[c_id], VV_adj=VV_adjs[c_id])
                   for c_id in range(n_clients)]
    elif ("GNN" in args.method) or ("PPDM" in args.method):
        clients = [Client(ModelTrainer, c_id, args,
                          train_datasets[c_id], valid_datasets[c_id],
                          test_datasets[c_id],
                          all_adj=all_adjs[c_id]) for c_id in range(n_clients)]
    else:
        clients = [Client(ModelTrainer, c_id, args,
                          train_datasets[c_id], valid_datasets[c_id],
                          test_datasets[c_id])
                   for c_id in range(n_clients)]
    # Initialize the aggretation weight
    init_clients_weight(clients)

    # Save the config of input arguments
    save_config(args)

    if "Fed" in args.method:
        server = Server(args, clients[0].get_params_shared())
    else:
        server = Server(args)
    server= Server(args)
    run_fl(clients, server, args)
def trainFedMa():
    from models.gnn.gnn_model import GNN
    #                              "client%d.pt" % self.c_id)
    # 加载参数
    args = arg_parse()
    # 设置cpu，gpu ，numpy 随机数种子
    seed_everything(args)
    # 设置日志文件
    init_logger(args)
    # ckpt_filename = os.path.join("./checkpoint/domain_", "FKCB", "LocalGNN_", "1732693711", "client0.pt")
    dir=os.path.join("checkpoint","domain_FKCB","LocalGNN"+ "_" + "1732693711","Food_ug.pt")
    ckpt_filename = os.path.join("checkpoint",
                                 "domain_FKCB"
                                 ,"LocalGNN"+
                                  "_" + "1732693711",
                                 "client0.pt")
    checkpoint = torch.load(ckpt_filename)


    # 加载数据集 里边有所有域的数据集 每个域的的数据集包括 user_ids, items, user_items, num_users, num_items
    train_datasets, valid_datasets, test_datasets = load_ratings_dataset(args)
    model = GNN(11880,1898, args)
    model.load_state_dict(checkpoint)
    # print(model.user_item_emb.weight)
    # Food_uemb=model.user_item_emb.weight[:1898]
    Food_uemb=model.user_item_emb
    Food_uG=UserGraph(args,"Food",file_name='Food_full_g1.npy')

    Food_full_g= Food_uG.get_users_adjacency_matrix()
    # np.save('Food_full_g1.npy', Food_full_g)
    D=Food_uG.get_D()
    L=Food_uG.get_Laplacian()

    Mamodel=Manifold(Food_uG,Food_uemb,D,L)
    opt = optim.Adam(Mamodel.parameters(), lr=0.1,weight_decay=0.001)

    for epoch in range(1000):
        opt.zero_grad()
        loss,decoder_l= Mamodel.loss()
        loss.backward()
        print("epoch :", epoch)
        print(Mamodel.encoder.fc1.weight)
        print(decoder_l)
        # if epoch % 100 ==0:
        #     print("epoch :",epoch)
        #     print(Mamodel.encoder.fc1.weight)
        #     print(decoder_l)
        # print( Mamodel.encoder.fc2.weight)
        # print(Mamodel.decoder.fc1.weight)
        # print( Mamodel.decoder.fc2.weight)
        print("-"*40)
        opt.step()

    # print(Food_uG.users_matrix)


if __name__ == "__main__":
    main()
    # trainFedMa()
