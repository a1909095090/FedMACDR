# -*- coding: utf-8 -*-
import os
import gc
import copy
import logging
import numpy as np
import torch
from dataloader import RecDataloader
from utils.io_utils import ensure_dir
from sklearn.cluster import SpectralClustering


class MaClient:
    def __init__(self, model_fn, c_id, args,
                 train_dataset, valid_dataset, test_dataset,
                 UU_adj=None, VV_adj=None, M=None, perturb_UU_adj=None,
                 all_adj=None):
        # Used for initializing embeddings of users and items
        self.args=args
        self.all_adj=all_adj
        self.num_users = train_dataset.num_users
        self.num_items = train_dataset.num_items
        self.domain = args.domains[c_id]
        self.trainer = model_fn(args, self.domain)
        self.c_id=c_id
        # self.model = self.trainer.model
        # self.model=self.trainer.
        self.method = args.method
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.checkpoint_dir = args.checkpoint_dir
        self.model_id = (args.model_id if len(args.model_id)
                         > 1 else "0" + args.model_id)


        self.train_dataloader = RecDataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = RecDataloader(
            valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = RecDataloader(
            test_dataset, batch_size=args.batch_size, shuffle=False)

        # Compute the number of samples for each client
        self.n_samples_train = len(train_dataset)
        self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)
        self.load_params()
        # The aggretation weight
        self.train_pop, self.valid_weight, self.test_weight = 0.0, 0.0, 0.0
        # Model evaluation results
        self.MRR, self.NDCG_5, self.NDCG_10, self.HR_1, self.HR_5, self.HR_10 \
            = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


    def train_epoch(self, round, args, global_params=None):
        """Trains one client with its own training data for one epoch.

        Args:
            round: Training round.
            args: Other arguments for training.
            global_params



            : Global model parameters used in `FedProx` method.
        """
        self.trainer.model.graph_convolution(self.all_adj)



        self.trainer.model.train()
        if "PPDM" in args.method:
            if (self.U_mu[0] is not None) and (self.U_sigma[0] is not None):
                similarity_matrix = \
                    torch.cdist(self.U_mu[0], self.U_mu[0], p=2)**2 \
                    + torch.cdist(self.U_sigma[0], self.U_sigma[0], p=2)**2
                similarity_matrix = torch.exp(-similarity_matrix /
                                              similarity_matrix.var())
                self.clustering_model.fit(similarity_matrix.detach().cpu())
                zeta = torch.LongTensor(
                    self.clustering_model.labels_).to(self.device)
                # Clustering center of each cluster
                tilde_u_mu = torch.stack([(self.U_mu[0].detach()[torch.eq(zeta, j)]).sum(
                    axis=0) for j in range(self.num_M)])
                tilde_u_sigma = torch.stack([(self.U_sigma[0].detach()[torch.eq(zeta, j)]).sum(
                    axis=0) for j in range(self.num_M)])
            else:
                zeta, tilde_u_mu, tilde_u_sigma = None, None, None

        for epoch in range(args.local_epoch):

            loss = 0
            step = 0
            for user_ids, interactions in self.train_dataloader:
                # if "PPDM" in args.method:
                #     batch_loss = self.trainer.train_batch(
                #         user_ids, interactions, round, args,
                #         all_adj=self.all_adj,
                #         zeta=zeta, tilde_u_mu=tilde_u_mu,
                #         tilde_u_sigma=tilde_u_sigma,
                #         global_params=global_params)
                # elif args.method == "FedHCDR":
                #     batch_loss = self.trainer.train_batch(
                #         user_ids, interactions, round, args,
                #         UU_adj=self.UU_adj, VV_adj=self.VV_adj,
                #         M=self.M, perturb_UU_adj=self.perturb_UU_adj,
                #         global_params=global_params)
                # elif ("HF" in args.method) or ("DHCF" in args.method):
                #     batch_loss = self.trainer.train_batch(
                #         user_ids, interactions, round, args,
                #         UU_adj=self.UU_adj, VV_adj=self.VV_adj,
                #         global_params=global_params)
                # elif ("GNN" in args.method) or ("PPDM" in args.method):
                #     batch_loss = self.trainer.train_batch(
                #         user_ids, interactions, round, args,
                #         all_adj=self.all_adj,
                #         global_params=global_params)
                # else:
                batch_loss = self.trainer.train_batch(
                    user_ids, interactions,
                    global_params=global_params)
                loss = batch_loss
                # step += 1

            gc.collect()
        logging.info("Epoch {}/{} - client {} -  Training Loss: {:.3f}".format(
            round, args.num_round, self.c_id, loss ))
        return self.n_samples_train

    def evaluation(self, mode="valid"):
        """Evaluates one client with its own valid/test data for one epoch.



        Args:
            mode: `valid` or `test`.
        """
        if mode == "valid":
            dataloader = self.valid_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader

        self.trainer.model.eval()

        if self.method == "FedHCDR":
            self.model.graph_convolution_hi(self.UU_adj, self.VV_adj)
            self.model.graph_convolution_lo(self.UU_adj, self.VV_adj, self.M)
        elif ("HF" in self.method) or ("DHCF" in self.method):
            self.model.graph_convolution(self.UU_adj, self.VV_adj)
        elif ("GNN" in self.method) or ("PPDM" in self.method):
            self.trainer.model.graph_convolution(self.all_adj)

        pred = []
        for user_ids, interactions in dataloader:
            predictions = self.trainer.test_batch(user_ids, interactions)
            pred = pred + predictions

        gc.collect()
        self.MRR, self.NDCG_5, self.NDCG_10, self.HR_1, self.HR_5, self.HR_10 \
            = self.cal_test_score(pred)
        return {"MRR": self.MRR, "HR @1": self.HR_1, "HR @5": self.HR_5,
                "HR @10":  self.HR_10, "NDCG @5":  self.NDCG_5,
                "NDCG @10": self.NDCG_10}


    def get_old_eval_log(self):
        """Returns the evaluation result of the lastest epoch.
        """
        return {"MRR": self.MRR, "HR @1": self.HR_1, "HR @5": self.HR_5,
                "HR @10":  self.HR_10, "NDCG @5":  self.NDCG_5,
                "NDCG @10": self.NDCG_10}

    @ staticmethod
    def cal_test_score(predictions):
        MRR = 0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        # `pred` indicates the rank of groundtruth items in the recommendation
        # list
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
        return MRR/valid_entity, NDCG_5 / valid_entity, \
            NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / \
            valid_entity, HR_10 / valid_entity

    def get_params_shared(self):
        """Returns the model parameters that need to be shared between clients.
        """
        # assert (self.method in ["FedHCDR", "FedHF", "FedDHCF", "FedMF",
        #                         "FedGNN", "FedPriCDR", "FedP2FCDR", "FedPPDM"])
        if self.method == "FedHCDR":
            return copy.deepcopy([
                self.model.lo_model.user_hyper_gcn_lo.state_dict()])
        elif self.method in ["FedHF", "FedDHCF"]:
            return copy.deepcopy([
                self.model.user_encoder.state_dict(),
                self.model.item_encoder.state_dict(),
            ])
        elif self.method == "FedMF":
            return copy.deepcopy([{"user_mlp_emb.weight":
                                   self.model.user_mlp_emb.weight},
                                  {"user_mf_emb.weight":
                                      self.model.user_mf_emb.weight}])
        elif self.method == "FedGNN":
            return copy.deepcopy([self.model.encoder.state_dict()])
        else:
            # return copy.deepcopy([dict()])
            users=np.arange(0,self.num_users-1)
            return self.trainer.get_shared_param()
    def get_reps_shared(self):
        """Returns the user representations that need to be shared
        between clients.
        """
        assert self.method in ["FedHCDR", "FedPriCDR", "FedP2FCDR", "FedPPDM"]
        if self.method == "FedHCDR":
            return copy.deepcopy([self.U_s[0].detach()])
        elif self.method in ["FedPriCDR", "FedP2FCDR"]:
            return copy.deepcopy([self.U_mlp[0].detach(),
                                  self.U_mf[0].detach()])
        elif self.method == "FedPPDM":
            return copy.deepcopy([self.U_mu[0].detach(),
                                  self.U_sigma[0].detach()])

    def set_global_params(self, global_params):
        """Assign the local shared model parameters with global model
        parameters.
        """
        assert (self.method in ["FedHCDR", "FedHF", "FedDHCF", "FedMF",
                                "FedGNN", "FedPriCDR", "FedP2FCDR", "FedPPDM"])
        if self.method == "FedHCDR":
            self.model.lo_model.user_hyper_gcn_lo.load_state_dict(
                global_params[0])
        elif self.method in ["FedHF", "FedDHCF"]:
            self.model.user_encoder.load_state_dict(global_params[0])
            self.model.item_encoder.load_state_dict(global_params[1])
        elif self.method == "FedMF":
            self.model.user_mlp_emb = self.model.user_mlp_emb.from_pretrained(
                global_params[0]["user_mlp_emb.weight"])
            self.model.user_mf_emb = self.model.user_mf_emb.from_pretrained(
                global_params[1]["user_mf_emb.weight"])
        elif self.method == "FedGNN":
            self.model.encoder.load_state_dict(global_params[0])
        else:
            self.model.global_emb=global_params

    def set_global_reps(self, global_rep):
        """Copy global user representations to local.
        """

        self.U_g[0] = copy.deepcopy(global_rep[0])

    def save_params(self):
        method_ckpt_path = os.path.join(self.checkpoint_dir,
                                        "domain_" +
                                        "".join([domain[0]
                                                for domain
                                                 in self.args.domains]),
                                        self.method + "_" + self.model_id)

        ensure_dir(method_ckpt_path, verbose=True)
        ckpt_filename = os.path.join(
            method_ckpt_path, "client%d.pt" % self.c_id)
        mackpt_filename = os.path.join(
            method_ckpt_path, "maclient%d.pt" % self.c_id)
        params = self.trainer.model.state_dict()
        maparams=self.trainer.state_dict()
        try:
            # torch.save(params, ckpt_filename)
            torch.save(maparams,mackpt_filename)
            print("Model saved to {}".format(ckpt_filename))
        except IOError:
            print("[ Warning: Saving failed... continuing anyway. ]")

    def load_params(self):
        ckpt_filename = os.path.join(self.checkpoint_dir,
                                     "domain_" +
                                     "".join([domain[0]
                                              for domain in self.args.domains]),
                                     self.method + "_" + self.model_id,
                                     "client%d.pt" % self.c_id)
        mackpt_filename = os.path.join(self.checkpoint_dir,
                                     "domain_" +
                                     "".join([domain[0]
                                              for domain in self.args.domains]),
                                     self.method + "_" + self.model_id,
                                     "maclient%d.pt" % self.c_id)
        discckpt_filename = os.path.join(self.checkpoint_dir,
                                       "domain_" +
                                       "".join([domain[0]
                                                for domain in self.args.domains]),
                                       self.method + "_" + self.model_id,
                                       "client%d_disc.pt" % self.c_id)
        try:
            # macheckpoint = torch.load(mackpt_filename)
            checkpoint = torch.load(ckpt_filename)
            discheckpoint = torch.load(discckpt_filename)
        except IOError:
            print("[ Fail: Cannot load model from {}. ]".format(ckpt_filename))
            print("[ Fail: Cannot load model from {}. ]".format(discckpt_filename))
            exit(1)
        # if self.trainer is not None:
        #     self.trainer.load_state_dict(macheckpoint)
        if self.trainer.model is not None:
            self.trainer.model.load_state_dict(checkpoint)
            for name, param in self.trainer.model.named_parameters():

                param.requires_grad = False
            self.trainer.discri.load_state_dict(discheckpoint)
            for name, param in self.trainer.discri.named_parameters():
                param.requires_grad = False
