# -*- coding: utf-8 -*-
import math
import numpy as np


class MaServer(object):
    def __init__(self, args, isFed=True):
        self.args = args
        self.isFed=True
        self.global_params = None


    def aggregate_params(self, clients):

        """Sums up parameters of models shared by all active clients at each
        epoch.

        Args:
            clients: A list of clients instances.
            random_cids: Randomly selected client ID in each training round.
        """
        if self.isFed == False:
            return
        for i in range(len(clients)):
            if i==0:
                self.global_params=clients[i].get_params_shared()
            else:
                self.global_params=self.global_params+clients[i].get_params_shared()
        self.global_params=self.global_params/len(clients)

        return self.global_params

    def choose_clients(self, n_clients, ratio=1.0):
        """Randomly chooses some clients.
        """
        choose_num = math.ceil(n_clients * ratio) #math.ceil() 向上取整。
        return np.random.permutation(n_clients)[:choose_num]

    def get_global_params(self):
        """Returns a reference to the parameters of the global model.
        """
        return self.global_params

    def get_global_reps(self):
        """Returns a reference to the parameters of the global model.
        """
        pass
        # return self.global_reps
