'''
Utilities for Loading data.

@author:
Jiancan Wu

@references:
'''
import numpy as np
import os
import scipy.sparse as sp
import pandas as pd


class LoadData(object):
    '''given the path of data, return the data format for CFM for Top-N recommendation
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X_user' and 'X_item' refers to features for context-aware user and item
    Test_data: same as Train_data
    '''

    # Two files are needed in the path
    def __init__(self, path, dataset, model_name):
        self.path = path + dataset + "/"
        self.trainfile = self.path + "train.dat"
        self.testfile = self.path + "test.dat"
        self.model_name = model_name
        self.num_users, self.num_items, self.num_context_features, self.num_fields, self.num_item_features = self.cnt_user_item_context()
        self.binded_context, self.binded_context_reverse = self.bind_contexts()
        self.binded_item_features = self.get_item_features()
        self.num_contexts = len(self.binded_context)
        self.context_matrix = self.construct_context_mat()
        self.sp_H_u_i, self.sp_H_u_c, self.sp_H_i_c = self.construct_u_i_c_mat()
        # self.binded_ins_user, self.binded_ins_item, self.binded_ins_context = self.bind_instance_u_i_c()
        self.norm_adj_mat = self.normalize_adj_mat()
        self.user_context_positive_list = self.get_positive_list(self.trainfile)  # userID positive itemID and contextID
        self.Train_data, self.Test_data = self.construct_data()

    def cnt_user_item_context(self):
        '''
        count the number of users and items
        '''
        df = pd.read_csv(self.trainfile)
        num_users = df['uid'].max() + 1
        num_items = df['iid'].max() + 1
        num_fields = len(df.columns) - 3  # the last column is item feature
        num_context_features = df[df.columns[-2]].max() + 1
        num_item_features = df[df.columns[-1]].max() - df[
            df.columns[-1]].min() + 1

        df_test = pd.read_csv(self.testfile)
        num_users = max(num_users, df_test['uid'].max() + 1)
        num_items = max(num_items, df_test['iid'].max() + 1)
        num_context_features = max(num_context_features, df_test[df_test.columns[-2]].max() + 1)
        num_item_features = max(num_item_features, df_test[df_test.columns[-1]].max() - df_test[
            df_test.columns[-1]].min() + 1)
        return num_users, num_items, num_context_features, num_fields, num_item_features

    def construct_u_i_c_mat(self):
        '''
        construct user-item interaction matrix R
        :return: a sparse matrix R
        '''
        H_ui = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        H_uc = sp.dok_matrix((self.num_users, self.num_contexts), dtype=np.float32)
        H_ic = sp.dok_matrix((self.num_items, self.num_contexts), dtype=np.float32)
        # H_cf_ui = sp.dok_matrix((self.num_context_features, self.num_users + self.num_items), dtype=np.float32)
        f = open(self.trainfile)
        line = f.readline()  # column name
        line = f.readline()
        while line:
            features = line.strip().split(',')
            user_id = int(features[0])
            item_id = int(features[1])
            context = "-".join([str(feature) for feature in features[2:-1]])  # the last column is item feature
            context_id = self.binded_context[context]
            H_ui[user_id, item_id] = 1.
            H_uc[user_id, context_id] = 1.
            H_ic[item_id, context_id] = 1.
            line = f.readline()
        f.close()
        return H_ui.tolil(), H_uc.tolil(), H_ic.tolil()

    def bind_contexts(self):
        def bind_c(filename, binded_context, binded_context_reverse):
            f = open(filename)
            line = f.readline()  # column name
            line = f.readline()
            n_context = len(binded_context)
            while line:
                features = line.strip().split(',')
                context = "-".join([str(item) for item in features[2:-1]])
                if context not in binded_context:
                    binded_context[context] = n_context
                    binded_context_reverse[n_context] = context
                    n_context += 1
                line = f.readline()
            f.close()
            return binded_context, binded_context_reverse

        binded_context, binded_context_reverse = {}, {}
        binded_context, binded_context_reverse = bind_c(
            self.trainfile, binded_context, binded_context_reverse)
        binded_context, binded_context_reverse = bind_c(
            self.testfile, binded_context, binded_context_reverse)
        return binded_context, binded_context_reverse

    def get_item_features(self):
        # scan training file
        f = open(self.trainfile)
        line = f.readline()  # column name
        line = f.readline()
        item_features = {}
        while line:
            features = line.strip().split(',')
            item_id = int(features[1])
            feature_id = int(features[-1])
            if item_id not in item_features:
                item_features[item_id] = feature_id
            else:
                if feature_id != item_features[item_id]:
                    return
            line = f.readline()
        f.close()
        cnt1 = len(item_features)
        # scan testing file
        f = open(self.testfile)
        line = f.readline()  # column name
        line = f.readline()
        while line:
            features = line.strip().split(',')
            item_id = int(features[1])
            feature_id = int(features[-1])
            if item_id not in item_features:
                item_features[item_id] = feature_id
            else:
                if feature_id != item_features[item_id]:
                    return
            line = f.readline()
        f.close()
        cnt2 = len(item_features)
        print('the number of items that only exist in testing set is %d, amount to %.1f %%' % (cnt2 - cnt1, 100 * (cnt2 - cnt1) / cnt2))
        return item_features
    
    def bind_instance_u_i_c(self):
        f = open(self.trainfile)
        line = f.readline()  # column name
        line = f.readline()
        binded_ins_user, binded_ins_item, binded_ins_context = [], [], []
        while line:
            features = line.strip().split(',')
            user_id = int(features[0])
            item_id = int (features[1])
            context = "-".join([str(c) for c in features[2:-1]])
            context_id = self.binded_context[context]
            binded_ins_user.append(user_id)
            binded_ins_item.append(item_id)
            binded_ins_context.append(context_id)
            line = f.readline()
        f.close()
        return binded_ins_user, binded_ins_item, binded_ins_context
    
    def normalize_adj_mat(self):
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized user-item adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_double(adj):
            rowsum = np.array(adj.sum(1))

            d_sqrt_inv = np.power(rowsum, -0.5).flatten()
            d_sqrt_inv[np.isinf(d_sqrt_inv)] = 0.
            d_mat_inv = sp.diags(d_sqrt_inv)

            norm_adj = d_mat_inv.dot(adj)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate double-normalized user-item adjacency matrix.')
            return norm_adj.tocoo()

        # construct adj matrix for user-item pair
        adj_mat = sp.dok_matrix(
            (self.num_users + self.num_items + self.num_contexts, self.num_users + self.num_items + self.num_contexts), 
            dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:self.num_users, self.num_users: self.num_users + self.num_items] = self.sp_H_u_i
        adj_mat[:self.num_users, self.num_users + self.num_items:] = self.sp_H_u_c
        adj_mat[self.num_users: self.num_users + self.num_items, :self.num_users] = self.sp_H_u_i.T
        adj_mat[self.num_users: self.num_users + self.num_items, self.num_users + self.num_items:] = self.sp_H_i_c
        adj_mat[self.num_users + self.num_items:, :self.num_users] = self.sp_H_u_c.T
        adj_mat[self.num_users + self.num_items:, self.num_users: self.num_users + self.num_items] = self.sp_H_i_c.T
        adj_mat = adj_mat.todok()
        
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_double(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def normalize_useritem_context_mat(self):
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print(
                'generate single-normalized useritem-context adjacency matrix.'
            )
            return norm_adj.tocoo()

        # normalize H_uc sp matrix
        adj_mat = self.sp_H_uc.todok()
        norm_H_uc_sp_mat = normalized_adj_single(adj_mat)

        # normalize H_ic sp matrix
        adj_mat = self.sp_H_ic.todok()
        norm_H_ic_sp_mat = normalized_adj_single(adj_mat)
        return norm_H_uc_sp_mat.tocsr(), norm_H_ic_sp_mat.tocsr()

    def construct_context_mat(self):
        context_mat = np.empty([self.num_contexts, self.num_fields], dtype=int)
        for context in self.binded_context:
            id = self.binded_context[context]
            context_mat[id] = [int(c) for c in context.split('-')]
        return context_mat

    def get_positive_list(self, file):
        '''
        Obtain positive item lists for each user
        :param file: train file
        :return:
        '''
        f = open(file)
        line = f.readline()  # column name
        line = f.readline()
        user_context_positive_list = {}
        while line:
            features = line.strip().split(',')
            user_id = int(features[0])
            item_id = int(features[1])
            context = "-".join([str(item) for item in features[2:-1]])
            context_id = self.binded_context[context]
            if user_id in user_context_positive_list:
                if context_id in user_context_positive_list[user_id]:
                    user_context_positive_list[user_id][context_id].append(
                        item_id)
                else:
                    user_context_positive_list[user_id][context_id] = [item_id]
            else:
                user_context_positive_list[user_id] = {}
                user_context_positive_list[user_id][context_id] = [item_id]

            line = f.readline()
        f.close()
        return user_context_positive_list

    def construct_data(self):
        '''
        Construct train and test data
        :return:
        '''
        X_user, X_context, X_item = self.read_data(self.trainfile)
        Train_data = self.construct_dataset(X_user, X_context, X_item)
        print("# of training:", len(X_user))

        X_user, X_context, X_item = self.read_data(self.testfile)
        Test_data = self.construct_dataset(X_user, X_context, X_item)
        print("# of test:", len(X_user))
        return Train_data, Test_data

    # lists of user and item
    def read_data(self, file):
        '''
        read raw data
        :param file: data file
        :return: structured data
        '''
        # read a data file;
        f = open(file)
        X_user = []
        X_item = []
        X_context = []
        line = f.readline()  # column name
        line = f.readline()
        while line:
            features = line.strip().split(',')
            user_id = int(features[0])
            item_id = int(features[1])
            context = "-".join([str(item) for item in features[2:-1]])
            context_id = self.binded_context[context]
            X_user.append(user_id)
            X_item.append(item_id)
            X_context.append(context_id)
            line = f.readline()
        f.close()
        return X_user, X_context, X_item

    def construct_dataset(self, X_user, X_context, X_item):
        '''
        Construct dataset
        :param X_user: user structured data
        :param X_item: item structured data
        :return:
        '''
        Data_Dic = {}
        Data_Dic['X_user'] = X_user
        Data_Dic['X_context'] = X_context
        Data_Dic['X_item'] = X_item
        return Data_Dic
