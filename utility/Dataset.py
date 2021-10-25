import numpy as np
import scipy.sparse as sp
import os
from utility.Tool import randint_choice, df_to_positive_dict, save_dict_to_file, load_dict_from_file, csr_to_user_dict
import pandas as pd
from time import time

KEEP_CONTEXT = {
    'yelp-nc': ['c_city', 'c_month', 'c_hour', 'c_DoW', 'c_last'],
    'yelp-oh': ['c_city', 'c_month', 'c_hour', 'c_DoW', 'c_last'],
    'amazon-book': ['c_year', 'c_month', 'c_day', 'c_DOW', 'c_last']
}


class GivenData(object):
    def __init__(self, dataset_name, path, data_format, separator, logger):
        self.dataset_name = dataset_name
        self.path = path
        self.data_format = data_format
        self.separator = separator
        self.logger = logger

    def load_data(self):
        side_info, all_data_dict = None, None
        self.logger.info("Loading interaction records from folder: %s "% (self.path))

        train_data = pd.read_csv(self.path + "train.dat", sep=self.separator[0])
        test_data = pd.read_csv(self.path + "test.dat", sep=self.separator[0])
        userid_dict = load_dict_from_file(self.path + 'userid_dict.txt')
        itemid_dict = load_dict_from_file(self.path + 'itemid_dict.txt')
        # self.logger.info('Loading full testset')

        all_data = pd.concat([train_data, test_data])

        num_users = len(userid_dict)
        num_items = len(itemid_dict) 
        num_valid_items = all_data["item_id"].max() + 1

        num_train = len(train_data["user_id"])
        num_test = len(test_data["user_id"])
        
        train_matrix = sp.csr_matrix(([1] * num_train, (train_data["user_id"], train_data["item_id"])), shape=(num_users, num_valid_items))
        test_matrix = sp.csr_matrix(([1] * num_test, (test_data["user_id"], test_data["item_id"])),  shape=(num_users, num_valid_items))
        
        if self.data_format == 'UIC':
            side_info, side_info_stats, all_data_dict = {}, {}, {}
            column = all_data.columns.values.tolist()
            context_column = column[2].split(self.separator[1])
            user_feature_column = column[3].split(self.separator[1]) if 'yelp' in self.dataset_name.lower() else None
            item_feature_column = column[-1].split(self.separator[1])

            keep_context = KEEP_CONTEXT[self.dataset_name.lower()]
            new_context_column = '-'.join(keep_context)
            all_data[context_column] = all_data[all_data.columns[2]].str.split(self.separator[1], expand=True)
            all_data[new_context_column] = all_data[keep_context].apply('-'.join, axis=1)

            # map context to id
            unique_context = all_data[new_context_column].unique()
            context2id = pd.Series(data=range(len(unique_context)), index=unique_context)
            # contextids = context2id.to_dict()
            all_data["context_id"] = all_data[new_context_column].map(context2id)
            train_data = all_data.iloc[:num_train, :]
            test_data = all_data.iloc[num_train:, :]

            if user_feature_column:
                user_feature = all_data.drop_duplicates(["user_id", '-'.join(user_feature_column)])
                user_feature = user_feature[["user_id", '-'.join(user_feature_column)]]
                user_feature[user_feature_column] = user_feature[user_feature.columns[-1]].str.split(self.separator[1], expand=True)
                user_feature.drop(user_feature.columns[[1]], axis=1, inplace=True)
            else:
                user_feature = None
            item_feature = all_data.drop_duplicates(["item_id", '-'.join(item_feature_column)])
            item_feature = item_feature[["item_id", '-'.join(item_feature_column)]]
            item_feature[item_feature_column] = item_feature[item_feature.columns[-1]].str.split(self.separator[1], expand=True)
            item_feature.drop(item_feature.columns[[1]], axis=1, inplace=True)
            context_feature = all_data.drop_duplicates(["context_id", new_context_column])[["context_id", new_context_column]]
            context_feature[keep_context] = context_feature[context_feature.columns[-1]].str.split(self.separator[1], expand=True)
            context_feature.drop(context_feature.columns[[1]], axis=1, inplace=True)
            if user_feature_column:
                side_info['user_feature'] = user_feature.set_index('user_id').astype(int)
                side_info_stats['num_user_features'] = side_info['user_feature'][user_feature_column[-1]].max() + 1
                side_info_stats['num_user_fields'] = len(user_feature_column)
            else:
                side_info['user_feature'] = None
                side_info_stats['num_user_features'] = 0
                side_info_stats['num_user_fields'] = 0
            
            side_info['item_feature'] = item_feature.set_index('item_id').astype(int)
            side_info['context_feature'] = context_feature.set_index('context_id').astype(int)
            side_info_stats['num_item_features'] = side_info['item_feature'][item_feature_column[-1]].max() + 1
            side_info_stats['num_item_fields'] = len(item_feature_column)
            side_info_stats['num_context_features'] = side_info['context_feature'][keep_context[-2]].max() + 1 + num_items
            side_info_stats['num_context_fields'] = len(keep_context)
            self.logger.info("\n" + "\n".join(["{}={}".format(key, value) for key, value in side_info_stats.items()]))
            self.logger.info("context feature name: " + ",".join([f.replace('c_', '') for f in keep_context]))
            all_data_dict['train_data'] = train_data[['user_id', 'item_id', 'context_id']]
            all_data_dict['test_data'] = test_data[['user_id', 'item_id', 'context_id']]
            # all_data_dict['positive_dict'] = df_to_positive_dict(all_data_dict['train_data'])
            try:
                t1 = time()
                all_data_dict['positive_dict'] = load_dict_from_file(self.path + '/user_pos_dict.txt')
                print('already load user positive dict', time() - t1)
            except Exception:
                all_data_dict['positive_dict'] = df_to_positive_dict(all_data_dict['train_data'])
                save_dict_to_file(all_data_dict['positive_dict'], self.path + '/user_pos_dict.txt')
            side_info['side_info_stats'] = side_info_stats
        
        num_ratings = len(train_data["user_id"]) + len(test_data["user_id"])
        self.logger.info("\"num_users\": %d,\"num_items\":%d,\"num_valid_items\":%d, \"num_ratings\":%d"%(num_users, num_items, num_valid_items, num_ratings))
        
        return train_matrix, test_matrix, all_data_dict, side_info, num_items

class Dataset(object):
    def __init__(self, conf, logger):
        """
        Constructor
        """
        self.logger = logger
        self.separator = conf.data_separator

        self.dataset_name = conf.dataset
        self.dataset_folder = conf.data_path
        
        data_splitter = GivenData(self.dataset_name, self.dataset_folder, conf.data_format, self.separator, self.logger)
        
        self.train_matrix, self.test_matrix, self.all_data_dict, self.side_info, self.num_items = data_splitter.load_data()
        # self.test_context_list = self.all_data_dict['test_data']['context_id'].tolist() if self.side_info is not None else None
        if self.side_info is None:
            self.test_context_dict = None
        else:
            self.test_context_dict = {}
            for user, context in zip(self.all_data_dict['test_data']['user_id'].tolist(), self.all_data_dict['test_data']['context_id'].tolist()):
                self.test_context_dict[user] = context

        self.num_users, self.num_valid_items = self.train_matrix.shape
        if self.side_info is not None:
            self.num_user_features = self.side_info['side_info_stats']['num_user_features']
            self.num_item_featuers = self.side_info['side_info_stats']['num_item_features']
            self.num_context_features = self.side_info['side_info_stats']['num_context_features']
        self.logger.info('Data Loading is Done!')