"""
@author: Zhongchuan Sun
"""
from utility.Tool import csr_to_user_dict, typeassert, pad_sequences, randint_choice
from scipy.sparse import csr_matrix
from utility.DataIterator import DataIterator
import numpy as np
from evaluator.backend import eval_score_matrix_loo


class AbstractEvaluator(object):
    """Basic class for evaluator.
    """
    def __init__(self):
        pass

    def metrics_info(self):
        raise NotImplementedError

    def evaluate(self, ranking_score):
        raise NotImplementedError

class LeaveOneOutEvaluator(AbstractEvaluator):
    """Evaluator for leave one out ranking task.
    """
    # @typeassert(train_matrix=csr_matrix, test_matrix=csr_matrix, top_k=(int, list, tuple))
    def __init__(self, train_matrix, test_matrix, test_context_dict=None, top_k=50):
        super(LeaveOneOutEvaluator, self).__init__()
        self.max_top = top_k if isinstance(top_k, int) else max(top_k)
        if isinstance(top_k, int):
            self.top_show = np.arange(top_k) + 1
        else:
            self.top_show = np.sort(top_k)
        self.user_pos_train = csr_to_user_dict(train_matrix)
        self.user_pos_test = csr_to_user_dict(test_matrix)
        self.test_context_dict = test_context_dict
        self.metrics_num = 3

    def metrics_info(self):
        HR = '\t'.join([("HitRatio@"+str(k)).ljust(12) for k in self.top_show])
        NDCG = '\t'.join([("NDCG@" + str(k)).ljust(12) for k in self.top_show])
        MRR = '\t'.join([("MRR@" + str(k)).ljust(12) for k in self.top_show])
        metric = '\t'.join([HR, NDCG, MRR])
        # return metric
        return "metrics:\t%s" % metric

    def evaluate(self, model):
        # B: batch size
        # N: the number of items
        test_batch_size = model.test_batch_size

        test_users = DataIterator(list(self.user_pos_test.keys()), batch_size=test_batch_size, shuffle=False, drop_last=False)
        batch_result = []
        for batch_users in test_users:
            test_items = []
            for user in batch_users:
                num_item = len(self.user_pos_test[user])
                if num_item != 1:
                    raise ValueError("the number of test item of user %d is %d" % (user, num_item))
                test_items.append(self.user_pos_test[user][0])
            ranking_score = model.predict(batch_users, None)  # (B,N)
            ranking_score = np.array(ranking_score)

            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.
            for idx, user in enumerate(batch_users):
                train_items = self.user_pos_train[user]
                ranking_score[idx][train_items] = -np.inf

            result = eval_score_matrix_loo(ranking_score, test_items, top_k=self.max_top, thread_num=None)  # (B,k*metric_num)
            batch_result.append(result)

        # concatenate the batch results to a matrix
        all_user_result = np.concatenate(batch_result, axis=0)
        final_result = np.mean(all_user_result, axis=0)  # mean

        final_result = np.reshape(final_result, newshape=[self.metrics_num, self.max_top])
        final_result = final_result[:, self.top_show - 1]
        final_result = np.reshape(final_result, newshape=[-1])
        buf = '\t'.join([("%.4f" % x).ljust(12) for x in final_result])
        return final_result, buf

    def evaluate4recall(self, model, num_recall, recall_type='MF'):
        # B: batch size
        # N: the number of items
        test_users = DataIterator(list(self.user_pos_test.keys()), batch_size=2048, shuffle=False, drop_last=False)
        batch_result = []
        for batch_users in test_users:
            result = []
            if recall_type == 'MF':
                ranking_score = model.predict(batch_users, None)  # (B,N)
                ranking_score = np.array(ranking_score)

                # set the ranking scores of training items to -inf,
                # then the training items will be sorted at the end of the ranking list.
                for idx, user in enumerate(batch_users):
                    ranking_score_cur_user = ranking_score[idx]
                    all_pos_items = self.user_pos_train[user]
                    all_pos_items.extend(self.user_pos_test[user])
                    ranking_score_cur_user[all_pos_items] = -np.inf     # mark scores of all positive items as -inf
                    recall_item_cur = np.argsort(-ranking_score_cur_user)   # sort in decent order, return id
                    recall_item_cur = recall_item_cur[0: num_recall - 1]    # only keep first (num_recall - 1) items
                    # recall_item_cur = np.append(recall_item_cur, self.user_pos_test[user])    # include test item
                    recall_item_cur = np.sort(recall_item_cur)
                    result.append(recall_item_cur)
            else:
                for idx, user in enumerate(batch_users):
                    all_pos_items = self.user_pos_train[user]
                    all_pos_items.extend(self.user_pos_test[user])
                    recall_item_cur = randint_choice(model.num_valid_items, num_recall - 1, replace=False, exclusion=all_pos_items)
                    recall_item_cur = np.sort(recall_item_cur)
                    result.append(recall_item_cur)

            batch_result.append(result)

        # concatenate the batch results to a matrix
        final_result = np.concatenate(batch_result, axis=0)
        return final_result

    def evaluate4CARS(self, model):
        # B: batch size
        # N: the number of items
        test_batch_size = model.test_batch_size

        pos_user_test_list = list(self.user_pos_test.keys())
        test_context_list = [self.test_context_dict[u] for u in pos_user_test_list]
        test_iter = DataIterator(pos_user_test_list, test_context_list, batch_size=test_batch_size, shuffle=False, drop_last=False)
        batch_result = []
        for batch_users, batch_contexts in test_iter:
            test_items = []
            for user in batch_users:
                num_item = len(self.user_pos_test[user])
                if num_item != 1:
                    raise ValueError("the number of test item of user %d is %d" % (user, num_item))
                test_items.append(self.user_pos_test[user][0])
            ranking_score = model.predict(batch_users, batch_contexts)  # (B,N)
            ranking_score = np.array(ranking_score)

            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.
            for idx, user in enumerate(batch_users):
                train_items = self.user_pos_train[user]
                ranking_score[idx][train_items] = -np.inf

            result = eval_score_matrix_loo(ranking_score, test_items, top_k=self.max_top, thread_num=None)  # (B,k*metric_num)
            batch_result.append(result)

        # concatenate the batch results to a matrix
        all_user_result = np.concatenate(batch_result, axis=0)
        final_result = np.mean(all_user_result, axis=0)  # mean

        final_result = np.reshape(final_result, newshape=[self.metrics_num, self.max_top])
        final_result = final_result[:, self.top_show - 1]
        final_result = np.reshape(final_result, newshape=[-1])
        buf = '\t'.join([("%.4f" % x).ljust(12) for x in final_result])
        return final_result, buf
