import numpy as np
import random
from utility.Tool import randint_choice, csr_to_user_dict


def _get_pairwise_all_data(dataset):
    user_input, item_input_pos, item_input_neg = [], [], []
    train_matrix = dataset.train_matrix
    num_items = dataset.num_valid_items
    num_users = dataset.num_users
    
    for u in range(num_users):
        items_by_u = train_matrix[u].indices
        num_items_by_u = len(items_by_u)
        if num_items_by_u > 0:
            user_input.extend([u] * num_items_by_u)
            item_input_pos.extend(items_by_u)
            item_input_neg.extend(randint_choice(num_items, num_items_by_u, replace=True, exclusion=items_by_u))
            
    return user_input, item_input_pos, item_input_neg

def _get_pairwise_all_data_context(dataset):
    user_input, context_input, item_input_pos, item_input_neg = [], [], [], []
    train_data = dataset.all_data_dict['train_data']
    user_input = train_data['user_id'].tolist()
    context_input = train_data['context_id'].tolist()
    item_input_pos = train_data['item_id'].tolist()

    num_items = dataset.num_valid_items
    
    for idx in range(len(user_input)):
        user_id = user_input[idx]
        context_id = context_input[idx]
        # pos = train_data[(train_data['user_id'] == user_id) & (train_data['context_id'] == context_id)]['item_id'].tolist()
        pos = dataset.all_data_dict['positive_dict'][user_id][context_id]
        neg_item_id = np.random.randint(num_items)
        while neg_item_id in pos:
            neg_item_id = np.random.randint(num_items)
        item_input_neg.append(neg_item_id)
            
    return user_input, context_input, item_input_pos, item_input_neg 

def _get_pointwise_all_data(dataset, num_negatives, phase='train'):
    user_input, item_input, labels = [], [], []
    if phase == 'train':
        train_matrix = dataset.train_matrix
    else:
        train_matrix = dataset.test_matrix
        
    num_items = dataset.num_valid_items
    num_users = dataset.num_users
    
    for u in range(num_users):
        items_by_u = train_matrix[u].indices
        num_items_by_u = len(items_by_u)
        if num_items_by_u > 0:
            negative_items = randint_choice(num_items, num_items_by_u * num_negatives, replace=True, exclusion=items_by_u)
            index = 0
            for i in items_by_u:
                # positive instance
                user_input.append(u)
                item_input.append(i)
                labels.append(1)
                # negative instance
                user_input.extend([u] * num_negatives)
                item_input.extend(negative_items[index: index + num_negatives])
                labels.extend([0] * num_negatives)
                index = index + num_negatives
    return user_input, item_input, labels

def _get_pointwise_all_data_context(dataset, num_negatives, phase='train'):
    user_input, context_input, item_input, labels = [], [], [], []
    if phase == 'train':
        train_data = dataset.all_data_dict['train_data']
    else:
        train_data = dataset.all_data_dict['test_data']
        user_pos_test = csr_to_user_dict(dataset.test_matrix)
    user_insts = train_data['user_id'].tolist()
    context_insts = train_data['context_id'].tolist()
    item_insts_pos = train_data['item_id'].tolist()

    num_items = dataset.num_valid_items
    
    for idx in range(len(user_insts)):
        user_id = user_insts[idx]
        context_id = context_insts[idx]
        user_input.extend([user_id] * (num_negatives + 1))
        context_input.extend([context_id] * (num_negatives + 1))
        item_input.append(item_insts_pos[idx])
        labels.append(1)
        try:
            user_pos = dataset.all_data_dict['positive_dict'][user_id][context_id]
        except Exception:
            user_pos = []
        if phase != 'train':
            user_pos = user_pos + user_pos_test[user_id]
        for _ in range(num_negatives):
            neg_item_id = np.random.randint(num_items)
            while neg_item_id in user_pos:
                neg_item_id = np.random.randint(num_items)
            item_input.append(neg_item_id)
            labels.append(0)
    return user_input, context_input, item_input, labels