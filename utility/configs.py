import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run GCM.")
    parser.add_argument('--env', type=int, default=1,
                        help='select the environment for this implementation (1 for PC, 2 for cluster)')
    parser.add_argument('--dataset', nargs='?', default='Yelp-OH',
                        help='Choose a dataset: Yelp-OH, or Yelp-NC, or Amazon-Book')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='Test batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--merge_type', nargs='?', default='sum',
                        help='can be sum, ip (i.e., inner product) or mlp(i.e., concat then MLP')
    parser.add_argument('--num_gcn_layers', type=int, default=1,
                        help='Number of GCN layers.')
    parser.add_argument('--gcn_layer_weight', nargs='?', default='[0.5,0.5]',
                        help='GCN layer weight when combine different gcn layer output')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (bpr_loss or log_loss).')
    parser.add_argument('--num_negatives', type=int, default=4,
                        help='the number of negatives for log loss, can be 1, 2 or 4')
    parser.add_argument('--decoder_type', nargs='?', default='FM',
                        help='Specify a decoder type.(IP for inner prodcut, FM for factorzation machine, MLP for multilayer perceptron)')
    parser.add_argument('--num_hidden_layers', type=int, default=1,
                        help='Number of hidden layers for NFM and GCM with MLP decoder.')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--reg', type=float, default=1e-3,
                        help='Regularization coefficient of embeddings.')
    parser.add_argument('--adj_norm_type', nargs='?', default='ls',
                        help='normalization type of adj matrix for NGCF, LightGCN and GCM, can be \
                            1. ls for left-single, \
                            2. ld for left-double, \
                            3. rs for right-single, \
                            4. rd for right-double and \
                            5. db for double')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='Test interval.')
    parser.add_argument('--stop_cnt', type=int, default=50,
                        help='stop count')
    parser.add_argument('--topk', nargs='?', default='[10,50]',
                        help='Top K')
    parser.add_argument('--init_method', nargs='?', default='xavier_normal',
                        help='1. tnormal for truncated_normal_initializer, 2.uniform for random_uniform_initializer 3.normal for random_normal_initializer, 4.xavier_normal, 5. xavier_uniform, 6.he_normal, 7.he_uniform')
    parser.add_argument('--stddev', type=float, default=0.01,
                        help='standard devariation for initialization')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='flag for saving model to pretrain file. 0: not save; 1: save')

    return parser.parse_args()


def post_process_for_config(args, model_name):
    args.model_name = model_name
    # select project path for different environment
    if args.env == 1:
        args.proj_path = 'D:/OneDrive - mail.ustc.edu.cn/PythonProjects/GCM-master/'
        args.data_path = 'D:/OneDrive - mail.ustc.edu.cn/PythonProjects/GCM-master/dataset/%s/' % args.dataset
    elif args.env == 2:
        args.proj_path = '/home/wujc/PythonProjects/GCM-master/'
        args.data_path = '/home/wujc/PythonProjects/GCM-master/dataset/%s/' % args.dataset
    else:
        raise ValueError("Environment is Not recognized!")

    args.data_separator = [',', '-']

    if model_name in ['MF', 'LightGCN']:
        args.data_format = 'UI'
    else:
        args.data_format = 'UIC'
    if model_name in ['GCM', 'LightGCN']:
        args.gcn_layer_weight = eval(args.gcn_layer_weight)
        if len(args.gcn_layer_weight) - 1 != args.num_gcn_layers:
            args.gcn_layer_weight = [1 / (args.num_gcn_layers + 1)] * (args.num_gcn_layers + 1)

    if args.adj_norm_type not in ['ls', 'ld', 'rs', 'rd', 'db']:
        raise ValueError("adj_norm_type is invalid.")
    return args