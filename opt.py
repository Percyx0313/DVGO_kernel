import argparse

def get_opts():
    parser=argparse.ArgumentParser()
    # basic infomation
    parser.add_argument('--config',required=True,help='data config file')
    parser.add_argument('--seed',type=int,default=720,help='Random seed')
    
    
    return parser.parse_args()
    