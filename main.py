import argparse
from utils.config import load_config
from utils.seed import set_seed
from training.train import run_training

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['train'],default='train'); p.add_argument('--config_dir',default='./configs')
    args=p.parse_args(); cfg=load_config(args.config_dir); set_seed(cfg.get('seed',42))
    if args.mode=='train': run_training(cfg)

if __name__=='__main__':
    main()
