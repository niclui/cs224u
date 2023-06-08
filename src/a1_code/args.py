import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str, default='prajjwal1/bert-mini')
    parser.add_argument('--prompting', type=str, default='r2')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/a1_bakeoff/bert-mini-cls.pt')
    parser.add_argument('--val-data', type=str, default='data/sentiment/cs224u-sentiment-test-unlabeled.csv')
    parser.add_argument('--pred-file', type=str, default='data/sentiment/test-bert-mini.csv')
    parser.add_argument('--lr', type=float, default=0.00005)
    args = parser.parse_args()
    return args
