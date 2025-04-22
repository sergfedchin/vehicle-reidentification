from processor import get_model
import argparse
import torch
import yaml


parser = argparse.ArgumentParser(description='ReID model trainer')
parser.add_argument('--config', default=None, help='Config Path')
args = parser.parse_args()
with open(args.config, "r") as stream:
    data = yaml.safe_load(stream)
model = get_model(data, torch.device("cuda"))


# def format_time(t: int) -> str:
#     return f'{f"{int(t // 3600)}h " if t >= 3600 else ""}'\
#            f'{f"{int(t // 60) % 60}m " if t >= 60 else ""}'\
#            f'{f"{int(t % 60)}" if t >= 10 else f"{t:.1f}"}s'

# print(format_time(2.1), format_time(45), format_time(123), format_time(30750), sep=' | ')
