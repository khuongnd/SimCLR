import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if len(os.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.argv[1]

from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
