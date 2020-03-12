import os
import pandas as pd
import json

from apps.utils import get_config


def parse_args():
    class Args:
        config = get_config(path=os.environ["OPENAI_CONFIG"])

        path = config['path']

    return Args()


def pprint(json_dict):
    print(json.dumps(json_dict, indent=4))


def calculate_stats(path):
    data = pd.read_csv(os.path.join(path, 'train_labels.csv'))
    stats = {
        "Number of samples": len(data),
        "Percent of verified": f"{data.verified.mean():.3f}"
    }

    for cl in data.columns[2:]:
        stats[f"Class {cl} percent"] = f"{data[cl].mean():.3f}"

    return stats


def main():
    args = parse_args()
    stats = calculate_stats(path=args.path)
    pprint(stats)


if __name__ == '__main__':
    main()
