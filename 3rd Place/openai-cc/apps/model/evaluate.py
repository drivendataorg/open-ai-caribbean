import os
from .src.utils import prepare_target_and_preds, calculate_score
from apps.utils import get_config


def parse_args():
    class Args:
        config = get_config(path=os.environ["OPENAI_CONFIG"])

        path = config['path']
        predictions_name = config['predictions-name']
        verified_only = config['evaluation']['verified_only']

    return Args()


def evaluate(path, preds_name, verified_only):
    preds_path = os.path.join(path, preds_name)
    target, preds, target_columns = prepare_target_and_preds(path, preds_path, verified_only)
    print(f'Evaluate: {preds_name} on {len(target)} samples')
    score = calculate_score(target, preds)
    print(f'Model scored: {score:.3f}\n')

    for id in range(len(target_columns)):
        print(f'Class {target_columns[id]}: {calculate_score(target[:, id], preds[:, id]):.3f}')


def main():
    args = parse_args()
    print(args.predictions_name)
    evaluate(
        path=args.path,
        preds_name=args.predictions_name,
        verified_only=args.verified_only
    )


if __name__ == '__main__':
    main()