from src.analysis.real_world_dataset_ablation import clustering_recommendation_ari
import argparse

# This script generates Table 2 from the paper, comparing model performance (F1-score, Hamming distance, ARI)
# across real-world datasets for: CNN, ResNet, No CNN, No residual, and No attention models.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clustering analysis on real world datasets.")
    parser.add_argument('--data-path', type=str, default="data/real_world_datasets", help='Path to dataset')
    parser.add_argument('--model-path', type=str, default="models/cnnresatt.pth", help='Path to model directory')
    parser.add_argument('--model-name', type=str, choices=['cnnresatt', 'baseline_cnn', 'no_cnn', 'no_res', 'no_att'], default='cnnresatt',
                        help='Type of model to use for evaluation (cnnresatt, baseline_cnn, no_cnn, no_res, no_att)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    clustering_recommendation_ari(  
        data_path=args.data_path,
        model_path=args.model_path,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
