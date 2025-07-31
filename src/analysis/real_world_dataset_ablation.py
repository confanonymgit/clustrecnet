from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
import warnings
warnings.filterwarnings("ignore")
from src.clust_utils import load_config, load_real_datasets, prepare_tensor, build_model, predict_algorithms
from src.clustering.algorithms import ClusteringAlgorithmFactory, ClusteringAlgorithms


def clustering_recommendation_ari(data_path: str, model_path: str, model_name: str, seed: int = 0) -> pd.DataFrame:
    
    datasets_path = Path(data_path)
    model_path = Path(model_path)
    all_data, dataset_names = load_real_datasets(datasets_path)
    x_padded = prepare_tensor(all_data, dataset_names)
    models = build_model(model_name, model_path)
    predictions = predict_algorithms(models, x_padded)

    # === Params ===
    algo_n_clusters_param = {
        "kmeans": "n_clusters",
        "kmedians": "n_clusters",
        "spectral_clustering": "n_clusters",
        "ward": "n_clusters",
        "agglomerative": "n_clusters",
        "birch": "n_clusters",
        "gaussian": "n_components"
    }

    algorithms = [
        "kmeans",
        "kmedians",
        "spectral_clustering",
        "ward",
        "agglomerative",
        "dbscan",
        "hdbscan",
        "optics",
        "birch",
        "gaussian"
    ]

    # === Initialize factory ===
    factory = ClusteringAlgorithmFactory()

    # === Create results dataframe ===
    df_results = pd.DataFrame({"": dataset_names})

    # === Main loop over datasets ===
    for idx, dataset_name in enumerate(dataset_names):
        print(f"\n Processing dataset: {dataset_name}")

        X = all_data[dataset_name]["X_standard"]
        labels_true = all_data[dataset_name]["labels"]

        # Step 1: select algorithm
        algo_idx = np.argmax(predictions[idx])
        selected_algo = algorithms[algo_idx]
        
        # Config DEF sweep
        cal_scores_def, sil_scores_def = [], []

        for n_clusters in range(2, 16):

            # Build DEF config:
            if selected_algo in ["dbscan", "hdbscan", "optics"]:
                continue
            param_name = algo_n_clusters_param[selected_algo]
            algo_config_def = {param_name: n_clusters}

            if selected_algo in ["kmeans", "kmedians", "spectral_clustering", "gaussian"]:
                algo_config_def["random_state"] = seed

            if selected_algo == "ward":
                algo_config_def["linkage"] = "ward"

            # === Run DEF config
            algo_instance_def = factory.create_algorithm(selected_algo, algo_config_def)
            labels_def = algo_instance_def.fit_predict(X)

            if len(np.unique(labels_def)) < 2:
                cal_def, sil_def = -1, -1
            else:
                cal_def = calinski_harabasz_score(X, labels_def)
                sil_def = silhouette_score(X, labels_def)

            cal_scores_def.append((n_clusters, cal_def))
            sil_scores_def.append((n_clusters, sil_def))

        # === Best n_clusters
        if selected_algo not in ["dbscan", "hdbscan", "optics"]:
            n_clusters_cal_def = max(cal_scores_def, key=lambda x: x[1])[0]
            n_clusters_sil_def = max(sil_scores_def, key=lambda x: x[1])[0]
        else:
            # Assign dummy values for clustering methods without n_clusters
            n_clusters_cal_def = n_clusters_sil_def = -1

        # === Step 3: 8 combinations of ARI ===
        if model_name == "baseline_cnn":
            sources = ["CAL"]   
        else:
            sources = ["CAL", "SIL"]
        for source in sources:
            if source == "CAL":
                n_clusters_sel = n_clusters_cal_def
            elif source == "SIL":
                n_clusters_sel = n_clusters_sil_def

            # === Build config for this run:
            if selected_algo in ["dbscan", "hdbscan", "optics"]:
                config_run = load_config(
                    file_path=f"configs/{dataset_name.lower()}.yaml", 
                    variables={"num_clusters": 6, "random_state": seed}
                )  # num_clusters dummy
                algo_config_run = config_run[selected_algo]
                if "n_neighbors" in algo_config_run:
                    algo_config_run["connectivity"] = ClusteringAlgorithms.get_connectivity_graph(X, algo_config_run["n_neighbors"])
                    del algo_config_run["n_neighbors"]
            else:
                config_run = load_config(
                    file_path=f"configs/{dataset_name.lower()}.yaml", 
                    variables={"num_clusters": n_clusters_sel, "random_state": seed}
                )
                algo_config_run = config_run[selected_algo]
                if "n_neighbors" in algo_config_run:
                    algo_config_run["connectivity"] = ClusteringAlgorithms.get_connectivity_graph(X, algo_config_run["n_neighbors"])
                    del algo_config_run["n_neighbors"]
                param_name = algo_n_clusters_param[selected_algo]
                algo_config_run[param_name] = n_clusters_sel
                if selected_algo == "ward":
                    algo_config_run["linkage"] = "ward"

            # === Run algorithm:
            algo_instance = factory.create_algorithm(selected_algo, algo_config_run)
            labels_pred = algo_instance.fit_predict(X)

            if len(np.unique(labels_pred)) < 2:
                ari_score = 0.0
            else:
                ari_score = adjusted_rand_score(labels_true, labels_pred)

            # === Save result:
            col_name = f"{model_name.upper()}/{source}/ARI"
            df_results.loc[idx, col_name] = ari_score

            print(f"✅ {col_name}: {ari_score:.4f}")

    print("\n🎉 All datasets processed!")

    # Save to CSV
    df_results.to_csv(f"results/real_world_data_clustering_analysis_{model_name}.csv", index=False)
