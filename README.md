# ClustRecNet: A Novel CNN-Residual Network with Attention Mechanism for Clustering Algorithm Recommendation

**ClustRecNet** is a deep learning framework that recommends the most suitable clustering algorithm for a given dataset. Trained on 34,000 synthetic datasets labeled with Adjusted Rand Index (ARI) scores from 10 clustering algorithms, it learns both local and global structural patterns using CNN, ResNet, and attention modules. It significantly outperforms traditional Cluster Validity Indices (CVIs) such as Silhouette and Calinski-Harabasz, as well as AutoML-based frameworks like ML2DAC, AutoML4Clust and AutoCluster.

---

## Key Features

- **End-to-End Learning**: Fully trainable deep neural network combining CNN, residual, and attention layers.
- **Large-Scale Training**: Built on 34,000 diverse synthetic datasets with known ground truth.
- **Generalization to Real-World Data**: Robust performance across multiple benchmarks.
- **State-of-the-Art Performance**: Consistently surpasses CVIs and AutoML baselines.
- **Modular Design**: Easily extendable to support new clustering algorithms or architectures.

---

## Repository Structure

```bash
```bash
ClustRecNet/
├── configs/             # Configuration files
├── data/                # Real-world and synthetic datasets
├── models/              # Trained model checkpoints
├── notebooks/           # Jupyter notebooks for evaluation and visualization
├── results/             # Output metrics and predictions
├── src/                 # Core source code
│   ├── analysis/        # Evaluation metrics and ablation study
│   ├── clust_utils/     # Clustering utilities and helpers
│   ├── clustering/      # Clustering algorithm implementations
│   ├── data_generation/ # Synthetic data generation pipeline
│   ├── tests/           # Unit and integration tests
│   └── training/        # Model definitions and training loops
├── main.py              # Entry point for training
├── ablation.py          # Run ablation studies on real datasets
├── requirements.txt     # Project dependencies
```

---

## Installation

1. Clone the Repository

```bash
git clone https://github.com/confanonymgit/clustrecnet.git
cd clustrecnet
```
2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Running the Pipeline

### Train the Model

To train on synthetic data with a selected model type:

```bash
python main.py --data-path="data/" --model-name="MODEL_NAME"
```

Available --model-name options:

- cnnresatt: CNN + ResNet + ATN (default)
- baseline_cnn: baseline CNN
- no_cnn: ResNet + ATN
- no_res: CNN + ATN
- no_att: CNN + ResNet

To reproduce the exact results:

```bash
Torch version: 2.6.0
Compiled with CUDA: 12.2
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
```

### Run Ablation Analysis

To evaluate model performance across real-world datasets (as shown in Table 2 of the paper):

```bash
python ablation.py --data-path="data/real_world_datasets" --model-path="models/MODEL_NAME.pth" --model-name="MODEL_NAME"
```

- Results will be saved in the `results/` directory.

- FFinal trained models will be saved in the `models` directory.

---

## 📊 Summary of Results on Real-World Datasets

| Model                | **Median** | **Mean**  |
|----------------------|------------|-----------|
| **ClustRecNet (CH)** | **0.1775** | **0.2295** |
| Baseline CNN         | 0.1119     | 0.1914    |
| AutoCluster          | 0.1477     | 0.1991    |
| AML4C                | 0.1102     | 0.1440    |
| ML2DAC               | 0.1403     | 0.1898    |

**ClustRecNet (CH)** consistently outperforms all baselines and AutoML-based clustering frameworks in both mean and median Adjusted Rand Index (ARI).

---

## Citation

```bash
# This section will be updated upon paper acceptance.
```


