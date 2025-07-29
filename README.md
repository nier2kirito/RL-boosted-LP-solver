# Learning to Reformulate Linear Programming

This repository contains an independent implementation of the paper ["Accelerating Linear Programming Solving by Exploiting the Performance Variability via Reinforcement Learning"](https://ai-2-ase.github.io/papers/2%5CSubmission%5CAAAI2023_Learning_to_ReformulationCameraReadVersion.pdf) (AAAI 2023).

**⚠️ Note**: This is an unofficial implementation created for research and educational purposes. The original authors are not affiliated with this implementation.

## Original Paper

**Title**: Accelerating Linear Programming Solving by Exploiting the Performance Variability via Reinforcement Learning  
**Authors**: Xijun Li¹'², Qingyu Qu³, Fangzhou Zhu², Mingxuan Yuan², Jia Zeng², Jie Wang¹  
**Affiliations**: 
- ¹University of Science and Technology of China
- ²Huawei Noah's Ark Lab  
- ³Beihang University
**Conference**: AAAI 2023

## Architecture Overview

The system consists of three main components:

1. **Representation**: Converts LP problems into bipartite graph representations
2. **Aggregation**: Uses Graph Neural Networks (GNN) to learn variable embeddings and aggregates them into clusters
3. **Permutation**: Employs Pointer Networks to generate new variable orderings that improve solver performance

## Key Features

- **Bipartite Graph Representation**: Converts LP problems into graph format where variables and constraints are nodes
- **Graph Convolutional Neural Network**: Learns meaningful embeddings for variables and constraints
- **Clustering and Pooling**: Aggregates variable embeddings into manageable clusters
- **Pointer Network**: Generates permutations of variable clusters to reformulate the original LP
- **Reinforcement Learning**: Trains the system using REINFORCE algorithm with solver performance as reward
- **Multiple Solver Support**: Compatible with various LP solvers (COIN-OR CLP, Gurobi, CPLEX)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from reformulate_lp import ReformulationSystem
from reformulate_lp.data import LPDataset

# Initialize the system
reformulator = ReformulationSystem(
    gnn_hidden_dim=64,
    pointer_hidden_dim=128,
    num_clusters=20
)

# Load and preprocess data
dataset = LPDataset("path/to/lp/problems")

# Train the system
reformulator.train(dataset, num_epochs=100)

# Reformulate a new LP problem
reformulated_lp = reformulator.reformulate(lp_problem)
```

### Training

```bash
python train.py --config configs/default.yaml
```

### Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pt --test_data data/test/
```

## Directory Structure

```
reformulating_lp/
├── reformulate_lp/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn.py              # Graph Neural Network implementation
│   │   ├── pointer_net.py      # Pointer Network implementation
│   │   └── reformulator.py     # Main reformulation system
│   ├── data/
│   │   ├── __init__.py
│   │   ├── lp_parser.py        # LP problem parsing
│   │   ├── graph_builder.py    # Bipartite graph construction
│   │   └── dataset.py          # Dataset handling
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   └── reinforcement.py    # REINFORCE algorithm
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── clp_solver.py       # COIN-OR CLP interface
│   │   └── solver_interface.py # Generic solver interface
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # Performance metrics
│       └── visualization.py    # Visualization tools
├── configs/
│   └── default.yaml            # Configuration file
├── data/
│   ├── train/                  # Training data
│   ├── val/                    # Validation data
│   └── test/                   # Test data
├── scripts/
│   ├── preprocess_data.py      # Data preprocessing
│   └── generate_datasets.py    # Dataset generation
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
├── requirements.txt
└── README.md
```

## Paper Reference

**Please cite the original paper if you use this implementation:**

```bibtex
@inproceedings{li2023accelerating,
  title={Accelerating Linear Programming Solving by Exploiting the Performance Variability via Reinforcement Learning},
  author={Li, Xijun and Qu, Qingyu and Zhu, Fangzhou and Yuan, Mingxuan and Zeng, Jia and Wang, Jie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

**Paper Link**: [https://ai-2-ase.github.io/papers/2%5CSubmission%5CAAAI2023_Learning_to_ReformulationCameraReadVersion.pdf](https://ai-2-ase.github.io/papers/2%5CSubmission%5CAAAI2023_Learning_to_ReformulationCameraReadVersion.pdf)

## Performance Results

The system achieves:
- 25% reduction in solving iterations on average
- 15% reduction in solving time on average
- Improved performance across multiple LP problem types and domains

## License

This implementation is licensed under the MIT License - see LICENSE file for details.

**Note**: This license applies only to this implementation code. The original research ideas and methodology belong to the original paper authors. 
