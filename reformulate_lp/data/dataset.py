"""
Dataset class for handling LP problems for training and evaluation.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import glob
import pickle

from .lp_parser import LPParser, generate_random_lp


class LPDataset(Dataset):
    """
    Dataset class for Linear Programming problems.
    
    Supports loading from various file formats and generating synthetic problems.
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 problems: Optional[List[Dict]] = None,
                 num_synthetic: int = 0,
                 synthetic_config: Optional[Dict] = None,
                 transform=None,
                 cache_dir: Optional[str] = None):
        """
        Initialize LP dataset.
        
        Args:
            data_path: Path to directory containing LP problem files
            problems: List of LP problem dictionaries
            num_synthetic: Number of synthetic problems to generate
            synthetic_config: Configuration for synthetic problem generation
            transform: Optional transform to apply to problems
            cache_dir: Directory to cache processed problems
        """
        self.data_path = data_path
        self.transform = transform
        self.cache_dir = cache_dir
        self.parser = LPParser()
        
        self.problems = []
        
        # Load problems from files
        if data_path and os.path.exists(data_path):
            self.problems.extend(self._load_from_directory(data_path))
        
        # Add provided problems
        if problems:
            self.problems.extend(problems)
        
        # Generate synthetic problems
        if num_synthetic > 0:
            synthetic_problems = self._generate_synthetic_problems(num_synthetic, synthetic_config)
            self.problems.extend(synthetic_problems)
        
        if len(self.problems) == 0:
            raise ValueError("No LP problems loaded. Provide data_path, problems, or set num_synthetic > 0")
        
        print(f"Loaded {len(self.problems)} LP problems")
    
    def __len__(self) -> int:
        return len(self.problems)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get LP problem by index.
        
        Args:
            idx: Problem index
            
        Returns:
            problem: LP problem dictionary
        """
        problem = self.problems[idx].copy()
        
        if self.transform:
            problem = self.transform(problem)
        
        return problem
    
    def _load_from_directory(self, data_path: str) -> List[Dict]:
        """Load LP problems from directory."""
        problems = []
        
        # Supported file patterns
        patterns = ['*.mps', '*.lp', '*.json', '*.npz']
        
        for pattern in patterns:
            files = glob.glob(os.path.join(data_path, pattern))
            for filepath in files:
                try:
                    problem = self.parser.parse_file(filepath)
                    problems.append(problem)
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
        
        return problems
    
    def _generate_synthetic_problems(self, num_problems: int, config: Optional[Dict] = None) -> List[Dict]:
        """Generate synthetic LP problems."""
        if config is None:
            config = {
                'num_variables_range': (5, 50),
                'num_constraints_range': (3, 30),
                'density_range': (0.2, 0.8),
                'seed': 42
            }
        
        problems = []
        np.random.seed(config.get('seed', 42))
        
        for i in range(num_problems):
            # Sample problem dimensions
            num_vars = np.random.randint(*config['num_variables_range'])
            num_cons = np.random.randint(*config['num_constraints_range'])
            density = np.random.uniform(*config['density_range'])
            
            # Generate problem
            problem = generate_random_lp(
                num_variables=num_vars,
                num_constraints=num_cons,
                density=density,
                seed=config.get('seed', 42) + i,
                ensure_feasible=config.get('ensure_feasible', True),
                max_retries=config.get('max_retries', 10),
                verify_with_solver=config.get('verify_with_solver', False)
            )
            
            problem['name'] = f'synthetic_{i:04d}'
            problems.append(problem)
        
        return problems
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.problems:
            return {}
        
        num_vars = [p['num_variables'] for p in self.problems]
        num_cons = [p['num_constraints'] for p in self.problems]
        
        stats = {
            'num_problems': len(self.problems),
            'num_variables': {
                'min': min(num_vars),
                'max': max(num_vars),
                'mean': np.mean(num_vars),
                'std': np.std(num_vars)
            },
            'num_constraints': {
                'min': min(num_cons),
                'max': max(num_cons),
                'mean': np.mean(num_cons),
                'std': np.std(num_cons)
            }
        }
        
        return stats
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
              test_ratio: float = 0.1, seed: int = 42) -> Tuple['LPDataset', 'LPDataset', 'LPDataset']:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            seed: Random seed
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        np.random.seed(seed)
        indices = np.random.permutation(len(self.problems))
        
        n_train = int(len(self.problems) * train_ratio)
        n_val = int(len(self.problems) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_problems = [self.problems[i] for i in train_indices]
        val_problems = [self.problems[i] for i in val_indices]
        test_problems = [self.problems[i] for i in test_indices]
        
        train_dataset = LPDataset(problems=train_problems, transform=self.transform)
        val_dataset = LPDataset(problems=val_problems, transform=self.transform)
        test_dataset = LPDataset(problems=test_problems, transform=self.transform)
        
        return train_dataset, val_dataset, test_dataset
    
    def save_cache(self, cache_path: str):
        """Save dataset to cache file."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self.problems, f)
    
    @classmethod
    def load_cache(cls, cache_path: str, transform=None):
        """Load dataset from cache file."""
        with open(cache_path, 'rb') as f:
            problems = pickle.load(f)
        return cls(problems=problems, transform=transform)


class LPCollator:
    """
    Collator for batching LP problems.
    
    Since LP problems can have different dimensions, we need custom batching.
    """
    
    def __init__(self, pad_to_max: bool = True):
        self.pad_to_max = pad_to_max
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of LP problems.
        
        Args:
            batch: List of LP problem dictionaries
            
        Returns:
            batched_data: Batched LP problems
        """
        if len(batch) == 1:
            # Single item - just add batch dimension
            problem = batch[0]
            return {
                'A': torch.tensor(problem['A']).unsqueeze(0).float(),
                'b': torch.tensor(problem['b']).unsqueeze(0).float(),
                'c': torch.tensor(problem['c']).unsqueeze(0).float(),
                'batch_size': 1,
                'problem_sizes': [(problem['num_constraints'], problem['num_variables'])],
                'names': [problem['name']]
            }
        
        # Multiple items - need to handle different sizes
        batch_size = len(batch)
        
        if self.pad_to_max:
            # Pad all problems to maximum size in batch
            max_constraints = max(p['num_constraints'] for p in batch)
            max_variables = max(p['num_variables'] for p in batch)
            
            A_batch = torch.zeros(batch_size, max_constraints, max_variables)
            b_batch = torch.zeros(batch_size, max_constraints)
            c_batch = torch.zeros(batch_size, max_variables)
            
            for i, problem in enumerate(batch):
                m, n = problem['A'].shape
                A_batch[i, :m, :n] = torch.tensor(problem['A'])
                b_batch[i, :m] = torch.tensor(problem['b'])
                c_batch[i, :n] = torch.tensor(problem['c'])
            
            return {
                'A': A_batch.float(),
                'b': b_batch.float(),
                'c': c_batch.float(),
                'batch_size': batch_size,
                'problem_sizes': [(p['num_constraints'], p['num_variables']) for p in batch],
                'names': [p['name'] for p in batch]
            }
        else:
            # Return list of problems (no padding)
            return {
                'problems': batch,
                'batch_size': batch_size
            }


def create_dataloader(dataset: LPDataset, 
                     batch_size: int = 1,
                     shuffle: bool = False,
                     num_workers: int = 0,
                     pad_to_max: bool = True) -> DataLoader:
    """
    Create DataLoader for LP dataset.
    
    Args:
        dataset: LP dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pad_to_max: Whether to pad problems to max size in batch
        
    Returns:
        dataloader: PyTorch DataLoader
    """
    collator = LPCollator(pad_to_max=pad_to_max)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator
    )


# Dataset transforms
class NormalizeLP:
    """Normalize LP problem coefficients."""
    
    def __init__(self, method='standard'):
        self.method = method
    
    def __call__(self, problem: Dict) -> Dict:
        problem = problem.copy()
        
        if self.method == 'standard':
            # Standardize A matrix
            A = problem['A']
            A_mean = np.mean(A[A != 0])
            A_std = np.std(A[A != 0])
            if A_std > 0:
                A = (A - A_mean) / A_std
                problem['A'] = A
            
            # Normalize b and c
            b = problem['b']
            c = problem['c']
            
            b_norm = np.linalg.norm(b)
            if b_norm > 0:
                problem['b'] = b / b_norm
            
            c_norm = np.linalg.norm(c)
            if c_norm > 0:
                problem['c'] = c / c_norm
        
        return problem


class AddNoise:
    """Add noise to LP problem coefficients."""
    
    def __init__(self, noise_level=0.01, seed=None):
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, problem: Dict) -> Dict:
        problem = problem.copy()
        
        # Add noise to A matrix
        A = problem['A']
        noise_A = self.rng.normal(0, self.noise_level, A.shape)
        problem['A'] = A + noise_A
        
        # Add noise to b and c
        b = problem['b']
        c = problem['c']
        
        noise_b = self.rng.normal(0, self.noise_level, b.shape)
        noise_c = self.rng.normal(0, self.noise_level, c.shape)
        
        problem['b'] = b + noise_b
        problem['c'] = c + noise_c
        
        return problem 