"""
Gurobi solver interface with deterministic mode support.

Implements the SolverInterface for Gurobi Optimizer with deterministic settings
to provide consistent and reproducible solving times.
"""

import numpy as np
from typing import Dict, Optional
import time

from .solver_interface import SolverInterface


class GurobiSolver(SolverInterface):
    """
    Interface to Gurobi Optimizer with deterministic mode.
    
    Uses Gurobi's deterministic settings to ensure consistent solving behavior
    and reproducible timing results across runs.
    """
    
    def __init__(self, 
                 time_limit: Optional[float] = None,
                 iteration_limit: Optional[int] = None,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 deterministic: bool = True,
                 method: int = -1,  # -1 = automatic, 0 = primal simplex, 1 = dual simplex, 2 = barrier
                 threads: int = 1):  # Use single thread for deterministic behavior
        """
        Initialize Gurobi solver.
        
        Args:
            time_limit: Maximum solving time in seconds
            iteration_limit: Maximum number of iterations
            tolerance: Numerical tolerance
            verbose: Whether to print solver output
            deterministic: Whether to enable deterministic mode
            method: Solving method (0=primal, 1=dual, 2=barrier, -1=auto)
            threads: Number of threads (1 for deterministic behavior)
        """
        super().__init__(time_limit, iteration_limit, tolerance, verbose)
        self.deterministic = deterministic
        self.method = method
        self.threads = threads
        
        # Try to import Gurobi
        self.gurobi_available = False
        try:
            import gurobipy as gp
            self.gp = gp
            self.gurobi_available = True
            if verbose:
                print("Using Gurobi Optimizer")
        except ImportError:
            raise ImportError("Gurobi (gurobipy) is required but not available")
    
    def solve(self, lp_problem: Dict) -> Dict:
        """
        Solve LP problem using Gurobi with deterministic settings.
        
        Args:
            lp_problem: Dictionary with 'A', 'b', 'c' matrices
            
        Returns:
            result: Solution result dictionary
        """
        if not self.gurobi_available:
            raise RuntimeError("Gurobi is not available")
        
        A = np.array(lp_problem['A'])
        b = np.array(lp_problem['b'])
        c = np.array(lp_problem['c'])
        
        try:
            # Create model
            with self.gp.Env(empty=True) as env:
                if not self.verbose:
                    env.setParam('OutputFlag', 0)
                env.start()
                
                model = self.gp.Model(env=env)
                
                # Set deterministic parameters
                if self.deterministic:
                    model.setParam('Seed', 12345)  # Fixed random seed
                    model.setParam('Threads', self.threads)  # Single thread for determinism
                    model.setParam('Method', self.method)  # Fixed method
                    model.setParam('NumericFocus', 3)  # Maximum numerical precision
                    model.setParam('Quad', 1)  # Use quad precision if needed
                
                # Set other parameters
                if self.time_limit is not None:
                    model.setParam('TimeLimit', self.time_limit)
                
                if self.iteration_limit is not None:
                    model.setParam('IterationLimit', self.iteration_limit)
                
                model.setParam('OptimalityTol', self.tolerance)
                model.setParam('FeasibilityTol', self.tolerance)
                
                if not self.verbose:
                    model.setParam('OutputFlag', 0)
                
                # Add variables: x >= 0
                num_vars = len(c)
                x = model.addVars(num_vars, name="x", lb=0.0)
                
                # Add constraints: Ax <= b
                for i in range(len(b)):
                    model.addConstr(
                        self.gp.quicksum(A[i, j] * x[j] for j in range(num_vars)) <= b[i],
                        name=f"constr_{i}"
                    )
                
                # Set objective: minimize c^T x
                model.setObjective(
                    self.gp.quicksum(c[j] * x[j] for j in range(num_vars)),
                    self.gp.GRB.MINIMIZE
                )
                
                # Solve
                start_time = time.time()
                model.optimize()
                solve_time = time.time() - start_time
                
                # Extract results
                status = model.Status
                
                if status == self.gp.GRB.OPTIMAL:
                    success = True
                    objective_value = model.ObjVal
                    solution = np.array([x[j].X for j in range(num_vars)])
                    status_msg = 'optimal'
                elif status == self.gp.GRB.INFEASIBLE:
                    success = False
                    objective_value = None
                    solution = None
                    status_msg = 'infeasible'
                elif status == self.gp.GRB.UNBOUNDED:
                    success = False
                    objective_value = None
                    solution = None
                    status_msg = 'unbounded'
                elif status == self.gp.GRB.TIME_LIMIT:
                    success = False
                    objective_value = model.ObjVal if model.SolCount > 0 else None
                    solution = np.array([x[j].X for j in range(num_vars)]) if model.SolCount > 0 else None
                    status_msg = 'time_limit'
                elif status == self.gp.GRB.ITERATION_LIMIT:
                    success = False
                    objective_value = model.ObjVal if model.SolCount > 0 else None
                    solution = np.array([x[j].X for j in range(num_vars)]) if model.SolCount > 0 else None
                    status_msg = 'iteration_limit'
                else:
                    success = False
                    objective_value = None
                    solution = None
                    status_msg = f'status_{status}'
                
                # Get iteration count
                iterations = int(model.IterCount) if hasattr(model, 'IterCount') else 0
                
                return {
                    'success': success,
                    'objective_value': objective_value,
                    'solution': solution,
                    'iterations': iterations,
                    'solve_time': solve_time,
                    'status': status_msg,
                    'solver_info': {
                        'solver_name': 'Gurobi',
                        'deterministic': self.deterministic,
                        'method': self.method,
                        'threads': self.threads,
                        'gurobi_status': status
                    }
                }
                
        except Exception as e:
            return {
                'success': False,
                'status': f'error: {str(e)}',
                'solve_time': 0.0,
                'objective_value': None,
                'solution': None,
                'iterations': None,
                'solver_info': {
                    'solver_name': 'Gurobi',
                    'error': str(e)
                }
            }
    
    def get_solver_info(self) -> Dict:
        """
        Get information about the Gurobi solver configuration.
        
        Returns:
            info: Dictionary with solver information
        """
        base_info = super().get_solver_info()
        base_info.update({
            'solver_name': 'GurobiSolver',
            'deterministic': self.deterministic,
            'method': self.method,
            'threads': self.threads,
            'gurobi_available': self.gurobi_available
        })
        return base_info 