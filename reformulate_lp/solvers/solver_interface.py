"""
Generic interface for LP solvers.

Provides a unified interface for different LP solvers to evaluate
the performance of reformulated problems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import time


class SolverInterface(ABC):
    """
    Abstract base class for LP solver interfaces.
    
    All solver implementations should inherit from this class
    and implement the solve method.
    """
    
    def __init__(self, 
                 time_limit: Optional[float] = None,
                 iteration_limit: Optional[int] = None,
                 tolerance: float = 1e-6,
                 verbose: bool = False):
        """
        Initialize solver interface.
        
        Args:
            time_limit: Maximum solving time in seconds
            iteration_limit: Maximum number of iterations
            tolerance: Numerical tolerance for optimality
            verbose: Whether to print solver output
        """
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.tolerance = tolerance
        self.verbose = verbose
    
    @abstractmethod
    def solve(self, lp_problem: Dict) -> Dict:
        """
        Solve the LP problem.
        
        Args:
            lp_problem: Dictionary containing 'A', 'b', 'c' matrices
            
        Returns:
            result: Dictionary containing solution information
                - success: bool, whether solve was successful
                - objective_value: float, optimal objective value
                - solution: array, optimal variable values
                - iterations: int, number of iterations taken
                - solve_time: float, time taken to solve
                - status: str, solver status message
        """
        pass
    
    def solve_with_timeout(self, lp_problem: Dict, timeout: Optional[float] = None) -> Dict:
        """
        Solve LP problem with timeout handling.
        
        Args:
            lp_problem: LP problem dictionary
            timeout: Timeout in seconds (overrides instance timeout)
            
        Returns:
            result: Solution result dictionary
        """
        if timeout is None:
            timeout = self.time_limit
        
        start_time = time.time()
        
        try:
            result = self.solve(lp_problem)
            solve_time = time.time() - start_time
            
            if timeout and solve_time > timeout:
                return {
                    'success': False,
                    'status': 'timeout',
                    'solve_time': solve_time,
                    'objective_value': None,
                    'solution': None,
                    'iterations': None
                }
            
            result['solve_time'] = solve_time
            return result
            
        except Exception as e:
            return {
                'success': False,
                'status': f'error: {str(e)}',
                'solve_time': time.time() - start_time,
                'objective_value': None,
                'solution': None,
                'iterations': None
            }
    
    def compare_solutions(self, original_lp: Dict, reformulated_lp: Dict) -> Dict:
        """
        Compare solving performance between original and reformulated LP.
        
        Args:
            original_lp: Original LP problem
            reformulated_lp: Reformulated LP problem
            
        Returns:
            comparison: Dictionary with comparison metrics
        """
        # Solve both problems
        original_result = self.solve_with_timeout(original_lp)
        reformulated_result = self.solve_with_timeout(reformulated_lp)
        
        comparison = {
            'original_result': original_result,
            'reformulated_result': reformulated_result,
            'both_successful': original_result['success'] and reformulated_result['success']
        }
        
        if comparison['both_successful']:
            # Compute improvement metrics
            orig_time = original_result['solve_time']
            reform_time = reformulated_result['solve_time']
            
            orig_iters = original_result.get('iterations', 0)
            reform_iters = reformulated_result.get('iterations', 0)
            
            comparison.update({
                'time_improvement': (orig_time - reform_time) / orig_time if orig_time > 0 else 0,
                'iteration_improvement': (orig_iters - reform_iters) / orig_iters if orig_iters > 0 else 0,
                'time_ratio': reform_time / orig_time if orig_time > 0 else float('inf'),
                'iteration_ratio': reform_iters / orig_iters if orig_iters > 0 else float('inf')
            })
        else:
            comparison.update({
                'time_improvement': -1.0,  # Penalty for failure
                'iteration_improvement': -1.0,
                'time_ratio': float('inf'),
                'iteration_ratio': float('inf')
            })
        
        return comparison
    
    def get_solver_info(self) -> Dict:
        """
        Get information about the solver.
        
        Returns:
            info: Dictionary with solver information
        """
        return {
            'solver_name': self.__class__.__name__,
            'time_limit': self.time_limit,
            'iteration_limit': self.iteration_limit,
            'tolerance': self.tolerance,
            'verbose': self.verbose
        } 