"""
COIN-OR CLP solver interface.

Implements the SolverInterface for the COIN-OR Linear Programming solver.
"""

import numpy as np
from typing import Dict, Optional
import time
import subprocess
import tempfile
import os

from .solver_interface import SolverInterface


class CLPSolver(SolverInterface):
    """
    Interface to COIN-OR CLP solver.
    
    Uses the CLP command-line interface or Python bindings if available.
    """
    
    def __init__(self, 
                 time_limit: Optional[float] = None,
                 iteration_limit: Optional[int] = None,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 use_python_bindings: bool = True):
        """
        Initialize CLP solver.
        
        Args:
            time_limit: Maximum solving time in seconds
            iteration_limit: Maximum number of iterations
            tolerance: Numerical tolerance
            verbose: Whether to print solver output
            use_python_bindings: Whether to use Python bindings (if available)
        """
        super().__init__(time_limit, iteration_limit, tolerance, verbose)
        self.use_python_bindings = use_python_bindings
        
        # Try to import CLP Python bindings
        self.clp_available = False
        self.cylp_available = False
        
        if use_python_bindings:
            try:
                import cylp
                from cylp.cy import CyClpSimplex
                self.cylp_available = True
                if verbose:
                    print("Using CyLP Python bindings")
            except ImportError:
                try:
                    import coinor.clp as clp
                    self.clp_available = True
                    if verbose:
                        print("Using COIN-OR CLP Python bindings")
                except ImportError:
                    if verbose:
                        print("Python bindings not available, using command-line interface")
    
    def solve(self, lp_problem: Dict) -> Dict:
        """
        Solve LP problem using CLP.
        
        Args:
            lp_problem: Dictionary with 'A', 'b', 'c' matrices
            
        Returns:
            result: Solution result dictionary
        """
        A = np.array(lp_problem['A'])
        b = np.array(lp_problem['b'])
        c = np.array(lp_problem['c'])
        
        if self.cylp_available:
            return self._solve_with_cylp(A, b, c)
        elif self.clp_available:
            return self._solve_with_clp_bindings(A, b, c)
        else:
            return self._solve_with_command_line(A, b, c)
    
    def _solve_with_cylp(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> Dict:
        """Solve using CyLP bindings."""
        try:
            from cylp.cy import CyClpSimplex
            from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
            
            # Create model
            model = CyLPModel()
            
            # Add variables
            num_vars = len(c)
            x = model.addVariable('x', num_vars)
            
            # Set non-negativity bounds: x >= 0
            x.lower = 0
            
            # Add constraints
            for i in range(len(b)):
                model += CyLPArray(A[i, :]) * x <= b[i]
            
            # Set objective
            model.objective = CyLPArray(c) * x
            
            # Create solver
            solver = CyClpSimplex(model)
            
            # Set solver options (skip time/iteration limits for now)
            solver.primalTolerance = self.tolerance
            solver.dualTolerance = self.tolerance
            
            if not self.verbose:
                solver.logLevel = 0
            
            # Solve
            start_time = time.time()
            status = solver.primal()
            solve_time = time.time() - start_time
            
            # Extract results - handle both string and integer status codes
            if status == 0 or status == 'optimal':  # Optimal
                # Try to get iteration count with fallback
                try:
                    iterations = solver.numberIterations
                except AttributeError:
                    try:
                        iterations = solver.getIterationCount()
                    except AttributeError:
                        try:
                            iterations = solver.iteration
                        except AttributeError:
                            iterations = 0  # Fallback if no iteration count available
                
                return {
                    'success': True,
                    'status': 'optimal',
                    'objective_value': solver.objectiveValue,
                    'solution': np.array(solver.primalVariableSolution['x']),
                    'iterations': iterations,
                    'solve_time': solve_time
                }
            else:
                # Try to get iteration count with fallback
                try:
                    iterations = solver.numberIterations
                except AttributeError:
                    try:
                        iterations = solver.getIterationCount()
                    except AttributeError:
                        try:
                            iterations = solver.iteration
                        except AttributeError:
                            iterations = 0  # Fallback if no iteration count available
                
                return {
                    'success': False,
                    'status': f'solver_status_{status}',
                    'objective_value': None,
                    'solution': None,
                    'iterations': iterations,
                    'solve_time': solve_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'status': f'error: {str(e)}',
                'objective_value': None,
                'solution': None,
                'iterations': 0,
                'solve_time': 0.0
            }
    
    def _solve_with_clp_bindings(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> Dict:
        """Solve using COIN-OR CLP Python bindings."""
        try:
            import coinor.clp as clp
            
            # Create problem
            prob = clp.CoinModel()
            
            # Add variables
            num_vars = len(c)
            for j in range(num_vars):
                prob.addCol(0, float('inf'), c[j], f'x{j}')
            
            # Add constraints
            for i in range(len(b)):
                indices = []
                coeffs = []
                for j in range(num_vars):
                    if abs(A[i, j]) > 1e-10:
                        indices.append(j)
                        coeffs.append(A[i, j])
                prob.addRow(indices, coeffs, -float('inf'), b[i])
            
            # Create solver and solve
            solver = clp.ClpSimplex()
            solver.loadFromCoinModel(prob)
            
            if not self.verbose:
                solver.setLogLevel(0)
            
            start_time = time.time()
            status = solver.primal()
            solve_time = time.time() - start_time
            
            if status == 0:  # Optimal
                # Try to get iteration count with fallback
                try:
                    iterations = solver.numberIterations()
                except (AttributeError, TypeError):
                    try:
                        iterations = solver.getIterationCount()
                    except (AttributeError, TypeError):
                        try:
                            iterations = solver.iteration
                        except (AttributeError, TypeError):
                            iterations = 0  # Fallback if no iteration count available
                
                return {
                    'success': True,
                    'status': 'optimal',
                    'objective_value': solver.objectiveValue(),
                    'solution': np.array([solver.primalColumnSolution()[j] for j in range(num_vars)]),
                    'iterations': iterations,
                    'solve_time': solve_time
                }
            else:
                # Try to get iteration count with fallback
                try:
                    iterations = solver.numberIterations()
                except (AttributeError, TypeError):
                    try:
                        iterations = solver.getIterationCount()
                    except (AttributeError, TypeError):
                        try:
                            iterations = solver.iteration
                        except (AttributeError, TypeError):
                            iterations = 0  # Fallback if no iteration count available
                
                return {
                    'success': False,
                    'status': f'solver_status_{status}',
                    'objective_value': None,
                    'solution': None,
                    'iterations': iterations,
                    'solve_time': solve_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'status': f'error: {str(e)}',
                'objective_value': None,
                'solution': None,
                'iterations': 0,
                'solve_time': 0.0
            }
    
    def _solve_with_command_line(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> Dict:
        """Solve using CLP command-line interface."""
        try:
            # Create temporary MPS file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mps', delete=False) as f:
                mps_file = f.name
                self._write_mps_file(A, b, c, f)
            
            # Prepare CLP command
            cmd = ['clp', mps_file, '-solve', '-solution', '-']
            
            if not self.verbose:
                cmd.extend(['-log', '0'])
            
            # Run CLP
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.time_limit)
            solve_time = time.time() - start_time
            
            # Clean up
            os.unlink(mps_file)
            
            # Parse output
            if result.returncode == 0:
                return self._parse_clp_output(result.stdout, solve_time)
            else:
                return {
                    'success': False,
                    'status': f'command_failed: {result.stderr}',
                    'objective_value': None,
                    'solution': None,
                    'iterations': 0,
                    'solve_time': solve_time
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'status': 'timeout',
                'objective_value': None,
                'solution': None,
                'iterations': 0,
                'solve_time': self.time_limit or 0.0
            }
        except Exception as e:
            return {
                'success': False,
                'status': f'error: {str(e)}',
                'objective_value': None,
                'solution': None,
                'iterations': 0,
                'solve_time': 0.0
            }
    
    def _write_mps_file(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, file):
        """Write problem in MPS format."""
        num_constraints, num_vars = A.shape
        
        # MPS file header
        file.write("NAME          PROBLEM\n")
        file.write("ROWS\n")
        file.write(" N  OBJ\n")
        
        # Write constraint rows
        for i in range(num_constraints):
            file.write(f" L  R{i:06d}\n")
        
        file.write("COLUMNS\n")
        
        # Write variables and coefficients
        for j in range(num_vars):
            var_name = f"X{j:06d}"
            
            # Objective coefficient
            if abs(c[j]) > 1e-10:
                file.write(f"    {var_name}  OBJ       {c[j]:.10e}\n")
            
            # Constraint coefficients
            for i in range(num_constraints):
                if abs(A[i, j]) > 1e-10:
                    file.write(f"    {var_name}  R{i:06d}    {A[i, j]:.10e}\n")
        
        file.write("RHS\n")
        
        # Write RHS values
        for i in range(num_constraints):
            if abs(b[i]) > 1e-10:
                file.write(f"    RHS1      R{i:06d}    {b[i]:.10e}\n")
        
        file.write("ENDATA\n")
    
    def _parse_clp_output(self, output: str, solve_time: float) -> Dict:
        """Parse CLP command-line output."""
        lines = output.strip().split('\n')
        
        # Look for solution status
        status = 'unknown'
        objective_value = None
        iterations = 0
        
        for line in lines:
            if 'Optimal solution found' in line:
                status = 'optimal'
            elif 'Primal infeasible' in line:
                status = 'infeasible'
            elif 'Dual infeasible' in line:
                status = 'unbounded'
            elif 'Objective value:' in line:
                try:
                    objective_value = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'iterations' in line.lower():
                try:
                    # Extract iteration count
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'iteration' in part.lower() and i > 0:
                            iterations = int(parts[i-1])
                            break
                except:
                    pass
        
        success = status == 'optimal'
        
        return {
            'success': success,
            'status': status,
            'objective_value': objective_value,
            'solution': None,  # Command-line interface doesn't return solution values easily
            'iterations': iterations,
            'solve_time': solve_time
        }


class CVXPYSolver(SolverInterface):
    """
    Fallback solver using CVXPY (if CLP is not available).
    """
    
    def __init__(self, 
                 time_limit: Optional[float] = None,
                 iteration_limit: Optional[int] = None,
                 tolerance: float = 1e-6,
                 verbose: bool = False):
        super().__init__(time_limit, iteration_limit, tolerance, verbose)
        
        try:
            import cvxpy as cp
            self.cvxpy_available = True
        except ImportError:
            self.cvxpy_available = False
            raise ImportError("CVXPY is required when CLP is not available")
    
    def solve(self, lp_problem: Dict) -> Dict:
        """Solve using CVXPY."""
        try:
            import cvxpy as cp
            
            A = np.array(lp_problem['A'])
            b = np.array(lp_problem['b'])
            c = np.array(lp_problem['c'])
            
            # Create variables
            x = cp.Variable(len(c))
            
            # Create problem
            objective = cp.Minimize(c.T @ x)
            constraints = [A @ x <= b, x >= 0]
            
            prob = cp.Problem(objective, constraints)
            
            # Solve
            start_time = time.time()
            prob.solve(verbose=self.verbose)
            solve_time = time.time() - start_time
            
            if prob.status == cp.OPTIMAL:
                return {
                    'success': True,
                    'status': 'optimal',
                    'objective_value': prob.value,
                    'solution': x.value,
                    'iterations': None,  # CVXPY doesn't expose iteration count
                    'solve_time': solve_time
                }
            else:
                return {
                    'success': False,
                    'status': prob.status,
                    'objective_value': None,
                    'solution': None,
                    'iterations': None,
                    'solve_time': solve_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'status': f'error: {str(e)}',
                'objective_value': None,
                'solution': None,
                'iterations': None,
                'solve_time': 0.0
            } 