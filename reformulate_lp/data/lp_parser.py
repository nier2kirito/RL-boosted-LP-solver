"""
LP problem parser for various file formats.

Supports parsing MPS, LP, and other linear programming file formats.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Union
import os
import re
import warnings


class LPParser:
    """
    Parser for Linear Programming problems in various formats.
    """
    
    def __init__(self):
        self.supported_formats = ['.mps', '.lp', '.json', '.npz']
        
    def parse_file(self, filepath: str) -> Dict:
        """
        Parse LP problem from file.
        
        Args:
            filepath: Path to the LP problem file
            
        Returns:
            lp_problem: Dictionary containing A, b, c matrices and metadata
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.mps':
            return self.parse_mps(filepath)
        elif ext == '.lp':
            return self.parse_lp(filepath)
        elif ext == '.json':
            return self.parse_json(filepath)
        elif ext == '.npz':
            return self.parse_npz(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def parse_mps(self, filepath: str) -> Dict:
        """
        Parse MPS format LP problem.
        
        Args:
            filepath: Path to MPS file
            
        Returns:
            lp_problem: Parsed LP problem
        """
        try:
            import cvxpy as cp
            
            # Use CVXPY to parse MPS files
            problem = cp.Problem()
            problem = cp.Problem.from_file(filepath)
            
            # Extract problem data
            data = problem.get_problem_data(cp.ECOS)
            
            A = data[0]['A'].toarray() if sp.issparse(data[0]['A']) else data[0]['A']
            b = data[0]['b'].flatten()
            c = data[0]['c'].flatten()
            
            return {
                'A': A,
                'b': b, 
                'c': c,
                'name': os.path.basename(filepath),
                'format': 'mps',
                'num_variables': A.shape[1],
                'num_constraints': A.shape[0]
            }
            
        except ImportError:
            raise ImportError("CVXPY is required for MPS parsing. Install with: pip install cvxpy")
        except Exception as e:
            raise ValueError(f"Error parsing MPS file: {e}")
    
    def parse_lp(self, filepath: str) -> Dict:
        """
        Parse LP format problem (simplified parser).
        
        Args:
            filepath: Path to LP file
            
        Returns:
            lp_problem: Parsed LP problem
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        # This is a simplified LP parser - for production use, consider using
        # a more robust parser like PuLP or CVXPY
        
        lines = content.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('\\')]
        
        # Parse objective
        obj_line = None
        for i, line in enumerate(lines):
            if 'minimize' in line.lower() or 'maximize' in line.lower():
                obj_line = line
                break
        
        if obj_line is None:
            raise ValueError("Could not find objective function")
        
        # Extract variable names and coefficients (simplified)
        variables = self._extract_variables_from_lp(lines)
        constraints = self._extract_constraints_from_lp(lines, variables)
        objective = self._extract_objective_from_lp(obj_line, variables)
        
        # Build matrices
        num_vars = len(variables)
        num_constraints = len(constraints)
        
        A = np.zeros((num_constraints, num_vars))
        b = np.zeros(num_constraints)
        c = np.zeros(num_vars)
        
        # Fill objective vector
        for var, coef in objective.items():
            if var in variables:
                c[variables.index(var)] = coef
        
        # Fill constraint matrix
        for i, constraint in enumerate(constraints):
            for var, coef in constraint['lhs'].items():
                if var in variables:
                    A[i, variables.index(var)] = coef
            b[i] = constraint['rhs']
        
        return {
            'A': A,
            'b': b,
            'c': c,
            'variables': variables,
            'name': os.path.basename(filepath),
            'format': 'lp',
            'num_variables': num_vars,
            'num_constraints': num_constraints
        }
    
    def parse_json(self, filepath: str) -> Dict:
        """
        Parse JSON format LP problem.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            lp_problem: Parsed LP problem
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Expected format: {"A": [[...]], "b": [...], "c": [...]}
        A = np.array(data['A'])
        b = np.array(data['b'])
        c = np.array(data['c'])
        
        return {
            'A': A,
            'b': b,
            'c': c,
            'name': data.get('name', os.path.basename(filepath)),
            'format': 'json',
            'num_variables': A.shape[1],
            'num_constraints': A.shape[0]
        }
    
    def parse_npz(self, filepath: str) -> Dict:
        """
        Parse NumPy NPZ format LP problem.
        
        Args:
            filepath: Path to NPZ file
            
        Returns:
            lp_problem: Parsed LP problem
        """
        data = np.load(filepath)
        
        A = data['A']
        b = data['b']
        c = data['c']
        
        return {
            'A': A,
            'b': b,
            'c': c,
            'name': data.get('name', os.path.basename(filepath)),
            'format': 'npz',
            'num_variables': A.shape[1],
            'num_constraints': A.shape[0]
        }
    
    def parse_dict(self, lp_dict: Dict) -> Dict:
        """
        Parse LP problem from dictionary format.
        
        Args:
            lp_dict: Dictionary containing LP problem data
            
        Returns:
            lp_problem: Standardized LP problem dictionary
        """
        A = np.array(lp_dict['A'])
        b = np.array(lp_dict['b'])
        c = np.array(lp_dict['c'])
        
        return {
            'A': A,
            'b': b,
            'c': c,
            'name': lp_dict.get('name', 'unnamed_problem'),
            'format': 'dict',
            'num_variables': A.shape[1],
            'num_constraints': A.shape[0]
        }
    
    def _extract_variables_from_lp(self, lines: List[str]) -> List[str]:
        """Extract variable names from LP format."""
        variables = set()
        
        for line in lines:
            # Simple regex to find variable names (x1, x2, etc.)
            vars_in_line = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', line)
            for var in vars_in_line:
                if var not in ['minimize', 'maximize', 'subject', 'to', 'end']:
                    variables.add(var)
        
        return sorted(list(variables))
    
    def _extract_constraints_from_lp(self, lines: List[str], variables: List[str]) -> List[Dict]:
        """Extract constraints from LP format."""
        constraints = []
        in_constraints = False
        
        for line in lines:
            if 'subject' in line.lower() and 'to' in line.lower():
                in_constraints = True
                continue
            
            if in_constraints and 'end' in line.lower():
                break
                
            if in_constraints and ('<=', '>=', '=') in line:
                constraint = self._parse_constraint_line(line, variables)
                if constraint:
                    constraints.append(constraint)
        
        return constraints
    
    def _extract_objective_from_lp(self, obj_line: str, variables: List[str]) -> Dict[str, float]:
        """Extract objective function coefficients."""
        objective = {}
        
        # Remove minimize/maximize
        obj_line = re.sub(r'(minimize|maximize)', '', obj_line, flags=re.IGNORECASE).strip()
        
        # Simple parsing - look for coefficient*variable patterns
        for var in variables:
            pattern = rf'([+-]?\s*\d*\.?\d*)\s*\*?\s*{var}\b'
            match = re.search(pattern, obj_line)
            if match:
                coef_str = match.group(1).replace(' ', '')
                if coef_str in ['', '+']:
                    coef = 1.0
                elif coef_str == '-':
                    coef = -1.0
                else:
                    coef = float(coef_str)
                objective[var] = coef
        
        return objective
    
    def _parse_constraint_line(self, line: str, variables: List[str]) -> Optional[Dict]:
        """Parse a single constraint line."""
        # Split by constraint operators
        if '<=' in line:
            lhs, rhs = line.split('<=')
            operator = '<='
        elif '>=' in line:
            lhs, rhs = line.split('>=')
            operator = '>='
        elif '=' in line:
            lhs, rhs = line.split('=')
            operator = '='
        else:
            return None
        
        # Parse LHS coefficients
        lhs_coeffs = {}
        for var in variables:
            pattern = rf'([+-]?\s*\d*\.?\d*)\s*\*?\s*{var}\b'
            match = re.search(pattern, lhs)
            if match:
                coef_str = match.group(1).replace(' ', '')
                if coef_str in ['', '+']:
                    coef = 1.0
                elif coef_str == '-':
                    coef = -1.0
                else:
                    coef = float(coef_str)
                lhs_coeffs[var] = coef
        
        # Parse RHS
        try:
            rhs_val = float(rhs.strip())
        except ValueError:
            return None
        
        return {
            'lhs': lhs_coeffs,
            'rhs': rhs_val,
            'operator': operator
        }
    
    def save_lp_problem(self, lp_problem: Dict, filepath: str, format: str = 'npz'):
        """
        Save LP problem to file.
        
        Args:
            lp_problem: LP problem dictionary
            filepath: Output file path
            format: Output format ('npz', 'json')
        """
        if format == 'npz':
            np.savez(filepath, 
                    A=lp_problem['A'],
                    b=lp_problem['b'], 
                    c=lp_problem['c'],
                    name=lp_problem.get('name', 'unnamed'))
        elif format == 'json':
            import json
            data = {
                'A': lp_problem['A'].tolist(),
                'b': lp_problem['b'].tolist(),
                'c': lp_problem['c'].tolist(),
                'name': lp_problem.get('name', 'unnamed'),
                'num_variables': lp_problem['A'].shape[1],
                'num_constraints': lp_problem['A'].shape[0]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {format}")


def generate_random_lp(num_variables: int, num_constraints: int, 
                      density: float = 0.3, seed: Optional[int] = None, 
                      ensure_feasible: bool = True, max_retries: int = 10,
                      verify_with_solver: bool = False) -> Dict:
    """
    Generate a random LP problem for testing.
    
    Args:
        num_variables: Number of variables
        num_constraints: Number of constraints
        density: Density of non-zero coefficients in constraint matrix
        seed: Random seed
        ensure_feasible: Whether to ensure the generated problem is feasible
        max_retries: Maximum number of retries if ensuring feasibility
        verify_with_solver: Whether to verify feasibility using an actual solver
        
    Returns:
        lp_problem: Random LP problem (guaranteed feasible if ensure_feasible=True)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if ensure_feasible:
        problem = _generate_feasible_lp(num_variables, num_constraints, density, seed, max_retries)
        
        # Optional verification with solver
        if verify_with_solver:
            problem = _verify_problem_feasibility(problem, max_retries)
        
        return problem
    else:
        # Original random generation (may be infeasible)
        return _generate_unrestricted_lp(num_variables, num_constraints, density)


def _generate_feasible_lp(num_variables: int, num_constraints: int, 
                         density: float, seed: Optional[int], max_retries: int) -> Dict:
    """Generate a feasible LP problem by constructing around a feasible solution."""
    
    if seed is not None:
        np.random.seed(seed)
    
    for attempt in range(max_retries):
        try:
            # Step 1: Generate a simple, guaranteed feasible solution
            # Use reasonable positive values that are easy to satisfy
            x_feasible = np.random.uniform(1.0, 5.0, num_variables)
            
            # Step 2: Create constraint matrix A with controlled structure
            # We'll create num_constraints regular constraints plus upper bounds
            A = np.zeros((num_constraints + num_variables, num_variables))
            
            # Regular constraints (first num_constraints rows)
            for i in range(num_constraints):
                # Ensure each constraint has at least one non-zero coefficient
                active_vars = max(1, int(density * num_variables))
                selected_vars = np.random.choice(num_variables, active_vars, replace=False)
                
                for j in selected_vars:
                    # Use positive coefficients to avoid issues
                    A[i, j] = np.random.uniform(0.1, 1.0)
            
            # Add upper bound constraints: x_j <= upper_bound (last num_variables rows)
            upper_bounds = x_feasible + np.random.uniform(2.0, 8.0, num_variables)
            for j in range(num_variables):
                A[num_constraints + j, j] = 1.0
            
            # Step 3: Compute RHS values
            # For regular constraints: generous slack from feasible solution
            regular_constraints = A[:num_constraints, :] @ x_feasible
            regular_slack = np.random.uniform(1.0, 5.0, num_constraints)
            regular_b = regular_constraints + regular_slack
            
            # For upper bound constraints
            upper_b = upper_bounds
            
            # Combine all RHS values
            b = np.concatenate([regular_b, upper_b])
            
            # Step 4: Generate objective coefficients
            # Create more balanced objective to prevent unbounded problems
            # Use a mix of positive and negative coefficients with reasonable magnitudes
            c = np.random.uniform(-0.3, 0.7, num_variables)  # Bias toward positive
            
            # Ensure we have both positive and negative coefficients for balance
            # But make sure positive coefficients dominate to prevent unbounded issues
            num_positive = max(1, int(0.6 * num_variables))  # At least 60% positive
            num_negative = min(int(0.3 * num_variables), num_variables - num_positive)  # At most 30% negative
            
            # Randomly assign signs while maintaining the balance
            indices = np.random.permutation(num_variables)
            for i in range(num_positive):
                c[indices[i]] = abs(c[indices[i]]) + 0.1  # Ensure positive and non-zero
            for i in range(num_positive, num_positive + num_negative):
                c[indices[i]] = -abs(c[indices[i]]) - 0.1  # Ensure negative and non-zero
            # Remaining coefficients stay as originally generated
            
            # Additional safeguard: ensure the objective is not too extreme
            c = np.clip(c, -1.0, 2.0)  # Reasonable bounds
            
            # Final check: ensure at least one positive coefficient exists
            if np.all(c <= 0):
                c[np.random.randint(num_variables)] = np.random.uniform(0.2, 1.0)
            
            # Step 5: Verify feasibility construction
            check_values = A @ x_feasible
            if not np.all(check_values <= b + 1e-10):
                print(f"Warning: Construction verification failed on attempt {attempt + 1}")
                continue
            
            # Step 6: Additional sanity checks
            if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(np.isnan(b)) or np.any(np.isinf(b)):
                continue
            
            if np.any(b <= 0):  # Ensure all RHS values are positive
                b = np.abs(b) + 1.0  # Make positive with some buffer
            
            # Keep ALL constraints including upper bounds to prevent unbounded problems
            # The upper bounds are essential for preventing dual infeasibility
            problem = {
                'A': A,  # Keep all constraints including upper bounds
                'b': b,  # Keep all RHS values
                'c': c,
                'name': f'feasible_lp_{num_variables}x{num_constraints}',
                'format': 'generated',
                'num_variables': num_variables,
                'num_constraints': A.shape[0],  # Update to reflect actual constraint count
                'feasible_solution': x_feasible
            }
            
            return problem
            
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print(f"Warning: Failed to generate feasible LP after {max_retries} attempts")
                return _generate_simple_feasible_lp(num_variables, num_constraints)
            continue
    
    # Fallback
    return _generate_simple_feasible_lp(num_variables, num_constraints)


def _generate_simple_feasible_lp(num_variables: int, num_constraints: int) -> Dict:
    """Generate a simple feasible LP as a last resort."""
    # Create a very simple feasible problem
    # Use identity-like structure to ensure feasibility
    
    x_feasible = np.ones(num_variables)  # Simple feasible solution: all variables = 1
    
    # Create a simple constraint matrix
    A = np.zeros((num_constraints, num_variables))
    
    # Add some simple constraints of the form: x_i <= 10
    for i in range(min(num_constraints, num_variables)):
        A[i, i] = 1.0
    
    # For additional constraints, add some random but safe constraints
    for i in range(num_variables, num_constraints):
        # Add constraint like: 0.5*x_1 + 0.5*x_2 + ... <= num_variables
        A[i, :] = 0.5
    
    # Set RHS to be safely above the feasible solution values
    b = A @ x_feasible + 10  # Large slack to ensure feasibility
    
    # Conservative objective to prevent unbounded problems
    # Use mostly positive coefficients with small magnitude
    c = np.random.uniform(0.1, 1.0, num_variables)  # All positive coefficients
    
    # Add a few negative coefficients for variety, but keep them small
    num_negative = min(3, num_variables // 4)  # At most 25% negative, max 3
    if num_negative > 0:
        neg_indices = np.random.choice(num_variables, num_negative, replace=False)
        c[neg_indices] = np.random.uniform(-0.3, -0.1, num_negative)  # Small negative values
    
    return {
        'A': A,
        'b': b,
        'c': c,
        'name': f'simple_feasible_lp_{num_variables}x{num_constraints}',
        'format': 'generated',
        'num_variables': num_variables,
        'num_constraints': num_constraints,
        'feasible_solution': x_feasible
    }


def _generate_unrestricted_lp(num_variables: int, num_constraints: int, density: float) -> Dict:
    """Generate unrestricted LP problem (original method, may be infeasible)."""
    # Generate sparse constraint matrix
    A = np.random.rand(num_constraints, num_variables)
    A[np.random.rand(num_constraints, num_variables) > density] = 0
    
    # Generate RHS and objective
    b = np.random.rand(num_constraints) * 10
    c = np.random.randn(num_variables)
    
    return {
        'A': A,
        'b': b,
        'c': c,
        'name': f'random_lp_{num_variables}x{num_constraints}',
        'format': 'generated',
        'num_variables': num_variables,
        'num_constraints': num_constraints
    } 


def _verify_problem_feasibility(problem: Dict, max_retries: int) -> Dict:
    """
    Verify that a generated problem is actually feasible using a solver.
    
    Args:
        problem: LP problem dictionary
        max_retries: Maximum retries if verification fails
        
    Returns:
        problem: Verified feasible problem (may be regenerated if original was infeasible)
    """
    try:
        # Import solver (lazy import to avoid dependency issues)
        from ..solvers.clp_solver import CLPSolver
        
        solver = CLPSolver(verbose=False, time_limit=5.0)  # Quick solve with timeout
        result = solver.solve(problem)
        
        if result['success']:
            # Problem is verified feasible
            problem['solver_verified'] = True
            return problem
        else:
            # Problem is not feasible, regenerate
            print(f"Warning: Generated problem was not feasible (status: {result['status']}), regenerating...")
            return _generate_feasible_lp(
                problem['num_variables'], 
                problem['num_constraints'], 
                0.3,  # Default density
                None,  # New random seed
                max_retries
            )
            
    except ImportError:
        # Solver not available, return problem as-is with warning
        print("Warning: CLP solver not available for verification, skipping solver check")
        problem['solver_verified'] = False
        return problem
    except Exception as e:
        # Solver error, return problem with warning
        print(f"Warning: Solver verification failed with error: {e}")
        problem['solver_verified'] = False
        return problem 