#!/usr/bin/env python3
"""
Test script to verify COIN-OR CLP is working with our implementation.
"""

print("Testing COIN-OR CLP installation...")

# Test 1: Check if CyLP is available
try:
    import cylp
    from cylp.cy import CyClpSimplex
    print("✓ CyLP Python bindings are available")
    cylp_available = True
except ImportError as e:
    print(f"✗ CyLP not available: {e}")
    cylp_available = False

# Test 2: Check if our CLPSolver works
try:
    from reformulate_lp.solvers.clp_solver import CLPSolver
    print("✓ CLPSolver class imported successfully")
    
    # Create a simple test problem
    import numpy as np
    
    # Simple LP: minimize x + y subject to x + y >= 1, x >= 0, y >= 0
    # Standard form: minimize [1, 1] * [x, y] subject to [-1, -1] * [x, y] <= -1
    A = np.array([[-1.0, -1.0]])  # Constraint matrix
    b = np.array([-1.0])          # RHS
    c = np.array([1.0, 1.0])      # Objective coefficients
    
    lp_problem = {
        'A': A,
        'b': b,
        'c': c,
        'name': 'test_problem'
    }
    
    print("✓ Test LP problem created")
    
    # Test solving
    solver = CLPSolver(verbose=True, time_limit=10.0)
    print("✓ CLPSolver initialized")
    
    result = solver.solve(lp_problem)
    print("✓ LP problem solved")
    
    print(f"Solution result: {result}")
    
    if result['success']:
        print("✓ LP solved successfully!")
        print(f"  Objective value: {result.get('objective_value', 'N/A')}")
        print(f"  Iterations: {result.get('iterations', 'N/A')}")
        print(f"  Solve time: {result.get('solve_time', 'N/A'):.4f}s")
        print(f"  Solution: {result.get('solution', 'N/A')}")
    else:
        print(f"✗ LP solving failed: {result.get('status', 'Unknown error')}")
        
except Exception as e:
    print(f"✗ Error testing CLPSolver: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test CVXPY fallback
try:
    from reformulate_lp.solvers.clp_solver import CVXPYSolver
    
    cvxpy_solver = CVXPYSolver(verbose=True)
    result = cvxpy_solver.solve(lp_problem)
    
    if result['success']:
        print("✓ CVXPY fallback solver works!")
    else:
        print(f"✗ CVXPY solver failed: {result.get('status', 'Unknown')}")
        
except Exception as e:
    print(f"✗ Error testing CVXPY: {e}")

# Test 4: Basic reformulation system test
try:
    from reformulate_lp import ReformulationSystem
    from reformulate_lp.data.lp_parser import generate_random_lp
    
    print("\nTesting ReformulationSystem...")
    
    # Generate a random problem
    test_lp = generate_random_lp(
        num_variables=10,
        num_constraints=5,
        density=0.3,
        seed=42
    )
    
    print("✓ Random LP problem generated")
    
    # Create reformulation system
    reformulator = ReformulationSystem(
        gnn_hidden_dim=32,
        pointer_hidden_dim=64,
        num_clusters=5
    )
    
    print("✓ ReformulationSystem created")
    
    # Test reformulation
    reformulated_lp, info = reformulator.reformulate(test_lp, temperature=1.0)
    
    print("✓ LP reformulation completed")
    print(f"  Permutation: {info['permutation']}")
    print(f"  Variable reordering: {reformulated_lp['variable_order']}")
    
except Exception as e:
    print(f"✗ Error testing ReformulationSystem: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("INSTALLATION TEST COMPLETED")
print("="*50)
print("If you see ✓ for most tests, your installation is working!")
print("You can now use:")
print("  python train.py --config configs/default.yaml")
print("  python evaluate.py --model_path <model_path>")
print("  python example_usage.py") 