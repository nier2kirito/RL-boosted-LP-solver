#!/usr/bin/env python3
"""
Simple test to verify that the modified generate_random_lp function 
generates feasible problems using only mathematical verification.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reformulate_lp.data.lp_parser import generate_random_lp


def test_mathematical_feasibility():
    """Test that generated problems are mathematically feasible."""
    
    print("Testing Mathematical Feasibility of Generated LP Problems")
    print("=" * 60)
    print("This test verifies that generated problems have:")
    print("1. A stored feasible solution x where x >= 0")
    print("2. The solution satisfies all constraints Ax <= b")
    print("3. Problem structure is valid")
    print()
    
    test_cases = [
        (5, 3, "Small"),
        (10, 5, "Medium"),  
        (20, 10, "Large"),
        (50, 25, "Very Large"),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for num_vars, num_cons, size_name in test_cases:
        print(f"{size_name} Problems ({num_vars} variables, {num_cons} constraints):")
        print("-" * 50)
        
        for test_id in range(5):  # Test 5 instances per size
            total_tests += 1
            test_passed = True
            
            # Generate feasible problem
            problem = generate_random_lp(
                num_variables=num_vars,
                num_constraints=num_cons,
                density=0.4,
                seed=42 + test_id,
                ensure_feasible=True,
                verify_with_solver=False
            )
            
            print(f"  Test {test_id + 1}: {problem['name']}")
            
            # Check problem structure
            A = problem['A']
            b = problem['b']
            c = problem['c']
            
            # Verify matrix dimensions
            if A.shape != (num_cons, num_vars):
                print(f"    ✗ Matrix A has wrong shape: {A.shape}, expected ({num_cons}, {num_vars})")
                test_passed = False
                continue
                
            if b.shape != (num_cons,):
                print(f"    ✗ Vector b has wrong shape: {b.shape}, expected ({num_cons},)")
                test_passed = False
                continue
                
            if c.shape != (num_vars,):
                print(f"    ✗ Vector c has wrong shape: {c.shape}, expected ({num_vars},)")
                test_passed = False
                continue
            
            print(f"    ✓ Matrix dimensions correct: A{A.shape}, b{b.shape}, c{c.shape}")
            
            # Check sparsity 
            sparsity = np.mean(A == 0)
            print(f"    ✓ Matrix sparsity: {sparsity:.1%} zeros")
            
            # Verify the stored feasible solution exists
            if 'feasible_solution' not in problem:
                print("    ✗ No feasible solution stored in problem")
                test_passed = False
                continue
                
            x_feas = problem['feasible_solution']
            print(f"    ✓ Feasible solution found with shape: {x_feas.shape}")
            
            # Check non-negativity constraint: x >= 0
            if not np.all(x_feas >= -1e-10):  # Small tolerance for numerical precision
                negative_vars = np.sum(x_feas < -1e-10)
                min_value = np.min(x_feas)
                print(f"    ✗ Feasible solution violates x >= 0: {negative_vars} variables negative, min = {min_value:.6f}")
                test_passed = False
                continue
            else:
                print(f"    ✓ All variables non-negative (min value: {np.min(x_feas):.6f})")
            
            # Check constraint satisfaction: Ax <= b
            Ax = A @ x_feas
            constraint_violations = Ax - b
            max_violation = np.max(constraint_violations)
            num_violated = np.sum(constraint_violations > 1e-6)
            
            if num_violated > 0:
                print(f"    ✗ {num_violated} constraints violated, max violation: {max_violation:.6f}")
                test_passed = False
                continue
            else:
                print(f"    ✓ All constraints satisfied (max slack: {-np.min(constraint_violations):.6f})")
            
            # Check that b values are reasonable (not negative, which would indicate issues)
            if np.any(b < 0):
                negative_b = np.sum(b < 0) 
                print(f"    ⚠ Warning: {negative_b} negative RHS values in b")
            
            # Check that A matrix has no all-zero rows
            zero_rows = np.sum(np.all(np.abs(A) < 1e-10, axis=1))
            if zero_rows > 0:
                print(f"    ⚠ Warning: {zero_rows} all-zero rows in constraint matrix")
            
            if test_passed:
                passed_tests += 1
                print("    ✅ Problem is mathematically feasible")
            else:
                print("    ❌ Problem failed feasibility test")
            
            print()
    
    print("=" * 60)
    print(f"SUMMARY: {passed_tests}/{total_tests} problems passed all feasibility tests")
    
    if passed_tests == total_tests:
        print("🎉 SUCCESS: All generated problems are mathematically feasible!")
        print("✅ The ensure_feasible=True option is working correctly.")
    else:
        print("⚠️ PARTIAL SUCCESS: Some problems had issues.")
        success_rate = (passed_tests / total_tests) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    return passed_tests == total_tests


def test_comparison_with_unrestricted():
    """Compare feasible vs unrestricted generation without solver."""
    
    print("\n" + "=" * 60)
    print("Comparing Feasible vs Unrestricted Generation (Mathematical Check)")
    print("=" * 60)
    
    num_tests = 10
    feasible_valid = 0
    unrestricted_valid = 0
    
    for i in range(num_tests):
        # Test feasible generation
        feasible_problem = generate_random_lp(
            num_variables=15, num_constraints=8,
            seed=200 + i, ensure_feasible=True
        )
        
        # Check if the feasible version has a valid stored solution
        if 'feasible_solution' in feasible_problem:
            x = feasible_problem['feasible_solution']
            A = feasible_problem['A']
            b = feasible_problem['b']
            
            # Check feasibility mathematically
            if np.all(x >= -1e-10) and np.all(A @ x <= b + 1e-6):
                feasible_valid += 1
        
        # Test unrestricted generation  
        unrestricted_problem = generate_random_lp(
            num_variables=15, num_constraints=8,
            seed=200 + i, ensure_feasible=False
        )
        
        # For unrestricted, we can't easily verify without a solver
        # But we can check if the problem structure is at least reasonable
        A = unrestricted_problem['A']
        b = unrestricted_problem['b']
        
        # Simple heuristic: if most b values are positive and A has reasonable values
        if np.mean(b > 0) > 0.5 and not np.any(np.isnan(A)) and not np.any(np.isinf(A)):
            unrestricted_valid += 1
    
    print(f"Feasible method: {feasible_valid}/{num_tests} problems have mathematically valid solutions")
    print(f"Unrestricted method: {unrestricted_valid}/{num_tests} problems have reasonable structure")
    print()
    
    if feasible_valid == num_tests:
        print("✅ Perfect: All problems with ensure_feasible=True have valid solutions!")
    else:
        print(f"⚠️ {num_tests - feasible_valid} feasible problems had issues")


if __name__ == "__main__":
    print("Testing LP Problem Generation with Mathematical Verification")
    print("(No solver required - using pure mathematical checks)")
    print()
    
    success = test_mathematical_feasibility()
    test_comparison_with_unrestricted()
    
    if success:
        print("\n" + "="*60)
        print("🎯 CONCLUSION: The modified generate_random_lp function works correctly!")
        print("✅ When ensure_feasible=True, it generates mathematically feasible problems")
        print("✅ All generated problems satisfy x >= 0 and Ax <= b constraints")
        print("✅ The feasible solution is stored and verified in each problem")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ Some issues found in the problem generation.")
        sys.exit(1) 