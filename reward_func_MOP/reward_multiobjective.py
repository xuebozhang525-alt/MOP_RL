import re
import numpy as np
import os
import json
import subprocess
import tempfile
import shutil
import sys

# Add the current directory to sys.path to allow importing local modules like executor and content_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# from executor import PythonExecutor
# from content_utils import extract_code_block 
# Avoid using content_utils.extract_code_block because it injects print statements 
# which might interfere with the file-based verification logic.

def extract_code_block(text, lang='python'):
    """
    Extract code block from text. Supports markdown ``` and xml <python> tags.
    """
    # Try markdown first
    pattern = r'```{}\s*(.*?)```'.format(lang)
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try xml tags
    pattern = r'<{0}>(.*?)</{0}>'.format(lang)
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: check if the whole text looks like code (simple heuristic)
    if "import gurobipy" in text or "from gurobipy" in text:
        return text.strip()
        
    return None

# 定义权重
W_FORMAT = 0.5
W_ACC = 1.0
W_PARADO = 2.0

def code_reward(code_excu_result):
    return 1.0 if code_excu_result == 'Done' else 0.0

def format_reward(processed_str: str) -> float:
    """
    Check the format of the model output.
    """
    # Check for markdown code blocks or xml tags
    if ("```python" in processed_str and "```" in processed_str) or \
       ("<python>" in processed_str and "</python>" in processed_str):
        return 1.0
    return 0.0

def parado_reward(solution_code, problem_dir):
    """
    Verify if the solution is Pareto optimal by running the verify.py script.
    """
    if not solution_code:
        return 0.0

    # Create a temporary directory for verification
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 1. Copy necessary files from problem_dir to temp_dir
            # We need verify.py and the initial input.json (template with parameters)
            verify_script = os.path.join(problem_dir, "verify.py")
            input_template = os.path.join(problem_dir, "input.json")
            
            if not os.path.exists(verify_script) or not os.path.exists(input_template):
                print(f"Missing verification files in {problem_dir}")
                return 0.0
                
            shutil.copy2(verify_script, os.path.join(temp_dir, "verify.py"))
            shutil.copy2(input_template, os.path.join(temp_dir, "input.json"))
            
            # 2. Write the model generated code to solver.py
            solver_path = os.path.join(temp_dir, "solver.py")
            with open(solver_path, "w", encoding="utf-8") as f:
                f.write(solution_code)
                
            # 3. Execute solver.py to generate input.json (with solution)
            # We use a timeout to prevent infinite loops
            run_res = subprocess.run(
                [sys.executable, "solver.py"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if run_res.returncode != 0:
                # Solver failed to run
                return 0.0
                
            # 4. Execute verify.py to check Pareto optimality
            verify_res = subprocess.run(
                [sys.executable, "verify.py"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if verify_res.returncode == 0 and "True" in verify_res.stdout:
                return 1.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error in parado_reward: {e}")
            return 0.0

def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Compute the total reward for a batch of solutions.
    """
    rewards = []
    
    # We need access to the problem directory for each data sample.
    # Assuming 'extra_infos' contains the 'problem_path' or we can derive it.
    # If not, we might need to modify the data loader to pass this info.
    # For now, let's assume extra_infos[i]['problem_path'] exists.
    
    for i in range(len(solution_strs)):
        sol_str = solution_strs[i]
        # ground_truth = ground_truths[i] # Not used for this specific reward
        extra_info = extra_infos[i]
        problem_path = extra_info.get('problem_path', None) 
        # If problem_path is not directly available, we might need to construct it from data_source or index
        
        # 1. Format Reward
        r_format = format_reward(sol_str)
        
        # Extract code
        code = extract_code_block(sol_str, 'python')
        
        # 2. Code Execution Reward (R_acc) - Check if code is runnable
        # We can reuse the logic inside parado_reward (step 3) for this, 
        # or treat it separately. Here, if parado_reward succeeds (step 3 passes), 
        # it implies code execution was successful.
        # But to be precise:
        r_acc = 0.0
        r_parado = 0.0
        
        if code and problem_path:
             with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Setup environment
                    if os.path.exists(os.path.join(problem_path, "verify.py")):
                        shutil.copy2(os.path.join(problem_path, "verify.py"), os.path.join(temp_dir, "verify.py"))
                    if os.path.exists(os.path.join(problem_path, "input.json")):
                        shutil.copy2(os.path.join(problem_path, "input.json"), os.path.join(temp_dir, "input.json"))
                    
                    # Write solver
                    with open(os.path.join(temp_dir, "solver.py"), "w", encoding="utf-8") as f:
                        f.write(code)
                    
                    # Run solver
                    solver_res = subprocess.run(
                        [sys.executable, "solver.py"],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if solver_res.returncode == 0:
                        r_acc = 1.0 # Code ran successfully
                        
                        # Run verify
                        verify_res = subprocess.run(
                            [sys.executable, "verify.py"],
                            cwd=temp_dir,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        
                        # Loose check for verification success
                        # Check for "True" (case-insensitive) in stdout
                        if verify_res.returncode == 0 and "true" in verify_res.stdout.lower():
                            r_parado = 1.0
                        # Debugging info (optional, printed to console during training)
                        elif verify_res.returncode != 0:
                            pass # Verify script failed
                        else:
                            pass # Verify script ran but output did not contain True
                            
                    else:
                        # Solver execution failed
                        # print(f"Solver Error: {solver_res.stderr}")
                        pass
                            
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
        
        total_reward = (W_FORMAT * r_format) + (W_ACC * r_acc) + (W_PARADO * r_parado)
        rewards.append(total_reward)
        
    return rewards