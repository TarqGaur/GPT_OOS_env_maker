import subprocess
import json
from LLM import addhistory


use_memory_access =[]
with open("exectest.json" , "w") as f:
    json.dump(use_memory_access ,f , indent=4)




# execution type 1 =>
def exec(cmdslist):
    allconsole = []

    for i in cmdslist:
        result = subprocess.run(i, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            allconsole.append(result.stdout.strip())
        else:
            allconsole.append(result.stderr.strip())

    problems = {}
    for idx, cmd in enumerate(cmdslist, start=1):
        output = allconsole[idx-1]

        problems[str(idx)] = {
            "command_executed": cmd,
            "output": output,
            "everything_good_and_no_errors_in_console": "error" not in output.lower() and "conflict" not in output.lower(),
            "analysis_required": True
        }

    prompt = f"""You are an expert Python package conflict resolver and dependency manager.

TASK: Analyze the command outputs below and provide solutions for any package conflicts, dependency issues, or installation errors.

ANALYSIS CRITERIA:
- Look for version conflicts between packages
- Identify incompatible dependency versions
- Detect missing dependencies or build requirements
- Check for Python version compatibility issues
- Identify deprecated package warnings
- Look for permission or environment issues

RESPONSE FORMAT: Respond ONLY in valid JSON with this exact structure:

{{
  "overall_status": "success" | "needs_attention" | "critical_error",
  "summary": "Brief description of findings",
  "issues_found": [
    {{
      "command_index": "1",
      "issue_type": "version_conflict" | "missing_dependency" | "build_error" | "permission_error" | "deprecation_warning" | "other",
      "severity": "low" | "medium" | "high" | "critical",
      "description": "Clear description of the specific issue",
      "affected_packages": ["package1", "package2"],
      "root_cause": "Explanation of why this happened"
    }}
  ],
  "recommended_solutions": [
    {{
      "solution_type": "commands" | "user_action" | "web_search",
      "priority": 1,
      "description": "What this solution does",
      "undo_commands": ["exact pip/conda commands to undo past priority changes, if first priority/no undo changes then set to 0"],
      "commands": ["exact pip/conda commands to run"],
      "user_actions": ["manual steps if needed"],
      "search_query": "search terms if web research needed",
      "expected_outcome": "what should happen after applying this solution"
    }}
  ],
  "prevention_tips": [
    "How to avoid similar issues in the future"
  ]
}}

IMPORTANT RULES:
1. If ALL commands executed successfully with no errors/warnings, set overall_status to "success" and keep issues_found empty
2. For version conflicts, suggest specific compatible version combinations
3. Always provide exact pip/conda commands, not generic advice
4. Prioritize solutions by effectiveness and safety
5. Include Python version requirements when relevant
6. Consider virtual environment isolation for complex conflicts
7. Also remember the priority commands will only be executed after testing its before priority so in undo_commands list all the changes to undo if required.

COMMAND OUTPUTS TO ANALYZE:
{json.dumps(problems, indent=2)}

Analyze each command output thoroughly and provide actionable solutions."""
    
    return addhistory(prompt, use_memory_access)

# execution type 2 =>
def exec_with_diagnostics(cmdslist, include_env_info=True):
    """
    Enhanced version that includes environment diagnostics
    """
    diagnostic_cmds = []
    
    if include_env_info:
        diagnostic_cmds = [
            "python --version",
            "pip --version", 
            "pip list | grep -E '(matplotlib|scipy|numpy|pandas)'",
            "pip check"
        ]
    
    all_cmds = diagnostic_cmds + cmdslist
    
    allconsole = []
    for i in all_cmds:
        result = subprocess.run(i, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            allconsole.append(result.stdout.strip())
        else:
            allconsole.append(result.stderr.strip())

    diagnostic_info = {}
    main_problems = {}
    
    if include_env_info:
        for idx, cmd in enumerate(diagnostic_cmds):
            diagnostic_info[f"diagnostic_{idx+1}"] = {
                "command": cmd,
                "output": allconsole[idx]
            }
    
    start_idx = len(diagnostic_cmds)
    for idx, cmd in enumerate(cmdslist):
        main_problems[str(idx+1)] = {
            "command_executed": cmd,
            "output": allconsole[start_idx + idx],
            "return_code": 0,  
            "analysis_required": True
        }

    enhanced_prompt = f"""You are an expert Python package conflict resolver with deep knowledge of package ecosystems.

ENVIRONMENT DIAGNOSTICS:
{json.dumps(diagnostic_info, indent=2) if include_env_info else "No diagnostic info provided"}

MAIN COMMAND ANALYSIS:
{json.dumps(main_problems, indent=2)}

ENHANCED ANALYSIS INSTRUCTIONS:
1. Use the diagnostic info to understand the current environment state
2. Cross-reference package versions to identify compatibility matrices
3. Consider transitive dependencies (packages that depend on your target packages)
4. Evaluate if the issue requires a clean environment or can be resolved in-place
5. Suggest virtual environment creation if conflicts are severe

RESPONSE FORMAT: Same JSON structure as before, but with more detailed analysis based on environment context.

Provide specific version recommendations that work together."""

    return addhistory(enhanced_prompt, use_memory_access)




if __name__ == "__main__":
    print("=== Basic Analysis ===")
    result1 = exec([
        "pip install matplotlib==3.1.0",
        "pip install scipy==1.11.2"
    ])
    print(result1)
    
    print("\n=== Enhanced Analysis with Diagnostics ===")
    result2 = exec_with_diagnostics([
        "pip install matplotlib==3.1.0", 
        "pip install scipy==1.11.2"
    ])
    print(result2)