
import subprocess
import json
from LLM import addhistory

try:
    with open("llmhist.json", "r") as f:
        use_memory_access = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    use_memory_access = []

def safe_json_parse(response_text):
    """Safely parse JSON response, handling common issues"""
    try:
        # First try direct parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # Try to find JSON block in the response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # If all else fails, return a structured error
        return {
            "overall_status": "critical_error",
            "summary": "Failed to parse LLM response as JSON",
            "issues_found": [],
            "recommended_solutions": [],
            "prevention_tips": ["Ensure LLM returns valid JSON format"],
            "raw_response": response_text
        }

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

CRITICAL: You MUST respond with ONLY valid JSON. No text before or after the JSON.

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
      "undo_commands": ["exact pip/conda commands to undo past priority changes, if first priority/no undo changes then set to empty array"],
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
7. RESPOND WITH ONLY JSON - NO OTHER TEXT

COMMAND OUTPUTS TO ANALYZE:
{json.dumps(problems, indent=2)}"""

    return addhistory(prompt, True, "exec", "exec.json")

def exec_with_diagnostics(cmdslist, include_env_info=True):
    """Enhanced version that includes environment diagnostics"""
    diagnostic_cmds = []
    
    if include_env_info:
        # Use cross-platform commands
        diagnostic_cmds = [
            "python --version",
            "pip --version", 
            "pip list",  # Remove grep for cross-platform compatibility
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

CRITICAL: You MUST respond with ONLY valid JSON. No text before or after the JSON.

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

RESPONSE FORMAT: Same JSON structure as the basic analysis, but with more detailed analysis based on environment context.

{{
  "overall_status": "success" | "needs_attention" | "critical_error",
  "summary": "Brief description of findings with environment context",
  "issues_found": [
    {{
      "command_index": "1",
      "issue_type": "version_conflict" | "missing_dependency" | "build_error" | "permission_error" | "deprecation_warning" | "python_version_conflict" | "other",
      "severity": "low" | "medium" | "high" | "critical",
      "description": "Clear description of the specific issue",
      "affected_packages": ["package1", "package2"],
      "root_cause": "Explanation of why this happened considering environment"
    }}
  ],
  "recommended_solutions": [
    {{
      "solution_type": "commands" | "user_action" | "web_search" | "environment_setup",
      "priority": 1,
      "description": "What this solution does",
      "undo_commands": [],
      "commands": ["exact pip/conda commands to run"],
      "user_actions": ["manual steps if needed"],
      "search_query": "search terms if web research needed",
      "expected_outcome": "what should happen after applying this solution"
    }}
  ],
  "prevention_tips": [
    "How to avoid similar issues in the future"
  ],
  "environment_recommendations": [
    "Specific advice based on current Python/pip versions"
  ]
}}

CRITICAL: RESPOND WITH ONLY VALID JSON - NO OTHER TEXT OR EXPLANATIONS"""

    return addhistory(enhanced_prompt, True, "diagnose", "diagnose.json")

if __name__ == "__main__":
    print("=== Basic Analysis ===")
    try:
        result1 = exec([
            "pip install tensorflow==1.5.0", 
            "pip install keras==2.0.0"
        ])

        if result1:
            last_key = sorted(result1.keys(), key=int)[-1]
            
            # Safely extract the response
            if "llm" in result1[last_key] and "main_response" in result1[last_key]["llm"]:
                main_response = result1[last_key]["llm"]["main_response"]
                
                # Try to parse as JSON if it's a string
                if isinstance(main_response, str):
                    try:
                        main_response = safe_json_parse(main_response)
                    except:
                        main_response = {"error": "Failed to parse JSON", "raw": main_response}
            else:
                main_response = {"error": "No main_response found", "data": result1[last_key]}

            with open("exectest1.json", "w") as f:
                json.dump(main_response, f, indent=4)
            
            print("✅ Basic analysis saved to exectest1.json")
        else:
            print("❌ No result from basic analysis")
            
    except Exception as e:
        print(f"❌ Error in basic analysis: {e}")
        with open("exectest1.json", "w") as f:
            json.dump({"error": str(e)}, f, indent=4)

    print("\n=== Enhanced Analysis with Diagnostics ===")
    try:
        result2 = exec_with_diagnostics([
            "pip install tensorflow==1.5.0", 
            "pip install keras==2.0.0"
        ])

        if result2:
            last_key2 = sorted(result2.keys(), key=int)[-1]  # Fixed: was using result1
            
            # Safely extract the response
            if "llm" in result2[last_key2] and "main_response" in result2[last_key2]["llm"]:
                main_response2 = result2[last_key2]["llm"]["main_response"]
                
                # Try to parse as JSON if it's a string
                if isinstance(main_response2, str):
                    try:
                        main_response2 = safe_json_parse(main_response2)
                    except:
                        main_response2 = {"error": "Failed to parse JSON", "raw": main_response2}
            else:
                main_response2 = {"error": "No main_response found", "data": result2[last_key2]}
            
            with open("exectest2.json", "w") as f:
                json.dump(main_response2, f, indent=4)
            
            print("✅ Enhanced analysis saved to exectest2.json")
        else:
            print("❌ No result from enhanced analysis")
            
    except Exception as e:
        print(f"❌ Error in enhanced analysis: {e}")
        with open("exectest2.json", "w") as f:
            json.dump({"error": str(e)}, f, indent=4)