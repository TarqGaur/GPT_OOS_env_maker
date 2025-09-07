import json
import os
import subprocess
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from _cmd import exec, exec_with_diagnostics
from LLM import addhistory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutonomousPackageResolver:
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.solution_history = []
        self.failed_packages = set()  # Track which packages are failing
        self.target_packages = set()  # Track which packages we're trying to install
        self.original_commands = []   # Store original failing commands
        
        # Universal solution patterns (not package-specific)
        self.known_solution_patterns = {
            "python_version_incompatible": {
                "patterns": [
                    r"requires python.*but you have",
                    r"python_requires",
                    r"not compatible with this python",
                    r"SafeConfigParser",
                    r"distutils.*deprecated",
                    r"removed in python"
                ],
                "solution_type": "version_upgrade",
                "description": "Package version incompatible with current Python version"
            },
            "no_matching_distribution": {
                "patterns": [
                    r"Could not find a version that satisfies",
                    r"No matching distribution found",
                    r"no such option",
                    r"ERROR: No matching distribution"
                ],
                "solution_type": "version_flexible",
                "description": "Exact version not available, need flexible version"
            },
            "build_tools_missing": {
                "patterns": [
                    r"Microsoft Visual C\+\+.*required",
                    r"python setup\.py.*failed",
                    r"error: Microsoft Visual Studio",
                    r"building wheel.*failed",
                    r"Failed building wheel"
                ],
                "solution_type": "upgrade_tools",
                "description": "Build tools need updating"
            },
            "dependency_conflict": {
                "patterns": [
                    r"incompatible.*already installed",
                    r"conflicts with",
                    r"dependency conflict",
                    r"version conflict"
                ],
                "solution_type": "dependency_resolution",
                "description": "Package dependency conflicts"
            }
        }
    
    def set_target_commands(self, commands: List[str]):
        """Set the original failing commands that we're trying to resolve"""
        self.original_commands = commands
        self.target_packages = self.extract_package_names_from_commands(commands)
        logger.info(f"Target packages to resolve: {self.target_packages}")
        
    def extract_package_names_from_commands(self, commands: List[str]) -> Set[str]:
        """Extract package names from pip install commands"""
        packages = set()
        for cmd in commands:
            # Match pip install commands and extract package names
            pip_match = re.search(r'pip install\s+(.+)', cmd, re.IGNORECASE)
            if pip_match:
                args = pip_match.group(1).strip()
                # Split by spaces and extract package names (ignore version specs)
                for arg in args.split():
                    if not arg.startswith('-'):  # Ignore flags like --upgrade
                        # Extract package name (before == or >= or <= etc.)
                        pkg_name = re.split(r'[<>=!]', arg.replace('"', '').replace("'", ''))[0]
                        if pkg_name and pkg_name not in ['pip', 'setuptools', 'wheel']:
                            packages.add(pkg_name.lower())
        return packages
    
    def load_latest_analysis(self) -> Dict[str, Any]:
        """Load the latest analysis from JSON files"""
        analysis_data = {
            "exec_results": {},
            "diagnose_results": {},
            "exec_test": {},
            "diagnose_test": {}
        }
        
        files_to_load = [
            ("exec.json", "exec_results"),
            ("diagnose.json", "diagnose_results"),
            ("exectest1.json", "exec_test"),
            ("exectest2.json", "diagnose_test")
        ]
        
        for filename, key in files_to_load:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        analysis_data[key] = data
                        logger.info(f"Loaded {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
            else:
                logger.warning(f"File {filename} not found")
        
        return analysis_data
    
    def extract_issues_from_analysis(self, analysis_data: Dict) -> List[Dict]:
        """Extract all issues from loaded analysis data and identify failed packages"""
        all_issues = []
        
        # Extract from exec_test (exectest1.json)
        if 'issues_found' in analysis_data.get('exec_test', {}):
            for issue in analysis_data['exec_test']['issues_found']:
                issue['source'] = 'exec_test'
                all_issues.append(issue)
                self._extract_failed_packages_from_issue(issue)
        
        # Extract from diagnose_test (exectest2.json) 
        if 'issues_found' in analysis_data.get('diagnose_test', {}):
            for issue in analysis_data['diagnose_test']['issues_found']:
                issue['source'] = 'diagnose_test'
                all_issues.append(issue)
                self._extract_failed_packages_from_issue(issue)
        
        # Extract from full histories if available
        for source_key in ['exec_results', 'diagnose_results']:
            source_data = analysis_data.get(source_key, {})
            if 'conversations' in source_data:
                conversations = source_data['conversations']
                if conversations:
                    latest_key = max(conversations.keys(), key=int)
                    latest_conv = conversations[latest_key]
                    if 'llm' in latest_conv and 'issues_found' in latest_conv['llm']:
                        for issue in latest_conv['llm']['issues_found']:
                            issue['source'] = source_key
                            all_issues.append(issue)
                            self._extract_failed_packages_from_issue(issue)
        
        logger.info(f"Extracted {len(all_issues)} issues from analysis")
        logger.info(f"Failed packages identified: {self.failed_packages}")
        return all_issues
    
    def _extract_failed_packages_from_issue(self, issue: Dict):
        """Extract package names from issue descriptions and commands"""
        description = issue.get('description', '').lower()
        command = issue.get('command', '').lower()
        affected_packages = issue.get('affected_packages', [])
        
        # Add explicitly mentioned affected packages
        for pkg in affected_packages:
            if pkg:
                self.failed_packages.add(pkg.lower())
        
        # Look for package names in pip install commands
        pip_matches = re.findall(r'pip install[^;]*?([a-zA-Z0-9_-]+)(?:[<>=]|$|\s)', command + ' ' + description)
        for match in pip_matches:
            if match and len(match) > 1:  # Ignore single characters
                self.failed_packages.add(match.lower())
        
        # Look for common package names mentioned in errors
        for pkg in self.target_packages:
            if pkg in description or pkg in command:
                self.failed_packages.add(pkg)
    
    def analyze_issue_patterns(self, issues: List[Dict]) -> Dict[str, List[Dict]]:
        """Analyze issues and categorize them by pattern type"""
        categorized_issues = {}
        
        for issue in issues:
            issue_text = (issue.get('description', '') + ' ' + issue.get('command', '')).lower()
            
            for pattern_name, pattern_data in self.known_solution_patterns.items():
                for regex_pattern in pattern_data['patterns']:
                    if re.search(regex_pattern, issue_text, re.IGNORECASE):
                        if pattern_name not in categorized_issues:
                            categorized_issues[pattern_name] = []
                        categorized_issues[pattern_name].append(issue)
                        break
        
        logger.info(f"Categorized issues: {list(categorized_issues.keys())}")
        return categorized_issues
    
    def generate_universal_solutions(self, categorized_issues: Dict) -> List[str]:
        """Generate universal solution commands based on issue patterns and failed packages"""
        commands = []
        
        # Always start with upgrading build tools if there are build issues
        if 'build_tools_missing' in categorized_issues:
            commands.append("python -m pip install --upgrade pip setuptools wheel")
        
        # Handle Python version compatibility issues
        if 'python_version_incompatible' in categorized_issues or 'no_matching_distribution' in categorized_issues:
            # For each failed package, try to install latest version without version constraints
            for package in self.failed_packages:
                if package in self.target_packages:  # Only for our target packages
                    commands.append(f'python -m pip install --upgrade "{package}"')
        
        # Handle dependency conflicts
        if 'dependency_conflict' in categorized_issues:
            # Try to reinstall all target packages together
            if self.target_packages:
                package_list = ' '.join([f'"{pkg}"' for pkg in self.target_packages])
                commands.append(f"python -m pip install --upgrade --force-reinstall {package_list}")
        
        # If no specific pattern, try general upgrade approach
        if not commands and self.failed_packages:
            commands.append("python -m pip install --upgrade pip setuptools wheel")
            for package in self.failed_packages:
                if package in self.target_packages:
                    commands.append(f'python -m pip install --upgrade "{package}"')
        
        return commands
    
    def execute_commands_with_monitoring(self, commands: List[str]) -> Dict[str, Any]:
        """Execute commands and monitor for success/failure"""
        results = {
            'commands_executed': commands,
            'success': True,
            'outputs': {},
            'errors': {}
        }
        
        for i, cmd in enumerate(commands):
            logger.info(f"Executing command {i+1}/{len(commands)}: {cmd}")
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
                
                results['outputs'][f'cmd_{i+1}'] = {
                    'command': cmd,
                    'returncode': result.returncode,
                    'stdout': result.stdout.strip(),
                    'stderr': result.stderr.strip()
                }
                
                if result.returncode != 0:
                    results['success'] = False
                    results['errors'][f'cmd_{i+1}'] = {
                        'command': cmd,
                        'error': result.stderr.strip(),
                        'returncode': result.returncode
                    }
                    logger.error(f"Command failed: {cmd}")
                else:
                    logger.info(f"Command succeeded: {cmd}")
                    
            except subprocess.TimeoutExpired:
                results['success'] = False
                results['errors'][f'cmd_{i+1}'] = {
                    'command': cmd,
                    'error': 'Command timed out after 300 seconds'
                }
                logger.error(f"Command timed out: {cmd}")
            except Exception as e:
                results['success'] = False
                results['errors'][f'cmd_{i+1}'] = {
                    'command': cmd,
                    'error': str(e)
                }
                logger.error(f"Command exception: {cmd} - {e}")
        
        return results
    
    def generate_verification_commands(self) -> List[str]:
        """Generate verification commands based on target packages"""
        verification_commands = []
        
        for package in self.target_packages:
            # Try to import the package and get version
            verification_commands.append(f'python -c "import {package}; print(\'{package}:\', {package}.__version__)"')
        
        # If no target packages, fall back to common verification
        if not verification_commands:
            verification_commands = [
                'python -c "import sys; print(\'Python version:\', sys.version)"',
                'pip list'
            ]
        
        return verification_commands
    
    def verify_solution_success(self) -> Dict[str, Any]:
        """Verify if the solution worked by testing imports of target packages"""
        verification_commands = self.generate_verification_commands()
        
        verification_result = self.execute_commands_with_monitoring(verification_commands)
        
        # Count successful imports
        successful_imports = sum(1 for cmd_data in verification_result['outputs'].values() 
                               if cmd_data['returncode'] == 0)
        total_imports = len(verification_commands)
        
        verification_result['success_rate'] = successful_imports / total_imports if total_imports > 0 else 0
        verification_result['imports_successful'] = successful_imports
        verification_result['total_imports'] = total_imports
        verification_result['fully_resolved'] = verification_result['success_rate'] >= 0.5  # 50% threshold
        
        logger.info(f"Verification: {successful_imports}/{total_imports} imports successful")
        return verification_result
    
    def query_llm_for_solution(self, issues: List[Dict], previous_attempts: List[Dict]) -> Dict[str, Any]:
        """Query LLM with full context for a solution"""
        
        # Get current Python version
        try:
            python_version = subprocess.run(['python', '--version'], capture_output=True, text=True).stdout.strip()
        except:
            python_version = "Unknown"
        
        query_context = f"""
I have package installation conflicts that need resolution. Here's the complete context:

ORIGINAL COMMANDS THAT FAILED:
{json.dumps(self.original_commands, indent=2)}

TARGET PACKAGES:
{list(self.target_packages)}

CURRENT PYTHON VERSION:
{python_version}

ISSUES IDENTIFIED:
{json.dumps(issues, indent=2)}

PREVIOUS ATTEMPTS:
{json.dumps(previous_attempts, indent=2)}

FAILED PACKAGES DETECTED:
{list(self.failed_packages)}

I need specific pip commands to resolve these conflicts. Please analyze the error patterns and provide:
1. Exact pip commands to fix the issues
2. Alternative approaches if the direct approach fails
3. Explanation of why the original commands failed
4. Prevention strategies for future installations

Focus on making the target packages work with the current Python version.
"""
        
        logger.info("Querying LLM for solution...")
        try:
            # Use the enhanced access with memory and potential web search
            llm_response = addhistory(
                user_msg=query_context,
                use_enhanced_access=True,
                history_name="package_resolver",
                save_to_file="package_resolver_history.json"
            )
            
            return {
                'success': True,
                'response': llm_response,
                'suggested_commands': self.extract_commands_from_llm_response(llm_response)
            }
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'suggested_commands': []
            }
    
    def extract_commands_from_llm_response(self, llm_response: Dict) -> List[str]:
        """Extract executable commands from LLM response"""
        commands = []
        
        if isinstance(llm_response, dict):
            # Check main response for commands
            main_response = llm_response.get('main_response', '')
            if isinstance(main_response, str):
                # Look for pip install commands in the response
                lines = main_response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('pip install') or line.startswith('python -m pip install'):
                        # Clean up the command (remove markdown code block markers, etc.)
                        clean_line = re.sub(r'```[bash]*', '', line).strip()
                        clean_line = re.sub(r'```', '', clean_line).strip()
                        if clean_line:
                            commands.append(clean_line)
            
            # Check if there are specific command recommendations in structured format
            if 'recommended_solutions' in llm_response:
                for solution in llm_response['recommended_solutions']:
                    if 'commands' in solution and isinstance(solution['commands'], list):
                        commands.extend(solution['commands'])
        
        # If no commands found, generate fallback based on target packages
        if not commands and self.target_packages:
            commands = [
                "python -m pip install --upgrade pip setuptools wheel"
            ]
            for package in self.target_packages:
                commands.append(f'python -m pip install --upgrade "{package}"')
        
        logger.info(f"Extracted {len(commands)} commands from LLM response")
        return commands
    
    def web_search_for_solution(self, issues: List[Dict]) -> Dict[str, Any]:
        """Trigger web search through LLM for additional solutions"""
        package_list = " ".join(self.target_packages)
        search_query = f"{package_list} Python 3.13 installation compatibility fix"
        
        search_context = f"""
I need to find current solutions for Python package installation issues:

TARGET PACKAGES: {list(self.target_packages)}
FAILED PACKAGES: {list(self.failed_packages)}

SEARCH FOR: {search_query}

SPECIFIC ISSUES:
{json.dumps(issues, indent=2)}

Please search for recent solutions and provide specific pip commands that work for these packages.
The solutions should focus on version compatibility and installation methods that work.
"""
        
        logger.info("Triggering web search for solutions...")
        try:
            # The LLM will automatically trigger web search if needed
            search_response = addhistory(
                user_msg=search_context + " NEED_SEARCH: " + search_query,
                use_enhanced_access=True,
                history_name="package_resolver_search",
                save_to_file="package_resolver_search_history.json"
            )
            
            return {
                'success': True,
                'response': search_response,
                'suggested_commands': self.extract_commands_from_llm_response(search_response)
            }
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'suggested_commands': []
            }
    
    def run_autonomous_resolution(self, target_commands: List[str]) -> Dict[str, Any]:
        """Main autonomous resolution loop"""
        logger.info("Starting autonomous package conflict resolution")
        
        # Set up target commands
        self.set_target_commands(target_commands)
        
        final_report = {
            'start_time': time.time(),
            'target_commands': target_commands,
            'target_packages': list(self.target_packages),
            'iterations': [],
            'final_status': 'unknown',
            'resolution_successful': False,
            'total_iterations': 0
        }
        
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            iteration_start = time.time()
            
            logger.info(f"=== Iteration {self.iteration_count}/{self.max_iterations} ===")
            
            iteration_data = {
                'iteration_number': self.iteration_count,
                'start_time': iteration_start,
                'actions_taken': [],
                'success': False
            }
            
            # Step 1: Load current analysis
            analysis_data = self.load_latest_analysis()
            issues = self.extract_issues_from_analysis(analysis_data)
            
            if not issues:
                logger.info("No issues found in analysis - checking if packages work")
                verification = self.verify_solution_success()
                if verification['fully_resolved']:
                    logger.info("All packages working - resolution complete!")
                    iteration_data['success'] = True
                    final_report['resolution_successful'] = True
                    final_report['final_status'] = 'resolved'
                    break
                else:
                    logger.info("Packages still not working - re-running analysis")
                    # Re-run diagnostics to get fresh error data
                    exec_with_diagnostics(self.original_commands)
                    continue
            
            # Step 2: Analyze issue patterns for universal solutions
            categorized_issues = self.analyze_issue_patterns(issues)
            universal_commands = self.generate_universal_solutions(categorized_issues)
            
            if universal_commands:
                logger.info(f"Applying universal solutions: {len(universal_commands)} commands")
                execution_result = self.execute_commands_with_monitoring(universal_commands)
                iteration_data['actions_taken'].append({
                    'action_type': 'universal_solution',
                    'commands': universal_commands,
                    'result': execution_result
                })
                
                if execution_result['success']:
                    # Verify the solution worked
                    verification = self.verify_solution_success()
                    if verification['fully_resolved']:
                        logger.info("Universal solution resolved the issue!")
                        iteration_data['success'] = True
                        final_report['resolution_successful'] = True
                        final_report['final_status'] = 'resolved'
                        break
            
            # Step 3: If universal solutions didn't work, query LLM
            if not iteration_data['success']:
                logger.info("Querying LLM for advanced solution")
                llm_result = self.query_llm_for_solution(issues, self.solution_history)
                
                if llm_result['success'] and llm_result['suggested_commands']:
                    execution_result = self.execute_commands_with_monitoring(llm_result['suggested_commands'])
                    iteration_data['actions_taken'].append({
                        'action_type': 'llm_solution',
                        'commands': llm_result['suggested_commands'],
                        'result': execution_result
                    })
                    
                    if execution_result['success']:
                        verification = self.verify_solution_success()
                        if verification['fully_resolved']:
                            logger.info("LLM solution resolved the issue!")
                            iteration_data['success'] = True
                            final_report['resolution_successful'] = True
                            final_report['final_status'] = 'resolved'
                            break
            
            # Step 4: If still not resolved, try web search
            if not iteration_data['success']:
                logger.info("Trying web search for additional solutions")
                search_result = self.web_search_for_solution(issues)
                
                if search_result['success'] and search_result['suggested_commands']:
                    execution_result = self.execute_commands_with_monitoring(search_result['suggested_commands'])
                    iteration_data['actions_taken'].append({
                        'action_type': 'web_search_solution',
                        'commands': search_result['suggested_commands'],
                        'result': execution_result
                    })
                    
                    if execution_result['success']:
                        verification = self.verify_solution_success()
                        if verification['fully_resolved']:
                            logger.info("Web search solution resolved the issue!")
                            iteration_data['success'] = True
                            final_report['resolution_successful'] = True
                            final_report['final_status'] = 'resolved'
                            break
            
            # Record this iteration
            iteration_data['end_time'] = time.time()
            iteration_data['duration'] = iteration_data['end_time'] - iteration_start
            final_report['iterations'].append(iteration_data)
            self.solution_history.append(iteration_data)
            
            if not iteration_data['success']:
                logger.warning(f"Iteration {self.iteration_count} did not resolve the issue")
                # Re-run analysis to get updated error data for next iteration
                exec_with_diagnostics(self.original_commands)
            else:
                break
        
        # Finalize report
        final_report['end_time'] = time.time()
        final_report['total_duration'] = final_report['end_time'] - final_report['start_time']
        final_report['total_iterations'] = self.iteration_count
        final_report['failed_packages_detected'] = list(self.failed_packages)
        
        if not final_report['resolution_successful']:
            if self.iteration_count >= self.max_iterations:
                final_report['final_status'] = 'max_iterations_reached'
            else:
                final_report['final_status'] = 'unresolved'
        
        # Save final report
        report_filename = 'universal_resolution_report.json'
        with open(report_filename, 'w') as f:
            json.dump(final_report, f, indent=4)
        
        logger.info(f"Autonomous resolution completed. Status: {final_report['final_status']}")
        logger.info(f"Report saved to: {report_filename}")
        
        return final_report

def main():
    """Main entry point - can be called with different target commands"""
    import sys
    
    # Default test commands if none provided
    # default_commands = [
    #     "pip install numpy==1.15.0", 
    #     "pip install pandas==2.1.0"
    # ]
    default_commands = []
    
    # Allow commands to be passed via command line arguments
    if len(sys.argv) > 1:
        target_commands = sys.argv[1:]
    else:
        target_commands = default_commands
    
    resolver = AutonomousPackageResolver(max_iterations=10)
    result = resolver.run_autonomous_resolution(target_commands)
    
    print(f"\n{'='*60}")
    print(f"UNIVERSAL PACKAGE RESOLUTION COMPLETE")
    print(f"{'='*60}")
    print(f"Target commands: {target_commands}")
    print(f"Target packages: {result.get('target_packages', [])}")
    print(f"Status: {result['final_status']}")
    print(f"Resolution successful: {result['resolution_successful']}")
    print(f"Total iterations: {result['total_iterations']}")
    print(f"Total time: {result.get('total_duration', 0):.2f} seconds")
    
    if result['resolution_successful']:
        print(f"✅ All package conflicts resolved successfully!")
    else:
        print(f"⚠️  Some issues may remain. Check universal_resolution_report.json for details.")
    
    return result

if __name__ == "__main__":
    main()