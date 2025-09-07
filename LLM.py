from openai import OpenAI
import json
import re
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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"  
)

# Enhanced system prompt that teaches the model about memory access
system_prompt = """
You are an AI assistant with access to a memory system. Always respond in **strict JSON format** only.

MEMORY ACCESS RULES:
- You will receive a "memory_summary" containing recent conversations
- If you need detailed information from older conversations, you can request specific memory indices
- To request memory access, include in your main_response: "NEED_MEMORY: [index1, index2, ...]"
- Available memory indices will be shown in the memory_summary

Follow this exact JSON schema:

{
  "main_response": "<your response here, include NEED_MEMORY: [indices] if you need older memories>",
  "memory_request": ["index1", "index2"] or null,
  "summarize_answer_prompt_in_100_words": "<summarize your answer in 100 words>",
  "summarize_question_prompt_in_100_words": "<summarize the user's question in 100 words>"
}

Rules:
1. Do not include anything outside the JSON object.
2. If you don't need memory access, set "memory_request" to null
3. If you need specific memories, list the indices in "memory_request" array
4. Always provide a complete response in "main_response"
"""

# Global variables for multiple histories
histories = {}
current_history = "default"
q_counters = {}

def init_history(history_name="default"):
    """Initialize a new history if it doesn't exist"""
    if history_name not in histories:
        histories[history_name] = {}
        q_counters[history_name] = 0
    return histories[history_name]

def switch_history(history_name):
    """Switch to a different history"""
    global current_history
    current_history = history_name
    init_history(history_name)
    print(f"🔄 Switched to history: {history_name}")

def get_current_history():
    """Get the current active history"""
    init_history(current_history)
    return histories[current_history]

def create_memory_summary(history_name=None):
    """Create a summary of available memories with indices for a specific history"""
    if history_name is None:
        history_name = current_history
    
    history = histories.get(history_name, {})
    
    if not history:
        return "No previous conversations available."
    
    summary = f"Available Memory Indices for '{history_name}':\n"
    for idx, conv in history.items():
        # Create a brief summary of each conversation
        user_msg = conv['user'][:100] + "..." if len(conv['user']) > 100 else conv['user']
        summary += f"Index {idx}: User asked about: {user_msg}\n"
    
    # Add recent context (last 2 conversations)
    last_keys = list(history.keys())[-2:]
    if last_keys:
        summary += "\nRecent Context:\n"
        for key in last_keys:
            summary += f"[{key}] User: {history[key]['user']}\n"
            summary += f"[{key}] AI: {history[key]['llm'].get('main_response', 'No response')}\n"
    
    return summary

def get_detailed_memory(indices, history_name=None):
    """Retrieve detailed memory for specific indices from a specific history"""
    if history_name is None:
        history_name = current_history
    
    history = histories.get(history_name, {})
    detailed_memories = {}
    for idx in indices:
        if str(idx) in history:
            detailed_memories[idx] = history[str(idx)]
    return detailed_memories

def send_with_memory_access(user_msg, history_name=None):
    """Enhanced send function with dynamic memory access"""
    if history_name is None:
        history_name = current_history
        
    try:
        # Step 1: Send initial request with memory summary
        memory_summary = create_memory_summary(history_name)
        
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Memory Summary:\n{memory_summary}"},
            {"role": "user", "content": f"Current Question: {user_msg}"}
        ]

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=initial_messages,
        )

        raw_content = json.loads(response.choices[0].message.content.strip())
        
        # Step 2: Check if model requested specific memories
        memory_request = raw_content.get('memory_request')
        
        if memory_request and isinstance(memory_request, list) and len(memory_request) > 0:
            print(f"🧠 AI requested memory access for indices: {memory_request}")
            
            # Get detailed memories
            detailed_memories = get_detailed_memory(memory_request, history_name)
            
            # Step 3: Send follow-up request with detailed memories
            followup_messages = [
                {"role": "system", "content": system_prompt.replace("NEED_MEMORY: [indices] if you need older memories", "provide your final answer")},
                {"role": "user", "content": f"Original Question: {user_msg}"},
                {"role": "user", "content": f"Memory Summary: {memory_summary}"},
                {"role": "user", "content": f"Detailed Memories Requested: {json.dumps(detailed_memories, indent=2)}"},
                {"role": "user", "content": "Now provide your complete final answer based on all available information."}
            ]

            final_response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=followup_messages,
            )

            final_content = json.loads(final_response.choices[0].message.content.strip())
            
            print("AI (with memory access):", final_content["main_response"])
            return final_content
        else:
            print("AI:", raw_content["main_response"])
            return raw_content

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e), "main_response": f"Error occurred: {e}"}

def send_simple(user_msg, history_name=None):
    """Original simple send function (for comparison)"""
    if history_name is None:
        history_name = current_history
        
    history = histories.get(history_name, {})
    
    try:
        last_keys = list(history.keys())[-4:]   # last 4 
        last_memories = {k: history[k] for k in last_keys}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(last_memories)},  # last 4
            {"role": "user", "content": user_msg}
        ]

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
        )

        raw_content = json.loads(response.choices[0].message.content.strip())
        print("AI:", raw_content["main_response"])
        return raw_content

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

def save_history_to_file(filename, history_name=None):
    """Save a specific history to a file"""
    if history_name is None:
        history_name = current_history
    
    history = histories.get(history_name, {})
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, "w") as f:
            json.dump({
                "history_name": history_name,
                "conversations": history,
                "total_conversations": len(history)
            }, f, indent=4)
        print(f"💾 History '{history_name}' saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving history: {e}")
        return False

def load_history_from_file(filename, history_name=None):
    """Load a history from a file"""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        
        if history_name is None:
            history_name = data.get("history_name", "loaded_history")
        
        histories[history_name] = data.get("conversations", {})
        
        # Update counter
        if histories[history_name]:
            q_counters[history_name] = max([int(k) for k in histories[history_name].keys()])
        else:
            q_counters[history_name] = 0
            
        print(f"📁 History loaded as '{history_name}' from {filename}")
        print(f"   Total conversations: {len(histories[history_name])}")
        return True
    except Exception as e:
        print(f"❌ Error loading history: {e}")
        return False

def addhistory(user_msg, use_memory_access=True, history_name=None, save_to_file=None):
    """Add conversation to history with optional memory access and file saving"""
    if history_name is None:
        history_name = current_history
    
    # Initialize history if it doesn't exist
    init_history(history_name)
    
    # Switch to the specified history temporarily if different from current
    original_history = current_history
    if history_name != current_history:
        switch_history(history_name)
    
    # Increment counter for this history
    q_counters[history_name] += 1
    q = q_counters[history_name]

    if use_memory_access:
        llm_reply = send_with_memory_access(user_msg, history_name)
    else:
        llm_reply = send_simple(user_msg, history_name)

    histories[history_name][str(q)] = {
        "user": user_msg,
        "llm": llm_reply
    }

    # Save to specific file if provided
    if save_to_file:
        save_history_to_file(save_to_file, history_name)
    else:
        # Save to default file for current history
        default_filename = f"{history_name}_history.json"
        save_history_to_file(default_filename, history_name)

    # Switch back to original history if we switched temporarily
    if history_name != original_history:
        switch_history(original_history)

    return histories[history_name]

def show_memory_stats(history_name=None):
    """Display current memory statistics for a specific history"""
    if history_name is None:
        history_name = current_history
    
    history = histories.get(history_name, {})
    print(f"\n📊 Memory Stats for '{history_name}':")
    print(f"Total conversations: {len(history)}")
    print(f"Available indices: {list(history.keys())}")

def show_all_histories():
    """Show all available histories"""
    print(f"\n📚 All Histories:")
    print(f"Current active: {current_history}")
    for name, history in histories.items():
        marker = "👉 " if name == current_history else "   "
        print(f"{marker}{name}: {len(history)} conversations")

def show_commands():
    """Show available commands"""
    print("\n🔧 Available Commands:")
    print("/stats [history_name] - Show memory statistics")
    print("/history [history_name] - Show conversation history")
    print("/histories - Show all available histories")
    print("/switch <history_name> - Switch to a different history")
    print("/clear [history_name] - Clear history (current if none specified)")
    print("/save <filename> [history_name] - Save history to file")
    print("/load <filename> [history_name] - Load history from file")
    print("/simple - Toggle simple mode (no memory access)")
    print("/help - Show this help")
    print("/quit - Exit the program")
    print("\nUsage Examples:")
    print("  addhistory('Hello', True, 'chat1', 'chat1.json')")
    print("  /switch project_work")
    print("  /save backup.json")

# Initialize default history
init_history()

# Main interaction loop
use_memory_access = True

# while True:
#     user_input = input(f"\nYou [{current_history}]: ").strip()
    
#     # Handle commands
#     if user_input.startswith('/'):
#         parts = user_input.split()
#         cmd = parts[0]
        
#         if cmd == '/quit':
#             break
#         elif cmd == '/stats':
#             history_name = parts[1] if len(parts) > 1 else None
#             show_memory_stats(history_name)
#         elif cmd == '/history':
#             history_name = parts[1] if len(parts) > 1 else current_history
#             history = histories.get(history_name, {})
#             print(f"\n📚 Conversation History for '{history_name}':")
#             for idx, conv in history.items():
#                 print(f"[{idx}] User: {conv['user'][:100]}...")
#                 print(f"[{idx}] AI: {conv['llm'].get('main_response', 'No response')[:100]}...")
#                 print()
#         elif cmd == '/histories':
#             show_all_histories()
#         elif cmd == '/switch':
#             if len(parts) > 1:
#                 switch_history(parts[1])
#             else:
#                 print("❓ Usage: /switch <history_name>")
#         elif cmd == '/clear':
#             history_name = parts[1] if len(parts) > 1 else current_history
#             if history_name in histories:
#                 histories[history_name].clear()
#                 q_counters[history_name] = 0
#                 print(f"🗑️ History '{history_name}' cleared!")
#             else:
#                 print(f"❓ History '{history_name}' not found!")
#         elif cmd == '/save':
#             if len(parts) > 1:
#                 filename = parts[1]
#                 history_name = parts[2] if len(parts) > 2 else current_history
#                 save_history_to_file(filename, history_name)
#             else:
#                 print("❓ Usage: /save <filename> [history_name]")
#         elif cmd == '/load':
#             if len(parts) > 1:
#                 filename = parts[1]
#                 history_name = parts[2] if len(parts) > 2 else None
#                 load_history_from_file(filename, history_name)
#             else:
#                 print("❓ Usage: /load <filename> [history_name]")
#         elif cmd == '/simple':
#             use_memory_access = not use_memory_access
#             mode = "Memory Access" if use_memory_access else "Simple"
#             print(f"🔄 Switched to {mode} mode")
#         elif cmd == '/help':
#             show_commands()
#         else:
#             print("❓ Unknown command. Type /help for available commands.")
#         continue
    
#     if user_input:
#         addhistory(user_input, use_memory_access)





















from openai import OpenAI
import json
import re
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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"  
)

# Enhanced system prompt that teaches the model about memory access
system_prompt = """
You are an AI assistant with access to a memory system. You MUST respond in **strict JSON format** only.

CRITICAL: Your response must be VALID JSON that can be parsed by json.loads(). Do not include any text before or after the JSON object.

MEMORY ACCESS RULES:
- You will receive a "memory_summary" containing recent conversations
- If you need detailed information from older conversations, you can request specific memory indices
- To request memory access, include in your main_response: "NEED_MEMORY: [index1, index2, ...]"
- Available memory indices will be shown in the memory_summary

Follow this exact JSON schema:

{
  "main_response": "your response here, include NEED_MEMORY: [indices] if you need older memories",
  "memory_request": ["index1", "index2"] or null,
  "summarize_answer_prompt_in_100_words": "summarize your answer in 100 words",
  "summarize_question_prompt_in_100_words": "summarize the user's question in 100 words"
}

CRITICAL RULES:
1. ONLY return valid JSON - nothing else
2. Do not include markdown formatting, code blocks, or explanations outside JSON
3. If you don't need memory access, set "memory_request" to null
4. If you need specific memories, list the indices in "memory_request" array
5. Always provide a complete response in "main_response"
6. Escape all quotes and special characters properly in JSON strings
"""

# Global variables for multiple histories
histories = {}
current_history = "default"
q_counters = {}

def safe_json_parse(content):
    """Safely parse JSON content with fallback handling"""
    if not content or not isinstance(content, str):
        return {"error": "Empty or invalid content", "main_response": "No valid response received"}
    
    try:
        # First try direct parsing
        return json.loads(content.strip())
    except json.JSONDecodeError as e:
        try:
            # Try to find JSON block in the response
            content = content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1])
            
            # Find JSON object boundaries
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # If all parsing fails, create a structured response
        return {
            "main_response": content,
            "memory_request": None,
            "summarize_answer_prompt_in_100_words": "Failed to parse JSON response from AI model",
            "summarize_question_prompt_in_100_words": "User query that resulted in unparseable response",
            "parsing_error": str(e),
            "raw_content": content
        }

def init_history(history_name="default"):
    """Initialize a new history if it doesn't exist"""
    if history_name not in histories:
        histories[history_name] = {}
        q_counters[history_name] = 0
    return histories[history_name]

def switch_history(history_name):
    """Switch to a different history"""
    global current_history
    current_history = history_name
    init_history(history_name)
    print(f"🔄 Switched to history: {history_name}")

def get_current_history():
    """Get the current active history"""
    init_history(current_history)
    return histories[current_history]

def create_memory_summary(history_name=None):
    """Create a summary of available memories with indices for a specific history"""
    if history_name is None:
        history_name = current_history
    
    history = histories.get(history_name, {})
    
    if not history:
        return "No previous conversations available."
    
    summary = f"Available Memory Indices for '{history_name}':\n"
    for idx, conv in history.items():
        # Create a brief summary of each conversation
        user_msg = conv['user'][:100] + "..." if len(conv['user']) > 100 else conv['user']
        summary += f"Index {idx}: User asked about: {user_msg}\n"
    
    # Add recent context (last 2 conversations)
    last_keys = list(history.keys())[-2:]
    if last_keys:
        summary += "\nRecent Context:\n"
        for key in last_keys:
            summary += f"[{key}] User: {history[key]['user']}\n"
            response = history[key]['llm'].get('main_response', 'No response')
            if isinstance(response, dict):
                response = json.dumps(response)
            summary += f"[{key}] AI: {response[:200]}...\n"
    
    return summary

def get_detailed_memory(indices, history_name=None):
    """Retrieve detailed memory for specific indices from a specific history"""
    if history_name is None:
        history_name = current_history
    
    history = histories.get(history_name, {})
    detailed_memories = {}
    for idx in indices:
        if str(idx) in history:
            detailed_memories[idx] = history[str(idx)]
    return detailed_memories

def send_with_memory_access(user_msg, history_name=None):
    """Enhanced send function with dynamic memory access"""
    if history_name is None:
        history_name = current_history
        
    try:
        # Step 1: Send initial request with memory summary
        memory_summary = create_memory_summary(history_name)
        
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Memory Summary:\n{memory_summary}"},
            {"role": "user", "content": f"Current Question: {user_msg}"}
        ]

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=initial_messages,
            temperature=0.1,  # Lower temperature for more consistent JSON
        )

        raw_content = response.choices[0].message.content.strip()
        parsed_content = safe_json_parse(raw_content)
        
        # Step 2: Check if model requested specific memories
        memory_request = parsed_content.get('memory_request')
        
        if memory_request and isinstance(memory_request, list) and len(memory_request) > 0:
            print(f"🧠 AI requested memory access for indices: {memory_request}")
            
            # Get detailed memories
            detailed_memories = get_detailed_memory(memory_request, history_name)
            
            # Step 3: Send follow-up request with detailed memories
            followup_messages = [
                {"role": "system", "content": system_prompt.replace("NEED_MEMORY: [indices] if you need older memories", "provide your final answer")},
                {"role": "user", "content": f"Original Question: {user_msg}"},
                {"role": "user", "content": f"Memory Summary: {memory_summary}"},
                {"role": "user", "content": f"Detailed Memories Requested: {json.dumps(detailed_memories, indent=2)}"},
                {"role": "user", "content": "Now provide your complete final answer based on all available information."}
            ]

            final_response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=followup_messages,
                temperature=0.1,
            )

            final_content = safe_json_parse(final_response.choices[0].message.content.strip())
            
            print("AI (with memory access):", final_content.get("main_response", "No response"))
            return final_content
        else:
            print("AI:", parsed_content.get("main_response", "No response"))
            return parsed_content

    except Exception as e:
        print(f"Error: {e}")
        return {
            "error": str(e), 
            "main_response": f"Error occurred: {e}",
            "memory_request": None,
            "summarize_answer_prompt_in_100_words": f"Error in AI communication: {str(e)}",
            "summarize_question_prompt_in_100_words": f"User query that caused error: {user_msg[:100]}"
        }

def send_simple(user_msg, history_name=None):
    """Original simple send function (for comparison)"""
    if history_name is None:
        history_name = current_history
        
    history = histories.get(history_name, {})
    
    try:
        last_keys = list(history.keys())[-4:]   # last 4 
        last_memories = {k: history[k] for k in last_keys}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(last_memories)},  # last 4
            {"role": "user", "content": user_msg}
        ]

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.1,
        )

        raw_content = response.choices[0].message.content.strip()
        parsed_content = safe_json_parse(raw_content)
        print("AI:", parsed_content.get("main_response", "No response"))
        return parsed_content

    except Exception as e:
        print(f"Error: {e}")
        return {
            "error": str(e),
            "main_response": f"Error occurred: {e}",
            "memory_request": None,
            "summarize_answer_prompt_in_100_words": f"Error in AI communication: {str(e)}",
            "summarize_question_prompt_in_100_words": f"User query that caused error: {user_msg[:100]}"
        }

def save_history_to_file(filename, history_name=None):
    """Save a specific history to a file"""
    if history_name is None:
        history_name = current_history
    
    history = histories.get(history_name, {})
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, "w") as f:
            json.dump({
                "history_name": history_name,
                "conversations": history,
                "total_conversations": len(history)
            }, f, indent=4)
        print(f"💾 History '{history_name}' saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving history: {e}")
        return False

def load_history_from_file(filename, history_name=None):
    """Load a history from a file"""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        
        if history_name is None:
            history_name = data.get("history_name", "loaded_history")
        
        histories[history_name] = data.get("conversations", {})
        
        # Update counter
        if histories[history_name]:
            q_counters[history_name] = max([int(k) for k in histories[history_name].keys()])
        else:
            q_counters[history_name] = 0
            
        print(f"📁 History loaded as '{history_name}' from {filename}")
        print(f"   Total conversations: {len(histories[history_name])}")
        return True
    except Exception as e:
        print(f"❌ Error loading history: {e}")
        return False

def addhistory(user_msg, use_memory_access=True, history_name=None, save_to_file=None):
    """Add conversation to history with optional memory access and file saving"""
    if history_name is None:
        history_name = current_history
    
    # Initialize history if it doesn't exist
    init_history(history_name)
    
    # Switch to the specified history temporarily if different from current
    original_history = current_history
    if history_name != current_history:
        switch_history(history_name)
    
    # Increment counter for this history
    q_counters[history_name] += 1
    q = q_counters[history_name]

    if use_memory_access:
        llm_reply = send_with_memory_access(user_msg, history_name)
    else:
        llm_reply = send_simple(user_msg, history_name)

    histories[history_name][str(q)] = {
        "user": user_msg,
        "llm": llm_reply
    }

    # Save to specific file if provided
    if save_to_file:
        save_history_to_file(save_to_file, history_name)
    else:
        # Save to default file for current history
        default_filename = f"{history_name}_history.json"
        save_history_to_file(default_filename, history_name)

    # Switch back to original history if we switched temporarily
    if history_name != original_history:
        switch_history(original_history)

    return histories[history_name]

def show_memory_stats(history_name=None):
    """Display current memory statistics for a specific history"""
    if history_name is None:
        history_name = current_history
    
    history = histories.get(history_name, {})
    print(f"\n📊 Memory Stats for '{history_name}':")
    print(f"Total conversations: {len(history)}")
    print(f"Available indices: {list(history.keys())}")

def show_all_histories():
    """Show all available histories"""
    print(f"\n📚 All Histories:")
    print(f"Current active: {current_history}")
    for name, history in histories.items():
        marker = "👉 " if name == current_history else "   "
        print(f"{marker}{name}: {len(history)} conversations")

def show_commands():
    """Show available commands"""
    print("\n🔧 Available Commands:")
    print("/stats [history_name] - Show memory statistics")
    print("/history [history_name] - Show conversation history")
    print("/histories - Show all available histories")
    print("/switch <history_name> - Switch to a different history")
    print("/clear [history_name] - Clear history (current if none specified)")
    print("/save <filename> [history_name] - Save history to file")
    print("/load <filename> [history_name] - Load history from file")
    print("/simple - Toggle simple mode (no memory access)")
    print("/help - Show this help")
    print("/quit - Exit the program")
    print("\nUsage Examples:")
    print("  addhistory('Hello', True, 'chat1', 'chat1.json')")
    print("  /switch project_work")
    print("  /save backup.json")

# Initialize default history
init_history()