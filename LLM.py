from openai import OpenAI
import json
import re
import os
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from requests.adapters import HTTPAdapter, Retry

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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"  
)

# Enhanced system prompt that teaches the model about memory access AND web search
system_prompt = """
You are an AI assistant with access to a memory system AND web search capability. You MUST respond in **strict JSON format** only.

CRITICAL: Your response must be VALID JSON that can be parsed by json.loads(). Do not include any text before or after the JSON object.

MEMORY ACCESS RULES:
- You will receive a "memory_summary" containing recent conversations
- If you need detailed information from older conversations, you can request specific memory indices
- To request memory access, include in your main_response: "NEED_MEMORY: [index1, index2, ...]"
- Available memory indices will be shown in the memory_summary

WEB SEARCH RULES:
- If you need current information, real-time data, or information not in your training, you can request web search
- To request web search, include in your main_response: "NEED_SEARCH: search_query_here"
- You can request both memory access AND web search in the same response
- Web search will provide you with current information from multiple sources

Follow this exact JSON schema:

{
  "main_response": "your response here, include NEED_MEMORY: [indices] and/or NEED_SEARCH: query if needed",
  "memory_request": ["index1", "index2"] or null,
  "search_request": "search_query" or null,
  "summarize_answer_prompt_in_100_words": "summarize your answer in 100 words",
  "summarize_question_prompt_in_100_words": "summarize the user's question in 100 words"
}

CRITICAL RULES:
1. ONLY return valid JSON - nothing else
2. Do not include markdown formatting, code blocks, or explanations outside JSON
3. If you don't need memory access, set "memory_request" to null
4. If you don't need web search, set "search_request" to null
5. If you need specific memories, list the indices in "memory_request" array
6. If you need web search, provide a clear search query in "search_request"
7. Always provide a complete response in "main_response"
8. Escape all quotes and special characters properly in JSON strings
"""

# Global variables for multiple histories
histories = {}
current_history = "default"
q_counters = {}

def scrap(search_query):
    """Enhanced web scraping function with better error handling"""
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/115.0 Safari/537.36"
        }

        params = {
            "engine": "google",
            "q": search_query,   
            "hl": "en",
            "gl": "us",
            "num": "3",     
            "api_key": "d71b08519c184d0b2d7d339e7c0e4dba161f48c0095680ae57419a6f70d3d501"
        }

        print(f"🔍 Searching for: {search_query}")
        search = GoogleSearch(params)
        results = search.get_dict()

        organic_results = results.get("organic_results", [])[:3]
        total_data = []

        for i, result in enumerate(organic_results, start=1):
            try:
                url = result.get("link")
                title = result.get("title")
                print(f"🔗 Site {i}: {title}\nURL: {url}")
                
                response = session.get(url, headers=headers, timeout=20)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract text from paragraphs and limit length
                text = " ".join([p.get_text().strip() for p in soup.find_all("p")])
                if len(text) > 2000:  # Limit text length to avoid token limits
                    text = text[:2000] + "..."
                
                total_data.append({
                    "title": title,
                    "url": url,
                    "content": text
                })
            except Exception as site_error:
                print(f"❌ Error scraping site {i}: {site_error}")
                continue

        return total_data
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return []

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
            "search_request": None,
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
    
    # Add recent context (last 5 conversations)
    last_keys = list(history.keys())[-5:]
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

def send_with_enhanced_access(user_msg, history_name=None):
    """Enhanced send function with dynamic memory access AND web search"""
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
            temperature=0.1,
        )

        raw_content = response.choices[0].message.content.strip()
        parsed_content = safe_json_parse(raw_content)
        
        # Step 2: Check if model requested specific memories or web search
        memory_request = parsed_content.get('memory_request')
        search_request = parsed_content.get('search_request')
        
        # Collect additional data
        additional_data = {}
        
        # Handle memory request
        if memory_request and isinstance(memory_request, list) and len(memory_request) > 0:
            print(f"🧠 AI requested memory access for indices: {memory_request}")
            additional_data['detailed_memories'] = get_detailed_memory(memory_request, history_name)
        
        # Handle search request
        if search_request and isinstance(search_request, str) and search_request.strip():
            print(f"🌐 AI requested web search for: {search_request}")
            search_results = scrap(search_request.strip())
            additional_data['search_results'] = search_results
        
        # Step 3: Send follow-up request with additional data if any was requested
        if additional_data:
            followup_messages = [
                {"role": "system", "content": system_prompt.replace(
                    "include NEED_MEMORY: [indices] and/or NEED_SEARCH: query if needed", 
                    "provide your final answer using the additional information provided"
                )},
                {"role": "user", "content": f"Original Question: {user_msg}"},
                {"role": "user", "content": f"Memory Summary: {memory_summary}"}
            ]
            
            if 'detailed_memories' in additional_data:
                followup_messages.append({
                    "role": "user", 
                    "content": f"Detailed Memories: {json.dumps(additional_data['detailed_memories'], indent=2)}"
                })
            
            if 'search_results' in additional_data:
                search_content = ""
                for i, result in enumerate(additional_data['search_results'], 1):
                    search_content += f"\nSource {i}: {result['title']}\nURL: {result['url']}\nContent: {result['content']}\n"
                followup_messages.append({
                    "role": "user",
                    "content": f"Web Search Results: {search_content}"
                })
            
            followup_messages.append({
                "role": "user",
                "content": "Now provide your complete final answer based on all available information."
            })

            final_response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=followup_messages,
                temperature=0.1,
            )

            final_content = safe_json_parse(final_response.choices[0].message.content.strip())
            
            # Add metadata about what was used
            final_content['used_memory'] = bool(memory_request)
            final_content['used_search'] = bool(search_request)
            
            print("AI (with enhanced access):", final_content.get("main_response", "No response"))
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
            "search_request": None,
            "summarize_answer_prompt_in_100_words": f"Error in AI communication: {str(e)}",
            "summarize_question_prompt_in_100_words": f"User query that caused error: {user_msg[:100]}"
        }

def send_with_memory_access(user_msg, history_name=None):
    """Original memory-only access function (for backward compatibility)"""
    return send_with_enhanced_access(user_msg, history_name)

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

        # Safely extract content with null checks
        if response and response.choices and len(response.choices) > 0:
            raw_content = response.choices[0].message.content
            if raw_content is not None:
                raw_content = raw_content.strip()
            else:
                raw_content = None
        else:
            raw_content = None

        parsed_content = safe_json_parse(raw_content)
        print("AI:", parsed_content.get("main_response", "No response"))
        return parsed_content

    except Exception as e:
        print(f"Error: {e}")
        return {
            "error": str(e),
            "main_response": f"Error occurred: {e}",
            "memory_request": None,
            "search_request": None,
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

def addhistory(user_msg, use_enhanced_access=True, history_name=None, save_to_file=None):
    """Add conversation to history with optional enhanced access (memory + web search) and file saving"""
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

    if use_enhanced_access:
        llm_reply = send_with_enhanced_access(user_msg, history_name)
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
    print("/simple - Toggle simple mode (no memory/search access)")
    print("/search <query> - Test web search functionality")
    print("/help - Show this help")
    print("/quit - Exit the program")
    print("\n🌟 New Features:")
    print("- AI can now request web search for current information")
    print("- Enhanced access mode includes both memory and web search")
    print("- Search results are integrated into AI responses")
    print("\nUsage Examples:")
    print("  addhistory('What is the current weather in New York?', True, 'weather_chat')")
    print("  addhistory('Tell me about recent AI developments', True)")
    print("  /search 'latest OpenAI news'")

# Test function for web search
def test_search(query):
    """Test the web search functionality"""
    print(f"\n🧪 Testing search for: {query}")
    results = scrap(query)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content: {result['content'][:300]}...")
    return results

# Initialize default history
init_history()

# Example usage
if __name__ == "__main__":
    while True:
        inp = str(input("msg"))
        addhistory(
    user_msg=inp, 
    use_enhanced_access=True, 
    history_name="testchat", 
    save_to_file="testchat.json"
        )
