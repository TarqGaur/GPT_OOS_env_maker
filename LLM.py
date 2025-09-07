from openai import OpenAI
import json
import re

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

history = {}
q = 0

def create_memory_summary():
    """Create a summary of available memories with indices"""
    if not history:
        return "No previous conversations available."
    
    summary = "Available Memory Indices:\n"
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

def get_detailed_memory(indices):
    """Retrieve detailed memory for specific indices"""
    detailed_memories = {}
    for idx in indices:
        if str(idx) in history:
            detailed_memories[idx] = history[str(idx)]
    return detailed_memories

def send_with_memory_access(user_msg):
    """Enhanced send function with dynamic memory access"""
    try:
        # Step 1: Send initial request with memory summary
        memory_summary = create_memory_summary()
        
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
            detailed_memories = get_detailed_memory(memory_request)
            
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

def send_simple(user_msg):
    """Original simple send function (for comparison)"""
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

def addhistory(user_msg, use_memory_access=True):
    """Add conversation to history with optional memory access"""
    global history, q
    q += 1

    if use_memory_access:
        llm_reply = send_with_memory_access(user_msg)
    else:
        llm_reply = send_simple(user_msg)

    history[str(q)] = {
        "user": user_msg,
        "llm": llm_reply
    }

    # Save to file
    with open("llmhist.json", "w") as f:
        json.dump(history, f, indent=4)

    return history

def show_memory_stats():
    """Display current memory statistics"""
    print(f"\n📊 Memory Stats:")
    print(f"Total conversations: {len(history)}")
    print(f"Available indices: {list(history.keys())}")
    
def show_commands():
    """Show available commands"""
    print("\n🔧 Available Commands:")
    print("/stats - Show memory statistics")
    print("/history - Show conversation history")
    print("/clear - Clear all history")
    print("/simple - Toggle simple mode (no memory access)")
    print("/help - Show this help")
    print("/quit - Exit the program")

# Main interaction loop
use_memory_access = True
print("🤖 Enhanced Memory-Access LLM Chat")
print("The AI can now dynamically access older conversations when needed!")
show_commands()

while True:
    user_input = input("\nYou: ").strip()
    
    # Handle commands
    if user_input.startswith('/'):
        if user_input == '/quit':
            break
        elif user_input == '/stats':
            show_memory_stats()
        elif user_input == '/history':
            print("\n📚 Conversation History:")
            for idx, conv in history.items():
                print(f"[{idx}] User: {conv['user'][:100]}...")
                print(f"[{idx}] AI: {conv['llm'].get('main_response', 'No response')[:100]}...")
                print()
        elif user_input == '/clear':
            history.clear()
            q = 0
            print("🗑️ History cleared!")
        elif user_input == '/simple':
            use_memory_access = not use_memory_access
            mode = "Memory Access" if use_memory_access else "Simple"
            print(f"🔄 Switched to {mode} mode")
        elif user_input == '/help':
            show_commands()
        else:
            print("❓ Unknown command. Type /help for available commands.")
        continue
    
    # Process regular conversation
    if user_input:
        addhistory(user_input, use_memory_access)