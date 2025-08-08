from typing import List, Dict, Any
from ..models.client_factory import get_llm_client, get_default_model
from ..models.LLM import LLM

class PromptRewriterLLM:
    def __init__(self, model: str = None):
        self.model = get_default_model()
        self.llm_client = get_llm_client()

    def rewrite(self, prompt: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Rewrite the prompt for optimal RAG search by EXPLICITLY considering chat history.

        Args:
            prompt: The latest user prompt to rewrite.
            chat_history: List of dicts. Each dict has 'role' ('user' or 'ai') and 'content'.

        Returns:
            str: The rewritten prompt.
        """
        # Create a string representation of the chat history
        # Convert our role format (user/ai) to standard format for the prompt
        history_str = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
            for msg in chat_history
        ])

        # Conservative prompt: do not invent unseen terms; preserve literals
        system_prompt = """You are an expert query rewriter for a RAG system.
Your goal is CONSERVATIVE clarification: keep all named entities, numbers, filenames (e.g., .pdf), and page references exactly as they appear. Do NOT introduce any new metrics, terms, or entities not present in the chat history or current query.
If the user query is already clear or too short to safely improve, return it UNCHANGED.
Output ONLY the rewritten query with no extra text."""

        user_prompt = f"""### Chat History
{history_str}

### Current User Query
{prompt}

### Rewritten Query:"""

        # Use our standard LLM function with deterministic settings
        options = {"temperature": 0.0, "max_tokens": 128}
        
        try:
            rewritten_prompt = LLM(
                self.llm_client,
                self.model,
                system_prompt,
                user_prompt,
                options=options,
                timeout=30.0,
                raw=True
            )
            
            return rewritten_prompt.strip()
            
        except Exception as e:
            print(f"❌ Query rewriting failed: {e}")
            # Fallback to original prompt
            return prompt

# Example usage
if __name__ == "__main__":
    chat_history = [
        {"role": "user", "content": "How do I use this repo?"},
        {"role": "ai", "content": "You need to install dependencies and run the main script."}
    ]
    prompt = "Explain how to process a PDF using this repo."
    rewriter = PromptRewriterLLM()
    rewritten = rewriter.rewrite(prompt, chat_history)
    print("Rewritten prompt for RAG search:", rewritten)