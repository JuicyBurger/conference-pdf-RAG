from typing import List, Dict, Any
from ollama import Client

class PromptRewriterLLM:
    def __init__(self, model: str = None):
        self.model = "llama3.1:8b-instruct-fp16"
        self.llm_client = Client(host="http://server_ip:11434")

    def rewrite(self, prompt: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Rewrite the prompt for optimal RAG search by EXPLICITLY considering chat history.

        Args:
            prompt: The latest user prompt to rewrite.
            chat_history: List of dicts. Each dict has 'role' ('user' or 'assistant') and 'content'.

        Returns:
            str: The rewritten prompt.
        """
        # Create a string representation of the chat history
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        # A robust prompt template that gives the LLM clear instructions and context
        prompt_template = (
            "You are an expert query rewriter for a RAG system. Your task is to synthesize the Chat History and the Current User Query into a single, self-contained, and highly specific query that is optimal for document retrieval.\n"
            "Analyze the conversation to resolve all contextual dependencies, such as pronouns (e.g., 'it', 'that'), ambiguous references, and follow-up questions. The final rewritten query must be perfectly understandable on its own.\n"
            "Response with the rewritten Query only without any explaination.\n"
            "### Chat History\n"
            f"{history_str}\n"
            "### Current User Query\n"
            f"{prompt}\n"
            "### Rewritten Query:"
        )

        # The messages list now contains only ONE clear, comprehensive instruction
        messages = [{"role": "user", "content": prompt_template}]

        # It's better to use a lower temperature for deterministic tasks like rewriting
        options = {"temperature": 0.0, "max_tokens": 128}
        
        response = self.llm_client.chat(model=self.model, messages=messages, options=options)
        rewritten_prompt = response["message"]["content"].strip()
        
        return rewritten_prompt

# Example usage
if __name__ == "__main__":
    chat_history = [
        {"role": "user", "content": "How do I use this repo?"},
        {"role": "assistant", "content": "You need to install dependencies and run the main script."}
    ]
    prompt = "Explain how to process a PDF using this repo."
    rewriter = PromptRewriterLLM()
    rewritten = rewriter.rewrite(prompt, chat_history)
    print("Rewritten prompt for RAG search:", rewritten)