from sentence_transformers import SentenceTransformer
from typing import Union, List
import torch

# Optimize model loading with performance settings
model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-zh",
    device="cuda"
)

# Enable optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # model.half()  # REMOVED: FP16 causes precision loss leading to Qdrant VectorStruct errors
    print("🚀 Using GPU acceleration for embeddings")
else:
    print("💻 Using CPU for embeddings")

def embed(text: Union[str, List[str]]):
    """
    Generate embeddings for single text or batch of texts.
    
    Args:
        text: Single text string or list of text strings
        
    Returns:
        Single embedding list or list of embedding lists
    """
    if isinstance(text, str):
        # Single text
        return model.encode(text, convert_to_tensor=False).tolist()
    else:
        # Batch of texts with optimized settings
        embeddings = model.encode(
            text,
            batch_size=32,  # Optimize batch size for memory/speed balance
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=True  # Ensure normalized embeddings
        )
        return embeddings.tolist()
