"""
Fine-tune Jina embedding model for Chinese financial documents
"""

import os
import json
import random
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import pandas as pd


class FinancialDocumentTrainer:
    """Fine-tune embedding model for Chinese financial documents"""
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-zh"):
        self.model = SentenceTransformer(model_name)
        self.training_examples = []
        self.eval_examples = []
    
    def generate_training_data_from_pdfs(self, pdf_paths: List[str]) -> List[InputExample]:
        """
        Generate training pairs from existing PDF documents
        """
        from ..data.pdf_ingestor import build_page_nodes
        
        examples = []
        
        for pdf_path in pdf_paths:
            print(f"📄 Processing {pdf_path}...")
            nodes = build_page_nodes(pdf_path)
            
            # Create positive pairs from same document/page
            doc_nodes = {}
            for node in nodes:
                key = f"{node['page']}"
                if key not in doc_nodes:
                    doc_nodes[key] = []
                doc_nodes[key].append(node['text'])
            
            # Generate positive pairs (same page content)
            for page_texts in doc_nodes.values():
                if len(page_texts) > 1:
                    for i in range(len(page_texts)):
                        for j in range(i + 1, len(page_texts)):
                            examples.append(InputExample(
                                texts=[page_texts[i], page_texts[j]], 
                                label=0.8  # High similarity for same page
                            ))
            
            # Generate negative pairs (different pages, different docs)
            all_texts = [node['text'] for node in nodes]
            for _ in range(min(100, len(all_texts) // 2)):  # Limit negative examples
                text1, text2 = random.sample(all_texts, 2)
                examples.append(InputExample(
                    texts=[text1, text2], 
                    label=0.2  # Low similarity for random pairs
                ))
        
        print(f"✅ Generated {len(examples)} training examples")
        return examples
    
    def create_qa_training_data(self, qa_files: List[str]) -> List[InputExample]:
        """
        Create training data from Q&A pairs (JSON files)
        """
        examples = []
        
        for qa_file in qa_files:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            for item in qa_data:
                question = item.get('question', '')
                answer = item.get('answer', '')
                
                if question and answer:
                    # Q&A pairs should be highly similar
                    examples.append(InputExample(
                        texts=[question, answer],
                        label=0.9
                    ))
        
        print(f"✅ Generated {len(examples)} Q&A training examples from JSON files")
        return examples
    
    def extract_qa_from_pdfs(self, qa_pdf_paths: List[str]) -> List[InputExample]:
        """
        Extract Q&A pairs from historical Q&A PDF documents
        """
        from ..data.pdf_ingestor import build_page_nodes
        import re
        
        examples = []
        
        for pdf_path in qa_pdf_paths:
            print(f"📄 Extracting Q&A from {pdf_path}...")
            nodes = build_page_nodes(pdf_path)
            
            for node in nodes:
                text = node['text']
                
                # Extract Q&A patterns from text
                qa_pairs = self._parse_qa_patterns(text)
                
                for question, answer in qa_pairs:
                    examples.append(InputExample(
                        texts=[question, answer],
                        label=0.95  # High confidence for historical Q&A
                    ))
        
        print(f"✅ Generated {len(examples)} Q&A training examples from PDF documents")
        return examples
    
    def _parse_qa_patterns(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse Q&A patterns from text using Chinese patterns
        """
        qa_pairs = []
        
        # Common Chinese Q&A patterns
        patterns = [
            # Q: ... A: ...
            r'[Qq問問題][:：]\s*([^A答]*?)\s*[Aa答][:：]\s*([^Q問]*?)(?=[QA問答]|$)',
            # 問題: ... 回答: ...
            r'問題[:：]\s*([^回]*?)\s*回答[:：]\s*([^問]*?)(?=問題|$)',
            # 問: ... 答: ...
            r'問[:：]\s*([^答]*?)\s*答[:：]\s*([^問]*?)(?=問|$)',
            # Question followed by answer paragraph
            r'([^。]*?[？?][^。]*?)([。][^？?]*?(?=[^。]*?[？?]|$))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                if len(match) == 2:
                    question = match[0].strip()
                    answer = match[1].strip()
                    
                    # Validate Q&A quality
                    if (len(question) > 5 and len(answer) > 10 and 
                        question != answer):
                        qa_pairs.append((question, answer))
        
        return qa_pairs
    
    def create_financial_term_pairs(self) -> List[InputExample]:
        """
        Create training pairs for financial terminology
        """
        financial_pairs = [
            # Financial metrics (high similarity)
            ("營業收入", "營收", 0.95),
            ("淨利潤", "稅後淨利", 0.9),
            ("毛利率", "毛利", 0.85),
            ("資產負債表", "資產負債", 0.9),
            ("現金流量", "現金流", 0.95),
            ("股東權益", "股東報酬", 0.85),
            
            # Business terms (medium similarity)
            ("市場佔有率", "市場份額", 0.9),
            ("競爭優勢", "核心競爭力", 0.8),
            ("業務成長", "營業成長", 0.85),
            
            # Unrelated terms (low similarity)
            ("營業收入", "員工人數", 0.1),
            ("股價", "天氣", 0.0),
            ("財務報表", "產品介紹", 0.2),
        ]
        
        examples = []
        for term1, term2, similarity in financial_pairs:
            examples.append(InputExample(
                texts=[term1, term2],
                label=similarity
            ))
        
        print(f"✅ Generated {len(examples)} financial term examples")
        return examples
    
    def prepare_training_data(self, pdf_paths: List[str], qa_files: List[str] = None, qa_pdf_paths: List[str] = None):
        """Prepare all training data"""
        all_examples = []
        
        # Add PDF-based examples (document structure learning)
        all_examples.extend(self.generate_training_data_from_pdfs(pdf_paths))
        
        # Add Q&A examples from JSON files
        if qa_files:
            all_examples.extend(self.create_qa_training_data(qa_files))
        
        # Add Q&A examples from historical Q&A PDFs (VERY VALUABLE!)
        if qa_pdf_paths:
            all_examples.extend(self.extract_qa_from_pdfs(qa_pdf_paths))
        
        # Add financial term examples
        all_examples.extend(self.create_financial_term_pairs())
        
        # Split into train/eval
        random.shuffle(all_examples)
        split_idx = int(0.8 * len(all_examples))
        
        self.training_examples = all_examples[:split_idx]
        self.eval_examples = all_examples[split_idx:]
        
        print(f"📊 Training data: {len(self.training_examples)} examples")
        print(f"📊 Evaluation data: {len(self.eval_examples)} examples")
    
    def fine_tune(self, output_path: str = "models/chinese-financial-embeddings", epochs: int = 3):
        """Fine-tune the model"""
        
        if not self.training_examples:
            raise ValueError("No training data available. Call prepare_training_data() first.")
        
        # Create data loader
        train_dataloader = DataLoader(self.training_examples, shuffle=True, batch_size=16)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Create evaluator
        if self.eval_examples:
            sentences1 = [ex.texts[0] for ex in self.eval_examples]
            sentences2 = [ex.texts[1] for ex in self.eval_examples]
            scores = [ex.label for ex in self.eval_examples]
            
            evaluator = EmbeddingSimilarityEvaluator(
                sentences1, sentences2, scores,
                name="chinese-financial-eval"
            )
        else:
            evaluator = None
        
        # Fine-tune
        print(f"🚀 Starting fine-tuning for {epochs} epochs...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            evaluator=evaluator,
            evaluation_steps=500,
            warmup_steps=100,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True
        )
        
        print(f"✅ Fine-tuning completed! Model saved to: {output_path}")
        return output_path


def main():
    """Main training script"""
    import glob
    
    # Get all document PDFs (regular business documents)
    pdf_paths = glob.glob("data/raw/*.pdf")
    
    # Get historical Q&A PDFs (separate folder recommended)
    qa_pdf_paths = glob.glob("data/qa_pdfs/*.pdf")  # Historical Q&A PDFs
    
    # Get Q&A JSON files
    qa_files = glob.glob("*.json")  # bulk_qa_zh.json, etc.
    
    if not pdf_paths and not qa_pdf_paths:
        print("❌ No PDF files found in data/raw/ or data/qa_pdfs/")
        return
    
    print("📂 Found training data:")
    print(f"  - Document PDFs: {len(pdf_paths)}")
    print(f"  - Q&A PDFs: {len(qa_pdf_paths)}")
    print(f"  - Q&A JSON files: {len(qa_files)}")
    
    # Initialize trainer
    trainer = FinancialDocumentTrainer()
    
    # Prepare training data with historical Q&A PDFs
    trainer.prepare_training_data(
        pdf_paths=pdf_paths, 
        qa_files=qa_files,
        qa_pdf_paths=qa_pdf_paths  # Historical Q&A PDFs!
    )
    
    # Fine-tune
    model_path = trainer.fine_tune(epochs=3)
    
    print(f"🎉 Fine-tuned model ready at: {model_path}")
    print("💡 Update src/embedder.py to use the new model:")
    print(f'    model = SentenceTransformer("{model_path}")')
    print("\n🔄 Then re-index all documents:")
    print('    python -m src.cli index "data/raw/*.pdf"')


if __name__ == "__main__":
    main() 