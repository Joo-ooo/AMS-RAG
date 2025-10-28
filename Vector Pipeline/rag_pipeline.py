import os
import re
import gc
from typing import List, Dict

import torch
import pandas as pd
from tqdm import tqdm
from pdf2image import convert_from_path
from dotenv import load_dotenv, find_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    get_response_synthesizer,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision
from ragas.llms import llm_factory
from ragas.embeddings import LlamaIndexEmbeddingsWrapper


class ConfigManager:
    """Manages environment variables and file paths."""
    def __init__(self, env_path: str = None):
        dotenv_path = env_path if env_path else find_dotenv()
        if not dotenv_path:
            raise FileNotFoundError("Could not find a .env file.")
        load_dotenv(dotenv_path=dotenv_path)
        
        self.project_root = os.path.dirname(dotenv_path)
        self.data_directory = self._get_path_from_env('VECTOR_DATASET_DIR')
        self.benchmark_filepath = self._get_path_from_env('VECTOR_BENCHMARK_DATASET_DIR')

        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        print(f"Project root set to: {self.project_root}")
        print(f"Data directory set to: {self.data_directory}")

    def _get_path_from_env(self, env_var: str) -> str:
        relative_path = os.getenv(env_var)
        if not relative_path:
            raise ValueError(f"Environment variable '{env_var}' not set in .env file.")
        return os.path.join(self.project_root, relative_path)

class ModelProvider:
    """Initializes and provides access to all models."""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.llm = self._init_llm()
        self.ocr_model, self.ocr_processor = self._init_ocr_model()
        self.embed_model = self._init_embedding_model()
        self.reranker = self._init_reranker()
        
        # Set global settings for LlamaIndex
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        print("Global LlamaIndex settings configured.")

    def _init_llm(self) -> HuggingFaceLLM:
        """Initializes the primary Language Model from a local directory."""
        model_path = os.getenv("LLAMA_3.1_8B_INSTRUCT_DIR")
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"LLM directory not found at path: {model_path}")

        llm = HuggingFaceLLM(
            model_name=model_path,
            tokenizer_name=model_path,
            device_map="auto",
            max_new_tokens=1024,
            model_kwargs={"token": self.config.hf_token, "dtype": torch.bfloat16},
            generate_kwargs={"temperature": 0.1, "repetition_penalty": 1.2, "do_sample": True},
        )
        print(f"Llama 3.1 LLM loaded from: {model_path}")
        return llm

    def _init_ocr_model(self):
        """Initializes the Nanonets OCR model from a local directory."""
        model_path = os.getenv("NANONETS_OCR_S_DIR")
        if not model_path or not os.path.isdir(model_path):
            print(f"Warning: OCR model directory not found at path: {model_path}. OCR will be disabled.")
            return None, None

        try:
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map="auto",
                dtype=torch.bfloat16
            )
            print(f"Nanonets OCR model loaded from: {model_path}")
            return model, processor
        except Exception as e:
            print(f"Failed to load Nanonets OCR model from {model_path}. Error: {e}")
            return None, None
            
    def _init_embedding_model(self) -> HuggingFaceEmbedding:
        """Initializes the sentence embedding model from a local directory."""
        model_path = os.getenv("MULTILINGUAL_E5_LARGE_DIR")
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"Embedding model directory not found at path: {model_path}")

        embed_model = HuggingFaceEmbedding(
            model_name=model_path, 
            token=self.config.hf_token
        )
        print(f"Embedding model loaded from: {model_path}")
        return embed_model

    def _init_reranker(self) -> SentenceTransformerRerank:
        """Initializes the reranker model from a local directory."""
        model_path = os.getenv("BGE_RERANKER_V2_M_DIR")
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"Reranker model directory not found at path: {model_path}")

        reranker = SentenceTransformerRerank(
            model=model_path, 
            top_n=5
        )
        print(f"Reranker model loaded from: {model_path}")
        return reranker

class DocumentIngestor:
    """Handles PDF ingestion and OCR processing."""
    
    NANONETS_PROMPT = "Extract the text from the above document... Watermarks should be wrapped in brackets." # Truncated for brevity

    def __init__(self, models: ModelProvider):
        if not models.ocr_model or not models.ocr_processor:
            raise ValueError("OCR model is not available in ModelProvider.")
        self.ocr_model = models.ocr_model
        self.ocr_processor = models.ocr_processor

    def _batch_ocr_on_pages(self, images: List, page_batch_size: int = 8) -> List[str]:
        """Processes a list of page images in batches using the OCR model."""
        all_texts = []
        for i in tqdm(range(0, len(images), page_batch_size), desc="OCR on Pages"):
            batch_images = images[i:i + page_batch_size]
            content = []
            for _ in batch_images:
                content.append({"type": "image"})
                content.append({"type": "text", "text": self.NANONETS_PROMPT})
            
            messages = [{"role": "user", "content": content}]
            text = self.ocr_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.ocr_processor(text=text, images=batch_images, padding=True, return_tensors="pt").to(self.ocr_model.device)
            
            output_ids = self.ocr_model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            generated_ids = [output_id[len(input_id):] for input_id, output_id in zip(inputs.input_ids, output_ids)]
            batch_outputs = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            all_texts.extend(batch_outputs)
            del batch_images, inputs, output_ids, generated_ids
            gc.collect()
        return all_texts

    def ingest_pdfs_from_directory(self, data_directory: str, page_batch_size: int = 8) -> List[Document]:
        """Ingests all PDFs from a directory, performs OCR, and returns Document objects."""
        all_documents = []
        files_to_process = [f for f in os.listdir(data_directory) if f.lower().endswith('.pdf')]
        print(f"Found {len(files_to_process)} PDF files to process.")

        for filename in tqdm(files_to_process, desc="Processing Files"):
            filepath = os.path.join(data_directory, filename)
            try:
                images_from_pdf = convert_from_path(filepath, dpi=150)
                page_texts = self._batch_ocr_on_pages(images_from_pdf, page_batch_size)
                full_text_content = "\n".join(page_texts)

                if full_text_content.strip():
                    document = Document(text=full_text_content, metadata={"filepath": filepath})
                    all_documents.append(document)
                else:
                    print(f"- WARNING: No content was extracted from {filename}")
                
                del images_from_pdf, page_texts
                gc.collect()

            except Exception as e:
                print(f"- FAILED to process {filename}. Error: {e}")
                continue
        
        print(f"--- Ingestion complete. Successfully loaded {len(all_documents)} documents. ---")
        return all_documents


class DualPathEnsembleRetriever(BaseRetriever):
    """Custom retriever that runs BM25 and semantic retrieval in parallel and merges results."""
    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: VectorIndexRetriever, bm25_weight: float = 2.0):
        self._bm25_retriever = bm25_retriever
        self._vector_retriever = vector_retriever
        self._bm25_weight = bm25_weight
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        bm25_hits = self._bm25_retriever.retrieve(query_bundle)
        vector_hits = self._vector_retriever.retrieve(query_bundle)

        for node in bm25_hits:
            node.score *= self._bm25_weight

        combined_hits: Dict[str, NodeWithScore] = {}
        for hit in bm25_hits + vector_hits:
            if hit.node.node_id not in combined_hits or hit.score > combined_hits[hit.node.node_id].score:
                combined_hits[hit.node.node_id] = hit
        
        sorted_hits = sorted(list(combined_hits.values()), key=lambda x: x.score, reverse=True)
        return sorted_hits


class RAGPipeline:
    """Orchestrates the entire RAG pipeline from ingestion to querying."""
    def __init__(self, config: ConfigManager, models: ModelProvider):
        self.config = config
        self.models = models
        self.nodes = None
        self.query_engine = None
        self.ingestor = DocumentIngestor(models)

    def setup_pipeline(self, documents: List[Document], chunk_size=1024, chunk_overlap=200):
        """Builds the index, retrievers, and query engine."""
        print("--- Setting up RAG pipeline ---")
        # 1. Chunking
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.nodes = node_parser.get_nodes_from_documents(documents)
        print(f"Created {len(self.nodes)} nodes with chunk_size={chunk_size}.")

        # 2. Indexing and Retrievers
        vector_index = VectorStoreIndex(self.nodes)
        vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
        bm25_retriever = BM25Retriever.from_defaults(nodes=self.nodes, similarity_top_k=5)
        
        # 3. Ensemble Retriever
        dual_retriever = DualPathEnsembleRetriever(bm25_retriever, vector_retriever)
        print("Initialized Dual-Path Ensemble Retriever.")

        # 4. Response Synthesizer
        custom_prompt = PromptTemplate(
            "You are a precise QA assistant. Use ONLY the context to answer the single question. "
            "Do not invent or add anything not present in the context. Output exactly one sentence and then stop.\n"
            "Context: {context_str}\n"
            "User's Question: {query_str}\n"
            "Answer: "
        )
        response_synthesizer = get_response_synthesizer(text_qa_template=custom_prompt)

        # 5. Base Query Engine (Ensemble + Reranker)
        ensemble_query_engine = RetrieverQueryEngine(
            retriever=dual_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[self.models.reranker]
        )
        
        # 6. Advanced Query Engine with HyDE
        hyde_prompt = PromptTemplate(
            "Drawing from the style of a Singapore Police Force annual crime report, write a concise, "
            "data-driven paragraph that directly answers the user's question. Focus on using specific "
            "crime-related keywords, percentages, and years. Do not invent statistics.\n"
            "Question: {context_str}\n"
            "Report Excerpt:"
        )
        hyde_transform = HyDEQueryTransform(llm=self.models.llm, include_original=True, hyde_prompt=hyde_prompt)
        
        self.query_engine = TransformQueryEngine(
            ensemble_query_engine,
            query_transform=hyde_transform
        )
        print("Advanced Query Engine with HyDE transform is ready.")

    def _post_process_answer(self, raw_answer: str, original_query: str) -> str:
        """Cleans and rephrases the final answer to be conversational."""
        answer_core = str(raw_answer).strip().rstrip('.')
        if not answer_core:
            return "Sorry, I couldn't extract an answer from the context."

        rephrase_prompt_text = (
            f"Rephrase the following information into a single, simple, conversational sentence. "
            f"Your response must contain the Extracted Answer exactly as it is provided. "
            f"Do not add any extra information.\n"
            f"Extracted answer: {answer_core}\n"
            f"Question: {original_query}\n"
            f"Conversational Answer:"
        )
        
        rephrased_raw = self.models.llm.complete(rephrase_prompt_text, max_new_tokens=32)
        s = ' '.join(str(rephrased_raw).strip().split())
        
        # Ensure the core answer is in the final output
        if answer_core not in s:
            s = f"The answer is {answer_core}."
        
        # Ensure it's a single sentence
        idx = s.find('.')
        s = s[:idx+1].strip() if idx != -1 else s + '.'
        
        return s

    def query(self, query_text: str) -> dict:
        """Executes a query against the pipeline and returns the response and source nodes."""
        if not self.query_engine:
            raise RuntimeError("Pipeline is not set up. Please run `setup_pipeline` first.")
            
        print(f"Executing query: {query_text}")
        response = self.query_engine.query(query_text)
        
        final_answer = self._post_process_answer(str(response), query_text)
        
        return {
            "answer": final_answer,
            "source_nodes": response.source_nodes,
        }