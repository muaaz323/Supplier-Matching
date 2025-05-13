"""
AI-Powered Supplier Matching Solution
---

A solution for matching sourcing events with relevant suppliers using NLP techniques.
"""

import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer
import PyPDF2
import os

# Load NLP models
try:
    nlp = spacy.load("en_core_web_md")  # Medium-sized English model
except:
    # If model not installed, download it
    import spacy.cli
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Load the sentence transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller model for faster performance

class SupplierMatcher:
    def __init__(self, supplier_file_path):
        """
        Initialize the SupplierMatcher with the supplier list.
        
        Args:
            supplier_file_path: Path to the Excel file containing supplier information
        """
        self.suppliers_df = pd.read_excel(supplier_file_path)
        self.prepare_supplier_data()
        self.embeddings_cache = {}
        
    def prepare_supplier_data(self):
        """
        Prepare and clean the supplier data for matching.
        """
        # Convert all column names to lowercase for consistency
        self.suppliers_df.columns = [col.lower() for col in self.suppliers_df.columns]
        
        
        key_columns = ['Supplier Name', 'Capability', 'Category', 'Description']
        
        # Fill NaN values with empty strings for text columns
        for col in self.suppliers_df.columns:
            if self.suppliers_df[col].dtype == 'object':
                self.suppliers_df[col] = self.suppliers_df[col].fillna('')
        
        # Create a combined text field for matching
        self.suppliers_df['combined_text'] = ''
        
        # Add all available text columns to combined text with appropriate weights
        for col in self.suppliers_df.columns:
            if col in key_columns:
                # Higher weight for key columns
                self.suppliers_df['combined_text'] += self.suppliers_df[col] + ' ' + self.suppliers_df[col] + ' '
            elif self.suppliers_df[col].dtype == 'object':
                # Add other text columns once
                self.suppliers_df['combined_text'] += self.suppliers_df[col] + ' '
                
        # Clean the combined text
        self.suppliers_df['combined_text'] = self.suppliers_df['combined_text'].apply(
            lambda x: ' '.join(re.sub(r'[^a-zA-Z0-9\s]', ' ', str(x).lower()).split())
        )
        
        # Create TF-IDF vectors for the suppliers
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.supplier_vectors = self.vectorizer.fit_transform(self.suppliers_df['combined_text'])
        
        # Create semantic embeddings for suppliers
        self.suppliers_df['embedding'] = self.suppliers_df['combined_text'].apply(
            lambda x: model.encode(x[:512])  # Limit length for performance
        )
        
    def read_pdf(self, pdf_path):
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def extract_event_data(self, pdf_path):
        """
        Extract structured data from a sourcing event PDF.
        
        Args:
            pdf_path: Path to the sourcing event PDF
            
        Returns:
            dict: Structured data from the event
        """
        text = self.read_pdf(pdf_path)
        
        # Extract basic info using regex patterns
        event_data = {
            'full_text': text,
            'name': '',
            'category': '',
            'tags': [],
            'keywords': [],
            'materials': [],
            'processes': []
        }
        
        # Common patterns in the PDFs
        name_patterns = [
            r'(?:Sourcing Event Name|Title|Eventname|Titel):\s*(.*?)(?:\n|$)',
            r'(?:Sourcing Event|Event Description):\s*(.*?)(?:\n|$)'
        ]
        
        category_patterns = [
            r'(?:Warengruppe|Category|Kategorie|Categories):\s*(.*?)(?:\n|$)',
            r'Business Units:\s*(.*?)(?:\n|$)'
        ]
        
        tag_patterns = [
            r'(?:Tags|Schlagwörter|Stichwörter|Tag)s?:\s*(.*?)(?:\n|$)'
        ]
        
        # Try to extract the event name
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not event_data['name']:
                event_data['name'] = match.group(1).strip()
        
        # Extract category
        for pattern in category_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['category'] = match.group(1).strip()
        
        # Extract tags
        for pattern in tag_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                tags = match.group(1).strip().split(',')
                event_data['tags'] = [tag.strip() for tag in tags]
        
        # Extract materials using NLP
        doc = nlp(text)
        
        # Common material keywords
        material_keywords = ['EPDM', 'EPP', 'Kunststoff', 'Alu', 'Stahl', 'material']
        
        # Extract materials mentioned
        for keyword in material_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                event_data['materials'].append(keyword)
        
        # Extract manufacturing processes
        process_keywords = ['extrusion', 'Co-Extrusion', 'mechanisch bearbeiten', 'Strangpressen']
        for keyword in process_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                event_data['processes'].append(keyword)
        
        # Get the processed text for matching
        event_data['processed_text'] = ' '.join([
            event_data['name'],
            event_data['category'],
            ' '.join(event_data['tags']),
            ' '.join(event_data['materials']),
            ' '.join(event_data['processes']),
            text  # Include full text but with lower weight
        ])
        
        return event_data
    
    def get_cached_embedding(self, text):
        """Get embedding from cache or compute it if not present"""
        if text not in self.embeddings_cache:
            self.embeddings_cache[text] = model.encode(text[:512])
        return self.embeddings_cache[text]
    
    def match_event_to_suppliers(self, event_data, top_n=5):
        """
        Match a sourcing event to the most relevant suppliers.
        
        Args:
            event_data: Structured data from the event
            top_n: Number of top suppliers to return
            
        Returns:
            list: Top N matching suppliers with scores
        """
        # Clean and prepare the event text
        clean_event_text = ' '.join(re.sub(r'[^a-zA-Z0-9\s]', ' ', 
                                           event_data['processed_text'].lower()).split())
        
        # Get TF-IDF vector for the event
        event_vector = self.vectorizer.transform([clean_event_text])
        
        # Calculate TF-IDF similarity
        tfidf_similarities = cosine_similarity(event_vector, self.supplier_vectors).flatten()
        
        # Get semantic similarity using sentence embeddings
        event_embedding = self.get_cached_embedding(clean_event_text)
        
        # Calculate semantic similarities
        semantic_similarities = np.array([
            cosine_similarity([event_embedding], [supplier_embedding])[0][0]
            for supplier_embedding in self.suppliers_df['embedding']
        ])
        
        # Combine similarity scores (weighted average)
        combined_scores = 0.4 * tfidf_similarities + 0.6 * semantic_similarities
        
        # Get top matches
        top_indices = combined_scores.argsort()[-top_n:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            supplier = self.suppliers_df.iloc[idx]
            results.append({
                'Rank': i + 1,
                'SupplierName': supplier['name'] if 'name' in supplier else f"Supplier_{idx}",
                'SupplierID': supplier['id'] if 'id' in supplier else idx,
                'MatchScore': float(combined_scores[idx]),
                'TfidfScore': float(tfidf_similarities[idx]),
                'SemanticScore': float(semantic_similarities[idx])
            })
        
        return convert_numpy_types(results)
    
    def process_all_events(self, event_folder, output_file=None):
        """
        Process all sourcing events in a folder and match them to suppliers.
        
        Args:
            event_folder: Folder containing the sourcing event PDFs
            output_file: Optional file path to save the results as JSON
            
        Returns:
            dict: Results for all events
        """
        results = {}
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(event_folder) if f.endswith('.pdf') and 'UseCase' in f]
        
        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(event_folder, pdf_file)
                event_data = self.extract_event_data(pdf_path)
                matches = self.match_event_to_suppliers(event_data)
                
                results[pdf_file] = {
                    'EventName': event_data['name'],
                    'Category': event_data['category'],
                    'Tags': event_data['tags'],
                    'Materials': event_data['materials'],
                    'Processes': event_data['processes'],
                    'Matches': matches
                }
                
                print(f"Processed {pdf_file}: found {len(matches)} matches")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        
        return results

    def evaluate_matches(self, ground_truth, results):
        """
        Evaluate matching performance if ground truth is provided.
        
        Args:
            ground_truth: Dictionary mapping event files to correct supplier IDs
            results: Results from process_all_events
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': []
        }
        
        for event_file, gt_suppliers in ground_truth.items():
            if event_file in results:
                predicted_suppliers = [match['SupplierID'] for match in results[event_file]['Matches']]
                
                # Calculate precision, recall, F1
                true_positives = len(set(predicted_suppliers) & set(gt_suppliers))
                precision = true_positives / len(predicted_suppliers) if predicted_suppliers else 0
                recall = true_positives / len(gt_suppliers) if gt_suppliers else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1_score'].append(f1)
        
        # Calculate averages
        for metric in metrics:
            if metrics[metric]:
                metrics[f'avg_{metric}'] = sum(metrics[metric]) / len(metrics[metric])
            else:
                metrics[f'avg_{metric}'] = 0
        
        return metrics

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj
