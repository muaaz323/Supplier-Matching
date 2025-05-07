# Supplier-Matching

This solution matches sourcing events (provided as PDF files) with the most relevant suppliers from a structured list, using Natural Language Processing (NLP) and semantic matching techniques.

## Overview

This AI-powered solution automates the supplier matching process by:

1. Reading and understanding unstructured sourcing event descriptions in PDF format
2. Analyzing structured supplier profiles 
3. Using semantic matching and NLP to find the most relevant suppliers for each event
4. Providing ranked matches with confidence scores

## Architecture

The solution consists of three main components:

1. **Core Matching Engine**: Python-based system that handles PDF processing, text extraction, and supplier matching using NLP techniques
2. **REST API**: FastAPI server that exposes the matching functionality via HTTP endpoints

## Technical Approach

### 1. Text Extraction and Processing

- PDFs are processed using PyPDF2 to extract raw text
- Structured data is extracted using regex patterns and NLP analysis
- Key information like event name, category, tags, materials, and processes are identified

### 2. Supplier Data Processing

- Supplier information is read from the Excel file and cleaned
- Key supplier attributes are weighted appropriately (capabilities, industries, categories, materials, etc.)
- Both TF-IDF vectors and semantic embeddings are created for each supplier

### 3. Matching Algorithm

The matching uses a hybrid approach combining:

- **TF-IDF Similarity**: Term frequency-inverse document frequency to match on keyword overlap
- **Semantic Similarity**: Using sentence transformers (MiniLM) to capture deeper semantic relationships
- **Weighted Scoring**: Combined scores weighted 40% TF-IDF and 60% semantic similarity

This hybrid approach ensures both keyword matching and deeper semantic understanding.


## Technologies Used

- **Python 3.10**: Core programming language
- **NLP Libraries**:
  - spaCy: For text processing and entity recognition
  - sentence-transformers: For semantic embeddings
  - scikit-learn: For TF-IDF and similarity metrics
- **PDF Processing**: PyPDF2
- **Data Handling**: Pandas, NumPy
- **API Framework**: FastAPI

## Setup and Installation

### Prerequisites

- Python 3.10+
- Required Python packages: `requirements.txt` containing:
  ```
  pandas
  numpy
  scikit-learn
  spacy
  sentence-transformers
  pypdf2
  fastapi
  uvicorn
  ```

### Installation Steps

1. Clone the repository
2. Install Python dependencies: `pip install -r requirements.txt`
3. Download spaCy model: `python -m spacy download en_core_web_md`
4. Place the supplier list and sourcing event PDFs in the appropriate directories
5. Start the API server: `python app.py`

## Usage

### API Endpoints

- `POST /match-event`: Upload a sourcing event PDF and get matches
- `POST /batch-process`: Process multiple events at once

### Example API Request

```python
import requests

# Match a single event
with open('UseCase1.pdf', 'rb') as f:
    files = {'event_file': f}
    data = {'top_n': 5}  # Get top 5 matches
    response = requests.post('http://localhost:8000/match-event', files=files, data=data)
    results = response.json()
    
# Print top matches
for match in results['matches']:
    print(f"{match['Rank']}. {match['SupplierName']} - Score: {match['MatchScore']:.2f}")
```

## Results

For each sourcing event, the system produces:

1. Structured data extracted from the PDF
2. Top 3-5 supplier matches with scores

Example output format:
```json
{
  "EventName": "O-Ring 8,00 x 1,00 mm",
  "Category": "EPDM",
  "Tags": ["2030", "Oringe", "EPDMs"],
  "Materials": ["EPDM"],
  "Processes": [],
  "Matches": [
    {
      "Rank": 1,
      "SupplierName": "RubberTech Inc.",
      "SupplierID": 5,
      "MatchScore": 0.92,
      "TfidfScore": 0.88,
      "SemanticScore": 0.95
    },
    {
      "Rank": 2,
      "SupplierName": "PolyTech Solutions",
      "SupplierID": 2,
      "MatchScore": 0.76,
      "TfidfScore": 0.71,
      "SemanticScore": 0.79
    },
    {
      "Rank": 3,
      "SupplierName": "GreenMetal Manufacturing",
      "SupplierID": 1,
      "MatchScore": 0.33,
      "TfidfScore": 0.25,
      "SemanticScore": 0.38
    }
  ]
}
```
