"""
FastAPI Server for AI-Powered Supplier Matching
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import shutil
import json
from typing import List, Optional
import io

# Import our supplier matcher class
from supplier_matcher import SupplierMatcher, convert_numpy_types

app = FastAPI(
    title="Supplier Matching API",
    description="API for matching sourcing events to relevant suppliers using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variable for the matcher
matcher = None

@app.on_event("startup")
async def startup_event():
    """Initialize the matcher on startup"""
    global matcher
    try:
        # Initialize the matcher with supplier data
        matcher = SupplierMatcher('materials//SupplierList.xlsx')
        print("Supplier matcher initialized successfully")
    except Exception as e:
        print(f"Error initializing supplier matcher: {str(e)}")
        # We'll initialize it on the first request if needed

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"status": "ok", "message": "Supplier Matching API is running"}

@app.post("/match-event")
async def match_event(
    event_file: UploadFile = File(...),
    top_n: int = Form(5)
):
    """
    Match a sourcing event (PDF) to relevant suppliers
    """
    global matcher
    
    # Initialize matcher if not already done
    if matcher is None:
        try:
            matcher = SupplierMatcher('materials//SupplierList.xlsx')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize matcher: {str(e)}")
    
    # Create a temporary file for the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # Write the uploaded file content to the temp file
        shutil.copyfileobj(event_file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Process the event
        event_data = matcher.extract_event_data(temp_path)
        matches = matcher.match_event_to_suppliers(event_data, top_n=top_n)

        matches = [convert_numpy_types(match) for match in matches]
        
        # Create the response
        response = {
            "event_name": event_data['name'],
            "category": event_data['category'],
            "tags": event_data['tags'],
            "matches": matches
        }
        
        # Clean up the temp file
        os.unlink(temp_path)
        
        return response
        
    except Exception as e:
        # Clean up temp file in case of error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing event: {str(e)}")

@app.post("/batch-process")
async def batch_process(
    event_files: List[UploadFile] = File(...),
    top_n: int = Form(5)
):
    """
    Process multiple sourcing events in batch
    """
    global matcher
    
    # Initialize matcher if not already done
    if matcher is None:
        try:
            matcher = SupplierMatcher('materials//SupplierList.xlsx')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize matcher: {str(e)}")
    
    results = {}
    temp_files = []
    
    try:
        # Process each uploaded file
        for event_file in event_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                shutil.copyfileobj(event_file.file, temp_file)
                temp_path = temp_file.name
                temp_files.append(temp_path)
            
            # Process the event
            event_data = matcher.extract_event_data(temp_path)
            matches = matcher.match_event_to_suppliers(event_data, top_n=top_n)

            matches = [convert_numpy_types(match) for match in matches]
            
            # Add to results
            results[event_file.filename] = {
                "event_name": event_data['name'],
                "category": event_data['category'],
                "tags": event_data['tags'],
                "matches": matches
            }
        
        # Clean up temp files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return convert_numpy_types(results)
        
    except Exception as e:
        # Clean up temp files in case of error
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing events: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)