from Bio.Seq import Seq
from pydantic import BaseModel, Field, field_validator

from typing import List, Optional, Dict, Any, Tuple

from langchain_core.tools import tool
import re
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# define class for protein sequence input data

class ProteinSequenceInput(BaseModel):
    sequence: str = Field(..., description="Input protein Sequence")  
    sequence_id: Optional[str] = Field(None, description="Input protein Sequence ID")
    sequence_name: Optional[str] = Field(None, description="Input protein Sequence Name")
    species: Optional[str] = Field(None, description="Input protein Sequence Species")

    @field_validator('sequence')
    def validate_sequence(cls, v):  
        if not isinstance(v, str):
            raise ValueError("Sequence must be a string")
        if not v.isalpha():
            raise ValueError("Sequence must contain only alphabetic characters")
        if not v.isupper():
            raise ValueError("Sequence must be in uppercase")
        if len(v) < 1:
            raise ValueError("Sequence must not be empty")

        standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        for char in v:
            if char not in standard_amino_acids:
                raise ValueError(f"Invalid amino acid: {char}. Only standard amino acids are allowed.")
            
        return v
    


# define tools

@tool
def preprocessing_protein_sequence(
    sequence: str,
    remove_any_number: bool = True,
    remove_any_special_characters: bool = True,
    remove_whitespace: bool = True,
    convert_to_upper: bool = True
) -> str:
    """
    Preprocess a protein sequence by removing unwanted characters and formatting.
    
    Args:
        sequence: Input protein sequence (may contain unwanted characters)
        remove_any_number: Remove numbers from sequence (default: True)
        remove_any_special_characters: Remove special characters (default: True)
        remove_whitespace: Remove whitespace characters (default: True)
        convert_to_upper: Convert sequence to uppercase (default: True)
    
    Returns:
        Cleaned protein sequence string
    """
    if not isinstance(sequence, str):
        raise ValueError("Input sequence must be a string")
    
    if remove_any_number:
        sequence = re.sub(r'\d+', '', sequence)
    
    if remove_any_special_characters:
        # Keep only letters (both cases)
        sequence = re.sub(r'[^a-zA-Z]', '', sequence)
    
    if remove_whitespace:
        sequence = re.sub(r'\s+', '', sequence)
    
    if convert_to_upper:
        sequence = sequence.upper()
    
    return sequence.strip()
import requests
#  STRING DB (Protein Interactions)
@tool
def search_string_db(protein_id: str) -> List[Dict[str, Any]]:
    """
    Search STRING database for protein interactions using protein ID."""

    url = f"https://string-db.org/api/json/network?identifiers={protein_id}"
    return requests.get(url).json()

#  CATH/Gene Ontology

@tool
def search_cath(domain_id: str) -> Dict[str, Any]:
    """
    Search CATH database for protein domain information using domain ID.
    
    Args:domain_id: CATH domain ID (e.g., "1a2bA")

    Returns:
        Dictionary containing CATH domain information

    """

    url = f"http://www.cathdb.info/version/v4_3_0/api/rest/domain/{domain_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Error fetching data from CATH: {response.status_code}")      
    
    return response.json()  

#  Wikipedia Search
@tool
def search_wikipedia(query: str) -> List[Dict[str, Any]]:
    """
    Search Wikipedia for a given query and return the top results.
    
    Args:
        query: Search query string
    
    Returns:
        List of dictionaries containing Wikipedia search results
    """
    wiki_search = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(),
        description="Search Wikipedia for a given query when it is required",
        top_k=5,
        return_only_outputs=True)
    return wiki_search.run(query=query)



def get_all_tools():
    """Returns all available tools"""
    return [search_wikipedia, search_cath]







