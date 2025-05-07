import re

def parse_fasta(file_path):
    """
    Parses a FASTA file and returns a dictionary with sequence headers as keys and sequences as values.
    Parameters:
    - file_path (str): Path to the FASTA file.
    Returns:
    - dict: A dictionary where keys are sequence headers and values are sequences.
    """
    fasta_dict = {}
    with open(file_path, 'r') as file:
        header = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # Header line
                if header:  # Save the previous sequence
                    fasta_dict[header] = ''.join(sequence)
                header = line[1:]  # Remove ">"
                sequence = []  # Reset sequence list
            else:
                sequence.append(line)
        if header:  # Save the last sequence
            fasta_dict[header] = ''.join(sequence)
    return fasta_dict

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


from Bio.SeqUtils.ProtParam import ProteinAnalysis
import json  
from typing import Dict,Any
# basic sequence analysis

def basic_sequence_analysis(sequence: str) -> Dict[str,Any]:
    """
    Perform basic sequence analysis on a protein sequence.
    use BioPython's ProteinAnalysis module to analyze the sequence.

    Computes key protein properties including length, amino acid composition,
    molecular weight, isoelectric point, extinction coefficient, instability index,
    aromaticity, and hydrophobicity (GRAVY score).

    Args:
        sequence (str): Input protein sequence (should be a valid amino acid sequence).

    Returns:
        str: JSON-formatted string containing analysis results with the following structure:        JSON-formatted string containing analysis results with the following structure:
            {
                "sequence": str,
                "length": int,
                "amino_acid_composition": Dict[str, int],
                "molecular_weight": float,
                "isoelectric_point": float,
                "extinction_coefficient": Dict[str, float],
                "instability_index": float,
                "aromaticity": float,
                "hydrophobicity_gravy": float
            }

    Raises:
        ValueError: If input sequence contains no valid amino acid characters.
    
    """

    analyzed_seq = ProteinAnalysis(sequence)
    
    result= {
        "sequence": sequence,
        "length": len(sequence) if sequence else 0 ,
        "amino_acid_composition": analyzed_seq.count_amino_acids(),
        "molecular_weight": analyzed_seq.molecular_weight(),
        "isoelectric_point": analyzed_seq.isoelectric_point(),
        "extinction_coefficient": analyzed_seq.molar_extinction_coefficient(),
        "instability_index": analyzed_seq.instability_index(),
        "aromaticity": analyzed_seq.aromaticity(),
        "hydrophobicity_gravy": analyzed_seq.gravy(),
    }
    return json.dumps(result, indent=4)