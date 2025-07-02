import json
import logging
from typing import Tuple, Dict, Any

def load_patient_data():
    """Load patient data from JSON file."""
    try:
        with open('data/patients.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("patients.json not found")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing patients.json: {e}")
        return []

def get_patient_report(patient_name: str) -> Tuple[Dict[str, Any], str]:
    """
    Retrieve patient report by name.
    Returns: (patient_report_dict, status)
    Status: 'found', 'not_found', 'multiple_found'
    """
    try:
        patients = load_patient_data()
        
        # Find exact matches (case insensitive)
        exact_matches = [p for p in patients if p['patient_name'].lower() == patient_name.lower()]
        
        if len(exact_matches) == 1:
            logging.info(f"Patient found: {patient_name}")
            return exact_matches[0], 'found'
        elif len(exact_matches) > 1:
            logging.warning(f"Multiple patients found with name: {patient_name}")
            return {}, 'multiple_found'
        
        # Try partial matches
        partial_matches = [p for p in patients if patient_name.lower() in p['patient_name'].lower()]
        
        if len(partial_matches) == 1:
            logging.info(f"Patient found (partial match): {patient_name}")
            return partial_matches[0], 'found'
        elif len(partial_matches) > 1:
            logging.warning(f"Multiple partial matches for: {patient_name}")
            return {}, 'multiple_found'
        
        logging.warning(f"Patient not found: {patient_name}")
        return {}, 'not_found'
        
    except Exception as e:
        logging.error(f"Error retrieving patient report: {e}")
        return {}, 'error'