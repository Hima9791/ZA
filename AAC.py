# STREAMLIT SCRIPT WITH UNIFIED PIPELINE LOGIC
#############################################

import streamlit as st
import pandas as pd
import re
import os
import base64
import requests
import gc
from io import BytesIO
import io # Make sure io is imported
import uuid # For potential unique ID generation if needed

#############################################
# HIDE GITHUB ICON & OTHER ELEMENTS
#############################################
hide_button = """
    <style>
    [data-testid="stBaseButton-header"] {
        display: none;
    }
    </style>
    """
st.markdown(hide_button, unsafe_allow_html=True)


#############################################
# 1) GITHUB HELPER FUNCTIONS (No changes needed)
#############################################

# --- download_mapping_file_from_github() --- (Keep as is)
def download_mapping_file_from_github() -> pd.DataFrame:
    """
    Downloads 'mapping.xlsx' from the GitHub repo specified in Streamlit secrets,
    returns a DataFrame parsed from that file.
    """
    st.write("DEBUG: Downloading mapping.xlsx from GitHub...")
    github_token = st.secrets["github"]["token"]
    owner = st.secrets["github"]["owner"]
    repo = st.secrets["github"]["repo"]
    file_path = st.secrets["github"]["file_path"]  # e.g. "mapping.xlsx"

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content_json = response.json()
        encoded_content = content_json["content"]
        decoded_bytes = base64.b64decode(encoded_content)

        local_file = "mapping.xlsx"
        try: # Write to disk immediately
            with open(local_file, "wb") as f:
                f.write(decoded_bytes)
        except Exception as e:
             st.error(f"Failed to write downloaded mapping to disk: {e}")
             st.stop()


        # Now parse the local file into a DataFrame
        try:
            df = pd.read_excel(local_file)
        except Exception as e:
            st.error(f"Failed to parse downloaded mapping file: {e}")
            st.stop()

        st.write("DEBUG: Download successful. mapping_df shape:", df.shape)
        return df
    else:
        st.error(f"Failed to download file from GitHub: {response.status_code} {response.text}")
        st.stop()

# --- update_mapping_file_on_github() --- (Keep as is)
def update_mapping_file_on_github(mapping_df: pd.DataFrame) -> bool:
    """
    Updates 'mapping.xlsx' on GitHub using a PUT request to the GitHub API.
    """
    st.write("DEBUG: Attempting to update mapping.xlsx on GitHub.")
    st.write("DEBUG: DataFrame shape before upload:", mapping_df.shape)

    github_token = st.secrets["github"]["token"]
    owner = st.secrets["github"]["owner"]
    repo = st.secrets["github"]["repo"]
    file_path = st.secrets["github"]["file_path"]

    # 1) Save DF to temp in-memory BytesIO buffer
    buffer = BytesIO()
    mapping_df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    content_bytes = buffer.read()

    # 2) Encode local file in base64
    encoded_content = base64.b64encode(content_bytes).decode("utf-8")

    # 3) Get the current file's SHA
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    current_response = requests.get(url, headers=headers)
    sha = None
    if current_response.status_code == 200:
        sha = current_response.json().get("sha")
        st.write("DEBUG: Current file SHA:", sha)
    elif current_response.status_code == 404:
         st.write("DEBUG: No existing file found. Creating a new one...")
    else:
        st.error(f"Failed to get current file info from GitHub: {current_response.status_code} {current_response.text}")
        # No temp file to remove here
        return False


    # 4) Prepare data payload
    data = {
        "message": "Update mapping file via Streamlit app",
        "content": encoded_content
    }
    if sha:
        data["sha"] = sha

    # 5) PUT request to update file
    update_response = requests.put(url, headers=headers, json=data)

    if update_response.status_code in [200, 201]:
        st.write("DEBUG: Update/creation successful:", update_response.status_code)
        # Force re-download next time or update session state
        if "mapping_df" in st.session_state:
            del st.session_state["mapping_df"]
        # Ensure local disk version is updated after successful upload
        save_mapping_to_disk(mapping_df, "mapping.xlsx") # Resave the confirmed version
        return True
    else:
        st.error(f"Failed to update file on GitHub: {update_response.status_code} {update_response.text}")
        return False

#############################################
# 2) GLOBAL MULTIPLIER MAPPING + READ MAPPING (No changes needed)
#############################################

MULTIPLIER_MAPPING = {
    'y': 1e-24, 'z': 1e-21, 'a': 1e-18, 'f': 1e-15,
    'p': 1e-12, 'n': 1e-9,  'µ': 1e-6,  'm': 1e-3,
    'c': 1e-2,  'd': 1e-1,  'da': 1e1,  'h': 1e2,
    'k': 1e3,   'M': 1e6,   'G': 1e9,   'T': 1e12,
    'P': 1e15,  'E': 1e18,  'Z': 1e21,  'Y': 1e24
}

LOCAL_BASE_UNITS = set()

# --- save_mapping_to_disk() --- (Keep as is)
def save_mapping_to_disk(df: pd.DataFrame, filename="mapping.xlsx"):
    """
    Helper: saves the current mapping DataFrame to a local 'mapping.xlsx',
    so the pipeline can read it from disk.
    """
    try:
        df.to_excel(filename, index=False, engine='openpyxl')
        st.write(f"DEBUG: Saved mapping to {filename}")
    except Exception as e:
        st.error(f"Error saving mapping to disk ({filename}): {e}")
        # Consider if stopping is appropriate or just log error
        # st.stop()

# --- read_mapping_file() --- (Keep as is, reads from local disk)
def read_mapping_file(mapping_file_path: str):
    """
    Reads the local 'mapping.xlsx', checks for required columns and multipliers.
    Returns set of base units and the multiplier mapping dict.
    """
    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError(f"Error: Mapping file '{mapping_file_path}' not found. It should be downloaded from GitHub first.")
    try:
        df = pd.read_excel(mapping_file_path)
    except Exception as e:
        raise ValueError(f"Error reading mapping file '{mapping_file_path}': {e}")

    required_cols = {'Base Unit Symbol', 'Multiplier Symbol'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"'{mapping_file_path}' must have columns: {required_cols}")

    base_units = set(str(x).strip() for x in df['Base Unit Symbol'].dropna().unique())

    # Combine known base units with LOCAL_BASE_UNITS if any
    combined_base_units = LOCAL_BASE_UNITS.union(base_units)

    st.write(f"DEBUG: Read mapping. Base units found: {len(combined_base_units)}. Multipliers known: {len(MULTIPLIER_MAPPING)}")
    return combined_base_units, MULTIPLIER_MAPPING

#############################################
# 3) EXISTING PARSING/CLASSIFICATION HELPERS (Keep necessary ones)
#    Functions like:
#    - classify_value_type_detailed
#    - extract_identifiers_detailed
#    - classify_sub_value
#    - count_main_items, count_conditions
#    - classify_main, classify_condition
#    - extract_numeric_info_for_value
#    - analyze_value_units
#    - extract_numeric_and_unit_analysis
#    - resolve_compound_unit
#    - replace_numbers_keep_sign_all
#    - remove_parentheses_detailed
#    - split_outside_parens
#    - safe_str
#    - fix_exceptions
#    Make sure all these are defined before they are called by analyze_and_code_value
#############################################

# =================== INSERT YOUR EXISTING HELPER FUNCTIONS HERE ==================
# For brevity, I am assuming these functions are present and correct as per your
# original script. Crucially ensure the following are kept and function correctly:
# * classify_value_type_detailed(raw_value: str) -> tuple
# * extract_numeric_info_for_value(raw_value, base_units, multipliers_dict) -> dict
# * analyze_value_units(raw_value, base_units, multipliers_dict) -> dict
# * resolve_compound_unit(raw_value, base_units, multipliers_dict) -> str
# * replace_numbers_keep_sign_all(s: str) -> str
# * split_outside_parens(text, delimiters) -> list
# * fix_exceptions(s: str) -> str
# * get_desired_order() -> list [Still needed for final output column ordering]

# --- Assume necessary helper functions are defined above this line ---
# Re-paste your existing definitions for these functions here.

# Placeholder functions - REPLACE with your actual implementations from Sections 3 & 4
def extract_numeric_and_unit_analysis(token, base_units, multipliers_dict):
     # YOUR ACTUAL IMPLEMENTATION HERE
     print(f"Placeholder: extract_numeric_and_unit_analysis for '{token}'")
     # Simplified example return
     if token.startswith("10kOhm"): return 10000.0, "k", "Ohm", 10000.0 * 1e3, False # Original Numeric, Multiplier, Base, Normalized, Error
     if token.startswith("5V"): return 5.0, "1", "V", 5.0, False
     if token == "V": return None, None, "V", None, False # Treat as unit only case handled inside analyze/extract_numeric_info...
     numeric_match = re.match(r'([+\-±]?\d+(?:\.\d+)?)', token)
     if numeric_match:
        return float(numeric_match.group(1).replace('±','')), "1", None, float(numeric_match.group(1).replace('±','')), False
     return None, None, None, None, True # Error

def remove_parentheses_detailed(text: str) -> str:
     # YOUR ACTUAL IMPLEMENTATION HERE
     return re.sub(r'\([^)]*\)', '', text)

def extract_numeric_info(part_text, base_units, multipliers_dict):
    # YOUR ACTUAL IMPLEMENTATION HERE
     print(f"Placeholder: extract_numeric_info for '{part_text}'")
     # Based on calling extract_numeric_and_unit_analysis on tokens
     text = remove_parentheses_detailed(part_text).strip()
     tokens = [text] # Simplified split
     # ... loop through tokens, call extract_numeric_and_unit_analysis ...
     # Dummy return
     if "10" in text:
         return {"numeric_values": [10.0], "multipliers": ["k"], "base_units": ["Ohm"], "normalized_values": [10000.0], "error_flags": [False], "type": "single"}
     else:
          return {"numeric_values": [], "multipliers": [], "base_units": [], "normalized_values": [], "error_flags": [], "type": "none"}


def extract_numeric_info_for_value(raw_value, base_units, multipliers_dict):
    # YOUR ACTUAL IMPLEMENTATION HERE
     print(f"Placeholder: extract_numeric_info_for_value for '{raw_value}'")
     main_part, cond_part = raw_value.split('@', 1) if '@' in raw_value else (raw_value, "")
     main_info = extract_numeric_info(main_part.strip(), base_units, multipliers_dict)
     cond_info = extract_numeric_info(cond_part.strip(), base_units, multipliers_dict)
     return { # Combine results
         "main_numeric": main_info["numeric_values"], "main_multipliers": main_info["multipliers"],
         "main_base_units": main_info["base_units"], "normalized_main": main_info["normalized_values"], "main_errors": main_info["error_flags"],
         "condition_numeric": cond_info["numeric_values"], "condition_multipliers": cond_info["multipliers"],
         "condition_base_units": cond_info["base_units"], "normalized_condition": cond_info["normalized_values"], "condition_errors": cond_info["error_flags"]
     }

def safe_str(item, placeholder="None"):
    # YOUR ACTUAL IMPLEMENTATION HERE
    return str(item) if item is not None else placeholder

def analyze_unit_part(part_text, base_units, multipliers_dict):
    # YOUR ACTUAL IMPLEMENTATION HERE
     print(f"Placeholder: analyze_unit_part for '{part_text}'")
     text = remove_parentheses_detailed(part_text).strip()
     tokens = [text] # Simplified split
     # ... call extract_numeric_and_unit_analysis ...
     if "Ohm" in text:
          units = ["Ohm"]
     elif "V" in text:
          units = ["V"]
     else: units = []
     distinct = set(u for u in units if u != "None")
     return {"units": units, "distinct_units": distinct, "is_consistent": len(distinct)<=1, "count": len(units), "type": "single"}

def analyze_value_units(raw_value, base_units, multipliers_dict):
     # YOUR ACTUAL IMPLEMENTATION HERE
     print(f"Placeholder: analyze_value_units for '{raw_value}'")
     main_part, cond_part = raw_value.split('@', 1) if '@' in raw_value else (raw_value, "")
     main_analysis = analyze_unit_part(main_part.strip(), base_units, multipliers_dict)
     cond_analysis = analyze_unit_part(cond_part.strip(), base_units, multipliers_dict)
     all_distinct = main_analysis["distinct_units"].union(cond_analysis["distinct_units"])
     return { # Combine results
         "main_units": main_analysis["units"], "main_distinct_units": main_analysis["distinct_units"],
         "main_units_consistent": main_analysis["is_consistent"], "main_unit_count": main_analysis["count"],
         "condition_units": cond_analysis["units"], "condition_distinct_units": cond_analysis["distinct_units"],
         "condition_units_consistent": cond_analysis["is_consistent"], "condition_unit_count": cond_analysis["count"],
         "all_distinct_units": all_distinct, "overall_consistent": len(all_distinct)<=1,
          "main_sub_analysis": "", "condition_sub_analysis": "" # Placeholder
     }

def resolve_compound_unit(normalized_unit_string, base_units, multipliers_dict):
     # YOUR ACTUAL IMPLEMENTATION HERE - Should analyze the raw string and structure
     print(f"Placeholder: resolve_compound_unit for '{normalized_unit_string}'")
     ua = analyze_value_units(normalized_unit_string, base_units, multipliers_dict)
     if not ua["all_distinct_units"]: return "None"
     # This logic needs refinement based on your original implementation
     main_repr = ",".join(sorted(list(ua["main_distinct_units"]))) if ua["main_distinct_units"] else ""
     cond_repr = ",".join(sorted(list(ua["condition_distinct_units"]))) if ua["condition_distinct_units"] else ""
     if main_repr and cond_repr: return f"{main_repr}@{cond_repr}"
     elif main_repr: return main_repr
     elif cond_repr: return cond_repr # Or perhaps format differently if only condition
     else: return "None"


def count_main_items(main_str: str) -> int:
    # YOUR ACTUAL IMPLEMENTATION HERE
    return len(split_outside_parens(main_str, [','])) if main_str else 0

def count_conditions(cond_str: str) -> int:
    # YOUR ACTUAL IMPLEMENTATION HERE
    return len(split_outside_parens(cond_str, [','])) if cond_str else 0

def classify_condition(cond_str: str) -> str:
    # YOUR ACTUAL IMPLEMENTATION HERE
    num_parts = count_conditions(cond_str)
    if num_parts == 0: return ""
    has_range = any(re.search(r'\s+to\s+', part) for part in split_outside_parens(cond_str, [',']))
    if num_parts > 1: return "Multiple Conditions"
    elif has_range: return "Range Condition"
    else: return "Single Condition"


def classify_main(main_str: str) -> str:
    # YOUR ACTUAL IMPLEMENTATION HERE
    num_parts = count_main_items(main_str)
    if num_parts == 0: return ""
    has_range = any(re.search(r'\s+to\s+', part) for part in split_outside_parens(main_str, [',']))
    # Simple checks, replace with your robust ones
    is_number = all(re.fullmatch(r'[+\-±]?\d+(?:\.\d+)?', part.strip()) for part in split_outside_parens(main_str, [',']))
    if num_parts > 1: return "Multi Value"
    elif has_range: return "Range Value"
    elif is_number: return "Number"
    else: return "Single Value" # Default


def classify_sub_value(subval: str):
    # YOUR ACTUAL IMPLEMENTATION HERE
     if "@" in subval:
         parts = subval.split('@', 1)
         main_part = parts[0].strip()
         cond_part = parts[1].strip() if len(parts) > 1 else ""
     else:
         main_part = subval.strip()
         cond_part = ""
     main_class = classify_main(main_part)
     cond_class = classify_condition(cond_part)
     # ... determine bool flags, counts ...
     classification = f"{main_class} with {cond_class}" if main_class and cond_class else main_class or "Invalid"
     return (classification,
             main_class=="Range Value", main_class=="Multi Value",
             cond_class=="Range Condition", cond_class=="Multiple Conditions",
             count_conditions(cond_part), count_main_items(main_part))

def extract_identifiers_detailed(text: str):
    # YOUR ACTUAL IMPLEMENTATION HERE
    return ', '.join(re.findall(r'\(([^()]*)\)', text))

def classify_value_type_detailed(raw_value: str):
     # YOUR ACTUAL IMPLEMENTATION HERE
     # Based on calling classify_sub_value on comma-separated chunks
     raw_value = str(raw_value).strip()
     if not raw_value:
         return ("", "", 0, 0, False, False, False, False, 0)
     identifiers = extract_identifiers_detailed(raw_value)
     # Simplistic splitting for placeholder - use your split_outside_parens logic
     subvals = [v.strip() for v in raw_value.split(',') if v.strip()]
     if not subvals: subvals = [raw_value] # Ensure at least one if original was not empty

     # ... loop through subvals, call classify_sub_value, aggregate results ...
     # Dummy aggregation
     all_res = [classify_sub_value(sv) for sv in subvals]
     final_class = "Multiple Mixed" if len(subvals)>1 else all_res[0][0] if all_res else "Invalid"
     return (final_class, identifiers, len(subvals),
             max(r[5] for r in all_res) if all_res else 0, # Condition count
             any(r[1] for r in all_res), # has range main
             any(r[2] for r in all_res), # has multi main (within subval? - clarify)
             any(r[3] for r in all_res), # has range condition
             any(r[4] for r in all_res), # has multi condition
             max(r[6] for r in all_res) if all_res else 0  # main item count
             )


def replace_numbers_keep_sign_all(s: str) -> str:
    # YOUR ACTUAL IMPLEMENTATION HERE
    return re.sub(r'([+-])?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', r'\1$', str(s))

def split_outside_parens(text, delimiters):
    # YOUR ACTUAL IMPLEMENTATION HERE
     # Basic placeholder - needs depth counting
     if not text: return []
     pattern = '|'.join(map(re.escape, delimiters))
     return [s.strip() for s in re.split(pattern, text) if s.strip()] # simplistic

def fix_exceptions(s):
    # YOUR ACTUAL IMPLEMENTATION HERE
     # Simple placeholder:
     s_new = s
     if "kOhm" in s: s_new = s.replace("kOhm", "k Ohm")
     return s_new

def get_desired_order():
    # YOUR ACTUAL IMPLEMENTATION HERE (Still needed for final column ordering)
    # Should include all possible codes generated by _get_codes_for_structure
     return [
         "SN-V", "SN-U", "SV-V", "SV-U", "CX-V", "CX-U",
         "RV-Vn", "RV-Un", "RV-Vx", "RV-Ux",
         "SC-V", "SC-U",
         "RC-Vn", "RC-Un", "RC-Vx", "RC-Ux",
         "MC-V1", "MC-U1", "MC-V2", "MC-U2", # etc.
         # M1-...M2-... prefixes handled in merge setup if needed, or dynamically added here if predictable max needed
         "ERR-PROC", "ERR-CLS", "ERR-CHUNK" # Example error codes
     ]


#==============================================================================


#############################################
# 4) UNIFIED PIPELINE CORE FUNCTION
#############################################

def _classify_value(value_str: str) -> dict:
    """Internal helper to perform classification."""
    try:
        (cls, ids, sv_count, cond_count, rng_main, multi_main,
         rng_cond, multi_cond, main_count) = classify_value_type_detailed(value_str)

        # Determine detailed type string
        if cls and cls not in ["Invalid Structure", "Classification Error", "Unknown", "Empty"]:
             dvt = f"{cls} [M:{main_count}][C:{cond_count}]"
             if sv_count > 1 : dvt += f" (x{sv_count})"
        elif not value_str:
             cls = "Empty"
             dvt = "Empty"
        else:
             cls = cls or "Unknown Classification"
             dvt = cls # Use the classification result directly if invalid/error

        return {
            "Classification": cls,
            "DetailedValueType": dvt,
            "Identifiers": ids,
            "SubValueCount": sv_count,
            "ConditionCount": cond_count,
            "HasRangeInMain": rng_main,
            "HasMultiValueInMain": multi_main,
            "HasRangeInCondition": rng_cond,
            "HasMultipleConditions": multi_cond,
            "MainItemCount": main_count, # Added for clarity
             "classification_error": False
        }
    except Exception as e:
         st.warning(f"Error during detailed classification for '{value_str}': {e}")
         # Return default error structure
         return {
            "Classification": "Classification Error",
            "DetailedValueType": f"Error: {e}",
            "Identifiers": extract_identifiers_detailed(value_str), # Try to get identifiers
            "SubValueCount": 1, # Assume 1 on error
            "ConditionCount": 0, "HasRangeInMain": False, "HasMultiValueInMain": False,
            "HasRangeInCondition": False, "HasMultipleConditions": False, "MainItemCount": 0,
            "classification_error": True
         }

def _extract_details(value_str: str, base_units: set, multipliers_dict: dict) -> dict:
    """Internal helper for numeric extraction, unit analysis, and normalization."""
    results = {}
    try:
        # Numeric Extraction
        num_info = extract_numeric_info_for_value(value_str, base_units, multipliers_dict)
        results.update({
            "MainNumericValues": ", ".join(safe_str(x) for x in num_info["main_numeric"]),
            "ConditionNumericValues": ", ".join(safe_str(x) for x in num_info["condition_numeric"]),
            "MainMultipliers": ", ".join(safe_str(x) for x in num_info["main_multipliers"]),
            "ConditionMultipliers": ", ".join(safe_str(x) for x in num_info["condition_multipliers"]),
            "MainBaseUnits": ", ".join(safe_str(x) for x in num_info["main_base_units"]),
            "ConditionBaseUnits": ", ".join(safe_str(x) for x in num_info["condition_base_units"]),
            "NormalizedMainValues": ", ".join(safe_str(x) for x in num_info["normalized_main"]),
            "NormalizedConditionValues": ", ".join(safe_str(x) for x in num_info["normalized_condition"]),
            "ParsingErrorFlag": any(num_info["main_errors"]) or any(num_info["condition_errors"]),
             # Store raw normalized lists for min/max calc
             "_normalized_all_numeric": [v for v in num_info["normalized_main"] + num_info["normalized_condition"] if isinstance(v, (int, float))],
             "_all_base_units_found": [u for u in num_info["main_base_units"] + num_info["condition_base_units"] if u and u != "None"]
        })

        # Unit Analysis
        ua = analyze_value_units(value_str, base_units, multipliers_dict)
        main_variation = "None"
        if ua["main_distinct_units"]:
            main_variation = ("Uniform: " + next(iter(ua["main_distinct_units"]))) if ua["main_units_consistent"] else ("Mixed: " + ", ".join(sorted(ua["main_distinct_units"])))
        condition_variation = "None"
        if ua["condition_distinct_units"]:
            condition_variation = ("Uniform: " + next(iter(ua["condition_distinct_units"]))) if ua["condition_units_consistent"] else ("Mixed: " + ", ".join(sorted(ua["condition_distinct_units"])))

        results.update({
            "MainUnits": ", ".join(safe_str(x) for x in ua["main_units"]),
            "MainDistinctUnitCount": len(ua["main_distinct_units"]),
            "MainUnitsConsistent": ua["main_units_consistent"],
            "ConditionUnits": ", ".join(safe_str(x) for x in ua["condition_units"]),
            "ConditionDistinctUnitCount": len(ua["condition_distinct_units"]),
            "ConditionUnitsConsistent": ua["condition_units_consistent"],
            "OverallUnitConsistency": ua["overall_consistent"],
            "SubValueUnitVariationSummary": f"Main: {main_variation}; Condition: {condition_variation}",
             # Store distinct units for overall summary
             "_all_distinct_units_set": ua["all_distinct_units"]
        })

        # Normalizations (String Representations)
        results["Normalized Unit"] = replace_numbers_keep_sign_all(value_str) # Dollar sign format
        results["Absolute Unit"] = resolve_compound_unit(value_str, base_units, multipliers_dict) # Resolved base units format

        results["extraction_error"] = False

    except Exception as e:
        st.warning(f"Error during numeric/unit extraction for '{value_str}': {e}")
        # Provide default values for all expected keys in case of error
        default_numeric = {"MainNumericValues": "Error", "ConditionNumericValues": "Error", "MainMultipliers": "Error",
                          "ConditionMultipliers": "Error", "MainBaseUnits": "Error", "ConditionBaseUnits": "Error",
                          "NormalizedMainValues": "Error", "NormalizedConditionValues": "Error", "ParsingErrorFlag": True,
                          "_normalized_all_numeric": [], "_all_base_units_found": []}
        default_unit = {"MainUnits": "Error", "MainDistinctUnitCount": 0, "MainUnitsConsistent": False,
                       "ConditionUnits": "Error", "ConditionDistinctUnitCount": 0, "ConditionUnitsConsistent": False,
                       "OverallUnitConsistency": False, "SubValueUnitVariationSummary": "Error",
                       "_all_distinct_units_set": set()}
        default_norm = {"Normalized Unit": "Error", "Absolute Unit": "Error"}
        results.update(default_numeric)
        results.update(default_unit)
        results.update(default_norm)
        results["extraction_error"] = True

    return results


def _build_analysis_dict(value_str:str, classification_data: dict, extraction_data: dict) -> dict:
    """Combines classification and extraction results into the final analysis dict."""
    analysis = {"Original Value": value_str} # Use a distinct name
    analysis.update(classification_data)
    analysis.update(extraction_data)

    # Calculate derived summaries
    all_norm_num = analysis.pop("_normalized_all_numeric", []) # Remove temp key
    all_units_found = analysis.pop("_all_base_units_found", []) # Remove temp key
    all_distinct_units = analysis.pop("_all_distinct_units_set", set()) # Remove temp key

    analysis["MinNormalizedValue"] = min(all_norm_num) if all_norm_num else None
    analysis["MaxNormalizedValue"] = max(all_norm_num) if all_norm_num else None
    analysis["SingleUnitForAllSubs"] = (len(all_distinct_units) <= 1)
    analysis["AllDistinctUnitsUsed"] = ", ".join(sorted(list(all_distinct_units))) if all_distinct_units else ""

    return analysis

def _get_codes_for_structure(is_range_main, main_count, is_range_cond, cond_count, is_num_only):
    """
    Determines the list of code pairs ([ValueCode, UnitCode]) based on structure.
    Refined logic based on Proposal 3. Needs careful mapping definition.
    Returns a list of lists, e.g., [['SV-V', 'SV-U'], ['SC-V', 'SC-U']]
    """
    codes = []

    # --- Main Part Codes ---
    if is_range_main: # e.g., "10A to 20A"
        codes.append(["RV-Vn", "RV-Un"]) # Min/Start
        codes.append(["RV-Vx", "RV-Ux"]) # Max/End
    elif main_count > 0 : # Handle Single Value or Number (excluding ranges handled above)
        if is_num_only: # Assuming classification determined this (e.g., from classify_main)
             codes.append(["SN-V", "SN-U"]) # Number only
        else:
             codes.append(["SV-V", "SV-U"]) # Single Value with Unit (Default if not number)
        # Note: 'Complex Single' could map here or have its own 'CX-' code if needed
        # Note: Multi Value Main (e.g., "10A, 20A @ 5V") is handled by splitting value_str first, this function gets called on chunks.

    # --- Condition Part Codes ---
    if is_range_cond: # e.g., "@ 5V to 10V"
         codes.append(["RC-Vn", "RC-Un"]) # Cond Range Min
         codes.append(["RC-Vx", "RC-Ux"]) # Cond Range Max
    elif cond_count == 1: # e.g., "@ 5V"
        codes.append(["SC-V", "SC-U"])   # Single Condition
    elif cond_count > 1: # e.g., "@ 5V, 10A"
         # Generate multiple condition codes dynamically
         for i in range(1, cond_count + 1):
              codes.append([f"MC-V{i}", f"MC-U{i}"])

    if not codes and (main_count > 0 or cond_count > 0):
        # Fallback if structure exists but no codes assigned (shouldn't happen ideally)
        st.warning(f"Warning: Could not assign codes for structure (main_range={is_range_main}, mc={main_count}, cond_range={is_range_cond}, cc={cond_count}, num_only={is_num_only})")
        codes.append(["UNK-V", "UNK-U"])

    return codes

def _generate_coded_rows(value_str: str, classification_data: dict, extraction_data: dict, base_units: set, multipliers_dict: dict) -> list:
    """Generates the list of coded rows (long format)."""
    coded_rows = []
    value_str_clean = fix_exceptions(value_str.strip())

    sub_value_count = classification_data.get("SubValueCount", 1)
    classification_name = classification_data.get("Classification", "Unknown")

    # Handle cases where classification itself indicated an error
    if classification_data.get("classification_error") or "Error" in classification_name:
        return [{"Category": classification_name, "Attribute": "Error", "Code": "ERR-CLS", "Value": classification_data.get("DetailedValueType","Unknown Classification Error")}]
    if extraction_data.get("extraction_error"):
        # Extraction failed, cannot generate reliable codes
        return [{"Category": classification_name, "Attribute": "Error", "Code": "ERR-EXT", "Value": "Numeric/Unit extraction failed"}]
    if classification_name == "Empty":
         return [] # No coded rows for empty values

    # Logic based on whether the value has multiple top-level comma-separated parts
    if sub_value_count > 1:
         # Split the original string using the same logic as classification
         chunks = split_outside_parens(value_str_clean, [','])
         # Refine check if split count matches expected count
         if len(chunks) != sub_value_count:
             st.warning(f"Sub-value count mismatch for '{value_str_clean}'. Expected {sub_value_count}, got {len(chunks)} chunks. Processing chunks found.")
             # Potentially add an error row or adjust processing

         for idx, chunk in enumerate(chunks):
             chunk = chunk.strip()
             if not chunk: continue
             chunk_prefix = f"M{idx+1}-" # Prefix for multiple chunks

             try:
                 # 1. Re-classify the chunk (needed to determine *its* structure for coding)
                 chunk_cls_data = _classify_value(chunk) # Get structure flags for the chunk
                 chunk_cat = chunk_cls_data.get("Classification", "Unknown Chunk")
                 if "Error" in chunk_cat or chunk_cat == "Unknown Chunk":
                     coded_rows.append({"Category": chunk_cat, "Attribute": "Error", "Code": f"{chunk_prefix}ERR-CHUNK", "Value": f"Could not classify chunk: {chunk}"})
                     continue

                 # 2. Extract block values/units for *this chunk* (using helper similar to old extract_block_texts/parse_value_unit)
                 # This needs careful implementation - get values associated with the structure identified in chunk_cls_data
                 # Simplified: use overall extraction data and attempt to split/match? No, re-extracting for chunk is safer.
                 # Let's try re-extraction for simplicity, although less efficient:
                 chunk_numeric_info = extract_numeric_info_for_value(chunk, base_units, multipliers_dict)

                 # Determine if chunk main part is numeric only (example logic)
                 chunk_main_vals = chunk_numeric_info.get("main_numeric", [])
                 chunk_main_units = [u for u in chunk_numeric_info.get("main_base_units", []) if u and u != "None"]
                 is_chunk_num_only = bool(chunk_main_vals) and not bool(chunk_main_units)

                 # 3. Get code structure for the chunk
                 code_pairs = _get_codes_for_structure(
                     chunk_cls_data.get("HasRangeInMain", False),
                     chunk_cls_data.get("MainItemCount", 0),
                     chunk_cls_data.get("HasRangeInCondition", False),
                     chunk_cls_data.get("ConditionCount", 0),
                     is_chunk_num_only
                 )

                 # 4. Extract actual value/unit strings corresponding to codes (tricky part!)
                 # Need a mapping from code position (e.g., first code pair is main value) to extracted numeric info
                 # This logic replaces extract_block_texts & parse_value_unit_identifier
                 extracted_values_for_chunk = []
                 # Main Part Values
                 main_vals_str = [safe_str(v) for v in chunk_numeric_info.get("main_numeric", [])]
                 main_units_str = [safe_str(u) for u in chunk_numeric_info.get("main_base_units", [])]
                 if chunk_cls_data.get("HasRangeInMain", False) and len(main_vals_str) >= 2:
                     extracted_values_for_chunk.append( (main_vals_str[0], main_units_str[0]) ) # Range Min
                     extracted_values_for_chunk.append( (main_vals_str[1], main_units_str[1]) ) # Range Max
                 elif main_vals_str:
                     extracted_values_for_chunk.append( (main_vals_str[0], main_units_str[0] if main_units_str else None) ) # Single Main

                 # Condition Part Values
                 cond_vals_str = [safe_str(v) for v in chunk_numeric_info.get("condition_numeric", [])]
                 cond_units_str = [safe_str(u) for u in chunk_numeric_info.get("condition_base_units", [])]
                 if chunk_cls_data.get("HasRangeInCondition", False) and len(cond_vals_str) >= 2:
                      extracted_values_for_chunk.append( (cond_vals_str[0], cond_units_str[0]) ) # Cond Range Min
                      extracted_values_for_chunk.append( (cond_vals_str[1], cond_units_str[1]) ) # Cond Range Max
                 elif cond_vals_str: # Handle Single or Multiple conditions
                      for i in range(len(cond_vals_str)):
                          extracted_values_for_chunk.append( (cond_vals_str[i], cond_units_str[i] if i < len(cond_units_str) else None) )


                 # 5. Create coded rows for the chunk
                 if len(code_pairs) != len(extracted_values_for_chunk):
                      st.warning(f"Code/Value count mismatch for chunk '{chunk}'. Codes: {len(code_pairs)}, Values: {len(extracted_values_for_chunk)}. Codes:{code_pairs}, Vals:{extracted_values_for_chunk}")
                      # Add error row for chunk or try partial assignment
                      coded_rows.append({"Category": chunk_cat, "Attribute": "Error", "Code": f"{chunk_prefix}ERR-MATCH", "Value": "Code/Value mismatch"})
                      continue

                 for i, (val_code, unit_code) in enumerate(code_pairs):
                      val_str, unit_str = extracted_values_for_chunk[i]
                      # Add value row
                      coded_rows.append({"Category": chunk_cat, "Attribute": "Value",
                                         "Code": chunk_prefix + val_code, "Value": val_str or ""})
                      # Add unit row
                      coded_rows.append({"Category": chunk_cat, "Attribute": "Unit",
                                         "Code": chunk_prefix + unit_code, "Value": unit_str or ""})

             except Exception as e:
                 st.error(f"Error processing chunk '{chunk}' from '{value_str_clean}': {e}")
                 coded_rows.append({"Category": "Error in Chunk Processing", "Attribute": "Error", "Code": f"{chunk_prefix}ERR-PROC", "Value": str(e)})
    else:
        # Single top-level value (may have internal structure like range or condition)
        try:
            # Determine if main part is numeric only (use overall extraction data)
            main_vals = [v for v in extraction_data.get("_normalized_all_numeric",[]) if extraction_data.get("MainNumericValues","").__contains__(safe_str(v))] # Crude check which part val belongs to
            main_units_found = any(u and u != "None" for u in extraction_data.get("MainBaseUnits","").split(', '))
            is_main_num_only = bool(main_vals) and not main_units_found


            code_pairs = _get_codes_for_structure(
                classification_data.get("HasRangeInMain", False),
                classification_data.get("MainItemCount", 0),
                classification_data.get("HasRangeInCondition", False),
                classification_data.get("ConditionCount", 0),
                is_main_num_only
            )

            # Extract value/unit strings corresponding to codes
            extracted_values = []
             # Main Part
            main_vals_str = [v.strip() for v in extraction_data.get("MainNumericValues","").split(',') if v.strip()]
            main_units_str = [u.strip() for u in extraction_data.get("MainBaseUnits","").split(',') if u.strip() and u.strip()!='None']
            if classification_data.get("HasRangeInMain", False) and len(main_vals_str) >= 2:
                 extracted_values.append( (main_vals_str[0], main_units_str[0] if main_units_str else "") )
                 extracted_values.append( (main_vals_str[1], main_units_str[1] if len(main_units_str)>1 else (main_units_str[0] if main_units_str else ""))) # Handle unit consistency assumptions
            elif main_vals_str:
                 extracted_values.append( (main_vals_str[0], main_units_str[0] if main_units_str else "") )

             # Condition Part
            cond_vals_str = [v.strip() for v in extraction_data.get("ConditionNumericValues","").split(',') if v.strip()]
            cond_units_str = [u.strip() for u in extraction_data.get("ConditionBaseUnits","").split(',') if u.strip() and u.strip()!='None']
            if classification_data.get("HasRangeInCondition", False) and len(cond_vals_str) >= 2:
                  extracted_values.append( (cond_vals_str[0], cond_units_str[0] if cond_units_str else "") )
                  extracted_values.append( (cond_vals_str[1], cond_units_str[1] if len(cond_units_str)>1 else (cond_units_str[0] if cond_units_str else "")))
            elif cond_vals_str: # Single or Multiple conditions
                 num_cond_vals = len(cond_vals_str)
                 for i in range(num_cond_vals):
                     unit = cond_units_str[i] if i < len(cond_units_str) else (cond_units_str[0] if cond_units_str else "") # Reuse first unit if fewer units than vals?
                     extracted_values.append( (cond_vals_str[i], unit) )


            if len(code_pairs) != len(extracted_values):
                 st.warning(f"Code/Value count mismatch for '{value_str_clean}'. Codes: {len(code_pairs)}, Values: {len(extracted_values)}. Codes:{code_pairs}, Vals:{extracted_values}")
                 coded_rows.append({"Category": classification_name, "Attribute": "Error", "Code": "ERR-MATCH", "Value": "Code/Value mismatch"})
            else:
                for i, (val_code, unit_code) in enumerate(code_pairs):
                    val_str, unit_str = extracted_values[i]
                    coded_rows.append({"Category": classification_name, "Attribute": "Value", "Code": val_code, "Value": val_str or ""})
                    coded_rows.append({"Category": classification_name, "Attribute": "Unit", "Code": unit_code, "Value": unit_str or ""})

        except Exception as e:
             st.error(f"Error processing single value '{value_str_clean}': {e}")
             coded_rows.append({"Category": "Error in Value Processing", "Attribute": "Error", "Code": "ERR-PROC", "Value": str(e)})

    # Filter out empty value/unit rows if desired (e.g., from UNK codes or errors where val/unit was empty)
    # final_coded_rows = [r for r in coded_rows if r.get("Value")] # Optional cleanup
    return coded_rows # Return all generated rows, including potential empties/errors

# --- Main analysis function ---
def analyze_and_code_value(value_str: str, base_units: set, multipliers_dict: dict) -> (dict, list):
    """
    Performs ALL analysis and coding for a single value string.
    Returns:
        - analysis_dict: Dictionary containing detailed analysis results (wide format).
        - coded_rows_list: List of dictionaries representing coded rows (long format).
    """
    # 1. Clean input
    value_str_clean = fix_exceptions(value_str.strip()) if value_str else ""

    if not value_str_clean:
         # Handle empty input string explicitly
        analysis_dict = {"Original Value": "", "Classification": "Empty", "DetailedValueType": "Empty"}
        # Add all other expected analysis keys with default empty/null values
        analysis_dict.update({k:"--" for k in ["Identifiers", "SubValueCount", "ConditionCount", "HasRangeInMain",
                      "HasMultiValueInMain", "HasRangeInCondition", "HasMultipleConditions",
                      "MainItemCount", "MainNumericValues", "ConditionNumericValues", "MainMultipliers",
                      "ConditionMultipliers", "MainBaseUnits", "ConditionBaseUnits",
                      "NormalizedMainValues", "NormalizedConditionValues", "ParsingErrorFlag",
                      "MainUnits", "MainDistinctUnitCount", "MainUnitsConsistent", "ConditionUnits",
                      "ConditionDistinctUnitCount", "ConditionUnitsConsistent", "OverallUnitConsistency",
                      "SubValueUnitVariationSummary", "Normalized Unit", "Absolute Unit",
                      "MinNormalizedValue", "MaxNormalizedValue", "SingleUnitForAllSubs",
                      "AllDistinctUnitsUsed"]})
        analysis_dict.update({"SubValueCount":0, "ConditionCount":0, "MainItemCount":0,"ParsingErrorFlag":False,
                              "MainUnitsConsistent":True, "ConditionUnitsConsistent":True, "OverallUnitConsistency":True,
                              "SingleUnitForAllSubs":True}) #Sensible defaults
        coded_rows_list = []
        return analysis_dict, coded_rows_list

    # 2. Classify structure
    classification_data = _classify_value(value_str_clean)

    # 3. Extract numeric details, units, and perform string normalizations
    extraction_data = _extract_details(value_str_clean, base_units, multipliers_dict)

    # 4. Build the final analysis dictionary (wide format)
    analysis_dict = _build_analysis_dict(value_str, classification_data, extraction_data) # Pass original value_str

    # 5. Generate the coded rows (long format)
    coded_rows_list = _generate_coded_rows(value_str_clean, classification_data, extraction_data, base_units, multipliers_dict)

    return analysis_dict, coded_rows_list

#############################################
# 5) BUFFER/COMBINE step is now integrated into Streamlit UI
#############################################

# REMOVED combine_results function


#############################################
# 6) STREAMLIT APP UI
#############################################

st.title("ACC Project - Unified Pipeline")

# Initialize/Load Mapping File
# Uses session state to avoid reloading mapping from disk/GitHub on every interaction.
@st.cache_data  # Cache the mapping read from disk
def get_mapping_data(file_path="mapping.xlsx"):
    """Reads mapping data from local disk, raising error if file missing."""
    if not os.path.exists(file_path):
         # Attempt download if local file missing at the start of a session run
         st.warning("Local mapping.xlsx not found. Attempting download from GitHub...")
         downloaded_df = download_mapping_file_from_github()
         if downloaded_df is not None:
              save_mapping_to_disk(downloaded_df, file_path) # Save it for subsequent runs
              return read_mapping_file(file_path)
         else:
              st.error("Failed to download mapping.xlsx from GitHub. Cannot proceed.")
              st.stop()
              return None, None # Should stop before here
    else:
        return read_mapping_file(file_path) # Read existing local file

try:
    # This will use the cached result unless the file timestamp changes
    # It now directly returns base_units, multipliers_dict
    base_units, multipliers_dict = get_mapping_data()
    # Check required columns within read_mapping_file or here if needed again
    required_cols_map = {'Base Unit Symbol', 'Multiplier Symbol'} # Ensure read_mapping validated this
    if base_units is None:
         st.error("Base units could not be loaded from mapping file. Check file and GitHub settings.")
         st.stop()

    # Store the DataFrame in session state for 'Manage Units' IF needed, but primary use is base_units/multipliers
    if "mapping_df" not in st.session_state or st.session_state.get("mapping_df") is None:
        if os.path.exists("mapping.xlsx"):
            st.session_state["mapping_df"] = pd.read_excel("mapping.xlsx")
        else:
            st.session_state["mapping_df"] = None # Indicate it's not loaded for editing

except FileNotFoundError as e:
    st.error(f"{e}")
    st.info("Attempting to download from GitHub...")
    try:
         initial_df = download_mapping_file_from_github()
         if initial_df is not None:
              save_mapping_to_disk(initial_df, "mapping.xlsx")
              st.session_state["mapping_df"] = initial_df
              base_units, multipliers_dict = read_mapping_file("mapping.xlsx")
              st.success("Successfully downloaded and loaded mapping file. Refresh might be needed.")
              st.rerun()
         else:
              st.error("Download failed. Cannot proceed without mapping file.")
              st.stop()
    except Exception as dl_e:
        st.error(f"Error during initial mapping download/load: {dl_e}")
        st.stop()

except ValueError as e:
     st.error(f"Error loading mapping data: {e}")
     st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred initializing mapping data: {e}")
    import traceback
    st.error(traceback.format_exc())
    st.stop()


# --- Check if mapping essentials are loaded ---
mapping_loaded = base_units is not None and multipliers_dict is not None
if not mapping_loaded:
     st.error("Mapping essential data (base units, multipliers) failed to load. Cannot proceed.")
     st.stop()


# --- UI Selection ---
operation = st.selectbox("Select Operation", ["Get Pattern", "Manage Units"])

############################
# OPERATION: GET PATTERN
############################
if operation == "Get Pattern":

    st.header("Get Pattern")
    st.write("Upload an Excel file containing a 'Value' column for processing.")

    input_file = st.file_uploader("Upload Input Excel File", type=["xlsx"], key="pattern_uploader")

    if input_file:
        final_output_filename = f"processed_{input_file.name}" # Name based on input
        # Temp local mapping file name
        mapping_filename = "mapping.xlsx"

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.info("Reading input file...")
            progress_bar.progress(5)
            # Read all sheets if necessary, or specify sheet
            try:
                 # Simplification: Assume processing first sheet or requires specific sheet name 'Sheet1'
                 # Modify this if multi-sheet processing is needed. Read sheet by sheet inside the loop.
                 xls = pd.ExcelFile(input_file) # Keep input_file BytesIO accessible
                 input_sheets = xls.sheet_names
                 st.info(f"Found sheets: {input_sheets}. Processing sheet: '{input_sheets[0]}'") # Process first sheet
                 input_df_orig = pd.read_excel(xls, sheet_name=input_sheets[0])

            except Exception as read_e:
                st.error(f"Failed to read Excel file: {read_e}")
                st.stop()


            if 'Value' not in input_df_orig.columns:
                 st.error("Input file must contain a 'Value' column.")
                 st.stop()

            # Add original index for tracking
            input_df = input_df_orig.reset_index().rename(columns={'index': 'original_index'})

            # Prepare for processing loop
            all_analysis_results = []
            all_coded_rows = []

            total_rows = len(input_df)
            status_text.info(f"Starting processing for {total_rows} rows...")
            progress_bar.progress(10)

            # --- Single Processing Loop ---
            for i, row in input_df.iterrows():
                original_index = row['original_index']
                value_str = str(row.get('Value', '')).strip()

                try:
                    # Core analysis and coding step
                    analysis_dict, coded_rows_list = analyze_and_code_value(
                        value_str, base_units, multipliers_dict
                    )

                    # Add identifier for merging
                    analysis_dict['original_index'] = original_index
                    for code_row in coded_rows_list:
                         code_row['original_index'] = original_index

                    all_analysis_results.append(analysis_dict)
                    all_coded_rows.extend(coded_rows_list)

                except Exception as row_e:
                     st.error(f"Error processing row {original_index} (Value: '{value_str}'): {row_e}")
                     # Append error placeholder to analysis results
                     error_analysis = {"original_index": original_index, "Original Value": value_str, "Classification": "Row Processing Error", "DetailedValueType": str(row_e)}
                      # Add all other analysis keys with default empty/null values
                     error_analysis.update({k:"Error" for k in ["Identifiers", "SubValueCount", "ConditionCount", "HasRangeInMain",
                                       "HasMultiValueInMain", "HasRangeInCondition", "HasMultipleConditions",
                                       "MainItemCount", "MainNumericValues", "ConditionNumericValues", "MainMultipliers",
                                       "ConditionMultipliers", "MainBaseUnits", "ConditionBaseUnits",
                                       "NormalizedMainValues", "NormalizedConditionValues", "ParsingErrorFlag",
                                       "MainUnits", "MainDistinctUnitCount", "MainUnitsConsistent", "ConditionUnits",
                                       "ConditionDistinctUnitCount", "ConditionUnitsConsistent", "OverallUnitConsistency",
                                       "SubValueUnitVariationSummary", "Normalized Unit", "Absolute Unit",
                                       "MinNormalizedValue", "MaxNormalizedValue", "SingleUnitForAllSubs",
                                       "AllDistinctUnitsUsed"]})
                     all_analysis_results.append(error_analysis)
                     # Optionally add an error row to coded results too?
                     # all_coded_rows.append({"original_index": original_index, "Category": "Row Error", "Attribute": "Error", "Code": "ERR-ROW", "Value": str(row_e)})


                # Update progress
                progress_percent = 10 + int(80 * (i + 1) / total_rows)
                progress_bar.progress(progress_percent)

            status_text.info("Processing complete. Finalizing output...")

            # --- Create DataFrames ---
            analysis_df = pd.DataFrame(all_analysis_results)
            coded_df = pd.DataFrame(all_coded_rows)

            # Ensure original_index exists for merging if rows were processed
            if 'original_index' not in analysis_df.columns and not analysis_df.empty:
                 st.error("Analysis DataFrame missing 'original_index'. Merge will likely fail.")
                 # Add it back if possible or debug why it was lost
                 analysis_df= analysis_df.reset_index().rename(columns={'index': 'original_index'})


            if 'original_index' not in coded_df.columns and not coded_df.empty:
                  st.warning("Coded DataFrame missing 'original_index'. Merging this part may fail.")


            # --- Merge Results ---
            # Merge coded rows with original data (using original_df which includes original_index)
            original_data_to_merge = input_df # Contains 'Value' and original_index + other cols
            df_merged = None

            if not coded_df.empty:
                 if 'original_index' in coded_df.columns and 'original_index' in original_data_to_merge.columns:
                     # Merge coded rows onto original data first
                     df_merged = original_data_to_merge.merge(coded_df, on='original_index', how='left')
                     # Add suffix to 'Value' from coded_df if needed (e.g., '_coded') if original still has 'Value'
                 else:
                     st.warning("Cannot merge coded data due to missing 'original_index'. Skipping coded data merge.")
                     df_merged = original_data_to_merge.copy() # Start with original data
            else:
                st.info("No coded rows were generated. Output will contain original data and analysis columns.")
                df_merged = original_data_to_merge.copy() # Start with original data


            # Merge analysis columns onto the result
            if not analysis_df.empty:
                if 'original_index' in analysis_df.columns and 'original_index' in df_merged.columns:
                    # Define analysis columns to bring in, excluding the merge key if it's redundant post-merge
                    analysis_cols = [col for col in analysis_df.columns if col != 'original_index']
                    df_merged = df_merged.merge(analysis_df[analysis_cols + ['original_index']], on='original_index', how='left', suffixes=('', '_analysis')) # Add suffix for potential conflicts like 'Classification'
                    # Handle potential duplicated columns (like Classification vs Classification_analysis) - choose one or rename
                    if 'Classification_analysis' in df_merged.columns:
                        # Example: Prefer analysis one, remove original from processing step if it existed in coded_df merge
                         if 'Classification' in df_merged.columns: df_merged = df_merged.drop('Classification', axis=1)
                         df_merged = df_merged.rename(columns={'Classification_analysis': 'Classification'})
                    if 'Category' not in df_merged.columns and 'Category' in coded_df.columns: # Check if category from coded_df needs bringing if dropped in suffix handling
                        # If needed re-merge specific column if it got dropped
                         pass


                else:
                    st.warning("Cannot merge analysis data due to missing 'original_index'. Skipping analysis data merge.")
                    # df_merged remains as is (original + possibly coded)
            else:
                 st.info("No analysis results were generated.")
                 # df_merged remains as is


            # Drop the merge key if no longer needed
            if 'original_index' in df_merged.columns:
                df_merged = df_merged.drop(columns=['original_index'])

            # --- Final Column Ordering ---
            # Start with preferred order, include columns dynamically present in df_merged
            base_order = ["Value"] # Start with original value
             # Add coded output columns if they exist
            if "Category" in df_merged.columns: base_order.extend(["Category", "Attribute", "Code", "Value_coded" if "Value_coded" in df_merged.columns else "Value"]) # Needs careful handling of Value cols
             # Rename coded value if needed: df_merged.rename(columns={'Value': 'Value_Coded'}, inplace=True) # Need conditional rename logic
             # Drop original Value if coded Value exists and is preferred in output?

            analysis_order = [
                "Classification", "DetailedValueType", "Normalized Unit", "Absolute Unit", "Identifiers",
                # ... add all other desired analysis columns ...
                "SubValueCount", "ConditionCount",
                "HasRangeInMain", "HasMultiValueInMain", "HasRangeInCondition", "HasMultipleConditions",
                "MainUnits", "MainDistinctUnitCount", "MainUnitsConsistent",
                "ConditionUnits", "ConditionDistinctUnitCount", "ConditionUnitsConsistent",
                "OverallUnitConsistency", "ParsingErrorFlag", "SubValueUnitVariationSummary",
                "MainNumericValues", "ConditionNumericValues", "MainMultipliers", "ConditionMultipliers",
                "MainBaseUnits", "ConditionBaseUnits", "NormalizedMainValues", "NormalizedConditionValues",
                "MinNormalizedValue", "MaxNormalizedValue", "SingleUnitForAllSubs", "AllDistinctUnitsUsed"
            ]

            # Get remaining original columns (excluding 'Value' if handled)
            original_cols = [col for col in input_df_orig.columns if col not in base_order and col not in analysis_order and col in df_merged.columns and col !='Value']

            # Combine and filter for existing columns
            final_ordered_cols = []
            potential_cols = base_order + original_cols + analysis_order
            for col in potential_cols:
                 if col in df_merged.columns and col not in final_ordered_cols:
                       final_ordered_cols.append(col)

            # Add any other columns missed (shouldn't happen if base/orig/analysis cover all)
            missed_cols = [col for col in df_merged.columns if col not in final_ordered_cols]
            final_ordered_cols.extend(missed_cols)


            st.write(f"DEBUG: Final column order attempt: {final_ordered_cols}")
            try:
                df_final = df_merged[final_ordered_cols]
            except KeyError as key_err:
                 st.error(f"Column ordering error: {key_err}. Falling back to unordered.")
                 df_final = df_merged # Use original merge order as fallback


            progress_bar.progress(95)

            # --- Save and Offer Download ---
            output_buffer = BytesIO()
            df_final.to_excel(output_buffer, index=False, engine='openpyxl')
            output_buffer.seek(0)

            progress_bar.progress(100)
            status_text.success("Processing Complete!")

            st.download_button(
                label=f"Download Results ({final_output_filename})",
                data=output_buffer,
                file_name=final_output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except FileNotFoundError as e: # Catch mapping file not found specifically if retry fails
             status_text.error(f"Pipeline Error: {e}. Ensure mapping.xlsx exists or can be downloaded.")
        except ValueError as e: # Catch mapping file format/content errors
             status_text.error(f"Pipeline Error: {e}. Check mapping.xlsx content/format.")
        except Exception as e:
            status_text.error(f"An unexpected error occurred during the 'Get Pattern' process: {e}")
            import traceback
            st.error(traceback.format_exc()) # Show detailed error in Streamlit logs/UI


############################
# OPERATION: MANAGE UNITS (Modified to use data editor)
############################
elif operation == "Manage Units":
    st.header("Manage Units (GitHub mapping file)")
    st.write("Edit the table below to add, remove, or modify base units. Click 'Save to GitHub' to persist changes.")
    st.warning("Directly editing 'Multiplier Symbol' in this table has no effect on the processing logic (multipliers are hardcoded). It only affects the content of mapping.xlsx.")


    # Ensure mapping DataFrame is loaded into session state for editing
    if "mapping_df" not in st.session_state or not isinstance(st.session_state.get("mapping_df"), pd.DataFrame):
         st.warning("Mapping data not loaded in session state for editing. Attempting to load from disk...")
         if os.path.exists("mapping.xlsx"):
             try:
                 st.session_state["mapping_df"] = pd.read_excel("mapping.xlsx")
                 st.info("Loaded mapping data from disk into editor.")
             except Exception as e:
                  st.error(f"Failed to load mapping.xlsx for editing: {e}")
                  st.session_state["mapping_df"] = None # Ensure it's None on error
         else:
              st.error("mapping.xlsx not found locally. Cannot edit units. Try running 'Get Pattern' once or check GitHub download.")
              st.session_state["mapping_df"] = None


    if isinstance(st.session_state.get("mapping_df"), pd.DataFrame):
        required_cols = {"Base Unit Symbol", "Multiplier Symbol"} # Ensure columns exist for editor

        if not required_cols.issubset(st.session_state["mapping_df"].columns):
             st.error(f"Mapping DataFrame is missing required columns: {required_cols}. Cannot display editor.")
        else:
            # --- Data Editor for Modifications ---
            st.subheader("Edit Mapping Data")
            edited_df = st.data_editor(
                st.session_state["mapping_df"],
                key="unit_editor",
                num_rows="dynamic", # Allow adding/deleting rows
                 # column_config={ # Optional: Configure specific columns if needed
                 #     "Base Unit Symbol": st.column_config.TextColumn(required=True),
                 # }
                disabled=["Multiplier Symbol"] # Example: Disable editing multiplier column
            )

            # --- Validation (Optional but recommended) ---
            validation_ok = True
            if edited_df is not None:
                # Check for duplicates in Base Unit Symbol
                 base_units_edited = edited_df["Base Unit Symbol"].dropna().astype(str)
                 if base_units_edited.duplicated().any():
                      st.error("Duplicate 'Base Unit Symbol' entries found. Please ensure all symbols are unique before saving.")
                      validation_ok = False
                 # Check for empty Base Unit Symbol
                 if (base_units_edited == "").any():
                      st.error("'Base Unit Symbol' cannot be empty. Please fill in or remove empty rows.")
                      validation_ok = False


            # --- Persist Changes ---
            st.subheader("Persist Changes")

             # Provide button to update session state based on editor (can be skipped if saving directly)
            # if st.button("Update Session State from Editor"):
            #     if edited_df is not None and validation_ok:
            #          st.session_state["mapping_df"] = edited_df.copy() # Update session state
            #          st.success("Session state updated with changes from editor.")
            #          # Optional: rerun to reflect changes if other parts depend on session state directly
            #          # st.rerun()
            #     elif not validation_ok:
            #          st.error("Validation failed. Cannot update session state.")
            #     else:
            #          st.warning("No changes detected in the editor.")


            # Option to download updated file locally from editor state
            if edited_df is not None:
                try:
                    buffer = BytesIO()
                    edited_df.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                    st.download_button(
                        label="Download Edited Mapping (Local Only)",
                        data=buffer,
                        file_name="edited_mapping.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                         disabled=not validation_ok
                    )
                except Exception as e:
                    st.error(f"Error creating download link for edited mapping: {e}")


            # Save changes back to GitHub
            if st.button("Save Edited Mapping to GitHub"):
                if edited_df is not None and validation_ok:
                    st.info("Attempting to save changes to GitHub...")
                     # Use the state directly from the editor ('edited_df') for upload
                    success = update_mapping_file_on_github(edited_df)
                    if success:
                        st.success("Mapping file updated on GitHub! The app will use the new version on next refresh/run.")
                         # Update session state to match what was uploaded
                        st.session_state["mapping_df"] = edited_df.copy()
                         # Clear cache for get_mapping_data to force re-read from disk next time
                        get_mapping_data.clear()
                        st.info("Cache cleared. Rerun required to reflect saved changes fully in processing.")
                        st.rerun() # Rerun to potentially reflect state change in editor display immediately

                    else:
                        st.error("Failed to update mapping file on GitHub. Check logs and permissions.")
                elif not validation_ok:
                     st.error("Validation failed. Cannot save to GitHub.")

                else: # Should not happen if button is clicked
                    st.error("Edited data is not available to save.")
    else:
        st.warning("Mapping data could not be loaded for editing.")
