# STREAMLIT SCRIPT WITH NEW PIPELINE LOGIC
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
# 1) GITHUB HELPER FUNCTIONS
#############################################

# --- download_mapping_file_from_github() --- (No changes needed)
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
        with open(local_file, "wb") as f:
            f.write(decoded_bytes)

        # Now parse the local file into a DataFrame
        try:
            df = pd.read_excel(local_file)
        except Exception as e:
            st.error(f"Failed to parse downloaded mapping file: {e}")
            st.stop()

        # os.remove(local_file) # Keep local file temporarily for pipeline steps
        st.write("DEBUG: Download successful. mapping_df shape:", df.shape)
        return df
    else:
        st.error(f"Failed to download file from GitHub: {response.status_code} {response.text}")
        st.stop()

# --- update_mapping_file_on_github() --- (No changes needed)
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

    # 1) Save DF to local file
    temp_file = "temp_mapping_upload.xlsx" # Use a different name to avoid conflict
    mapping_df.to_excel(temp_file, index=False, engine='openpyxl')

    # 2) Encode local file in base64
    with open(temp_file, "rb") as f:
        content_bytes = f.read()
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
        os.remove(temp_file) # Clean up local temp file
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
    os.remove(temp_file)  # clean up local file

    if update_response.status_code in [200, 201]:
        st.write("DEBUG: Update/creation successful:", update_response.status_code)
        # Force re-download next time or update session state
        if "mapping_df" in st.session_state:
            del st.session_state["mapping_df"]
        return True
    else:
        st.error(f"Failed to update file on GitHub: {update_response.status_code} {update_response.text}")
        return False


#############################################
# 2) GLOBAL MULTIPLIER MAPPING + READ MAPPING
#############################################

MULTIPLIER_MAPPING = {
    'y': 1e-24, 'z': 1e-21, 'a': 1e-18, 'f': 1e-15,
    'p': 1e-12, 'n': 1e-9,  'µ': 1e-6,  'm': 1e-3,
    'c': 1e-2,  'd': 1e-1,  'da': 1e1,  'h': 1e2,
    'k': 1e3,   'M': 1e6,   'G': 1e9,   'T': 1e12,
    'P': 1e15,  'E': 1e18,  'Z': 1e21,  'Y': 1e24
}

LOCAL_BASE_UNITS = set()  # If you want any built-ins, put them here.

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
        st.stop()


def read_mapping_file(mapping_file_path: str):
    """
    Reads the local 'mapping.xlsx', checks for required columns and unknown multipliers.
    """
    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError(f"Error: '{mapping_file_path}' not found.")
    try:
        df = pd.read_excel(mapping_file_path)
    except Exception as e:
        raise ValueError(f"Error reading mapping file '{mapping_file_path}': {e}")

    required_cols = {'Base Unit Symbol', 'Multiplier Symbol'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"'{mapping_file_path}' must have columns: {required_cols}")

    base_units = set(str(x).strip() for x in df['Base Unit Symbol'].dropna().unique())
    file_multipliers = set(str(x).strip() for x in df['Multiplier Symbol'].dropna().unique() if pd.notna(x)) # Ensure Multiplier Symbol exists

    # Combine hardcoded multipliers with those from the file (if any)
    # Note: MULTIPLIER_MAPPING is currently hardcoded, file 'Multiplier Symbol' is not used for factors
    combined_multiplier_keys = set(MULTIPLIER_MAPPING.keys())

    # Check if multipliers mentioned in the file are defined in our hardcoded map
    undefined = file_multipliers - combined_multiplier_keys
    if undefined:
        # Allow multipliers to exist in the sheet even if not in MULTIPLIER_MAPPING for now
        # raise ValueError(f"Undefined multipliers in '{mapping_file_path}': {undefined}")
        st.warning(f"Note: Multipliers found in mapping file but not in hardcoded MULTIPLIER_MAPPING: {undefined}. They won't be used for normalization.")

    st.write(f"DEBUG: Read mapping. Base units found: {len(base_units)}. Multipliers known: {len(combined_multiplier_keys)}")
    return base_units, MULTIPLIER_MAPPING


#############################################
# 3) FIRST PIPELINE: FIXED PROCESSING
#############################################

# REMOVED PATTERNS_12 - No longer needed as we use detailed classification

# REMOVED detect_value_type - No longer needed

# --- Utility functions for Pipeline 1 (Fixed Processing) ---
# (Keep extract_numeric_and_unit_analysis, extract_numeric_info, safe_str,
#  extract_numeric_info_for_value, analyze_unit_part, analyze_value_units,
#  process_unit_token_no_paren, process_unit_token, split_outside_parens,
#  resolve_compound_unit, count_main_items, count_conditions,
#  classify_condition, classify_main, classify_sub_value,
#  remove_parentheses_detailed, extract_identifiers_detailed,
#  classify_value_type_detailed, fix_exceptions,
#  extract_block_texts, parse_value_unit_identifier, generate_mapping,
#  fill_mapping_for_part, get_code_prefixes_for_category, display_mapping,
#  get_desired_order )
# --- These functions are mostly kept as is, but `process_single_key` is modified ---

# ========== Re-adding necessary helper functions from section 3 & 4 ==========
# (Ensure these are defined before they are called)

def extract_numeric_and_unit_analysis(token, base_units, multipliers_dict):
    token = token.strip()
    # Regex to capture optional sign, numeric part, and the rest (unit, etc.)
    # Allows scientific notation like 1.23e-4
    pattern = re.compile(r'^(?P<numeric>[+\-±]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)(?P<rest>.*)$')
    m = pattern.match(token)
    if not m:
        # Check if it's just a unit or something non-numeric starting
        if any(token.startswith(p) for p in multipliers_dict.keys()) or token in base_units:
             return None, None, token, None, False # Treat as unit only
        return None, None, None, None, True # True error

    numeric_str = m.group("numeric")
    rest = m.group("rest").strip()

    try:
        # Use float for potentially scientific notation
        numeric_val = float(numeric_str.replace('±','')) # Handle potential ± for conversion
    except ValueError:
        numeric_val = None

    if numeric_val is None:
        return None, None, None, None, True # Error in numeric conversion

    # Case 1: Only a number was found
    if not rest:
        return numeric_val, "1", None, numeric_val, False # No unit, multiplier is 1

    # Case 2: Number followed by something - try to parse unit and multiplier
    multiplier_symbol = ""
    base_unit = rest # Start assuming the whole 'rest' is the base unit
    multiplier_factor = 1.0
    sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
    found_prefix = False

    # Try matching known prefixes
    for prefix in sorted_prefixes:
        if rest.startswith(prefix):
            possible_base = rest[len(prefix):].strip() # Potential base unit after prefix
            # Check if the remaining part is a known base unit OR if it's empty (e.g., "10k" implies "10 k")
            if possible_base in base_units : #or not possible_base: # Allow empty base if needed? No, require base unit.
                # Ensure we don't misinterpret a base unit as a prefix (e.g., 'm' in 'meter')
                # This check helps if a base unit symbol itself is a prefix (less common)
                if possible_base in base_units: # Redundant check, but clearer
                    multiplier_symbol = prefix
                    base_unit = possible_base
                    multiplier_factor = multipliers_dict[prefix]
                    found_prefix = True
                    break
                # If possible_base was empty, maybe handle it? For now, require a base unit.

    # If no prefix was matched, check if the entire 'rest' is a base unit
    if not found_prefix:
        if rest in base_units:
            base_unit = rest
            # No prefix symbol, multiplier remains 1.0
            multiplier_symbol = "1" # Explicitly '1' for no prefix
        else:
            # The 'rest' part is not a known base unit and didn't start with a known prefix
            # It might be an invalid unit or just text
            # Decide how to handle: error or assume no unit? Let's flag as error.
             st.write(f"DEBUG: Unknown unit part '{rest}' in token '{token}'")
             return numeric_val, None, None, None, True # Error: Unrecognized unit part

    # Calculate normalized value if possible
    normalized_value = None
    if numeric_val is not None and multiplier_factor is not None:
         try:
            normalized_value = numeric_val * multiplier_factor
         except TypeError:
             st.warning(f"DEBUG: Type error during normalization for {numeric_val} * {multiplier_factor}")
             normalized_value = None # Or handle differently

    # Return results: original numeric, multiplier symbol found, base unit found, normalized value, error flag
    return numeric_val, multiplier_symbol if multiplier_symbol else "1", base_unit, normalized_value, False

def remove_parentheses_detailed(text: str) -> str:
    return re.sub(r'\([^)]*\)', '', text)

def extract_numeric_info(part_text, base_units, multipliers_dict):
    # First remove content within parentheses for analysis
    text = remove_parentheses_detailed(part_text).strip()

    if not text:
        return {
            "numeric_values": [], "multipliers": [],
            "base_units": [], "normalized_values": [],
            "error_flags": [], "type": "none"
        }

    # Determine structure: range, multiple values, or single
    # Use split_outside_parens to handle delimiters correctly if needed, but here we use simple string checks after removing parens
    if " to " in text:
        # Need robust splitting around ' to ' respecting spaces
        tokens = [t.strip() for t in re.split(r'\s+to\s+', text) if t.strip()]
        info_type = "range"
    elif "," in text:
         # Split by comma, respecting potential spaces around it
        tokens = [t.strip() for t in text.split(",") if t.strip()]
        info_type = "multiple"
    else:
        tokens = [text]
        info_type = "single"

    # Initialize lists to store results for each token
    numeric_values = []
    multipliers = []
    base_units_list = []
    normalized_values = []
    error_flags = []

    # Process each token found
    for token in tokens:
        num_val, multiplier_symbol, base_unit, norm_val, err_flag = extract_numeric_and_unit_analysis(token, base_units, multipliers_dict)

        numeric_values.append(num_val)
        # Use "1" if no multiplier symbol was identified but parsing was ok
        multipliers.append(multiplier_symbol if multiplier_symbol else ("1" if not err_flag and num_val is not None else "None"))
         # Base unit might be None if only number, or if error
        base_units_list.append(base_unit if base_unit else "None")
        normalized_values.append(norm_val)
        error_flags.append(err_flag)

    # Return dictionary of results
    return {
        "numeric_values": numeric_values,
        "multipliers": multipliers,
        "base_units": base_units_list,
        "normalized_values": normalized_values,
        "error_flags": error_flags,
        "type": info_type # Indicates if it was parsed as 'single', 'range', or 'multiple' parts
    }

def safe_str(item, placeholder="None"):
    return str(item) if item is not None else placeholder

def extract_numeric_info_for_value(raw_value, base_units, multipliers_dict):
    # Handle potential multiple sub-values separated by commas *outside* of conditions
    # Example: "10A @ 5V, 20A @ 5V" should be treated as two distinct sub-values
    # Example: "10A @ 5V, 6V" should NOT be split here if the comma is part of the condition

    # Let's assume classify_value_type_detailed correctly identifies the structure.
    # This function should process ONE logical value string which might contain '@' internally.
    # If the input `raw_value` truly represents multiple independent values (like "10A, 20A"),
    # the calling function (`detailed_analysis_pipeline`) should handle splitting first.
    # Here, we focus on parsing a single potentially complex value string.

    # We simplify: Assume `raw_value` is ONE logical entry. Split by '@' if present.
    raw_value = str(raw_value).strip() # Ensure string type
    main_part = raw_value
    cond_part = ""

    if "@" in raw_value:
        # Split only on the first '@'
        parts = raw_value.split('@', 1)
        main_part = parts[0].strip()
        if len(parts) > 1:
            cond_part = parts[1].strip()

    # Process the main part and the condition part separately
    main_info = extract_numeric_info(main_part, base_units, multipliers_dict)
    cond_info = extract_numeric_info(cond_part, base_units, multipliers_dict) # Will return empty lists if cond_part is ""

    # Combine results. Note these lists contain results for *all* tokens within main/condition parts
    # e.g., for "10 to 20A @ 5V", main_info will have two entries, cond_info one entry.
    return {
        "main_numeric": main_info["numeric_values"],
        "main_multipliers": main_info["multipliers"],
        "main_base_units": main_info["base_units"],
        "normalized_main": main_info["normalized_values"],
        "main_errors": main_info["error_flags"],
        "condition_numeric": cond_info["numeric_values"],
        "condition_multipliers": cond_info["multipliers"],
        "condition_base_units": cond_info["base_units"],
        "normalized_condition": cond_info["normalized_values"],
        "condition_errors": cond_info["error_flags"],
    }

# --- change1 seems to have been incorporated into process_unit_token_no_paren ---
def process_unit_token_no_paren(token, base_units, multipliers_dict):
    token = token.strip()

    # Handle currency symbol ($) explicitly if needed, otherwise treat like other units.
    # Let's assume '$' is NOT a base unit unless added to mapping.xlsx.
    # If it needs special handling (like always having a space), add logic here.
    # Example: if token.startswith('$'): ... return "$ " + base_unit ...

    # Direct match for base unit
    if token in base_units:
        return token # Return the base unit itself

    # Check for prefix + base unit
    sorted_prefixes = sorted(multipliers_dict.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if token.startswith(prefix):
            possible_base = token[len(prefix):].strip()
            if possible_base in base_units:
                # We found a valid prefix and base unit combination.
                # The goal is often to return the *base unit* for consistency.
                return possible_base # Return the identified base unit

    # If no match, it's an unknown unit or just text without a unit
    # Return None or a specific marker? Let's return None.
    # st.write(f"DEBUG: Unit part '{token}' not resolved to a base unit.")
    return None # Indicate no base unit could be resolved


def analyze_unit_part(part_text, base_units, multipliers_dict):
    # Remove parentheses content first
    text = remove_parentheses_detailed(part_text).strip()
    if not text:
        return {
            "units": [], "distinct_units": set(), # Use set for distinct
            "is_consistent": True, "count": 0,
            "type": "none"
        }

    # Split based on structure (range, multiple, single)
    # Use same logic as extract_numeric_info for splitting
    if " to " in text:
        tokens = [t.strip() for t in re.split(r'\s+to\s+', text) if t.strip()]
        part_type = "range"
    elif "," in text:
        tokens = [t.strip() for t in text.split(",") if t.strip()]
        part_type = "multiple"
    else:
        tokens = [text]
        part_type = "single"

    units = []
    for token in tokens:
        # Need to extract only the unit part from the token (e.g., "10kOhm" -> "Ohm")
        # We can reuse extract_numeric_and_unit_analysis or just focus on the unit part
        _, _, base_unit, _, err_flag = extract_numeric_and_unit_analysis(token, base_units, multipliers_dict)

        if not err_flag and base_unit:
            units.append(base_unit)
        elif token in base_units: # Handle cases like just "V"
             units.append(token)
        else:
            units.append("None") # Or handle error cases differently

    distinct_units = set(u for u in units if u != "None")
    is_consistent = (len(distinct_units) <= 1)
    count = len(units)

    return {
        "units": units, # List of resolved base units for each token
        "distinct_units": distinct_units, # Set of unique base units found
        "is_consistent": is_consistent, # True if zero or one distinct base unit
        "count": count, # Number of tokens processed
        "type": part_type # How the part was structured
    }


def analyze_value_units(raw_value, base_units, multipliers_dict):
    # Similar structure to extract_numeric_info_for_value
    raw_value = str(raw_value).strip()
    main_part = raw_value
    cond_part = ""

    if "@" in raw_value:
        parts = raw_value.split('@', 1)
        main_part = parts[0].strip()
        if len(parts) > 1:
            cond_part = parts[1].strip()

    # Analyze units in main and condition parts
    main_analysis = analyze_unit_part(main_part, base_units, multipliers_dict)
    cond_analysis = analyze_unit_part(cond_part, base_units, multipliers_dict)

    # Combine unit information
    all_main_units = main_analysis["units"]
    all_condition_units = cond_analysis["units"]

    main_distinct = main_analysis["distinct_units"]
    condition_distinct = cond_analysis["distinct_units"]

    main_consistent = main_analysis["is_consistent"]
    condition_consistent = cond_analysis["is_consistent"]

    # Overall consistency considers both parts
    all_distinct_units = main_distinct.union(condition_distinct)
    overall_consistent = (len(all_distinct_units) <= 1)


    return {
        "main_units": all_main_units,
        "main_distinct_units": main_distinct,
        "main_units_consistent": main_consistent,
        "main_unit_count": main_analysis["count"],
        # Keep sub_analysis if needed for detailed debugging, maybe as string
        "main_sub_analysis": str(main_analysis),

        "condition_units": all_condition_units,
        "condition_distinct_units": condition_distinct,
        "condition_units_consistent": condition_consistent,
        "condition_unit_count": cond_analysis["count"],
        "condition_sub_analysis": str(cond_analysis),

        "all_distinct_units": all_distinct_units, # Set of all unique units found
        "overall_consistent": overall_consistent # True if <= 1 unique unit across main and condition
    }


def process_unit_token(token, base_units, multipliers_dict):
    """
    Processes a token potentially containing value, unit, and parentheses,
    returning a standardized representation focusing on the unit.
    This seems complex and possibly overlaps with analyze_unit_part.
    Let's simplify or clarify its purpose.

    If the goal is just to get the base unit, `process_unit_token_no_paren`
    or `extract_numeric_and_unit_analysis` might be sufficient.

    If the goal is formatting (like adding space for Ohm), let's refine.
    Assume the primary goal is base unit identification for now.
    """
    # Let's reuse the robust extraction logic:
    _, _, base_unit, _, err_flag = extract_numeric_and_unit_analysis(token, base_units, multipliers_dict)

    if not err_flag and base_unit:
        # Add special formatting if needed
        # if base_unit.lower() == "ohm":
        #    return f"$ {base_unit}" # Example formatting
        return base_unit # Return the resolved base unit
    elif token in base_units:
         return token # It was just a base unit
    else:
        # Handle errors or unknowns
        return f"Error: Unit unresolved in '{token}'"


def split_outside_parens(text, delimiters):
    """Splits text by delimiters, ignoring delimiters inside parentheses."""
    tokens = []
    current = ""
    i = 0
    depth = 0
    # Ensure delimiters are handled correctly (e.g., regex might be better for complex cases)
    # Using string searching here. Sort delimiters by length descending to match longest first.
    sorted_delims = sorted(delimiters, key=len, reverse=True)

    while i < len(text):
        char = text[i]

        if char == '(':
            depth += 1
            current += char
            i += 1
        elif char == ')':
            # Ensure depth doesn't go below zero
            depth = max(0, depth - 1)
            current += char
            i += 1
        elif depth == 0:
            # Check if any delimiter matches at the current position
            matched_delim = None
            for delim in sorted_delims:
                # Ensure we don't go past the end of the string
                if i + len(delim) <= len(text) and text[i : i + len(delim)] == delim:
                     # Add whitespace checks if delimiters require surrounding spaces (e.g., ' to ')
                     # Simple example: check for space before 'to' if delim is ' to '
                     # is_standalone = (i == 0 or text[i-1].isspace()) and \
                     #                (i + len(delim) == len(text) or text[i + len(delim)].isspace())
                     # For now, assume exact match is sufficient
                    matched_delim = delim
                    break

            if matched_delim:
                # Found a delimiter outside parentheses
                if current.strip(): # Add the part before the delimiter if not empty
                    tokens.append(current.strip())
                # tokens.append(matched_delim) # Optionally add the delimiter itself as a token
                current = "" # Reset current part
                i += len(matched_delim) # Move index past the delimiter
            else:
                # Not a delimiter, add char to current part
                current += char
                i += 1
        else:
            # Inside parentheses, just add the char
            current += char
            i += 1

    # Add the last part if any
    if current.strip():
        tokens.append(current.strip())

    # Filter out empty strings that might result from splitting
    return [token for token in tokens if token]


def resolve_compound_unit(normalized_unit_string, base_units, multipliers_dict):
    """
    Takes a string like "$V @ $A to $A" and resolves the base units.
    This function's original purpose (using $) seems tied to the normalization step.
    Let's adapt it to work on the *original* value string to identify the structure
    and resolve units within each part.

    Revised Purpose: Take a raw value string, parse its structure,
    and return a string representing the resolved *base units* in that structure.
    Example: "10V @ 5mA to 10mA" -> "V @ A to A"
    """
    raw_value = str(normalized_unit_string).strip() # Input is the raw value string
    resolved_parts = []

    # Split by '@' first, keeping the delimiter conceptually
    if "@" in raw_value:
        parts = raw_value.split('@', 1)
        main_part_str = parts[0].strip()
        cond_part_str = parts[1].strip() if len(parts) > 1 else ""

        # Resolve units in the main part
        main_analysis = analyze_unit_part(main_part_str, base_units, multipliers_dict)
        # Represent main part units (handle range/multi within main)
        if main_analysis["type"] == "range":
             # Assuming consistency check passed or we take the first distinct unit
             main_unit_repr = " to ".join(main_analysis["distinct_units"]) if main_analysis["distinct_units"] else "None"
        elif main_analysis["type"] == "multiple":
             main_unit_repr = ", ".join(main_analysis["distinct_units"]) if main_analysis["distinct_units"] else "None"
        else: # single or none
             main_unit_repr = next(iter(main_analysis["distinct_units"]), "None")
        resolved_parts.append(main_unit_repr)

        # Add the "@" separator
        resolved_parts.append("@")

        # Resolve units in the condition part
        cond_analysis = analyze_unit_part(cond_part_str, base_units, multipliers_dict)
        # Represent condition part units
        if cond_analysis["type"] == "range":
             cond_unit_repr = " to ".join(cond_analysis["distinct_units"]) if cond_analysis["distinct_units"] else "None"
        elif cond_analysis["type"] == "multiple":
             cond_unit_repr = ", ".join(cond_analysis["distinct_units"]) if cond_analysis["distinct_units"] else "None"
        else: # single or none
             cond_unit_repr = next(iter(cond_analysis["distinct_units"]), "None")
        resolved_parts.append(cond_unit_repr)

    else:
        # No '@', process the whole string as the main part
        main_analysis = analyze_unit_part(raw_value, base_units, multipliers_dict)
        if main_analysis["type"] == "range":
             main_unit_repr = " to ".join(main_analysis["distinct_units"]) if main_analysis["distinct_units"] else "None"
        elif main_analysis["type"] == "multiple":
             main_unit_repr = ", ".join(main_analysis["distinct_units"]) if main_analysis["distinct_units"] else "None"
        else: # single or none
             main_unit_repr = next(iter(main_analysis["distinct_units"]), "None")
        resolved_parts.append(main_unit_repr)

    # Join the parts with spaces, filtering out "None" unless it's the only part
    final_repr = " ".join(p for p in resolved_parts if p != "None")
    if not final_repr and "None" in resolved_parts:
         return "None" # If everything was None, return "None"
    elif not final_repr:
         return "" # Handle empty case

    return final_repr.replace(" @ ", "@") # Condense space around @ if desired


def count_main_items(main_str: str) -> int:
    main_str = remove_parentheses_detailed(main_str).strip()
    if not main_str:
        return 0
    # Use robust splitting for range 'to'
    if " to " in main_str:
        # Check if it's a range value like "10 to 20 V" -> counts as 1 structured item
        # Or if it's multiple items like "10, 20 to 30" -> needs careful split
        # Let's use split_outside_parens for comma first, then check for 'to' within parts
        parts = split_outside_parens(main_str, [','])
        count = 0
        for part in parts:
             # Crude check for range: if ' to ' exists, count as one item representing the range
             if re.search(r'\s+to\s+', part):
                 count += 1
             else:
                 count += 1 # Count non-range parts as one item
        return count if count > 0 else (1 if main_str else 0) # Ensure at least 1 if string is non-empty

    elif "," in main_str:
        # Count comma-separated items
        return len(split_outside_parens(main_str, [',']))
    else:
        # Single item (could be number or value)
        return 1

def count_conditions(cond_str: str) -> int:
    cond_str = remove_parentheses_detailed(cond_str).strip()
    if not cond_str:
        return 0
    # Conditions are typically comma-separated or a single range
    if " to " in cond_str:
         # Does "5V to 10V" count as 1 condition (a range) or 2? Assume 1 range condition.
         # Does "5V, 10V to 15V" count as 2 conditions?
         # Let's split by comma first.
         parts = split_outside_parens(cond_str, [','])
         return len(parts) # Count number of comma-separated condition clauses
    elif "," in cond_str:
         return len(split_outside_parens(cond_str, [',']))
    else:
         # Single condition (value or range)
         return 1 if cond_str else 0


def classify_condition(cond_str: str) -> str:
    cond_str = remove_parentheses_detailed(cond_str).strip()
    if not cond_str:
        return "" # No condition

    # Use split_outside_parens to count parts accurately
    comma_parts = split_outside_parens(cond_str, [','])
    num_comma_parts = len(comma_parts)

    has_range = any(re.search(r'\s+to\s+', part) for part in comma_parts)

    if num_comma_parts > 1:
        return "Multiple Conditions" # e.g., "5V, 10A" or "5V, 10V to 15V"
    elif num_comma_parts == 1:
        if has_range:
            return "Range Condition" # e.g., "5V to 10V"
        else:
            return "Single Condition" # e.g., "5V"
    else: # Should not happen if cond_str is not empty
        return "Unknown Condition Structure"

def classify_main(main_str: str) -> str:
    main_str = remove_parentheses_detailed(main_str).strip()
    if not main_str:
        return "" # Empty main part

    comma_parts = split_outside_parens(main_str, [','])
    num_comma_parts = len(comma_parts)
    has_range = any(re.search(r'\s+to\s+', part) for part in comma_parts)
    # Check if *any* part contains letters/units (excluding 'to')
    has_unit_chars = any(re.search(r'[a-zA-Zµ°%]', re.sub(r'\s+to\s+', '', part)) for part in comma_parts)


    if num_comma_parts > 1:
        return "Multi Value" # e.g., "10A, 20A" or "10, 20 to 30"
    elif num_comma_parts == 1:
        if has_range:
            return "Range Value" # e.g., "10A to 20A" or "10 to 20" (if units are implicit/global)
        else:
            # Single part, check if it contains units or is just a number
             # Regex looks for letters, micro sign, degree, percent
            if has_unit_chars:
                return "Single Value" # e.g., "10A", "50%"
            # Check if it's purely numeric (allowing sign, decimal)
            elif re.fullmatch(r'[+\-±]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?', comma_parts[0]):
                return "Number" # e.g., "10", "-5.5"
            else:
                # Contains other characters, maybe mixed alphanumeric?
                return "Complex Single" # Or "Unknown Single"
    else:
        return "Unknown Main Structure"


def classify_sub_value(subval: str):
    # Don't remove parentheses here, do it inside classify_main/condition if needed
    if "@" in subval:
        parts = subval.split('@', 1)
        main_part = parts[0].strip()
        cond_part = parts[1].strip() if len(parts) > 1 else ""
    else:
        main_part = subval.strip()
        cond_part = ""

    # Classify each part
    main_class = classify_main(main_part)
    cond_class = classify_condition(cond_part)

    # Determine characteristics
    has_range_in_main = (main_class == "Range Value")
    has_multi_value_in_main = (main_class == "Multi Value") # Based on comma splitting

    has_range_in_condition = (cond_class == "Range Condition")
    has_multiple_conditions = (cond_class == "Multiple Conditions") # Based on comma splitting

    # Count items/conditions
    main_item_count = count_main_items(main_part)
    cond_item_count = count_conditions(cond_part) # Number of condition clauses

    # Combine classification strings
    if main_class and cond_class:
        classification = f"{main_class} with {cond_class}"
    elif main_class:
        classification = main_class
    else:
        classification = "Invalid Structure" # Or empty?

    return (classification,
            has_range_in_main,
            has_multi_value_in_main,
            has_range_in_condition,
            has_multiple_conditions,
            cond_item_count,
            main_item_count)

def extract_identifiers_detailed(text: str):
    # Find all non-nested parentheses content
    return re.findall(r'\(([^()]*)\)', text) # Finds content inside (...)

def classify_value_type_detailed(raw_value: str):
    raw_value = str(raw_value).strip()
    if not raw_value:
        return ("", "", 0, 0, False, False, False, False, 0)

    # Extract identifiers first
    found_parens_content = extract_identifiers_detailed(raw_value)
    identifiers = ', '.join(found_parens_content)

    # For classification, work with the string *without* parentheses content removed,
    # as split_outside_parens handles them.
    # Sub-values are typically split by comma *unless* the comma is part of a condition definition.
    # Example: "10A @ 5V, 20A @ 5V" -> 2 sub-values
    # Example: "10A @ 5V, 6V" -> 1 sub-value (comma is part of multi-condition)

    # Let's use split_outside_parens based on comma as the primary sub-value separator,
    # *unless* an '@' sign is present. If '@' is present, the splitting logic changes.
    subvals = []
    if "@" in raw_value:
        # If '@' is present, treat the whole thing as one sub-value for this function's purpose.
        # The internal logic of classify_sub_value will handle main/condition parts.
        # However, we need to consider cases like "A @ X, B @ Y". This *should* be two sub-values.
        # Let's refine the split: Split by comma only if the comma is NOT preceded by '@' closely.
        # This is tricky. Let's stick to the simpler model for now:
        # Use split_outside_parens with comma delimiter.
        subvals = split_outside_parens(raw_value, [','])
        # Now, check if the split was appropriate. If a part *doesn't* contain '@' but others do, maybe recombine?
        # Example: "A @ X, Y" -> split is ["A @ X", "Y"]. Is "Y" a separate sub-value or part of X's condition?
        # Let's assume split_outside_parens is the definition of sub-values for now.
        pass # Use the result of split_outside_parens
    else:
        # No '@', split by comma straightforwardly
        subvals = split_outside_parens(raw_value, [','])

    # Ensure we have at least one subval if the raw_value wasn't empty
    if not subvals and raw_value:
        subvals = [raw_value]

    sub_value_count = len(subvals)
    if sub_value_count == 0: # Should not happen if raw_value was not empty
        return ("", identifiers, 0, 0, False, False, False, False, 0)

    # Analyze each sub-value
    sub_classifications = []
    sub_range_in_main = []
    sub_multi_in_main = []
    sub_range_in_condition = []
    sub_multi_cond = []
    sub_cond_counts = []
    sub_main_item_counts = []

    for sv in subvals:
        (cls,
         has_range_m, has_multi_m,
         has_range_c, has_multi_c,
         cond_count, main_item_count) = classify_sub_value(sv)

        sub_classifications.append(cls)
        sub_range_in_main.append(has_range_m)
        sub_multi_in_main.append(has_multi_m)
        sub_range_in_condition.append(has_range_c)
        sub_multi_cond.append(has_multi_c)
        sub_cond_counts.append(cond_count)
        sub_main_item_counts.append(main_item_count)

    # Determine overall characteristics
    final_class = ""
    if sub_value_count == 1:
        final_class = sub_classifications[0]
    else:
        # Multiple sub-values
        unique_classes = set(sub_classifications)
        if len(unique_classes) == 1:
            # e.g., all are "Single Value with Single Condition"
            final_class = f"Multiple ({sub_value_count}x) {next(iter(unique_classes))}"
        else:
            # e.g., "Single Value, Number with Condition"
            final_class = f"Multiple Mixed ({sub_value_count}x)" # Could list the types if needed

    # Aggregate counts (use max, sum, or list?)
    # For simplicity, let's report if *any* sub-value has the feature.
    has_range_in_main_overall = any(sub_range_in_main)
    has_multi_value_in_main_overall = any(sub_multi_in_main) # If any subval's main part had commas
    has_range_in_condition_overall = any(sub_range_in_condition)
    has_multiple_conditions_overall = any(sub_multi_cond) # If any subval's condition part had commas

    # Aggregate counts - maybe use max or a summary string
    final_cond_item_count_agg = max(sub_cond_counts) if sub_cond_counts else 0
    final_main_item_count_agg = max(sub_main_item_counts) if sub_main_item_counts else 0


    return (final_class, # The overall classification string
            identifiers, # Comma-separated string of content in ()
            sub_value_count, # How many comma-separated sub-values were detected
            final_cond_item_count_agg, # Max conditions found in any sub-value
            has_range_in_main_overall, # Any sub-value had range in its main part?
            has_multi_value_in_main_overall, # Any sub-value had multi-value in its main part?
            has_range_in_condition_overall, # Any sub-value had range in its condition?
            has_multiple_conditions_overall, # Any sub-value had multiple conditions?
            final_main_item_count_agg) # Max main items found in any sub-value

def fix_exceptions(s):
    # Add space before Ohm if preceded by a digit or known prefix
    # Be careful not to add space if it's already there, e.g., "10 k Ohm"
    # Look for a digit or prefix immediately followed by Ohm (case-insensitive)
    prefixes_pattern = "|".join(re.escape(p) for p in MULTIPLIER_MAPPING.keys())
    # Pattern: (digit OR prefix) optionally followed by space, then Ohm/ohm/OHM, but NOT preceded by a space already
    # This is getting complex. Simpler: Add space if digit/prefix is directly followed by Ohm.
    pattern = re.compile(r"(\d|" + prefixes_pattern + r")([Oo][Hh][Mm])")
    s = pattern.sub(r"\1 \2", s) # Add space

    # Add other specific fixes here if needed
    # s = s.replace("...", "...")
    return s


# MODIFIED process_single_key
def process_single_key(main_key: str, base_units, multipliers_dict):
    main_key_original = main_key # Keep original for output row
    main_key_clean = fix_exceptions(main_key.strip())

    # --- CHANGE: Use detailed classification logic ---
    try:
        (category, _, sub_value_count, _, _, _, _, _, _) = classify_value_type_detailed(main_key_clean)
        if not category: # Handle empty or invalid structures identified by detailed classifier
             category = "Unknown"
    except Exception as e:
        st.error(f"Error during detailed classification for '{main_key_clean}': {e}")
        category = "Classification Error"
    # --- END CHANGE ---

    if category == "Unknown" or category == "Classification Error":
        return [{
            "Main Key": main_key_original,
            "Category": category,
            "Attribute": "Value",
            "Code": "ERR",
            "Value": "Could not classify structure." if category=="Unknown" else f"Error: {e}"
        }]

    # --- CHANGE: Handle multiple sub-values based on detailed classification's sub_value_count ---
    # The category might be like "Multiple (2x) Single Value"
    # We still need to split the original string to process parts if sub_value_count > 1
    all_rows = []
    if sub_value_count > 1:
         # Use the same splitting logic used inside classify_value_type_detailed
         # For now, assume comma split outside parens is the primary mechanism
         chunks = split_outside_parens(main_key_clean, [','])
         if len(chunks) != sub_value_count:
             st.warning(f"Sub-value count mismatch for '{main_key_clean}'. Expected {sub_value_count}, got {len(chunks)} chunks. Processing chunks.")
             # Proceed with chunks found, may indicate complex structure not fully captured by simple comma split

         for idx, chunk in enumerate(chunks):
             chunk = chunk.strip()
             if not chunk: continue

             try:
                 # Re-classify the chunk to get its specific category if needed,
                 # although the overall category already told us it was multiple/mixed.
                 # Let's use the chunk's specific classification for block extraction.
                 (chunk_cat, _, _, _, _, _, _, _, _) = classify_value_type_detailed(chunk)
                 if not chunk_cat: chunk_cat = "Unknown Chunk"

                 if chunk_cat != "Unknown Chunk":
                    block_texts = extract_block_texts(chunk, chunk_cat) # Use specific chunk category
                    parsed_parts = []
                    for bt in block_texts:
                         part_val, part_unit = parse_value_unit_identifier(bt, base_units, multipliers_dict)
                         parsed_parts.append((part_val, part_unit))

                    # Generate mapping, using the chunk's category
                    code_map = generate_mapping(parsed_parts, chunk_cat)

                    # Prefix codes with M{idx+1}-
                    new_code_map = {}
                    for code, val in code_map.items():
                         new_code_map[f"M{idx+1}-{code}"] = val

                    # Get desired order, prefixed
                    desired_order_chunk = [f"M{idx+1}-{c}" for c in get_desired_order()]

                    # Create rows for this chunk
                    rows = display_mapping(new_code_map, desired_order_chunk, chunk_cat, main_key_original) # Use original key
                    all_rows.extend(rows)
                 else:
                     # Handle error for this specific chunk
                      error_row = {
                         "Main Key": main_key_original, "Category": f"Error in Chunk {idx+1}",
                         "Attribute": "Value", "Code": "ERR-CHK",
                         "Value": f"Could not classify chunk: {chunk}"
                      }
                      all_rows.append(error_row)

             except Exception as e:
                 st.error(f"Error processing chunk '{chunk}' from '{main_key_original}': {e}")
                 error_row = {
                     "Main Key": main_key_original, "Category": f"Error in Chunk {idx+1}",
                     "Attribute": "Value", "Code": "ERR-PROC",
                     "Value": str(e)
                 }
                 all_rows.append(error_row)
         return all_rows

    else:
        # Single sub-value (sub_value_count is 1 or 0 for empty)
        try:
            # Use the category determined at the start
            block_texts = extract_block_texts(main_key_clean, category)
            parsed_parts = []
            for bt in block_texts:
                part_val, part_unit = parse_value_unit_identifier(bt, base_units, multipliers_dict)
                parsed_parts.append((part_val, part_unit))

            code_map = generate_mapping(parsed_parts, category)
            rows = display_mapping(code_map, get_desired_order(), category, main_key_original) # Use original key
            return rows
        except Exception as e:
             st.error(f"Error processing single key '{main_key_original}': {e}")
             return [{
                 "Main Key": main_key_original, "Category": "Processing Error",
                 "Attribute": "Value", "Code": "ERR",
                 "Value": str(e)
             }]


# --- Helper functions for Fixed Pipeline that rely on Category Name ---
# (extract_block_texts, get_code_prefixes_for_category, generate_mapping, display_mapping)
# These should work correctly as long as the `category` string passed to them
# matches the expected names (e.g., "Single Value", "Range Value Single Condition")
# generated by `classify_value_type_detailed`.

def extract_block_texts(main_key, category_name):
    """Extracts logical parts based on the structure defined by category_name."""
    main_key = main_key.strip()
    parts = []

    # Use regex or string splitting based on the known structure of the category
    if " with " in category_name:
        main_type, cond_type = category_name.split(" with ", 1)
        if "@" in main_key:
            main_part, cond_part = main_key.split("@", 1)
            main_part = main_part.strip()
            cond_part = cond_part.strip()
        else:
            # Should not happen if category includes " with "
            return [main_key] # Fallback

        # Split main part based on main_type
        if main_type.startswith("Range Value"):
            parts.extend(re.split(r'\s+to\s+', main_part))
        elif main_type.startswith("Multi Value"):
             parts.extend(split_outside_parens(main_part, [',']))
        else: # Single Value or Number
            parts.append(main_part)

        # Split condition part based on cond_type
        if cond_type.startswith("Range Condition"):
             parts.extend(re.split(r'\s+to\s+', cond_part))
        elif cond_type.startswith("Multiple Conditions"):
             parts.extend(split_outside_parens(cond_part, [',']))
        else: # Single Condition
            parts.append(cond_part)

    elif category_name.startswith("Range Value"): # No condition part
        parts.extend(re.split(r'\s+to\s+', main_key))
    elif category_name.startswith("Multi Value"): # No condition part
         parts.extend(split_outside_parens(main_key, [',']))
    elif category_name.startswith("Single Value") or category_name.startswith("Number"): # No condition part
        parts.append(main_key)
    elif category_name.startswith("Multiple"): # Handle cases like "Multiple (2x) Single Value"
         # This function is designed for one value string. The calling function should handle splitting multiples.
         # If called with a multiple category, maybe just return the key? Or try splitting?
         # Let's assume the caller handles the split and this receives a single logical chunk category.
         # Fallback for unexpected multiple category:
         parts.append(main_key)
    else: # Unknown or simple Number
        parts.append(main_key)

    return [p.strip() for p in parts if p.strip()]

def parse_value_unit_identifier(raw_chunk, base_units, multipliers_dict):
    """
    Parses a raw string chunk (e.g., '50 mOhm (typ)') into value_with_prefix and base_unit.
    value_with_prefix includes the number and multiplier symbol as found.
    base_unit is the resolved base unit string.
    """
    # 1. Extract identifiers (parentheses content) if needed elsewhere, but remove for parsing
    identifier_content = extract_identifiers_detailed(raw_chunk) # Content like ['typ']
    chunk_no_paren = remove_parentheses_detailed(raw_chunk).strip()

    # 2. Use the robust extraction function
    num_val, multi_sym, base_unit, norm_val, err_flag = extract_numeric_and_unit_analysis(
        chunk_no_paren, base_units, multipliers_dict
    )

    # 3. Reconstruct the "value_with_prefix" string from the original chunk
    value_with_prefix = ""
    if num_val is not None:
        # Find the numeric part in the original (no paren) string
        # Need to handle float conversion back to string carefully if needed for matching
        num_str = ""
        # Try finding exact numeric string representations
        if isinstance(num_val, int) or num_val.is_integer():
            num_str = str(int(num_val))
        else:
            num_str = str(num_val)
        # Add sign if present originally (e.g. from extract_numeric...)
        # This is tricky. Let's just take the numeric value found.

        # Find where the unit/multiplier starts
        rest_part = ""
        if multi_sym != "1" and multi_sym is not None:
            rest_part += multi_sym
        if base_unit:
             # Add space correctly? Based on fix_exceptions logic?
             # Let's just append. Spacing handled by extract_numeric... implicitly somewhat.
             rest_part += base_unit

        # Search for the numeric part and the rest part in the chunk_no_paren
        # This is complex. Let's simplify: return the extracted num_val and base_unit.
        # The fixed pipeline needs the *string* representation.

        # Alternative: Return the numeric string and multiplier string found by extract...
        num_str_extracted = ""
        multi_str_extracted = ""
        pattern_num = re.compile(r'^(?P<numeric>[+\-±]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)')
        m_num = pattern_num.match(chunk_no_paren)
        if m_num:
             num_str_extracted = m_num.group("numeric")

        if multi_sym != "1" and multi_sym is not None:
            multi_str_extracted = multi_sym # Assumes multi_sym is the string as found

        # Combine them - requires knowing the space?
        # Let's use a simpler approach for the fixed pipeline output:
        # Return the numeric string found and the base unit.
        # The 'Code' mapping might need adjustment if it expects prefix info.
        value_for_output = num_str_extracted
        base_unit_for_output = base_unit if base_unit else "" # Return empty string if no unit

        return (value_for_output, base_unit_for_output)

    elif chunk_no_paren in base_units: # Handle case where chunk is just a unit
        return ("", chunk_no_paren)
    else:
        # Could not parse a number, maybe it's just text?
        # Return the original chunk (no paren) as value, empty unit? Or flag error?
        # Let's return original (no paren) as value, signifying non-standard structure.
        return (chunk_no_paren, "")


def get_code_prefixes_for_category(category_name):
    # This mapping needs to align with the category names produced by classify_value_type_detailed
    # Add entries for all expected classifications.
    # Using simplified prefixes for clarity. Adjust codes (e.g., V0, Vn0, Vx0) as needed.

    # --- Single Value Structures ---
    if category_name == "Number":
        return [{"prefix": "SN-", "codes": ["SN-V", "SN-U"], "attributes": ["Value", "Unit"]}]
    elif category_name == "Single Value":
        return [{"prefix": "SV-", "codes": ["SV-V", "SV-U"], "attributes": ["Value", "Unit"]}]
    elif category_name == "Complex Single": # Handle the new category from classify_main
         return [{"prefix": "CX-", "codes": ["CX-V", "CX-U"], "attributes": ["Value", "Unit"]}]

    # --- Range Value Structures ---
    elif category_name == "Range Value": # e.g., "10A to 20A"
        return [
            {"prefix": "RV-", "codes": ["RV-Vn", "RV-Un"], "attributes": ["Value", "Unit"]}, # Min/Start
            {"prefix": "RV-", "codes": ["RV-Vx", "RV-Ux"], "attributes": ["Value", "Unit"]}  # Max/End
        ]

    # --- Multi Value Structures ---
    elif category_name == "Multi Value": # e.g., "10A, 20A"
        # The calling function (process_single_key) handles splitting these.
        # This function should ideally receive the category of a *chunk*.
        # If called with "Multi Value", what should it return? Let's assume it's called for a chunk.
        # Fallback if called directly with "Multi Value":
        return [{"prefix": "MV-", "codes": ["MV-V", "MV-U"], "attributes": ["Value", "Unit"]}] # Placeholder

    # --- Structures with Conditions ---
    elif category_name == "Number with Single Condition":
        return [
            {"prefix": "SN-", "codes": ["SN-V", "SN-U"], "attributes": ["Value", "Unit"]}, # Number
            {"prefix": "SC-", "codes": ["SC-V", "SC-U"], "attributes": ["Value", "Unit"]}  # Condition
        ]
    elif category_name == "Single Value with Single Condition":
        return [
            {"prefix": "SV-", "codes": ["SV-V", "SV-U"], "attributes": ["Value", "Unit"]}, # Value
            {"prefix": "SC-", "codes": ["SC-V", "SC-U"], "attributes": ["Value", "Unit"]}  # Condition
        ]
    elif category_name == "Range Value with Single Condition":
        return [
            {"prefix": "RV-", "codes": ["RV-Vn", "RV-Un"], "attributes": ["Value", "Unit"]}, # Range Min
            {"prefix": "RV-", "codes": ["RV-Vx", "RV-Ux"], "attributes": ["Value", "Unit"]}, # Range Max
            {"prefix": "SC-", "codes": ["SC-V", "SC-U"], "attributes": ["Value", "Unit"]}   # Condition
        ]
    elif category_name == "Number with Range Condition": # New combination needed
         return [
             {"prefix": "SN-", "codes": ["SN-V", "SN-U"], "attributes": ["Value", "Unit"]}, # Number
             {"prefix": "RC-", "codes": ["RC-Vn", "RC-Un"], "attributes": ["Value", "Unit"]}, # Cond Range Min
             {"prefix": "RC-", "codes": ["RC-Vx", "RC-Ux"], "attributes": ["Value", "Unit"]}  # Cond Range Max
         ]
    elif category_name == "Single Value with Range Condition":
         return [
             {"prefix": "SV-", "codes": ["SV-V", "SV-U"], "attributes": ["Value", "Unit"]}, # Value
             {"prefix": "RC-", "codes": ["RC-Vn", "RC-Un"], "attributes": ["Value", "Unit"]}, # Cond Range Min
             {"prefix": "RC-", "codes": ["RC-Vx", "RC-Ux"], "attributes": ["Value", "Unit"]}  # Cond Range Max
         ]
    elif category_name == "Range Value with Range Condition":
         return [
             {"prefix": "RV-", "codes": ["RV-Vn", "RV-Un"], "attributes": ["Value", "Unit"]}, # Value Range Min
             {"prefix": "RV-", "codes": ["RV-Vx", "RV-Ux"], "attributes": ["Value", "Unit"]}, # Value Range Max
             {"prefix": "RC-", "codes": ["RC-Vn", "RC-Un"], "attributes": ["Value", "Unit"]}, # Cond Range Min
             {"prefix": "RC-", "codes": ["RC-Vx", "RC-Ux"], "attributes": ["Value", "Unit"]}  # Cond Range Max
         ]
    elif category_name == "Number with Multiple Conditions":
         # How to handle multiple conditions? Use MC- prefix?
         return [
             {"prefix": "SN-", "codes": ["SN-V", "SN-U"], "attributes": ["Value", "Unit"]}, # Number
             # Conditions handled by MC- prefix logic in generate_mapping
         ]
    elif category_name == "Single Value with Multiple Conditions":
         return [
             {"prefix": "SV-", "codes": ["SV-V", "SV-U"], "attributes": ["Value", "Unit"]}, # Value
             # Conditions handled by MC- prefix logic
         ]
    elif category_name == "Range Value with Multiple Conditions":
         return [
             {"prefix": "RV-", "codes": ["RV-Vn", "RV-Un"], "attributes": ["Value", "Unit"]}, # Value Range Min
             {"prefix": "RV-", "codes": ["RV-Vx", "RV-Ux"], "attributes": ["Value", "Unit"]}, # Value Range Max
             # Conditions handled by MC- prefix logic
         ]

    # --- Multiple Top-Level Values (handled by caller, but add fallback) ---
    elif category_name.startswith("Multiple"):
         # Extract the base category if possible, e.g., "Multiple (2x) Single Value" -> "Single Value"
         match = re.search(r'\)\s*(.*)$', category_name)
         base_cat = match.group(1) if match else "Unknown Multiple Base"
         st.write(f"DEBUG: Multiple category detected, base: {base_cat}")
         # Return codes for the base category? Or special handling?
         # For now, return a generic multiple code. The caller prefixes with M1-, M2- etc.
         return [{"prefix": "MULTI-", "codes": ["MULTI-V", "MULTI-U"], "attributes": ["Value", "Unit"]}]

    # --- Fallback ---
    else:
        st.warning(f"Unknown category '{category_name}' in get_code_prefixes_for_category. Using default.")
        return [{"prefix": "UNK-", "codes": ["UNK-V", "UNK-U"], "attributes": ["Value", "Unit"]}]

def generate_mapping(parsed_parts, category_name):
    """Generates the code-to-value mapping based on parsed parts and category."""
    blocks = get_code_prefixes_for_category(category_name)
    mapping = {}
    part_idx = 0
    block_idx = 0
    mc_counter = 1 # For extra conditions (Multiple Conditions)

    while part_idx < len(parsed_parts):
        part_tuple = parsed_parts[part_idx]

        if block_idx < len(blocks):
            block_info = blocks[block_idx]
            new_map = fill_mapping_for_part(part_tuple, block_info)
            mapping.update(new_map)
            part_idx += 1
            block_idx += 1
        elif "Multiple Conditions" in category_name:
            # Ran out of defined blocks, but category expects more parts (conditions)
            block_info = {
                "prefix": "MC-", # Multi-Condition prefix
                "codes": [f"MC-V{mc_counter}", f"MC-U{mc_counter}"],
                "attributes": ["Value", "Unit"]
            }
            new_map = fill_mapping_for_part(part_tuple, block_info)
            mapping.update(new_map)
            part_idx += 1
            mc_counter += 1
        else:
            # Ran out of blocks and not expecting multiple conditions - extra parts?
            st.warning(f"Warning: More parsed parts ({len(parsed_parts)}) than blocks ({len(blocks)}) for category '{category_name}'. Ignoring extra part: {part_tuple}")
            part_idx += 1 # Skip this part

    return mapping


def fill_mapping_for_part(part_tuple, block_info):
    """Fills the mapping for a single parsed part using block info."""
    (val_str, base_unit_str) = part_tuple
    result = {}
    # Ensure we have enough codes for Value and Unit
    if len(block_info["codes"]) >= 2 and len(block_info["attributes"]) >= 2:
         # Assume codes[0] is for Value, codes[1] is for Unit based on attribute order
         value_code = block_info["codes"][block_info["attributes"].index("Value")]
         unit_code = block_info["codes"][block_info["attributes"].index("Unit")]
         result[value_code] = {"value": val_str}
         result[unit_code] = {"value": base_unit_str}
    else:
         # Fallback or error if block_info structure is wrong
         st.error(f"Error: Block info structure incorrect for filling mapping: {block_info}")
         # Create generic codes if possible
         result[f"{block_info.get('prefix','ERR')}-V"] = {"value": val_str}
         result[f"{block_info.get('prefix','ERR')}-U"] = {"value": base_unit_str}

    return result


def display_mapping(mapping_dict, desired_order, category, main_key):
    """Formats the mapping dictionary into rows for the output DataFrame."""
    def get_attribute_from_code(code):
        # Simplified attribute guessing based on code suffix
        if code.endswith("-V") or code.endswith("-Vn") or code.endswith("-Vx") or re.search(r'-V\d+$', code):
             return "Value"
        elif code.endswith("-U") or code.endswith("-Un") or code.endswith("-Ux") or re.search(r'-U\d+$', code):
             return "Unit"
        return "Unknown Attribute" # Fallback

    output = []
    used_codes = set()

    # Add rows for codes in the desired order that are present in the mapping
    for code in desired_order:
        if code in mapping_dict:
            attr = get_attribute_from_code(code)
            row = {
                "Main Key": main_key, # Original input string
                "Category": category, # Classification name
                "Attribute": attr,    # Value or Unit
                "Code": code,         # The specific code (e.g., SV-V, RC-Un)
                "Value": mapping_dict[code].get("value", "") # The parsed value or unit string
            }
            output.append(row)
            used_codes.add(code)

    # Add any extra codes found in the mapping but not in the desired order
    extras = sorted([c for c in mapping_dict if c not in used_codes])
    for code in extras:
        attr = get_attribute_from_code(code)
        row = {
            "Main Key": main_key,
            "Category": category,
            "Attribute": attr,
            "Code": code,
            "Value": mapping_dict[code].get("value", "")
        }
        output.append(row)

    return output

def get_desired_order():
    # Update this list based on the codes defined in get_code_prefixes_for_category
    # Include potential codes for all structures and MC codes
    return [
        # Number
        "SN-V", "SN-U",
        # Single Value
        "SV-V", "SV-U",
         # Complex Single
         "CX-V", "CX-U",
        # Range Value (main)
        "RV-Vn", "RV-Un", "RV-Vx", "RV-Ux",
        # Single Condition
        "SC-V", "SC-U",
        # Range Condition
        "RC-Vn", "RC-Un", "RC-Vx", "RC-Ux",
        # Multi-Condition (dynamic) - add first few expected ones
        "MC-V1", "MC-U1", "MC-V2", "MC-U2", "MC-V3", "MC-U3",
         # MULTI (placeholder for base type in a multiple structure - might not appear directly)
         "MULTI-V", "MULTI-U",
         # Unknown/Error
         "UNK-V", "UNK-U", "ERR"
    ]

# MODIFIED process_fixed_pipeline_bytes
def process_fixed_pipeline_bytes(file_bytes: bytes):
    """
    Runs the first pipeline (fixed processing) on the input file bytes.
    Reads mapping.xlsx from disk.
    Produces processed data (structured rows with codes).
    Returns a list of dictionary rows.
    """
    print("Running Fixed Processing Pipeline...")
    st.write("DEBUG: Starting Fixed Processing Pipeline...")
    all_processed_rows = []
    output_filename_temp = 'temp_processed_output.xlsx' # Avoid conflict if run multiple times

    try:
        # Read mapping file from disk (saved by Streamlit UI)
        base_units, multipliers_dict = read_mapping_file("mapping.xlsx")
        combined_base_units = LOCAL_BASE_UNITS.union(base_units)
        st.write(f"DEBUG: Using {len(combined_base_units)} base units for fixed pipeline.")

        # Read input excel from bytes
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        all_sheets = xls.sheet_names
        print(f"Found sheets: {all_sheets}")
        st.write(f"DEBUG: Input sheets: {all_sheets}")

        total_rows_processed = 0
        chunk_size = 500 # Process in chunks

        for sheet in all_sheets:
            print(f"Processing sheet: '{sheet}'")
            st.write(f"DEBUG: Processing sheet: '{sheet}'")
            try:
                sheet_df = pd.read_excel(xls, sheet_name=sheet)
                # Ensure 'Value' column exists
                if 'Value' not in sheet_df.columns:
                     st.warning(f"Sheet '{sheet}' skipped: Missing 'Value' column.")
                     continue

                sheet_rows_processed = 0
                for i in range(0, len(sheet_df), chunk_size):
                    chunk = sheet_df.iloc[i:i + chunk_size]
                    for row_index, row in chunk.iterrows():
                        main_key = str(row.get('Value', '')).strip()
                        if not main_key:
                            continue

                        # Run the core processing logic for a single key
                        results = process_single_key(main_key, combined_base_units, multipliers_dict)

                        # Add original row data and sheet info to each result row
                        orig_data = row.to_dict()
                        for r in results:
                            new_row = orig_data.copy() # Start with original data
                            new_row.update(r)          # Add/overwrite with processed results (Main Key, Category, Code, Value, Attribute)
                            new_row["Sheet"] = sheet   # Add sheet name
                            all_processed_rows.append(new_row)
                        sheet_rows_processed += 1
                    gc.collect() # Garbage collect after each chunk
                total_rows_processed += sheet_rows_processed
                st.write(f"DEBUG: Processed {sheet_rows_processed} rows from sheet '{sheet}'.")

            except Exception as e:
                st.error(f"Error processing sheet '{sheet}': {e}")
                continue # Skip to next sheet on error

        st.write(f"DEBUG: Fixed Processing Pipeline finished. Total rows generated: {len(all_processed_rows)}")

        if not all_processed_rows:
             st.warning("Fixed processing generated no output rows.")
             return None # Return None if no data

        # Create DataFrame from all rows
        processed_df = pd.DataFrame(all_processed_rows)

        # Save to a temporary Excel file (optional, could return df directly)
        # writer = pd.ExcelWriter(output_filename_temp, engine='openpyxl')
        # processed_df.to_excel(writer, sheet_name='Combined', index=False)
        # writer.close()
        # st.write(f"DEBUG: Saved temporary processed file to {output_filename_temp}")

        # Return the DataFrame
        return processed_df


    except FileNotFoundError as e:
         st.error(f"Pipeline Error: {e}")
         st.stop()
    except ValueError as e:
         st.error(f"Pipeline Error: {e}")
         st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred in the fixed pipeline: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()


# ========== END FIRST PIPELINE (Fixed) ==========


#############################################
# 4) SECOND PIPELINE: DETAILED ANALYSIS
#############################################

# REMOVED basic_classification function

# --- Helper functions for Detailed Analysis ---
# (Keep replace_numbers_keep_sign_all, replace_numbers_keep_sign_outside_parens)
# These are now used inside detailed_analysis_pipeline

def replace_numbers_keep_sign_all(s: str) -> str:
    # Replace floating point numbers (including scientific notation) with '$', keeping sign
    # Regex captures optional sign, digits, optional decimal part, optional exponent
    return re.sub(r'([+-])?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', r'\1$', str(s))

def replace_numbers_keep_sign_outside_parens(s: str) -> str:
    s = str(s) # Ensure string
    result = []
    in_paren = False
    i = 0
    while i < len(s):
        char = s[i]
        if char == '(':
            in_paren = True
            result.append(char)
            i += 1
        elif char == ')':
            in_paren = False
            result.append(char)
            i += 1
        else:
            if in_paren:
                result.append(char)
                i += 1
            else:
                # Look for number pattern at current position
                 # Use the same robust regex as replace_numbers_keep_sign_all
                match_ = re.match(r'([+-]?)\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', s[i:])
                if match_:
                    sign = match_.group(1) if match_.group(1) else '' # Handle no sign
                    result.append(sign + '$')
                    i += len(match_.group(0)) # Move index past the matched number
                else:
                    result.append(char)
                    i += 1
    return "".join(result)


# --- detailed_analysis_pipeline function (incorporates normalization) ---
def detailed_analysis_pipeline(df, base_units, multipliers_dict):
    """Performs detailed analysis, including classification and normalization."""
    st.write("DEBUG: Starting Detailed Analysis Pipeline...")
    results = [] # List to store row-by-row results as dicts

    # Check if 'Value' column exists
    if 'Value' not in df.columns:
        st.error("Detailed Analysis Error: Input DataFrame missing 'Value' column.")
        return pd.DataFrame() # Return empty DataFrame

    i = 0
    for index, row in df.iterrows():
        val_str = str(row.get('Value', '')).strip()
        row_results = {"OriginalIndex": index} # Store original index if needed
        row_results["Value"] = val_str # Keep original value

        if not val_str:
            # Handle empty value rows if necessary, e.g., fill with defaults
            row_results.update({
                'Classification': "Empty", 'Identifiers': "", 'SubValueCount': 0,
                'ConditionCount': 0, 'HasRangeInMain': False, 'HasMultiValueInMain': False,
                'HasRangeInCondition': False, 'HasMultipleConditions': False,
                'DetailedValueType': "Empty", 'Normalized Unit': "", 'Absolute Unit': "",
                'MainUnits': "", 'MainDistinctUnitCount': 0, 'MainUnitsConsistent': True,
                'ConditionUnits': "", 'ConditionDistinctUnitCount': 0, 'ConditionUnitsConsistent': True,
                'OverallUnitConsistency': True, 'ParsingErrorFlag': False,
                'SubValueUnitVariationSummary': "Main: None; Condition: None",
                 'MainNumericValues': "", 'ConditionNumericValues': "", 'MainMultipliers': "",
                 'ConditionMultipliers': "", 'MainBaseUnits': "", 'ConditionBaseUnits': "",
                 'NormalizedMainValues': "", 'NormalizedConditionValues': "",
                 'MinNormalizedValue': None, 'MaxNormalizedValue': None,
                 'SingleUnitForAllSubs': True, 'AllDistinctUnitsUsed': ""
            })
            results.append(row_results)
            continue

        try:
            # 1. Classification and Structure Analysis
            (cls, ids, sv_count, final_cond_item_count,
             rng_main, multi_main, rng_cond, multi_cond,
             final_main_item_count) = classify_value_type_detailed(val_str)

            row_results['Classification'] = cls
            row_results['Identifiers'] = ids
            row_results['SubValueCount'] = sv_count
            # Use consistent naming? final_cond_item_count vs ConditionCount
            row_results['ConditionCount'] = final_cond_item_count
            row_results['HasRangeInMain'] = rng_main
            row_results['HasMultiValueInMain'] = multi_main
            row_results['HasRangeInCondition'] = rng_cond
            row_results['HasMultipleConditions'] = multi_cond
            # Add main item count?
            # row_results['MainItemCount'] = final_main_item_count

            # Create DetailedValueType string
            if cls and cls != "Invalid Structure":
                dvt = f"{cls} [M:{final_main_item_count}][C:{final_cond_item_count}]"
                if sv_count > 1 : dvt += f" (x{sv_count})"
            else:
                dvt = cls # Show "Invalid Structure" or empty if cls was empty
            row_results['DetailedValueType'] = dvt

            # 2. Normalization (previously in basic_classification)
            row_results['Normalized Unit'] = replace_numbers_keep_sign_all(val_str)
            # row_results['Normalized Unit_edit'] = replace_numbers_keep_sign_outside_parens(val_str) # If needed

            # 3. Absolute Unit Resolution
            # Use the refined resolve_compound_unit
            row_results["Absolute Unit"] = resolve_compound_unit(val_str, base_units, multipliers_dict)

            # 4. Unit Analysis (Main vs Condition)
            ua = analyze_value_units(val_str, base_units, multipliers_dict)
            row_results["MainUnits"] = ", ".join(safe_str(x) for x in ua["main_units"])
            row_results["MainDistinctUnitCount"] = len(ua["main_distinct_units"])
            row_results["MainUnitsConsistent"] = ua["main_units_consistent"]
            row_results["ConditionUnits"] = ", ".join(safe_str(x) for x in ua["condition_units"])
            row_results["ConditionDistinctUnitCount"] = len(ua["condition_distinct_units"])
            row_results["ConditionUnitsConsistent"] = ua["condition_units_consistent"]
            # row_results["MainSubAnalysis"] = ua["main_sub_analysis"] # Keep if debugging needed
            # row_results["ConditionSubAnalysis"] = ua["condition_sub_analysis"] # Keep if debugging needed
            row_results["OverallUnitConsistency"] = ua["overall_consistent"]


            # 5. Numeric Value Extraction & Normalization
            num_info = extract_numeric_info_for_value(val_str, base_units, multipliers_dict)
            row_results["MainNumericValues"] = ", ".join(safe_str(x) for x in num_info["main_numeric"])
            row_results["ConditionNumericValues"] = ", ".join(safe_str(x) for x in num_info["condition_numeric"])
            row_results["MainMultipliers"] = ", ".join(safe_str(x) for x in num_info["main_multipliers"])
            row_results["ConditionMultipliers"] = ", ".join(safe_str(x) for x in num_info["condition_multipliers"])
            row_results["MainBaseUnits"] = ", ".join(safe_str(x) for x in num_info["main_base_units"])
            row_results["ConditionBaseUnits"] = ", ".join(safe_str(x) for x in num_info["condition_base_units"])
            row_results["NormalizedMainValues"] = ", ".join(safe_str(x) for x in num_info["normalized_main"])
            row_results["NormalizedConditionValues"] = ", ".join(safe_str(x) for x in num_info["normalized_condition"])

            parsing_error = any(num_info["main_errors"]) or any(num_info["condition_errors"])
            row_results["ParsingErrorFlag"] = parsing_error

            # 6. Summaries and Derived Metrics
            # Unit variation summary
            main_variation = "None"
            if ua["main_distinct_units"]:
                main_variation = "Uniform: " + next(iter(ua["main_distinct_units"])) if ua["main_units_consistent"] else "Mixed: " + ", ".join(ua["main_distinct_units"])
            condition_variation = "None"
            if ua["condition_distinct_units"]:
                 condition_variation = "Uniform: " + next(iter(ua["condition_distinct_units"])) if ua["condition_units_consistent"] else "Mixed: " + ", ".join(ua["condition_distinct_units"])
            row_results["SubValueUnitVariationSummary"] = f"Main: {main_variation}; Condition: {condition_variation}"

            # Min/Max and overall units
            all_normalized_numeric = [v for v in num_info["normalized_main"] + num_info["normalized_condition"] if isinstance(v, (int, float))]
            all_base_units_used = [u for u in num_info["main_base_units"] + num_info["condition_base_units"] if u and u != "None"]

            min_val = min(all_normalized_numeric) if all_normalized_numeric else None
            max_val = max(all_normalized_numeric) if all_normalized_numeric else None
            distinct_units_all = set(all_base_units_used)
            is_single_unit = (len(distinct_units_all) <= 1)

            row_results["MinNormalizedValue"] = min_val
            row_results["MaxNormalizedValue"] = max_val
            row_results["SingleUnitForAllSubs"] = is_single_unit
            row_results["AllDistinctUnitsUsed"] = ", ".join(sorted(list(distinct_units_all))) if distinct_units_all else ""

            results.append(row_results)

        except Exception as e:
             st.error(f"Error analyzing row {index}, Value: '{val_str}': {e}")
             # Append partial results or error marker?
             error_results = {"OriginalIndex": index, "Value": val_str, "Classification": "Analysis Error", "DetailedValueType": str(e)}
             results.append(error_results) # Add row with error info
        i += 1
        if i % 100 == 0: # Print progress
             st.write(f"DEBUG: Detailed analysis progress: {i}/{len(df)} rows")


    st.write(f"DEBUG: Detailed Analysis Pipeline finished. Generated {len(results)} result rows.")
    analysis_df = pd.DataFrame(results)

    # Merge back with original non-'Value' columns
    if 'Value' in df.columns:
         original_other_cols = df.drop(columns=['Value']).reset_index(drop=True)
         # Ensure analysis_df also has a sequential index if 'OriginalIndex' wasn't used or is messy
         analysis_df = analysis_df.reset_index(drop=True)
         # Merge based on index
         final_df = pd.concat([original_other_cols, analysis_df], axis=1)
         # Drop OriginalIndex if it was added
         if "OriginalIndex" in final_df.columns:
             final_df = final_df.drop(columns=["OriginalIndex"])
    else:
         final_df = analysis_df # Only analysis results if no original 'Value' col


    return final_df


# MODIFIED detailed_analysis function
def detailed_analysis(input_df: pd.DataFrame, mapping_file: str, output_file: str):
    """
    Performs detailed analysis on the input DataFrame using the mapping file.
    Saves the results to the output Excel file.
    Returns the path to the output file.
    """
    st.write("DEBUG: Starting detailed_analysis function...")
    try:
        base_units, multipliers_dict = read_mapping_file(mapping_file)
        st.write(f"DEBUG: Using {len(base_units)} base units for detailed analysis.")
    except Exception as e:
        st.error(f"Error reading mapping file '{mapping_file}': {e}")
        return None # Indicate failure

    # Input is now a DataFrame
    if not isinstance(input_df, pd.DataFrame):
         st.error("Detailed Analysis Error: Input must be a pandas DataFrame.")
         return None
    if 'Value' not in input_df.columns:
        st.error("Detailed Analysis Error: Input DataFrame must contain a column named 'Value'.")
        return None

    # Run the main analysis pipeline
    try:
        analysis_df = detailed_analysis_pipeline(input_df.copy(), base_units, multipliers_dict) # Use copy
    except Exception as e:
         st.error(f"Error during detailed analysis pipeline execution: {e}")
         import traceback
         st.error(traceback.format_exc())
         return None

    if analysis_df.empty:
         st.warning("Detailed analysis resulted in an empty DataFrame.")
         # Decide if saving an empty file is desired or return None
         # return None

    # Save the results
    try:
        analysis_df.to_excel(output_file, index=False)
        print(f"[✓] Detailed analysis saved to '{output_file}'.")
        st.write(f"DEBUG: Detailed analysis saved to '{output_file}'. Shape: {analysis_df.shape}")
        return output_file # Return path on success
    except Exception as e:
        st.error(f"Error writing detailed analysis to '{output_file}': {e}")
        return None # Indicate failure


#############################################
# 5) BUFFER: COMBINE RESULTS
#############################################

# MODIFIED combine_results
def combine_results(processed_df: pd.DataFrame, # Changed input to DataFrame
                    analysis_file: str,
                    output_file="final_combined.xlsx"):
    """Combines processed data (from fixed pipeline) with detailed analysis results."""
    st.write("DEBUG: Starting combine_results function...")
    try:
        # 1) Processed data is already a DataFrame
        if not isinstance(processed_df, pd.DataFrame) or processed_df.empty:
            st.error("Combine Error: Invalid or empty processed DataFrame provided.")
            return None

        # 2) Read analysis results file
        try:
            df_analysis = pd.read_excel(analysis_file)
            st.write(f"DEBUG: Read analysis file '{analysis_file}'. Shape: {df_analysis.shape}")
        except FileNotFoundError:
            st.error(f"Combine Error: Analysis file not found at '{analysis_file}'.")
            return None
        except Exception as e:
            st.error(f"Combine Error: Failed to read analysis file '{analysis_file}': {e}")
            return None

        if df_analysis.empty:
            st.warning("Combine Warning: Analysis DataFrame is empty. Combining with processed data only.")
            # If analysis is empty, maybe just save the processed_df?
            # Let's proceed with merge, it will likely result in NaNs for analysis columns.

        # 3) Merge
        # Ensure keys exist in both DataFrames
        if 'Main Key' not in processed_df.columns:
             st.error("Combine Error: 'Main Key' column missing in processed data.")
             return None
        if 'Value' not in df_analysis.columns:
             st.error("Combine Error: 'Value' column missing in analysis data (needed for merging).")
             # If 'Value' is missing, maybe try merging on index if appropriate? Assuming Value is the key.
             return None

        st.write("DEBUG: Columns in processed_df:", processed_df.columns.tolist())
        st.write("DEBUG: Columns in df_analysis:", df_analysis.columns.tolist())

        # Define columns to bring from analysis. Include the newly added ones.
        analysis_cols_to_merge = [
            "Value", # Key column from analysis side
            "Classification",
            "DetailedValueType",
            "Identifiers",
            "SubValueCount",
            "ConditionCount",
            "HasRangeInMain", "HasMultiValueInMain", "HasRangeInCondition", "HasMultipleConditions",
            "Normalized Unit",
            "Absolute Unit",
            "MainUnits", "MainDistinctUnitCount", "MainUnitsConsistent",
            "ConditionUnits", "ConditionDistinctUnitCount", "ConditionUnitsConsistent",
            "OverallUnitConsistency",
            "ParsingErrorFlag",
            "SubValueUnitVariationSummary",
            "MainNumericValues", "ConditionNumericValues", "MainMultipliers", "ConditionMultipliers",
            "MainBaseUnits", "ConditionBaseUnits", "NormalizedMainValues", "NormalizedConditionValues",
            "MinNormalizedValue", "MaxNormalizedValue", "SingleUnitForAllSubs", "AllDistinctUnitsUsed"
        ]
        # Select only existing columns from analysis_df to avoid KeyErrors
        existing_analysis_cols = [col for col in analysis_cols_to_merge if col in df_analysis.columns]
        if not existing_analysis_cols or "Value" not in existing_analysis_cols:
             st.error("Combine Error: Analysis data is missing key columns for merging.")
             return None
        st.write(f"DEBUG: Merging using these analysis columns: {existing_analysis_cols}")


        # Perform the merge
        df_merged = processed_df.merge(
            df_analysis[existing_analysis_cols],
            how="left",
            left_on="Main Key", # From processed_df
            right_on="Value",   # From analysis_df
            suffixes=("_processed", "_analysis") # Add suffixes to handle potential conflicts beyond the key
        )
        st.write(f"DEBUG: Merged DataFrame shape: {df_merged.shape}")


        # 4) Clean up merged columns
        # Drop the redundant 'Value' column from the analysis side if it exists
        if 'Value_analysis' in df_merged.columns:
            df_merged.drop(columns=["Value_analysis"], inplace=True)
        elif 'Value' in df_merged.columns and 'Main Key' in df_merged.columns and df_merged['Value'].equals(df_merged['Main Key']):
             # If suffixes weren't added and Value == Main Key, drop Value? Or keep Main Key?
             # Let's assume Main Key is the definitive original identifier.
             pass # Keep both for now, decide on final columns below

        # Handle potential duplicated columns if suffixes weren't always applied (e.g., 'Category')
        # Example: If 'Category' exists in both, merge might create 'Category_processed', 'Category_analysis'
        # Or it might just take the left one. Check columns:
        st.write(f"DEBUG: Columns after merge & potential drop: {df_merged.columns.tolist()}")


        # 5) Define and Reorder final columns
        # Start with columns from the processed_df structure
        final_columns_order = [
            "Main Key",         # Original input value string
            # --- Analysis columns (insert near the start) ---
            "Classification",
            "DetailedValueType",
            "Normalized Unit",
            "Absolute Unit",
            "Identifiers",
            # --- Processed structure columns ---
            "Category",         # Category determined by fixed pipeline (based on detailed classification name)
            "Attribute",        # Value or Unit
            "Code",             # SV-V, RV-Un, etc.
            "Value",            # The parsed value/unit string from fixed pipeline for this specific code/attribute
            # --- Original data & Sheet ---
            # Include other columns from the original input file dynamically?
            # For now, just add Sheet. Add others if needed.
            "Sheet",
            # --- Add other analysis columns at the end ---
             "SubValueCount", "ConditionCount",
             "HasRangeInMain", "HasMultiValueInMain", "HasRangeInCondition", "HasMultipleConditions",
             "MainUnits", "MainDistinctUnitCount", "MainUnitsConsistent",
             "ConditionUnits", "ConditionDistinctUnitCount", "ConditionUnitsConsistent",
             "OverallUnitConsistency", "ParsingErrorFlag", "SubValueUnitVariationSummary",
             "MainNumericValues", "ConditionNumericValues", "MainMultipliers", "ConditionMultipliers",
             "MainBaseUnits", "ConditionBaseUnits", "NormalizedMainValues", "NormalizedConditionValues",
             "MinNormalizedValue", "MaxNormalizedValue", "SingleUnitForAllSubs", "AllDistinctUnitsUsed"
        ]

        # Filter this list to only include columns that actually exist in df_merged
        existing_final_columns = [col for col in final_columns_order if col in df_merged.columns]

        # Add any columns from df_merged that were missed in the desired order list
        # (e.g., original columns from the input file)
        original_input_cols = [col for col in df_merged.columns if col not in existing_final_columns]

        # Combine existing desired order + original input cols + any remaining analysis cols
        final_ordered_cols = existing_final_columns + sorted(original_input_cols)

        # Ensure no duplicates in final list
        final_ordered_cols = list(dict.fromkeys(final_ordered_cols))


        st.write(f"DEBUG: Final column order: {final_ordered_cols}")
        df_final = df_merged[final_ordered_cols]

        # 6) Write final output
        df_final.to_excel(output_file, index=False)
        print(f"[✓] Final combined file saved to '{output_file}'")
        st.write(f"DEBUG: Final combined file saved to '{output_file}'. Shape: {df_final.shape}")
        return output_file # Return path on success

    except Exception as e:
        st.error(f"An unexpected error occurred during combine_results: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None



#############################################
# 6) STREAMLIT APP UI
#############################################

st.title("ACC Project - Unified Pipeline")

# Initialize mapping_df in session_state
if "mapping_df" not in st.session_state or st.session_state["mapping_df"] is None:
    try:
        st.session_state["mapping_df"] = download_mapping_file_from_github()
        # Immediately save to disk after download for pipeline use
        if st.session_state["mapping_df"] is not None:
             save_mapping_to_disk(st.session_state["mapping_df"], "mapping.xlsx")
        else:
             st.error("Failed to download mapping file. Cannot proceed.")
             st.stop()
    except Exception as e:
        st.error(f"Error initializing mapping file: {e}")
        st.session_state["mapping_df"] = None # Ensure it's None on error
        st.stop()


# Check if mapping_df is valid before proceeding
if st.session_state["mapping_df"] is None:
    st.error("Mapping data could not be loaded. Please check GitHub configuration and connection.")
    st.stop()
else:
    # Validate required columns after ensuring df is loaded
    required_cols = {"Base Unit Symbol", "Multiplier Symbol"}
    if not required_cols.issubset(st.session_state["mapping_df"].columns):
        st.error(f"Mapping file from GitHub must contain columns: {required_cols}. Please fix the file on GitHub.")
        # Allow viewing/managing but disable Get Pattern?
        # st.stop() # Or allow manage units? Let's allow manage units.
        pass


operation = st.selectbox("Select Operation", ["Get Pattern", "Manage Units"])

############################
# OPERATION: GET PATTERN
############################
if operation == "Get Pattern":
    # Re-check mapping file validity specifically for this operation
    if st.session_state["mapping_df"] is None or not required_cols.issubset(st.session_state["mapping_df"].columns):
         st.error("Cannot run 'Get Pattern' because the mapping file is missing or invalid. Please check/fix it via 'Manage Units' or on GitHub.")
         st.stop()


    st.header("Get Pattern")
    st.write("Upload an Excel file containing a 'Value' column for processing.")

    input_file = st.file_uploader("Upload Input Excel File", type=["xlsx"], key="pattern_uploader")

    if input_file:
        # Define filenames
        user_input_filename = "user_input_temp.xlsx"
        # processed_output_filename = "processed_output.xlsx" # Fixed pipeline returns df now
        analysis_output_filename = "detailed_analysis_output.xlsx"
        final_output_filename = "final_combined_output.xlsx"
        mapping_filename = "mapping.xlsx" # Local mapping file name

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.info("Reading input file...")
            progress_bar.progress(5)
            file_bytes = input_file.read()

            # Save user input bytes to disk for detailed analysis step if needed
            with open(user_input_filename, "wb") as f:
                 f.write(file_bytes)
            input_df_for_analysis = pd.read_excel(user_input_filename) # Read it back for analysis
            st.write(f"DEBUG: Read user input file. Shape: {input_df_for_analysis.shape}")


            # Step 1: Ensure mapping is on disk (should be handled at startup/download)
            status_text.info("Ensuring local mapping file is present...")
            progress_bar.progress(10)
            if not os.path.exists(mapping_filename):
                 save_mapping_to_disk(st.session_state["mapping_df"], mapping_filename)


            # Step 2: Run first pipeline (Fixed Processing)
            status_text.info("Running fixed processing pipeline...")
            progress_bar.progress(20)
            # This now returns a DataFrame
            processed_df = process_fixed_pipeline_bytes(file_bytes)
            if processed_df is None or processed_df.empty:
                st.error("Fixed Processing Pipeline did not return valid data. Aborting.")
                st.stop()
            progress_bar.progress(40)
            st.write(f"DEBUG: Fixed processing pipeline complete. Output shape: {processed_df.shape}")


            # Step 3: Run second pipeline (Detailed Analysis)
            status_text.info("Running detailed analysis pipeline...")
            progress_bar.progress(50)
            # detailed_analysis reads the user input df, uses mapping.xlsx, writes analysis_output_filename
            analysis_result_path = detailed_analysis(
                input_df=input_df_for_analysis, # Pass the DataFrame read earlier
                mapping_file=mapping_filename,
                output_file=analysis_output_filename
            )
            if analysis_result_path is None:
                st.error("Detailed Analysis Pipeline failed. Aborting.")
                st.stop()
            progress_bar.progress(80)
            st.write(f"DEBUG: Detailed analysis complete. Output saved to {analysis_result_path}")

            # Step 4: Combine Results
            status_text.info("Combining results...")
            progress_bar.progress(90)
            final_result_path = combine_results(
                processed_df=processed_df, # Pass the DataFrame from fixed pipeline
                analysis_file=analysis_output_filename,
                output_file=final_output_filename
            )
            if final_result_path is None:
                st.error("Combining results failed. Aborting.")
                st.stop()
            progress_bar.progress(100)
            status_text.success("Processing Complete!")


            # Step 5: Offer download
            try:
                with open(final_result_path, "rb") as fp:
                    final_bytes = fp.read()
                st.download_button(
                    label=f"Download Results ({os.path.basename(final_result_path)})",
                    data=final_bytes,
                    file_name=os.path.basename(final_result_path), # Use the actual output name
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error preparing download link: {e}")

            # Clean up temporary files (optional)
            # for f in [user_input_filename, analysis_output_filename, final_output_filename]:
            #     if os.path.exists(f): os.remove(f)

        except Exception as e:
            status_text.error(f"An error occurred during the 'Get Pattern' process: {e}")
            import traceback
            st.error(traceback.format_exc()) # Show detailed error in Streamlit logs/UI

############################
# OPERATION: MANAGE UNITS
############################
elif operation == "Manage Units":
    st.header("Manage Units (GitHub mapping file)")

    # Display current mapping from session state
    st.subheader("Current Mapping File (from GitHub / Session State)")
    if st.session_state["mapping_df"] is not None:
         st.dataframe(st.session_state["mapping_df"])
    else:
         st.warning("Mapping data not loaded.")
         if st.button("Retry Download from GitHub"):
              # Clear state and rerun to trigger download
              if "mapping_df" in st.session_state: del st.session_state["mapping_df"]
              st.rerun()


    # --- Add New Unit ---
    st.subheader("Add New Base Unit")
    with st.form("add_unit_form"):
        new_unit = st.text_input("Enter new Base Unit Symbol").strip()
        # Optional: Add multiplier symbol if needed for the file structure, though not used by current logic
        # new_multiplier = st.text_input("Associated Multiplier Symbol (optional, e.g., 'k')").strip()
        submit_new = st.form_submit_button("Add Unit to Local Session")

    if submit_new and new_unit:
        if st.session_state["mapping_df"] is not None:
            # Check if unit already exists
            if new_unit in st.session_state["mapping_df"]["Base Unit Symbol"].astype(str).unique():
                 st.warning(f"Unit '{new_unit}' already exists.")
            else:
                # Add new row. Ensure all required columns exist.
                new_row_data = {"Base Unit Symbol": new_unit, "Multiplier Symbol": None} # Add other cols if mapping has more
                # Ensure new row matches columns of existing DataFrame
                current_cols = st.session_state["mapping_df"].columns
                new_row_df_data = {col: [new_row_data.get(col)] for col in current_cols}
                new_row_df = pd.DataFrame(new_row_df_data)

                st.session_state["mapping_df"] = pd.concat(
                    [st.session_state["mapping_df"], new_row_df],
                    ignore_index=True
                )
                st.success(f"New unit '{new_unit}' added to the current session. Save to GitHub to persist.")
                st.rerun() # Rerun to update displayed dataframe
        else:
            st.error("Mapping data not available to add unit.")
    elif submit_new:
        st.error("Base Unit Symbol cannot be empty.")


    # --- Delete Unit ---
    st.subheader("Delete Base Unit")
    if st.session_state["mapping_df"] is not None and not st.session_state["mapping_df"].empty:
        existing_units = sorted(st.session_state["mapping_df"]["Base Unit Symbol"].dropna().astype(str).unique())
        if existing_units:
            to_delete = st.selectbox("Select a unit to delete from local session", ["--Select--"] + existing_units, key="delete_unit_select")

            if st.button("Delete Selected Unit from Local Session"):
                if to_delete != "--Select--":
                    before_shape = st.session_state["mapping_df"].shape
                    st.session_state["mapping_df"] = st.session_state["mapping_df"][
                        st.session_state["mapping_df"]["Base Unit Symbol"] != to_delete
                    ].reset_index(drop=True) # Reset index after deletion
                    after_shape = st.session_state["mapping_df"].shape
                    st.success(f"Unit '{to_delete}' deleted from the current session. (Rows before: {before_shape[0]}, after: {after_shape[0]}). Save to GitHub to persist.")
                    st.rerun() # Update display
                else:
                    st.warning("Please select a unit to delete.")
        else:
            st.info("No base units found in the current mapping to delete.")
    else:
        st.info("Mapping data is empty or not loaded.")


    # --- Save Changes ---
    st.subheader("Persist Changes")
    st.warning("Changes made above are only in the current session.")

    # Option to download updated file locally
    if st.session_state["mapping_df"] is not None:
        try:
            towrite = BytesIO()
            st.session_state["mapping_df"].to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button(
                label="Download Current Mapping File (Local Session)",
                data=towrite,
                file_name="session_mapping.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
             st.error(f"Error creating download link for session mapping: {e}")


    # Save changes back to GitHub
    if st.button("Save Current Session Mapping to GitHub"):
        if st.session_state["mapping_df"] is not None:
            st.info("Attempting to save changes to GitHub...")
            # Ensure required columns are still present before upload
            if not required_cols.issubset(st.session_state["mapping_df"].columns):
                 st.error(f"Cannot save to GitHub: Mapping file is missing required columns: {required_cols}.")
            else:
                success = update_mapping_file_on_github(st.session_state["mapping_df"])
                if success:
                    st.success("Mapping file updated on GitHub! The app will use the new version on next full refresh or run.")
                    # Clear local state to force re-download next time? Or assume success?
                    # del st.session_state["mapping_df"] # Force re-download on next action
                else:
                    st.error("Failed to update mapping file on GitHub. Check logs and permissions.")
        else:
            st.error("No mapping data in session to save.")
