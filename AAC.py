# STREAMLIT SCRIPT - OPTION 2: REFINED UNIFIED PIPELINE
#############################################

import streamlit as st
import pandas as pd
import re
import os
import base64
import requests
import gc
from io import BytesIO
import io
import uuid # Use if original_index approach fails

#############################################
# HIDE GITHUB ICON & OTHER ELEMENTS (Keep as is)
#############################################
# ... (code from your script) ...
hide_button = """ <style> [data-testid="stBaseButton-header"] { display: none; } </style> """
st.markdown(hide_button, unsafe_allow_html=True)

#############################################
# 1) GITHUB HELPER FUNCTIONS (Keep as is)
#############################################
# --- download_mapping_file_from_github() --- (Keep as is)
# --- update_mapping_file_on_github() --- (Keep as is)
# ... (full functions from your script, including dummy versions if needed) ...
def download_mapping_file_from_github():
     # YOUR ACTUAL IMPLEMENTATION HERE (OR DUMMY)
     st.write("DEBUG: Placeholder - download_mapping_file_from_github")
     try:
        df = pd.DataFrame({ 'Base Unit Symbol': ['V', 'A', 'Ohm', 's', 'F', 'Hz'], 'Multiplier Symbol': [None, None, None, 'm', 'p', 'k']})
        save_mapping_to_disk(df, "mapping.xlsx")
        return df
     except Exception as e: st.error(f"Dummy DF Error: {e}"); return None

def save_mapping_to_disk(df, filename="mapping.xlsx"):
      # YOUR ACTUAL IMPLEMENTATION HERE
      st.write(f"DEBUG: Placeholder - save_mapping_to_disk to {filename}")
      try: df.to_excel(filename, index=False)
      except Exception as e: st.error(f"Dummy Save Error: {e}")

def update_mapping_file_on_github(df):
      # YOUR ACTUAL IMPLEMENTATION HERE
      st.write("DEBUG: Placeholder - update_mapping_file_on_github")
      return True # Simulate success

#############################################
# 2) GLOBAL MULTIPLIER & MAPPING READ (Keep as is)
#############################################
# --- MULTIPLIER_MAPPING --- (Keep as is)
# --- LOCAL_BASE_UNITS --- (Keep as is)
# --- save_mapping_to_disk() ---
# --- read_mapping_file() --- (Keep as is)
# ... (definitions from your script) ...
MULTIPLIER_MAPPING = { 'm': 1e-3, 'k': 1e3, 'M': 1e6, 'µ': 1e-6, 'n': 1e-9, 'p':1e-12, 'G':1e9}
LOCAL_BASE_UNITS = set()
def read_mapping_file(mapping_file_path: str):
     # YOUR ACTUAL IMPLEMENTATION HERE
     st.write(f"DEBUG: Placeholder - read_mapping_file from {mapping_file_path}")
     if not os.path.exists(mapping_file_path): raise FileNotFoundError(f"'{mapping_file_path}' not found")
     df = pd.read_excel(mapping_file_path); req = {'Base Unit Symbol', 'Multiplier Symbol'}
     if not req.issubset(df.columns): raise ValueError(f"Missing cols: {req}")
     units = set(df['Base Unit Symbol'].dropna().astype(str).unique()) | LOCAL_BASE_UNITS
     return units, MULTIPLIER_MAPPING

#############################################
# 3) CORE PARSING/CLASSIFICATION HELPERS (Keep required originals)
#############################################
# --- PASTE YOUR ORIGINAL WORKING HELPERS HERE ---
# Needs: classify_value_type_detailed, extract_identifiers_detailed,
# classify_sub_value, count_main_items, count_conditions,
# classify_main, classify_condition, extract_numeric_info_for_value,
# analyze_value_units, extract_numeric_and_unit_analysis, resolve_compound_unit,
# replace_numbers_keep_sign_all, remove_parentheses_detailed,
# split_outside_parens, safe_str, fix_exceptions, get_desired_order
# ... (Use same placeholders or real functions as in Option 1) ...
# Placeholder functions - REPLACE with your actual implementations
def extract_numeric_and_unit_analysis(token, base_units, multipliers_dict): # Placeholder
    if token.startswith("10kOhm"): return 10000.0, "k", "Ohm", 10000.0 * 1e3, False
    if token.startswith("5V"): return 5.0, "1", "V", 5.0, False
    if token == "V": return None, None, "V", None, False
    nm = re.match(r'([+\-±]?\d+(?:\.\d+)?)', token)
    if nm: return float(nm.group(1).replace('±','')), "1", None, float(nm.group(1).replace('±','')), False
    return None, None, None, None, True
def remove_parentheses_detailed(text: str) -> str: return re.sub(r'\([^)]*\)', '', text) # Placeholder
def extract_numeric_info(part_text, base_units, multipliers_dict): # Placeholder
     text = remove_parentheses_detailed(part_text).strip(); tokens = [text] # Simplified split
     if "10" in text: return {"numeric_values": [10.0], "multipliers": ["k"], "base_units": ["Ohm"], "normalized_values": [10000.0], "error_flags": [False], "type": "single"}
     else: return {"numeric_values": [], "multipliers": [], "base_units": [], "normalized_values": [], "error_flags": [], "type": "none"}
def extract_numeric_info_for_value(raw_value, base_units, multipliers_dict): # Placeholder
     main_part, cond_part = raw_value.split('@', 1) if '@' in raw_value else (raw_value, "")
     main_info = extract_numeric_info(main_part.strip(), base_units, multipliers_dict)
     cond_info = extract_numeric_info(cond_part.strip(), base_units, multipliers_dict)
     return {"main_numeric": main_info["numeric_values"], "main_multipliers": main_info["multipliers"], "main_base_units": main_info["base_units"], "normalized_main": main_info["normalized_values"], "main_errors": main_info["error_flags"], "condition_numeric": cond_info["numeric_values"], "condition_multipliers": cond_info["multipliers"], "condition_base_units": cond_info["base_units"], "normalized_condition": cond_info["normalized_values"], "condition_errors": cond_info["error_flags"]}
def safe_str(item, placeholder="None"): return str(item) if item is not None else placeholder # Placeholder
def analyze_unit_part(part_text, base_units, multipliers_dict): # Placeholder
     text = remove_parentheses_detailed(part_text).strip(); tokens = [text] # Simplified split
     units = ["Ohm"] if "Ohm" in text else (["V"] if "V" in text else [])
     distinct = set(u for u in units if u != "None")
     return {"units": units, "distinct_units": distinct, "is_consistent": len(distinct)<=1, "count": len(units), "type": "single"}
def analyze_value_units(raw_value, base_units, multipliers_dict): # Placeholder
     main_part, cond_part = raw_value.split('@', 1) if '@' in raw_value else (raw_value, "")
     main_analysis = analyze_unit_part(main_part.strip(), base_units, multipliers_dict)
     cond_analysis = analyze_unit_part(cond_part.strip(), base_units, multipliers_dict)
     all_distinct = main_analysis["distinct_units"].union(cond_analysis["distinct_units"])
     return {"main_units": main_analysis["units"], "main_distinct_units": main_analysis["distinct_units"], "main_units_consistent": main_analysis["is_consistent"], "main_unit_count": main_analysis["count"], "condition_units": cond_analysis["units"], "condition_distinct_units": cond_analysis["distinct_units"], "condition_units_consistent": cond_analysis["is_consistent"], "condition_unit_count": cond_analysis["count"], "all_distinct_units": all_distinct, "overall_consistent": len(all_distinct)<=1, "main_sub_analysis": "", "condition_sub_analysis": ""}
def resolve_compound_unit(normalized_unit_string, base_units, multipliers_dict): # Placeholder
     ua = analyze_value_units(normalized_unit_string, base_units, multipliers_dict); main_repr, cond_repr = ",".join(sorted(list(ua["main_distinct_units"]))), ",".join(sorted(list(ua["condition_distinct_units"]))); return f"{main_repr}@{cond_repr}" if main_repr and cond_repr else main_repr or cond_repr or "None"
def count_main_items(main_str: str) -> int: return len(split_outside_parens(main_str, [','])) if main_str else 0 # Placeholder
def count_conditions(cond_str: str) -> int: return len(split_outside_parens(cond_str, [','])) if cond_str else 0 # Placeholder
def classify_condition(cond_str: str) -> str: # Placeholder
    num_parts = count_conditions(cond_str); hr = any(re.search(r'\s+to\s+', part) for part in split_outside_parens(cond_str, [','])); return "" if num_parts==0 else ("Multiple Conditions" if num_parts>1 else ("Range Condition" if hr else "Single Condition"))
def classify_main(main_str: str) -> str: # Placeholder
    num_parts = count_main_items(main_str); hr = any(re.search(r'\s+to\s+', part) for part in split_outside_parens(main_str, [','])); isn = all(re.fullmatch(r'[+\-±]?\d+(?:\.\d+)?', part.strip()) for part in split_outside_parens(main_str, [','])); return "" if num_parts == 0 else ("Multi Value" if num_parts > 1 else ("Range Value" if hr else ("Number" if isn else "Single Value")))
def classify_sub_value(subval: str): # Placeholder
    main_part, cond_part = subval.split('@', 1) if '@' in subval else (subval, "")
    mc, cc = classify_main(main_part.strip()), classify_condition(cond_part.strip()); cls = f"{mc} with {cc}" if mc and cc else mc or "Invalid"
    return (cls, mc=="Range Value", mc=="Multi Value", cc=="Range Condition", cc=="Multiple Conditions", count_conditions(cond_part.strip()), count_main_items(main_part.strip()))
def extract_identifiers_detailed(text: str): return ', '.join(re.findall(r'\(([^()]*)\)', text)) # Placeholder
def classify_value_type_detailed(raw_value: str): # Placeholder - Same simplified as option 1
    if not raw_value: return ("", "", 0, 0, False, False, False, False, 0)
    ids = extract_identifiers_detailed(raw_value)
    subvals = [v.strip() for v in raw_value.split(',') if v.strip()] if ',' in raw_value and '@' not in raw_value else ([raw_value] if raw_value else [])
    if not subvals: subvals=[raw_value]
    all_res = [classify_sub_value(sv) for sv in subvals]
    fc = "Multiple Mixed" if len(subvals)>1 else all_res[0][0] if all_res else "Invalid"
    return (fc, ids, len(subvals), max(r[5] for r in all_res) if all_res else 0, any(r[1] for r in all_res), any(r[2] for r in all_res), any(r[3] for r in all_res), any(r[4] for r in all_res), max(r[6] for r in all_res) if all_res else 0)
def replace_numbers_keep_sign_all(s: str) -> str: return re.sub(r'([+-])?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', r'\1$', str(s)) # Placeholder
def split_outside_parens(text, delimiters): # Placeholder
    if not text: return []
    pattern = '|'.join(map(re.escape, delimiters)); return [s.strip() for s in re.split(pattern, text) if s.strip()]
def fix_exceptions(s): return s # Placeholder
def get_desired_order(): # Placeholder - same as Option 1
     return ["SN-V", "SN-U", "SV-V", "SV-U", "CX-V", "CX-U", "RV-Vn", "RV-Un", "RV-Vx", "RV-Ux", "SC-V", "SC-U", "RC-Vn", "RC-Un", "RC-Vx", "RC-Ux", "MC-V1", "MC-U1", "MC-V2", "MC-U2", "ERR-PROC", "ERR-CLS", "ERR-CHUNK"]

#############################################
# 4) UNIFIED PIPELINE CORE FUNCTION (MODIFIED FOR OPTION 2)
#############################################

# --- _classify_value (keep as is) ---
# --- _extract_details (keep as is) ---
# --- _build_analysis_dict (keep as is) ---
# --- PLACEHOLDER definitions for these are included below ---
def _classify_value(value_str: str) -> dict: # Placeholder
    try:
        (cls, ids, sv_count, cond_count, rng_main, multi_main, rng_cond, multi_cond, main_count) = classify_value_type_detailed(value_str)
        dvt = f"{cls} [M:{main_count}][C:{cond_count}] (x{sv_count})" if cls and cls not in ["Invalid Structure", "Empty"] else cls or "Unknown Classification"
        return {"Classification": cls or "Unknown", "DetailedValueType": dvt, "Identifiers": ids, "SubValueCount": sv_count, "ConditionCount": cond_count, "HasRangeInMain": rng_main, "HasMultiValueInMain": multi_main, "HasRangeInCondition": rng_cond, "HasMultipleConditions": multi_cond, "MainItemCount": main_count, "classification_error": False}
    except Exception as e: return {"Classification": "Classification Error", "DetailedValueType": f"Error: {e}", "Identifiers": "", "SubValueCount": 1, "ConditionCount": 0, "HasRangeInMain": False, "HasMultiValueInMain": False, "HasRangeInCondition": False, "HasMultipleConditions": False, "MainItemCount": 0, "classification_error": True }
def _extract_details(value_str: str, base_units: set, multipliers_dict: dict) -> dict: # Placeholder
     try:
         num_info = extract_numeric_info_for_value(value_str, base_units, multipliers_dict)
         ua = analyze_value_units(value_str, base_units, multipliers_dict)
         results = {"MainNumericValues": ", ".join(safe_str(x) for x in num_info["main_numeric"]), "ConditionNumericValues": ", ".join(safe_str(x) for x in num_info["condition_numeric"]), # etc...
                    "ParsingErrorFlag": any(num_info["main_errors"]) or any(num_info["condition_errors"]), "_normalized_all_numeric": [v for v in num_info["normalized_main"] + num_info["normalized_condition"] if isinstance(v, (int, float))],
                    "MainUnits": ", ".join(safe_str(x) for x in ua["main_units"]), "_all_distinct_units_set": ua["all_distinct_units"],
                    "Normalized Unit": replace_numbers_keep_sign_all(value_str), "Absolute Unit": resolve_compound_unit(value_str, base_units, multipliers_dict),
                    "extraction_error": False}
         return results
     except Exception as e: return {"ParsingErrorFlag": True, "_normalized_all_numeric":[], "_all_distinct_units_set": set(), "Normalized Unit": "Error", "Absolute Unit": "Error", "extraction_error": True}
def _build_analysis_dict(value_str:str, classification_data: dict, extraction_data: dict) -> dict: # Placeholder
     analysis = {"Original Value": value_str}; analysis.update(classification_data); analysis.update(extraction_data)
     all_norm_num = analysis.pop("_normalized_all_numeric", []); all_distinct_units = analysis.pop("_all_distinct_units_set", set())
     analysis["MinNormalizedValue"] = min(all_norm_num) if all_norm_num else None; analysis["MaxNormalizedValue"] = max(all_norm_num) if all_norm_num else None
     analysis["SingleUnitForAllSubs"] = (len(all_distinct_units) <= 1); analysis["AllDistinctUnitsUsed"] = ", ".join(sorted(list(all_distinct_units))) if all_distinct_units else ""
     # Add other keys from _extract_details with default values if missing
     for k in ["MainNumericValues", "ConditionNumericValues", "MainMultipliers","ConditionMultipliers", "MainBaseUnits", "ConditionBaseUnits", "NormalizedMainValues", "NormalizedConditionValues","ParsingErrorFlag","MainUnits", "MainDistinctUnitCount", "MainUnitsConsistent", "ConditionUnits","ConditionDistinctUnitCount", "ConditionUnitsConsistent", "OverallUnitConsistency","SubValueUnitVariationSummary", "Normalized Unit", "Absolute Unit"]:
         if k not in analysis: analysis[k] = "N/A" # Or appropriate default
     return analysis

# --- _get_codes_for_structure (copied from Option 1 Stage 1, used internally) ---
def _get_codes_for_structure(is_range_main, main_count, is_range_cond, cond_count, is_num_only): # Placeholder logic
     codes = []
     if is_range_main: codes.extend([["RV-Vn", "RV-Un"], ["RV-Vx", "RV-Ux"]])
     elif main_count > 0: codes.append(["SN-V", "SN-U"] if is_num_only else ["SV-V", "SV-U"])
     if is_range_cond: codes.extend([["RC-Vn", "RC-Un"], ["RC-Vx", "RC-Ux"]])
     elif cond_count == 1: codes.append(["SC-V", "SC-U"])
     elif cond_count > 1: codes.extend([[f"MC-V{i}", f"MC-U{i}"] for i in range(1, cond_count + 1)])
     if not codes and (main_count > 0 or cond_count > 0): codes.append(["UNK-V", "UNK-U"])
     return codes


# --- _generate_coded_rows (MODIFIED FOR OPTION 2) ---
def _generate_coded_rows(value_str: str, overall_classification_data: dict, extraction_data: dict, base_units: set, multipliers_dict: dict) -> list:
    """
    Generates the list of coded rows (long format).
    CRITICAL: Each row MUST include 'Chunk_Category' reflecting the classification of its specific chunk.
    """
    coded_rows = []
    value_str_clean = fix_exceptions(value_str.strip())

    # Use overall classification only to determine if splitting is needed
    sub_value_count = overall_classification_data.get("SubValueCount", 1)

    # Handle global errors identified earlier
    if overall_classification_data.get("classification_error"):
         return [{"Chunk_Category": overall_classification_data.get("Classification"), "Attribute": "Error", "Code": "ERR-CLS-U", "Value": overall_classification_data.get("DetailedValueType")}]
    if extraction_data.get("extraction_error"):
         return [{"Chunk_Category": overall_classification_data.get("Classification"), "Attribute": "Error", "Code": "ERR-EXT-U", "Value": "Numeric/Unit extraction failed"}]
    if overall_classification_data.get("Classification") == "Empty":
         return []

    if sub_value_count > 1:
         chunks = split_outside_parens(value_str_clean, [','])
         if len(chunks) != sub_value_count:
             st.warning(f"[Unified] Sub-value count mismatch for '{value_str_clean}'. Expected {sub_value_count}, got {len(chunks)}.")

         for idx, chunk in enumerate(chunks):
             chunk = chunk.strip()
             if not chunk: continue
             chunk_prefix = f"M{idx+1}-"

             try:
                 # === RE-CLASSIFY THE CHUNK ===
                 chunk_cls_data = _classify_value(chunk) # Classify this specific chunk
                 chunk_category = chunk_cls_data.get("Classification", "Unknown Chunk") #<<< GET CHUNK CATEGORY

                 if "Error" in chunk_category or chunk_category == "Unknown Chunk":
                      # Store error row WITH chunk category info
                      coded_rows.append({"Chunk_Category": chunk_category, "Attribute": "Error", "Code": f"{chunk_prefix}ERR-CHUNK-U", "Value": f"Could not classify chunk: {chunk}"})
                      continue

                 # === Re-extract numeric/unit for this chunk ===
                 # Needed to associate values correctly with chunk's structure
                 chunk_numeric_info = extract_numeric_info_for_value(chunk, base_units, multipliers_dict)
                 chunk_main_units = [u for u in chunk_numeric_info.get("main_base_units", []) if u and u != "None"]
                 is_chunk_num_only = bool(chunk_numeric_info.get("main_numeric")) and not bool(chunk_main_units)

                 # === Get codes based on CHUNK's structure ===
                 code_pairs = _get_codes_for_structure(
                      chunk_cls_data.get("HasRangeInMain", False), chunk_cls_data.get("MainItemCount", 0),
                      chunk_cls_data.get("HasRangeInCondition", False), chunk_cls_data.get("ConditionCount", 0),
                      is_chunk_num_only
                 )

                 # === Extract value/unit strings for CHUNK ===
                 # (Same complex logic as _extract_value_unit_pairs_for_coding in Option 1, but applied to chunk)
                 extracted_values_for_chunk = [] # Placeholder for extraction result like [(val,unit), (val,unit)...]
                 # --- Add complex extraction logic here based on chunk_numeric_info & chunk_cls_data ---
                 # Simplified Placeholder Extraction for Chunk:
                 main_vals = [safe_str(v) for v in chunk_numeric_info["main_numeric"]]
                 main_units = [safe_str(u) for u in chunk_numeric_info["main_base_units"]]
                 cond_vals = [safe_str(v) for v in chunk_numeric_info["condition_numeric"]]
                 cond_units = [safe_str(u) for u in chunk_numeric_info["condition_base_units"]]
                 if chunk_cls_data["HasRangeInMain"]: # Simplified range assumption
                     if len(main_vals)>=2: extracted_values_for_chunk.extend([(main_vals[0],main_units[0] if main_units else ""),(main_vals[1],main_units[-1] if main_units else "")])
                     else: extracted_values_for_chunk.append((main_vals[0], main_units[0] if main_units else "")) # Handle incomplete range?
                 elif main_vals: extracted_values_for_chunk.append((main_vals[0], main_units[0] if main_units else ""))
                 # Condition... similarly add based on chunk_cls_data structure and extracted cond_vals/units
                 if chunk_cls_data["HasRangeInCondition"]: # Simplified
                      if len(cond_vals)>=2: extracted_values_for_chunk.extend([(cond_vals[0],cond_units[0] if cond_units else ""),(cond_vals[1],cond_units[-1] if cond_units else "")])
                 elif cond_vals: extracted_values_for_chunk.extend([(cond_vals[i], cond_units[i] if i < len(cond_units) else (cond_units[0] if cond_units else "")) for i in range(len(cond_vals))])
                 # ----------------------------------------------

                 # === Generate rows for the chunk ===
                 if len(code_pairs) != len(extracted_values_for_chunk):
                       st.warning(f"[Unified] Code/Value count mismatch for chunk '{chunk}'. Codes: {len(code_pairs)}, Values: {len(extracted_values_for_chunk)}.")
                       coded_rows.append({"Chunk_Category": chunk_category, "Attribute": "Error", "Code": f"{chunk_prefix}ERR-MATCH-U", "Value": "Code/Value mismatch"})
                       continue

                 for i, (val_code, unit_code) in enumerate(code_pairs):
                       val_str, unit_str = extracted_values_for_chunk[i]
                       row_base = {
                            "Chunk_Category": chunk_category, #<<< STORE CHUNK CATEGORY
                            "Code": chunk_prefix + val_code, # Value code with prefix
                       }
                       coded_rows.append({**row_base, "Attribute": "Value", "Value": val_str or ""})
                       # Modify code for unit row
                       row_base["Code"] = chunk_prefix + unit_code # Unit code with prefix
                       coded_rows.append({**row_base, "Attribute": "Unit", "Value": unit_str or ""})

             except Exception as e:
                   st.error(f"[Unified] Error processing chunk '{chunk}' from '{value_str_clean}': {e}")
                   # Assign overall classification as fallback category? Or specific error category?
                   coded_rows.append({"Chunk_Category": "Chunk Processing Error", "Attribute": "Error", "Code": f"{chunk_prefix}ERR-PROC-U", "Value": str(e)})
    else:
        # Single top-level value (no comma splitting needed)
        try:
             chunk_category = overall_classification_data.get("Classification", "Unknown") # Category is the overall one here
             # --- Get codes based on OVERALL structure ---
             # (Re-use logic to check if main part numeric etc. based on extraction_data)
             main_vals_exist = extraction_data.get("MainNumericValues", "") not in ["", "Error"]
             main_units_exist = any(u.strip() not in ['None', ''] for u in extraction_data.get("MainBaseUnits", "").split(','))
             is_main_num_only = main_vals_exist and not main_units_exist

             code_pairs = _get_codes_for_structure(
                   overall_classification_data.get("HasRangeInMain", False), overall_classification_data.get("MainItemCount", 0),
                   overall_classification_data.get("HasRangeInCondition", False), overall_classification_data.get("ConditionCount", 0),
                   is_main_num_only
             )
             # --- Extract value/unit strings for the whole value string ---
             extracted_values = [] # Placeholder for list of (val,unit) pairs
             # --- Add complex extraction logic here based on extraction_data & overall_classification_data structure ---
             # Simplified Placeholder Extraction (similar to chunk version, using overall data):
             main_vals = [v.strip() for v in extraction_data.get("MainNumericValues","").split(',') if v.strip()]
             main_units = [u.strip() for u in extraction_data.get("MainBaseUnits","").split(',') if u.strip() and u.strip()!='None']
             cond_vals = [v.strip() for v in extraction_data.get("ConditionNumericValues","").split(',') if v.strip()]
             cond_units = [u.strip() for u in extraction_data.get("ConditionBaseUnits","").split(',') if u.strip() and u.strip()!='None']
             if overall_classification_data["HasRangeInMain"]:
                 if len(main_vals)>=2: extracted_values.extend([(main_vals[0],main_units[0] if main_units else ""),(main_vals[1],main_units[-1] if main_units else "")])
                 else: extracted_values.append((main_vals[0], main_units[0] if main_units else "")) # Incomplete range?
             elif main_vals: extracted_values.append((main_vals[0], main_units[0] if main_units else ""))
             # Condition...
             if overall_classification_data["HasRangeInCondition"]:
                  if len(cond_vals)>=2: extracted_values.extend([(cond_vals[0],cond_units[0] if cond_units else ""),(cond_vals[1],cond_units[-1] if cond_units else "")])
             elif cond_vals: extracted_values.extend([(cond_vals[i], cond_units[i] if i < len(cond_units) else (cond_units[0] if cond_units else "")) for i in range(len(cond_vals))])
             # -----------------------------------------------------------

             # --- Generate rows ---
             if len(code_pairs) != len(extracted_values):
                  st.warning(f"[Unified] Code/Value count mismatch for '{value_str_clean}'. Codes: {len(code_pairs)}, Values: {len(extracted_values)}.")
                  coded_rows.append({"Chunk_Category": chunk_category, "Attribute": "Error", "Code": "ERR-MATCH-U", "Value": "Code/Value mismatch"})
             else:
                  for i, (val_code, unit_code) in enumerate(code_pairs):
                      val_str, unit_str = extracted_values[i]
                      coded_rows.append({"Chunk_Category": chunk_category, "Attribute": "Value", "Code": val_code, "Value": val_str or ""})
                      coded_rows.append({"Chunk_Category": chunk_category, "Attribute": "Unit", "Code": unit_code, "Value": unit_str or ""})

        except Exception as e:
              st.error(f"[Unified] Error processing single value '{value_str_clean}': {e}")
              coded_rows.append({"Chunk_Category": "Value Processing Error", "Attribute": "Error", "Code": "ERR-PROC-U", "Value": str(e)})

    return coded_rows


# --- Main analysis function ---
def analyze_and_code_value(value_str: str, base_units: set, multipliers_dict: dict) -> (dict, list):
    """ Option 2: Performs ALL analysis and coding for a single value string."""
    value_str_clean = fix_exceptions(value_str.strip()) if value_str else ""
    if not value_str_clean:
         # Handle empty input (copied from Option 1 code)
        analysis_dict = {"Original Value": "", "Classification": "Empty", "DetailedValueType": "Empty", **{k:"--" for k in ["Identifiers", "SubValueCount", ...]}} # etc.
        coded_rows_list = []
        return analysis_dict, coded_rows_list

    # Overall Classification (needed by generate_coded_rows to know if split needed)
    classification_data = _classify_value(value_str_clean)
    # Extraction (runs once for whole string, needed by analysis AND code gen)
    extraction_data = _extract_details(value_str_clean, base_units, multipliers_dict)
    # Build Analysis Dict (based on overall classification and extraction)
    analysis_dict = _build_analysis_dict(value_str, classification_data, extraction_data)
    # Generate Coded Rows (Handles internal splitting/re-classification and assigns Chunk_Category)
    coded_rows_list = _generate_coded_rows(value_str_clean, classification_data, extraction_data, base_units, multipliers_dict)
    return analysis_dict, coded_rows_list

#############################################
# 5) BUFFER/COMBINE step integrated into Streamlit UI below
#############################################


#############################################
# 6) STREAMLIT APP UI (MODIFIED FOR OPTION 2 MERGE)
#############################################

st.title("ACC Project - Pipeline (Refined Unified)")

# --- Mapping File Initialization (Use @st.cache_data version) ---
@st.cache_data
def get_mapping_data(file_path="mapping.xlsx"):
     # ... (Use the definition from unified option 1 / Option 1 above) ...
     st.write(f"DEBUG: Placeholder - get_mapping_data from {file_path}")
     if not os.path.exists(file_path):
         downloaded_df = download_mapping_file_from_github(); save_mapping_to_disk(downloaded_df, file_path)
         if downloaded_df is None: st.stop()
     return read_mapping_file(file_path)
try: # ... (Rest of initialization block ) ...
     base_units, multipliers_dict = get_mapping_data()
     if base_units is None: st.stop()
     if "mapping_df" not in st.session_state or st.session_state.get("mapping_df") is None:
          if os.path.exists("mapping.xlsx"): st.session_state["mapping_df"] = pd.read_excel("mapping.xlsx")
          else: st.session_state["mapping_df"] = None
except Exception as e: st.error(f"Mapping init error: {e}"); st.stop()
mapping_loaded = base_units is not None and multipliers_dict is not None
if not mapping_loaded: st.stop()

operation = st.selectbox("Select Operation", ["Get Pattern", "Manage Units"])

############################
# OPERATION: GET PATTERN
############################
if operation == "Get Pattern":
    st.header("Get Pattern")
    st.write("Upload an Excel file containing a 'Value' column for processing.")
    input_file = st.file_uploader("Upload Input Excel File", type=["xlsx"], key="pattern_uploader_unified")

    if input_file:
        final_output_filename = f"processed_Unified_{input_file.name}"
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.info("Reading input file...")
            progress_bar.progress(5)
            try: # Read input
                 xls = pd.ExcelFile(input_file); input_sheets = xls.sheet_names
                 st.info(f"Found sheets: {input_sheets}. Processing sheet: '{input_sheets[0]}'")
                 input_df_orig = pd.read_excel(xls, sheet_name=input_sheets[0])
                 input_df = input_df_orig.reset_index().rename(columns={'index': 'original_index'})
            except Exception as read_e: st.error(f"Read Error: {read_e}"); st.stop()
            if 'Value' not in input_df_orig.columns: st.error("'Value' column missing."); st.stop()

            # Prepare for processing loop
            all_analysis_results = []; all_coded_rows = []
            total_rows = len(input_df)
            status_text.info(f"Starting processing for {total_rows} rows...")
            progress_bar.progress(10)

            # --- Single Processing Loop ---
            for i, row in input_df.iterrows():
                original_index = row['original_index']
                value_str = str(row.get('Value', '')).strip()
                try:
                    analysis_dict, coded_rows_list = analyze_and_code_value(value_str, base_units, multipliers_dict)
                    # Add identifier
                    analysis_dict['original_index'] = original_index
                    for code_row in coded_rows_list: code_row['original_index'] = original_index
                    all_analysis_results.append(analysis_dict)
                    all_coded_rows.extend(coded_rows_list)
                except Exception as row_e: # Handle row processing error
                    st.error(f"[Unified] Error processing row {original_index} (Value: '{value_str}'): {row_e}")
                    # Append placeholders if needed
                progress_percent = 10 + int(80 * (i + 1) / total_rows)
                progress_bar.progress(progress_percent)

            status_text.info("Processing complete. Finalizing output...")

            # --- Create DataFrames ---
            analysis_df = pd.DataFrame(all_analysis_results)
            coded_df = pd.DataFrame(all_coded_rows) # This df now contains 'Chunk_Category'

            # --- Refined Merge Strategy ---
            # 1. Prepare original data frame (with index)
            original_data_to_merge = input_df

            # 2. Merge coded rows onto original data
            df_merged = None
            if not coded_df.empty and 'original_index' in coded_df.columns and 'original_index' in original_data_to_merge.columns:
                 df_merged = original_data_to_merge.merge(coded_df, on='original_index', how='left', suffixes=('_orig','')) # Avoid suffix on coded cols if possible
            elif not coded_df.empty:
                  st.warning("Cannot merge coded data (missing index). Output may lack coded columns.")
                  df_merged = original_data_to_merge.copy() # Keep original if coded merge fails
            else: # coded_df is empty
                 st.info("No coded rows generated.")
                 df_merged = original_data_to_merge.copy()


            # 3. Merge analysis data onto the result
            if not analysis_df.empty and 'original_index' in analysis_df.columns and df_merged is not None and 'original_index' in df_merged.columns:
                # Define analysis columns to bring in, **excluding** overall 'Classification' maybe
                # And 'Original Value' as we keep the input 'Value'
                analysis_cols = [col for col in analysis_df.columns if col not in [
                                'original_index', 'Original Value', 'Classification', 'classification_error', 'extraction_error']]
                cols_to_merge = analysis_cols + ['original_index']

                df_merged = df_merged.merge(analysis_df[cols_to_merge], on='original_index', how='left', suffixes=('', '_analysis'))
            elif not analysis_df.empty:
                  st.warning("Cannot merge analysis data (missing index or prior merge failed). Output may lack analysis columns.")
                  # df_merged remains as is

            if df_merged is None: # Should not happen if input_df was valid
                st.error("Merge process failed unexpectedly."); st.stop()


            # 4. Cleanup, Rename, and Reorder Columns
            if 'Chunk_Category' in df_merged.columns:
                  # Rename Chunk_Category to Category (as in original output)
                  df_merged.rename(columns={'Chunk_Category': 'Category'}, inplace=True)
            else: # If coded_df was empty or merge failed, Category won't exist
                  df_merged['Category'] = 'N/A' # Add placeholder


            # Rename 'Value' from coded rows (if no suffix was added and didn't conflict)
            if 'Value' in df_merged.columns and 'Code' in df_merged.columns and 'Value_orig' in df_merged.columns: # Heuristic: If code column present and Value_orig, then Value is likely from coded part
                  df_merged.rename(columns={'Value': 'Attribute Value'}, inplace=True)
                  df_merged.rename(columns={'Value_orig': 'Value'}, inplace=True) # Rename original back to 'Value'
            elif 'Value' in df_merged.columns and 'Value_orig' in df_merged.columns: # Keep original value if no coded value evident
                 df_merged.drop(columns=['Value'], inplace=True)
                 df_merged.rename(columns={'Value_orig': 'Value'}, inplace=True)


            # Define final order - MUST include 'Category' (formerly Chunk_Category)
            preferred_order = [
                "Value",    # Original input value
                "Category", # Specific category from coded row
                "Attribute",
                "Code",
                "Attribute Value", # Renamed from coded row's 'Value'
                # --- Add Original Columns ---
                # --- Add Analysis Columns ---
                "DetailedValueType", "Normalized Unit", "Absolute Unit", "Identifiers",
                # etc. - use same list as option 1's combine_results
                "SubValueCount", "ConditionCount", "HasRangeInMain", "HasMultiValueInMain",
                "HasRangeInCondition", "HasMultipleConditions",
                "ParsingErrorFlag", "MinNormalizedValue", "MaxNormalizedValue", "AllDistinctUnitsUsed"
            ]

            # Dynamically build the final list (similar logic to Option 1 combine_results)
            original_cols = [col for col in input_df_orig.columns if col not in ['Value']] # Exclude original 'Value' here
            final_col_list = []
            current_cols = df_merged.columns.tolist()

            # Build list checking existence in df_merged, ensuring preferred order
            for col in preferred_order:
                if col in current_cols and col not in final_col_list: final_col_list.append(col)
            for col in original_cols:
                 if col in current_cols and col not in final_col_list: final_col_list.append(col)
            # Add remaining analysis columns not explicitly in preferred_order
            remaining_cols = [col for col in current_cols if col not in final_col_list and col !='original_index']
            final_col_list.extend(remaining_cols)

            try: # Select final columns
                 df_final = df_merged[final_col_list]
            except KeyError as e:
                 st.error(f"Final column selection error: {e}. Using default order.")
                 df_final = df_merged.drop(columns=['original_index'], errors='ignore')


            # --- Save and Offer Download ---
            progress_bar.progress(95)
            output_buffer = BytesIO()
            df_final.to_excel(output_buffer, index=False, engine='openpyxl')
            output_buffer.seek(0)
            progress_bar.progress(100)
            status_text.success("Processing Complete!")
            st.download_button(
                label=f"Download Results ({final_output_filename})",
                data=output_buffer, file_name=final_output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e: # General catch-all
            status_text.error(f"An unexpected error occurred during 'Get Pattern' (Unified): {e}")
            import traceback; st.error(traceback.format_exc())

############################
# OPERATION: MANAGE UNITS (Keep data editor version)
############################
elif operation == "Manage Units":
    # --- PASTE THE 'Manage Units' SECTION FROM THE UNIFIED CODE HERE ---
    # It uses st.data_editor and works independently of the chosen pipeline approach.
    st.header("Manage Units (GitHub mapping file)")
    st.write("Edit the table below...")
    # ... (rest of Manage Units UI using st.data_editor) ...
    # ... Use key="unit_editor_unified" if needed ...
    if "mapping_df" not in st.session_state or not isinstance(st.session_state.get("mapping_df"), pd.DataFrame):
         st.warning("Mapping data not loaded in session state for editing...") # Load attempt...
    if isinstance(st.session_state.get("mapping_df"), pd.DataFrame):
         # ... Check required cols ...
         edited_df = st.data_editor(st.session_state["mapping_df"], key="unit_editor_unified", num_rows="dynamic")
         # ... Validation logic ...
         validation_ok = True # Placeholder
         # ... Download button ...
         # ... Save button -> update_mapping_file_on_github(edited_df)...
         if st.button("Save Edited Mapping to GitHub"):
                if edited_df is not None and validation_ok:
                    st.info("Attempting to save changes to GitHub...")
                    success = update_mapping_file_on_github(edited_df) # Placeholder func defined above
                    if success:
                        st.success("Mapping file updated on GitHub!")
                        st.session_state["mapping_df"] = edited_df.copy()
                        get_mapping_data.clear(); st.rerun() # Clear cache and rerun
                    else: st.error("Failed to update mapping file on GitHub.")
                elif not validation_ok: st.error("Validation failed.")
                else: st.error("Edited data is not available.")
    else: st.warning("Mapping data could not be loaded for editing.")
