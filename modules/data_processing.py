import re
import math
import statistics
import csv
import io
import uuid
import pandas as pd
from typing import List, Dict, Any, Tuple

# -----------------------------------------------------------------------------
# DATA PARSING & EXTRACTION
# -----------------------------------------------------------------------------

def extract_metadata(filename: str) -> Dict[str, Any]:
    """
    Extracts metadata (Sample, Temp, Field, State, etc.) from the filename.
    """
    meta = {
        "sample": "Sample",
        "temp": None,
        "field": None,
        "voltage": None,
        "state": None,
        "direction": "unknown",
        "meas_type": None
    }
    
    name_clean = filename.replace(".dat", "")
    
    # 1. Sample Name
    parts = re.split(r'[_\s]+', name_clean)
    if parts and parts[0]:
        meta["sample"] = parts[0]
        
    # Sample overrides
    if "SRO" in name_clean.upper() and meta["sample"] == "Sample": meta["sample"] = "SrRuO3"
    if "STO" in name_clean.upper() and meta["sample"] == "Sample": meta["sample"] = "SrTiO3"

    # 2. Temperature
    m_temp = re.search(r"(?:T=)?(\d+(?:\.\d+)?)\s*K", name_clean, re.IGNORECASE)
    if m_temp:
        meta["temp"] = float(m_temp.group(1))

    # 3. Magnetic Field
    m_field = re.search(r"(\d+(?:\.\d+)?)\s*(T|Oe|Tesla)", name_clean, re.IGNORECASE)
    if m_field:
        val = float(m_field.group(1))
        unit = m_field.group(2).lower()
        if "oe" in unit:
            meta["field"] = f"{val:g}Oe"
        else:
            meta["field"] = f"{val:g}T"

    # 4. Voltage / Current
    m_volt = re.search(r"(\d+(?:\.\d+)?)\s*(V|mV|uA|mA)", name_clean, re.IGNORECASE)
    if m_volt:
        meta["voltage"] = f"{float(m_volt.group(1)):g}{m_volt.group(2)}"

    # 5. State / Geometry
    states = []
    if re.search(r"Comp", name_clean, re.IGNORECASE): states.append("Comp")
    if re.search(r"Tens", name_clean, re.IGNORECASE): states.append("Tens")
    if re.search(r"IP|InPlane", name_clean, re.IGNORECASE): states.append("IP")
    if re.search(r"OOP|OutPlane", name_clean, re.IGNORECASE): states.append("OOP")
    if states:
        meta["state"] = " ".join(states)

    # 6. Direction
    if re.search(r"UP", name_clean, re.IGNORECASE):
        meta["direction"] = "UP"
    elif re.search(r"DOWN|DN", name_clean, re.IGNORECASE):
        meta["direction"] = "DOWN"
        
    # 7. Measurement Type
    if re.search(r"RT", name_clean, re.IGNORECASE):
        meta["meas_type"] = "RT"
    elif re.search(r"RH", name_clean, re.IGNORECASE):
        meta["meas_type"] = "RH"
        
    return meta

def generate_label(metadata: Dict[str, Any]) -> str:
    """Generates a clean label from extracted metadata."""
    components = []
    
    # Sample
    if metadata.get("sample"):
        components.append(metadata["sample"])
        
    # Measurement Type + Field
    if metadata.get("meas_type"):
        if metadata.get("field") and metadata["meas_type"] == "RT":
             components.append(f"RT{metadata['field']}")
        else:
             components.append(metadata["meas_type"])
             if metadata.get("field"): components.append(metadata["field"])
    elif metadata.get("field"):
        components.append(metadata["field"])
        
    if metadata.get("voltage"):
        components.append(metadata["voltage"])
        
    if metadata.get("state"):
        components.append(metadata["state"])
        
    if metadata.get("temp"):
        components.append(f"{int(math.ceil(metadata['temp']))}K")
        
    if metadata.get("direction") and metadata["direction"] != "unknown":
        components.append(metadata["direction"])
        
    return " ".join(components)

def choose_field_column(cols: List[str]) -> int:
    preferred = ["Magnetic Field (T)", "Magnetic Field (Oe)", "Field (T)", "Field (Oe)"]
    for p in preferred:
        if p in cols:
            return cols.index(p)
    
    hits = [(i, c) for i, c in enumerate(cols) if "field" in c.lower()]
    if hits:
        def score(item: Tuple[int, str]) -> Tuple[int, int, int]:
            _, c = item
            lc = c.lower()
            return (0 if "(t" in lc else 1, 0 if "(oe" in lc else 1, len(lc))
        hits.sort(key=score)
        return hits[0][0]
    return -1

def choose_temperature_column(cols: List[str]) -> int:
    preferred = ["Temperature (K)", "Temp (K)", "T (K)"]
    for p in preferred:
        if p in cols:
            return cols.index(p)
    
    hits = [(i, c) for i, c in enumerate(cols) if "temp" in c.lower()]
    if hits:
        return hits[0][0]
    return -1

def _resist_candidates(cols: List[str]) -> List[int]:
    return [i for i, c in enumerate(cols) if "resist" in c.lower()]

def _resist_tiebreak_key(colname: str) -> Tuple[int, int]:
    lc = colname.lower()
    # Prefer Resistance (0) over Resistivity (1) to match plot labels and avoid missing geometry data
    kind = 0 if "resistance" in lc else (1 if "resistivity" in lc else 2)
    bridge = 0 if "bridge 1" in lc else 1
    return (kind, bridge)

def choose_best_resistance_column(cols: List[str], rows: List[List[str]], max_rows: int = 4000) -> int:
    cand = _resist_candidates(cols)
    if not cand:
        return -1

    counts = {i: 0 for i in cand}
    seen = 0
    for row in rows:
        if not row:
            continue
        if len(row) < len(cols):
            row = row + [""] * (len(cols) - len(row))
        for i in cand:
            try:
                v = float(row[i])
                if math.isfinite(v):
                    counts[i] += 1
            except Exception:
                pass
        seen += 1
        if seen >= max_rows:
            break

    ranked = sorted(
        cand,
        key=lambda i: (-counts.get(i, 0),) + _resist_tiebreak_key(cols[i]) + (i,),
    )
    best = ranked[0]
    if counts.get(best, 0) == 0:
        return -1
    return best

def field_to_tesla(h: float, field_col_name: str) -> float:
    lc = field_col_name.lower()
    if "(oe" in lc or "oersted" in lc:
        return h * 1e-4
    return h

def parse_multivu_content(content: str, filename: str) -> Dict[str, Any]:
    lines = content.splitlines()

    data_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "[Data]":
            data_idx = i
            break
    if data_idx is None or data_idx + 1 >= len(lines):
        raise ValueError("No [Data] section found.")

    csv_block = "\n".join(lines[data_idx + 1 :])
    reader = csv.reader(io.StringIO(csv_block))
    header = next(reader, None)
    if not header:
        raise ValueError("Missing header after [Data].")

    cols = [c.strip() for c in header]
    rows = list(reader)

    # Heuristic Column Selection
    # We try to guess which columns correspond to Field and Resistance based on their names.
    # This allows the app to handle slightly different file formats automatically.
    field_i = choose_field_column(cols)
    if field_i < 0:
        raise ValueError("No magnetic field column found.")
    field_name = cols[field_i]

    resist_i = choose_best_resistance_column(cols, rows)
    if resist_i < 0:
        raise ValueError("No valid resistance column found.")
    resist_name = cols[resist_i]

    temp_i = cols.index("Temperature (K)") if "Temperature (K)" in cols else -1
    tvals: List[float] = []

    H_T: List[float] = []
    R: List[float] = []
    
    # Store full data for custom plotting
    full_data_rows = []
    skipped_count = 0

    for row in rows:
        if not row:
            continue
        if len(row) < len(cols):
            row = row + [""] * (len(cols) - len(row))
            
        # Convert row to float where possible for full dataframe
        clean_row = []
        for val in row:
            try:
                v = float(val)
                if math.isfinite(v):
                    clean_row.append(v)
                else:
                    clean_row.append(None)
            except:
                clean_row.append(None)
        full_data_rows.append(clean_row)

        try:
            h = float(row[field_i])
            r = float(row[resist_i])
        except Exception:
            skipped_count += 1
            continue

        if not (math.isfinite(h) and math.isfinite(r)):
            skipped_count += 1
            continue

        H_T.append(field_to_tesla(h, field_name))
        R.append(r)

        if temp_i >= 0:
            try:
                tv = float(row[temp_i])
                if math.isfinite(tv):
                    tvals.append(tv)
            except Exception:
                pass

    if not H_T:
        raise ValueError("No numeric data points found.")
        
    # Create Full DataFrame
    df_full = pd.DataFrame(full_data_rows, columns=cols)

    # Extract Metadata
    meta = extract_metadata(filename)
    
    # Fallback for Temperature if not in filename
    if meta['temp'] is None and tvals:
        meta['temp'] = float(statistics.median(tvals))
        
    # Generate Label
    label = generate_label(meta)

    return {
        "id": str(uuid.uuid4()),
        "fileName": filename,
        "label": label,
        "temperatureK": meta['temp'],
        "direction": meta['direction'],
        "fieldCol": field_name,
        "rCol": resist_name,
        "H_T": H_T,
        "R": R,
        "full_df": df_full,
        "metadata": meta,
        "skipped_rows": skipped_count
    }
