import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import io
import csv
import math
import re
import statistics
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple

# Set page config to wide mode by default
st.set_page_config(layout="wide", page_title="PPMS Analysis Tool")


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
    kind = 0 if "resistivity" in lc else (1 if "resistance" in lc else 2)
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
            continue

        if not (math.isfinite(h) and math.isfinite(r)):
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
        "metadata": meta
    }

# -----------------------------------------------------------------------------
# STREAMLIT APPLICATION
# -----------------------------------------------------------------------------

st.set_page_config(page_title="PPMS Analysis Tool", layout="wide")

def persistent_selectbox(label, options, persistent_key, **kwargs):
    """
    A wrapper around st.selectbox that persists its value across reruns 
    even when the widget key changes (e.g. due to uploader_key rotation).
    Uses a dedicated dictionary in session_state to store values safely.
    """
    # 1. Initialize persistent store if needed
    if 'persistent_values' not in st.session_state:
        st.session_state['persistent_values'] = {}
    
    store = st.session_state['persistent_values']
    uploader_key = st.session_state.get('uploader_key', 0)
    widget_key = f"{persistent_key}_{uploader_key}"
    
    # 2. Check if we have a value from the CURRENT widget (user interaction)
    if widget_key in st.session_state:
        store[persistent_key] = st.session_state[widget_key]
        
    # 3. Retrieve the value to display
    # Priority: Store > Default
    current_val = store.get(persistent_key)
    
    # 4. Validate current value against options
    if current_val not in options:
        # Fallback to default
        default_idx = kwargs.get('index', 0)
        if 0 <= default_idx < len(options):
            current_val = options[default_idx]
        elif options:
            current_val = options[0]
        else:
            current_val = None
        
        # Update store with valid default
        store[persistent_key] = current_val

    # 5. Calculate index for Streamlit
    idx = 0
    if current_val in options:
        idx = options.index(current_val)
        
    # 6. Clean kwargs
    kwargs.pop('index', None)
    kwargs.pop('key', None)
    
    # 7. Render widget
    selected_val = st.selectbox(label, options, index=idx, key=widget_key, **kwargs)
    
    # 8. Sync back to store (handles the case where we just initialized with default)
    store[persistent_key] = selected_val
    
    return selected_val

def persistent_input(widget_func, persistent_key, **kwargs):
    """
    Wrapper for value-based widgets (toggle, text, number, checkbox, color).
    Persists value in st.session_state['persistent_values'].
    Rotates key with uploader_key to force re-render with stored value.
    """
    if 'persistent_values' not in st.session_state:
        st.session_state['persistent_values'] = {}
    store = st.session_state['persistent_values']
    
    uploader_key = st.session_state.get('uploader_key', 0)
    widget_key = f"{persistent_key}_{uploader_key}"
    
    # 1. Check for interaction on CURRENT widget
    if widget_key in st.session_state:
        store[persistent_key] = st.session_state[widget_key]
        
    # 2. Get value to display (Store > Default)
    if persistent_key in store:
        kwargs['value'] = store[persistent_key]
        
    # 3. Render
    val = widget_func(key=widget_key, **kwargs)
    
    # 4. Sync
    store[persistent_key] = val
    return val

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'all_datasets': [],
        'uploader_key': 0,
        'batch_counter': 0,
        'plot_ids': [1],
        'next_plot_id': 2,
        'custom_batches': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Ensure Unique IDs for all datasets (Migration/Safety check)
seen_ids = set()
for d in st.session_state.all_datasets:
    if 'id' not in d or str(d['id']).lower().endswith('.dat') or d['id'] in seen_ids:
        d['id'] = str(uuid.uuid4())
    seen_ids.add(d['id'])

st.title("PPMS Analysis Tool")
st.markdown("Upload `.dat` files to visualize and analyze transport measurements (R-T, MR, I-V, etc.).")

# --- Sidebar: Data Manager ---
st.sidebar.header("Data Manager")

# File Uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload .dat files", 
    type=["dat"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
    help="Drag and drop files or folders here."
)

# Process Uploaded Files
if uploaded_files:
    is_batch = len(uploaded_files) > 1
    
    if is_batch:
        # Calculate next available batch ID
        existing_batch_ids = set(d.get('batch_id', 0) for d in st.session_state.all_datasets)
        existing_batch_ids.update(st.session_state.custom_batches.keys())
        existing_batch_ids.discard(0)
        
        batch_id = 1
        while batch_id in existing_batch_ids:
            batch_id += 1
        
        # Guess folder name from prefix
        filenames = [f.name for f in uploaded_files]
        prefix = os.path.commonprefix(filenames)
        
        if prefix:
            batch_name = f"📂 {prefix.strip('_- ')}"
        else:
            batch_name = f"📂 Batch Import #{batch_id}"
    else:
        batch_id = 0
        batch_name = "📄 File by file import"
    
    new_files_count = 0
    for uploaded_file in uploaded_files:
        if any(d['fileName'] == uploaded_file.name for d in st.session_state.all_datasets):
            continue
            
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            data = parse_multivu_content(content, uploaded_file.name)
            
            data['batch_id'] = batch_id
            data['batch_name'] = batch_name 
            
            st.session_state.all_datasets.append(data)
            new_files_count += 1
        except Exception as e:
            st.error(f"Error parsing {uploaded_file.name}: {e}")
    
    if new_files_count > 0:
        st.sidebar.success(f"Added {new_files_count} new files.")
        st.session_state.uploader_key += 1
        st.rerun()

# Folder Management
organize_mode = True

# --- Callbacks ---
def create_folder_callback():
    new_name = st.session_state.get("new_folder_name_input", "New Folder")
    
    # Calculate next available batch ID
    existing_batch_ids = set(d.get('batch_id', 0) for d in st.session_state.all_datasets)
    existing_batch_ids.update(st.session_state.custom_batches.keys())
    existing_batch_ids.discard(0)
    
    new_id = 1
    while new_id in existing_batch_ids:
        new_id += 1
        
    st.session_state.custom_batches[new_id] = f"📂 {new_name}"

def move_file_callback(file_id, target_bid, target_name):
    for d in st.session_state.all_datasets:
        if d['id'] == file_id:
            d['batch_id'] = target_bid
            d['batch_name'] = target_name
            break
    
    # Force refresh of selection state
    for key in list(st.session_state.keys()):
        if key.startswith("sel_"):
            if isinstance(st.session_state[key], list):
                st.session_state[key] = list(st.session_state[key])

def delete_file_callback(file_id):
    # Find the file name before deleting
    file_to_delete = None
    for d in st.session_state.all_datasets:
        if d['id'] == file_id:
            file_to_delete = d['fileName']
            break
            
    # Remove from dataset
    st.session_state.all_datasets = [d for d in st.session_state.all_datasets if d['id'] != file_id]
    
    # Clean up selections in plots
    for key in list(st.session_state.keys()):
        if key.startswith("sel_"):
            current_selection = st.session_state[key]
            if isinstance(current_selection, list):
                if file_to_delete:
                    new_selection = [f for f in current_selection if f != file_to_delete]
                else:
                    new_selection = current_selection
                st.session_state[key] = new_selection

def delete_batch_callback(batch_id):
    # Identify files to be deleted
    files_to_delete = [d['fileName'] for d in st.session_state.all_datasets if d.get('batch_id') == batch_id]

    # Delete files in the batch
    st.session_state.all_datasets = [d for d in st.session_state.all_datasets if d.get('batch_id') != batch_id]
    
    # Delete the batch entry
    if 'custom_batches' in st.session_state and batch_id in st.session_state.custom_batches:
        del st.session_state.custom_batches[batch_id]

    # Clean up selections in plots
    if files_to_delete:
        for key in list(st.session_state.keys()):
            if key.startswith("sel_"):
                current_selection = st.session_state[key]
                if isinstance(current_selection, list):
                    new_selection = [f for f in current_selection if f not in files_to_delete]
                    st.session_state[key] = new_selection

def move_files_batch_callback(file_ids, target_bid, target_name):
    """Move multiple files to a target folder."""
    for d in st.session_state.all_datasets:
        if d['id'] in file_ids:
            d['batch_id'] = target_bid
            d['batch_name'] = target_name
            
    # Force refresh of selection state
    for key in list(st.session_state.keys()):
        if key.startswith("sel_"):
            if isinstance(st.session_state[key], list):
                st.session_state[key] = list(st.session_state[key])

def delete_files_batch_callback(file_ids):
    """Delete multiple files."""
    files_to_delete = [d['fileName'] for d in st.session_state.all_datasets if d['id'] in file_ids]
    
    # Remove from dataset
    st.session_state.all_datasets = [d for d in st.session_state.all_datasets if d['id'] not in file_ids]
    
    # Clean up selections in plots
    for key in list(st.session_state.keys()):
        if key.startswith("sel_"):
            current_selection = st.session_state[key]
            if isinstance(current_selection, list):
                new_selection = [f for f in current_selection if f not in files_to_delete]
                st.session_state[key] = new_selection

# Create New Folder UI
with st.sidebar.popover("➕ Create New Folder", width='stretch'):
    st.text_input("Folder Name", "New Folder", key="new_folder_name_input")
    st.button("Create", width='stretch', on_click=create_folder_callback)

def rename_batch_callback(batch_id, old_name):
    new_name = st.session_state.get(f"rename_{batch_id}")
    
    if not new_name or new_name == old_name:
        return

    # Update files
    for d in st.session_state.all_datasets:
        if d.get('batch_id') == batch_id:
            d['batch_name'] = new_name
    
    # Update custom batches
    if 'custom_batches' in st.session_state and batch_id in st.session_state.custom_batches:
        st.session_state.custom_batches[batch_id] = new_name
        
    # Update Plot Filters
    for key in list(st.session_state.keys()):
        if key.startswith("batch_filter_") or key.startswith("batch_filter_cust_"):
            if st.session_state[key] == old_name:
                st.session_state[key] = new_name

# --- Dialogs for File Management ---
dialog_decorator = None
if hasattr(st, "dialog"):
    dialog_decorator = st.dialog
elif hasattr(st, "experimental_dialog"):
    dialog_decorator = st.experimental_dialog

if dialog_decorator:
    @dialog_decorator("Manage File")
    def manage_file_dialog(file_data: Dict[str, Any], current_batch_id: int, all_batches_info: Dict[int, Dict[str, Any]]):
        st.write(f"**File:** {file_data['fileName']}")
        
        st.markdown("---")
        st.subheader("Move File")
        
        target_options = {bid: info['name'] for bid, info in all_batches_info.items() if bid != current_batch_id}
        
        if current_batch_id != 0:
             target_options[0] = "📄 File by file import"
        
        if target_options:
            target_bid = st.selectbox(
                "Select Target Folder", 
                options=list(target_options.keys()), 
                format_func=lambda x: target_options[x], 
                key=f"dlg_mv_sel_{file_data['id']}"
            )
            
            target_name = target_options[target_bid]
            if st.button("Confirm Move", key=f"dlg_mv_btn_{file_data['id']}", type="primary"):
                move_file_callback(file_data['id'], target_bid, target_name)
                st.rerun()
        else:
            st.info("No other folders to move to.")

        st.markdown("---")
        st.subheader("Delete File")
        st.warning("This action cannot be undone.")
        if st.button("Confirm Delete", key=f"dlg_rm_{file_data['id']}", type="primary"):
            delete_file_callback(file_data['id'])
            st.rerun()

if dialog_decorator:
    @dialog_decorator("Batch Actions")
    def manage_batch_dialog(datasets: List[Dict[str, Any]], batches: Dict[int, Dict[str, Any]]):
        st.write("Select multiple files to move or delete them at once.")
        
        # 1. Select Files
        # Create a mapping of ID -> Display Name
        file_options = {d['id']: f"{d['fileName']} ({d.get('batch_name', 'Unknown')})" for d in datasets}
        
        selected_ids = st.multiselect(
            "Select Files",
            options=list(file_options.keys()),
            format_func=lambda x: file_options[x],
            key="dlg_batch_action_files",
            placeholder="Choose files..."
        )
        
        if selected_ids:
            st.markdown("---")
            st.subheader("Move Selected")
            
            # Target options (All folders)
            target_options = {bid: info['name'] for bid, info in batches.items()}
            # Ensure "File by file" is available
            if 0 not in target_options:
                 target_options[0] = "📄 File by file import"
                 
            target_bid = st.selectbox(
                "Target Folder", 
                options=list(target_options.keys()), 
                format_func=lambda x: target_options[x], 
                key="dlg_batch_move_target"
            )
            
            target_name = target_options[target_bid]
            if st.button("Move Files", key="dlg_batch_move_btn", type="primary"):
                move_files_batch_callback(selected_ids, target_bid, target_name)
                st.rerun()
                
            st.markdown("---")
            st.subheader("Delete Selected")
            st.warning(f"Are you sure you want to delete {len(selected_ids)} files?")
            
            if st.button(f"Delete {len(selected_ids)} Files", key="dlg_batch_delete_btn", type="primary"):
                delete_files_batch_callback(selected_ids)
                st.rerun()
        else:
            st.info("Please select at least one file.")

def render_file_actions(file_data: Dict[str, Any], current_batch_id: int, all_batches_info: Dict[int, Dict[str, Any]]):
    """Helper to render the Move/Delete actions for a file."""
    # Ensure ID is a string
    file_id = str(file_data.get('id', uuid.uuid4()))
    
    # Use a Dialog if available
    if dialog_decorator:
        if st.button("⚙️", key=f"btn_manage_{file_id}", help="Manage File"):
            manage_file_dialog(file_data, current_batch_id, all_batches_info)
    else:
        # Fallback for older Streamlit versions
        with st.popover("⋮", help="Manage File"):
            st.markdown("**Move File**")
            
            target_options = {bid: info['name'] for bid, info in all_batches_info.items() if bid != current_batch_id}
            
            # Ensure "File by file" (0) is available if we are in a batch
            if current_batch_id != 0:
                 target_options[0] = "📄 File by file import"
            
            if target_options:
                c_dest, c_go = st.columns([0.7, 0.3], vertical_alignment="bottom")
                with c_dest:
                    target_bid = st.selectbox(
                        "Target", 
                        options=list(target_options.keys()), 
                        format_func=lambda x: target_options[x], 
                        key=f"mv_sel_{file_id}", 
                        label_visibility="collapsed"
                    )
                with c_go:
                    target_name = target_options[target_bid]
                    st.button(
                        "Move", 
                        key=f"mv_btn_{file_id}", 
                        width='stretch', 
                        on_click=move_file_callback, 
                        args=(file_data['id'], target_bid, target_name)
                    )
            else:
                st.info("No other folders.")

            st.markdown("<hr style='margin: 5px 0; border: none; border-top: 1px solid #f0f0f0;'>", unsafe_allow_html=True)
            st.button(
                "Delete File", 
                key=f"rm_{file_id}", 
                type="primary", 
                width='stretch', 
                on_click=delete_file_callback, 
                args=(file_data['id'],)
            )

# Data Management & Explorer
datasets = st.session_state.all_datasets

# Group by Batch
batches = {}
batch_order = []

# 1. Collect existing batches from files
for d in datasets:
    bid = d.get('batch_id', 0)
    if bid == 0:
        bname = "📄 File by file import"
    else:
        bname = d.get('batch_name', f"📂 Batch #{bid}")
        
    if bid not in batches:
        batches[bid] = {'name': bname, 'files': []}
        if bid not in batch_order:
            batch_order.append(bid)
    batches[bid]['files'].append(d)

# 2. Add empty custom batches
if 'custom_batches' in st.session_state:
    for bid, bname in st.session_state.custom_batches.items():
        if bid not in batches:
            batches[bid] = {'name': bname, 'files': []}
            batch_order.append(bid)

if datasets or batches:
    # Ensure unique labels
    label_counts = {}
    for d in datasets:
        l = d['label']
        label_counts[l] = label_counts.get(l, 0) + 1

    for d in datasets:
        l = d['label']
        if label_counts[l] > 1:
            d['label'] = f"{l} ({d['fileName']})"
    
    # Explorer UI
    st.sidebar.markdown("### Loaded Data")
    
    # Clear All Button
    c_info, c_clear = st.sidebar.columns([0.4, 0.6])
    with c_info:
        st.write(f"**Total Files:** {len(datasets)}")
    with c_clear:
        if st.button("🗑️ Clear All", help="Remove all loaded data", width='stretch'):
            st.session_state.all_datasets = []
            st.session_state.batch_counter = 0
            st.session_state.custom_batches = {}
            st.rerun()

    # --- Batch Actions Button ---
    if dialog_decorator:
        if st.sidebar.button("⚡ Batch Actions", width='stretch'):
            manage_batch_dialog(datasets, batches)
    else:
        # Fallback for older versions
        with st.sidebar.expander("⚡ Batch Actions", expanded=False):
            # 1. Select Files
            file_options = {d['id']: f"{d['fileName']} ({d.get('batch_name', 'Unknown')})" for d in datasets}
            
            selected_ids = st.multiselect(
                "Select Files",
                options=list(file_options.keys()),
                format_func=lambda x: file_options[x],
                key="batch_action_files",
                placeholder="Choose files..."
            )
            
            if selected_ids:
                st.markdown("---")
                # Move Action
                st.caption("Move Selected")
                
                target_options = {bid: info['name'] for bid, info in batches.items()}
                if 0 not in target_options:
                     target_options[0] = "📄 File by file import"
                     
                c_dest, c_go = st.columns([0.7, 0.3], vertical_alignment="bottom")
                with c_dest:
                    target_bid = st.selectbox(
                        "Target Folder", 
                        options=list(target_options.keys()), 
                        format_func=lambda x: target_options[x], 
                        key="batch_move_target",
                        label_visibility="collapsed"
                    )
                with c_go:
                    target_name = target_options[target_bid]
                    st.button(
                        "Move", 
                        key="batch_move_btn", 
                        width='stretch', 
                        on_click=move_files_batch_callback, 
                        args=(selected_ids, target_bid, target_name)
                    )
                    
                st.markdown("---")
                # Delete Action
                if st.button(f"Delete {len(selected_ids)} Files", key="batch_delete_btn", type="primary", width='stretch', on_click=delete_files_batch_callback, args=(selected_ids,)):
                    pass
            else:
                st.caption("Select files to see actions.")

    # Display "File by file" first if it exists
    if 0 in batches:
        with st.sidebar.expander(batches[0]['name'], expanded=True):
            for d in batches[0]['files']:
                c_name, c_act = st.columns([0.85, 0.15])
                with c_name:
                    st.text(f"📄 {d['fileName']}")
                with c_act:
                    render_file_actions(d, 0, batches)
    
    # Display other batches
    for bid in batch_order:
        if bid == 0: continue
        
        b_name = batches[bid]['name']
        with st.sidebar.expander(b_name, expanded=True):
            # Rename Feature
            col_ren, col_del = st.columns([0.85, 0.15])
            with col_ren:
                st.text_input("Folder Name", value=b_name, key=f"rename_{bid}", label_visibility="collapsed", on_change=rename_batch_callback, args=(bid, b_name))
            with col_del:
                st.button("🗑️", key=f"del_batch_{bid}", help="Delete Folder", on_click=delete_batch_callback, args=(bid,))
            
            for d in batches[bid]['files']:
                c_name, c_act = st.columns([0.85, 0.15])
                with c_name:
                    st.text(f"📄 {d['fileName']}")
                with c_act:
                    render_file_actions(d, bid, batches)
    
    st.sidebar.caption("⚠️ Refreshing the page will clear the data.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
**Author :** Amir MEDDAS  
*LPS - Laboratoire de Physique des Solides*
*C2N - Centre de Nanosciences et de Nanotechnologies*    
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/amir-meddas-80876424b/)
""")

else:
    st.sidebar.info("Upload files to begin.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
**Author :** Amir MEDDAS  
*LPS - Laboratoire de Physique des Solides*                          
*C2N - Centre de Nanosciences et de Nanotechnologies*  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/amir-meddas-80876424b/)
""")
    st.stop()

# --- Sidebar: Footer ---
# (Moved to top)

# -----------------------------------------------------------------------------
# PLOTTING INTERFACE
# -----------------------------------------------------------------------------

def add_plot_callback():
    existing_ids = set(st.session_state.plot_ids)
    new_id = 1
    while new_id in existing_ids:
        new_id += 1
    st.session_state.plot_ids.append(new_id)

def remove_plot_callback(plot_id_str):
    pid = int(plot_id_str)
    if pid in st.session_state.plot_ids:
        st.session_state.plot_ids.remove(pid)
        
        # Cleanup session state for this plot
        keys_to_del = []
        for key in st.session_state.keys():
            if key.endswith(f"_{pid}") or f"_{pid}_" in key:
                keys_to_del.append(key)
        
        for key in keys_to_del:
            del st.session_state[key]

def duplicate_plot_callback(plot_id):
    existing_ids = set(st.session_state.plot_ids)
    new_id = 1
    while new_id in existing_ids:
        new_id += 1
    st.session_state.plot_ids.append(new_id)
    
    # Copy state
    for key in list(st.session_state.keys()):
        # Exclude buttons
        if any(x in key for x in ["dup_", "add_btn_", "del_btn_"]): continue
        
        # Case 1: Key ends with _{plot_id} (Standard widgets)
        if key.endswith(f"_{plot_id}"):
            base = key[:-len(plot_id)] # remove old id (keep the underscore)
            new_key = f"{base}{new_id}"
            st.session_state[new_key] = st.session_state[key]
        
        # Case 2: Key contains _{plot_id}_ (Dynamic widgets like legends: leg_{plot_id}_{file_id})
        elif f"_{plot_id}_" in key:
            new_key = key.replace(f"_{plot_id}_", f"_{new_id}_")
            st.session_state[new_key] = st.session_state[key]

def get_batch_map(datasets: List[Dict[str, Any]], custom_batches: Dict[int, str] = None) -> Dict[Any, str]:
    """Returns a mapping of batch_id to batch_name."""
    unique_batches = {}
    
    # 1. From Datasets
    for d in datasets:
        bid = d.get('batch_id', 0)
        bname = d.get('batch_name', 'Unknown')
        unique_batches[bid] = bname
        
    # 2. From Custom Batches (Empty folders)
    if custom_batches:
        for bid, bname in custom_batches.items():
            unique_batches[bid] = bname
    
    return unique_batches

def create_plot_interface(plot_id: str, available_datasets: List[Dict[str, Any]], width: int, height: int) -> Optional[go.Figure]:
    """Creates a self-contained plotting interface and returns the figure."""
    
    with st.container(border=True):
        # Header with Actions
        c_head_title, c_head_actions = st.columns([0.7, 0.3], vertical_alignment="center")
        
        with c_head_title:
            # Editable Plot Name
            c_h_text, c_h_edit = st.columns([0.8, 0.2], vertical_alignment="center")
            with c_h_text:
                plot_name = st.session_state.get(f"pname_{plot_id}", f"Plot {plot_id}")
                st.markdown(f"<h3 style='margin: 0; padding: 0; line-height: 1.5;'>{plot_name}</h3>", unsafe_allow_html=True)
            with c_h_edit:
                with st.popover("✏️", help="Rename Plot", width='stretch'):
                    st.text_input("Name", value=plot_name, key=f"pname_{plot_id}")
        
        with c_head_actions:
            # Action Buttons
            b_add, b_rem, b_dup = st.columns(3)
            with b_add:
                st.button("➕", key=f"add_btn_{plot_id}", help="Add a new plot", on_click=add_plot_callback, width='stretch')
            with b_rem:
                st.button("➖", key=f"del_btn_{plot_id}", help="Remove this plot", on_click=remove_plot_callback, args=(plot_id,), width='stretch')
            with b_dup:
                st.button("📋", key=f"dup_{plot_id}", help="Duplicate this plot", on_click=duplicate_plot_callback, args=(plot_id,), width='stretch')
        
        # Row 0: Analysis Mode
        analysis_mode = persistent_selectbox(
            "Analysis Mode",
            ["Custom Columns", "Standard MR Analysis", "Standard R-T Analysis"],
            index=0,
            persistent_key=f"mode_{plot_id}"
        )
        
        # --- Unified File Selection ---
        # Filter by Folder (Batch)
        batch_map = get_batch_map(available_datasets, st.session_state.get('custom_batches', {}))
        
        # Options: "ALL" + sorted IDs
        batch_ids = sorted(batch_map.keys())
        options = ["ALL"] + batch_ids
        
        def format_batch(option):
            if option == "ALL":
                return "All Folders"
            return batch_map.get(option, f"Batch {option}")

        # Persistent State Key
        persistent_batch_key = f"batch_filter_{plot_id}"
        
        # Migration/Validation Logic
        current_val = st.session_state.get(persistent_batch_key)
        
        # If value is a string (old name) or invalid, try to migrate or reset
        if (isinstance(current_val, str) and current_val != "ALL" and current_val not in options) or (current_val not in options and current_val is not None):
            found_id = "ALL"
            if isinstance(current_val, str):
                for bid, bname in batch_map.items():
                    if bname == current_val:
                        found_id = bid
                        break
            st.session_state[persistent_batch_key] = found_id
        
        if st.session_state.get(persistent_batch_key) not in options:
             st.session_state[persistent_batch_key] = "ALL"
             
        # Dynamic Widget Key (Rotates with Uploader Key to force re-render on new data)
        # This prevents the widget from resetting to empty when options change significantly
        widget_batch_key = f"batch_filter_{plot_id}_{st.session_state.uploader_key}"
        
        # Sync Widget -> State
        if widget_batch_key in st.session_state:
            st.session_state[persistent_batch_key] = st.session_state[widget_batch_key]
            
        selected_batch_id = st.selectbox(
            "Filter by Folder", 
            options, 
            format_func=format_batch,
            index=options.index(st.session_state[persistent_batch_key]),
            key=widget_batch_key
        )
        
        if selected_batch_id != "ALL":
            filtered_datasets = [d for d in available_datasets if d.get('batch_id') == selected_batch_id]
        else:
            filtered_datasets = available_datasets

        # Use Raw Filenames for Selection
        # Options must include currently selected files even if they are filtered out
        filtered_filenames = [d['fileName'] for d in filtered_datasets]
        
        # Validate and Fix File Selection in Session State
        persistent_sel_key = f"sel_{plot_id}"
        current_selection = st.session_state.get(persistent_sel_key, [])
        
        # 1. Remove files that don't exist anymore globally
        all_existing_files = {d['fileName'] for d in available_datasets}
        valid_selection = [f for f in current_selection if f in all_existing_files]
        
        if len(valid_selection) != len(current_selection):
            st.session_state[persistent_sel_key] = valid_selection
            current_selection = valid_selection
        
        # 2. Ensure options include the selection
        combined_options = sorted(list(set(filtered_filenames + current_selection)))
        
        # Dynamic Widget Key for Multiselect
        widget_sel_key = f"sel_{plot_id}_{st.session_state.uploader_key}"
        
        # Sync Widget -> State
        if widget_sel_key in st.session_state:
            st.session_state[persistent_sel_key] = st.session_state[widget_sel_key]
        
        selected_filenames = st.multiselect(
            f"Select Curves for Plot {plot_id}", 
            options=combined_options,
            default=current_selection,
            key=widget_sel_key
        )
        
        # Map back to datasets
        selected_datasets = [d for d in available_datasets if d['fileName'] in selected_filenames]
        
        # Row 1: Axes & Style
        c1, c2 = st.columns([2, 1])
        
        # Variables for Custom Mode
        custom_x_col = None
        custom_y_col = None
        
        if analysis_mode == "Standard MR Analysis":
            # R0 Method
            r0_method = persistent_selectbox(
                "R0 Calculation Method",
                ["Closest to 0T", "Mean within Window", "First Point"],
                index=0,
                persistent_key=f"r0_meth_{plot_id}"
            )
            r0_window = 0.01
            if r0_method == "Mean within Window":
                r0_window = st.number_input("Zero Field Window (T)", value=0.01, step=0.005, format="%.4f", key=f"r0_win_{plot_id}")

            with c1:
                y_axis_mode = persistent_selectbox(
                    "Y-Axis Mode",
                    ["Magnetoresistance (MR %)", "Resistance (Ω)", "Normalized (R/R0)", "Derivative (dR/dH)"],
                    index=0,
                    persistent_key=f"y_mode_{plot_id}"
                )
            with c2:
                x_axis_unit = persistent_selectbox(
                    "X-Axis Unit",
                    ["Tesla (T)", "Oersted (Oe)"],
                    index=0,
                    persistent_key=f"x_unit_{plot_id}"
                )
        elif analysis_mode == "Standard R-T Analysis":
            with c1:
                y_axis_mode = persistent_selectbox(
                    "Y-Axis Mode",
                    ["Resistance (Ω)", "Normalized (R/R_300K)", "Derivative (dR/dT)"],
                    index=0,
                    persistent_key=f"y_mode_{plot_id}"
                )
            with c2:
                # Use disabled selectbox for alignment
                st.selectbox("X-Axis", ["Temperature (K)"], disabled=True, key=f"x_unit_{plot_id}")
        else:
            # Custom Columns Mode
            ref_cols = []
            
            if selected_datasets:
                df_ref = selected_datasets[0]['full_df']
                valid_cols = df_ref.dropna(axis=1, how='all').columns.tolist()
                ref_cols = valid_cols
            elif available_datasets:
                df_ref = available_datasets[0]['full_df']
                valid_cols = df_ref.dropna(axis=1, how='all').columns.tolist()
                ref_cols = valid_cols

            # Check for Oe column to offer Tesla conversion
            display_cols = list(ref_cols)
            has_oe = any("Oe" in c or "Oersted" in c for c in ref_cols)
            if has_oe:
                display_cols.append("Magnetic Field (T)")

            # Smart Defaults for Custom Columns
            def get_smart_index(cols, keywords):
                for i, c in enumerate(cols):
                    if any(k in c.lower() for k in keywords):
                        return i
                return 0

            default_y_idx = get_smart_index(display_cols, ["resist", "ohm", "voltage"])
            default_x_idx = get_smart_index(display_cols, ["temp", "field", "tesla", "oe"])
            
            # Ensure X and Y are different if possible
            if default_x_idx == default_y_idx and len(display_cols) > 1:
                default_x_idx = (default_y_idx + 1) % len(display_cols)

            with c1:
                custom_y_col = persistent_selectbox("Y Column", display_cols, index=default_y_idx, persistent_key=f"y_col_{plot_id}")
            with c2:
                custom_x_col = persistent_selectbox("X Column", display_cols, index=default_x_idx, persistent_key=f"x_col_{plot_id}")
            
            # Oe to T conversion removed (handled by virtual column)

        # Row 2: Processing
        c4, c5 = st.columns([1, 1], vertical_alignment="bottom")
        with c4:
            smooth_window = persistent_input(st.number_input, f"smooth_{plot_id}", label="Smoothing (pts)", min_value=0, value=0, step=1, help="Moving average window size.")
        with c5:
            # Common Toggles
            show_linear_fit = persistent_input(st.toggle, f"fit_{plot_id}", label="Show Linear Fit", value=False, help="Fit Y = aX + b")
            
            if analysis_mode == "Standard MR Analysis":
                symmetrize = persistent_input(st.toggle, f"sym_{plot_id}", label="Symmetrize Data", value=False, help="R(H) = (R(H) + R(-H))/2")
                plot_derivative = False
            elif analysis_mode == "Standard R-T Analysis":
                symmetrize = False
                plot_derivative = False
            else:
                symmetrize = False
                plot_derivative = persistent_input(st.toggle, f"deriv_{plot_id}", label="Plot Derivative (dY/dX)", value=False, help="Plot dY/dX vs X")

        # Fit Settings (Conditional)
        fit_range_min = None
        fit_range_max = None
        
        if show_linear_fit:
            with st.popover("📏 Fit Settings", width='stretch'):
                st.markdown("**Linear Fit Range (X-Axis)**")
                c_fmin, c_fmax = st.columns(2)
                with c_fmin:
                    fit_range_min = persistent_input(st.number_input, f"fmin_{plot_id}", label="Min X", value=None, placeholder="Start")
                with c_fmax:
                    fit_range_max = persistent_input(st.number_input, f"fmax_{plot_id}", label="Max X", value=None, placeholder="End")
                st.caption("Leave empty to fit the entire range.")

        if not selected_datasets:
            st.info("Select at least one file to display the plot.")
            return None

        # --- Legend Customization ---
        custom_legends = {}
        with st.popover("🖊️ Legend Labels", width='stretch'):
            for d in selected_datasets:
                default_leg = d['label']
                custom_leg = persistent_input(st.text_input, f"leg_{plot_id}_{d['id']}", label=f"Label for {d['fileName']}", value=default_leg)
                custom_legends[d['id']] = custom_leg

        # --- Customization ---
        with st.expander("🎨 Plot Customization", expanded=False):
            # Row 1: Titles & Theme
            col_cust1, col_cust2, col_cust3 = st.columns(3)
            with col_cust1:
                custom_title = persistent_input(st.text_input, f"title_{plot_id}", label="Plot Title", value="", placeholder="Auto-generated if empty")
                title_font_size = persistent_input(st.number_input, f"title_font_{plot_id}", label="Title Font Size", value=20, min_value=10, max_value=50)
            with col_cust2:
                template_mode = persistent_selectbox(
                    "Theme", 
                    ["Auto (Global)", "plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"], 
                    index=0, 
                    persistent_key=f"theme_{plot_id}"
                )
                show_legend = persistent_input(st.checkbox, f"legend_{plot_id}", label="Show Legend", value=True)
            with col_cust3:
                plot_mode = persistent_selectbox(
                    "Plot Style", 
                    ["Lines", "Markers", "Lines+Markers"], 
                    index=0,
                    persistent_key=f"style_{plot_id}"
                )
                if "Lines" in plot_mode:
                    line_width = persistent_input(st.number_input, f"lw_{plot_id}", label="Line Width", value=2.0, min_value=0.5, max_value=10.0, step=0.5)
                else:
                    line_width = 2.0
                
                if "Markers" in plot_mode:
                    marker_size = persistent_input(st.number_input, f"ms_{plot_id}", label="Marker Size", value=6, min_value=1, max_value=20, step=1)
                else:
                    marker_size = 6

            st.markdown("---")
            
            # Row 2: Axes
            col_cust4, col_cust5, col_cust6 = st.columns(3)
            with col_cust4:
                custom_xlabel = persistent_input(st.text_input, f"xlabel_{plot_id}", label="X-Axis Label", value="", placeholder="Auto-generated if empty")
                axis_title_size = persistent_input(st.number_input, f"axis_title_font_{plot_id}", label="Axis Title Size", value=16, min_value=8, max_value=40)
                
                # X Limits
                use_xlim = persistent_input(st.checkbox, f"use_xlim_{plot_id}", label="Set X Limits")
                if use_xlim:
                    c_xmin, c_xmax = st.columns(2)
                    with c_xmin:
                        xlim_min = persistent_input(st.number_input, f"xlim_min_{plot_id}", label="Min", value=-9.0, format="%.2f")
                    with c_xmax:
                        xlim_max = persistent_input(st.number_input, f"xlim_max_{plot_id}", label="Max", value=9.0, format="%.2f")
                else:
                    xlim_min, xlim_max = None, None

            with col_cust5:
                custom_ylabel = persistent_input(st.text_input, f"ylabel_{plot_id}", label="Y-Axis Label", value="", placeholder="Auto-generated if empty")
                tick_font_size = persistent_input(st.number_input, f"tick_font_{plot_id}", label="Tick Label Size", value=14, min_value=8, max_value=30)
                
                # Y Limits
                use_ylim = persistent_input(st.checkbox, f"use_ylim_{plot_id}", label="Set Y Limits")
                if use_ylim:
                    c_ymin, c_ymax = st.columns(2)
                    with c_ymin:
                        ylim_min = persistent_input(st.number_input, f"ylim_min_{plot_id}", label="Min", value=0.0, format="%.2e")
                    with c_ymax:
                        ylim_max = persistent_input(st.number_input, f"ylim_max_{plot_id}", label="Max", value=100.0, format="%.2e")
                else:
                    ylim_min, ylim_max = None, None

            with col_cust6:
                show_grid = persistent_input(st.checkbox, f"grid_{plot_id}", label="Show Grid", value=True)
                grid_color = persistent_input(st.color_picker, f"grid_color_{plot_id}", label="Grid Color", value="#E5E5E5")

        fig = go.Figure()
        
        # Prepare data for export
        export_data = {}

        for d in selected_datasets:
            x_data = None
            y_data = None
            x_label = ""
            y_label = ""

            if analysis_mode == "Standard MR Analysis":
                df = pd.DataFrame({"H_T": d["H_T"], "R": d["R"]})
                
                # Symmetrization (R_sym(H) = (R(H) + R(-H))/2)
                if symmetrize:
                    # 1. Sort by Field
                    df = df.sort_values(by="H_T")
                    
                    max_h = df["H_T"].abs().max()
                    
                    df_sorted = df.sort_values("H_T")
                    h_sorted = df_sorted["H_T"].values
                    r_sorted = df_sorted["R"].values
                    
                    target_h = np.linspace(0, max_h, num=int(max_h/0.01) + 100)
                    
                    # Interpolate
                    r_plus = np.interp(target_h, h_sorted, r_sorted)
                    r_minus = np.interp(-target_h, h_sorted, r_sorted)
                    
                    r_sym = (r_plus + r_minus) / 2.0
                    
                    # Reconstruct full symmetric curve
                    final_h = np.concatenate([-target_h[::-1], target_h])
                    final_r = np.concatenate([r_sym[::-1], r_sym])
                    
                    # Update df for subsequent processing
                    df = pd.DataFrame({"H_T": final_h, "R": final_r})

                # Calculate R0
                r0 = 1.0
                if r0_method == "First Point":
                    r0 = df["R"].iloc[0]
                elif r0_method == "Closest to 0T":
                    idx = df["H_T"].abs().idxmin()
                    r0 = df["R"].iloc[idx]
                elif r0_method == "Mean within Window":
                    mask = df["H_T"].abs() <= r0_window
                    if mask.any():
                        r0 = df.loc[mask, "R"].mean()
                    else:
                        idx = df["H_T"].abs().idxmin()
                        r0 = df["R"].iloc[idx]

                # Calculate X
                x_data = df["H_T"]
                x_label = "Field (T)"
                if x_axis_unit == "Oersted (Oe)":
                    x_data = x_data * 10000
                    x_label = "Field (Oe)"

                # Calculate Y
                y_data = df["R"]
                y_label = "Resistance (Ω)"
                
                if y_axis_mode == "Magnetoresistance (MR %)":
                    y_data = 100 * (df["R"] - r0) / r0
                    y_label = "MR (%)"
                elif y_axis_mode == "Normalized (R/R0)":
                    y_data = df["R"] / r0
                    y_label = "R / R0"
                elif y_axis_mode == "Derivative (dR/dH)":
                    # Simple finite difference
                    dy = df["R"].diff()
                    dx = df["H_T"].diff()
                    y_data = dy / dx
                    y_label = "dR/dH (Ω/T)"
                    # Remove first NaN
                    y_data = y_data.fillna(0)
            
            elif analysis_mode == "Standard R-T Analysis":
                if 'full_df' not in d:
                    st.warning(f"Full data not available for {d['label']}")
                    continue
                
                full_df = d['full_df']
                cols = full_df.columns.tolist()
                temp_idx = choose_temperature_column(cols)
                
                if temp_idx < 0:
                    st.warning(f"No Temperature column found in {d['label']}")
                    continue
                    
                temp_col = cols[temp_idx]
                
                # Use the resistance column identified during parsing
                r_col = d['rCol']
                if r_col not in full_df.columns:
                    st.warning(f"Resistance column '{r_col}' not found in {d['label']}")
                    continue
                
                # Create working DF
                df = pd.DataFrame({"T": full_df[temp_col], "R": full_df[r_col]})
                df = df.dropna().sort_values("T")
                
                x_data = df["T"]
                x_label = "Temperature (K)"
                
                if y_axis_mode == "Resistance (Ω)":
                    y_data = df["R"]
                    y_label = "Resistance (Ω)"
                elif y_axis_mode == "Normalized (R/R_300K)":
                    # Find R at closest T to 300K
                    idx_300 = (df["T"] - 300).abs().idxmin()
                    r_300 = df.loc[idx_300, "R"]
                    y_data = df["R"] / r_300
                    y_label = "R / R(300K)"
                elif y_axis_mode == "Derivative (dR/dT)":
                    dy = df["R"].diff()
                    dx = df["T"].diff()
                    y_data = dy / dx
                    y_label = "dR/dT (Ω/K)"
                    y_data = y_data.fillna(0)

            else:
                # Custom Columns Mode
                if 'full_df' not in d:
                    st.warning(f"Full data not available for {d['label']}")
                    continue
                
                full_df = d['full_df']
                
                # Helper to get data (handling virtual columns)
                def get_col_data(col_name, df):
                    if col_name == "Magnetic Field (T)":
                        # Find Oe column
                        for c in df.columns:
                            if "Oe" in c or "Oersted" in c:
                                return df[c] * 1e-4, "Magnetic Field (T)"
                        return None, None
                    elif col_name in df.columns:
                        return df[col_name], col_name
                    return None, None

                x_data, x_label = get_col_data(custom_x_col, full_df)
                y_data, y_label = get_col_data(custom_y_col, full_df)

                if x_data is None:
                    st.warning(f"Column '{custom_x_col}' not found in {d['label']}")
                    continue
                if y_data is None:
                    st.warning(f"Column '{custom_y_col}' not found in {d['label']}")
                    continue
                
                # Drop NaNs
                mask = x_data.notna() & y_data.notna()
                x_data = x_data[mask]
                y_data = y_data[mask]

                if plot_derivative:
                    # Sort by X for derivative calculation
                    temp_df = pd.DataFrame({'x': x_data, 'y': y_data}).sort_values('x')
                    x_sorted = temp_df['x']
                    y_sorted = temp_df['y']
                    
                    dy = y_sorted.diff()
                    dx = x_sorted.diff()
                    
                    # Calculate derivative
                    deriv = dy / dx
                    
                    # Handle division by zero or infinite values
                    deriv = deriv.replace([np.inf, -np.inf], np.nan)
                    
                    x_data = x_sorted
                    y_data = deriv
                    y_label = f"d({y_label})/d({x_label})"
                    
                    # Drop NaNs created by diff
                    mask_d = y_data.notna()
                    x_data = x_data[mask_d]
                    y_data = y_data[mask_d]

            # Smoothing
            if smooth_window > 1:
                y_data = y_data.rolling(window=int(smooth_window), center=True).mean()

            # Plot Trace
            mode_map = {"Lines": "lines", "Markers": "markers", "Lines+Markers": "lines+markers"}
            
            # Legend Name
            legend_name = custom_legends.get(d['id'], d['fileName'])
            
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode=mode_map[plot_mode],
                name=legend_name,
                hovertemplate=f"{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4e}}<extra></extra>",
                line=dict(width=line_width) if "Lines" in plot_mode else None,
                marker=dict(size=marker_size) if "Markers" in plot_mode else None
            ))
            
            # Linear Fit
            if show_linear_fit and x_data is not None and y_data is not None:
                # Remove NaNs for fitting
                mask_fit = x_data.notna() & y_data.notna()
                
                # Apply Range Filter
                if fit_range_min is not None:
                    mask_fit &= (x_data >= fit_range_min)
                if fit_range_max is not None:
                    mask_fit &= (x_data <= fit_range_max)
                
                xf = x_data[mask_fit]
                yf = y_data[mask_fit]
                
                if len(xf) > 1:
                    # Polyfit degree 1
                    slope, intercept = np.polyfit(xf, yf, 1)
                    y_fit = slope * xf + intercept
                    
                    # Plot the fit line only within the range
                    fig.add_trace(go.Scatter(
                        x=xf,
                        y=y_fit,
                        mode='lines',
                        name=f"Fit {legend_name}",
                        line=dict(dash='dash', width=2, color='red'),
                        hoverinfo='skip'
                    ))
                    
                    # Enhanced Annotation
                    eq_text = f"<b>y = {slope:.3e} x + {intercept:.3e}</b>"
                    
                    # Position annotation near the center of the fit segment
                    mid_idx = len(xf) // 2
                    
                    fig.add_annotation(
                        x=xf.iloc[mid_idx],
                        y=y_fit.iloc[mid_idx],
                        text=eq_text,
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        ax=0,
                        ay=-40,
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="black",
                        borderwidth=1,
                        font=dict(size=14, color="black")
                    )

            # Add to export
            clean_label = d['label'].replace(" ", "_")
            export_data[f"{clean_label}_X"] = x_data.values
            export_data[f"{clean_label}_Y"] = y_data.values

        # Apply Customization
        final_title = custom_title if custom_title else f"{y_label} vs {x_label}"
        final_xlabel = custom_xlabel if custom_xlabel else x_label
        final_ylabel = custom_ylabel if custom_ylabel else y_label

        # Determine Theme
        final_template = template_mode
        if template_mode == "Auto (Global)":
            final_template = "plotly" 
        
        # Construct uirevision to preserve zoom/pan across reruns
        uirevision_key = f"{plot_id}_{analysis_mode}"
        if analysis_mode == "Standard MR Analysis":
             uirevision_key += f"_{x_axis_unit}_{y_axis_mode}"
        elif analysis_mode == "Standard R-T Analysis":
             uirevision_key += f"_{y_axis_mode}"
        else:
             uirevision_key += f"_{custom_x_col}_{custom_y_col}"

        layout_args = dict(
            title=dict(
                text=final_title,
                font=dict(size=title_font_size),
                x=0.5, # Center title
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(text=final_xlabel, font=dict(size=axis_title_size)),
                tickfont=dict(size=tick_font_size),
                showgrid=show_grid,
                gridcolor=grid_color,
                range=[xlim_min, xlim_max] if use_xlim else None
            ),
            yaxis=dict(
                title=dict(text=final_ylabel, font=dict(size=axis_title_size)),
                tickfont=dict(size=tick_font_size),
                showgrid=show_grid,
                gridcolor=grid_color,
                range=[ylim_min, ylim_max] if use_ylim else None
            ),
            showlegend=show_legend,
            hovermode="closest",
            height=height,
            width=width,
            template=final_template,
            uirevision=uirevision_key
        )

        fig.update_layout(**layout_args)

        # Smart Filename for Download
        safe_title = "".join([c for c in final_title if c.isalnum() or c in (' ', '-', '_')]).strip().replace(" ", "_")
        if not safe_title:
            safe_title = f"plot_{plot_id}"
            
        config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f"MR_Analysis_{safe_title}",
                'height': height,
                'width': width,
                'scale': 2
            }
        }

        st.plotly_chart(fig, width="stretch", config=config, key=f"chart_{plot_id}")
        
        # Export Options
        st.write("") # Spacer
        c_left, c_center, c_right = st.columns([1, 2, 1])
        with c_center:
            # Export of plotted data (Origin compatible .dat)
            if export_data:
                df_export = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in export_data.items() ]))
                # Use tab separator for Origin compatibility
                dat_exp = df_export.to_csv(index=False, sep='\t').encode('utf-8')
                st.download_button(
                    label=f"Download Plot {plot_id} Data (.dat)",
                    data=dat_exp,
                    file_name=f"plot_{plot_id}_data.dat",
                    mime="text/plain",
                    width='stretch'
                )
        return fig

# --- Main Layout ---

# Layout Settings
with st.expander("Layout & Export Settings", expanded=True):
    col_lay1, col_lay2, col_lay3 = st.columns(3)
    with col_lay1:
        num_cols = st.number_input("Columns per Row", min_value=1, max_value=3, value=1)
    with col_lay2:
        plot_width = st.number_input("Plot Width (px)", min_value=300, value=600)
    with col_lay3:
        plot_height = st.number_input("Plot Height (px)", min_value=300, value=600)

# Render Plots in Grid
generated_figures = []
plot_indices = st.session_state.plot_ids

# Create rows
rows = [plot_indices[i:i + num_cols] for i in range(0, len(plot_indices), num_cols)]

for row_indices in rows:
    cols = st.columns(num_cols)
    for i, plot_idx in enumerate(row_indices):
        with cols[i]:
            fig = create_plot_interface(str(plot_idx), datasets, plot_width, plot_height)
            if fig:
                generated_figures.append(fig)

# --- Data Table (Global) ---
with st.expander("View Raw Data Metadata"):
    if datasets:
        # Filter by Folder
        batch_map = get_batch_map(datasets, st.session_state.get('custom_batches', {}))
        
        batch_ids = sorted(batch_map.keys())
        options = ["ALL"] + batch_ids
        
        def format_batch_meta(option):
            if option == "ALL":
                return "All Folders"
            return batch_map.get(option, f"Batch {option}")
            
        selected_batch_id = st.selectbox("Filter by Folder", options, format_func=format_batch_meta, index=0, key="meta_batch_filter")
        
        if selected_batch_id != "ALL":
            filtered_datasets = [d for d in datasets if d.get('batch_id') == selected_batch_id]
        else:
            filtered_datasets = datasets
            
        if filtered_datasets:
            dataset_names = [d['fileName'] for d in filtered_datasets]
            selected_meta_idx = st.selectbox("Select File to Inspect", range(len(filtered_datasets)), format_func=lambda i: dataset_names[i], key="meta_file_sel")
            d = filtered_datasets[selected_meta_idx]
            
            st.markdown("### Metadata")
            metadata = {
                "File Name": d["fileName"],
                "Label": d["label"],
                "Temperature (K)": d["temperatureK"],
                "Direction": d["direction"],
                "Field Column": d["fieldCol"],
                "Resistance Column": d["rCol"],
                "Batch": d.get("batch_name", "None")
            }
            st.json(metadata)
        
            st.markdown("### Data Preview")
            if 'full_df' in d:
                # Filter out completely empty columns
                df_display = d['full_df'].dropna(axis=1, how='all')
                st.dataframe(df_display, width='stretch', height=300)
            else:
                st.info("Full dataframe not available.")
        else:
            st.info("No files in this folder.")

