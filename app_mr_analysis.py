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
from typing import List, Dict, Any, Optional, Tuple

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# (Default files removed as requested)

# -----------------------------------------------------------------------------
# PARSING LOGIC (Adapted from original script)
# -----------------------------------------------------------------------------

def parse_temperature_from_filename(name: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*K", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)K", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None

def infer_direction_from_filename(name: str) -> str:
    up = name.upper()
    if "DOWN" in up or "DN" in up:
        return "DOWN"
    if "UP" in up:
        return "UP"
    return "unknown"

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

def generate_smart_label(filename: str, temp: Optional[float], direction: str) -> str:
    # 1. Extract Sample (First alphanumeric block usually)
    sample = "Sample"
    parts = re.split(r'[_\s]+', filename)
    if parts and parts[0].isalnum():
        sample = parts[0]
            
    # Specific overrides
    if "SRO" in filename.upper() and sample == "Sample": sample = "SrRuO3"
    if "STO" in filename.upper() and sample == "Sample": sample = "SrTiO3"

    # 2. Extract Voltage (e.g. 15V)
    voltage = ""
    m_volt = re.search(r"(\d+V)", filename, re.IGNORECASE)
    if m_volt:
        voltage = m_volt.group(1)

    # 3. Extract State (Comp/Tens)
    state = ""
    if re.search(r"Comp", filename, re.IGNORECASE):
        state = "Comp"
    elif re.search(r"Tens", filename, re.IGNORECASE):
        state = "Tens"

    # 4. Extract Measurement Type (e.g. RT0T, RT5T)
    meas_type = ""
    m_meas = re.search(r"(RT\d+T)", filename, re.IGNORECASE)
    if m_meas:
        meas_type = m_meas.group(1)

    # 5. Format Temp
    temp_str = ""
    if temp is not None:
        t_val = int(math.ceil(temp))
        temp_str = f"{t_val}K"
    
    # 6. Format Direction
    dir_str = direction if direction != "unknown" else ""
    
    # Combine
    components = [sample, meas_type, voltage, state, temp_str, dir_str]
    # Filter empty
    components = [c for c in components if c]
    
    return " ".join(components)

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

    tempK = parse_temperature_from_filename(filename)
    if tempK is None and tvals:
        tempK = float(statistics.median(tvals))

    direction = infer_direction_from_filename(filename)
    label = generate_smart_label(filename, tempK, direction)

    return {
        "id": filename,
        "fileName": filename,
        "label": label,
        "temperatureK": tempK,
        "direction": direction,
        "fieldCol": field_name,
        "rCol": resist_name,
        "H_T": H_T,
        "R": R,
        "full_df": df_full
    }

# -----------------------------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------------------------

st.set_page_config(page_title="PPMS Analysis Tool", layout="wide")

st.title("📊 PPMS Analysis Tool")
st.markdown("Upload `.dat` files to analyze MR, Resistance, and Normalization.")

# --- Sidebar: Configuration ---
st.sidebar.header("Configuration")

# Initialize Session State for Data
if 'all_datasets' not in st.session_state:
    st.session_state.all_datasets = []

# File Selection
st.sidebar.subheader("Data Source")
uploaded_files = st.sidebar.file_uploader("Upload .dat files", type=["dat"], accept_multiple_files=True)

# Process Uploaded Files
if uploaded_files:
    new_files_count = 0
    for uploaded_file in uploaded_files:
        # Check if file already loaded to avoid duplicates
        if any(d['fileName'] == uploaded_file.name for d in st.session_state.all_datasets):
            continue
            
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            data = parse_multivu_content(content, uploaded_file.name)
            st.session_state.all_datasets.append(data)
            new_files_count += 1
        except Exception as e:
            st.error(f"Error parsing {uploaded_file.name}: {e}")
    
    if new_files_count > 0:
        st.sidebar.success(f"Added {new_files_count} new files.")

# Clear Data Button
if st.sidebar.button("Clear All Data"):
    st.session_state.all_datasets = []
    st.rerun()

# Data Explorer / Filter
datasets = st.session_state.all_datasets

# Sort datasets
datasets.sort(key=lambda d: (d['temperatureK'] if d['temperatureK'] is not None else 9999, d['fileName']))

# Ensure unique labels
label_counts = {}
for d in datasets:
    l = d['label']
    label_counts[l] = label_counts.get(l, 0) + 1

for d in datasets:
    l = d['label']
    if label_counts[l] > 1:
        d['label'] = f"{l} ({d['fileName']})"

if not datasets:
    st.info("Please upload files to begin.")
    st.stop()

# Sidebar Data Explorer
with st.sidebar.expander("📂 Data Explorer", expanded=False):
    st.write(f"Total Files: {len(datasets)}")
    
    # Group by Sample
    samples = sorted(list(set(d['label'].split()[0] for d in datasets)))
    selected_sample_filter = st.selectbox("Filter by Sample", ["All"] + samples)
    
    filtered_view = datasets
    if selected_sample_filter != "All":
        filtered_view = [d for d in datasets if d['label'].startswith(selected_sample_filter)]
    
    st.dataframe(
        pd.DataFrame([
            {"Label": d['label'], "Temp (K)": d['temperatureK'], "File": d['fileName']} 
            for d in filtered_view
        ]),
        hide_index=True
    )

# --- Sidebar: Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Author:** Amir MEDDAS  
*C2N - Centre de Nanosciences et de Nanotechnologies*  
*LPS - Laboratoire de Physique des Solides*  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/amir-meddas-80876424b/)
""")

# -----------------------------------------------------------------------------
# PLOTTING INTERFACE
# -----------------------------------------------------------------------------

def add_plot_callback():
    new_id = st.session_state.next_plot_id
    st.session_state.plot_ids.append(new_id)
    st.session_state.next_plot_id += 1

def remove_plot_callback(plot_id_str):
    pid = int(plot_id_str)
    if pid in st.session_state.plot_ids:
        st.session_state.plot_ids.remove(pid)

def duplicate_plot_callback(plot_id):
    new_id = st.session_state.next_plot_id
    st.session_state.plot_ids.append(new_id)
    st.session_state.next_plot_id += 1
    
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

def create_plot_interface(plot_id: str, available_datasets: List[Dict[str, Any]], width: int, height: int) -> Optional[go.Figure]:
    """Creates a self-contained plotting interface and returns the figure."""
    
    with st.container(border=True):
        # Header with Actions
        c_head1, c_head2, c_head3, c_head4 = st.columns([0.55, 0.15, 0.15, 0.15], vertical_alignment="center")
        with c_head1:
            st.markdown(f"### Plot {plot_id}")
        
        # Add Button
        with c_head2:
            st.button("➕", key=f"add_btn_{plot_id}", help="Add a new plot", on_click=add_plot_callback)

        # Remove Button
        with c_head3:
            st.button("➖", key=f"del_btn_{plot_id}", help="Remove this plot", on_click=remove_plot_callback, args=(plot_id,))

        # Duplicate Button
        with c_head4:
            st.button("📋", key=f"dup_{plot_id}", help="Duplicate this plot", on_click=duplicate_plot_callback, args=(plot_id,))
        
        # Row 0: Analysis Mode
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Custom Columns", "Standard MR Analysis"],
            index=0,
            key=f"mode_{plot_id}"
        )
        
        # Row 1: Axes & Style
        c1, c2 = st.columns([2, 1])
        
        # Variables for Custom Mode
        custom_x_col = None
        custom_y_col = None
        
        if analysis_mode == "Standard MR Analysis":
            # R0 Method (Moved from Global)
            r0_method = st.selectbox(
                "R0 Calculation Method",
                ["Closest to 0T", "Mean within Window", "First Point"],
                index=0,
                key=f"r0_meth_{plot_id}"
            )
            r0_window = 0.01
            if r0_method == "Mean within Window":
                r0_window = st.number_input("Zero Field Window (T)", value=0.01, step=0.005, format="%.4f", key=f"r0_win_{plot_id}")

            with c1:
                y_axis_mode = st.selectbox(
                    "Y-Axis Mode",
                    ["Magnetoresistance (MR %)", "Resistance (Ω)", "Normalized (R/R0)", "Derivative (dR/dH)"],
                    index=0,
                    key=f"y_mode_{plot_id}"
                )
            with c2:
                x_axis_unit = st.selectbox(
                    "X-Axis Unit",
                    ["Tesla (T)", "Oersted (Oe)"],
                    index=0,
                    key=f"x_unit_{plot_id}"
                )
        else:
            # Custom Columns Mode
            # Get columns from the first available dataset as reference
            ref_cols = []
            
            # File Selector (Multiselect) - Moved up to determine available columns
            options = [d['label'] for d in available_datasets]
            default_sel = options if plot_id == "1" else []
            selected_labels = st.multiselect(f"Select Curves for Plot {plot_id}", options, default=default_sel, key=f"sel_{plot_id}")
            selected_datasets = [d for d, label in zip(available_datasets, options) if label in selected_labels]

            if selected_datasets:
                # Find common non-empty columns or just take from first
                # Let's take the first one for simplicity but filter for non-empty
                df_ref = selected_datasets[0]['full_df']
                valid_cols = df_ref.dropna(axis=1, how='all').columns.tolist()
                ref_cols = valid_cols
            elif available_datasets:
                 # Fallback if nothing selected yet
                df_ref = available_datasets[0]['full_df']
                valid_cols = df_ref.dropna(axis=1, how='all').columns.tolist()
                ref_cols = valid_cols

            with c1:
                custom_y_col = st.selectbox("Y Column", ref_cols, index=0 if ref_cols else 0, key=f"y_col_{plot_id}")
            with c2:
                custom_x_col = st.selectbox("X Column", ref_cols, index=1 if len(ref_cols) > 1 else 0, key=f"x_col_{plot_id}")
            
            # Oe to T conversion
            convert_oe_to_t = st.checkbox("Convert X from Oe to Tesla (x 10^-4)", value=False, key=f"conv_oe_{plot_id}")

        # Row 2: Processing (Smoothing & Symmetrize)
        c4, c5 = st.columns([1, 1], vertical_alignment="bottom")
        with c4:
            smooth_window = st.number_input("Smoothing (pts)", min_value=0, value=0, step=1, key=f"smooth_{plot_id}", help="Moving average window size.")
        with c5:
            if analysis_mode == "Standard MR Analysis":
                symmetrize = st.toggle("Symmetrize Data", value=False, key=f"sym_{plot_id}", help="R(H) = (R(H) + R(-H))/2")
                plot_derivative = False
            else:
                symmetrize = False
                plot_derivative = st.toggle("Plot Derivative (dY/dX)", value=False, key=f"deriv_{plot_id}", help="Plot dY/dX vs X")

        # File Selector (Multiselect) - Only show if not already shown in Custom Mode
        if analysis_mode == "Standard MR Analysis":
            options = [d['label'] for d in available_datasets]
            default_sel = options if plot_id == "1" else []
            selected_labels = st.multiselect(f"Select Curves for Plot {plot_id}", options, default=default_sel, key=f"sel_{plot_id}")
            selected_datasets = [d for d, label in zip(available_datasets, options) if label in selected_labels]

        if not selected_datasets:
            st.info("Select at least one file to display the plot.")
            return None

        # --- Legend Customization ---
        custom_legends = {}
        with st.expander("🖊️ Legend Labels", expanded=False):
            for d in selected_datasets:
                # Default legend logic
                if d['temperatureK']:
                    t_val = int(math.ceil(d['temperatureK']))
                    default_leg = f"{t_val}K"
                else:
                    default_leg = d['fileName']
                
                # Input
                custom_leg = st.text_input(f"Label for {d['label']}", value=default_leg, key=f"leg_{plot_id}_{d['id']}")
                custom_legends[d['id']] = custom_leg

        # --- Customization ---
        with st.expander("🎨 Plot Customization", expanded=False):
            # Row 1: Titles & Theme
            col_cust1, col_cust2, col_cust3 = st.columns(3)
            with col_cust1:
                custom_title = st.text_input("Plot Title", value="", placeholder="Auto-generated if empty", key=f"title_{plot_id}")
                title_font_size = st.number_input("Title Font Size", value=20, min_value=10, max_value=50, key=f"title_font_{plot_id}")
            with col_cust2:
                template_mode = st.selectbox("Theme", ["Auto (Global)", "plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"], index=0, key=f"theme_{plot_id}")
                show_legend = st.checkbox("Show Legend", value=True, key=f"legend_{plot_id}")
            with col_cust3:
                plot_mode = st.selectbox(
                    "Plot Style", 
                    ["Lines", "Markers", "Lines+Markers"], 
                    index=0,
                    key=f"style_{plot_id}"
                )
                if "Lines" in plot_mode:
                    line_width = st.number_input("Line Width", value=2.0, min_value=0.5, max_value=10.0, step=0.5, key=f"lw_{plot_id}")
                else:
                    line_width = 2.0
                
                if "Markers" in plot_mode:
                    marker_size = st.number_input("Marker Size", value=6, min_value=1, max_value=20, step=1, key=f"ms_{plot_id}")
                else:
                    marker_size = 6

            st.markdown("---")
            
            # Row 2: Axes
            col_cust4, col_cust5, col_cust6 = st.columns(3)
            with col_cust4:
                custom_xlabel = st.text_input("X-Axis Label", value="", placeholder="Auto-generated if empty", key=f"xlabel_{plot_id}")
                axis_title_size = st.number_input("Axis Title Size", value=16, min_value=8, max_value=40, key=f"axis_title_font_{plot_id}")
                
                # X Limits
                use_xlim = st.checkbox("Set X Limits", key=f"use_xlim_{plot_id}")
                if use_xlim:
                    c_xmin, c_xmax = st.columns(2)
                    with c_xmin:
                        xlim_min = st.number_input("Min", value=-9.0, format="%.2f", key=f"xlim_min_{plot_id}")
                    with c_xmax:
                        xlim_max = st.number_input("Max", value=9.0, format="%.2f", key=f"xlim_max_{plot_id}")
                else:
                    xlim_min, xlim_max = None, None

            with col_cust5:
                custom_ylabel = st.text_input("Y-Axis Label", value="", placeholder="Auto-generated if empty", key=f"ylabel_{plot_id}")
                tick_font_size = st.number_input("Tick Label Size", value=14, min_value=8, max_value=30, key=f"tick_font_{plot_id}")
                
                # Y Limits
                use_ylim = st.checkbox("Set Y Limits", key=f"use_ylim_{plot_id}")
                if use_ylim:
                    c_ymin, c_ymax = st.columns(2)
                    with c_ymin:
                        ylim_min = st.number_input("Min", value=0.0, format="%.2e", key=f"ylim_min_{plot_id}")
                    with c_ymax:
                        ylim_max = st.number_input("Max", value=100.0, format="%.2e", key=f"ylim_max_{plot_id}")
                else:
                    ylim_min, ylim_max = None, None

            with col_cust6:
                show_grid = st.checkbox("Show Grid", value=True, key=f"grid_{plot_id}")
                grid_color = st.color_picker("Grid Color", value="#E5E5E5", key=f"grid_color_{plot_id}")

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
            
            else:
                # Custom Columns Mode
                if 'full_df' not in d:
                    st.warning(f"Full data not available for {d['label']}")
                    continue
                
                full_df = d['full_df']
                
                if custom_x_col not in full_df.columns:
                    st.warning(f"Column '{custom_x_col}' not found in {d['label']}")
                    continue
                if custom_y_col not in full_df.columns:
                    st.warning(f"Column '{custom_y_col}' not found in {d['label']}")
                    continue
                    
                x_data = full_df[custom_x_col]
                y_data = full_df[custom_y_col]
                x_label = custom_x_col
                y_label = custom_y_col
                
                # Drop NaNs
                mask = x_data.notna() & y_data.notna()
                x_data = x_data[mask]
                y_data = y_data[mask]

                # Convert Oe to T if requested
                if convert_oe_to_t:
                    x_data = x_data * 1e-4
                    if "Oe" in x_label:
                        x_label = x_label.replace("Oe", "T")
                    elif "Oersted" in x_label:
                        x_label = x_label.replace("Oersted", "T")
                    else:
                        x_label = f"{x_label} (T)"

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
            
            # Add to export
            # Use a simpler key for Origin compatibility (no spaces if possible, but Origin handles them)
            # We'll use a structured key to parse later if needed, or just clean names
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
            # Use plotly_white as default base, or let user pick specific theme
            final_template = "plotly" 
        
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
            template=final_template
        )

        fig.update_layout(**layout_args)

        # Smart Filename for Download
        safe_title = "".join([c for c in final_title if c.isalnum() or c in (' ', '-', '_')]).strip().replace(" ", "_")
        if not safe_title:
            safe_title = f"plot_{plot_id}"
            
        config = {
            'toImageButtonOptions': {
                'format': 'png', # one of png, svg, jpeg, webp
                'filename': f"MR_Analysis_{safe_title}",
                'height': height,
                'width': width,
                'scale': 2 # Higher resolution
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
                    use_container_width=True
                )
        return fig

# --- Main Layout ---

# Layout Settings
with st.expander("Layout & Export Settings", expanded=True):
    col_lay1, col_lay2, col_lay3 = st.columns(3)
    with col_lay1:
        num_cols = st.number_input("Columns per Row", min_value=1, max_value=3, value=2)
    with col_lay2:
        plot_width = st.number_input("Plot Width (px)", min_value=300, value=600)
    with col_lay3:
        plot_height = st.number_input("Plot Height (px)", min_value=300, value=600)

# Dynamic Plot Management
if 'plot_ids' not in st.session_state:
    st.session_state.plot_ids = [1]
if 'next_plot_id' not in st.session_state:
    st.session_state.next_plot_id = 2

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

# --- Global HTML Export ---
# (Removed as requested)

# --- Data Table (Global) ---
with st.expander("View Raw Data Metadata"):
    if datasets:
        dataset_names = [d['label'] for d in datasets]
        selected_meta_idx = st.selectbox("Select File to Inspect", range(len(datasets)), format_func=lambda i: dataset_names[i])
        d = datasets[selected_meta_idx]
        
        metadata = {
            "File Name": d["fileName"],
            "Label": d["label"],
            "Temperature (K)": d["temperatureK"],
            "Direction": d["direction"],
            "Field Column": d["fieldCol"],
            "Resistance Column": d["rCol"],
            "Data Points": len(d["H_T"])
        }
        st.json(metadata)
        st.write(f"Raw data preview for {d['label']}:")
        st.dataframe(pd.DataFrame({"Field (T)": d["H_T"], "Resistance": d["R"]}).head(100))

