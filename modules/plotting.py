import streamlit as st
import pandas as pd
import numpy as np
import copy
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
from modules.utils import persistent_selectbox, persistent_input, save_session_state
from modules.data_processing import choose_temperature_column

# Check for st.fragment
try:
    from streamlit import fragment
except ImportError:
    # Use dummy decorator if not available (for older Streamlit versions)
    def fragment(func):
        return func

# -----------------------------------------------------------------------------
# CACHED DATA PROCESSING
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def process_one_curve(
    df: pd.DataFrame,
    analysis_mode: str,
    x_unit: str,
    y_mode: str,
    r0_method: str,
    r0_window: float,
    custom_x_col: Optional[str],
    custom_y_col: Optional[str],
    is_derivative: bool,
    temp_col_idx: int = -1,
    r_col_name: str = ""
) -> Tuple[Optional[pd.Series], Optional[pd.Series], str, str]:
    """
    Processes a single dataframe to return X, Y series and labels.
    Cached for performance.
    """
    x_data, y_data = None, None
    x_label, y_label = "", ""

    if df.empty:
        return None, None, "", ""

    try:
        if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"]:
            # Check availability of standard columns
            if "H_T" not in df.columns or "R" not in df.columns:
                 return None, None, "", ""

            # Standard MR/RH Processing
            
            # 1. R0 Calculation
            r0 = 1.0
            try:
                if r0_method == "First Point": 
                    r0 = df["R"].iloc[0]
                elif r0_method == "Closest to 0T": 
                    r0 = df["R"].iloc[df["H_T"].abs().idxmin()]
                elif r0_method == "Mean within Window":
                    mask = df["H_T"].abs() <= r0_window
                    r0 = df.loc[mask, "R"].mean() if mask.any() else df["R"].iloc[df["H_T"].abs().idxmin()]
                elif r0_method == "Max Resistance": 
                    r0 = df["R"].max()
            except Exception:
                r0 = 1.0
    
            # 2. X Axis
            x_data = df["H_T"]
            x_label = "Field (T)"
            if x_unit == "Oersted (Oe)":
                x_data = x_data * 10000
                x_label = "Field (Oe)"
    
            # 3. Y Axis
            if y_mode == "Magnetoresistance (MR %)":
                y_data = 100 * (df["R"] - r0) / r0
                y_label = "MR (%)"
            elif y_mode == "Normalized (R/R0)":
                y_data = df["R"] / r0
                y_label = "R / R0"
            elif y_mode == "Derivative (dR/dH)":
                y_data = df["R"].diff() / df["H_T"].diff()
                y_label = "dR/dH (Ω/T)"
                y_data = y_data.fillna(0)
            else:
                y_data = df["R"]
                y_label = "Resistance (Ω)"
    
        elif analysis_mode == "Standard R-T Analysis":
            if temp_col_idx >= 0 and r_col_name in df.columns:
                # Extract and Sort
                temp_col = df.columns[temp_col_idx]
                sub_df = pd.DataFrame({"T": df[temp_col], "R": df[r_col_name]}).dropna().sort_values("T")
                
                if not sub_df.empty:
                    x_data = sub_df["T"]
                    x_label = "Temperature (K)"
                    
                    if y_mode == "Resistance (Ω)":
                        y_data = sub_df["R"]
                        y_label = "Resistance (Ω)"
                    elif y_mode == "Normalized (R/R_300K)":
                        try:
                             # Find closest to 300
                             r_300 = sub_df.loc[(sub_df["T"] - 300).abs().idxmin(), "R"]
                        except:
                             r_300 = 1.0
                        y_data = sub_df["R"] / r_300
                        y_label = "R / R(300K)"
                    elif y_mode == "Derivative (dR/dT)":
                        y_data = (sub_df["R"].diff() / sub_df["T"].diff()).fillna(0)
                        y_label = "dR/dT (Ω/K)"
    
        else: # Custom Columns
            # Helper to get col data
            def get_col_data(col_name, dframe):
                if not col_name: return None, None
                if col_name == "Magnetic Field (T)":
                    for c in dframe.columns:
                        if "Oe" in c or "Oersted" in c: return dframe[c] * 1e-4, "Magnetic Field (T)"
                    return None, None
                elif col_name in dframe.columns: return dframe[col_name], col_name
                return None, None
    
            x_d, x_l = get_col_data(custom_x_col, df)
            y_d, y_l = get_col_data(custom_y_col, df)
    
            if x_d is not None and y_d is not None:
                 # Align
                 mask = x_d.notna() & y_d.notna()
                 x_data, y_data = x_d[mask], y_d[mask]
                 x_label, y_label = x_l, y_l
                 
                 if is_derivative:
                     # Sort by X for derivative
                     temp_df = pd.DataFrame({'x': x_data, 'y': y_data}).sort_values('x')
                     
                     deriv = temp_df['y'].diff() / temp_df['x'].diff()
                     deriv = deriv.replace([np.inf, -np.inf], np.nan)
                     
                     x_data = temp_df['x']
                     # We need to filter NaNs created by diff
                     mask_d = deriv.notna()
                     x_data = x_data[mask_d]
                     y_data = deriv[mask_d]
                     y_label = f"d({y_label})/d({x_label})"
    except Exception:
        # Failsafe for any calculation error
        return None, None, "", ""

    return x_data, y_data, x_label, y_label


# -----------------------------------------------------------------------------
# PLOTTING INTERFACE
# -----------------------------------------------------------------------------

def perform_paste(plot_id, key_x, key_y, is_persistent_widget=False):
    """
    Helper to paste last clicked coordinates into widgets.
    Handles both standard widgets and persistent_input widgets.
    """
    lc = st.session_state.get(f"last_click_{plot_id}")
    if not lc: return

    val_x = lc["x"]
    val_y = lc["y"]

    if is_persistent_widget:
        # Logic for persistent_input widgets
        if 'persistent_values' not in st.session_state:
            st.session_state['persistent_values'] = {}
        
        # 1. Update the persistent store
        st.session_state['persistent_values'][key_x] = val_x
        st.session_state['persistent_values'][key_y] = val_y
        
        # 2. Update the actual widget key to prevent persistent_input from overwriting the store
        #    with the old widget value on the next run.
        uploader_key = st.session_state.get('uploader_key', 0)
        wkey_x = f"{key_x}_{uploader_key}"
        wkey_y = f"{key_y}_{uploader_key}"
        
        st.session_state[wkey_x] = val_x
        st.session_state[wkey_y] = val_y
        
        save_session_state()
    else:
        # Logic for standard widgets
        st.session_state[key_x] = val_x
        st.session_state[key_y] = val_y

def add_plot_callback():
    existing_ids = set(st.session_state.plot_ids)
    new_id = 1
    while new_id in existing_ids:
        new_id += 1
    st.session_state.plot_ids.append(new_id)
    save_session_state()

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
            
        # Cleanup persistent values
        if 'persistent_values' in st.session_state:
            p_store = st.session_state['persistent_values']
            p_keys_to_del = []
            for key in p_store.keys():
                if key.endswith(f"_{pid}") or f"_{pid}_" in key:
                    p_keys_to_del.append(key)
            for key in p_keys_to_del:
                del p_store[key]
    save_session_state()

def toggle_rename_callback(plot_id):
    # Save title if we are closing edit mode
    if st.session_state.get(f"ren_mode_{plot_id}", False):
        key = f"pname_{plot_id}"
        if key in st.session_state:
            if 'persistent_values' not in st.session_state:
                st.session_state['persistent_values'] = {}
            st.session_state['persistent_values'][key] = st.session_state[key]

    key = f"ren_mode_{plot_id}"
    st.session_state[key] = not st.session_state.get(key, False)

def close_rename_callback(plot_id):
    st.session_state[f"ren_mode_{plot_id}"] = False
    
    # Persist Title
    key = f"pname_{plot_id}"
    if key in st.session_state:
        if 'persistent_values' not in st.session_state:
            st.session_state['persistent_values'] = {}
        st.session_state['persistent_values'][key] = st.session_state[key]
    save_session_state()

def duplicate_plot_callback(plot_id):
    existing_ids = set(st.session_state.plot_ids)
    new_id = 1
    while new_id in existing_ids:
        new_id += 1
    st.session_state.plot_ids.append(new_id)
    
    # Copy persistent values
    if 'persistent_values' in st.session_state:
        p_store = st.session_state['persistent_values']
        new_entries = {}
        for key, val in p_store.items():
            # Deep copy mutable values to avoid shared state
            val_copy = copy.deepcopy(val)
            
            if key.endswith(f"_{plot_id}"):
                base = key[:-len(str(plot_id))] # remove old id
                new_key = f"{base}{new_id}"
                new_entries[new_key] = val_copy
            elif f"_{plot_id}_" in key:
                new_key = key.replace(f"_{plot_id}_", f"_{new_id}_")
                new_entries[new_key] = val_copy
        p_store.update(new_entries)
    
    # Copy state
    for key in list(st.session_state.keys()):
        # Exclude buttons and temporary states
        if any(x in key for x in ["dup_", "add_btn_", "del_btn_", "ren_btn_", "paste_", "add_annot_btn_", "del_annot_", "dl_", "chart_", "ren_mode_", "del_batch_", "rename_"]): continue
        
        new_key = None
        # Case 1: Key ends with _{plot_id} (Standard widgets)
        if key.endswith(f"_{plot_id}"):
            base = key[:-len(str(plot_id))] # remove old id
            new_key = f"{base}{new_id}"
        
        # Case 2: Key contains _{plot_id}_ (Dynamic widgets like legends: leg_{plot_id}_{file_id})
        elif f"_{plot_id}_" in key:
            new_key = key.replace(f"_{plot_id}_", f"_{new_id}_")
            
        if new_key:
            val = st.session_state[key]
            # Deep copy for mutable types (lists, dicts) to ensure independence
            if isinstance(val, (list, dict)):
                st.session_state[new_key] = copy.deepcopy(val)
            else:
                st.session_state[new_key] = val
                
    save_session_state()

def move_plot_callback(plot_id, direction):
    """Moves a plot up (-1) or down (1) in the list."""
    if 'plot_ids' in st.session_state:
        ids = st.session_state.plot_ids
        try:
            current_idx = ids.index(plot_id)
            target_idx = current_idx + direction
            
            if 0 <= target_idx < len(ids):
                ids[current_idx], ids[target_idx] = ids[target_idx], ids[current_idx]
                save_session_state()
        except ValueError:
            pass

def toggle_collapse_callback(plot_id):
    """Toggles the collapsed state of a plot card."""
    key = f"is_collapsed_{plot_id}"
    st.session_state[key] = not st.session_state.get(key, False)
    
    # Persist state
    if 'persistent_values' not in st.session_state:
        st.session_state['persistent_values'] = {}
    st.session_state['persistent_values'][key] = st.session_state[key]
    save_session_state()

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
    """Creates a self-contained plotting interface and returns the figure. Wrapper to handle toolbar separately."""
    
    with st.container(border=True):
        # --- Header Row (Non-Fragmented) ---
        c_col, c_title, c_ren, c_up, c_down, c_add, c_dup, c_rem = st.columns([0.4, 3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], vertical_alignment="center", gap="small")
        
        # Collapse State
        p_col_key = f"is_collapsed_{plot_id}"
        # Sync with persistent values if present
        if 'persistent_values' in st.session_state and p_col_key in st.session_state['persistent_values']:
            st.session_state[p_col_key] = st.session_state['persistent_values'][p_col_key]
             
        is_collapsed = st.session_state.get(p_col_key, False)
        
        with c_col:
            icon = "▶️" if is_collapsed else "🔽"
            st.button(icon, key=f"col_btn_{plot_id}", help="Collapse/Expand", on_click=toggle_collapse_callback, args=(plot_id,), use_container_width=True)

        with c_title:
            p_key = f"pname_{plot_id}"
            if 'persistent_values' in st.session_state and p_key in st.session_state['persistent_values']:
                plot_name = st.session_state['persistent_values'][p_key]
            else:
                plot_name = st.session_state.get(p_key, f"Plot {plot_id}")

            if st.session_state.get(f"ren_mode_{plot_id}", False):
                if p_key not in st.session_state: st.session_state[p_key] = plot_name
                st.text_input("Name", value=plot_name, key=p_key, label_visibility="collapsed", on_change=close_rename_callback, args=(plot_id,))
            else:
                # Added ID for anchor scrolling
                st.markdown(f"<h3 id='plot_{plot_id}' style='margin: 0; padding: 0; line-height: 1.5;'>{plot_name}</h3>", unsafe_allow_html=True)
        
        with c_ren: st.button("✏️", key=f"ren_btn_{plot_id}", help="Rename Plot", on_click=toggle_rename_callback, args=(plot_id,), use_container_width=True)
        with c_up: st.button("⬆️", key=f"up_btn_{plot_id}", help="Move Up", on_click=move_plot_callback, args=(plot_id, -1), use_container_width=True)
        with c_down: st.button("⬇️", key=f"down_btn_{plot_id}", help="Move Down", on_click=move_plot_callback, args=(plot_id, 1), use_container_width=True)
        with c_add: st.button("➕", key=f"add_btn_{plot_id}", help="Add a new plot", on_click=add_plot_callback, use_container_width=True)
        with c_dup: st.button("📋", key=f"dup_{plot_id}", help="Duplicate this plot", on_click=duplicate_plot_callback, args=(plot_id,), use_container_width=True)
        with c_rem: st.button("➖", key=f"del_btn_{plot_id}", help="Remove this plot", on_click=remove_plot_callback, args=(plot_id,), use_container_width=True)
        
        # --- Content (Fragmented) ---
        # Wrapped in a container to ensure visual continuity within the bordered parent
        if not is_collapsed:
            with st.container():
                render_plot_internal(plot_id, available_datasets, width, height, plot_name)
        return None

@fragment
def render_plot_internal(plot_id: str, available_datasets: List[Dict[str, Any]], width: int, height: int, plot_name: str) -> Optional[go.Figure]:
    """Internal fragmented rendering logic."""
    analysis_mode = persistent_selectbox("Analysis Mode", ["Custom Columns", "Standard MR Analysis", "Standard R-T Analysis"], index=0, persistent_key=f"mode_{plot_id}")
    
    # --- Batch & File Selection ---
    batch_map = get_batch_map(available_datasets, st.session_state.get('custom_batches', {}))
    batch_ids = sorted(batch_map.keys())
    options = ["ALL"] + batch_ids
    
    def format_batch(option):
        return "All Folders" if option == "ALL" else batch_map.get(option, f"Batch {option}")

    persistent_batch_key = f"batch_filter_{plot_id}"
    current_val = st.session_state.get(persistent_batch_key)
    
    # Validate batch selection
    if (isinstance(current_val, str) and current_val != "ALL" and current_val not in options) or (current_val not in options and current_val is not None):
        found_id = "ALL"
        if isinstance(current_val, str):
            for bid, bname in batch_map.items():
                if bname == current_val:
                    found_id = bid
                    break
        st.session_state[persistent_batch_key] = found_id
    
    if st.session_state.get(persistent_batch_key) not in options: st.session_state[persistent_batch_key] = "ALL"
         
    widget_batch_key = f"batch_filter_{plot_id}_{st.session_state.uploader_key}"
    if widget_batch_key in st.session_state: st.session_state[persistent_batch_key] = st.session_state[widget_batch_key]
        
    sb_kwargs = {
        "label": "Filter by Folder",
        "options": options,
        "format_func": format_batch,
        "key": widget_batch_key,
        "on_change": save_session_state
    }
    if widget_batch_key not in st.session_state: sb_kwargs["index"] = options.index(st.session_state[persistent_batch_key])
        
    selected_batch_id = st.selectbox(**sb_kwargs)
    
    filtered_datasets = [d for d in available_datasets if d.get('batch_id') == selected_batch_id] if selected_batch_id != "ALL" else available_datasets
    filtered_filenames = [d['fileName'] for d in filtered_datasets]
    
    persistent_sel_key = f"sel_{plot_id}"
    current_selection = st.session_state.get(persistent_sel_key, [])
    all_existing_files = {d['fileName'] for d in available_datasets}
    valid_selection = [f for f in current_selection if f in all_existing_files]
    
    if len(valid_selection) != len(current_selection):
        st.session_state[persistent_sel_key] = valid_selection
        current_selection = valid_selection
    
    combined_options = sorted(list(set(filtered_filenames + current_selection)))
    widget_sel_key = f"sel_{plot_id}_{st.session_state.uploader_key}_{str(selected_batch_id)}"
    
    def update_selection():
        st.session_state[persistent_sel_key] = st.session_state[widget_sel_key]
        save_session_state()
    
    ms_kwargs = {
        "label": f"Select Curves for {plot_name}",
        "options": combined_options,
        "key": widget_sel_key,
        "on_change": update_selection
    }
    if widget_sel_key not in st.session_state: ms_kwargs["default"] = current_selection
        
    selected_filenames = st.multiselect(**ms_kwargs)
    selected_datasets = [d for d in available_datasets if d['fileName'] in selected_filenames]

    # --- TABS LAYOUT ---
    tab_data, tab_analysis, tab_style, tab_export = st.tabs(["📊 Data", "📈 Analysis", "🎨 Styling", "💾 Export"])

    # State Variables Init
    custom_x_col, custom_y_col = None, None
    r0_method, r0_window = "First Point", 0.01
    y_axis_mode, x_axis_unit = "Resistance (Ω)", "Tesla (T)"
    symmetrize_files = []
    plot_derivative = False
    
    # --- TAB 1: DATA ---
    with tab_data:
        c1, c2 = st.columns([2, 1])
        if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"]:
            # R0 Method
            default_r0_idx = 0
            global_r0 = st.session_state.get("global_r0_method")
            r0_options = ["Closest to 0T", "Mean within Window", "First Point", "Max Resistance"]
            if global_r0 in r0_options: default_r0_idx = r0_options.index(global_r0)

            r0_method = persistent_selectbox("R0 Calculation Method", r0_options, index=default_r0_idx, persistent_key=f"r0_meth_{plot_id}")
            if r0_method == "Mean within Window":
                r0_window = st.number_input("Zero Field Window (T)", value=0.01, step=0.005, format="%.4f", key=f"r0_win_{plot_id}")
            
            with c1:
                y_opts = ["Magnetoresistance (MR %)", "Resistance (Ω)", "Normalized (R/R0)", "Derivative (dR/dH)"] if analysis_mode == "Standard MR Analysis" else ["Resistance (Ω)", "Normalized (R/R0)", "Derivative (dR/dH)"]
                y_axis_mode = persistent_selectbox("Y-Axis Mode", y_opts, index=0, persistent_key=f"y_mode_{plot_id}")
            with c2:
                default_unit_idx = 0
                global_unit = st.session_state.get("global_field_unit")
                unit_options = ["Tesla (T)", "Oersted (Oe)"]
                if global_unit in unit_options: default_unit_idx = unit_options.index(global_unit)
                x_axis_unit = persistent_selectbox("X-Axis Unit", unit_options, index=default_unit_idx, persistent_key=f"x_unit_{plot_id}")
        
        elif analysis_mode == "Standard R-T Analysis":
            with c1:
                y_axis_mode = persistent_selectbox("Y-Axis Mode", ["Resistance (Ω)", "Normalized (R/R_300K)", "Derivative (dR/dT)"], index=0, persistent_key=f"y_mode_{plot_id}")
            with c2:
                st.selectbox("X-Axis", ["Temperature (K)"], disabled=True, key=f"x_unit_{plot_id}")
        
        else: # Custom Columns
            ref_cols = []
            if selected_datasets: df_ref = selected_datasets[0]['full_df']
            elif available_datasets: df_ref = available_datasets[0]['full_df']
            else: df_ref = pd.DataFrame()
            
            if not df_ref.empty:
                ref_cols = df_ref.dropna(axis=1, how='all').columns.tolist()

            display_cols = list(ref_cols)
            if any("Oe" in c or "Oersted" in c for c in ref_cols): display_cols.append("Magnetic Field (T)")
            
            if not display_cols:
                st.warning("No columns available for plotting. Please check the data file.")
            else:
                def get_smart_index(cols, keywords):
                    for i, c in enumerate(cols):
                        if any(k in c.lower() for k in keywords): return i
                    return 0

                default_y_idx = get_smart_index(display_cols, ["resist", "ohm", "voltage"])
                default_x_idx = get_smart_index(display_cols, ["temp", "field", "tesla", "oe"])
                if default_x_idx == default_y_idx and len(display_cols) > 1: default_x_idx = (default_y_idx + 1) % len(display_cols)

                with c1: custom_y_col = persistent_selectbox("Y Column", display_cols, index=default_y_idx, persistent_key=f"y_col_{plot_id}")
                with c2: custom_x_col = persistent_selectbox("X Column", display_cols, index=default_x_idx, persistent_key=f"x_col_{plot_id}")

# --- GET SYMMETRIZATION & DERIVATIVE SETTINGS ---
# We need this to process the data correctly
with tab_analysis:
    __, __, c_proc = st.columns(3)
    with c_proc:
        if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"]:
            # We need to access this info *before* loop, so we assume st.popover and multiselect 
            # will yield state. We need to actually render them to get state if not already set?
            # Actually, we can check session state directly if it exists, or render it.
            # To be safe and keep order, we'll render the widget here but validation data logic happens *after*.
            with st.popover("🪞 Symmetrize", use_container_width=True):
                st.markdown("Select files to symmetrize:")
                sym_options = {d['id']: d['fileName'] for d in selected_datasets}
                sym_key = f"sym_files_{plot_id}"
                if 'persistent_values' not in st.session_state: st.session_state['persistent_values'] = {}
                saved_sym = st.session_state['persistent_values'].get(sym_key, [])
                valid_sym = [sid for sid in saved_sym if sid in sym_options]
                
                sym_widget_key = f"widget_sym_{plot_id}_{st.session_state.uploader_key}"
                sym_kwargs = {
                    "label": "Files", "options": list(sym_options.keys()), "format_func": lambda x: sym_options[x],
                    "key": sym_widget_key, "label_visibility": "collapsed"
                }
                if sym_widget_key not in st.session_state: sym_kwargs["default"] = valid_sym
                    
                selected_sym_ids = st.multiselect(**sym_kwargs)
                
                if st.session_state['persistent_values'].get(sym_key) != selected_sym_ids:
                    st.session_state['persistent_values'][sym_key] = selected_sym_ids
                    save_session_state()
                symmetrize_files = selected_sym_ids
            plot_derivative = False
        elif analysis_mode == "Standard R-T Analysis":
            plot_derivative = False
        else:
            plot_derivative = persistent_input(st.toggle, f"deriv_{plot_id}", label="Derivative", value=False, help="Plot dY/dX vs X")


# --- DATA PROCESSING LOOP (OPTIMIZED) ---
processed_entries = [] # List of dicts with keys: x, y, x_label, y_label, d_id, suffix

global_x_min, global_x_max = float('inf'), float('-inf')
global_y_min, global_y_max = float('inf'), float('-inf')

if selected_datasets:
    for d in selected_datasets:
        datasets_to_process = []
        
        # Prepare standard DF if needed
        if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"]:
             if 'H_T' in d and 'R' in d:
                  # Reconstruct standardized dataframe
                  try:
                      std_df = pd.DataFrame({'H_T': d['H_T'], 'R': d['R']})
                      datasets_to_process.append((std_df, "", False))
                  except Exception:
                      # Fallback
                      if 'full_df' in d: datasets_to_process.append((d['full_df'], "", False))
             elif 'full_df' in d:
                  datasets_to_process.append((d['full_df'], "", False))
        elif 'full_df' in d:
             datasets_to_process.append((d['full_df'], "", False))

        final_datasets_to_process = []
        for df_curr, suffix, is_sym in datasets_to_process:
            final_datasets_to_process.append((df_curr, suffix, is_sym))
            # Add symmetrized copy if needed and we have H_T
            if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"] and d['id'] in symmetrize_files and not is_sym and "H_T" in df_curr.columns:
                df_sym = df_curr.copy()
                df_sym["H_T"] = -df_sym["H_T"]
                final_datasets_to_process.append((df_sym, " (Sym)", True))
        
        for df_curr, suffix, is_sym in final_datasets_to_process:
            # Use our cached helper
            p_x, p_y, p_lx, p_ly = process_one_curve(
                df_curr, analysis_mode, x_axis_unit, y_axis_mode, 
                r0_method, r0_window, custom_x_col, custom_y_col, 
                plot_derivative,
                temp_col_idx=choose_temperature_column(df_curr.columns.tolist()),
                r_col_name=d.get('rCol', 'R')
            )
            
            if p_x is not None and p_y is not None:
                # Update Ranges
                if not p_x.empty:
                    global_x_min = min(global_x_min, p_x.min())
                    global_x_max = max(global_x_max, p_x.max())
                if not p_y.empty:
                    global_y_min = min(global_y_min, p_y.min())
                    global_y_max = max(global_y_max, p_y.max())
                
                processed_entries.append({
                    "x": p_x, "y": p_y, 
                    "lx": p_lx, "ly": p_ly,
                    "id": d['id'], "name": d['fileName'],
                    "suffix": suffix
                })

# Default Ranges
if global_x_min == float('inf'): global_x_min, global_x_max = -10.0, 10.0
if global_y_min == float('inf'): global_y_min, global_y_max = 0.0, 100.0

# Smart Formatting Helpers
def get_smart_format(val_min, val_max):
    val_range = abs(val_max - val_min)
    if val_range == 0: return "%.4f"
    if val_range < 1e-3: return "%.4e"
    if val_range > 1000: return "%.2f"
    return "%.4f"
    
def get_smart_step(val_min, val_max):
    val_range = abs(val_max - val_min)
    if val_range == 0: return 0.1
    if val_range < 1: return 0.001
    if val_range < 100: return 0.1
    return 1.0

x_fmt, x_step = get_smart_format(global_x_min, global_x_max), get_smart_step(global_x_min, global_x_max)
y_fmt, y_step = get_smart_format(global_y_min, global_y_max), get_smart_step(global_y_min, global_y_max)

# --- TAB 2: ANALYSIS (With Calculated Ranges) ---
show_linear_fit, show_parabolic_fit = False, False
fit_range_min, fit_range_max = None, None
pfit_range_min, pfit_range_max = None, None
linear_fit_settings, parabolic_fit_settings = {}, {}

with tab_analysis:
    c_fit1, c_fit2, __ = st.columns(3)
    with c_fit1: show_linear_fit = persistent_input(st.toggle, f"fit_{plot_id}", label="Linear Fit", value=False)
    with c_fit2: show_parabolic_fit = persistent_input(st.toggle, f"pfit_{plot_id}", label="Parabolic Fit", value=False)
    
    if show_linear_fit or show_parabolic_fit:
        st.markdown("###### Fit Settings")
        fit_options = {d['id']: d['fileName'] for d in selected_datasets}
        
        if show_linear_fit:
            with st.expander("Linear Fit Configuration", expanded=True):
                # Same logic as original but simplified
                fit_sel_key = f"fit_sel_{plot_id}"
                saved_fit_sel = st.session_state['persistent_values'].get(fit_sel_key, [])
                valid_fit_sel = [sid for sid in saved_fit_sel if sid in fit_options]
                
                fit_widget_key = f"widget_fit_sel_{plot_id}_{st.session_state.uploader_key}"
                fit_kwargs = {"label":"Select Curves", "options":list(fit_options.keys()), "format_func":lambda x: fit_options[x], "key":fit_widget_key, "default": valid_fit_sel} if fit_widget_key not in st.session_state else {"label":"Select Curves", "options":list(fit_options.keys()), "format_func":lambda x: fit_options[x], "key":fit_widget_key}
                
                selected_fit_ids = st.multiselect(**fit_kwargs)
                if st.session_state['persistent_values'].get(fit_sel_key) != selected_fit_ids:
                    st.session_state['persistent_values'][fit_sel_key] = selected_fit_ids
                    save_session_state()
                    
                c_fmin, c_fmax = st.columns(2)
                with c_fmin: fit_range_min = persistent_input(st.number_input, f"fmin_{plot_id}", label="Min X", value=None, placeholder="Start", format=x_fmt, step=x_step)
                with c_fmax: fit_range_max = persistent_input(st.number_input, f"fmax_{plot_id}", label="Max X", value=None, placeholder="End", format=x_fmt, step=x_step)
                
                for fid in selected_fit_ids:
                    st.caption(f"Settings for: {fit_options[fid]}")
                    c_fc, c_fs, c_fw = st.columns(3)
                    with c_fc: f_color = persistent_input(st.color_picker, f"fit_col_{plot_id}_{fid}", label="Line Color", value="#FF0000")
                    with c_fs: f_style = persistent_selectbox("Line Style", ["dash", "solid", "dot", "dashdot"], index=0, persistent_key=f"fit_style_{plot_id}_{fid}")
                    with c_fw: f_width = persistent_input(st.number_input, f"fit_width_{plot_id}_{fid}", label="Width", value=2.0, step=0.5)

                    c_ax, c_ay, c_btn = st.columns([1, 1, 1], vertical_alignment="bottom")
                    with c_ax: f_annot_x = persistent_input(st.number_input, f"fit_ax_{plot_id}_{fid}", label="Annot X", value=None, placeholder="Auto", format=x_fmt, step=x_step)
                    with c_ay: f_annot_y = persistent_input(st.number_input, f"fit_ay_{plot_id}_{fid}", label="Annot Y", value=None, placeholder="Auto", format=y_fmt, step=y_step)
                    with c_btn:
                        last_click = st.session_state.get(f"last_click_{plot_id}")
                        sel_str = f"{x_fmt % last_click['x']}, {y_fmt % last_click['y']}" if last_click else "No point"
                        st.markdown(f"<div style='font-size: 13px; margin-bottom: 0.5rem;'>Sel: {sel_str}</div>", unsafe_allow_html=True)
                        st.button("📍 Paste", key=f"paste_fit_{plot_id}_{fid}", on_click=perform_paste, use_container_width=True, args=(plot_id, f"fit_ax_{plot_id}_{fid}", f"fit_ay_{plot_id}_{fid}", True))
                    
                    linear_fit_settings[fid] = {"color": f_color, "style": f_style, "width": f_width, "annot_x": f_annot_x, "annot_y": f_annot_y}

        if show_parabolic_fit:
            with st.expander("Parabolic Fit Configuration", expanded=True):
                pfit_sel_key = f"pfit_sel_{plot_id}"
                saved_pfit_sel = st.session_state['persistent_values'].get(pfit_sel_key, [])
                valid_pfit_sel = [sid for sid in saved_pfit_sel if sid in fit_options]
                
                pfit_widget_key = f"widget_pfit_sel_{plot_id}_{st.session_state.uploader_key}"
                pfit_kwargs = {"label":"Select Curves", "options":list(fit_options.keys()), "format_func":lambda x: fit_options[x], "key":pfit_widget_key, "default": valid_pfit_sel} if pfit_widget_key not in st.session_state else {"label":"Select Curves", "options":list(fit_options.keys()), "format_func":lambda x: fit_options[x], "key":pfit_widget_key}
                
                selected_pfit_ids = st.multiselect(**pfit_kwargs)
                if st.session_state['persistent_values'].get(pfit_sel_key) != selected_pfit_ids:
                    st.session_state['persistent_values'][pfit_sel_key] = selected_pfit_ids
                    save_session_state()
                
                c_pmin, c_pmax = st.columns(2)
                with c_pmin: pfit_range_min = persistent_input(st.number_input, f"pfmin_{plot_id}", label="Min X", value=None, placeholder="Start", format=x_fmt, step=x_step)
                with c_pmax: pfit_range_max = persistent_input(st.number_input, f"pfmax_{plot_id}", label="Max X", value=None, placeholder="End", format=x_fmt, step=x_step)
                
                for fid in selected_pfit_ids:
                    st.caption(f"Settings for: {fit_options[fid]}")
                    c_fc, c_fs, c_fw = st.columns(3)
                    with c_fc: pf_color = persistent_input(st.color_picker, f"pfit_col_{plot_id}_{fid}", label="Line Color", value="#00FF00")
                    with c_fs: pf_style = persistent_selectbox("Line Style", ["dot", "solid", "dash", "dashdot"], index=0, persistent_key=f"pfit_style_{plot_id}_{fid}")
                    with c_fw: pf_width = persistent_input(st.number_input, f"pfit_width_{plot_id}_{fid}", label="Width", value=3.0, step=0.5)

                    c_ax, c_ay, c_btn = st.columns([1, 1, 1], vertical_alignment="bottom")
                    with c_ax: pf_annot_x = persistent_input(st.number_input, f"pfit_ax_{plot_id}_{fid}", label="Annot X", value=None, placeholder="Auto", format=x_fmt, step=x_step)
                    with c_ay: pf_annot_y = persistent_input(st.number_input, f"pfit_ay_{plot_id}_{fid}", label="Annot Y", value=None, placeholder="Auto", format=y_fmt, step=y_step)
                    with c_btn:
                        last_click = st.session_state.get(f"last_click_{plot_id}")
                        sel_str = f"{x_fmt % last_click['x']}, {y_fmt % last_click['y']}" if last_click else "No point"
                        st.markdown(f"<div style='font-size: 13px; margin-bottom: 0.5rem;'>Sel: {sel_str}</div>", unsafe_allow_html=True)
                        st.button("📍 Paste", key=f"paste_pfit_{plot_id}_{fid}", on_click=perform_paste, use_container_width=True, args=(plot_id, f"pfit_ax_{plot_id}_{fid}", f"pfit_ay_{plot_id}_{fid}", True))
                    
                    parabolic_fit_settings[fid] = {"color": pf_color, "style": pf_style, "width": pf_width, "annot_x": pf_annot_x, "annot_y": pf_annot_y}

# --- TAB 3: STYLING ---
curve_settings, custom_legends = {}, {}
with tab_style:
    if not selected_datasets: st.info("Select curves first.")
    else:
        for d in selected_datasets:
            with st.expander(f"Curve: {d['fileName']}", expanded=False):
                c_col1, c_col2, c_col3 = st.columns([1, 1, 2])
                with c_col1: use_custom_color = persistent_input(st.checkbox, f"use_col_{plot_id}_{d['id']}", label="Custom Color", value=False)
                with c_col2: curve_color = persistent_input(st.color_picker, f"color_{plot_id}_{d['id']}", label="Pick Color", value="#000000") if use_custom_color else None
                with c_col3: curve_smooth = persistent_input(st.number_input, f"smooth_{plot_id}_{d['id']}", label="Smoothing (pts)", min_value=0, value=0, step=1)
                
                custom_leg = persistent_input(st.text_input, f"leg_{plot_id}_{d['id']}", label="Legend Label", value=d['label'])
                custom_legends[d['id']] = custom_leg
                curve_settings[d['id']] = {"color": curve_color, "smoothing": curve_smooth}
        
        st.markdown("---")
        st.markdown("##### Global Settings")
        with st.expander("**Plot Appearance**", expanded=False):
            col_cust1, col_cust2, col_cust3 = st.columns(3)
            with col_cust1:
                custom_title = persistent_input(st.text_input, f"title_{plot_id}", label="Plot Title", value="", placeholder="Auto")
                title_font_size = persistent_input(st.number_input, f"title_font_{plot_id}", label="Title Size", value=20, min_value=10)
            with col_cust2:
                template_mode = persistent_selectbox("Theme", ["Auto (Global)", "plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn"], index=0, persistent_key=f"theme_{plot_id}")
                show_legend = persistent_input(st.checkbox, f"legend_{plot_id}", label="Show Legend", value=True)
            with col_cust3:
                plot_mode = persistent_selectbox("Style", ["Lines", "Markers", "Lines+Markers"], index=0, persistent_key=f"style_{plot_id}")
                line_width = persistent_input(st.number_input, f"lw_{plot_id}", label="Line Width", value=2.0, step=0.5)
                marker_size = persistent_input(st.number_input, f"ms_{plot_id}", label="Marker Size", value=6, step=1)
        
        with st.expander("**Axes**", expanded=False):
            col_cust4, col_cust5, col_cust6 = st.columns(3)
            with col_cust4:
                custom_xlabel = persistent_input(st.text_input, f"xlabel_{plot_id}", label="X Label", value="", placeholder="Auto")
                axis_title_size = persistent_input(st.number_input, f"axis_title_font_{plot_id}", label="Label Size", value=16)
                use_xlim = persistent_input(st.checkbox, f"use_xlim_{plot_id}", label="Set X Limits")
                if use_xlim:
                    c_xmin, c_xmax = st.columns(2)
                    with c_xmin: xlim_min = persistent_input(st.number_input, f"xlim_min_{plot_id}", label="Min", value=-9.0, format=x_fmt, step=x_step)
                    with c_xmax: xlim_max = persistent_input(st.number_input, f"xlim_max_{plot_id}", label="Max", value=9.0, format=x_fmt, step=x_step)
                else: xlim_min, xlim_max = None, None
            
            with col_cust5:
                custom_ylabel = persistent_input(st.text_input, f"ylabel_{plot_id}", label="Y Label", value="", placeholder="Auto")
                tick_font_size = persistent_input(st.number_input, f"tick_font_{plot_id}", label="Tick Size", value=14)
                use_ylim = persistent_input(st.checkbox, f"use_ylim_{plot_id}", label="Set Y Limits")
                if use_ylim:
                    c_ymin, c_ymax = st.columns(2)
                    with c_ymin: ylim_min = persistent_input(st.number_input, f"ylim_min_{plot_id}", label="Min", value=0.0, format=y_fmt, step=y_step)
                    with c_ymax: ylim_max = persistent_input(st.number_input, f"ylim_max_{plot_id}", label="Max", value=100.0, format=y_fmt, step=y_step)
                else: ylim_min, ylim_max = None, None
                
            with col_cust6:
                show_grid = persistent_input(st.checkbox, f"grid_{plot_id}", label="Show Grid", value=True)
                grid_color = persistent_input(st.color_picker, f"grid_color_{plot_id}", label="Grid Color", value="#E5E5E5")

        with st.expander("**Text Annotation**", expanded=False):
            if f"annotations_list_{plot_id}" not in st.session_state:
                st.session_state[f"annotations_list_{plot_id}"] = []

            if st.button("Add Annotation", key=f"add_annot_btn_{plot_id}"):
                st.session_state[f"annotations_list_{plot_id}"].append({
                    "text": "New Text", "x": global_x_min, "y": global_y_min, "color": "#000000", "size": 14,
                    "bold": False, "italic": False, "font": "Arial"
                })
                st.rerun()

            to_delete = []
            for i, annot in enumerate(st.session_state[f"annotations_list_{plot_id}"]):
                annot.setdefault("bold", False)
                annot.setdefault("italic", False)
                annot.setdefault("font", "Arial")

                st.markdown(f"**Annotation {i+1}**")
                annot["text"] = st.text_input(f"Text", value=annot["text"], key=f"annot_txt_{plot_id}_{i}")
                
                c_xy1, c_xy2, c_btn = st.columns([1, 1, 1], vertical_alignment="bottom")
                with c_xy1: annot["x"] = st.number_input("X", value=float(annot["x"]), format=x_fmt, step=x_step, key=f"annot_x_{plot_id}_{i}")
                with c_xy2: annot["y"] = st.number_input("Y", value=float(annot["y"]), format=y_fmt, step=y_step, key=f"annot_y_{plot_id}_{i}")
                with c_btn:
                    last_click = st.session_state.get(f"last_click_{plot_id}")
                    sel_str = f"{x_fmt % last_click['x']}, {y_fmt % last_click['y']}" if last_click else "No point"
                    st.markdown(f"<div style='font-size: 13px; margin-bottom: 0.5rem;'>Sel: {sel_str}</div>", unsafe_allow_html=True)
                    st.button("📍 Paste", key=f"paste_click_{plot_id}_{i}", on_click=perform_paste, use_container_width=True, args=(plot_id, f"annot_x_{plot_id}_{i}", f"annot_y_{plot_id}_{i}", False))

                c_style_row = st.columns([1, 1, 2, 0.6, 0.6, 0.5], vertical_alignment="bottom")
                with c_style_row[0]: annot["color"] = st.color_picker("Color", value=annot["color"], key=f"annot_col_{plot_id}_{i}")
                with c_style_row[1]: annot["size"] = st.number_input("Size", value=int(annot["size"]), min_value=5, key=f"annot_sz_{plot_id}_{i}")
                with c_style_row[2]: annot["font"] = st.selectbox("Font", ["Arial", "Times New Roman", "Courier New", "Verdana", "Georgia"], index=0 if annot["font"] == "Arial" else 1, key=f"annot_font_{plot_id}_{i}")
                with c_style_row[3]: annot["bold"] = st.checkbox("B", value=annot["bold"], key=f"annot_bold_{plot_id}_{i}")
                with c_style_row[4]: annot["italic"] = st.checkbox("I", value=annot["italic"], key=f"annot_italic_{plot_id}_{i}")
                with c_style_row[5]:
                    if st.button("🗑️", key=f"del_annot_{plot_id}_{i}", help="Delete Annotation"): to_delete.append(i)
                st.divider()

            if to_delete:
                for index in sorted(to_delete, reverse=True): del st.session_state[f"annotations_list_{plot_id}"][index]
                st.rerun()

# --- TAB 4: EXPORT ---
with tab_export: st.markdown("###### Export Data")

# --- FINAL PLOTTING ---
if not processed_entries:
    st.info("Select at least one file to display the plot.")
    return None
    
fig = go.Figure()
export_df_dict = {}

# 1. Plot Curves
for entry in processed_entries:
    x_data, y_data = entry["x"], entry["y"]
    d_id = entry["id"]
    
    c_settings = curve_settings.get(d_id, {})
    c_color = c_settings.get("color")
    c_smooth = c_settings.get("smoothing", 0)
    
    if c_smooth > 1: y_data = y_data.rolling(window=int(c_smooth), center=True).mean()
    
    base_legend = custom_legends.get(d_id, entry["name"])
    legend_name = f"{base_legend}{entry['suffix']}"
    
    # Style Determination
    line_style = dict(width=line_width, dash='dash' if entry['suffix'] else None)
    if c_color: line_style['color'] = c_color
    
    marker_style = dict(size=marker_size)
    if c_color: marker_style['color'] = c_color
    
    trace_mode = "lines" if plot_mode == "Lines" else "markers" if plot_mode == "Markers" else "lines+markers"
    
    if plot_mode == "Lines":
        trace_mode = "lines+markers" # Hybrid for selection
        marker_style = dict(size=max(marker_size, 8), opacity=0)
        if c_color: marker_style['color'] = c_color

    # Use valid WEBGL Scattergl
    fig.add_trace(go.Scattergl(
        x=x_data, y=y_data, mode=trace_mode, name=legend_name,
        hovertemplate=f"{entry['lx']}: %{{x:.4f}}<br>{entry['ly']}: %{{y:.4e}}<extra></extra>",
        line=line_style if "Lines" in plot_mode else None,
        marker=marker_style if "Markers" in plot_mode or plot_mode == "Lines" else None
    ))
    
    # Fits
    if show_linear_fit and d_id in linear_fit_settings:
        cfg = linear_fit_settings[d_id]
        mask = x_data.notna() & y_data.notna()
        if fit_range_min is not None: mask &= (x_data >= fit_range_min)
        if fit_range_max is not None: mask &= (x_data <= fit_range_max)
        xf, yf = x_data[mask], y_data[mask]
        
        if len(xf) > 1:
            slope, intercept = np.polyfit(xf, yf, 1)
            y_fit = slope * xf + intercept
            fig.add_trace(go.Scattergl(x=xf, y=y_fit, mode='lines', name=f"Linear: {legend_name}", 
                                     line=dict(dash=cfg['style'], width=cfg['width'], color=cfg['color']), hoverinfo='skip'))

            # Annotations
            annot_x_f = cfg.get('annot_x')
            annot_y_f = cfg.get('annot_y')
            mid_idx = len(xf) // 2
            arrow_x, arrow_y = xf.iloc[mid_idx], y_fit.iloc[mid_idx]

            if annot_x_f is None or annot_y_f is None:
                fig.add_annotation(x=arrow_x, y=arrow_y, text=f"<b>y = {slope:.3e} x + {intercept:.3e}</b>", 
                                   showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=cfg['color'], 
                                   ax=0, ay=-40, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor=cfg['color'], borderwidth=1, 
                                   font=dict(size=14, color=cfg['color']))
            else:
                fig.add_annotation(x=arrow_x, y=arrow_y, text=f"<b>y = {slope:.3e} x + {intercept:.3e}</b>", 
                                   showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=cfg['color'], 
                                   ax=annot_x_f, ay=annot_y_f, axref="x", ayref="y", 
                                   bgcolor="rgba(255, 255, 255, 0.8)", bordercolor=cfg['color'], borderwidth=1, 
                                   font=dict(size=14, color=cfg['color']))
            
    if show_parabolic_fit and d_id in parabolic_fit_settings:
        cfg = parabolic_fit_settings[d_id]
        mask = x_data.notna() & y_data.notna()
        if pfit_range_min is not None: mask &= (x_data >= pfit_range_min)
        if pfit_range_max is not None: mask &= (x_data <= pfit_range_max)
        xf, yf = x_data[mask], y_data[mask]
        
        if len(xf) > 2:
            a, b, c = np.polyfit(xf, yf, 2)
            y_fit = a * xf**2 + b * xf + c
            fig.add_trace(go.Scattergl(x=xf, y=y_fit, mode='lines', name=f"Parabolic: {legend_name}",
                                     line=dict(dash=cfg['style'], width=cfg['width'], color=cfg['color']), hoverinfo='skip'))

            # Annotations
            annot_x_pf = cfg.get('annot_x')
            annot_y_pf = cfg.get('annot_y')
            mid_idx = len(xf) // 2
            arrow_x, arrow_y = xf.iloc[mid_idx], y_fit.iloc[mid_idx]

            if annot_x_pf is None or annot_y_pf is None:
                fig.add_annotation(x=arrow_x, y=arrow_y, text=f"<b>y = {a:.2e} x² + {b:.2e} x + {c:.2e}</b>", 
                                   showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=cfg['color'], 
                                   ax=0, ay=40, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor=cfg['color'], borderwidth=1, 
                                   font=dict(size=14, color=cfg['color']))
            else:
                fig.add_annotation(x=arrow_x, y=arrow_y, text=f"<b>y = {a:.2e} x² + {b:.2e} x + {c:.2e}</b>", 
                                   showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=cfg['color'], 
                                   ax=annot_x_pf, ay=annot_y_pf, axref="x", ayref="y", 
                                   bgcolor="rgba(255, 255, 255, 0.8)", bordercolor=cfg['color'], borderwidth=1, 
                                   font=dict(size=14, color=cfg['color']))

    # Prepare Export
    clean_label = legend_name.replace(" ", "_").replace("(", "").replace(")", "")
    export_df_dict[f"{clean_label}_X"] = x_data.values
    export_df_dict[f"{clean_label}_Y"] = y_data.values

# Layout Updates
final_title = custom_title if custom_title else f"{processed_entries[0]['ly']} vs {processed_entries[0]['lx']}"
final_xlabel = custom_xlabel if custom_xlabel else processed_entries[0]['lx']
final_ylabel = custom_ylabel if custom_ylabel else processed_entries[0]['ly']

final_template = template_mode if template_mode != "Auto (Global)" else "plotly"

# Add Text Annotations
if f"annotations_list_{plot_id}" in st.session_state:
    for annot in st.session_state[f"annotations_list_{plot_id}"]:
        if annot["text"]:
            styled_text = annot["text"]
            if annot.get("bold", False): styled_text = f"<b>{styled_text}</b>"
            if annot.get("italic", False): styled_text = f"<i>{styled_text}</i>"
            
            fig.add_annotation(
                x=annot["x"], y=annot["y"], text=styled_text, showarrow=False, 
                font=dict(size=annot["size"], color=annot["color"], family=annot.get("font", "Arial")), 
                bgcolor="rgba(255, 255, 255, 0.5)"
            )

fig.update_layout(
    title=dict(text=final_title, font=dict(size=title_font_size), x=0.5, xanchor='center'),
    xaxis=dict(title=dict(text=final_xlabel, font=dict(size=axis_title_size)), tickfont=dict(size=tick_font_size), 
               showgrid=show_grid, gridcolor=grid_color, range=[xlim_min, xlim_max] if use_xlim else None),
    yaxis=dict(title=dict(text=final_ylabel, font=dict(size=axis_title_size)), tickfont=dict(size=tick_font_size), 
               showgrid=show_grid, gridcolor=grid_color, range=[ylim_min, ylim_max] if use_ylim else None),
    showlegend=show_legend, hovermode="closest", height=height, width=width, template=final_template
)

# 4. Handle Selection/Click Events
selection = st.plotly_chart(fig, use_container_width=True, key=f"chart_{plot_id}", on_select="rerun", selection_mode="points")
if selection and selection.get("selection") and selection["selection"]["points"]:
    point = selection["selection"]["points"][0]
    st.session_state[f"last_click_{plot_id}"] = {"x": point["x"], "y": point["y"]}

# Export Buttons
if export_df_dict:
    with tab_export:
        export_msg_placeholder = st.empty() # Placeholder for cleanliness
        
        df_exp = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in export_df_dict.items() ]))
        
        # DAT Export
        dat_exp = df_exp.to_csv(index=False, sep='\t').encode('utf-8')
        st.download_button(
            label="Download Data (.dat)",
            data=dat_exp,
            file_name=f"plot_{plot_id}_data.dat",
            mime="text/plain",
            key=f"dl_dat_optimized_{plot_id}"
        )

        # CSV Export
        csv_exp = df_exp.to_csv(index=False, sep=',').encode('utf-8')
        st.download_button(
            label="Download Data (.csv)", 
            data=csv_exp, 
            file_name=f"plot_{plot_id}_data.csv", 
            mime="text/csv", 
            key=f"dl_csv_optimized_{plot_id}"
        )
        
        # HTML Export
        html_exp = fig.to_html(include_plotlyjs="cdn", full_html=True)
        st.download_button(
            label="Download Interactive Plot (.html)",
            data=html_exp,
            file_name=f"plot_{plot_id}.html",
            mime="text/html",
            key=f"dl_html_optimized_{plot_id}"
        )
        
return fig
