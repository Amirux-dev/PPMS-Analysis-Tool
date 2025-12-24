import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from modules.utils import persistent_selectbox, persistent_input, save_session_state
from modules.data_processing import choose_temperature_column

# -----------------------------------------------------------------------------
# PLOTTING INTERFACE
# -----------------------------------------------------------------------------

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
            if key.endswith(f"_{plot_id}"):
                base = key[:-len(str(plot_id))] # remove old id
                new_key = f"{base}{new_id}"
                new_entries[new_key] = val
            elif f"_{plot_id}_" in key:
                new_key = key.replace(f"_{plot_id}_", f"_{new_id}_")
                new_entries[new_key] = val
        p_store.update(new_entries)
    
    # Copy state
    for key in list(st.session_state.keys()):
        # Exclude buttons and temporary states
        if any(x in key for x in ["dup_", "add_btn_", "del_btn_", "ren_btn_"]): continue
        
        # Case 1: Key ends with _{plot_id} (Standard widgets)
        if key.endswith(f"_{plot_id}"):
            base = key[:-len(plot_id)] # remove old id (keep the underscore)
            new_key = f"{base}{new_id}"
            st.session_state[new_key] = st.session_state[key]
        
        # Case 2: Key contains _{plot_id}_ (Dynamic widgets like legends: leg_{plot_id}_{file_id})
        elif f"_{plot_id}_" in key:
            new_key = key.replace(f"_{plot_id}_", f"_{new_id}_")
            st.session_state[new_key] = st.session_state[key]
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
    """Creates a self-contained plotting interface and returns the figure."""
    
    with st.container(border=True):
        c_title, c_ren, c_add, c_rem, c_dup = st.columns([1.7, 1, 1, 1, 1], vertical_alignment="center", gap="small")
        
        with c_title:
            p_key = f"pname_{plot_id}"
            
            if 'persistent_values' in st.session_state and p_key in st.session_state['persistent_values']:
                plot_name = st.session_state['persistent_values'][p_key]
            else:
                plot_name = st.session_state.get(p_key, f"Plot {plot_id}")

            if st.session_state.get(f"ren_mode_{plot_id}", False):
                if p_key not in st.session_state:
                     st.session_state[p_key] = plot_name
                     
                st.text_input("Name", value=plot_name, key=p_key, label_visibility="collapsed", on_change=close_rename_callback, args=(plot_id,))
            else:
                st.markdown(f"<h3 style='margin: 0; padding: 0; line-height: 1.5;'>{plot_name}</h3>", unsafe_allow_html=True)
        
        with c_ren:
            st.button("✏️", key=f"ren_btn_{plot_id}", help="Rename Plot", on_click=toggle_rename_callback, args=(plot_id,), use_container_width=True)
        with c_add:
            st.button("➕", key=f"add_btn_{plot_id}", help="Add a new plot", on_click=add_plot_callback, use_container_width=True)
        with c_rem:
            st.button("➖", key=f"del_btn_{plot_id}", help="Remove this plot", on_click=remove_plot_callback, args=(plot_id,), use_container_width=True)
        with c_dup:
            st.button("📋", key=f"dup_{plot_id}", help="Duplicate this plot", on_click=duplicate_plot_callback, args=(plot_id,), use_container_width=True)
        
        analysis_mode = persistent_selectbox(
            "Analysis Mode",
            ["Custom Columns", "Standard MR Analysis", "Standard R-T Analysis"],
            index=0,
            persistent_key=f"mode_{plot_id}"
        )
        
        batch_map = get_batch_map(available_datasets, st.session_state.get('custom_batches', {}))
        
        batch_ids = sorted(batch_map.keys())
        options = ["ALL"] + batch_ids
        
        def format_batch(option):
            if option == "ALL":
                return "All Folders"
            return batch_map.get(option, f"Batch {option}")

        persistent_batch_key = f"batch_filter_{plot_id}"
        
        current_val = st.session_state.get(persistent_batch_key)
        
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
             
        widget_batch_key = f"batch_filter_{plot_id}_{st.session_state.uploader_key}"
        
        if widget_batch_key in st.session_state:
            st.session_state[persistent_batch_key] = st.session_state[widget_batch_key]
            
        selected_batch_id = st.selectbox(
            "Filter by Folder", 
            options, 
            format_func=format_batch,
            index=options.index(st.session_state[persistent_batch_key]),
            key=widget_batch_key,
            on_change=save_session_state
        )
        
        if selected_batch_id != "ALL":
            filtered_datasets = [d for d in available_datasets if d.get('batch_id') == selected_batch_id]
        else:
            filtered_datasets = available_datasets

        filtered_filenames = [d['fileName'] for d in filtered_datasets]
        
        persistent_sel_key = f"sel_{plot_id}"
        current_selection = st.session_state.get(persistent_sel_key, [])
        
        all_existing_files = {d['fileName'] for d in available_datasets}
        valid_selection = [f for f in current_selection if f in all_existing_files]
        
        if len(valid_selection) != len(current_selection):
            st.session_state[persistent_sel_key] = valid_selection
            current_selection = valid_selection
        
        combined_options = sorted(list(set(filtered_filenames + current_selection)))
        
        # Dynamic Widget Key for Multiselect
        # We append uploader_key to force the widget to re-mount when new files are added.
        # Without this, the 'options' list updates but the widget might not refresh its internal state correctly.
        widget_sel_key = f"sel_{plot_id}_{st.session_state.uploader_key}"
        
        if widget_sel_key in st.session_state:
            st.session_state[persistent_sel_key] = st.session_state[widget_sel_key]
        
        selected_filenames = st.multiselect(
            f"Select Curves for Plot {plot_id}", 
            options=combined_options,
            default=current_selection,
            key=widget_sel_key,
            on_change=save_session_state
        )
        
        selected_datasets = [d for d in available_datasets if d['fileName'] in selected_filenames]
        
        c1, c2 = st.columns([2, 1])
        
        custom_x_col = None
        custom_y_col = None
        
        if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"]:
            # R0 Method
            # Use Global Default if available
            default_r0_idx = 0
            global_r0 = st.session_state.get("global_r0_method")
            r0_options = ["Closest to 0T", "Mean within Window", "First Point", "Max Resistance"]
            if global_r0 in r0_options:
                default_r0_idx = r0_options.index(global_r0)

            r0_method = persistent_selectbox(
                "R0 Calculation Method",
                r0_options,
                index=default_r0_idx,
                persistent_key=f"r0_meth_{plot_id}"
            )
            r0_window = 0.01
            if r0_method == "Mean within Window":
                r0_window = st.number_input("Zero Field Window (T)", value=0.01, step=0.005, format="%.4f", key=f"r0_win_{plot_id}")

            with c1:
                if analysis_mode == "Standard MR Analysis":
                    y_opts = ["Magnetoresistance (MR %)", "Resistance (Ω)", "Normalized (R/R0)", "Derivative (dR/dH)"]
                else:
                    y_opts = ["Resistance (Ω)", "Normalized (R/R0)", "Derivative (dR/dH)"]

                y_axis_mode = persistent_selectbox(
                    "Y-Axis Mode",
                    y_opts,
                    index=0,
                    persistent_key=f"y_mode_{plot_id}"
                )
            with c2:
                # Use Global Default for Unit
                default_unit_idx = 0
                global_unit = st.session_state.get("global_field_unit")
                unit_options = ["Tesla (T)", "Oersted (Oe)"]
                if global_unit in unit_options:
                    default_unit_idx = unit_options.index(global_unit)

                x_axis_unit = persistent_selectbox(
                    "X-Axis Unit",
                    unit_options,
                    index=default_unit_idx,
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
                st.selectbox("X-Axis", ["Temperature (K)"], disabled=True, key=f"x_unit_{plot_id}")
        else:
            ref_cols = []
            
            if selected_datasets:
                df_ref = selected_datasets[0]['full_df']
                valid_cols = df_ref.dropna(axis=1, how='all').columns.tolist()
                ref_cols = valid_cols
            elif available_datasets:
                df_ref = available_datasets[0]['full_df']
                valid_cols = df_ref.dropna(axis=1, how='all').columns.tolist()
                ref_cols = valid_cols

            display_cols = list(ref_cols)
            has_oe = any("Oe" in c or "Oersted" in c for c in ref_cols)
            if has_oe:
                display_cols.append("Magnetic Field (T)")

            def get_smart_index(cols, keywords):
                for i, c in enumerate(cols):
                    if any(k in c.lower() for k in keywords):
                        return i
                return 0

            default_y_idx = get_smart_index(display_cols, ["resist", "ohm", "voltage"])
            default_x_idx = get_smart_index(display_cols, ["temp", "field", "tesla", "oe"])
            
            if default_x_idx == default_y_idx and len(display_cols) > 1:
                default_x_idx = (default_y_idx + 1) % len(display_cols)

            with c1:
                custom_y_col = persistent_selectbox("Y Column", display_cols, index=default_y_idx, persistent_key=f"y_col_{plot_id}")
            with c2:
                custom_x_col = persistent_selectbox("X Column", display_cols, index=default_x_idx, persistent_key=f"x_col_{plot_id}")
            
        st.markdown("###### Processing & Fits")
        c_smooth, c_fit1, c_fit2, c_proc = st.columns(4, vertical_alignment="bottom")
        
        with c_smooth:
            smooth_window = persistent_input(st.number_input, f"smooth_{plot_id}", label="Smoothing (pts)", min_value=0, value=0, step=1, help="Moving average window size.")
        
        with c_fit1:
            show_linear_fit = persistent_input(st.toggle, f"fit_{plot_id}", label="Linear Fit", value=False, help="Fit Y = aX + b")
        
        with c_fit2:
            show_parabolic_fit = persistent_input(st.toggle, f"pfit_{plot_id}", label="Parabolic Fit", value=False, help="Fit Y = aX² + bX + c")
            
        with c_proc:
            symmetrize_files = []
            if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"]:
                with st.popover("🪞 Symmetrize", use_container_width=True):
                    st.markdown("Select files to symmetrize:")
                    sym_options = {d['id']: d['fileName'] for d in selected_datasets}
                    
                    sym_key = f"sym_files_{plot_id}"
                    if 'persistent_values' not in st.session_state:
                        st.session_state['persistent_values'] = {}
                    
                    saved_sym = st.session_state['persistent_values'].get(sym_key, [])
                    valid_sym = [sid for sid in saved_sym if sid in sym_options]
                    
                    selected_sym_ids = st.multiselect(
                        "Files",
                        options=list(sym_options.keys()),
                        format_func=lambda x: sym_options[x],
                        default=valid_sym,
                        key=f"widget_sym_{plot_id}_{st.session_state.uploader_key}",
                        label_visibility="collapsed"
                    )
                    
                    if st.session_state['persistent_values'].get(sym_key) != selected_sym_ids:
                        st.session_state['persistent_values'][sym_key] = selected_sym_ids
                        save_session_state()
                        
                    symmetrize_files = selected_sym_ids
                
                plot_derivative = False
            elif analysis_mode == "Standard R-T Analysis":
                plot_derivative = False
            else:
                plot_derivative = persistent_input(st.toggle, f"deriv_{plot_id}", label="Derivative", value=False, help="Plot dY/dX vs X")

        fit_range_min = None
        fit_range_max = None
        pfit_range_min = None
        pfit_range_max = None
        
        if show_linear_fit or show_parabolic_fit:
            with st.popover("📏 Fit Settings", width='stretch'):
                if show_linear_fit:
                    st.markdown("**Linear Fit Range (X-Axis)**")
                    c_fmin, c_fmax = st.columns(2)
                    with c_fmin:
                        fit_range_min = persistent_input(st.number_input, f"fmin_{plot_id}", label="Min X (Linear)", value=None, placeholder="Start")
                    with c_fmax:
                        fit_range_max = persistent_input(st.number_input, f"fmax_{plot_id}", label="Max X (Linear)", value=None, placeholder="End")
                    st.caption("Leave empty to fit the entire range.")
                
                if show_parabolic_fit:
                    if show_linear_fit: st.markdown("---")
                    st.markdown("**Parabolic Fit Range (X-Axis)**")
                    c_pmin, c_pmax = st.columns(2)
                    with c_pmin:
                        pfit_range_min = persistent_input(st.number_input, f"pfmin_{plot_id}", label="Min X (Parabolic)", value=None, placeholder="Start")
                    with c_pmax:
                        pfit_range_max = persistent_input(st.number_input, f"pfmax_{plot_id}", label="Max X (Parabolic)", value=None, placeholder="End")
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
            items_to_plot = [] # List of tuples: (x_data, y_data, x_label, y_label, suffix)

            if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"]:
                # Base Data
                df_base = pd.DataFrame({"H_T": d["H_T"], "R": d["R"]})
                
                # List of dataframes to process: (df, suffix)
                dfs_to_process = [(df_base, "")]
                
                # Check if this specific file is selected for symmetrization
                if d['id'] in symmetrize_files:
                    # Simple Mirror Logic: (H, R) -> (-H, R)
                    # Plots the symmetric of the curve with respect to the Y-axis
                    df_sym = df_base.copy()
                    df_sym["H_T"] = -df_sym["H_T"]
                    
                    dfs_to_process.append((df_sym, " (Sym)"))

                for df, suffix in dfs_to_process:
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
                    elif r0_method == "Max Resistance":
                        r0 = df["R"].max()

                    # Calculate X
                    x_d = df["H_T"]
                    x_l = "Field (T)"
                    if x_axis_unit == "Oersted (Oe)":
                        x_d = x_d * 10000
                        x_l = "Field (Oe)"

                    # Calculate Y
                    y_d = None
                    y_l = ""
                    
                    if y_axis_mode == "Magnetoresistance (MR %)":
                        y_d = 100 * (df["R"] - r0) / r0
                        y_l = "MR (%)"
                    elif y_axis_mode == "Normalized (R/R0)":
                        y_d = df["R"] / r0
                        y_l = "R / R0"
                    elif y_axis_mode == "Derivative (dR/dH)":
                        dy = df["R"].diff()
                        dx = df["H_T"].diff()
                        y_d = dy / dx
                        y_l = "dR/dH (Ω/T)"
                        y_d = y_d.fillna(0)
                    else: # Resistance
                        y_d = df["R"]
                        y_l = "Resistance (Ω)"
                    
                    items_to_plot.append((x_d, y_d, x_l, y_l, suffix))
            
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
                
                y_data = None
                y_label = ""

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
                
                items_to_plot.append((x_data, y_data, x_label, y_label, ""))

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
                
                items_to_plot.append((x_data, y_data, x_label, y_label, ""))

            # --- PLOTTING LOOP ---
            for x_data, y_data, x_label, y_label, suffix in items_to_plot:
                if x_data is None or y_data is None:
                    continue

                # Smoothing
                if smooth_window > 1:
                    y_data = y_data.rolling(window=int(smooth_window), center=True).mean()

                # Plot Trace
                mode_map = {"Lines": "lines", "Markers": "markers", "Lines+Markers": "lines+markers"}
                
                # Legend Name
                base_legend = custom_legends.get(d['id'], d['fileName'])
                legend_name = f"{base_legend}{suffix}"
                
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode=mode_map[plot_mode],
                    name=legend_name,
                    hovertemplate=f"{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4e}}<extra></extra>",
                    line=dict(width=line_width, dash='dash' if suffix else None) if "Lines" in plot_mode else None,
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

                # Parabolic Fit
                if show_parabolic_fit and x_data is not None and y_data is not None:
                    # Remove NaNs for fitting
                    mask_pfit = x_data.notna() & y_data.notna()
                    
                    # Apply Range Filter
                    if pfit_range_min is not None:
                        mask_pfit &= (x_data >= pfit_range_min)
                    if pfit_range_max is not None:
                        mask_pfit &= (x_data <= pfit_range_max)
                    
                    xpf = x_data[mask_pfit]
                    ypf = y_data[mask_pfit]
                    
                    if len(xpf) > 2: # Need at least 3 points for parabola
                        # Polyfit degree 2
                        a, b, c = np.polyfit(xpf, ypf, 2)
                        y_pfit = a * xpf**2 + b * xpf + c
                        
                        # Plot the fit line only within the range
                        fig.add_trace(go.Scatter(
                            x=xpf,
                            y=y_pfit,
                            mode='lines',
                            name=f"ParaFit {legend_name}",
                            line=dict(dash='dot', width=3, color='green'),
                            hoverinfo='skip'
                        ))
                        
                        # Enhanced Annotation
                        eq_text = f"<b>y = {a:.2e} x² + {b:.2e} x + {c:.2e}</b>"
                        
                        # Position annotation near the center of the fit segment
                        mid_idx = len(xpf) // 2
                        
                        fig.add_annotation(
                            x=xpf.iloc[mid_idx],
                            y=y_pfit.iloc[mid_idx],
                            text=eq_text,
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            ax=0,
                            ay=40, # Offset downwards
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="blue",
                            borderwidth=1,
                            font=dict(size=14, color="blue")
                        )

                # Add to export
                clean_label = d['label'].replace(" ", "_") + suffix.replace(" ", "_").replace("(", "").replace(")", "")
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
        
        # --- Statistics Table ---
        if export_data:
            with st.expander("📊 Statistics Table", expanded=False):
                stats_data = []
                for key, val in export_data.items():
                    if key.endswith("_Y"):
                        label = key[:-2] # Remove _Y
                        y_vals = val
                        x_vals = export_data.get(f"{label}_X", [])
                        
                        if len(y_vals) > 0:
                            stats_data.append({
                                "Curve": label,
                                "Min X": f"{np.min(x_vals):.4g}" if len(x_vals) > 0 else "-",
                                "Max X": f"{np.max(x_vals):.4g}" if len(x_vals) > 0 else "-",
                                "Min Y": f"{np.min(y_vals):.4g}",
                                "Max Y": f"{np.max(y_vals):.4g}",
                                "Mean Y": f"{np.mean(y_vals):.4g}",
                                "Std Dev": f"{np.std(y_vals):.4g}"
                            })
                
                if stats_data:
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

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
