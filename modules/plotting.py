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
        # --- Header Row ---
        c_title, c_ren, c_add, c_rem, c_dup = st.columns([1.7, 1, 1, 1, 1], vertical_alignment="center", gap="small")
        
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
                st.markdown(f"<h3 style='margin: 0; padding: 0; line-height: 1.5;'>{plot_name}</h3>", unsafe_allow_html=True)
        
        with c_ren: st.button("✏️", key=f"ren_btn_{plot_id}", help="Rename Plot", on_click=toggle_rename_callback, args=(plot_id,), use_container_width=True)
        with c_add: st.button("➕", key=f"add_btn_{plot_id}", help="Add a new plot", on_click=add_plot_callback, use_container_width=True)
        with c_rem: st.button("➖", key=f"del_btn_{plot_id}", help="Remove this plot", on_click=remove_plot_callback, args=(plot_id,), use_container_width=True)
        with c_dup: st.button("📋", key=f"dup_{plot_id}", help="Duplicate this plot", on_click=duplicate_plot_callback, args=(plot_id,), use_container_width=True)
        
        # --- Global Settings for this Plot ---
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
        
        if st.session_state.get(persistent_batch_key) not in options:
             st.session_state[persistent_batch_key] = "ALL"
             
        widget_batch_key = f"batch_filter_{plot_id}_{st.session_state.uploader_key}"
        if widget_batch_key in st.session_state:
            st.session_state[persistent_batch_key] = st.session_state[widget_batch_key]
            
        selected_batch_id = st.selectbox("Filter by Folder", options, format_func=format_batch, index=options.index(st.session_state[persistent_batch_key]), key=widget_batch_key, on_change=save_session_state)
        
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
        widget_sel_key = f"sel_{plot_id}_{st.session_state.uploader_key}"
        
        if widget_sel_key in st.session_state:
            st.session_state[persistent_sel_key] = st.session_state[widget_sel_key]
        
        selected_filenames = st.multiselect(f"Select Curves for {plot_name}", options=combined_options, default=current_selection, key=widget_sel_key, on_change=save_session_state)
        selected_datasets = [d for d in available_datasets if d['fileName'] in selected_filenames]

        # --- TABS LAYOUT ---
        tab_data, tab_analysis, tab_style, tab_export = st.tabs(["📊 Data", "📈 Analysis", "🎨 Styling", "💾 Export"])

        # Initialize variables to ensure scope availability
        custom_x_col, custom_y_col = None, None
        r0_method, r0_window = "First Point", 0.01
        y_axis_mode, x_axis_unit = "Resistance (Ω)", "Tesla (T)"
        
        show_linear_fit, show_parabolic_fit = False, False
        fit_range_min, fit_range_max = None, None
        pfit_range_min, pfit_range_max = None, None
        linear_fit_settings = {}
        parabolic_fit_settings = {}
        symmetrize_files = []
        plot_derivative = False
        
        curve_settings = {}
        custom_legends = {}
        
        annot_text, annot_x, annot_y, annot_color, annot_size = "", 0.0, 0.0, "#000000", 14
        
        custom_title, title_font_size = "", 20
        template_mode, show_legend = "Auto (Global)", True
        plot_mode, line_width, marker_size = "Lines", 2.0, 6
        custom_xlabel, axis_title_size, use_xlim, xlim_min, xlim_max = "", 16, False, -9.0, 9.0
        custom_ylabel, tick_font_size, use_ylim, ylim_min, ylim_max = "", 14, False, 0.0, 100.0
        show_grid, grid_color = True, "#E5E5E5"

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
                if selected_datasets:
                    df_ref = selected_datasets[0]['full_df']
                    ref_cols = df_ref.dropna(axis=1, how='all').columns.tolist()
                elif available_datasets:
                    df_ref = available_datasets[0]['full_df']
                    ref_cols = df_ref.dropna(axis=1, how='all').columns.tolist()

                display_cols = list(ref_cols)
                if any("Oe" in c or "Oersted" in c for c in ref_cols): display_cols.append("Magnetic Field (T)")

                def get_smart_index(cols, keywords):
                    for i, c in enumerate(cols):
                        if any(k in c.lower() for k in keywords): return i
                    return 0

                default_y_idx = get_smart_index(display_cols, ["resist", "ohm", "voltage"])
                default_x_idx = get_smart_index(display_cols, ["temp", "field", "tesla", "oe"])
                if default_x_idx == default_y_idx and len(display_cols) > 1: default_x_idx = (default_y_idx + 1) % len(display_cols)

                with c1: custom_y_col = persistent_selectbox("Y Column", display_cols, index=default_y_idx, persistent_key=f"y_col_{plot_id}")
                with c2: custom_x_col = persistent_selectbox("X Column", display_cols, index=default_x_idx, persistent_key=f"x_col_{plot_id}")

        # --- TAB 2: ANALYSIS ---
        with tab_analysis:
            c_fit1, c_fit2, c_proc = st.columns(3)
            with c_fit1: show_linear_fit = persistent_input(st.toggle, f"fit_{plot_id}", label="Linear Fit", value=False, help="Fit Y = aX + b")
            with c_fit2: show_parabolic_fit = persistent_input(st.toggle, f"pfit_{plot_id}", label="Parabolic Fit", value=False, help="Fit Y = aX² + bX + c")
            with c_proc:
                if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"]:
                    with st.popover("🪞 Symmetrize", use_container_width=True):
                        st.markdown("Select files to symmetrize:")
                        sym_options = {d['id']: d['fileName'] for d in selected_datasets}
                        sym_key = f"sym_files_{plot_id}"
                        if 'persistent_values' not in st.session_state: st.session_state['persistent_values'] = {}
                        saved_sym = st.session_state['persistent_values'].get(sym_key, [])
                        valid_sym = [sid for sid in saved_sym if sid in sym_options]
                        selected_sym_ids = st.multiselect("Files", options=list(sym_options.keys()), format_func=lambda x: sym_options[x], default=valid_sym, key=f"widget_sym_{plot_id}_{st.session_state.uploader_key}", label_visibility="collapsed")
                        if st.session_state['persistent_values'].get(sym_key) != selected_sym_ids:
                            st.session_state['persistent_values'][sym_key] = selected_sym_ids
                            save_session_state()
                        symmetrize_files = selected_sym_ids
                    plot_derivative = False
                elif analysis_mode == "Standard R-T Analysis":
                    plot_derivative = False
                else:
                    plot_derivative = persistent_input(st.toggle, f"deriv_{plot_id}", label="Derivative", value=False, help="Plot dY/dX vs X")

            if show_linear_fit or show_parabolic_fit:
                st.markdown("###### Fit Settings")
                if show_linear_fit:
                    with st.expander("Linear Fit Configuration", expanded=True):
                        fit_options = {d['id']: d['fileName'] for d in selected_datasets}
                        fit_sel_key = f"fit_sel_{plot_id}"
                        saved_fit_sel = st.session_state['persistent_values'].get(fit_sel_key, [])
                        valid_fit_sel = [sid for sid in saved_fit_sel if sid in fit_options]
                        selected_fit_ids = st.multiselect("Select Curves to Fit", options=list(fit_options.keys()), format_func=lambda x: fit_options[x], default=valid_fit_sel, key=f"widget_fit_sel_{plot_id}_{st.session_state.uploader_key}")
                        
                        if st.session_state['persistent_values'].get(fit_sel_key) != selected_fit_ids:
                            st.session_state['persistent_values'][fit_sel_key] = selected_fit_ids
                            save_session_state()
                        
                        c_fmin, c_fmax = st.columns(2)
                        with c_fmin: fit_range_min = persistent_input(st.number_input, f"fmin_{plot_id}", label="Min X", value=None, placeholder="Start")
                        with c_fmax: fit_range_max = persistent_input(st.number_input, f"fmax_{plot_id}", label="Max X", value=None, placeholder="End")
                        
                        for fid in selected_fit_ids:
                            st.caption(f"Settings for: {fit_options[fid]}")
                            c_fc, c_fs = st.columns(2)
                            with c_fc: f_color = persistent_input(st.color_picker, f"fit_col_{plot_id}_{fid}", label="Line Color", value="#FF0000")
                            
                            c_ax, c_ay, c_btn = st.columns([1, 1, 1])
                            with c_ax: f_annot_x = persistent_input(st.number_input, f"fit_ax_{plot_id}_{fid}", label="Annot X", value=None, placeholder="Auto")
                            with c_ay: f_annot_y = persistent_input(st.number_input, f"fit_ay_{plot_id}_{fid}", label="Annot Y", value=None, placeholder="Auto")
                            with c_btn:
                                st.write("")
                                st.write("")
                                
                                def paste_fit_callback(pid, fid):
                                    lc = st.session_state.get(f"last_click_{pid}")
                                    if lc:
                                        if 'persistent_values' not in st.session_state: st.session_state['persistent_values'] = {}
                                        st.session_state['persistent_values'][f"fit_ax_{pid}_{fid}"] = lc["x"]
                                        st.session_state['persistent_values'][f"fit_ay_{pid}_{fid}"] = lc["y"]

                                st.button("📍 Paste", key=f"paste_fit_{plot_id}_{fid}", help="Paste clicked coordinates", on_click=paste_fit_callback, args=(plot_id, fid))

                            linear_fit_settings[fid] = {"color": f_color, "annot_x": f_annot_x, "annot_y": f_annot_y}
                
                if show_parabolic_fit:
                    with st.expander("Parabolic Fit Configuration", expanded=True):
                        fit_options = {d['id']: d['fileName'] for d in selected_datasets}
                        pfit_sel_key = f"pfit_sel_{plot_id}"
                        saved_pfit_sel = st.session_state['persistent_values'].get(pfit_sel_key, [])
                        valid_pfit_sel = [sid for sid in saved_pfit_sel if sid in fit_options]
                        selected_pfit_ids = st.multiselect("Select Curves to Fit", options=list(fit_options.keys()), format_func=lambda x: fit_options[x], default=valid_pfit_sel, key=f"widget_pfit_sel_{plot_id}_{st.session_state.uploader_key}")
                        
                        if st.session_state['persistent_values'].get(pfit_sel_key) != selected_pfit_ids:
                            st.session_state['persistent_values'][pfit_sel_key] = selected_pfit_ids
                            save_session_state()

                        c_pmin, c_pmax = st.columns(2)
                        with c_pmin: pfit_range_min = persistent_input(st.number_input, f"pfmin_{plot_id}", label="Min X", value=None, placeholder="Start")
                        with c_pmax: pfit_range_max = persistent_input(st.number_input, f"pfmax_{plot_id}", label="Max X", value=None, placeholder="End")

                        for fid in selected_pfit_ids:
                            st.caption(f"Settings for: {fit_options[fid]}")
                            c_fc, c_fs = st.columns(2)
                            with c_fc: pf_color = persistent_input(st.color_picker, f"pfit_col_{plot_id}_{fid}", label="Line Color", value="#00FF00")
                            
                            c_ax, c_ay, c_btn = st.columns([1, 1, 1])
                            with c_ax: pf_annot_x = persistent_input(st.number_input, f"pfit_ax_{plot_id}_{fid}", label="Annot X", value=None, placeholder="Auto")
                            with c_ay: pf_annot_y = persistent_input(st.number_input, f"pfit_ay_{plot_id}_{fid}", label="Annot Y", value=None, placeholder="Auto")
                            with c_btn:
                                st.write("")
                                st.write("")
                                
                                def paste_pfit_callback(pid, fid):
                                    lc = st.session_state.get(f"last_click_{pid}")
                                    if lc:
                                        if 'persistent_values' not in st.session_state: st.session_state['persistent_values'] = {}
                                        st.session_state['persistent_values'][f"pfit_ax_{pid}_{fid}"] = lc["x"]
                                        st.session_state['persistent_values'][f"pfit_ay_{pid}_{fid}"] = lc["y"]

                                st.button("📍 Paste", key=f"paste_pfit_{plot_id}_{fid}", help="Paste clicked coordinates", on_click=paste_pfit_callback, args=(plot_id, fid))

                            parabolic_fit_settings[fid] = {"color": pf_color, "annot_x": pf_annot_x, "annot_y": pf_annot_y}

        # --- PRE-CALCULATE DATA RANGES FOR ANNOTATION SCALING ---
        global_x_min, global_x_max = float('inf'), float('-inf')
        global_y_min, global_y_max = float('inf'), float('-inf')
        
        # We need to do a quick pass to estimate ranges without full plotting overhead
        # This is a simplified version of the plotting loop logic just for ranges
        if selected_datasets:
            for d in selected_datasets:
                # Simplified logic to extract X/Y based on mode
                x_temp, y_temp = None, None
                
                if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"]:
                    df = pd.DataFrame({"H_T": d["H_T"], "R": d["R"]})
                    r0 = 1.0
                    if r0_method == "First Point": r0 = df["R"].iloc[0]
                    elif r0_method == "Closest to 0T": r0 = df["R"].iloc[df["H_T"].abs().idxmin()]
                    elif r0_method == "Mean within Window":
                        mask = df["H_T"].abs() <= r0_window
                        r0 = df.loc[mask, "R"].mean() if mask.any() else df["R"].iloc[df["H_T"].abs().idxmin()]
                    elif r0_method == "Max Resistance": r0 = df["R"].max()

                    x_temp = df["H_T"] * (10000 if x_axis_unit == "Oersted (Oe)" else 1)
                    if y_axis_mode == "Magnetoresistance (MR %)": y_temp = 100 * (df["R"] - r0) / r0
                    elif y_axis_mode == "Normalized (R/R0)": y_temp = df["R"] / r0
                    elif y_axis_mode == "Derivative (dR/dH)": y_temp = df["R"].diff() / df["H_T"].diff()
                    else: y_temp = df["R"]
                
                elif analysis_mode == "Standard R-T Analysis":
                    if 'full_df' in d and d['rCol'] in d['full_df'].columns:
                        full_df = d['full_df']
                        cols = full_df.columns.tolist()
                        temp_idx = choose_temperature_column(cols)
                        if temp_idx >= 0:
                            df = pd.DataFrame({"T": full_df[cols[temp_idx]], "R": full_df[d['rCol']]}).dropna()
                            x_temp = df["T"]
                            if y_axis_mode == "Resistance (Ω)": y_temp = df["R"]
                            elif y_axis_mode == "Normalized (R/R_300K)":
                                r_300 = df.loc[(df["T"] - 300).abs().idxmin(), "R"]
                                y_temp = df["R"] / r_300
                            elif y_axis_mode == "Derivative (dR/dT)": y_temp = df["R"].diff() / df["T"].diff()

                else: # Custom
                    if 'full_df' in d:
                        full_df = d['full_df']
                        # Helper to get col data (simplified)
                        def get_col_simple(cname, df):
                            if cname == "Magnetic Field (T)":
                                for c in df.columns:
                                    if "Oe" in c or "Oersted" in c: return df[c] * 1e-4
                                return None
                            return df[cname] if cname in df.columns else None
                        
                        x_temp = get_col_simple(custom_x_col, full_df)
                        y_temp = get_col_simple(custom_y_col, full_df)
                        if plot_derivative and x_temp is not None and y_temp is not None:
                             # Very rough derivative range estimation
                             temp_df = pd.DataFrame({'x': x_temp, 'y': y_temp}).sort_values('x')
                             y_temp = temp_df['y'].diff() / temp_df['x'].diff()

                if x_temp is not None and not x_temp.empty:
                    global_x_min = min(global_x_min, x_temp.min())
                    global_x_max = max(global_x_max, x_temp.max())
                if y_temp is not None and not y_temp.empty:
                    global_y_min = min(global_y_min, y_temp.min())
                    global_y_max = max(global_y_max, y_temp.max())

        # Default ranges if no data
        if global_x_min == float('inf'): global_x_min, global_x_max = -10.0, 10.0
        if global_y_min == float('inf'): global_y_min, global_y_max = 0.0, 100.0

        # Determine format string based on range
        def get_smart_format(val_min, val_max):
            val_range = abs(val_max - val_min)
            if val_range == 0: return "%.4f"
            if val_range < 1e-3: return "%.4e"
            if val_range < 1: return "%.6f"
            if val_range > 1000: return "%.2f"
            return "%.4f"

        def get_smart_step(val_min, val_max):
            val_range = abs(val_max - val_min)
            if val_range == 0: return 0.1
            if val_range < 1e-5: return 1e-7
            if val_range < 1e-3: return 1e-5
            if val_range < 1: return 0.001
            if val_range < 10: return 0.01
            if val_range < 100: return 0.1
            return 1.0

        x_fmt = get_smart_format(global_x_min, global_x_max)
        y_fmt = get_smart_format(global_y_min, global_y_max)
        x_step = get_smart_step(global_x_min, global_x_max)
        y_step = get_smart_step(global_y_min, global_y_max)

        # --- TAB 3: STYLING ---
        with tab_style:
            st.markdown("###### Curve Styling")
            if not selected_datasets:
                st.info("Select curves first.")
            else:
                for d in selected_datasets:
                    with st.expander(f"Curve: {d['fileName']}", expanded=False):
                        c_col1, c_col2, c_col3 = st.columns([1, 1, 2])
                        with c_col1:
                            use_custom_color = persistent_input(st.checkbox, f"use_col_{plot_id}_{d['id']}", label="Custom Color", value=False)
                        with c_col2:
                            if use_custom_color:
                                curve_color = persistent_input(st.color_picker, f"color_{plot_id}_{d['id']}", label="Pick Color", value="#000000")
                            else:
                                curve_color = None
                        with c_col3:
                            curve_smooth = persistent_input(st.number_input, f"smooth_{plot_id}_{d['id']}", label="Smoothing (pts)", min_value=0, value=0, step=1)
                        
                        # Legend Label
                        default_leg = d['label']
                        custom_leg = persistent_input(st.text_input, f"leg_{plot_id}_{d['id']}", label="Legend Label", value=default_leg)
                        custom_legends[d['id']] = custom_leg
                        
                        curve_settings[d['id']] = {"color": curve_color, "smoothing": curve_smooth}

            st.markdown("---")
            st.markdown("##### Global Plot Settings")

            with st.expander("**Plot Appearance**", expanded=False):
                col_cust1, col_cust2, col_cust3 = st.columns(3)
                with col_cust1:
                    custom_title = persistent_input(st.text_input, f"title_{plot_id}", label="Plot Title", value="", placeholder="Auto")
                    title_font_size = persistent_input(st.number_input, f"title_font_{plot_id}", label="Title Size", value=20, min_value=10)
                with col_cust2:
                    template_mode = persistent_selectbox("Theme", ["Auto (Global)", "plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"], index=0, persistent_key=f"theme_{plot_id}")
                    show_legend = persistent_input(st.checkbox, f"legend_{plot_id}", label="Show Legend", value=True)
                with col_cust3:
                    plot_mode = persistent_selectbox("Style", ["Lines", "Markers", "Lines+Markers"], index=0, persistent_key=f"style_{plot_id}")
                    if "Lines" in plot_mode: line_width = persistent_input(st.number_input, f"lw_{plot_id}", label="Line Width", value=2.0, step=0.5)
                    if "Markers" in plot_mode: marker_size = persistent_input(st.number_input, f"ms_{plot_id}", label="Marker Size", value=6, step=1)

            with st.expander("**Axes**", expanded=False):
                col_cust4, col_cust5, col_cust6 = st.columns(3)
                with col_cust4:
                    custom_xlabel = persistent_input(st.text_input, f"xlabel_{plot_id}", label="X Label", value="", placeholder="Auto")
                    axis_title_size = persistent_input(st.number_input, f"axis_title_font_{plot_id}", label="Label Size", value=16)
                    use_xlim = persistent_input(st.checkbox, f"use_xlim_{plot_id}", label="Set X Limits")
                    if use_xlim:
                        c_xmin, c_xmax = st.columns(2)
                        with c_xmin: xlim_min = persistent_input(st.number_input, f"xlim_min_{plot_id}", label="Min", value=-9.0, format="%.2f")
                        with c_xmax: xlim_max = persistent_input(st.number_input, f"xlim_max_{plot_id}", label="Max", value=9.0, format="%.2f")
                    else: xlim_min, xlim_max = None, None

                with col_cust5:
                    custom_ylabel = persistent_input(st.text_input, f"ylabel_{plot_id}", label="Y Label", value="", placeholder="Auto")
                    tick_font_size = persistent_input(st.number_input, f"tick_font_{plot_id}", label="Tick Size", value=14)
                    use_ylim = persistent_input(st.checkbox, f"use_ylim_{plot_id}", label="Set Y Limits")
                    if use_ylim:
                        c_ymin, c_ymax = st.columns(2)
                        with c_ymin: ylim_min = persistent_input(st.number_input, f"ylim_min_{plot_id}", label="Min", value=0.0, format="%.2e")
                        with c_ymax: ylim_max = persistent_input(st.number_input, f"ylim_max_{plot_id}", label="Max", value=100.0, format="%.2e")
                    else: ylim_min, ylim_max = None, None

                with col_cust6:
                    show_grid = persistent_input(st.checkbox, f"grid_{plot_id}", label="Show Grid", value=True)
                    grid_color = persistent_input(st.color_picker, f"grid_color_{plot_id}", label="Grid Color", value="#E5E5E5")

            with st.expander("**Text Annotation**", expanded=False):
                if f"annotations_list_{plot_id}" not in st.session_state:
                    st.session_state[f"annotations_list_{plot_id}"] = []

                if st.button("Add Annotation", key=f"add_annot_btn_{plot_id}"):
                    st.session_state[f"annotations_list_{plot_id}"].append({
                        "text": "New Text", "x": 0.0, "y": 0.0, "color": "#000000", "size": 14,
                        "bold": False, "italic": False, "font": "Arial"
                    })
                    st.rerun()

                to_delete = []
                for i, annot in enumerate(st.session_state[f"annotations_list_{plot_id}"]):
                    # Ensure new keys exist for old annotations
                    annot.setdefault("bold", False)
                    annot.setdefault("italic", False)
                    annot.setdefault("font", "Arial")

                    st.markdown(f"**Annotation {i+1}**")
                    c1, c2 = st.columns([0.85, 0.15], vertical_alignment="bottom")
                    with c1:
                        annot["text"] = st.text_input(f"Text", value=annot["text"], key=f"annot_txt_{plot_id}_{i}")
                    with c2:
                        if st.button("🗑️", key=f"del_annot_{plot_id}_{i}"):
                            to_delete.append(i)
                    
                    c_xy1, c_xy2, c_btn = st.columns([1, 1, 1], vertical_alignment="bottom")
                    with c_xy1:
                        annot["x"] = st.number_input("X", value=float(annot["x"]), format=x_fmt, step=x_step, key=f"annot_x_{plot_id}_{i}")
                    with c_xy2:
                        annot["y"] = st.number_input("Y", value=float(annot["y"]), format=y_fmt, step=y_step, key=f"annot_y_{plot_id}_{i}")
                    with c_btn:
                        last_click = st.session_state.get(f"last_click_{plot_id}")
                        help_text = "1. Click a data point on the plot.\n2. Click this button to paste coordinates."
                        if last_click:
                            st.caption(f"Sel: {last_click['x']:.2f}, {last_click['y']:.2f}")
                        else:
                            st.caption("No point selected")
                        
                        def paste_callback(pid, idx):
                            lc = st.session_state.get(f"last_click_{pid}")
                            if lc:
                                st.session_state[f"annot_x_{pid}_{idx}"] = lc["x"]
                                st.session_state[f"annot_y_{pid}_{idx}"] = lc["y"]

                        st.button("📍 Paste Click", key=f"paste_click_{plot_id}_{i}", help=help_text, on_click=paste_callback, args=(plot_id, i))

                    c_col, c_size = st.columns(2)
                    with c_col:
                        annot["color"] = st.color_picker("Color", value=annot["color"], key=f"annot_col_{plot_id}_{i}")
                    with c_size:
                        annot["size"] = st.number_input("Size", value=int(annot["size"]), min_value=5, key=f"annot_sz_{plot_id}_{i}")
                    
                    c_font, c_style = st.columns([2, 1], vertical_alignment="bottom")
                    with c_font:
                        annot["font"] = st.selectbox("Font", ["Arial", "Times New Roman", "Courier New", "Verdana", "Georgia"], index=0 if annot["font"] == "Arial" else 1, key=f"annot_font_{plot_id}_{i}")
                    with c_style:
                        c_b, c_i = st.columns(2)
                        with c_b: annot["bold"] = st.checkbox("Bold", value=annot["bold"], key=f"annot_bold_{plot_id}_{i}")
                        with c_i: annot["italic"] = st.checkbox("Italic", value=annot["italic"], key=f"annot_italic_{plot_id}_{i}")

                    st.divider()

                if to_delete:
                    for index in sorted(to_delete, reverse=True):
                        del st.session_state[f"annotations_list_{plot_id}"][index]
                    st.rerun()

        # --- TAB 4: EXPORT ---
        with tab_export:
            st.markdown("###### Export Data")
            export_msg_placeholder = st.empty()
            export_msg_placeholder.info("Download options will appear here after the plot is generated.")

        if not selected_datasets:
            st.info("Select at least one file to display the plot.")
            return None

        # --- PLOT GENERATION ---
        fig = go.Figure()
        export_data = {}

        for d in selected_datasets:
            items_to_plot = [] # List of tuples: (x_data, y_data, x_label, y_label, suffix)

            if analysis_mode in ["Standard MR Analysis", "Standard R-H Analysis"]:
                df_base = pd.DataFrame({"H_T": d["H_T"], "R": d["R"]})
                dfs_to_process = [(df_base, "")]
                if d['id'] in symmetrize_files:
                    df_sym = df_base.copy()
                    df_sym["H_T"] = -df_sym["H_T"]
                    dfs_to_process.append((df_sym, " (Sym)"))

                for df, suffix in dfs_to_process:
                    r0 = 1.0
                    if r0_method == "First Point": r0 = df["R"].iloc[0]
                    elif r0_method == "Closest to 0T": r0 = df["R"].iloc[df["H_T"].abs().idxmin()]
                    elif r0_method == "Mean within Window":
                        mask = df["H_T"].abs() <= r0_window
                        r0 = df.loc[mask, "R"].mean() if mask.any() else df["R"].iloc[df["H_T"].abs().idxmin()]
                    elif r0_method == "Max Resistance": r0 = df["R"].max()

                    x_d, x_l = df["H_T"], "Field (T)"
                    if x_axis_unit == "Oersted (Oe)": x_d, x_l = x_d * 10000, "Field (Oe)"

                    y_d, y_l = None, ""
                    if y_axis_mode == "Magnetoresistance (MR %)": y_d, y_l = 100 * (df["R"] - r0) / r0, "MR (%)"
                    elif y_axis_mode == "Normalized (R/R0)": y_d, y_l = df["R"] / r0, "R / R0"
                    elif y_axis_mode == "Derivative (dR/dH)":
                        y_d, y_l = df["R"].diff() / df["H_T"].diff(), "dR/dH (Ω/T)"
                        y_d = y_d.fillna(0)
                    else: y_d, y_l = df["R"], "Resistance (Ω)"
                    
                    items_to_plot.append((x_d, y_d, x_l, y_l, suffix))
            
            elif analysis_mode == "Standard R-T Analysis":
                if 'full_df' not in d: continue
                full_df = d['full_df']
                cols = full_df.columns.tolist()
                temp_idx = choose_temperature_column(cols)
                if temp_idx < 0: continue
                temp_col = cols[temp_idx]
                r_col = d['rCol']
                if r_col not in full_df.columns: continue
                
                df = pd.DataFrame({"T": full_df[temp_col], "R": full_df[r_col]}).dropna().sort_values("T")
                x_data, x_label = df["T"], "Temperature (K)"
                y_data, y_label = None, ""

                if y_axis_mode == "Resistance (Ω)": y_data, y_label = df["R"], "Resistance (Ω)"
                elif y_axis_mode == "Normalized (R/R_300K)":
                    r_300 = df.loc[(df["T"] - 300).abs().idxmin(), "R"]
                    y_data, y_label = df["R"] / r_300, "R / R(300K)"
                elif y_axis_mode == "Derivative (dR/dT)":
                    y_data, y_label = (df["R"].diff() / df["T"].diff()).fillna(0), "dR/dT (Ω/K)"
                
                items_to_plot.append((x_data, y_data, x_label, y_label, ""))

            else: # Custom Columns
                if 'full_df' not in d: continue
                full_df = d['full_df']
                def get_col_data(col_name, df):
                    if col_name == "Magnetic Field (T)":
                        for c in df.columns:
                            if "Oe" in c or "Oersted" in c: return df[c] * 1e-4, "Magnetic Field (T)"
                        return None, None
                    elif col_name in df.columns: return df[col_name], col_name
                    return None, None

                x_data, x_label = get_col_data(custom_x_col, full_df)
                y_data, y_label = get_col_data(custom_y_col, full_df)

                if x_data is not None and y_data is not None:
                    mask = x_data.notna() & y_data.notna()
                    x_data, y_data = x_data[mask], y_data[mask]

                    if plot_derivative:
                        temp_df = pd.DataFrame({'x': x_data, 'y': y_data}).sort_values('x')
                        deriv = temp_df['y'].diff() / temp_df['x'].diff()
                        deriv = deriv.replace([np.inf, -np.inf], np.nan)
                        x_data, y_data, y_label = temp_df['x'], deriv, f"d({y_label})/d({x_label})"
                        mask_d = y_data.notna()
                        x_data, y_data = x_data[mask_d], y_data[mask_d]
                    
                    items_to_plot.append((x_data, y_data, x_label, y_label, ""))

            # --- PLOTTING LOOP ---
            for x_data, y_data, x_label, y_label, suffix in items_to_plot:
                if x_data is None or y_data is None: continue

                c_settings = curve_settings.get(d['id'], {})
                c_color = c_settings.get("color")
                c_smooth = c_settings.get("smoothing", 0)

                if c_smooth > 1: y_data = y_data.rolling(window=int(c_smooth), center=True).mean()

                mode_map = {"Lines": "lines", "Markers": "markers", "Lines+Markers": "lines+markers"}
                base_legend = custom_legends.get(d['id'], d['fileName'])
                legend_name = f"{base_legend}{suffix}"
                
                line_style = dict(width=line_width, dash='dash' if suffix else None)
                if c_color: line_style['color'] = c_color
                
                marker_style = dict(size=marker_size)
                if c_color: marker_style['color'] = c_color

                # Determine mode and marker style
                trace_mode = mode_map[plot_mode]
                trace_marker = marker_style if "Markers" in plot_mode else None
                
                if plot_mode == "Lines":
                    # Hack: Use lines+markers with invisible markers to allow point selection
                    trace_mode = "lines+markers"
                    trace_marker = dict(size=max(marker_size, 8), opacity=0)
                    if c_color: trace_marker['color'] = c_color

                fig.add_trace(go.Scatter(
                    x=x_data, y=y_data, mode=trace_mode, name=legend_name,
                    hovertemplate=f"{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4e}}<extra></extra>",
                    line=line_style if "Lines" in plot_mode else None,
                    marker=trace_marker
                ))
                
                # Linear Fit
                if show_linear_fit and d['id'] in linear_fit_settings and x_data is not None and y_data is not None:
                    fit_cfg = linear_fit_settings[d['id']]
                    mask_fit = x_data.notna() & y_data.notna()
                    if fit_range_min is not None: mask_fit &= (x_data >= fit_range_min)
                    if fit_range_max is not None: mask_fit &= (x_data <= fit_range_max)
                    xf, yf = x_data[mask_fit], y_data[mask_fit]
                    
                    if len(xf) > 1:
                        slope, intercept = np.polyfit(xf, yf, 1)
                        y_fit = slope * xf + intercept
                        fig.add_trace(go.Scatter(x=xf, y=y_fit, mode='lines', name=f"Fit {legend_name}", line=dict(dash='dash', width=2, color=fit_cfg.get('color', 'red')), hoverinfo='skip'))
                        
                        annot_x_f = fit_cfg.get('annot_x')
                        annot_y_f = fit_cfg.get('annot_y')
                        if annot_x_f is None or annot_y_f is None:
                            mid_idx = len(xf) // 2
                            annot_x_f, annot_y_f, ay_offset = xf.iloc[mid_idx], y_fit.iloc[mid_idx], -40
                        else: ay_offset = 0
                        
                        fig.add_annotation(x=annot_x_f, y=annot_y_f, text=f"<b>y = {slope:.3e} x + {intercept:.3e}</b>", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, ax=0, ay=ay_offset, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1, font=dict(size=14, color="black"))

                # Parabolic Fit
                if show_parabolic_fit and d['id'] in parabolic_fit_settings and x_data is not None and y_data is not None:
                    pfit_cfg = parabolic_fit_settings[d['id']]
                    mask_pfit = x_data.notna() & y_data.notna()
                    if pfit_range_min is not None: mask_pfit &= (x_data >= pfit_range_min)
                    if pfit_range_max is not None: mask_pfit &= (x_data <= pfit_range_max)
                    xpf, ypf = x_data[mask_pfit], y_data[mask_pfit]
                    
                    if len(xpf) > 2:
                        a, b, c = np.polyfit(xpf, ypf, 2)
                        y_pfit = a * xpf**2 + b * xpf + c
                        fig.add_trace(go.Scatter(x=xpf, y=y_pfit, mode='lines', name=f"ParaFit {legend_name}", line=dict(dash='dot', width=3, color=pfit_cfg.get('color', 'green')), hoverinfo='skip'))
                        
                        annot_x_pf = pfit_cfg.get('annot_x')
                        annot_y_pf = pfit_cfg.get('annot_y')
                        if annot_x_pf is None or annot_y_pf is None:
                            mid_idx = len(xpf) // 2
                            annot_x_pf, annot_y_pf, ay_offset = xpf.iloc[mid_idx], y_pfit.iloc[mid_idx], 40
                        else: ay_offset = 0

                        fig.add_annotation(x=annot_x_pf, y=annot_y_pf, text=f"<b>y = {a:.2e} x² + {b:.2e} x + {c:.2e}</b>", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, ax=0, ay=ay_offset, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor=pfit_cfg.get('color', 'green'), borderwidth=1, font=dict(size=14, color=pfit_cfg.get('color', 'green')))

                clean_label = d['label'].replace(" ", "_") + suffix.replace(" ", "_").replace("(", "").replace(")", "")
                export_data[f"{clean_label}_X"] = x_data.values
                export_data[f"{clean_label}_Y"] = y_data.values

        # Apply Customization
        final_title = custom_title if custom_title else f"{y_label} vs {x_label}"
        final_xlabel = custom_xlabel if custom_xlabel else x_label
        final_ylabel = custom_ylabel if custom_ylabel else y_label

        if f"annotations_list_{plot_id}" in st.session_state:
            for annot in st.session_state[f"annotations_list_{plot_id}"]:
                if annot["text"]:
                    # Apply styling
                    styled_text = annot["text"]
                    if annot.get("bold", False): styled_text = f"<b>{styled_text}</b>"
                    if annot.get("italic", False): styled_text = f"<i>{styled_text}</i>"
                    
                    fig.add_annotation(
                        x=annot["x"], y=annot["y"], 
                        text=styled_text, 
                        showarrow=False, 
                        font=dict(
                            size=annot["size"], 
                            color=annot["color"],
                            family=annot.get("font", "Arial")
                        ), 
                        bgcolor="rgba(255, 255, 255, 0.5)"
                    )

        final_template = template_mode if template_mode != "Auto (Global)" else "plotly"
        uirevision_key = f"{plot_id}_{analysis_mode}"
        if analysis_mode == "Standard MR Analysis": uirevision_key += f"_{x_axis_unit}_{y_axis_mode}"
        elif analysis_mode == "Standard R-T Analysis": uirevision_key += f"_{y_axis_mode}"
        else: uirevision_key += f"_{custom_x_col}_{custom_y_col}"

        fig.update_layout(
            title=dict(text=final_title, font=dict(size=title_font_size), x=0.5, xanchor='center'),
            xaxis=dict(title=dict(text=final_xlabel, font=dict(size=axis_title_size)), tickfont=dict(size=tick_font_size), showgrid=show_grid, gridcolor=grid_color, range=[xlim_min, xlim_max] if use_xlim else None),
            yaxis=dict(title=dict(text=final_ylabel, font=dict(size=axis_title_size)), tickfont=dict(size=tick_font_size), showgrid=show_grid, gridcolor=grid_color, range=[ylim_min, ylim_max] if use_ylim else None),
            showlegend=show_legend, hovermode="closest", height=height, width=width, template=final_template, uirevision=uirevision_key
        )

        safe_title = "".join([c for c in final_title if c.isalnum() or c in (' ', '-', '_')]).strip().replace(" ", "_") or f"plot_{plot_id}"
        config = {'toImageButtonOptions': {'format': 'png', 'filename': f"MR_Analysis_{safe_title}", 'height': height, 'width': width, 'scale': 2}}
        
        selection = st.plotly_chart(fig, width="stretch", config=config, key=f"chart_{plot_id}", on_select="rerun", selection_mode="points")
        if selection and selection.get("selection") and selection["selection"]["points"]:
            point = selection["selection"]["points"][0]
            st.session_state[f"last_click_{plot_id}"] = {"x": point["x"], "y": point["y"]}
        
        # --- Statistics Table (In Export Tab) ---
        if export_data:
            with tab_export:
                export_msg_placeholder.empty() # Clear the initial message
                
                st.write("")
                
                df_export = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in export_data.items() ]))
                
                # 1. DAT Export (Blue)
                with st.container():
                    st.info("📄 **DAT Export** (Tab-separated values)")
                    dat_exp = df_export.to_csv(index=False, sep='\t').encode('utf-8')
                    st.download_button(
                        label=f"Download {plot_name} Data (.dat)",
                        data=dat_exp,
                        file_name=f"{safe_title}.dat",
                        mime="text/plain",
                        key=f"dl_dat_{plot_id}"
                    )

                # 2. CSV Export (Green)
                with st.container():
                    st.success("📊 **CSV Export** (Comma-separated values)")
                    csv_exp = df_export.to_csv(index=False, sep=',').encode('utf-8')
                    st.download_button(
                        label=f"Download {plot_name} Data (.csv)",
                        data=csv_exp,
                        file_name=f"{safe_title}.csv",
                        mime="text/csv",
                        key=f"dl_csv_{plot_id}"
                    )

                # 3. HTML Export (Orange/Warning)
                with st.container():
                    st.warning("🌐 **HTML Export** (Interactive Plot)")
                    html_exp = fig.to_html(include_plotlyjs="cdn", full_html=True)
                    st.download_button(
                        label=f"Download {plot_name} Plot (.html)",
                        data=html_exp,
                        file_name=f"{safe_title}.html",
                        mime="text/html",
                        key=f"dl_html_{plot_id}"
                    )
        return fig
