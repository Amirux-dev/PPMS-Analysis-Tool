import streamlit as st
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple

# Import Modules
# (Modules updated)
from modules.data_processing import parse_multivu_content
from modules.utils import (
    save_session_state, load_session_state, init_session_state, recover_session_state,
    persistent_selectbox, persistent_input
)
from modules.plotting import create_plot_interface, get_batch_map

# Set page config to wide mode by default
st.set_page_config(layout="wide", page_title="PPMS Analysis Tool", page_icon="📈", initial_sidebar_state="expanded")

# Reduce top whitespace and adjust sidebar width
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 33vw !important;
            min-width: 33vw !important;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STREAMLIT APPLICATION
# -----------------------------------------------------------------------------

init_session_state()
# Load previous state if available
if not st.session_state.all_datasets:
    load_session_state()
recover_session_state()

# Ensure Unique IDs for all datasets (Migration/Safety check)
seen_ids = set()
for d in st.session_state.all_datasets:
    if 'id' not in d or str(d['id']).lower().endswith('.dat') or d['id'] in seen_ids:
        d['id'] = str(uuid.uuid4())
    seen_ids.add(d['id'])

st.title("PPMS Analysis Tool")
st.markdown("Upload `.dat` files to visualize and analyze transport measurements (R-T, MR, I-V, etc.).")

# --- Sidebar: General Settings ---
with st.sidebar.expander("⚙️ General Settings", expanded=False):
    global_field_unit = st.selectbox("Default Field Unit", ["Tesla (T)", "Oersted (Oe)"], index=0, key="global_field_unit")
    global_r0_method = st.selectbox("Default R0 Method", ["Closest to 0T", "Mean within Window", "First Point", "Max Resistance"], index=0, key="global_r0_method")


# --- Sidebar: Data Manager ---
st.sidebar.header("Data Manager")

# Duplicate Strategy
duplicate_strategy = st.sidebar.radio(
    "Duplicate Handling:",
    ["Skip", "Overwrite"],
    index=0,
    horizontal=True,
    help="Choose 'Overwrite' to update existing files with the same name."
)

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
    # Determine Batch Strategy
    if len(uploaded_files) == 1:
        # Single file -> "File by File Import" (ID 0)
        batch_id = 0
        batch_name = "📂 File by File Import"
    else:
        # Multiple files -> New Sequential Batch
        st.session_state.batch_counter += 1
        batch_id = st.session_state.batch_counter
        batch_name = f"📂 Batch Import #{batch_id}"
    
    new_files_count = 0
    updated_files_count = 0
    skipped_count = 0
    
    for uploaded_file in uploaded_files:
        fname = os.path.basename(uploaded_file.name)
        
        existing_file = next((d for d in st.session_state.all_datasets if d['fileName'] == fname), None)
        
        if existing_file and duplicate_strategy == "Skip":
            skipped_count += 1
            continue
            
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            data = parse_multivu_content(content, fname)
            
            data['batch_id'] = batch_id
            data['batch_name'] = batch_name 
            
            if existing_file and duplicate_strategy == "Overwrite":
                # CRITICAL: Preserve the original ID of the file being overwritten.
                # Why? Because all active plots reference files by their UUID ('id').
                # If we generated a new UUID, the file would disappear from all existing plots.
                # By keeping the old ID, the plots automatically update with the new data.
                data['id'] = existing_file['id']
                idx = st.session_state.all_datasets.index(existing_file)
                st.session_state.all_datasets[idx] = data
                updated_files_count += 1
            else:
                st.session_state.all_datasets.append(data)
                new_files_count += 1
        except Exception as e:
            st.error(f"Error parsing {fname}: {e}")
    
    if new_files_count > 0 or updated_files_count > 0 or skipped_count > 0:
        if new_files_count > 0 or updated_files_count > 0:
            save_session_state()
            
        msg = []
        if new_files_count > 0: msg.append(f"{new_files_count} added to '{batch_name}'")
        if updated_files_count > 0: msg.append(f"{updated_files_count} updated")
        if skipped_count > 0: msg.append(f"{skipped_count} skipped")
        
        st.sidebar.success(f"Done: {', '.join(msg)}.")
        st.session_state.uploader_key += 1
        st.rerun()

# Folder Management
organize_mode = True

# --- Callbacks ---
def create_folder_callback():
    new_name = st.session_state.get("new_folder_name_input", "New Folder")
    
    existing_batch_ids = set(d.get('batch_id', 0) for d in st.session_state.all_datasets)
    existing_batch_ids.update(st.session_state.custom_batches.keys())
    existing_batch_ids.discard(0)
    
    new_id = 1
    while new_id in existing_batch_ids:
        new_id += 1
        
    st.session_state.custom_batches[new_id] = f"📂 {new_name}"
    save_session_state()

def move_file_callback(file_id, target_bid, target_name):
    for d in st.session_state.all_datasets:
        if d['id'] == file_id:
            d['batch_id'] = target_bid
            d['batch_name'] = target_name
            break
    
    for key in list(st.session_state.keys()):
        if key.startswith("sel_"):
            if isinstance(st.session_state[key], list):
                st.session_state[key] = list(st.session_state[key])
    save_session_state()

def delete_file_callback(file_id):
    file_to_delete = None
    for d in st.session_state.all_datasets:
        if d['id'] == file_id:
            file_to_delete = d['fileName']
            break
            
    st.session_state.all_datasets = [d for d in st.session_state.all_datasets if d['id'] != file_id]
    
    for key in list(st.session_state.keys()):
        if key.startswith("sel_"):
            current_selection = st.session_state[key]
            if isinstance(current_selection, list):
                if file_to_delete:
                    new_selection = [f for f in current_selection if f != file_to_delete]
                else:
                    new_selection = current_selection
                st.session_state[key] = new_selection
    save_session_state()

def delete_batch_callback(batch_id):
    files_to_delete = [d['fileName'] for d in st.session_state.all_datasets if d.get('batch_id') == batch_id]

    st.session_state.all_datasets = [d for d in st.session_state.all_datasets if d.get('batch_id') != batch_id]
    
    if 'custom_batches' in st.session_state and batch_id in st.session_state.custom_batches:
        del st.session_state.custom_batches[batch_id]

    if files_to_delete:
        for key in list(st.session_state.keys()):
            if key.startswith("sel_"):
                current_selection = st.session_state[key]
                if isinstance(current_selection, list):
                    new_selection = [f for f in current_selection if f not in files_to_delete]
                    st.session_state[key] = new_selection
    save_session_state()

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
    save_session_state()

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
    save_session_state()

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
    save_session_state()

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
        
        # Filter by Folder
        filter_options = {-1: "All Files"}
        filter_options.update({bid: info['name'] for bid, info in batches.items()})
        
        selected_filter = st.selectbox("Filter by Folder", options=list(filter_options.keys()), format_func=lambda x: filter_options[x], key="dlg_batch_filter")
        
        if selected_filter != -1:
            filtered_datasets = [d for d in datasets if d.get('batch_id', 0) == selected_filter]
        else:
            filtered_datasets = datasets
        
        # 1. Select Files
        # Create a mapping of ID -> Display Name
        file_options = {d['id']: f"{d['fileName']} ({d.get('batch_name', 'Unknown')})" for d in filtered_datasets}
        
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
            save_session_state()
            st.rerun()

    if st.sidebar.button("💾 Save Analysis State", help="Force save current state to disk", width='stretch'):
        save_session_state()
        st.toast("State saved manually!", icon="💾")

    # --- Batch Actions Button ---
    if dialog_decorator:
        if st.sidebar.button("⚡ Batch Actions", width='stretch'):
            manage_batch_dialog(datasets, batches)
    else:
        # Fallback for older versions
        with st.sidebar.expander("⚡ Batch Actions", expanded=False):
            # Filter by Folder
            filter_options = {-1: "All Files"}
            filter_options.update({bid: info['name'] for bid, info in batches.items()})
            
            selected_filter = st.selectbox("Filter by Folder", options=list(filter_options.keys()), format_func=lambda x: filter_options[x], key="sb_batch_filter")
            
            if selected_filter != -1:
                filtered_datasets = [d for d in datasets if d.get('batch_id', 0) == selected_filter]
            else:
                filtered_datasets = datasets

            # 1. Select Files
            file_options = {d['id']: f"{d['fileName']} ({d.get('batch_name', 'Unknown')})" for d in filtered_datasets}
            
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

    # Scrollable Container for Files
    with st.sidebar.container(height=500):
        # Display "File by file" first if it exists
        if 0 in batches:
            with st.expander(batches[0]['name'], expanded=False):
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
            with st.expander(b_name, expanded=False):
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


# Plotting Interface imported from modules


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

if not plot_indices:
    st.info("No plots available. Click below to add one.")
    if st.button("➕ Add Plot", key="add_plot_empty", type="primary"):
        from modules.plotting import add_plot_callback
        add_plot_callback()
        st.rerun()

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

