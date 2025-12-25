import streamlit as st
import pickle
import os
import uuid

# -----------------------------------------------------------------------------
# SESSION PERSISTENCE
# -----------------------------------------------------------------------------

# Ensure the state file is stored in the application root directory
# This prevents issues depending on where the streamlit command is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(BASE_DIR)
STATE_FILE = os.path.join(APP_ROOT, "session_state.pkl")
PROJECTS_DIR = os.path.join(APP_ROOT, "projects")

if not os.path.exists(PROJECTS_DIR):
    os.makedirs(PROJECTS_DIR)

def get_current_state_dict():
    """Helper to construct the state dictionary for saving."""
    # Capture plot-specific state that isn't in persistent_values
    plot_states = {}
    if 'plot_ids' in st.session_state:
        for pid in st.session_state.plot_ids:
            for prefix in ["sel_", "batch_filter_", "pname_", "ren_mode_", "annotations_list_"]:
                key = f"{prefix}{pid}"
                if key in st.session_state:
                    plot_states[key] = st.session_state[key]

    return {
        'all_datasets': st.session_state.all_datasets,
        'plot_ids': st.session_state.plot_ids,
        'next_plot_id': st.session_state.next_plot_id,
        'custom_batches': st.session_state.custom_batches,
        'persistent_values': st.session_state.get('persistent_values', {}),
        'batch_counter': st.session_state.batch_counter,
        'plot_states': plot_states
    }

def save_session_state():
    """Saves the current session state to a local pickle file AND active project if autosave is on."""
    state_to_save = get_current_state_dict()
    
    # 1. Save to recovery file (fast, always happens)
    try:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(state_to_save, f)
    except Exception as e:
        print(f"Error saving recovery state: {e}")

    # 2. Auto-save to Project File
    active_project = st.session_state.get('active_project')
    autosave = st.session_state.get('autosave_enabled', False)
    
    if active_project and autosave:
        project_path = os.path.join(PROJECTS_DIR, active_project)
        try:
            with open(project_path, 'wb') as f:
                pickle.dump(state_to_save, f)
        except Exception as e:
            print(f"Error auto-saving project: {e}")

def list_projects():
    """Returns list of .ppms files in projects directory."""
    if not os.path.exists(PROJECTS_DIR): return []
    return [f for f in os.listdir(PROJECTS_DIR) if f.endswith('.ppms')]

def load_project_file(filename):
    """Loads a specific project file."""
    path = os.path.join(PROJECTS_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                saved_state = pickle.load(f)
            return apply_loaded_state(saved_state)
        except Exception as e:
            st.error(f"Error loading project {filename}: {e}")
    return False

def save_current_project(filename):
    """Manually save current state to a project file."""
    if not filename.endswith('.ppms'):
        filename += '.ppms'
    
    path = os.path.join(PROJECTS_DIR, filename)
    state_to_save = get_current_state_dict()
    try:
        with open(path, 'wb') as f:
            pickle.dump(state_to_save, f)
        return True
    except Exception as e:
        st.error(f"Error saving project: {e}")
        return False

def delete_project_file(filename):
    """Deletes a project file."""
    path = os.path.join(PROJECTS_DIR, filename)
    if os.path.exists(path):
        try:
            os.remove(path)
            return True
        except Exception as e:
            st.error(f"Error deleting project: {e}")
    return False

def apply_loaded_state(saved_state):
    """Applies a loaded state dictionary to the current session."""
    try:
        st.session_state.all_datasets = saved_state.get('all_datasets', [])
        st.session_state.plot_ids = saved_state.get('plot_ids', [1])
        st.session_state.next_plot_id = saved_state.get('next_plot_id', 2)
        st.session_state.custom_batches = saved_state.get('custom_batches', {})
        st.session_state.persistent_values = saved_state.get('persistent_values', {})
        st.session_state.batch_counter = saved_state.get('batch_counter', 0)
        
        # Restore plot states
        if 'plot_states' in saved_state:
            for k, v in saved_state['plot_states'].items():
                st.session_state[k] = v
        
        # Force save to local disk immediately
        save_session_state()
        return True
    except Exception as e:
        st.error(f"Error applying state: {e}")
        return False

def load_session_state():
    """Loads session state from local pickle file if it exists."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                saved_state = pickle.load(f)
            
            # Restore if keys are missing or empty
            if not st.session_state.all_datasets and saved_state.get('all_datasets'):
                return apply_loaded_state(saved_state)
        except Exception as e:
            st.error(f"Error loading state: {e}")
    return False

def recover_session_state():
    """Recover plot_ids and next_plot_id from persistent_values if they seem lost."""
    if 'persistent_values' not in st.session_state:
        return

    store = st.session_state.persistent_values
    # Start with current plot_ids or default [1]
    recovered_ids = set(st.session_state.get('plot_ids', [1]))
    
    # Scan store for keys like 'fit_{id}_...' to find lost plots
    for key in store.keys():
        if key.startswith("fit_"):
            parts = key.split("_")
            # fit_{id}_{param} - check if second part is a digit (plot_id)
            if len(parts) > 1 and parts[1].isdigit():
                pid = int(parts[1])
                recovered_ids.add(pid)
    
    # Update plot_ids if we found more
    st.session_state.plot_ids = sorted(list(recovered_ids))
    
    # Update next_plot_id to be safe
    current_max = max(st.session_state.plot_ids) if st.session_state.plot_ids else 0
    if st.session_state.get('next_plot_id', 1) <= current_max:
        st.session_state.next_plot_id = current_max + 1

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'all_datasets': [],
        'uploader_key': 0,
        'project_loader_key': 0,
        'batch_counter': 0,
        'plot_ids': [1],
        'next_plot_id': 2,
        'custom_batches': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -----------------------------------------------------------------------------
# UI HELPERS
# -----------------------------------------------------------------------------

def persistent_selectbox(label, options, persistent_key, **kwargs):
    """
    A wrapper around st.selectbox that persists its value across reruns 
    even when the widget key changes (e.g. due to uploader_key rotation).
    Uses a dedicated dictionary in session_state to store values safely.
    
    Why is this needed?
    Streamlit widgets reset if their 'key' changes or if they disappear from the UI.
    We rotate keys (add uploader_key) to force updates when data changes, but we 
    want to keep the user's selection. This function bridges that gap.
    """
    # 1. Initialize persistent store if needed
    if 'persistent_values' not in st.session_state:
        st.session_state['persistent_values'] = {}
    
    store = st.session_state['persistent_values']
    # We append uploader_key to the widget ID. This forces Streamlit to treat it as a "new" widget
    # whenever a file is uploaded, ensuring it refreshes its options list correctly.
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
    # Avoid passing 'index' if the key is already in session_state to prevent warnings
    if widget_key in st.session_state:
        selected_val = st.selectbox(label, options, key=widget_key, **kwargs)
    else:
        selected_val = st.selectbox(label, options, index=idx, key=widget_key, **kwargs)
    
    # 8. Sync back to store (handles the case where we just initialized with default)
    if store.get(persistent_key) != selected_val:
        store[persistent_key] = selected_val
        save_session_state()
    else:
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
        # Only pass 'value' if the widget key is NOT already in session state
        # This prevents the "created with a default value but also had its value set via Session State API" warning
        if widget_key not in st.session_state:
            kwargs['value'] = store[persistent_key]
        
    # 3. Render
    val = widget_func(key=widget_key, **kwargs)
    
    # 4. Sync
    if store.get(persistent_key) != val:
        store[persistent_key] = val
        save_session_state()
    else:
        store[persistent_key] = val
    return val
