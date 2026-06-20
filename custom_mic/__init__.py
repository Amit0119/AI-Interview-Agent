import os
import streamlit.components.v1 as components

_RELEASE = False

if not _RELEASE:
    # Use the absolute path to the frontend folder
    _component_func = components.declare_component(
        "custom_mic",
        path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    )
else:
    # Not used for now
    pass

def custom_mic_button(key=None):
    """Create a new instance of "custom_mic_button"."""
    component_value = _component_func(key=key, default=None)
    return component_value
