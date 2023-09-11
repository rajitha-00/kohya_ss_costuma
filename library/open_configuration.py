import logging
import json
from library.common_gui import get_file_path, update_my_data
import gradio as gr
import os

# Set up logging
log = logging.getLogger(__name__)

def open_configuration(
    # ... [all your parameters]
):
    """
    Open the configuration from a file and update the parameters.

    Parameters:
    - ask_for_file: Whether to ask the user for the file path.
    - apply_preset: Whether to apply a preset configuration.
    - file_path: Path to the configuration file.
    - ... [all other parameters]

    Returns:
    - tuple: Updated values for all parameters.
    """
    # Get list of function parameters and values
    parameters = list(locals().items())

    ask_for_file = True if ask_for_file.get('label') == 'True' else False
    apply_preset = True if apply_preset.get('label') == 'True' else False

    # Check if we are "applying" a preset or a config
    if apply_preset:
        log.info(f'Applying preset {training_preset}...')
        file_path = f'./presets/lora/{training_preset}.json'
    else:
        # If not applying a preset, set the `training_preset` field to an empty string
        training_preset_index = parameters.index(('training_preset', training_preset))
        parameters[training_preset_index] = ('training_preset', '')

    original_file_path = file_path

    if ask_for_file:
        file_path = get_file_path(file_path)

    if not file_path == '' and not file_path == None:
        # Load variables from JSON file
        with open(file_path, 'r') as f:
            my_data = json.load(f)
            log.info('Loading config...')
            my_data = update_my_data(my_data)
    else:
        file_path = original_file_path
        my_data = {}

    values = [file_path]
    for key, value in parameters:
        if not key in ['ask_for_file', 'apply_preset', 'file_path']:
            json_value = my_data.get(key)
            values.append(json_value if json_value is not None else value)

    # This next section is about making the LoCon parameters visible if LoRA_type = 'Standard'
    if my_data.get('LoRA_type', 'Standard') == 'LoCon':
        values.append(gr.Row.update(visible=True))
    else:
        values.append(gr.Row.update(visible=False))

    return tuple(values)
