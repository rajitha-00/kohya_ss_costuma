import logging
import json
from library.common_gui import get_file_path, update_my_data
import gradio as gr
import os

# Set up logging
log = logging.getLogger(__name__)

def open_configuration(
    ask_for_file,
    apply_preset,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,
    dim_from_weights,
    color_aug,
    flip_aug,
    clip_skip,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    v_pred_like_loss,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    lr_scheduler_args,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    LoRA_type,
    factor,
    use_cp,
    decompose_both,
    train_on_input,
    conv_dim,
    conv_alpha,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    down_lr_weight,
    mid_lr_weight,
    up_lr_weight,
    block_lr_zero_threshold,
    block_dims,
    block_alphas,
    conv_block_dims,
    conv_block_alphas,
    weighted_captions,
    unit,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    scale_weight_norms,
    network_dropout,
    rank_dropout,
    module_dropout,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    full_bf16,
    min_timestep,
    max_timestep,
    training_preset,
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
