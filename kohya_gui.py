import gradio as gr
import os
from lora_gui import lora_tab
from library.class_lora_tab import LoRATools
import argparse
from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

def LoRA_UI(**kwargs):
    css = ''

    headless = kwargs.get('headless', False)
    log.info(f'headless: {headless}')

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            log.info('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Interface(css=css, title='LoRA GUI', theme=gr.themes.Default())

    with interface:
        with gr.Tab('LoRA'):
            lora_tab(headless=headless)

    # Show the interface
    launch_kwargs = {}
    username = kwargs.get('username')
    password = kwargs.get('password')
    server_port = kwargs.get('server_port', 0)
    inbrowser = kwargs.get('inbrowser', False)
    share = kwargs.get('share', False)
    server_name = kwargs.get('listen')

    launch_kwargs['server_name'] = server_name
    if username and password:
        launch_kwargs['auth'] = (username, password)
    if server_port > 0:
        launch_kwargs['server_port'] = server_port
    if inbrowser:
        launch_kwargs['inbrowser'] = inbrowser
    if share:
        launch_kwargs['share'] = share
    interface.launch(**launch_kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen', type=str, default='127.0.0.1', help='IP to listen on for connections to Gradio')
    parser.add_argument('--username', type=str, default='', help='Username for authentication')
    parser.add_argument('--password', type=str, default='', help='Password for authentication')
    parser.add_argument('--server_port', type=int, default=0, help='Port to run the server listener on')
    parser.add_argument('--inbrowser', action='store_true', help='Open in browser')
    parser.add_argument('--share', action='store_true', help='Share the gradio UI')
    parser.add_argument('--headless', action='store_true', help='Is the server headless')

    args = parser.parse_args()

    LoRA_UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
        headless=args.headless,
    )
