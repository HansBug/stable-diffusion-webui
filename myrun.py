import os
import re
import signal
import threading

import os
import threading
import time
import importlib
import signal
import threading
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from modules.paths import script_path

from modules import devices, sd_samplers, upscaler
import modules.codeformer_model as codeformer
import modules.extras
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.shared as shared
import modules.txt2img

import modules.ui
from modules import devices
from modules import modelloader
from modules.paths import script_path
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

DATASET_DIRECTORY = '/data/sync/pc_sync/novelai/datasets'
LOG_DIRECTORY = '/data/sync/pc_sync/novelai/textual_inversion'
PROMPT_TEMPLATE_FILE = '/home/hansbug/wtf-projects/stable-diffusion-webui/textual_inversion_templates/subject_filewords.txt'

EMBEDDING_DIR = '/home/hansbug/wtf-projects/stable-diffusion-webui/embeddings'

queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def initialize():
    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    shared.face_restorers.append(modules.face_restoration.FaceRestoration())
    modelloader.load_upscalers()

    modules.scripts.load_scripts()

    modules.sd_models.load_model()
    shared.opts.onchange("sd_model_checkpoint",
                         wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))
    shared.opts.onchange("sd_hypernetwork", wrap_queued_call(
        lambda: modules.hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
    shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def is_embedding_exist(name: str) -> bool:
    return os.path.exists(os.path.join(EMBEDDING_DIR, f'{name}.pt'))


def train_new_operator(name: str, steps: int = 13500, learning_rate=0.005, batch_size=1, log_per=250):
    from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
    if not is_embedding_exist(name):
        # game = 'arknights' # if name == 'vanilla' else 'azur_lane'
        # game = 'girls\'_frontline'
        game = 'fate/grand_order'
        if game:
            init_text = f'photo of {name} in {game}'
        else:
            init_text = f'photo of {name}'
        create_embedding(name, init_text=init_text, num_vectors_per_token=8, overwrite_old=False)
        hijack = modules.sd_hijack.model_hijack
        hijack.embedding_db.load_textual_inversion_embeddings()

    dataset_directory = os.path.join(DATASET_DIRECTORY, f'{name}_processed')
    train_embedding(
        name, str(learning_rate), batch_size,
        dataset_directory, LOG_DIRECTORY,
        512, 512, steps, log_per, log_per, PROMPT_TEMPLATE_FILE,

        save_image_with_stored_embedding=True,
        preview_from_txt2img=False,
        preview_prompt='',
        preview_negative_prompt='',
        preview_steps=20,
        preview_sampler_index=0,
        preview_cfg_scale=7,
        preview_seed=-1.0,
        preview_width=512,
        preview_height=512,
    )


DATASET_PATTERN = re.compile(r'^(?P<name>[\s\S]+)_processed$')

if __name__ == '__main__':
    initialize()
    hijack = modules.sd_hijack.model_hijack
    hijack.embedding_db.load_textual_inversion_embeddings()

    datasets = [
        # 'agir', 'azuma', 'anchorage', 
        # 'brest', 'drake', 'friedrich_der_grosse', 
        # 'hakuryu', 'kronshtadt', 
        'musashi', 'new_jersey', 'plymouth', 
        'shimakaze', 'shinano', 'ulrich_von_hutten', 'vanguard', 

        # 'hoshiguma', 'frostnova', 'projekt_red',
        # *os.listdir(DATASET_DIRECTORY),
    ]

    for fn in datasets:
        # matching = DATASET_PATTERN.fullmatch(fn)
        # if not matching:
        #     continue
        # name = matching.group('name')

        name = fn
        print(f'Training {name} ...')
        train_new_operator(name, steps=20000)

