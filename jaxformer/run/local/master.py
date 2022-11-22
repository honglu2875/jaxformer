# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import jax

from jaxformer.utils import print_time, emulate_tpu_on_cpu
from jaxformer.models.factory import create_model
from jaxformer.models.decoder.inter.checkpoint import try_save_ckpt as try_save_ckpt_decoder, load_ckpt as load_ckpt_decoder

class LocalMaster:

    def __init__(self, mesh_shape, config):

        with print_time(f'Allocating jax devices'):
            print(f'Jax.device_count() = {jax.device_count()}')
            self.devices = np.array(jax.devices()).reshape(mesh_shape)

        with jax.experimental.maps.Mesh(self.devices, ('dp', 'pt', 'mp')):
            model, optimizer, lr_schedule = create_model(config=config)[0]

        self.lr_schedule = lr_schedule
        self.model = model
        self.optimizer=optimizer



    def train(self, data):
        with jax.experimental.maps.Mesh(self.devices, ('dp', 'pt', 'mp')):
            step, loss, grad_global_norm = self.model.train({'x': data[:, :, :-1], 'y': data[:, :, 1:]})
            lr = float(self.lr_schedule(step))
            return step, loss, lr, grad_global_norm


    def profile(self, data):
        with jax.experimental.maps.Mesh(self.devices, ('dp', 'pt', 'mp')):
            return self.model.profile({'x': data[:, :, :-1], 'y': data[:, :, 1:]})


    def save(self, step, path, wandb_run_id, data_files, data_file, data_batch):
        with print_time(f'Writing ckpt json at step={step}'):
            with open(f'{path}/ckpt.json', 'w') as f:
                json.dump({'process_count': int(jax.process_count), 'step': int(step), 'wandb_run_id': wandb_run_id, 'data_files': data_files, 'data_file': data_file, 'data_batch': data_batch}, f)
        
        return try_save_ckpt_decoder(self.model.state,path=path)


    def load(self, path, step=None, ignore_optimizer=False):
        return load_ckpt(state_old=self.model.state,path=path,step_overwrite=step,ignore_optimizer=False)


    def stats(self):
        with jax.experimental.maps.Mesh(self.devices, ('dp', 'pt', 'mp')):
            return self.model.stats()


def create_master(config):
    
    if 'use_cuda' in config and config['use_cuda']:
        mp = len(jax.devices())
        mesh_shape = (1, 1, mp)
    else:
        tpu_size_logical = config['tpu_size_logical']
        tpu_cores = config['tpu_cores']
        rep = config['opt_params_partitions']

        if config['debug_emulate_tpu_on_cpu']:
            with print_time(f'Emulating tpu on cpu with {tpu_cores} cores'):
                emulate_tpu_on_cpu(cores=tpu_cores)
        
        dp = tpu_size_logical // tpu_cores // rep
        mp = tpu_cores
        mesh_shape = (dp, rep, mp)

    with print_time(f'Creating local worker'):
        print(f'mesh_shape={mesh_shape}')
        master = LocalMaster(mesh_shape, config)

    return master