from phc.utils.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgym.torch_utils import *

import os
import os.path as osp
import time
from datetime import datetime
import numpy as np
from torch import optim
import torch
from torch import nn
from phc.env.tasks.humanoid_amp_task import HumanoidAMPTask

import learning.replay_buffer as replay_buffer
import learning.common_agent as common_agent

from tensorboardX import SummaryWriter
import copy
from phc.utils.torch_utils import project_to_norm
import learning.amp_datasets as amp_datasets
from phc.learning.loss_functions import kl_multi
from smpl_sim.utils.math_utils import LinearAnneal
from phc.utils.pc_anomaly import (
    SmplToPointCloud,
    PointCloudMotionClassifier,
    build_pc_backbone,
    pc_normalize_torch,
    POINTNET2_FEAT_DIM,
)
from phc.utils.smpl_lidar_sim import simulate_lidar_scan, vertical_angles_deg_for_lidar

def load_my_state_dict(target, saved_dict):
    for name, param in saved_dict.items():
        if name not in target:
            continue

        if target[name].shape == param.shape:
            target[name].copy_(param)


class AMPAgent(common_agent.CommonAgent):

    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        if self.config.get('use_seq_rl', False):
            # Use the is_rnn to force the dataset to have sequencal format. 
            self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, True, self.ppo_device, self.seq_len)
        else:
            self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)


        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.ppo_device)  # Override and get new value

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        norm_disc_reward = config.get('norm_disc_reward', False)
        if (norm_disc_reward):
            self._disc_reward_mean_std = RunningMeanStd((1,)).to(self.ppo_device)
        else:
            self._disc_reward_mean_std = None

        self.save_kin_info = self.vec_env.env.task.cfg.env.get("save_kin_info", False)
        self.only_kin_loss = self.vec_env.env.task.cfg.env.get("only_kin_loss", False)
        self.temp_running_mean = self.vec_env.env.task.temp_running_mean # use temp running mean to make sure the obs used for training is the same as calc gradient.

        kin_lr = float(self.vec_env.env.task.kin_lr)
        
        if self.save_kin_info:
            self.kin_dict_info = None
            self.kin_optimizer = torch.optim.Adam(self.model.a2c_network.parameters(), kin_lr)

        # Point-cloud anomaly branch (SMPL → point cloud → classifier)
        self.use_pc_anomaly_loss = config.get('use_pc_anomaly_loss', False)
        self.pc_use_as_reward = bool(config.get('pc_use_as_reward', False))
        self.pc_reward_coef = float(config.get('pc_reward_coef', 0.0))
        self.pc_anomaly_loss_coef = float(config.get('pc_anomaly_loss_coef', 0.0))
        self.pc_human_class_idx = int(config.get('pc_human_class_idx', 0))
        self.pc_num_points = int(config.get('pc_num_points', 1024))
        self.pc_local_coord = bool(config.get('pc_local_coord', True))
        self.pc_num_classes = int(config.get('pc_num_classes', 2))
        self.pc_temporal_pool = config.get('pc_temporal_pool', 'mean')
        self.pc_classifier_weights = config.get('pc_classifier_weights', None)
        self.pc_smpl_model_path = config.get('pc_smpl_model_path', 'data/smpl')
        self.pc_backbone = config.get('pc_backbone', 'pointnet2')
        self.pc_pointnet_pretrained = config.get('pc_pointnet_pretrained', None)
        self.pc_chunk_size = config.get('pc_chunk_size', None)
        self.pc_use_root_local_frame = bool(config.get('pc_use_root_local_frame', True))
        self.pc_axis_perm = tuple(int(x) for x in config.get('pc_axis_perm', [0, 1, 2]))
        self.pc_axis_sign = tuple(float(x) for x in config.get('pc_axis_sign', [1.0, 1.0, 1.0]))
        self.pc_debug_check_grad = bool(config.get('pc_debug_check_grad', False))
        self.pc_debug_check_grad_interval = int(config.get('pc_debug_check_grad_interval', 50))
        self.pc_log_sample_size = int(config.get('pc_log_sample_size', 8))
        self.pc_dataset_save_enable = bool(config.get('pc_dataset_save_enable', False))
        self.pc_dataset_save_dir = str(config.get('pc_dataset_save_dir', 'output/pc_anomaly_dataset'))
        self.pc_dataset_save_per_max = float(config.get('pc_dataset_save_per_max', 0.15))
        self.pc_dataset_save_mean_max = float(config.get('pc_dataset_save_mean_max', 0.0))
        self.pc_dataset_save_max_files = int(config.get('pc_dataset_save_max_files', 5000))
        self._pc_dataset_save_count = 0
        # 与点云数据集分开落盘：仅 SMPL 状态（仍用下方 per_max / mean_max 与点云保存同一套筛选）
        self.pc_smpl_save_enable = bool(config.get('pc_smpl_save_enable', False))
        self.pc_smpl_save_dir = str(config.get('pc_smpl_save_dir', 'output/pc_smpl_dataset'))
        self.pc_smpl_save_max_files = int(config.get('pc_smpl_save_max_files', 5000))
        self._pc_smpl_save_count = 0
        self.pc_lidar_save_enable = bool(config.get('pc_lidar_save_enable', False))
        self.pc_lidar_save_dir = str(config.get('pc_lidar_save_dir', 'output/pc_lidar_dataset'))
        self.pc_lidar_save_max_files = int(config.get('pc_lidar_save_max_files', 5000))
        self._pc_lidar_save_count = 0
        self.pc_lidar_distance = float(config.get('pc_lidar_distance', 10.0))
        self.pc_lidar_sensor_height = float(config.get('pc_lidar_sensor_height', 1.3))
        self.pc_lidar_lines = int(config.get('pc_lidar_lines', 64))
        self.pc_lidar_fov_up_deg = float(config.get('pc_lidar_fov_up_deg', 2.0))
        self.pc_lidar_fov_down_deg = float(config.get('pc_lidar_fov_down_deg', -24.9))
        self.pc_lidar_horizontal_res_deg = float(config.get('pc_lidar_horizontal_res_deg', 0.5))
        self.pc_lidar_min_range = float(config.get('pc_lidar_min_range', 0.2))
        self.pc_lidar_max_range = float(config.get('pc_lidar_max_range', 80.0))
        self.pc_lidar_ray_chunk_size = int(config.get('pc_lidar_ray_chunk_size', 256))
        self.pc_lidar_scan_padding_deg = float(config.get('pc_lidar_scan_padding_deg', 1.0))
        self.pc_lidar_max_envs = int(config.get('pc_lidar_max_envs', 4))
        self._pc_grad_check_step = 0
        self._enable_pc_branch = self.use_pc_anomaly_loss or self.pc_use_as_reward

        self.smpl_to_pc = None
        self.pc_classifier = None

        if self._enable_pc_branch:
            if not self.save_kin_info:
                raise RuntimeError(
                    "use_pc_anomaly_loss/pc_use_as_reward 启用时需要在环境配置中设置 env.save_kin_info=True，"
                    "以便在 rollout 期间保存 SMPL 状态（kin_dict）。"
                )

            humanoid_env = self.vec_env.env.task
            # 形状参数维度：humanoid_shapes[..., 0] 为 gender，其余为 betas
            humanoid_shapes = humanoid_env.humanoid_shapes
            num_betas = int(humanoid_shapes.shape[1] - 1)

            self.smpl_to_pc = SmplToPointCloud(
                smpl_model_path=self.pc_smpl_model_path,
                num_betas=num_betas,
                num_points=self.pc_num_points,
                local_coord=self.pc_local_coord,
                device=self.ppo_device,
            )

            backbone = build_pc_backbone(
                backbone_type=self.pc_backbone,
                feat_dim=config.get('pc_feat_dim', 256),
                in_channels=3,
                pretrained_extractor_path=self.pc_pointnet_pretrained,
            )
            self.pc_classifier = PointCloudMotionClassifier(
                backbone=backbone,
                in_channels=3,
                feat_dim=getattr(backbone, 'feat_dim', config.get('pc_feat_dim', 256)),
                num_classes=self.pc_num_classes,
                temporal_pool=self.pc_temporal_pool,
            ).to(self.ppo_device)

            if self.pc_classifier_weights is not None:
                state = torch.load(self.pc_classifier_weights, map_location=self.ppo_device)
                # 允许权重文件中嵌套一层 'state_dict'
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                self.pc_classifier.load_state_dict(state, strict=False)

            # 训练 PULSE 时冻结分类器参数，只回传到 PULSE/SMPL
            for p in self.pc_classifier.parameters():
                p.requires_grad_(False)
            # 分类器只做推理，不参与训练；保持 eval，避免 BN 在小 batch 下漂移
            self.pc_classifier.eval()

            # 使用第一个 humanoid 的 betas 作为统一形状（不区分个体）
            with torch.no_grad():
                betas = humanoid_shapes[0, 1:1 + num_betas].to(self.ppo_device)
                # 不依赖 nn.Module.register_buffer，直接存为普通张量属性
                self.pc_betas = betas.unsqueeze(0)
            self.pc_axis_sign_tensor = torch.tensor(self.pc_axis_sign, dtype=torch.float32, device=self.ppo_device).view(1, 1, 3)

        # ZL Hack
        if self.vec_env.env.task.fitting:
            print("#################### Fitting and freezing!! ####################")
            checkpoint = torch_ext.load_checkpoint(self.vec_env.env.task.models_path[0])
            
            self.set_stats_weights(checkpoint)  # loads mean std. essential for distilling knowledge. will not load if has a shape mismatch.
            self.freeze_state_weights()  # freeze the mean stds.
            load_my_state_dict(self.model.state_dict(), checkpoint['model'])  # loads everything (model, std, ect.). that can be load from the last model.
            # self.value_mean_std # not freezing value function though.
        
        return
    
    def set_stats_weights(self, weights):
        if self.normalize_input:
            if weights['running_mean_std']['running_mean'].shape == self.running_mean_std.state_dict()['running_mean'].shape:
                self.running_mean_std.load_state_dict(weights['running_mean_std'])
            else:
                print("shape mismatch, can not load input mean std")
                
        if self.normalize_value:
            self.value_mean_std.load_state_dict(weights['reward_mean_std'])

        if self.has_central_value:
            self.central_value_net.set_stats_weights(weights['assymetric_vf_mean_std'])
 
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])
            
        if self._normalize_amp_input:
            if weights['amp_input_mean_std']['running_mean'].shape == self._amp_input_mean_std.state_dict()['running_mean'].shape:
                self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
            else:
                print("shape mismatch, can not load AMP mean std")
            

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.load_state_dict(weights['disc_reward_mean_std'])
            
    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        
        if "kin_optimizer" in self.__dict__:
            print("!!!saving kin_optimizer!!! Remove this message asa p!!")
            state['kin_optimizer'] = self.kin_optimizer.state_dict()

        return state

    def set_full_state_weights(self, weights):
        super().set_full_state_weights(weights)
        if "kin_optimizer" in weights:
            print("!!!loading kin_optimizer!!! Remove this message asa p!!")
            self.kin_optimizer.load_state_dict(weights['kin_optimizer'])
        

    def freeze_state_weights(self):
        if self.normalize_input:
            self.running_mean_std.freeze()
        if self.normalize_value:
            self.value_mean_std.freeze()
        if self.has_central_value:
            raise NotImplementedError()
        if self.mixed_precision:
            raise NotImplementedError()

    def unfreeze_state_weights(self):
        if self.normalize_input:
            self.running_mean_std.unfreeze()
        if self.normalize_value:
            self.value_mean_std.unfreeze()
        if self.has_central_value:
            raise NotImplementedError()
        if self.mixed_precision:
            raise NotImplementedError()

    def init_tensors(self):
        super().init_tensors()
        self._build_amp_buffers()

        if self.save_kin_info:
            B, S, _ = self.experience_buffer.tensor_dict['obses'].shape
            kin_dict = self.vec_env.env.task.kin_dict
            kin_dict_size = np.sum([v.reshape(v.shape[0], -1).shape[-1] for k, v in kin_dict.items()])
            self.experience_buffer.tensor_dict['kin_dict'] = torch.zeros((B, S, kin_dict_size)).to(self.experience_buffer.tensor_dict['obses'])
            self.tensor_list += ['kin_dict']
            
        if self.vec_env.env.task.z_type == "vae":
            B, S, _ = self.experience_buffer.tensor_dict['obses'].shape
            self.experience_buffer.tensor_dict['z_noise'] = torch.zeros(B, S, self.model.a2c_network.embedding_size).to(self.experience_buffer.tensor_dict['obses'])
            self.tensor_list += ['z_noise']
            
        return

    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.eval()
        if self._enable_pc_branch and self.pc_classifier is not None:
            self.pc_classifier.eval()

        return

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.train()
        # 即使 agent 切到 train，点云分类器仍固定 eval（只做打分）
        if self._enable_pc_branch and self.pc_classifier is not None:
            self.pc_classifier.eval()

        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()

        if (self._norm_disc_reward()):
            state['disc_reward_mean_std'] = self._disc_reward_mean_std.state_dict()

        return state
    

    def play_steps_rnn(self):
        self.set_eval()
        mb_rnn_states = []
        epinfos = []
        self.experience_buffer.tensor_dict['values'].fill_(0)
        self.experience_buffer.tensor_dict['rewards'].fill_(0)
        self.experience_buffer.tensor_dict['dones'].fill_(1)
        step_time = 0.0

        update_list = self.update_list

        batch_size = self.num_agents * self.num_actors
        mb_rnn_masks = None

        mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = self.init_rnn_step(batch_size, mb_rnn_states) # mb_rnn_states means "memory bank" rnn states

        ### ZL
        done_indices = []
        terminated_flags = torch.zeros(self.num_actors, device=self.device)
        reward_raw = torch.zeros(1, device=self.device)
        pc_reward_penalty = torch.zeros(1, device=self.device)
        pc_rollout_p_human = torch.zeros(1, device=self.device)
        pc_rollout_p_human_samples = []
        pc_rollout_p_human_min = torch.tensor(float("inf"), device=self.device)
        pc_rollout_p_human_max = torch.tensor(float("-inf"), device=self.device)
        pc_rollout_p_human_nonzero = torch.zeros(1, device=self.device)
        pc_rollout_p_human_count = torch.zeros(1, device=self.device)

        for n in range(self.horizon_length):
            
            
            
            self.obs = self.env_reset(done_indices)
            
            # self.rnn_states[0][:, :, -1] = n; print('debugg!!!!')
            # self.rnn_states[0][:, :, -2] = torch.arange(self.num_actors)
            
            seq_indices, full_tensor = self.process_rnn_indices(mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states)  # this should upate mb_rnn_states
            if full_tensor:
                break
            
            if self.has_central_value:
                self.central_value_net.pre_step_rnn(self.last_rnn_indices, self.last_state_indices)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            
            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data_rnn('obses', indices, play_mask, self.obs['obs'])

            for k in update_list:
                self.experience_buffer.update_data_rnn(k, indices, play_mask, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data_rnn('states', indices[::self.num_agents], play_mask[::self.num_agents] // self.num_agents, self.obs['states'])

            if self.only_kin_loss:
                # pure behavior cloning, kinemaitc loss. 
                self.obs, rewards, self.dones, infos = self.env_step(res_dict['mus'])
            else:
                self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            
                
            shaped_rewards = self.rewards_shaper(rewards)
            if self.pc_use_as_reward:
                with torch.no_grad():
                    if 'kin_dict' not in infos:
                        raise RuntimeError("pc_use_as_reward=True 但 infos 中没有 kin_dict，请确认 env.save_kin_info=True。")
                    if n == 0:
                        self._dump_last_kin_frame(infos['kin_dict'])
                    pc_batch, p_human_step, pc_viz_batch = self._forward_pc_normalized_and_p_human(infos['kin_dict'])
                    self._maybe_save_pc_dataset_batch(p_human_step, n, pc_batch, pc_viz_batch)
                    self._maybe_save_smpl_dataset_batch(infos['kin_dict'], p_human_step, n)
                    self._maybe_save_lidar_dataset_batch(infos['kin_dict'], p_human_step, n)
                    penalty = self.pc_reward_coef * p_human_step.unsqueeze(-1)
                    shaped_rewards = shaped_rewards - penalty
                    pc_reward_penalty += penalty.mean()
                    pc_rollout_p_human += p_human_step.mean()
                    remain = self.pc_log_sample_size - len(pc_rollout_p_human_samples)
                    if remain > 0:
                        pc_rollout_p_human_samples.extend(
                            p_human_step.detach().flatten()[:remain].cpu().tolist()
                        )
                    pc_rollout_p_human_min = torch.minimum(pc_rollout_p_human_min, p_human_step.min())
                    pc_rollout_p_human_max = torch.maximum(pc_rollout_p_human_max, p_human_step.max())
                    pc_rollout_p_human_nonzero += (p_human_step > 1e-8).float().sum()
                    pc_rollout_p_human_count += torch.tensor(float(p_human_step.numel()), device=self.device)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()
            self.experience_buffer.update_data_rnn('rewards', indices, play_mask, shaped_rewards)
            self.experience_buffer.update_data_rnn('next_obses', indices, play_mask, self.obs['obs'])
            self.experience_buffer.update_data_rnn('dones', indices, play_mask, self.dones.byte())
            self.experience_buffer.update_data_rnn('amp_obs', indices, play_mask, infos['amp_obs'])

            ### ZL
            terminated = infos['terminate'].float()
            terminated_flags += terminated
            reward_raw_mean = infos['reward_raw'].mean(dim=0)

            if reward_raw.shape != reward_raw_mean.shape:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean

            terminated = terminated.unsqueeze(-1)
            input_dict = {"obs": self.obs['obs'], "rnn_states": self.rnn_states}
            next_vals = self._eval_critic(input_dict)  # ZL this has issues? (maybe not, since we are passing the states in.)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data_rnn('next_values', indices, play_mask, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.process_rnn_dones(all_done_indices, indices, seq_indices)

            if self.has_central_value:
                self.central_value_net.post_step_rnn(all_done_indices)

            self.algo_observer.process_infos(infos, done_indices)

            fdones = self.dones.float()
            not_dones = 1.0 - self.dones.float()

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            if self.only_kin_loss:
                self.experience_buffer.update_data_rnn('kin_dict', indices, play_mask, torch.cat([v.reshape(v.shape[0], -1) for k, v in infos['kin_dict'].items()], dim = -1))
                if self.kin_dict_info is None:
                    self.kin_dict_info = {k: (v.shape, v.reshape(v.shape[0], -1).shape) for k, v in infos['kin_dict'].items()}

            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)

            done_indices = done_indices[:, 0]
            

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values
        
        # self.experience_buffer.tensor_dict['actions']: is num_env, Batch, feat. That's why we swap and flatten, mb_rnn_states is already in that format. 
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list) # swap to step, num_envs, feat
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['rnn_states'] = mb_rnn_states
        
        batch_dict['rnn_masks'] = mb_rnn_masks # ZL: this should be swap and flattened, but it's all ones for now
        batch_dict['terminated_flags'] = terminated_flags
        batch_dict['reward_raw'] =reward_raw / self.horizon_length
        if self.pc_use_as_reward:
            batch_dict['pc_reward_penalty'] = pc_reward_penalty / self.horizon_length
            batch_dict['pc_rollout_p_human'] = pc_rollout_p_human / self.horizon_length
            batch_dict['pc_rollout_p_human_samples'] = pc_rollout_p_human_samples
            batch_dict['pc_rollout_p_human_min'] = (
                pc_rollout_p_human_min
                if torch.isfinite(pc_rollout_p_human_min)
                else torch.tensor(0.0, device=self.device)
            )
            batch_dict['pc_rollout_p_human_max'] = (
                pc_rollout_p_human_max
                if torch.isfinite(pc_rollout_p_human_max)
                else torch.tensor(0.0, device=self.device)
            )
            batch_dict['pc_rollout_p_human_nonzero_ratio'] = (
                pc_rollout_p_human_nonzero / torch.clamp_min(pc_rollout_p_human_count, 1.0)
            )
        
        batch_dict['played_frames'] = n * self.num_actors * self.num_agents
        batch_dict['step_time'] = step_time
        

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        batch_dict['mb_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)
        
        return batch_dict

    def play_steps(self):
        self.set_eval()
        humanoid_env = self.vec_env.env.task

        epinfos = []
        done_indices = []
        update_list = self.update_list
        terminated_flags = torch.zeros(self.num_actors, device=self.device)
        reward_raw = torch.zeros(1, device=self.device)
        pc_reward_penalty = torch.zeros(1, device=self.device)
        pc_rollout_p_human = torch.zeros(1, device=self.device)
        pc_rollout_p_human_samples = []
        pc_rollout_p_human_min = torch.tensor(float("inf"), device=self.device)
        pc_rollout_p_human_max = torch.tensor(float("-inf"), device=self.device)
        pc_rollout_p_human_nonzero = torch.zeros(1, device=self.device)
        pc_rollout_p_human_count = torch.zeros(1, device=self.device)
        for n in range(self.horizon_length):

            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
                
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])
            
            if self.only_kin_loss and self.save_kin_info:
                # pure behavior cloning, kinemaitc loss. 
                self.obs, rewards, self.dones, infos = self.env_step(res_dict['mus'])
            else:
                self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
                
            shaped_rewards = self.rewards_shaper(rewards)
            if self.pc_use_as_reward:
                with torch.no_grad():
                    if 'kin_dict' not in infos:
                        raise RuntimeError("pc_use_as_reward=True 但 infos 中没有 kin_dict，请确认 env.save_kin_info=True。")
                    if n == 0:
                        self._dump_last_kin_frame(infos['kin_dict'])
                    pc_batch, p_human_step, pc_viz_batch = self._forward_pc_normalized_and_p_human(infos['kin_dict'])
                    self._maybe_save_pc_dataset_batch(p_human_step, n, pc_batch, pc_viz_batch)
                    self._maybe_save_smpl_dataset_batch(infos['kin_dict'], p_human_step, n)
                    self._maybe_save_lidar_dataset_batch(infos['kin_dict'], p_human_step, n)
                    penalty = self.pc_reward_coef * p_human_step.unsqueeze(-1)
                    shaped_rewards = shaped_rewards - penalty
                    pc_reward_penalty += penalty.mean()
                    pc_rollout_p_human += p_human_step.mean()
                    remain = self.pc_log_sample_size - len(pc_rollout_p_human_samples)
                    if remain > 0:
                        pc_rollout_p_human_samples.extend(
                            p_human_step.detach().flatten()[:remain].cpu().tolist()
                        )
                    pc_rollout_p_human_min = torch.minimum(pc_rollout_p_human_min, p_human_step.min())
                    pc_rollout_p_human_max = torch.maximum(pc_rollout_p_human_max, p_human_step.max())
                    pc_rollout_p_human_nonzero += (p_human_step > 1e-8).float().sum()
                    pc_rollout_p_human_count += torch.tensor(float(p_human_step.numel()), device=self.device)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
            
            if self.save_kin_info:
                self.experience_buffer.update_data('kin_dict', n, torch.cat([v.reshape(v.shape[0], -1) for k, v in infos['kin_dict'].items()], dim = -1))
                
                if self.kin_dict_info is None:
                    self.kin_dict_info = {k: (v.shape, v.reshape(v.shape[0], -1).shape) for k, v in infos['kin_dict'].items()}

                
            terminated = infos['terminate'].float()
            terminated_flags += terminated

            reward_raw_mean = infos['reward_raw'].mean(dim=0)
            if reward_raw.shape != reward_raw_mean.shape:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean
            terminated = terminated.unsqueeze(-1)

            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)
            
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['terminated_flags'] = terminated_flags
        batch_dict['reward_raw'] =reward_raw / self.horizon_length
        if self.pc_use_as_reward:
            batch_dict['pc_reward_penalty'] = pc_reward_penalty / self.horizon_length
            batch_dict['pc_rollout_p_human'] = pc_rollout_p_human / self.horizon_length
            batch_dict['pc_rollout_p_human_samples'] = pc_rollout_p_human_samples
            batch_dict['pc_rollout_p_human_min'] = (
                pc_rollout_p_human_min
                if torch.isfinite(pc_rollout_p_human_min)
                else torch.tensor(0.0, device=self.device)
            )
            batch_dict['pc_rollout_p_human_max'] = (
                pc_rollout_p_human_max
                if torch.isfinite(pc_rollout_p_human_max)
                else torch.tensor(0.0, device=self.device)
            )
            batch_dict['pc_rollout_p_human_nonzero_ratio'] = (
                pc_rollout_p_human_nonzero / torch.clamp_min(pc_rollout_p_human_count, 1.0)
            )
        batch_dict['played_frames'] = self.batch_size
        
        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)
        batch_dict['mb_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)
        
        return batch_dict

    def prepare_dataset(self, batch_dict):
        
        
        dataset_dict = super().prepare_dataset(batch_dict)
        dataset_dict['amp_obs'] = batch_dict['amp_obs']
        dataset_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        dataset_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']

        if self.save_kin_info:
            dataset_dict['kin_dict'] = batch_dict['kin_dict']
        
        if self.vec_env.env.task.z_type == "vae":
            dataset_dict['z_noise'] = batch_dict['z_noise']
            
        self.dataset.update_values_dict(dataset_dict, rnn_format = True, horizon_length = self.horizon_length, num_envs = self.num_actors)
        # self.dataset.update_values_dict(dataset_dict)

        return

    def train_epoch(self):
        self.pre_epoch(self.epoch_num)
        play_time_start = time.time()

        ### ZL: do not update state weights during play

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self._update_amp_demos()
        num_obs_samples = batch_dict['amp_obs'].shape[0]
        amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)['amp_obs']
        batch_dict['amp_obs_demo'] = amp_obs_demo

        if (self._amp_replay_buffer.get_total_count() == 0):
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
        else:
            batch_dict['amp_obs_replay'] = self._amp_replay_buffer.sample(num_obs_samples)['amp_obs']

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        # if self.is_rnn:
        # frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)
            
        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_amp_obs(batch_dict['amp_obs'])

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        train_info['terminated_flags'] = batch_dict['terminated_flags']
        train_info['reward_raw'] = batch_dict['reward_raw']
        train_info['mb_rewards'] = batch_dict['mb_rewards']
        train_info['returns'] = batch_dict['returns']
        if self.pc_use_as_reward and 'pc_reward_penalty' in batch_dict:
            train_info['pc_reward_penalty'] = batch_dict['pc_reward_penalty']
        if self.pc_use_as_reward and 'pc_rollout_p_human' in batch_dict:
            train_info['pc_rollout_p_human'] = batch_dict['pc_rollout_p_human']
        if self.pc_use_as_reward and 'pc_rollout_p_human_samples' in batch_dict:
            train_info['pc_rollout_p_human_samples'] = batch_dict['pc_rollout_p_human_samples']
        if self.pc_use_as_reward and 'pc_rollout_p_human_min' in batch_dict:
            train_info['pc_rollout_p_human_min'] = batch_dict['pc_rollout_p_human_min']
        if self.pc_use_as_reward and 'pc_rollout_p_human_max' in batch_dict:
            train_info['pc_rollout_p_human_max'] = batch_dict['pc_rollout_p_human_max']
        if self.pc_use_as_reward and 'pc_rollout_p_human_nonzero_ratio' in batch_dict:
            train_info['pc_rollout_p_human_nonzero_ratio'] = batch_dict['pc_rollout_p_human_nonzero_ratio']
        self._record_train_batch_info(batch_dict, train_info)
        self.post_epoch(self.epoch_num)
        
        if self.save_kin_info:
            print_str = "Kin: " + " \t".join([f"{k}: {torch.mean(torch.tensor(train_info[k])):.4f}" for k, v in train_info.items() if k.startswith("kin")])
            print(print_str)

        if self.use_pc_anomaly_loss and "pc_anomaly_loss" in train_info:
            pc_anom = torch.stack(train_info["pc_anomaly_loss"]).mean().item()
            pc_ph = torch.stack(train_info["pc_p_human"]).mean().item()
            print(f"PC: p_human={pc_ph:.4f} (→人形)  pc_anom_loss={pc_anom:.4f}")
            if "pc_grad_abs_sum" in train_info:
                gsum = torch.stack(train_info["pc_grad_abs_sum"]).mean().item()
                gmax = torch.stack(train_info["pc_grad_abs_max"]).mean().item()
                gr = torch.stack(train_info["pc_grad_nonzero_ratio"]).mean().item()
                print(f"PCGradCheck(avg): abs_sum={gsum:.6e} abs_max={gmax:.6e} nonzero_ratio={gr:.4f}")
        if self.pc_use_as_reward and "pc_reward_penalty" in train_info:
            pc_pen_val = train_info["pc_reward_penalty"]
            if isinstance(pc_pen_val, torch.Tensor):
                pc_pen = pc_pen_val.mean().item()
            else:
                pc_pen = torch.mean(torch.tensor(pc_pen_val)).item()
            pc_roll_val = train_info.get("pc_rollout_p_human", None)
            pc_roll = None
            if pc_roll_val is not None:
                if isinstance(pc_roll_val, torch.Tensor):
                    pc_roll = pc_roll_val.mean().item()
                else:
                    pc_roll = torch.mean(torch.tensor(pc_roll_val)).item()
            if pc_roll is None:
                print(f"PCReward: penalty={pc_pen:.6e} (reward -= pc_reward_coef * p_human)")
            else:
                sample_vals = train_info.get("pc_rollout_p_human_samples", [])
                if isinstance(sample_vals, torch.Tensor):
                    sample_vals = sample_vals.detach().cpu().flatten().tolist()
                sample_vals = sample_vals[: self.pc_log_sample_size]
                sample_str = ", ".join([f"{v:.3f}" for v in sample_vals]) if len(sample_vals) > 0 else "-"
                print(f"PCReward: p_human_samples=[{sample_str}] mean={pc_roll:.3f} penalty={pc_pen:.3f}")

        return train_info

    def pre_epoch(self, epoch_num):
        # print("freeze running mean/std")

        if self.vec_env.env.task.humanoid_type in ["smpl", "smplh", "smplx"]:
            humanoid_env = self.vec_env.env.task
            if (epoch_num > 1) and epoch_num % humanoid_env.shape_resampling_interval == 1: # + 1 to evade the evaluations. 
            # if (epoch_num > 0) and epoch_num % humanoid_env.shape_resampling_interval == 0 and not (epoch_num % (self.save_freq)): # Remove the resampling for this. 
                # Different from AMP, always resample motion no matter the motion type.
                print("Resampling Shape")
                humanoid_env.resample_motions()
                # self.current_rewards # Fixing these values such that they do not get whacked by the
                # self.current_lengths
            if humanoid_env.getup_schedule:
                humanoid_env.update_getup_schedule(epoch_num, getup_udpate_epoch=humanoid_env.getup_udpate_epoch)
                if epoch_num > humanoid_env.getup_udpate_epoch:  # ZL fix janky hack
                    self._task_reward_w = 0.5
                    self._disc_reward_w = 0.5
                else:
                    self._task_reward_w = 0
                    self._disc_reward_w = 1

        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)  # Freeze running mean/std, so that the actor does not use the updated mean/std
        self.running_mean_std_temp.freeze()

    def post_epoch(self, epoch_num):
        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)  # Unfreeze running mean/std
        self.running_mean_std_temp.freeze()
        

    def _preproc_obs(self, obs_batch, use_temp=False):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v, use_temp = use_temp)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0

        if self.normalize_input:
            obs_batch_proc = obs_batch[:, :self.running_mean_std.mean_size]
            if use_temp:
                obs_batch_out = self.running_mean_std_temp(obs_batch_proc)
                obs_batch_orig = self.running_mean_std(obs_batch_proc)  # running through mean std, but do not use its value. use temp
            else:
                obs_batch_out = self.running_mean_std(obs_batch_proc)  # running through mean std, but do not use its value. use temp
            obs_batch_out = torch.cat([obs_batch_out, obs_batch[:, self.running_mean_std.mean_size:]], dim=-1)

        return obs_batch_out

    def _forward_pc_normalized_and_p_human(self, kin_dict):
        """SMPL→点云；再得到分类用点云与 p_human。返回 (pc_cls, p_human, pc_viz)。

        pc_viz：仅 SmplToPointCloud 输出，与 scripts/viz_pointcloud_one_frame.py 里 3D 散点图一致。
        pc_cls：root 局部 + 轴变换 + 单位球归一化后送入 PointNet++ 的张量（与奖励/训练一致）。
        """
        def quat_rotate_xyzw(q, v):
            q_xyz = q[..., 0:3]
            q_w = q[..., 3:4]
            t = 2.0 * torch.cross(q_xyz, v, dim=-1)
            return v + q_w * t + torch.cross(q_xyz, t, dim=-1)

        root_pos = kin_dict['root_pos']
        root_rot = kin_dict['root_rot']
        dof_pos = kin_dict['dof_pos']
        B = root_pos.shape[0]
        betas = self.pc_betas.expand(B, -1)
        chunk_size = self.pc_chunk_size or B
        pc_viz_chunks = []
        pc_cls_chunks = []
        logits_list = []
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            rp = root_pos[start:end].to(self.ppo_device)
            rr = root_rot[start:end].to(self.ppo_device)
            dp = dof_pos[start:end].to(self.ppo_device)
            bt = betas[start:end]
            pc_viz = self.smpl_to_pc(root_pos=rp, root_rot_xyzw=rr, dof_pos=dp, betas=bt)
            pc_viz_chunks.append(pc_viz)
            pc = pc_viz
            if self.pc_use_root_local_frame:
                if not self.pc_local_coord:
                    pc = pc - rp.unsqueeze(1)
                rr_n = torch.nn.functional.normalize(rr, p=2, dim=-1)
                rr_inv = torch.cat([-rr_n[:, 0:3], rr_n[:, 3:4]], dim=-1)
                rr_inv = rr_inv[:, None, :].expand(-1, pc.shape[1], -1)
                pc = quat_rotate_xyzw(rr_inv.reshape(-1, 4), pc.reshape(-1, 3)).view(pc.shape[0], pc.shape[1], 3)
            pc = pc[:, :, list(self.pc_axis_perm)] * self.pc_axis_sign_tensor
            pc_cls = pc_normalize_torch(pc)
            pc_cls_chunks.append(pc_cls)
            logits_list.append(self.pc_classifier(pc_cls))
        pc_cls_all = torch.cat(pc_cls_chunks, dim=0)
        pc_viz_all = torch.cat(pc_viz_chunks, dim=0)
        logits_pc = torch.cat(logits_list, dim=0)
        probs_pc = torch.softmax(logits_pc, dim=-1)
        p_human = probs_pc[:, self.pc_human_class_idx]
        return pc_cls_all, p_human, pc_viz_all

    def _calc_pc_human_prob_from_kin_dict(self, kin_dict):
        _, p_human, _ = self._forward_pc_normalized_and_p_human(kin_dict)
        return p_human

    def _pc_save_batch_passes_threshold(self, p_human_step):
        if not (p_human_step < self.pc_dataset_save_per_max).all().item():
            return False
        if self.pc_dataset_save_mean_max > 0.0 and not (p_human_step.mean() < self.pc_dataset_save_mean_max).item():
            return False
        return True

    def _maybe_save_pc_dataset_batch(self, p_human_step, horizon_step, pc_cls_all, pc_viz_all):
        if not self.pc_dataset_save_enable:
            return
        if self._pc_dataset_save_count >= self.pc_dataset_save_max_files:
            return
        if not self._pc_save_batch_passes_threshold(p_human_step):
            return
        try:
            out_root = osp.join(osp.dirname(__file__), "..", "..", self.pc_dataset_save_dir)
            os.makedirs(out_root, exist_ok=True)
            fn = osp.join(
                out_root,
                f"pc_ep{int(self.epoch_num):06d}_h{int(horizon_step):03d}_{self._pc_dataset_save_count:06d}.npz",
            )
            np.savez_compressed(
                fn,
                pc=pc_viz_all.detach().cpu().numpy().astype(np.float32),
                pc_cls=pc_cls_all.detach().cpu().numpy().astype(np.float32),
                p_human=p_human_step.detach().cpu().numpy().astype(np.float32),
                epoch=np.int32(self.epoch_num),
                horizon_step=np.int32(horizon_step),
                per_max=np.float32(self.pc_dataset_save_per_max),
                mean_max=np.float32(self.pc_dataset_save_mean_max),
            )
            self._pc_dataset_save_count += 1
            if self._pc_dataset_save_count <= 5 or self._pc_dataset_save_count % 50 == 0:
                print(f"[pc_dataset] saved {fn} (n={p_human_step.shape[0]}, mean_p_human={p_human_step.mean().item():.4f})")
        except Exception as e:
            print(f"[pc_dataset] save failed: {e}")

    def _maybe_save_smpl_dataset_batch(self, kin_dict, p_human_step, horizon_step):
        """筛选条件与点云数据集相同；写入单独 npz（不含 pc）。"""
        if not self.pc_smpl_save_enable:
            return
        if self._pc_smpl_save_count >= self.pc_smpl_save_max_files:
            return
        if not self._pc_save_batch_passes_threshold(p_human_step):
            return
        try:
            out_root = osp.join(osp.dirname(__file__), "..", "..", self.pc_smpl_save_dir)
            os.makedirs(out_root, exist_ok=True)
            fn = osp.join(
                out_root,
                f"smpl_ep{int(self.epoch_num):06d}_h{int(horizon_step):03d}_{self._pc_smpl_save_count:06d}.npz",
            )
            rp = kin_dict["root_pos"].detach().cpu().numpy().astype(np.float32)
            rr = kin_dict["root_rot"].detach().cpu().numpy().astype(np.float32)
            dp = kin_dict["dof_pos"].detach().cpu().numpy().astype(np.float32)
            B = int(rp.shape[0])
            betas_np = self.pc_betas.expand(B, -1).detach().cpu().numpy().astype(np.float32)
            np.savez_compressed(
                fn,
                root_pos=rp,
                root_rot_xyzw=rr,
                dof_pos=dp,
                smpl_betas=betas_np,
                p_human=p_human_step.detach().cpu().numpy().astype(np.float32),
                epoch=np.int32(self.epoch_num),
                horizon_step=np.int32(horizon_step),
                per_max=np.float32(self.pc_dataset_save_per_max),
                mean_max=np.float32(self.pc_dataset_save_mean_max),
            )
            self._pc_smpl_save_count += 1
            if self._pc_smpl_save_count <= 5 or self._pc_smpl_save_count % 50 == 0:
                print(f"[pc_smpl] saved {fn} (n={B}, mean_p_human={p_human_step.mean().item():.4f})")
        except Exception as e:
            print(f"[pc_smpl] save failed: {e}")

    def _maybe_save_lidar_dataset_batch(self, kin_dict, p_human_step, horizon_step):
        """射线–网格 LiDAR 仿真点云；筛选与 pc_dataset 相同；不参与奖励。"""
        if not self.pc_lidar_save_enable:
            return
        if self._pc_lidar_save_count >= self.pc_lidar_save_max_files:
            return
        if not self._pc_save_batch_passes_threshold(p_human_step):
            return
        try:
            root_pos = kin_dict['root_pos']
            root_rot = kin_dict['root_rot']
            dof_pos = kin_dict['dof_pos']
            B = int(root_pos.shape[0])
            betas = self.pc_betas.expand(B, -1)
            vert_angles = vertical_angles_deg_for_lidar(
                self.pc_lidar_lines, self.pc_lidar_fov_up_deg, self.pc_lidar_fov_down_deg
            )
            faces_np = self.smpl_to_pc.mesh_faces_numpy()
            lidar_seed = int(self.epoch_num) * 1_000_003 + int(horizon_step) * 97 + self._pc_lidar_save_count

            with torch.no_grad():
                verts = self.smpl_to_pc.mesh_vertices_world(
                    root_pos.to(self.ppo_device),
                    root_rot.to(self.ppo_device),
                    dof_pos.to(self.ppo_device),
                    betas.to(self.ppo_device),
                )

            n_env = min(B, max(1, self.pc_lidar_max_envs))
            merge_pts = []
            merge_range = []
            merge_ring = []
            merge_az = []
            merge_int = []
            merge_face = []
            merge_env = []
            for b in range(n_env):
                v_np = verts[b].detach().cpu().numpy().astype(np.float32)
                lid = simulate_lidar_scan(
                    v_np,
                    faces_np,
                    distance=self.pc_lidar_distance,
                    sensor_height=self.pc_lidar_sensor_height,
                    vertical_angles_deg=vert_angles,
                    horizontal_res_deg=self.pc_lidar_horizontal_res_deg,
                    min_range=self.pc_lidar_min_range,
                    max_range=self.pc_lidar_max_range,
                    range_noise_std=0.0,
                    dropout=0.0,
                    scan_padding_deg=self.pc_lidar_scan_padding_deg,
                    ray_chunk_size=self.pc_lidar_ray_chunk_size,
                    seed=lidar_seed + b,
                )
                pts = lid['points']
                if pts.shape[0] == 0:
                    continue
                merge_pts.append(pts)
                merge_range.append(lid['range'])
                merge_ring.append(lid['ring'])
                merge_az.append(lid['azimuth'])
                merge_int.append(lid['intensity'])
                merge_face.append(lid['face_index'])
                merge_env.append(np.full((pts.shape[0],), b, dtype=np.int32))

            if not merge_pts:
                print("[pc_lidar] skip save: zero hits for all sampled envs")
                return

            out_root = osp.join(osp.dirname(__file__), "..", "..", self.pc_lidar_save_dir)
            os.makedirs(out_root, exist_ok=True)
            fn = osp.join(
                out_root,
                f"lidar_ep{int(self.epoch_num):06d}_h{int(horizon_step):03d}_{self._pc_lidar_save_count:06d}.npz",
            )
            np.savez_compressed(
                fn,
                lidar_points=np.concatenate(merge_pts, axis=0).astype(np.float32),
                lidar_range=np.concatenate(merge_range, axis=0).astype(np.float32),
                lidar_ring=np.concatenate(merge_ring, axis=0).astype(np.int32),
                lidar_azimuth_deg=np.concatenate(merge_az, axis=0).astype(np.float32),
                lidar_intensity=np.concatenate(merge_int, axis=0).astype(np.float32),
                lidar_face_index=np.concatenate(merge_face, axis=0).astype(np.int32),
                lidar_env_index=np.concatenate(merge_env, axis=0).astype(np.int32),
                lidar_vertical_angles_deg=vert_angles.astype(np.float32),
                p_human=p_human_step.detach().cpu().numpy().astype(np.float32),
                epoch=np.int32(self.epoch_num),
                horizon_step=np.int32(horizon_step),
                per_max=np.float32(self.pc_dataset_save_per_max),
                mean_max=np.float32(self.pc_dataset_save_mean_max),
                lidar_distance_m=np.float32(self.pc_lidar_distance),
                lidar_sensor_height_m=np.float32(self.pc_lidar_sensor_height),
                lidar_num_envs_saved=np.int32(len(merge_pts)),
            )
            self._pc_lidar_save_count += 1
            if self._pc_lidar_save_count <= 5 or self._pc_lidar_save_count % 50 == 0:
                npts = int(sum(x.shape[0] for x in merge_pts))
                print(f"[pc_lidar] saved {fn} (points={npts}, envs_merged={len(merge_pts)})")
        except Exception as e:
            print(f"[pc_lidar] save failed: {e}")

    def _dump_last_kin_frame(self, kin_dict):
        try:
            import joblib
            out_dir = osp.join(osp.dirname(__file__), "..", "..", "output", "pc_classifier")
            os.makedirs(out_dir, exist_ok=True)
            root_pos = kin_dict["root_pos"]
            root_rot = kin_dict["root_rot"]
            dof_pos = kin_dict["dof_pos"]
            joblib.dump(
                {
                    "root_pos": root_pos[0].detach().cpu().numpy(),
                    "root_rot": root_rot[0].detach().cpu().numpy(),
                    "dof_pos": dof_pos[0].detach().cpu().numpy(),
                    "betas": self.pc_betas[0].detach().cpu().numpy(),
                },
                osp.join(out_dir, "last_kin_frame.pkl"),
            )
        except Exception:
            pass

    def calc_gradients(self, input_dict):
        
        self.set_train()
        humanoid_env = self.vec_env.env.task

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch_processed = self._preproc_obs(obs_batch, use_temp=self.temp_running_mean)
        input_dict['obs_processed'] = obs_batch_processed

        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip
        
        self.train_result = {}
        kin_loss_info = None
        if self.only_kin_loss:
            # pure behavior cloning, kinemaitc loss.
            batch_dict = {}
            batch_dict['obs_orig'] = obs_batch
            batch_dict['obs'] = input_dict['obs_processed']
            batch_dict['kin_dict'] = input_dict['kin_dict']
            
            # if humanoid_env.z_type == "vae":
            #     batch_dict['z_noise'] = input_dict['z_noise']
            
            rnn_len = self.horizon_length
            rnn_len = 1
            if self.is_rnn:
                batch_dict['rnn_states'] = input_dict['rnn_states']
                batch_dict['seq_length'] = rnn_len

            kin_loss_info = self._optimize_kin(batch_dict)
            self.train_result.update( {'entropy': torch.tensor(0).float(), 'kl': torch.tensor(0).float(), 'last_lr': self.last_lr, 'lr_mul': torch.tensor(0).float()})
            
        else:
            batch_dict = {
                'is_train': True,
                'amp_steps': self.vec_env.env.task._num_amp_obs_steps,
                'prev_actions': actions_batch,
                'obs': obs_batch_processed,
                'amp_obs': amp_obs,
                'amp_obs_replay': amp_obs_replay,
                'amp_obs_demo': amp_obs_demo,
                "obs_orig": obs_batch,
            }
        
            rnn_masks = None
            rnn_len = self.horizon_length
            rnn_len = 1
            if self.is_rnn:
                rnn_masks = input_dict['rnn_masks']
                batch_dict['rnn_states'] = input_dict['rnn_states']
                batch_dict['seq_length'] = rnn_len
                
                
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                res_dict = self.model(batch_dict) # current model if RNN, has BPTT enabled. 
                
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']
                disc_agent_logit = res_dict['disc_agent_logit']
                disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
                disc_demo_logit = res_dict['disc_demo_logit']

                if not rnn_masks is None:
                    rnn_mask_bool = rnn_masks.squeeze().bool()
                    old_action_log_probs_batch, action_log_probs, advantage, values, entropy, mu, sigma, return_batch, old_mu_batch, old_sigma_batch = \
                        old_action_log_probs_batch[rnn_mask_bool], action_log_probs[rnn_mask_bool], advantage[rnn_mask_bool], values[rnn_mask_bool], \
                            entropy[rnn_mask_bool], mu[rnn_mask_bool], sigma[rnn_mask_bool], return_batch[rnn_mask_bool], old_mu_batch[rnn_mask_bool], old_sigma_batch[rnn_mask_bool]
                    
                    # flatten values for computing loss
                    
                a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
                a_loss = a_info['actor_loss']

                c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
                c_loss = c_info['critic_loss']

                b_loss = self.bound_loss(mu)

                a_loss = torch.mean(a_loss)
                c_loss = torch.mean(c_loss)
                b_loss = torch.mean(b_loss)
                entropy = torch.mean(entropy)

                disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
                
                disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
                disc_loss = disc_info['disc_loss']

                loss = (
                    a_loss
                    + self.critic_coef * c_loss
                    - self.entropy_coef * entropy
                    + self.bounds_loss_coef * b_loss
                    + self._disc_coef * disc_loss
                )

                # Point-cloud anomaly loss: encourage motions classified as non-human
                pc_anom_loss = None
                if self.use_pc_anomaly_loss:
                    if 'kin_dict' not in input_dict:
                        raise RuntimeError(
                            "use_pc_anomaly_loss=True 但当前 batch 中没有 kin_dict；"
                            "请确认 env.save_kin_info=True 并已在 AMPAgent.init_tensors 中启用 kin_dict 缓存。"
                        )

                    kin_dict_flat = input_dict['kin_dict']
                    kin_dict = self._assamble_kin_dict(kin_dict_flat)

                    root_pos = kin_dict['root_pos']      # (B, 3)
                    root_rot = kin_dict['root_rot']      # (B, 4) xyzw
                    dof_pos = kin_dict['dof_pos']        # (B, J*3)
                    p_human = self._calc_pc_human_prob_from_kin_dict(kin_dict)
                    pc_anom_loss = p_human.mean()
                    loss = loss + self.pc_anomaly_loss_coef * pc_anom_loss
                
                
                a_clip_frac = torch.mean(a_info['actor_clipped'].float())

                a_info['actor_loss'] = a_loss
                a_info['actor_clip_frac'] = a_clip_frac
                c_info['critic_loss'] = c_loss

                if pc_anom_loss is not None:
                    disc_info['pc_anomaly_loss'] = pc_anom_loss.detach()
                    disc_info['pc_p_human'] = p_human.mean().detach()
                    if self.pc_debug_check_grad:
                        disc_info['pc_grad_abs_sum'] = torch.tensor(0.0, device=self.ppo_device)
                        disc_info['pc_grad_abs_max'] = torch.tensor(0.0, device=self.ppo_device)
                        disc_info['pc_grad_nonzero_ratio'] = torch.tensor(0.0, device=self.ppo_device)
                    if self.pc_debug_check_grad and (self._pc_grad_check_step % max(self.pc_debug_check_grad_interval, 1) == 0):
                        pc_weighted_loss = self.pc_anomaly_loss_coef * pc_anom_loss
                        grad_abs_sum = 0.0
                        grad_abs_max = 0.0
                        grad_nonzero = 0
                        grad_total = 0
                        if pc_weighted_loss.requires_grad:
                            model_params = [p for p in self.model.parameters() if p.requires_grad]
                            grad_list = torch.autograd.grad(
                                pc_weighted_loss,
                                model_params,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            for g in grad_list:
                                if g is None:
                                    continue
                                abs_g = g.abs()
                                grad_abs_sum += abs_g.sum().item()
                                grad_abs_max = max(grad_abs_max, abs_g.max().item())
                                grad_nonzero += int((abs_g > 0).any().item())
                                grad_total += 1
                        else:
                            # 直接记录“这条 loss 对当前图不可导”，用于诊断而非中断训练
                            grad_total = 0
                        disc_info['pc_grad_abs_sum'] = torch.tensor(grad_abs_sum, device=self.ppo_device)
                        disc_info['pc_grad_abs_max'] = torch.tensor(grad_abs_max, device=self.ppo_device)
                        disc_info['pc_grad_nonzero_ratio'] = torch.tensor(
                            (float(grad_nonzero) / float(max(grad_total, 1))),
                            device=self.ppo_device,
                        )
                        if pc_weighted_loss.requires_grad:
                            print(
                                f"PCGradCheck: step={self._pc_grad_check_step} "
                                f"abs_sum={grad_abs_sum:.6e} abs_max={grad_abs_max:.6e} "
                                f"nonzero_ratio={grad_nonzero}/{max(grad_total, 1)}"
                            )
                        else:
                            print(
                                f"PCGradCheck: step={self._pc_grad_check_step} "
                                "pc_weighted_loss has no grad_fn (no differentiable path to model params)"
                            )
                    try:
                        import joblib
                        out_dir = osp.join(osp.dirname(__file__), "..", "..", "output", "pc_classifier")
                        os.makedirs(out_dir, exist_ok=True)
                        joblib.dump({
                            "root_pos": root_pos[0].cpu().numpy(),
                            "root_rot": root_rot[0].cpu().numpy(),
                            "dof_pos": dof_pos[0].cpu().numpy(),
                            "betas": self.pc_betas[0].detach().cpu().numpy(),
                        }, osp.join(out_dir, "last_kin_frame.pkl"))
                    except Exception:
                        pass

                if self.multi_gpu:
                    self.optimizer.zero_grad()
                else:
                    for param in self.model.parameters():
                        param.grad = None

            self._pc_grad_check_step += 1
            self.scaler.scale(loss).backward()
            
            with torch.no_grad():
                reduce_kl = not self.is_rnn
                kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
                if self.is_rnn:
                    kl_dist = kl_dist.mean()
            
                    
            #TODO: Refactor this ugliest code of the year
            if self.truncate_grads:
                if self.multi_gpu:
                    self.optimizer.synchronize()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                    with self.optimizer.skip_synchronize():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            self.train_result.update( {'entropy': entropy, 'kl': kl_dist, 'last_lr': self.last_lr, 'lr_mul': lr_mul, 'b_loss': b_loss})
            self.train_result.update(a_info)
            self.train_result.update(c_info)
            self.train_result.update(disc_info)
            
        if self.save_kin_info and self.only_kin_loss and (kin_loss_info is not None):
            self.train_result.update(kin_loss_info)

        return

    def _assamble_kin_dict(self, kin_dict_flat):
        B = kin_dict_flat.shape[0]
        len_acc = 0
        kin_dict = {}
        for k, v in self.kin_dict_info.items():
            kin_dict[k] = kin_dict_flat[:, len_acc:(len_acc + v[1][-1])].view(B, *v[0][1:])
            len_acc += v[1][-1]
        return kin_dict

    def _optimize_kin(self, batch_dict):
        info_dict = {}
        humanoid_env = self.vec_env.env.task
        if humanoid_env.distill: 
            kin_dict = self._assamble_kin_dict(batch_dict['kin_dict'])
            gt_action = kin_dict['gt_action']

            kin_body_rot_geo_loss, kin_vel_loss_l2 = 0.0, 0.0
            if humanoid_env.z_type == "vae":
                pred_action, pred_action_sigma, extra_dict = self.model.a2c_network.eval_actor(batch_dict, return_extra = True)
                # kin_body_loss = (pred_action - gt_action).pow(2).mean() * 10  ## MSE
                kin_action_loss = torch.norm(pred_action - gt_action, dim=-1).mean()  ## RMSE
                
                vae_mu, vae_log_var = extra_dict['vae_mu'], extra_dict['vae_log_var']
                if humanoid_env.use_vae_prior or humanoid_env.use_vae_fixed_prior:
                    prior_mu, prior_log_var = self.model.a2c_network.compute_prior(batch_dict)
                    KLD = kl_multi(vae_mu, vae_log_var, prior_mu, prior_log_var).mean()
                else:
                    KLD = -0.5 * torch.sum(1 + vae_log_var - vae_mu.pow(2) - vae_log_var.exp()) / vae_mu.shape[0]
                    
                ar1_prior, regu_prior = 0, 0 
                if humanoid_env.use_ar1_prior:
                    time_zs = vae_mu.view(self.minibatch_size // self.horizon_length, self.horizon_length, -1)
                    phi = 0.99
                    
                    error = time_zs[:, 1:] - time_zs[:, :-1] * phi
                    
                    idxes = kin_dict['progress_buf'].view(self.minibatch_size // self.horizon_length, self.horizon_length, -1)
                    
                    not_consecs = ((idxes[:, 1:] - idxes[:, :-1]) != 1).view(-1)
                    error = error.view(-1, error.shape[-1])
                    error[not_consecs] = 0
                    
                    starteres = ((idxes <= 2)[:, 1:] + (idxes <= 2)[:, :-1]).view(-1) # make sure the "drop" is not affected. 
                    error[starteres] = 0
                    
                    ar1_prior = torch.norm(error, dim=-1).mean() 
                    info_dict["kin_ar1"] = ar1_prior
                    
                if humanoid_env.use_vae_prior_regu:
                    prior_mean_regu = ((prior_mu ** 2).mean() + (vae_mu ** 2).mean()) * 0.001 # penalize large prior values
                    prior_var_regu = ((prior_log_var ** 2).mean() + (vae_log_var ** 2).mean()) * 0.001 # penalize large variance values
                    regu_prior = prior_mean_regu + prior_var_regu
                    info_dict["kin_prior_regu"] = regu_prior
                
                kin_loss = kin_action_loss +  KLD * humanoid_env.kld_coefficient + ar1_prior * humanoid_env.ar1_coefficient + regu_prior * 0.005
                
                
                info_dict["kin_action_loss"] = kin_action_loss
                info_dict["kin_KLD"] = KLD
                
                if KLD > 100:
                    import ipdb; ipdb.set_trace()
                    print("KLD is too large, clipping to 10")
                
                ######### KLD annealing #######
                if humanoid_env.kld_anneal:
                    anneal_start_epoch = 2500
                    anneal_end_epoch = 5000
                    min_val = humanoid_env.kld_coefficient_min
                    if self.epoch_num > anneal_start_epoch:
                        humanoid_env.kld_coefficient = (0.01 - min_val) * max((anneal_end_epoch -self.epoch_num) / (anneal_end_epoch - anneal_start_epoch), 0) + min_val
                    info_dict["kin_kld_w"] = humanoid_env.kld_coefficient
                ######### KLD annealing #######
                
                
                    
                    
            else:
                raise NotImplementedError()    
                
            self.kin_optimizer.zero_grad()
            kin_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.kin_optimizer.step()
            
            info_dict.update({"kin_loss": kin_loss})
            
        return info_dict



    def _load_config_params(self, config):
        super()._load_config_params(config)

        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert (self._amp_minibatch_size <= self.minibatch_size)

        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        
        config['task_obs_size_detail'] = self.vec_env.env.task.get_task_obs_size_detail()
        if self.vec_env.env.task.has_task:
            config['self_obs_size'] = self.vec_env.env.task.get_self_obs_size()
            config['task_obs_size'] = self.vec_env.env.task.get_task_obs_size()

        return config

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()
        return


    def _oracle_loss(self, obs):
        oracle_a, _ = self.oracle_model.a2c_network.eval_actor({"obs": obs})
        model_a, _ = self.model.a2c_network.eval_actor({"obs": obs})
        oracle_loss = (oracle_a - model_a).pow(2).mean(dim=-1) * 50
        return {'oracle_loss': oracle_loss}

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        '''
        disc_agent_logit: replay and current episode logit (fake examples)
        disc_demo_logit: disc_demo_logit logit 
        obs_demo: gradient penalty demo obs (real examples)
        '''
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights)) # make weight small??
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit), create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]

        ### ZL Hack for zeroing out gradient penalty on the shape (406,)
        # if self.vec_env.env.task.__dict__.get("smpl_humanoid", False):
        #     humanoid_env = self.vec_env.env.task
        #     B, feat_dim = disc_demo_grad.shape
        #     shape_obs_dim = 17
        #     if humanoid_env.has_shape_obs:
        #         amp_obs_dim = int(feat_dim / humanoid_env._num_amp_obs_steps)
        #         for i in range(humanoid_env._num_amp_obs_steps):
        #             disc_demo_grad[:,
        #                            ((i + 1) * amp_obs_dim -
        #                             shape_obs_dim):((i + 1) * amp_obs_dim)] = 0

        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)

        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        # print(f"agent_loss: {disc_loss_agent.item():.3f}  | disc_loss_demo {disc_loss_demo.item():.3f}")
        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.detach(),
            'disc_logit_loss': disc_logit_loss.detach(),
            'disc_agent_acc': disc_agent_acc.detach(),
            'disc_demo_acc': disc_demo_acc.detach(),
            'disc_agent_logit': disc_agent_logit.detach(),
            'disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info
    
    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros(batch_shape + self._amp_observation_space.shape, device=self.ppo_device)
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)  # Demo is the data from the dataset. Real samples

        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['amp_obs']
        return

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return

    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        return

    def _norm_disc_reward(self):
        return self._disc_reward_mean_std is not None

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']

        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {'disc_rewards': disc_r}
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))

            if (self._norm_disc_reward()):
                self._disc_reward_mean_std.train()
                norm_disc_r = self._disc_reward_mean_std(disc_r.flatten())
                disc_r = norm_disc_r.reshape(disc_r.shape)
                disc_r = 0.5 * disc_r + 0.25

            disc_r *= self._disc_reward_scale

        return disc_r

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        if (amp_obs.shape[0] > buf_size):
            rand_idx = torch.randperm(amp_obs.shape[0])
            rand_idx = rand_idx[:buf_size]
            amp_obs = amp_obs[rand_idx]

        self._amp_replay_buffer.store({'amp_obs': amp_obs})
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return
    
    def _assemble_train_info(self, train_info, frame):
        train_info_dict = super()._assemble_train_info(train_info, frame)
        
        if "disc_loss" in train_info:
            disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
            train_info_dict.update({
                "disc_loss": torch_ext.mean_list(train_info['disc_loss']).item(),
                "disc_agent_acc": torch_ext.mean_list(train_info['disc_agent_acc']).item(),
                "disc_demo_acc": torch_ext.mean_list(train_info['disc_demo_acc']).item(),
                "disc_agent_logit": torch_ext.mean_list(train_info['disc_agent_logit']).item(),
                "disc_demo_logit": torch_ext.mean_list(train_info['disc_demo_logit']).item(),
                "disc_grad_penalty": torch_ext.mean_list(train_info['disc_grad_penalty']).item(),
                "disc_logit_loss": torch_ext.mean_list(train_info['disc_logit_loss']).item(),
                "disc_reward_mean": disc_reward_mean.item(),
                "disc_reward_std": disc_reward_std.item(),
            })
        
        if "returns" in train_info:
            train_info_dict['returns'] = train_info['returns'].mean().item()
            
        if "mb_rewards" in train_info:
            train_info_dict['mb_rewards'] = train_info['mb_rewards'].mean().item()
        
        # if 'terminated_flags' in train_info:
        #     train_info_dict["success_rate"] =  1 - torch.mean((train_info['terminated_flags'] > 0).float()).item()
        
        if "reward_raw" in train_info:
            for idx, v in enumerate(train_info['reward_raw'].cpu().numpy().tolist()):
                train_info_dict[f"ind_reward.{idx}"] =  v
        
        if "sym_loss" in train_info:
            train_info_dict['sym_loss'] = torch_ext.mean_list(train_info['sym_loss']).item()
        return train_info_dict

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            # print("disc_pred: ", disc_pred, disc_reward)
        return
