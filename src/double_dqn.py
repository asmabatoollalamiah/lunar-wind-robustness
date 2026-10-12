"""Double DQN on top of Stable-Baselines3's DQN.

SB3's DQN uses the original target  r + g * max_a Q_target(s', a).
Double DQN (van Hasselt et al., 2016) picks the action with the online
network and scores it with the target network:
    r + g * Q_target(s',argmax_a Q_online(s', a))
Only the target computation changes, so I override train().
"""
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            data =self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                next_online = self.q_net(data.next_observations)
                best_next = next_online.argmax(dim=1, keepdim=True)
                next_target = self.q_net_target(data.next_observations)
                next_q = th.gather(next_target, dim=1, index=best_next)
                target_q = data.rewards + (1 - data.dones) * self.gamma * next_q
            current_q = th.gather(self.q_net(data.observations), dim=1,
                                  index=data.actions.long())
            loss = F.smooth_l1_loss(current_q, target_q)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
