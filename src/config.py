"""Hyperparameters, taken from RL Baselines3 Zoo (LunarLander entries).
The same values are used for the calm and randomised conditions."""
PPO_KWARGS = dict(
    n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999,
    n_epochs=4, ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]),
)
PPO_N_ENVS = 16

DQN_KWARGS = dict(
    learning_rate=6.3e-4, batch_size=128, buffer_size=50_000,
    learning_starts=0, gamma=0.99, target_update_interval=250,
    train_freq=4,gradient_steps=-1, exploration_fraction=0.12,
    exploration_initial_eps=1.0, exploration_final_eps=0.1,
    policy_kwargs=dict(net_arch=[256, 256]),
)

EVAL_WINDS = [0, 5, 10, 15, 17.5, 20, 22.5]
