import os
import time

from Callbacks import InfoLoggingCallback
from mini_cyborg_v2 import DefenderCyborgEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Limit thread usage to prevent CPU overload
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

seed = 4
# Create and wrap your environment
env = make_vec_env(
    DefenderCyborgEnv, n_envs=4
)  # If you have multiple CPU cores, increase n_envs
logdir = "logs/train_PPO_v2_seed_" + str(seed) + "_" + time.strftime("%Y%m%d_%H%M%S")

# Instantiate the PPO model (MlpPolicy is suitable for vector observations)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=seed,
    tensorboard_log=logdir,
    device="cpu",
    ent_coef=0.01,
)  # Set verbose to see progress

info_callback = InfoLoggingCallback(verbose=1)

# Train for 10,000 timesteps (adjust to fit your needs)
model.learn(total_timesteps=3_000_000, callback=info_callback)

# Save the model
model.save("ppo_cyborg_agent")
