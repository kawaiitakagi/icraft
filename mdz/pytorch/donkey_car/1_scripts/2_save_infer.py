import argparse
import importlib
import os
import sys
sys.path.append(R"../0_sac_donkey_car")
from torch import nn
import numpy as np
import torch as th
import onnxruntime as ort
import yaml
from copy import deepcopy

from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.type_aliases import PyTorchObs


from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import StoreDict, get_model_path

class CustomSAC(nn.Module):
    def __init__(self, base_model,deterministic):
        super(CustomSAC, self).__init__()
        self.base_model = base_model.policy 
        self.deterministic = deterministic

    def forward(self, x):
        x = self.base_model(x,True)
        return x

def enjoy() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="donkey-mountain-track-v0")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="..\\weights\\models")
    parser.add_argument("--algo", help="RL Algorithm", default="sac", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=True, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=True, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    args = parser.parse_args()

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder
    log_path = args.folder+R'\sac\donkey-mountain-track-v0_1'
    # 参数加载


    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # gym环境及参数加载
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)



    # obs = env.reset()
    obs = env.reset()
    sac_path = R"..\3_deploy\modelzoo\donkey_car\imodel"
    ort_session = ort.InferenceSession(sac_path+'\\sac_donkey_car.onnx')
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    #推理
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0





    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0



    generator = range(args.n_timesteps)

    try:
        for _ in generator:


            obs_onnx = deepcopy(obs)
            outputs = ort_session.run([output_name], {input_name: obs_onnx})
            action = outputs[0].reshape((-1, *env.action_space.shape))
            low, high = env.action_space.low, env.action_space.high
            action_onnx=  low + (0.5 * (action + 1.0) * (high - low))
       
            obs, reward, done, infos = env.step(action_onnx)

            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if args.n_envs == 1:
                if _>0 and _%200==0 and args.verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0

    except KeyboardInterrupt:
        pass


    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()


if __name__ == "__main__":
    enjoy()
