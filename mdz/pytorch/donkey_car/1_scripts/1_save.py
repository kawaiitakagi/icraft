import argparse
import importlib
import os
import sys
sys.path.append(R"../0_sac_donkey_car")
from torch import nn
import numpy as np
import torch
import torch as th

import yaml
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

    # 模型及参数加载
    _, model_path, log_path = get_model_path(
        args.exp_id,
        folder,
        algo,
        env_name,
        args.load_best,
    )
    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

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

    if algo in off_policy_algos:
        kwargs.update(dict(buffer_size=1))
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=args.device, **kwargs)
    obs = env.reset()

    #pt模型导出
    from ae.autoencoder import load_ae

    autoencoder = load_ae(R"..\weights\models\sac\donkey-mountain-track-v0_1\ae-64_1711535971_best.pkl")
    dumyin = np.fromfile(R"..\2_compile\qtset\donkey_car\48.ftmp", dtype=np.float32)
    dumyin = torch.from_numpy(dumyin.reshape(1,3,80,160))
    dumyin_out = autoencoder(dumyin)

    ftmp_path = R"..\3_deploy\modelzoo\donkey_car\io\inputs\ae"
    dumyin.numpy().astype(np.float32).tofile(ftmp_path + "\\encoder_input.ftmp")
    dumyin_out.detach().numpy().astype(np.float32).tofile(ftmp_path + "\\ae_output.ftmp")

    y = torch.jit.trace(autoencoder, dumyin ,strict=False) 
    torch.jit.save(y, "../2_compile/fmodel/ae.pt")
    print("TorchScript export success, saved in ../2_compile/fmodel/ae.pt")

    #onnx模型导出
    deterministic = True
    sac_path = R"..\3_deploy\modelzoo\donkey_car\imodel"
    model = CustomSAC(model,deterministic=deterministic)
    try:
        with th.no_grad():
            model.eval()
            obs_tensor, vectorized_env = model.base_model.actor.obs_to_tensor(obs)
            th.onnx.export(model, obs_tensor, sac_path+"\\sac_donkey_car.onnx", verbose=True,opset_version=17)
            print('onnx export success, saved in %s' % sac_path)
            ftmp_path = R"..\3_deploy\modelzoo\donkey_car\io\inputs\sac"
            obs_tensor.numpy().astype(np.float32).tofile(ftmp_path + "\\obs_onnx.ftmp")
    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    enjoy()
