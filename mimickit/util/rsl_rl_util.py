import os
import sys


def configure_rsl_rl_path(default_repo_root=None):
    if (default_repo_root is None):
        util_dir = os.path.dirname(__file__)
        mimickit_dir = os.path.dirname(util_dir)
        repo_root = os.path.dirname(mimickit_dir)
        default_repo_root = os.path.join(repo_root, "third_party", "rsl_rl")

    repo_root = os.environ.get("RSL_RL_ROOT", default_repo_root)
    repo_root = os.path.abspath(repo_root)

    pkg_root = os.path.join(repo_root, "rsl_rl")
    if (os.path.isdir(pkg_root) and repo_root not in sys.path):
        sys.path.insert(0, repo_root)

    return repo_root
