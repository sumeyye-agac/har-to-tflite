from __future__ import annotations

import platform
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_tensorflow_determinism(seed: int) -> bool:
    try:
        import tensorflow as tf
    except Exception:
        return False
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return True


def write_env_snapshot(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    freeze = _pip_freeze()
    payload = "\n".join(
        [
            f"python={sys.version.replace(chr(10), ' ')}",
            f"platform={platform.platform()}",
            "",
            "[pip-freeze]",
            freeze,
            "",
        ]
    )
    p.write_text(payload, encoding="utf-8")
    return p


def _pip_freeze() -> str:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=30)
        return out.strip()
    except Exception as exc:
        return f"<failed: {exc}>"
