dc := "docker compose"

default: sh

[no-exit-message, no-cd]
sh:
    {{ dc }} exec pi-server bash || true

[no-cd]
build:
    {{ dc }} build pi-server

[no-cd]
restart:
    {{ dc }} restart pi-server

[no-exit-message, no-cd]
logs:
    {{ dc }} logs -f --tail=100 pi-server

[no-cd]
up:
    {{ dc }} up -d pi-server

[no-cd]
down:
    {{ dc }} down pi-server

# ── pi0 LoRA 训练流程 (宿主机 uv 环境, 不进 pi-server 容器) ──
# 用法/参数覆盖/环境约束见 vla/README.md, 决策见 docs/adr/0011-pi0-lora-training-flow.md.

# 训练配置名与数据集 (LeRobot 格式 <root>/<repo_id>), 均可命令行覆盖: just train config=... repo_id=...
config  := "pi0_naviai_gripper_lora_tcp"
repo_id := "naviai/tcp_gripper_wa1_grasp_the_spoon"
weights := "gs://openpi-assets/checkpoints/pi0_base/params"

# 锁单卡, 可 just train gpu=0 覆盖
gpu   := "1"
# 仅 wandb 上行走代理, 权重 GCS 直连不走
proxy := "http://127.0.0.1:7890"

export HF_LEROBOT_HOME := "./data"
norm_stats := "assets/" + config + "/" + repo_id + "/norm_stats.json"

# 预下载 pi0_base 权重 (GCS 匿名直连, 已缓存则跳过)
weights:
    #!/usr/bin/env bash
    cache="$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_base/params"
    if [ -d "$cache" ]; then
      echo ">> 权重已缓存: $cache, 跳过"
    else
      echo ">> 下载 {{ weights }} (GCS 直连)"
      uv run python -c "from openpi.shared import download; print(download.maybe_download('{{ weights }}'))"
    fi

# 计算 norm_stats (本地数据, 仅缺失时跑, 锁单卡)
norm-stats:
    #!/usr/bin/env bash
    if [ -f "{{ norm_stats }}" ]; then
      echo ">> norm_stats 已存在: {{ norm_stats }}, 跳过"
    else
      echo ">> 计算 norm_stats -> {{ norm_stats }} (单卡 cuda:{{ gpu }})"
      CUDA_VISIBLE_DEVICES={{ gpu }} uv run python scripts/compute_norm_stats.py --config-name={{ config }}
    fi

# 单卡训练, exp-name 默认带时间戳版本; 覆盖: just train myexp gpu=0
train exp="":
    #!/usr/bin/env bash
    exp_name="{{ exp }}"
    [ -n "$exp_name" ] || exp_name="tcp_grasp_spoon_$(date +%Y%m%d_%H%M%S)"
    echo ">> 训练 exp-name=$exp_name, GPU=cuda:{{ gpu }}, wandb=online(proxy)"
    CUDA_VISIBLE_DEVICES={{ gpu }} \
    HTTPS_PROXY={{ proxy }} HTTP_PROXY={{ proxy }} NO_PROXY=googleapis.com,.googleapis.com \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py {{ config }} --exp-name="$exp_name"

# 接最近一次 run 继续, 或 just resume <name>
resume exp="":
    #!/usr/bin/env bash
    exp_name="{{ exp }}"
    [ -n "$exp_name" ] || exp_name="$(ls -t checkpoints/{{ config }} 2>/dev/null | head -1)"
    [ -n "$exp_name" ] || { echo "没有可恢复的 run, 用 just resume <name> 指定"; exit 1; }
    echo ">> 恢复 exp-name=$exp_name, GPU=cuda:{{ gpu }}"
    CUDA_VISIBLE_DEVICES={{ gpu }} \
    HTTPS_PROXY={{ proxy }} HTTP_PROXY={{ proxy }} NO_PROXY=googleapis.com,.googleapis.com \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py {{ config }} --exp-name="$exp_name" --resume
