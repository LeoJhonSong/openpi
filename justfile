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

# ── pi0 训练流程 (宿主机 uv 环境, 不进 pi-server 容器) ──
# 用法/参数覆盖/环境约束见 vla/README.md, 命名方案见 docs/adr/0012-unified-data-model-naming.md.

# config 只承载配方 <algo>_<space>_<tuning>; repo_id = naviai/<dataset_id>, 经 --data.repo-id 覆盖喂入,
# 故一个 config 服务多数据集. 均可命令行覆盖: just train config=... repo_id=... gpu=...
# 默认: pi0 tcp_gripper LoRA (单卡)
config  := "pi0_tcp_gripper_lora"
repo_id := "naviai/wa1_grasp_the_spoon_260623_tcp_gripper"
gpu     := "1"

# ── 手动换 tcp_finger 全量 (双卡 FSDP, 配置见 misc/naviai_config.py:pi0_tcp_finger_full) 时, 改上面三行为: ──
# config  := "pi0_tcp_finger_full"
# repo_id := "naviai/wa1_grasp_the_spoon_260623_tcp_finger"
# gpu     := "0,2"

weights := "gs://openpi-assets/checkpoints/pi0_base/params"
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

# 计算 norm_stats (本地数据, 仅缺失时跑); repo_id 覆盖喂入, 与训练同源
norm-stats:
    #!/usr/bin/env bash
    if [ -f "{{ norm_stats }}" ]; then
      echo ">> norm_stats 已存在: {{ norm_stats }}, 跳过"
    else
      echo ">> 计算 norm_stats -> {{ norm_stats }} (cuda:{{ gpu }})"
      CUDA_VISIBLE_DEVICES={{ gpu }} uv run python scripts/compute_norm_stats.py --config-name={{ config }} --repo-id={{ repo_id }}
    fi

# 训练, 先补跑 weights + norm-stats (依赖, 各自裸 shell 不挂代理), 再在 detached tmux 会话拉起.
# 会话名 = exp-name (默认即 model_id <dataset_id>-<config>-<时间戳>); 覆盖: just train myexp gpu=0
train exp="": weights norm-stats
    #!/usr/bin/env bash
    dataset_id="{{ repo_id }}"; dataset_id="${dataset_id##*/}"
    exp_name="{{ exp }}"
    [ -n "$exp_name" ] || exp_name="${dataset_id}-{{ config }}-$(date +%y%m%d%H%M)"
    if tmux has-session -t "$exp_name" 2>/dev/null; then
      echo ">> tmux 会话 '$exp_name' 已存在, 先 tmux attach -t $exp_name 或 tmux kill-session -t $exp_name"; exit 1
    fi
    echo ">> 训练 exp-name=$exp_name, repo_id={{ repo_id }}, GPU=cuda:{{ gpu }}, wandb=online(proxy)"
    tmux new-session -d -s "$exp_name" \
      "CUDA_VISIBLE_DEVICES={{ gpu }} HTTPS_PROXY={{ proxy }} HTTP_PROXY={{ proxy }} NO_PROXY=googleapis.com,.googleapis.com XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py {{ config }} --exp-name=$exp_name --data.repo-id={{ repo_id }}; exec bash"
    echo ">> 已在 tmux 会话 '$exp_name' 启动训练, 查看: tmux attach -t $exp_name"

# 接最近一次 run 继续 (同样在 detached tmux 会话里), 或 just resume <name>
resume exp="":
    #!/usr/bin/env bash
    exp_name="{{ exp }}"
    [ -n "$exp_name" ] || exp_name="$(ls -t checkpoints/{{ config }} 2>/dev/null | head -1)"
    [ -n "$exp_name" ] || { echo "没有可恢复的 run, 用 just resume <name> 指定"; exit 1; }
    if tmux has-session -t "$exp_name" 2>/dev/null; then
      echo ">> tmux 会话 '$exp_name' 已存在, 先 tmux attach -t $exp_name 或 tmux kill-session -t $exp_name"; exit 1
    fi
    echo ">> 恢复 exp-name=$exp_name, GPU=cuda:{{ gpu }}"
    tmux new-session -d -s "$exp_name" \
      "CUDA_VISIBLE_DEVICES={{ gpu }} HTTPS_PROXY={{ proxy }} HTTP_PROXY={{ proxy }} NO_PROXY=googleapis.com,.googleapis.com XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py {{ config }} --exp-name=$exp_name --data.repo-id={{ repo_id }} --resume; exec bash"
    echo ">> 已在 tmux 会话 '$exp_name' 恢复训练, 查看: tmux attach -t $exp_name"
