# OpenPi 目录约定

本目录是 uv 项目 (`pyproject.toml` + `uv.lock`), 根目录下有虚拟环境 `.venv`.

GPU 节点本身就是 K8s 容器 (无 Docker), pi0 训练直接在宿主 uv 环境跑, 不进 `pi-server` 容器.

- 运行脚本一律 `uv run python xxx.py`, 不直接调用 `python`
- 装包优先 `uv add <pkg>` (写入 `pyproject.toml` 并装进 `.venv`)
- 临时装包用 `uv pip install --python /home/zhs/projects/erban-caring/vla/openpi/.venv <pkg>`, 显式指定本 venv
- 禁止 `pip install` / `python -m pip install`, 避免装进系统 Python
