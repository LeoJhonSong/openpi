"""验证训练时的绿幕增强: 抠绿幕 + 贴随机背景, 肉眼确认替换效果与背景多样性.

用与训练完全相同的ChromaKeyReplaceBg (经greenaug.build_transform构造, 读数据集
greenaug.json参数 + 约定背景目录), 对真实帧跑增强. 每路相机出一张对照图: 每行一帧,
左列原图, 右侧若干列是同一帧配不同随机背景的结果, 确认前景 (机器人/物体) 保留, 绿幕
被换成随机背景, 且背景每次不同.

用法 (用项目venv, 需先下载MIL textures并软链到data/mil_textures):
    uv run python tools/greenaug_aug_check.py --dataset data/naviai/tcp_gripper_wa1_grasp_the_spoon

输出: out/<image_key>.png (base_0_rgb / right_wrist_0_rgb)
"""

import argparse
from pathlib import Path
import random

import cv2
import greenaug_io
import numpy as np

from openpi.training import greenaug

# 与LeRobotNaviAIGripperDataConfig一致: 模型图像key -> 数据集相机列名
KEY_TO_CAMERA = {"base_0_rgb": "realsense_up", "right_wrist_0_rgb": "right_wrist"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="LeRobot V2数据集根目录")
    p.add_argument("--background-dir", default=greenaug.DEFAULT_BACKGROUND_DIR, help="背景图目录")
    p.add_argument("--out", type=Path, default=Path("outputs/greenaug_aug"), help="输出目录")
    p.add_argument("--num-frames", type=int, default=8, help="抽多少帧 (每帧取自不同episode)")
    p.add_argument("--variants", type=int, default=4, help="每帧配多少个随机背景")
    p.add_argument("--seed", type=int, default=0, help="随机种子 (复现)")
    args = p.parse_args()

    random.seed(args.seed)
    transform = greenaug.build_transform(args.dataset, args.background_dir, KEY_TO_CAMERA)
    if transform is None:
        raise SystemExit(f"{args.dataset}/{greenaug.PARAMS_FILENAME}不存在, 请先用greenaug_tune.py调参")
    print(f"背景图{len(transform.background_paths)}张 <- {args.background_dir}")

    eps = greenaug_io.list_episodes(args.dataset)
    if not eps:
        raise SystemExit(f"{args.dataset}/data/chunk-*/下没有episode parquet文件")
    eps = eps[: args.num_frames]
    args.out.mkdir(parents=True, exist_ok=True)

    cams_present = set(greenaug_io.available_cameras(eps[0], KEY_TO_CAMERA.values()))
    for image_key, cam in KEY_TO_CAMERA.items():
        if image_key not in transform.key_to_params or cam not in cams_present:
            continue
        rows = []
        for ep in eps:
            frame = greenaug_io.decode_rgb(greenaug_io.read_camera_pngs(ep, cam)[0]["bytes"])
            cells = [greenaug_io.label(frame, "original")]
            for _ in range(args.variants):
                # 跑与训练同一条增强: 单key的image dict, transform原地抠图+换背景
                data = {"image": {image_key: frame.copy()}}
                cells.append(transform(data)["image"][image_key])
            rows.append(np.concatenate(cells, axis=1))
        sheet = np.concatenate(rows, axis=0)
        dst = args.out / f"{image_key}.png"
        cv2.imwrite(str(dst), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"[{image_key}] {cam}: {len(eps)}帧, 每帧{args.variants}个随机背景 -> {dst}")


if __name__ == "__main__":
    main()
