"""用调好的greenaug.json参数批量抠像, 把抠掉的背景涂红, 铺成对照图核对抠像蒙版准不准.

每个episode的每路相机生成一张contact sheet: 每条轨迹等距取N帧 (默认10), 上行原图,
下行抠像后背景涂红, 逐帧上下对照, 一张静态图看完该轨迹, 核对涂红区域恰好是绿幕,
前景 (机器人/物体) 没被误涂. 注意这里只涂红mask, 不换随机背景 (那是greenaug_aug_check
的活), 专门用来盯抠像边界. 抠像复用openpi.training.greenaug, 与greenaug_tune.py
网页里, 与训练时用的逐像素一致.

参数来自数据集根目录的greenaug.json (由greenaug_tune.py调参写出), key=图像列名.

用法 (用项目venv):
    uv run python tools/greenaug_matte_check.py \
        --dataset data/naviai/tcp_gripper_wa1_grasp_the_spoon --out outputs/greenaug_matte_check

输出结构: out/<相机>/<episode>.png
"""

import argparse
from pathlib import Path

import cv2
import greenaug_io
import numpy as np

from openpi.training import greenaug


def contact_sheet(originals, mattes, fidx):
    """上行原图 (标帧号), 下行抠图, 逐帧上下对照拼成一张."""
    top = np.concatenate([greenaug_io.label(o, f"f{fi:04d}") for o, fi in zip(originals, fidx, strict=True)], axis=1)
    bottom = np.concatenate(mattes, axis=1)
    return np.concatenate([top, bottom], axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="LeRobot V2数据集根目录")
    p.add_argument("--out", type=Path, default=Path("outputs/greenaug_matte_check"), help="输出目录")
    p.add_argument("--num-frames", type=int, default=10, help="每条轨迹等距取多少帧")
    p.add_argument("--max-episodes", type=int, default=0, help="只处理前N条 (0=全部, 调试用)")
    args = p.parse_args()

    params = greenaug.load_params(args.dataset)
    if not params:
        raise SystemExit(f"{args.dataset}/{greenaug.PARAMS_FILENAME}不存在或为空, 请先用greenaug_tune.py调参")

    eps = greenaug_io.list_episodes(args.dataset)
    if not eps:
        raise SystemExit(f"{args.dataset}/data/chunk-*/下没有episode parquet文件")
    if args.max_episodes:
        eps = eps[: args.max_episodes]

    cameras = greenaug_io.available_cameras(eps[0], list(params))
    if not cameras:
        raise SystemExit(f"greenaug.json里的相机列在数据集中都找不到: {list(params)}")
    for cam in cameras:
        (args.out / cam).mkdir(parents=True, exist_ok=True)

    print(f"数据集{args.dataset}: {len(eps)}个episode, 相机{cameras}, 每条取{args.num_frames}帧")
    saved = 0
    for ei, ep in enumerate(eps):
        epname = ep.stem
        fidx = greenaug_io.frame_indices(greenaug_io.count_frames(ep), args.num_frames)
        for cam in cameras:
            pngs = greenaug_io.read_camera_pngs(ep, cam)
            originals = [greenaug_io.decode_rgb(pngs[i]["bytes"]) for i in fidx]
            mattes = [greenaug.matte_red(o, params[cam]) for o in originals]
            sheet = contact_sheet(originals, mattes, fidx)
            cv2.imwrite(str(args.out / cam / f"{epname}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
            saved += 1
        if (ei + 1) % 25 == 0 or ei + 1 == len(eps):
            print(f"  [{ei + 1}/{len(eps)}] {epname}")

    print(f"已保存{saved}张对照图 ({len(eps)}个episode x {len(cameras)}个相机) -> {args.out}")


if __name__ == "__main__":
    main()
