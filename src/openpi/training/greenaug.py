"""GreenAug绿幕抠像 + 随机背景替换, 作为训练专属图像数据增强.

把绿幕背景抠掉, 换成随机背景图, 强迫模型不依赖采集时的绿幕背景, 提升场景泛化
(参考GreenAug-Random, https://arxiv.org/abs/2407.07868). 只在训练data loader里
跑, 推理路径不经过这里, 真机看到的是真实背景, 不做任何抠像.

色度键算法源自经典实现 http://gc-films.com/chromakey.html: 在YCbCr的CbCr平面上算像素到key色的距离.
原算法用tola/tolb双阈值产生软alpha mask做羽化合成 (OBS的同族实现还会消溢色, 见
https://github.com/obsproject/obs-studio/blob/master/plugins/obs-filters/data/chroma_key_filter.effect);
我们这里只做硬二值抠图 (距绿色够远=前景, 否则=背景被换掉), 因为软合成的半透明边缘是
推理时不存在的人造物, 会污染训练分布 (决策见ADR 0012). 单一阈值tol即可决定二值边界,
故只保留keycolor/tol两个参数, 无需torch/cv2/chromakey依赖, dataloader worker里全是
numpy + PIL, 不碰GPU.

每路相机的抠图参数 (keycolor/tol) 与数据集绑定, 放在数据集根目录的greenaug.json
(key=图像列名), 由greenaug_tune.py交互调参写出.
"""

from functools import lru_cache
import json
import pathlib
import random

import numpy as np
from PIL import Image
from PIL import ImageColor

# 数据集图像列前缀; 相机标识用列名 (realsense_up/left_wrist/right_wrist), 拼出完整列名
IMAGE_COLUMN_PREFIX = "observation.images."
# 抠图参数文件名, 与数据集绑定, 放数据集根目录 (同norm_stats跟数据走的思路)
PARAMS_FILENAME = "greenaug.json"
# 随机背景集的约定位置 (相对训练cwd). 背景集是跨数据集共享资源, 通常软链到大存储.
# 用的是MIL材质纹理 (来自 http://rail.eecs.berkeley.edu/datasets/mil_data.zip 里的纹理图).
DEFAULT_BACKGROUND_DIR = "data/mil_textures"


# 取整只发生在这里: Cb/Cr两个色度通道各自floor成整数 (对齐chromakey.torch).
# 由整数Cb/Cr算出的dist是浮点 (sqrt(整数)), foreground_mask里dist>tol是浮点比较, tol不取整.
def _rgb_to_cbcr(rgb):
    """RGB(...,3) float [0,255] -> (Cb, Cr), 公式同chromakey.torch.rgb_to_ycbcr (含floor).

    注意Cb/Cr用未取整的Y计算, 最后各自floor (与原库逐步一致)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = np.floor((b - y) * 0.564 + 128.0)
    cr = np.floor((r - y) * 0.713 + 128.0)
    return cb, cr


def chroma_distance(image, keycolor):
    """每像素到key色的CbCr距离 (H, W) float. image: uint8 HWC RGB.

    距离越小越接近key色 (绿幕背景), 越大越像前景. CbCr公式同chromakey.torch (含floor)."""
    img = image.astype(np.float64)
    cb, cr = _rgb_to_cbcr(img)
    key_rgb = np.asarray(ImageColor.getrgb(keycolor), dtype=np.float64)
    kcb, kcr = _rgb_to_cbcr(key_rgb)
    return np.sqrt((cb - kcb) ** 2 + (cr - kcr) ** 2)


def foreground_mask(image, keycolor, tol):
    """前景bool mask (H, W): 到绿色的色度距离 > tol即前景, 否则判为绿幕背景被换掉.

    tol等价于旧tola/tolb/thresh三参数硬二值化后的单一距离边界 (见模块docstring)."""
    return chroma_distance(image, keycolor) > tol


def composite(image, foreground, background):
    """前景像素保留原图, 背景像素换成background. 三者均uint8 HWC, 尺寸一致."""
    return np.where(foreground[..., None], image, background).astype(np.uint8)


def composite_red(image, foreground):
    """背景涂亮红, 供调参/渲染时肉眼检查抠像质量."""
    out = image.copy()
    out[~foreground] = (255, 0, 0)
    return out


def matte_red(image, params):
    """按一路相机参数dict (keycolor/tol) 抠图, 背景涂红. 调参与抠像核对共用."""
    fg = foreground_mask(image, params["keycolor"], params["tol"])
    return composite_red(image, fg)


def load_params(dataset_root):
    """读数据集根的greenaug.json: {相机列名: {keycolor, tol}}.

    文件不存在返回空dict (对没有绿幕参数的数据集向后兼容, 调用方据此跳过抠像)."""
    path = pathlib.Path(dataset_root) / PARAMS_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_params(dataset_root, camera, params):
    """把一路相机的参数写回数据集根的greenaug.json (合并已有, 不覆盖其他相机)."""
    path = pathlib.Path(dataset_root) / PARAMS_FILENAME
    all_params = load_params(dataset_root)
    all_params[camera] = params
    path.write_text(json.dumps(all_params, indent=2, ensure_ascii=False) + "\n")
    return path


@lru_cache(maxsize=512)
def _load_background(path, h, w):
    """解码背景图并resize到 (h, w) 的uint8 RGB. LRU缓存避免重复解码同一张."""
    im = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


class ChromaKeyReplaceBg:
    """训练专属图像增强: 对指定相机的图像做色度键抠像, 贴随机背景.

    抠的是keycolor指定的单色背景 (默认绿幕, 但任意纯色背景同理). 只在训练data
    loader里跑, 推理路径不经过 (真机看真实背景). 每路相机各自独立随机抽一张背景,
    强迫模型不依赖采集时的背景. 作用在uint8 HWC RGB图像上, 排在jax的
    ColorJitter等颜色增强之前.
    """

    def __init__(self, key_to_params, background_dir=DEFAULT_BACKGROUND_DIR):
        """key_to_params: {image_key: {keycolor, tol}}; background_dir: 背景图目录 (默认MIL纹理集)."""
        self.key_to_params = key_to_params
        self.background_paths = [str(p) for p in sorted(pathlib.Path(background_dir).glob("**/*.png"))]
        if not self.background_paths:
            raise FileNotFoundError(f"背景目录无png图: {background_dir} (先下载MIL textures并软链到此)")

    def __call__(self, data):
        images = data["image"]
        for key, params in self.key_to_params.items():
            if key not in images:
                continue
            img = images[key]
            h, w = img.shape[:2]
            background = _load_background(random.choice(self.background_paths), h, w)
            fg = foreground_mask(img, params["keycolor"], params["tol"])
            images[key] = composite(img, fg, background)
        return data


def build_transform(dataset_root, background_dir, key_to_camera):
    """从数据集greenaug.json构造ChromaKeyReplaceBg; 无参数文件则返回None (跳过增强).

    key_to_camera: {image_key: 相机列名}, 把image dict的key映射到greenaug.json里的相机参数.
    """
    params = load_params(dataset_root)
    key_to_params = {key: params[camera] for key, camera in key_to_camera.items() if camera in params}
    if not key_to_params:
        return None
    return ChromaKeyReplaceBg(key_to_params, background_dir)
