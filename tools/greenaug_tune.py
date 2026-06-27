"""GreenAug chroma-key交互式调参器 (网页版).

在服务器上启动一个Flask小服务, 浏览器里用滑块 + 吸管实时调参.
直接读LeRobot V2数据集的parquet, 按"每行一个episode, 横向是该轨迹等距抽的6帧,
竖叠10个episode"铺成矩阵, 只展示"背景涂红"的分割结果. 这样横看能看出同一条
轨迹随时间的变化, 竖看能看出不同episode (不同套餐具/背景) 之间的差异.
顶部留一张参考原图供吸管取色 (取点击处3x3平均色). 点网格里任意一帧可把参考图切成它.

episode间隔抽样: 数据集是"同一套餐具连拍几条再换一套", 序号相邻的episode高度相似.
所以一页的10条在序号上等距跨开 (间隔 = 总数 // 10), 一屏横跨所有套餐具; 翻页换偏移
补采空隙, 30页 (以300条为例) 循环无重复覆盖全部episode.

抠像逻辑复用openpi.training.greenaug, 与greenaug_matte_check.py, 与训练时用的逐像素一致.
每路相机参数 (keycolor/tol) 读写数据集根目录的greenaug.json, 调完点
"保存到greenaug.json"即落盘, preview和训练直接读这份文件.

用法 (用项目venv):
  uv run python tools/greenaug_tune.py --dataset data/naviai/tcp_gripper_wa1_grasp_the_spoon
  # VS Code里把端口 (默认7860) 转发到本地, 浏览器打开 http://localhost:7860
"""

import argparse
import base64
import io
import math
from pathlib import Path

import cv2
from flask import Flask
from flask import jsonify
from flask import request
import greenaug_io
import numpy as np
from PIL import Image

from openpi.training import greenaug

app = Flask(__name__)

ROWS = 10  # 一页竖叠多少个episode
COLS = 6  # 每个episode横向等距抽多少帧
# 候选相机列名 (实际可用的由数据集parquet schema过滤)
CAMERAS = ["realsense_up", "left_wrist", "right_wrist"]
# 没有现成参数时的滑块初值
DEFAULT_PARAMS = {"keycolor": "#0c5a2b", "tol": 20}

STATE = {"dataset": None, "scale": 2, "episodes": [], "params": {}, "cache": {}}


def b64_png(rgb: np.ndarray, scale: int | None = None) -> str:
    """RGB ndarray -> base64 PNG (带data URI前缀).

    scale=None用全局STATE['scale']放大; scale=1输出原始分辨率
    (吸管取色的原图必须用原始分辨率, 否则前端反算坐标对不上)."""
    s = STATE["scale"] if scale is None else scale
    if s != 1:
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def page_count() -> int:
    """总页数 = 总数 // ROWS (每页等距挑ROWS条, stride即页数)."""
    return max(1, len(STATE["episodes"]) // ROWS)


def page_episode_indices(page: int):
    """第page页 (0-based) 等距挑出的episode索引列表.

    stride = 页数; 第page页取page, page+stride, page+2*stride, ...,
    跨开相邻的相似episode, 且各页无重复."""
    total = len(STATE["episodes"])
    stride = page_count()
    return [page + k * stride for k in range(ROWS) if page + k * stride < total]


def load_page(cam: str, page: int):
    """读一页: [(epname, [(frame_idx, rgb), ...]), ...], 按 (cam, page) 缓存原图."""
    key = (cam, page)
    if key in STATE["cache"]:
        return STATE["cache"][key]
    result = []
    for ei in page_episode_indices(page):
        ep = STATE["episodes"][ei]
        fidx = greenaug_io.frame_indices(greenaug_io.count_frames(ep), COLS)
        pngs = greenaug_io.read_camera_pngs(ep, cam)
        frames = [(fi, greenaug_io.decode_rgb(pngs[fi]["bytes"])) for fi in fidx]
        result.append((ep.stem, frames))
    STATE["cache"][key] = result
    return result


def _cbcr(rgb):
    """RGB -> (Cb, Cr), 与greenaug._rgb_to_cbcr同公式 (含floor)."""
    r, g, b = (float(v) for v in rgb)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = math.floor((b - y) * 0.564 + 128.0)
    cr = math.floor((r - y) * 0.713 + 128.0)
    return cb, cr


def suggest_from_samples(samples):
    """多个绿样本 -> (keycolor_hex, 建议tol).

    keycolor = 逐通道中位数; tol = 各样本到该keycolor的色度距离最大值 (向上取整),
    即刚好罩住所有采样到的绿, 把它们都判为背景的单一阈值."""
    arr = np.asarray(samples, dtype=np.float32)  # (N, 3)
    key = np.median(arr, axis=0).round().astype(int)  # 逐通道中位数
    kcb, kcr = _cbcr(key)
    dists = [math.hypot(cb - kcb, cr - kcr) for cb, cr in (_cbcr(s) for s in arr)]
    tol = math.ceil(max(dists)) if dists else 0
    hexc = "#" + "".join(f"{int(v):02x}" for v in key)
    return hexc, tol


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>GreenAug调参</title>
<style>
 body{font-family:monospace;background:#1e1e1e;color:#ddd;margin:16px}
 h2{margin:0 0 8px}
 .top{display:flex;gap:18px;align-items:flex-start;margin-bottom:12px}
 .ref h3{margin:4px;font-weight:normal;color:#9cdcfe}
 canvas,img{border:1px solid #555;image-rendering:pixelated}
 #refcv{cursor:crosshair}
 .row{display:flex;align-items:center;gap:6px;margin-bottom:6px}
 .row .ep{width:150px;color:#9cdcfe;font-size:11px;flex-shrink:0;word-break:break-all}
 .cell{text-align:center}
 .cell img{cursor:pointer}
 .cell small{color:#888;font-size:11px}
 .panel{min-width:380px}
 .ctrl{display:flex;align-items:center;gap:10px;margin:6px 0}
 .ctrl label{width:135px;display:inline-block}
 .ctrl input[type=range]{width:300px}
 .ctrl .val{width:60px;color:#ce9178}
 #cfg{width:760px;background:#252526;color:#b5cea8;border:1px solid #555;padding:8px;margin-top:10px}
 .key{display:inline-block;width:26px;height:18px;border:1px solid #888;vertical-align:middle}
 button{background:#0e639c;color:#fff;border:0;padding:6px 12px;cursor:pointer}
 button:disabled{background:#444;cursor:default}
 select{background:#333;color:#ddd;border:1px solid #555;padding:4px}
 #pager{margin:6px 0 12px;display:flex;align-items:center;gap:12px}
 #pageinfo{color:#888}
 #status{color:#888;margin:10px 0}
 #saveinfo{color:#6a9955;margin-left:10px}
</style></head><body>
<h2>GreenAug chroma-key实时调参 (背景涂红=被抠掉)</h2>

<div class=top>
 <div class=ref>
   <h3>参考原图 (点击累积取样, 3x3平均)</h3>
   <canvas id=refcv></canvas>
   <div style=margin-top:6px>
     相机: <select id=cam></select>
     &nbsp;<button onclick=resetSamples()>重置取样</button>
     &nbsp;<span id=ninfo style=color:#888>已采0点</span>
   </div>
 </div>
 <div class=panel>
  <div class=ctrl><label>keycolor</label>
   <input type=color id=keycolor value="#0c5a2b">
   <span class=key id=keyswatch></span><span class=val id=keyhex>#0c5a2b</span></div>
  <div class=ctrl><label>tol (色度容差)</label>
   <input type=range id=tol min=0 max=80 step=0.1 value=20><span class=val id=tol_v>20</span></div>
  <div><button onclick=saveParams()>保存到greenaug.json</button><span id=saveinfo></span></div>
  <textarea id=cfg rows=2 readonly></textarea>
 </div>
</div>

<div id=pager>
 <button id=prev onclick=gotoPage(-1)>← 上一页</button>
 <span id=pageinfo>-</span>
 <button id=next onclick=gotoPage(1)>下一页 →</button>
</div>

<div id=status></div>
<div id=grid></div>

<script>
const native=new Image();
const off=document.createElement('canvas');
let octx,NW,NH,SCALE,CAM,PAGE=0,NPAGES=1;
let samples=[];  // 累积的绿样本 [[r,g,b],...]

function updateNinfo(){document.getElementById('ninfo').textContent='已采'+samples.length+'点';}

function resetSamples(){samples=[]; updateNinfo();}

async function applySuggest(){
  if(!samples.length) return;
  const s=samples.map(p=>p.join(',')).join(';');
  const j=await (await fetch('/suggest?samples='+encodeURIComponent(s))).json();
  document.getElementById('keycolor').value=j.keycolor;
  document.getElementById('tol').value=j.tol;  // 自动建议的tol
  updateNinfo();
  refresh();
}

function val(id){return document.getElementById(id).value;}

function curParams(){
  return {keycolor:val('keycolor'), tol:+val('tol')};
}

function gotoPage(delta){
  PAGE=(PAGE+delta+NPAGES)%NPAGES;  // 循环翻页, 到底回第一页
  refresh();
}

async function refresh(){
  document.getElementById('tol_v').textContent=val('tol');
  const hex=val('keycolor');
  document.getElementById('keyhex').textContent=hex;
  document.getElementById('keyswatch').style.background=hex;
  document.getElementById('saveinfo').textContent='';
  document.getElementById('pageinfo').textContent='第'+(PAGE+1)+' / '+NPAGES+'页 (间隔抽样)';
  document.getElementById('status').textContent='加载中...';
  const q=new URLSearchParams({camera:CAM,page:PAGE,keycolor:hex,tol:val('tol')});
  const j=await (await fetch('/process?'+q)).json();
  const g=document.getElementById('grid'); g.innerHTML='';
  j.rows.forEach((row,ri)=>{
    const r=document.createElement('div'); r.className='row';
    const ep=document.createElement('div'); ep.className='ep'; ep.textContent=row.ep;
    r.appendChild(ep);
    row.frames.forEach((fr,ci)=>{
      const d=document.createElement('div'); d.className='cell';
      const im=document.createElement('img'); im.src=fr.src;
      im.title='点击 -> 用这张原图取色'; im.onclick=()=>setRef(ri,ci);  // 切参考图
      const cap=document.createElement('small'); cap.textContent=fr.name;
      d.appendChild(im); d.appendChild(document.createElement('br')); d.appendChild(cap);
      r.appendChild(d);
    });
    g.appendChild(r);
  });
  document.getElementById('cfg').value=JSON.stringify({[CAM]:curParams()});
  document.getElementById('status').textContent='';
}

['keycolor','tol'].forEach(
  id=>document.getElementById(id).addEventListener('input',refresh));

async function saveParams(){  // 把当前相机参数写回数据集greenaug.json
  const q=new URLSearchParams({camera:CAM, ...curParams()});
  const j=await (await fetch('/save?'+q)).json();
  document.getElementById('saveinfo').textContent='已写入'+j.path;
}

async function setRef(ri,ci){  // 把顶部参考图换成点中那帧的原图 (保留已采样本, 可跨图累积取色)
  const q=new URLSearchParams({camera:CAM,page:PAGE,row:ri,col:ci});
  const j=await (await fetch('/ref?'+q)).json();
  NW=j.w; NH=j.h;
  native.onload=()=>setupEyedropper();
  native.src=j.ref_src;
}

function setupEyedropper(){
  const cv=document.getElementById('refcv');
  cv.width=NW*SCALE; cv.height=NH*SCALE;
  const ctx=cv.getContext('2d'); ctx.imageSmoothingEnabled=false;
  ctx.drawImage(native,0,0,cv.width,cv.height);
  off.width=NW; off.height=NH; octx=off.getContext('2d');
  octx.drawImage(native,0,0);
  cv.onclick=(e)=>{
    const rect=cv.getBoundingClientRect();
    let x=Math.floor((e.clientX-rect.left)/rect.width*NW);
    let y=Math.floor((e.clientY-rect.top)/rect.height*NH);
    x=Math.max(1,Math.min(NW-2,x)); y=Math.max(1,Math.min(NH-2,y));
    const d=octx.getImageData(x-1,y-1,3,3).data;  // 3x3平均
    let r=0,gg=0,b=0;
    for(let k=0;k<9;k++){r+=d[k*4];gg+=d[k*4+1];b+=d[k*4+2];}
    const avg=[r,gg,b].map(v=>Math.round(v/9));
    samples.push(avg);          // 累积取样, 服务端用中位数合成keycolor + 建议tol
    applySuggest();
  };
}

async function loadCam(cam){
  CAM=cam; PAGE=0; resetSamples();  // 换相机=换绿幕, 回第一页并清空已采样本
  const meta=await (await fetch('/meta?camera='+encodeURIComponent(cam))).json();
  NW=meta.w; NH=meta.h; SCALE=meta.scale; NPAGES=meta.n_pages;
  document.getElementById('keycolor').value=meta.keycolor;
  document.getElementById('tol').value=meta.tol;
  native.onload=()=>{setupEyedropper(); refresh();};
  native.src=meta.ref_src;
}

async function init(){
  const cams=await (await fetch('/cameras')).json();
  const sel=document.getElementById('cam');
  cams.forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o);});
  sel.onchange=()=>loadCam(sel.value);
  loadCam(cams[0]);
}
init();
</script></body></html>"""


def camera_params(cam: str) -> dict:
    """该相机的当前参数: greenaug.json里有就用, 没有给默认."""
    return {**DEFAULT_PARAMS, **STATE["params"].get(cam, {})}


@app.route("/")
def index():
    return PAGE


@app.route("/cameras")
def cameras():
    if not STATE["episodes"]:
        return jsonify([])
    return jsonify(greenaug_io.available_cameras(STATE["episodes"][0], CAMERAS))


@app.route("/meta")
def meta():
    cam = request.args.get("camera", CAMERAS[0])
    page = load_page(cam, 0)
    if not page:
        return jsonify(error="数据集没有episode"), 404
    ref = page[0][1][0][1]  # 第一个episode的第一帧, 作吸管取色参考
    h, w = ref.shape[:2]
    p = camera_params(cam)
    return jsonify(
        w=w,
        h=h,
        scale=STATE["scale"],
        n_pages=page_count(),
        ref_src=b64_png(ref, scale=1),  # 原始分辨率, 放大交给前端canvas
        keycolor=p["keycolor"],
        tol=p["tol"],
    )


@app.route("/ref")
def ref():
    """点击网格某帧 -> 返回该帧原图 (原始分辨率) 供吸管取色. 原图已在load_page缓存."""
    a = request.args
    cam = a.get("camera", CAMERAS[0])
    page, row, col = int(a.get("page", 0)), int(a.get("row", 0)), int(a.get("col", 0))
    rgb = load_page(cam, page)[row][1][col][1]
    return jsonify(ref_src=b64_png(rgb, scale=1), w=rgb.shape[1], h=rgb.shape[0])


@app.route("/suggest")
def suggest():
    """samples="r,g,b;r,g,b;..." -> 中位数keycolor + 建议tol."""
    raw = request.args.get("samples", "").strip()
    pts = [[int(c) for c in p.split(",")] for p in raw.split(";") if p]
    if not pts:
        return jsonify(error="no samples"), 400
    hexc, tol = suggest_from_samples(pts)
    return jsonify(keycolor=hexc, tol=tol, n=len(pts))


@app.route("/process")
def process():
    a = request.args
    cam = a.get("camera", CAMERAS[0])
    page = int(a.get("page", 0))
    p = {"keycolor": a.get("keycolor", "#0c5a2b"), "tol": float(a.get("tol", 20))}
    rows = []
    for epname, frames in load_page(cam, page):
        cells = [{"name": f"f{fi:04d}", "src": b64_png(greenaug.matte_red(im, p))} for fi, im in frames]
        rows.append({"ep": epname, "frames": cells})
    return jsonify(rows=rows)


@app.route("/save")
def save():
    """把当前相机参数写回数据集根的greenaug.json, 并更新内存缓存."""
    a = request.args
    cam = a.get("camera", CAMERAS[0])
    p = {"keycolor": a.get("keycolor"), "tol": float(a.get("tol"))}
    STATE["params"][cam] = p
    path = greenaug.save_params(STATE["dataset"], cam, p)
    return jsonify(path=str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True, help="LeRobot V2数据集根目录")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--scale", type=int, default=2, help="显示放大倍数 (最近邻)")
    args = p.parse_args()

    eps = greenaug_io.list_episodes(args.dataset)
    if not eps:
        raise SystemExit(f"{args.dataset}/data/chunk-*/下没有episode parquet文件")
    STATE["dataset"] = args.dataset
    STATE["scale"] = args.scale
    STATE["episodes"] = eps
    STATE["params"] = greenaug.load_params(args.dataset)

    print(f"数据集{args.dataset}: {len(eps)}个episode, 已有参数相机{list(STATE['params'])}")
    print(f"矩阵{ROWS}行 (episode) x {COLS}列 (轨迹等距帧), 间隔抽样{page_count()}页循环")
    print(f"打开浏览器: http://localhost:{args.port}  (VS Code转发{args.port}端口)")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
