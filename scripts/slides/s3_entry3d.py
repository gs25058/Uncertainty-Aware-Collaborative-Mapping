#!/usr/bin/env python3
"""Slide figure 3: screenshot of the human-like 3D entry map (web/traverse_3d_corridor915f.html).

    python3 scripts/slides/s3_entry3d.py [--variant sgbm] [--out paper/figures/slides/s3_entry3d.png]

Runs with the SYSTEM python3 (it has Playwright) and the installed google-chrome.
Headless WebGL works here only through SwiftShader (--use-angle=swiftshader).
Only the PRESENTATION is changed in the page, never the verdict:
  - white background, no fog, the side rail / badges hidden;
  - ceiling cut by the page's own height clip (floor + CLIP_M), so floor tiles show;
  - "standable but not reached" tiles recoloured grey (slide colour rule);
  - a small legend overlaid bottom-right.
Camera: spherical orbit of the page (theta, phi, r, target); phi is measured from
the zenith, so phi = 55 deg is 35 deg above the horizon.
"""
import argparse
import math
import time

from PIL import Image
from playwright.sync_api import sync_playwright

URL = "file:///src/gs25058/cr_RNE/covor_slam/web/traverse_3d_corridor915f.html"
CLIP_M = 0.60

# 39 CSS px x DSF 1.5 = 58 px = 14 pt at 300 dpi
LEGEND = """
<div id="slide-legend" style="position:fixed;right:28px;bottom:24px;background:rgba(255,255,255,0.94);
  border:1px solid #cfcfcf;border-radius:6px;padding:20px 26px;font-family:'NanumGothic',sans-serif;
  font-size:39px;color:#111;line-height:1.5;z-index:10">
  <div><span style="display:inline-block;width:34px;height:34px;background:#5fe3a1;vertical-align:-5px;margin-right:14px"></span>서서 도달</div>
  <div><span style="display:inline-block;width:34px;height:34px;background:#e8cd6e;vertical-align:-5px;margin-right:14px"></span>숙여서 도달</div>
  <div><span style="display:inline-block;width:34px;height:34px;background:#e8590c;vertical-align:-5px;margin-right:14px"></span>기어서 도달</div>
  <div><span style="display:inline-block;width:34px;height:34px;background:#4dabf7;vertical-align:-5px;margin-right:14px"></span>옆으로 틀어 도달</div>
  <div><span style="display:inline-block;width:34px;height:34px;background:#c0eb75;vertical-align:-5px;margin-right:14px"></span>관측 안 된 바닥 건넘 (≤ 0.5 m)</div>
  <div><span style="display:inline-block;width:34px;height:34px;background:#b4b4b4;vertical-align:-5px;margin-right:14px"></span>설 수 있으나 미도달</div>
  <div><span style="display:inline-block;width:34px;height:34px;background:#8fa6b2;vertical-align:-5px;margin-right:14px"></span>몸 높이 장애물</div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sgbm", help="gt | sgbm | sgbm_rt | ideal")
    ap.add_argument("--out", default="paper/figures/slides/s3_entry3d.png")
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--phi-deg", type=float, default=55.0)
    ap.add_argument("--r", type=float, default=22.0)
    ap.add_argument("--target", default="-0.6,-8.0,-1.6")
    args = ap.parse_args()
    tx, ty, tz = (float(v) for v in args.target.split(","))
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                              args=["--use-gl=angle", "--use-angle=swiftshader",
                                    "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
                                    "--no-sandbox"])
        pg = b.new_page(viewport={"width": 2560, "height": 1440}, device_scale_factor=1.5)
        pg.goto(URL, wait_until="load", timeout=120000)
        pg.wait_for_function("typeof built !== 'undefined' && built.length === DATA.variants.length",
                             timeout=120000)
        js = """([key, theta, phi, r, tx, ty, tz, clip, legend]) => {
          const idx = DATA.variants.findIndex(v => v.key === key);
          if (idx < 0) throw new Error('no variant ' + key);
          document.querySelector('.rail').style.display = 'none';
          document.getElementById('zbadge').style.display = 'none';
          document.querySelector('.hint').style.display = 'none';
          document.getElementById('app').style.gridTemplateColumns = '1fr';
          document.getElementById('app').style.height = window.innerHeight + 'px';
          document.getElementById('stage').style.height = window.innerHeight + 'px';
          document.body.style.background = '#ffffff';
          document.getElementById('stage').style.background = '#ffffff';
          renderer.setClearColor(0xffffff, 1);
          scene.fog = null;
          select(idx);
          const grey = new THREE.Color(0xb4b4b4);
          for (const bv of built) {
            for (let k = 0; k < bv.unreach.count; k++) bv.unreach.setColorAt(k, grey);
            if (bv.unreach.instanceColor) bv.unreach.instanceColor.needsUpdate = true;
          }
          state.zclip = DATA.variants[idx].floor_z + clip;
          applyLayers();
          cam.theta = theta; cam.phi = phi; cam.r = r; cam.target.set(tx, ty, tz);
          resize(); placeCamera();
          document.getElementById('stage').insertAdjacentHTML('beforeend', legend);
          return DATA.variants[idx].label;
        }"""
        label = pg.evaluate(js, [args.variant, args.theta, math.radians(args.phi_deg), args.r,
                                 tx, ty, tz, CLIP_M, LEGEND])
        time.sleep(3)
        pg.screenshot(path=args.out, full_page=False)
        b.close()
    Image.open(args.out).save(args.out, dpi=(300, 300))   # dpi tag only, pixels unchanged
    print("captured %s -> %s" % (label, args.out))


if __name__ == "__main__":
    main()
