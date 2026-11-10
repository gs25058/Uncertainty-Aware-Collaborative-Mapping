#!/usr/bin/env python3
"""Exercise the offline viewer in Chrome and save actual browser previews.

Requires websocket-client only for this optional browser audit; not for scene
generation, sensing, or opening the HTML viewer.
"""
import argparse
import base64
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
import websocket


def main():
    p=argparse.ArgumentParser();p.add_argument('--viewer',default='web/fire_buildings_3d.html');p.add_argument('--out',default='reports/fire_buildings_20261107');a=p.parse_args()
    out=Path(a.out);(out/'previews').mkdir(parents=True,exist_ok=True)
    chrome=shutil.which('google-chrome') or shutil.which('chromium')
    if not chrome:raise RuntimeError('Chrome/Chromium is required for this optional audit')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='covor_fire_chrome_') as profile,(out/'browser.log').open('w') as log:
        process=subprocess.Popen([chrome,'--headless','--no-sandbox','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--use-angle=swiftshader','--enable-unsafe-swiftshader',f'--remote-debugging-port={port}',f'--remote-allow-origins=http://localhost:{port}',f'--user-data-dir={profile}','--window-size=1500,1000','about:blank'],stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+20;page=None
            while time.monotonic()<deadline:
                try:
                    pages=json.load(urllib.request.urlopen(f'http://localhost:{port}/json/list',timeout=1));page=next(p for p in pages if p['type']=='page');break
                except Exception:time.sleep(.1)
            if page is None:raise RuntimeError('Chrome debugging endpoint did not start')
            ws=websocket.create_connection(page['webSocketDebuggerUrl'],origin=f'http://localhost:{port}',timeout=20)
            counter=0;errors=[]
            def command(method,params=None):
                nonlocal counter
                counter+=1;ws.send(json.dumps(dict(id=counter,method=method,params=params or {})))
                while True:
                    r=json.loads(ws.recv())
                    if r.get('method')=='Runtime.exceptionThrown':errors.append(r['params'])
                    if r.get('id')==counter:
                        if 'error' in r:raise RuntimeError(r['error'])
                        return r.get('result',{})
            def evaluate(expression):
                r=command('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
                if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
                return r['result'].get('value')
            def wait_ready(scene_id=None):
                deadline=time.monotonic()+30
                while time.monotonic()<deadline:
                    s=evaluate('window.FIRE_VIEWER_STATUS')
                    if s and s.get('ready') and (scene_id is None or s['scene']==scene_id):return s
                    time.sleep(.05)
                raise TimeoutError('Viewer did not finish rendering')
            def screenshot(name):
                data=command('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']
                (out/'previews'/name).write_bytes(base64.b64decode(data))
            command('Page.enable');command('Runtime.enable')
            command('Page.navigate',dict(url=Path(a.viewer).resolve().as_uri()))
            wait_ready();catalog=evaluate('window.FIRE_SCENES.map(s=>({id:s.id,levels:s.levels}))');checks=[]
            for index,s in enumerate(catalog):
                evaluate(f"document.getElementById('scene').value={index};document.getElementById('scene').dispatchEvent(new Event('change'))")
                wait_ready(s['id'])
                # Full building and each floor use the same buffers and material controls.
                evaluate("document.querySelector('#levels button[data-floor=\"-1\"]').click()")
                time.sleep(.08);screenshot(s['id']+'_overview.png')
                for f in range(s['levels']):
                    evaluate(f"document.querySelector('#levels button[data-floor=\"{f}\"]').click()")
                    result=evaluate("({error:document.getElementById('view').getContext('webgl2').getError(),description:document.getElementById('readout').textContent})")
                    assert result['error']==0,result
                evaluate("document.querySelector('#levels button[data-floor=\"0\"]').click()")
                if index in (0,2,5):screenshot(s['id']+'_L1.png')
                evaluate("document.querySelector('#levels button[data-floor=\"-1\"]').click()")
                for control in ('cut','explode','roof','furniture','fire','smoke','labels'):
                    for _ in range(2):
                        evaluate(f"document.getElementById('{control}').click();new Promise(resolve=>requestAnimationFrame(()=>resolve(true)))")
                        assert evaluate("document.getElementById('view').getContext('webgl2').getError()") == 0
                assert evaluate("document.getElementById('view').getContext('webgl2').getError()") == 0
                checks.append(dict(scene=s['id'],floors_tested=s['levels'],webgl_error=0,controls_tested=True))
                print('browser PASS',s['id'],flush=True)
            evaluate("location.hash='scene=1&floor=0'");wait_ready(catalog[0]['id'])
            assert '1층' in evaluate("document.getElementById('readout').textContent")
            evaluate(f"location.hash='scene={len(catalog)}&floor=-1'");wait_ready(catalog[-1]['id'])
            assert not errors,errors
            (out/'browser_checks.json').write_text(json.dumps(dict(scenes=len(checks),runtime_errors=errors,checks=checks,hash_navigation_after_refresh=True),ensure_ascii=False,indent=2)+'\n')
            ws.close()
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
