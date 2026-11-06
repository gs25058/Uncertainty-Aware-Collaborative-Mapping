#!/usr/bin/env python3
"""Extract a few SmokeSeer examples via HTTP ranges, never download whole GB ZIPs."""
import argparse
import html
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile

ARCHIVES={"real":("1AUQyy07dEDNHkeliUcaldPaDFG7lsY9g",1700099829),
          "synthetic":("1ljZrRI8hFp3GWPirDqi0mmd_nobh2DZV",3609993464)}


class RemoteZip(io.RawIOBase):
    def __init__(self,url,size):
        self.url,self.size,self.pos=url,size,0
        self.downloaded=0
    def seekable(self):
        return True
    def readable(self):
        return True
    def tell(self):
        return self.pos
    def seek(self,offset,whence=0):
        self.pos=(0 if whence==0 else self.pos if whence==1 else self.size)+offset
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n<=0:
            return b""
        if n>40_000_000 or self.downloaded+n>100_000_000:
            raise RuntimeError("reference download exceeds 100 MB budget")
        req=urllib.request.Request(self.url,headers={"Range":f"bytes={self.pos}-{self.pos+n-1}"})
        with urllib.request.urlopen(req,timeout=45) as r:
            if r.status!=206:
                raise RuntimeError(f"server ignored Range (HTTP {r.status}); refused whole ZIP")
            data=r.read(n)
        self.pos+=len(data)
        self.downloaded+=len(data)
        return data


def confirmed_url(fid):
    url=f"https://drive.google.com/uc?export=download&id={fid}"
    with urllib.request.urlopen(url,timeout=30) as r:
        prefix=r.read(20000)
        if prefix.startswith(b"PK"):
            return r.url
    text=prefix.decode()
    form=re.search(r'<form[^>]*action="([^"]+)"',text)
    fields=dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"',text))
    if not form:
        raise RuntimeError("Drive did not provide a download form")
    return html.unescape(form.group(1))+"?"+urllib.parse.urlencode(fields)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--sources-only",action="store_true")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    source_urls={"Demo_fog_simulation_Cityscapes.m":"https://raw.githubusercontent.com/sakaridis/fog_simulation-SFSU_synthetic/master/source/Demo_fog_simulation_Cityscapes.m",
        "haze_linear.m":"https://raw.githubusercontent.com/sakaridis/fog_simulation-SFSU_synthetic/master/source/Fog_simulation/haze_linear.m",
        "transmission_homogeneous_medium.m":"https://raw.githubusercontent.com/sakaridis/fog_simulation-SFSU_synthetic/master/source/Fog_simulation/transmission_homogeneous_medium.m"}
    sources={}
    for name,url in source_urls.items():
        try:
            content=urllib.request.urlopen(url,timeout=30).read()
            (out/name).write_bytes(content)
            sources[name]=dict(url=url,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content))
        except Exception as e:
            sources[name]=dict(url=url,error=str(e))
    (out/"sources.json").write_text(json.dumps(sources,indent=2)+"\n")
    if args.sources_only:return
    report={}
    for name,(fid,size) in ARCHIVES.items():
        try:
            remote=RemoteZip(confirmed_url(fid),size)
            with zipfile.ZipFile(remote) as archive:
                (out/f"{name}_members.json").write_text(json.dumps(
                    [dict(name=p.filename,bytes=p.file_size,compressed=p.compress_size,
                          method=p.compress_type) for p in archive.infolist()],indent=2)+"\n")
                files=[p for p in archive.infolist() if p.filename.lower().endswith((".png",".jpg",".jpeg"))
                       and "__MACOSX" not in p.filename
                       and "thermal" not in Path(p.filename).name.lower()
                       and (name!="real" or Path(p.filename).parent.name=="images")]
                # Spread across sequence/scene directories, and retain names.
                chosen=[]
                for info in files:
                    if "_gt." in info.filename:
                        continue
                    if not chosen or str(Path(info.filename).parent)!=str(Path(chosen[-1].filename).parent):
                        chosen.append(info)
                    if len(chosen)>=3:
                        break
                if not chosen:
                    raise RuntimeError("no intensity image in ZIP")
                saved=[]
                for i,info in enumerate(chosen):
                    data=archive.read(info)
                    path=out/f"{name}_{i}{Path(info.filename).suffix}"
                    path.write_bytes(data)
                    saved.append(dict(archive_member=info.filename,local=str(path),bytes=len(data)))
                report[name]=dict(status="downloaded",file_id=fid,archive_bytes=size,
                                  transferred_bytes=remote.downloaded,examples=saved)
        except Exception as e:
            report[name]=dict(status="blocked",reason=str(e),file_id=fid,archive_bytes=size)
        (out/"download_report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(name,report[name],flush=True)


if __name__=="__main__":
    main()
