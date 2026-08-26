#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小说工作室（Novel Studio）—— memagent 专用写作客户端 v1。

本地 Web IDE：书架管理（婴儿新建 / 打开成年库）、模型配置（提炼强模型与
正文性价比模型分离）、写作台（一键生成+审校评分+进度轮询）、章节阅读器、
精读导入（强模型结构化分析他人小说：节奏/人物弧光/伏笔网络→技法入库）、
记忆仪表盘。仅绑定 127.0.0.1，零新增第三方依赖。

用法：
    python novel_studio.py --home novel_studio --port 8600
    浏览器打开 http://127.0.0.1:8600
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME = ROOT / "novel_studio"
CFG_PATH = HOME / "studio_config.json"

DEFAULT_CFG = {
    "draft_model": "",      # 正文：性价比模型
    "extract_model": "",    # 精读提炼：最强模型
    "judge_model": "",      # 审校评委
    "active_work": "",
}
CFG = dict(DEFAULT_CFG)
ACTIVE: dict[str, str] = {"path": ""}
JOB: dict = {"running": False, "type": "", "status": "idle",
             "log": [], "result": None}


# ---------- 工具 ----------

def _cfg_load() -> None:
    global CFG
    if CFG_PATH.is_file():
        try:
            CFG.update(json.loads(CFG_PATH.read_text(encoding="utf-8")))
        except ValueError:
            pass


def _cfg_save() -> None:
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(json.dumps(CFG, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def _works() -> list[dict]:
    out = []
    if HOME.is_dir():
        for d in sorted(HOME.iterdir()):
            if d.is_dir() and (d / "memory.json").is_file():
                ch = d / "works" / "chapters"
                n = len(list(ch.glob("第*章.md"))) if ch.is_dir() else 0
                out.append({"name": d.name, "path": str(d), "chapters": n})
    return out


def _books_base() -> Path:
    """章节扫描与生成的书籍基目录：显式 wdir 优先于 <work>/works。"""
    if ACTIVE.get("wdir"):
        return Path(ACTIVE["wdir"])
    return Path(ACTIVE.get("path") or "") / "works"


def _chapters_dir() -> Path:
    """兼容三种布局：wdir 平铺 chapters、<work>/chapters、<work>/works/chapters。"""
    d = Path(ACTIVE.get("path") or "")
    cands = []
    if ACTIVE.get("wdir"):
        cands.append(Path(ACTIVE["wdir"]) / "chapters")
    cands += [d / "chapters", d / "works" / "chapters"]
    for c in cands:
        if c.is_dir():
            return c
    return cands[-1]


def _active_agent():
    from memagent.agent import AgentConfig, MemoryAgent
    from memagent.llm import LLMClassifier
    from memagent.responder import LLMResponder

    p = ACTIVE.get("path")
    if not p:
        raise RuntimeError("未选择作品")
    draft = CFG.get("draft_model") or None
    responder = LLMResponder(model=draft, timeout=300.0) if draft else None
    # 写入分类走离线关键词：避免每条 remember 都打一次 LLM（单条 6s×批量）
    return MemoryAgent(
        persist_path=ACTIVE.get("store") or str(Path(p) / "memory.json"),
        cfg=AgentConfig(chapter_save_dir=str(_books_base()),
                        evolve_on_sleep=False),
        classifier=LLMClassifier(api_key=""),
        responder=responder,
    )


def _job_start(jtype: str, fn, *a):
    if JOB["running"]:
        return False
    JOB.update(running=True, type=jtype, status="running", log=[],
               result=None)

    def wrap():
        try:
            JOB["result"] = fn(*a)
            JOB["status"] = "done"
        except Exception as e:
            JOB["result"] = {"error": f"{type(e).__name__}: {e}"}
            JOB["log"].append(f"❌ {traceback.format_exc(limit=3)}")
            JOB["status"] = "error"
        finally:
            JOB["running"] = False
    threading.Thread(target=wrap, daemon=True).start()
    return True


def _job_log(msg: str) -> None:
    JOB["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")


# ---------- 后台任务 ----------

def _task_generate(words: int) -> dict:
    _job_log("初始化写作 agent…")
    agent = _active_agent()
    title = Path(ACTIVE["path"]).name
    _job_log(f"开始生成《{title}》下一章（目标 {words} 字，含审校循环）")
    result = agent.write_chapter(target_words=words, with_web=False)
    agent.save()
    _job_log("✅ 完成：" + json.dumps(
        {k: result.get(k) for k in ("ok", "chapter", "words", "path")},
        ensure_ascii=False))
    return result


def _split_book(text: str, chunk: int = 3200) -> list[str]:
    parts = re.split(r"(?=第[0-9一二两三四五六七八九十百千]+章)", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = [text[i:i + chunk] for i in range(0, len(text), chunk)]
    merged, buf = [], ""
    for p in parts:
        if len(buf) + len(p) > chunk and buf:
            merged.append(buf)
            buf = p
        else:
            buf += "\n" + p
    if buf.strip():
        merged.append(buf)
    return merged


def _task_deepread(txt_path: str, label: str, max_chunks: int) -> dict:
    from memagent.responder import LLMResponder

    agent = _active_agent()   # 技法注入到当前打开的作品
    model = CFG.get("extract_model") or CFG.get("draft_model")
    if not model:
        raise RuntimeError("请先在设置里配置精读模型")
    rsp = LLMResponder(model=model, timeout=240.0)
    raw = Path(txt_path).read_bytes()
    text = None
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        raise RuntimeError("无法解码该文件（utf-8/gbk 均失败）")

    chunks = _split_book(text)[:max_chunks]
    _job_log(f"《{label}》共切分 {len(chunks)} 段，逐段深度精读…")
    prompt_tpl = (
        "你是网文写作技法分析专家。下面是一部畅销小说的片段，请做结构化"
        "精读，只输出 JSON（不要多余文字）：\n"
        '{"pacing": ["节奏技巧一句话", ...], '
        '"dialogue": ["对话技巧一句话", ...], '
        '"foreshadow": ["伏笔手法一句话(不含具体剧情)", ...], '
        '"emotion": ["情绪调动手法一句话", ...]}\n'
        "每类最多 4 条，必须抽象为可复用的写法规律，禁止复述情节。\n\n片段：\n")
    stats = {"segments": 0, "learned": 0}
    for i, ck in enumerate(chunks, 1):
        reply = ""
        for attempt in (1, 2):
            try:
                reply = rsp.respond(prompt_tpl + ck[:3400])
                break
            except Exception as e:
                _job_log(f"  段{i} 调用失败({attempt}/2)：{str(e)[:60]}")
                time.sleep(3)
        m = re.search(r"\{.*\}", reply, re.S)
        if not m:
            _job_log(f"  段{i} 未解析出 JSON，跳过")
            continue
        try:
            data = json.loads(m.group(0))
        except ValueError:
            continue
        for cat, items in data.items():
            if not isinstance(items, list):
                continue
            for it in items:
                s = str(it).strip()
                if len(s) >= 8:
                    agent.remember_skill(f"[{label}/{cat}] {s}",
                                         importance=0.75)
                    stats["learned"] += 1
        stats["segments"] += 1
        _job_log(f"  段{i}/{len(chunks)} 完成，累计技法 {stats['learned']} 条")
        agent.save()
    agent.save()
    _job_log("✅ 精读完成")
    return stats


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静默访问日志
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        route = self.path.split("?")[0]
        if route == "/" or route == "/index.html":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            if route == "/api/state":
                works = _works()
                ext = ACTIVE.get("path", "")
                self._json({"cfg": CFG, "works": works,
                            "active": ext,
                            "active_store": ACTIVE.get("store", ""),
                            "active_name": (Path(ext).name if ext else ""),
                            "job": {k: JOB[k] for k in
                                    ("running", "type", "status",
                                     "log", "result")}})
            elif route == "/api/chapters":
                base = _chapters_dir()
                files = list(base.glob("第*章.md"))
                for sub in base.glob("*"):
                    if (sub / "chapters").is_dir():
                        files += (sub / "chapters").glob("第*章.md")
                seen_p = set()
                files = [f for f in files
                         if not (f.resolve() in seen_p
                                 or seen_p.add(f.resolve()))]
                items = []
                for f in sorted(files,
                                key=lambda x: int(re.search(r"\d+",
                                                  x.stem).group())):
                    t = f.read_text(encoding="utf-8")
                    first = t.splitlines()[0].lstrip("# ").strip()
                    items.append({"no": int(re.search(r"\d+", f.stem).group()),
                                  "title": first, "chars": len(t),
                                  "path": str(f)})
                self._json({"chapters": items})
            elif route == "/api/chapter":
                q = dict(p.split("=", 1) for p in self.path.split("?")[1].split("&"))
                f = _chapters_dir() / f"第{q['no']}章.md"
                self._json({"content": f.read_text(encoding="utf-8")
                            if f.is_file() else ""})
            elif route == "/api/dashboard":
                agent = _active_agent()
                out = Path(ACTIVE["path"]) / "dashboard.html"
                from memagent.interactive import render_interactive_html
                render_interactive_html(agent, str(out))
                self._json({"ok": True, "path": str(out)})
            else:
                self._json({"error": "unknown"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        try:
            b = self._body()
            if self.path == "/api/work/new":
                from memagent.agent import AgentConfig, MemoryAgent
                from memagent.llm import LLMClassifier

                name = re.sub(r'[\\/:*?"<>|]', "_",
                              str(b.get("name", "")).strip())
                if not name:
                    return self._json({"error": "需要作品名"}, 400)
                d = HOME / name
                d.mkdir(parents=True, exist_ok=True)
                agent = MemoryAgent(
                    classifier=LLMClassifier(api_key=""),
                    persist_path=str(d / "memory.json"),
                    cfg=AgentConfig(chapter_save_dir=str(d / "works"),
                                    evolve_on_sleep=False))
                for key, field in (("premise", "故事设定"), ("protagonist", "主角"),
                                   ("antagonist", "反派"), ("worldview", "世界观"),
                                   ("style", "文风")):
                    v = str(b.get(key, "")).strip()
                    if v:
                        agent.remember_setting(f"{field}：{v}", importance=0.9)
                agent.save()
                ACTIVE["path"] = str(d)
                CFG["active_work"] = str(d)
                _cfg_save()
                return self._json({"ok": True, "path": str(d)})
            if self.path == "/api/work/open":
                p = str(b.get("path", "")).strip()
                cand = Path(p)
                store = None
                if cand.is_file() and cand.suffix == ".json":
                    store = cand.resolve()                 # 直接给库文件（如主库）
                    wdir = store.parent if store.name != "memory.json" \
                        else store.parent
                elif (cand / "memory.json").is_file():
                    store = (cand / "memory.json").resolve()
                    wdir = cand.resolve()
                elif (HOME / p / "memory.json").is_file():
                    store = (HOME / p / "memory.json").resolve()
                    wdir = (HOME / p).resolve()
                if not store:
                    return self._json({"error":
                        "找不到该作品（需目录含 memory.json，或直接给 .json）"},
                        404)
                ACTIVE["path"] = str(wdir)
                ACTIVE["store"] = str(store)
                wd = str(b.get("works_dir") or "").strip()
                ACTIVE["wdir"] = str(Path(wd).resolve()) if wd else ""
                CFG["active_work"] = str(store)
                _cfg_save()
                return self._json({"ok": True, "path": ACTIVE["path"],
                                   "store": str(store),
                                   "wdir": ACTIVE["wdir"]})
            if self.path == "/api/config":
                for k in ("draft_model", "extract_model", "judge_model"):
                    if k in b:
                        CFG[k] = str(b[k]).strip()
                _cfg_save()
                return self._json({"ok": True, "cfg": CFG})
            if self.path == "/api/generate":
                words = int(b.get("words", 2200))
                if not ACTIVE.get("path"):
                    return self._json({"error": "未选择作品"}, 400)
                ok = _job_start("generate", _task_generate, words)
                return self._json({"started": ok},
                                  200 if ok else 409)
            if self.path == "/api/import/deepread":
                txt = str(b.get("txt_path", "")).strip()
                if not Path(txt).is_file():
                    return self._json({"error": "文件不存在"}, 400)
                if not ACTIVE.get("path"):
                    return self._json({"error": "先选择要注入技法的作品"}, 400)
                ok = _job_start("deepread", _task_deepread, txt,
                                str(b.get("label") or Path(txt).stem),
                                int(b.get("max_chunks", 10)))
                return self._json({"started": ok}, 200 if ok else 409)
            self._json({"error": "unknown"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>memagent 小说工作室</title>
<style>
 :root{--bg:#14161b;--panel:#1d2027;--line:#2b2f3a;--fg:#dfe3ea;
       --dim:#8b93a3;--acc:#e8b04b;--ok:#7dc98f;}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font-family:"Microsoft YaHei",system-ui,sans-serif;display:flex;height:100vh}
 .side{width:230px;background:var(--panel);border-right:1px solid var(--line);
   padding:14px;overflow-y:auto}
 .side h1{font-size:15px;color:var(--acc);margin:2px 0 12px}
 .w{padding:8px 10px;border-radius:6px;cursor:pointer;margin-bottom:6px;font-size:13px}
 .w:hover{background:#262a34}.w.on{background:#333a49;border-left:3px solid var(--acc)}
 .main{flex:1;padding:18px 26px;overflow-y:auto}
 button{background:var(--acc);border:none;color:#222;padding:8px 16px;
   border-radius:6px;cursor:pointer;font-weight:bold}
 button.sec{background:#333a49;color:var(--fg)}
 button:disabled{opacity:.45;cursor:not-allowed}
 input,textarea{background:#101218;border:1px solid var(--line);color:var(--fg);
   border-radius:6px;padding:8px;width:100%;font-size:13px}
 textarea{min-height:56px;resize:vertical}
 label{font-size:12px;color:var(--dim);display:block;margin:10px 0 4px}
 table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
 td,th{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left}
 tr:hover td{cursor:pointer;background:#232733}
 #log{background:#101218;border:1px solid var(--line);border-radius:6px;
   padding:10px;font-size:12px;white-space:pre-wrap;max-height:180px;
   overflow-y:auto;color:var(--ok)}
 h2{font-size:15px;border-left:3px solid var(--acc);padding-left:8px}
 .row{display:flex;gap:14px}.row>div{flex:1}
 #chapview{white-space:pre-wrap;line-height:1.9;font-size:14px;background:
   var(--panel);padding:20px;border-radius:8px;margin-top:12px}
</style></head><body>
<div class="side"><h1>📖 小说工作室</h1><div id="works"></div>
 <button class="sec" style="width:100%" onclick="show('new')">＋ 新建作品（婴儿）</button>
</div>
<div class="main" id="main"></div>
<script>
let WORKS=[],ACTIVE="",JOB=null,TAB="write";
const $=id=>document.getElementById(id);
async function api(p,b){const r=await fetch(p,{method:b?"POST":"GET",
  headers:{"Content-Type":"application/json"},body:b?JSON.stringify(b):null});
  return r.json()}
async function refresh(){const s=await api("/api/state");
  WORKS=s.works;ACTIVE=s.active_name||s.active;JOB=s.job;
  $("works").innerHTML=s.works.map(w=>
   `<div class="w ${s.active.endsWith(w.path)?'on':''}"
     onclick="openW('${w.path.replace(/\\/g,'\\\\')}')">
     📕 ${w.name}<span style="float:right;color:var(--dim)">${w.chapters}章</span></div>`
  ).join("")||"<div style='color:var(--dim);font-size:12px'>尚无作品</div>";
  if(TAB==="write")drawWrite();}
function openW(p){api("/api/work/open",{path:p}).then(()=>{refresh();TAB="write";drawWrite();})}
function show(t){TAB=t;t==="new"?drawNew():t==="cfg"?drawCfg():t==="read"?drawRead():t==="imp"?drawImp():drawWrite();}
async function drawWrite(){
 const ch=(ACTIVE)?(await api("/api/chapters")).chapters:[];
 $("main").innerHTML=`<h2>写作台 ${ACTIVE?`· ${ACTIVE}`:"（未选择作品）"}</h2>
  <div class="row"><div><label>目标字数</label><input id="words" value="2200"></div>
  <div style="align-self:flex-end"><button id="gen" onclick="gen()">✍ 生成下一章</button>
  <button class="sec" onclick="sleepNow()">😴 睡眠巩固</button></div></div>
  <div id="log" style="margin-top:12px">${(JOB&&JOB.log||[]).join("\n")}</div>
  <h2 style="margin-top:22px">章节列表</h2>
  <table><tr><th>章</th><th>标题</th><th>字数</th></tr>
  ${ch.map(c=>`<tr onclick=viewCh(${c.no})><td>${c.no}</td><td>${c.title}</td><td>${c.chars}</td></tr>`).join("")}
  </table><div id="chapview"></div>`;
 poll();}
function viewCh(no){fetch("/api/chapter?no="+no).then(r=>r.json()).then(d=>{
 $("chapview").textContent=d.content||"（空）"});}
async function gen(){$("gen").disabled=true;
 await api("/api/generate",{words:+$("words").value});poll(true);}
async function sleepNow(){await fetch("/api/sleep",{method:"POST"});
 $("log").textContent+="\n已触发睡眠巩固";}
function poll(active){const iv=setInterval(async()=>{
 const s=await api("/api/state");JOB=s.job;
 const lg=$("log");if(lg&&TAB==="write")lg.textContent=(JOB.log||[]).join("\n");
 if(!JOB.running){clearInterval(iv);
   if(active){refresh();alert("任务完成："+JOB.status)}else drawWrite?0:0;}
 },1500);}
function drawNew(){$("main").innerHTML=`<h2>新建作品（从婴儿开始）</h2>
 <div class="row"><div><label>作品名</label><input id="n_name"></div></div>
 <label>故事设定</label><textarea id="n_premise"></textarea>
 <div class="row"><div><label>主角</label><textarea id="n_pro"></textarea>
 </div><div><label>反派</label><textarea id="n_ant"></textarea></div></div>
 <label>世界观</label><textarea id="n_world"></textarea>
 <label>文风要求</label><textarea id="n_style"></textarea>
 <p><button onclick="createW()">🌱 创建并开始</button>
 <span style="color:var(--dim);font-size:12px">　旧作不受影响，可随时从左侧切回</span></p>`;}
async function createW(){const b={};["name","premise","pro","ant","world","style"]
 .forEach(k=>b[k===""?"":k]=$("n_"+k)?$("n_"+k).value.trim():"");
 b.name=b.name=$("n_name").value;b.premise=$("n_premise").value;
 b.protagonist=$("n_pro").value;b.antagonist=$("n_ant").value;
 b.worldview=$("n_world").value;b.style=$("n_style").value;
 const r=await api("/api/work/new",b);r.ok?(refresh(),drawWrite()):alert(r.error)}
function drawCfg(){const c=JOB&&window.CFG||window.CFG||{};
 $("main").innerHTML=`<h2>设置 · 模型分工</h2>
 <label>正文初稿（性价比模型）</label><input id="c_draft" value="${window.CFG?.draft_model||''}">
 <label>精读提炼（最强模型）</label><input id="c_ext" value="${window.CFG?.extract_model||''}">
 <label>审校评委</label><input id="c_judge" value="${window.CFG?.judge_model||''}">
 <p><button onclick="saveCfg()">保存</button></p>`;}
async function saveCfg(){await api("/api/config",{draft_model:$("c_draft").value,
 extract_model:$("c_ext").value,judge_model:$("c_judge").value});
 alert("已保存");refresh()}
function drawRead(){/* 章节阅读器由写作台的章节表点击进入 */}
function drawImp(){$("main").innerHTML=`<h2>精读导入 · 读别人的好小说</h2>
 <label>小说 txt 路径</label><input id="i_txt" placeholder="D:\\books\\某畅销书.txt">
 <div class="row"><div><label>来源标签</label><input id="i_label"></div>
 <div><label>最多分析段数</label><input id="i_max" value="10"></div></div>
 <p><button onclick="imp()">📚 开始深度精读（用最强模型）</button></p>
 <div id="log" style="margin-top:12px">${(JOB&&JOB.log||[]).join("\n")}</div>
 <p style="color:var(--dim);font-size:12px">技法将注入当前选中作品；建议先用「设置」把精读模型换成最强的。</p>`;
 poll();}
async function imp(){$("log")&&( $("log").textContent="启动中…");
 const r=await api("/api/import/deepread",{txt_path:$("i_txt").value,
  label:$("i_label").value,max_chunks:+$("i_max").value});
 if(r.error)alert(r.error);poll(true);}
refresh();setInterval(refresh,8000);
</script></body></html>
"""


def main(argv=None) -> int:
    global HOME
    try:
        from memagent.cli import enable_utf8

        enable_utf8()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="memagent 小说工作室")
    ap.add_argument("--home", default=str(HOME))
    ap.add_argument("--port", type=int, default=8600)
    args = ap.parse_args(argv)
    HOME = Path(args.home).resolve()
    HOME.mkdir(parents=True, exist_ok=True)
    _cfg_load()
    if CFG.get("active_work"):
        ACTIVE["path"] = CFG["active_work"]

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"📖 小说工作室：http://127.0.0.1:{args.port}  （home={HOME}）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
