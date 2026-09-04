#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量精读批量任务：E:\\自动小说\\novels\\history -> 修真门派掌门路 技法库

- 过滤色情内容（敏感词密度）
- 分块深度精读（强模型结构化分析：节奏/对话/伏笔/情绪）
- 增量持久化（每本一存，崩溃可续跑）
"""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(r"E:\自动小说\novels\history")
STUDIO_HOME = Path(r"E:\神经网络\novel_studio")
WORK = STUDIO_HOME / "修真门派掌门路"
PERSIST = WORK / "memory.json"
PROGRESS = WORK / "deepread_progress.json"
LOGFILE = WORK / "deepread_batch.log"

BAD_WORDS = ["操", "肏", "插", "射", "高潮", "淫", "轮奸", "强奸", "口交", "肛交", "颜射", "吞精"]
MODEL = "sensenova-6.8-flash-lite"
MAX_CHUNKS_PER_BOOK = 5  # FIX: was 10 -> 9472 bloat

PROMPT = (
    "你是网文写作技法分析专家。下面是一部畅销小说的片段，请做结构化精读，只输出 JSON：\n"
    '{"pacing": ["节奏技巧一句话", ...], "dialogue": ["对话技巧一句话", ...], '
    '"foreshadow": ["伏笔手法一句话(不含具体剧情)", ...], '
    '"emotion": ["情绪调动手法一句话", ...]}\n'
    "每类最多 4 条，必须抽象为可复用的写法规律，禁止复述情节。\n\n片段：\n"
)

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line+"\n")

def is_clean(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    if len(text) < 100000:
        return False
    # 全文敏感词密度
    if sum(text.count(w) for w in BAD_WORDS) > 10:
        return False
    return True

def split_book(text: str, chunk: int = 3200):
    parts = re.split(r"(?=第[0-9一二两三四五六七八九十百千]+章)", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = [text[i:i+chunk] for i in range(0, len(text), chunk)]
    merged, buf = [], ""
    for p in parts:
        if len(buf)+len(p) > chunk and buf:
            merged.append(buf); buf = p
        else:
            buf += "\n"+p
    if buf.strip(): merged.append(buf)
    return merged

def main():
    WORK.mkdir(parents=True, exist_ok=True)
    # 加载进度
    done = set()
    if PROGRESS.is_file():
        try: done = set(json.loads(PROGRESS.read_text(encoding="utf-8")).get("done", []))
        except: pass
    log(f"已完成 {len(done)} 本，待处理...")

    # 扫描
    all_txts = list(ROOT.rglob("*.txt"))
    log(f"扫描到 {len(all_txts)} 个txt")
    clean = []
    for f in all_txts:
        if str(f) in done: continue
        if is_clean(f):
            clean.append(f)
    log(f"过滤后 {len(clean)} 本 待精读（已跳过 {len(done)} 本）")
    # 按大小排序，大部头优先
    clean.sort(key=lambda p: p.stat().st_size, reverse=True)

    from memagent.responder import LLMResponder
    rsp = LLMResponder(model=MODEL, timeout=180, max_tokens=8192)

    total_learned = 0
    for idx, path in enumerate(clean, 1):
        label = path.stem
        log(f"[{idx}/{len(clean)}] 开始《{label}》")
        try:
            raw = path.read_bytes()
            text = None
            for enc in ("utf-8", "gb18030", "gbk"):
                try: text = raw.decode(enc); break
                except: continue
            if not text: 
                log(f"  解码失败，跳过"); continue
            chunks = split_book(text)[:MAX_CHUNKS_PER_BOOK]
            learned_this = 0
            for ci, ck in enumerate(chunks, 1):
                for attempt in (1, 2, 3):
                    try:
                        reply = rsp.respond(PROMPT + ck[:3400])
                        break
                    except Exception as e:
                        msg = str(e)
                        if "429" in msg or "限流" in msg:
                            wait = 60 if attempt==1 else 120
                            log(f"  段{ci} 限流，等待{wait}s 重试 {attempt}/3")
                            time.sleep(wait)
                            continue
                        else:
                            log(f"  段{ci} 失败({attempt}/3): {msg[:80]}")
                            time.sleep(5)
                            continue
                else:
                    log(f"  段{ci} 3次均失败，跳过"); continue
                m = re.search(r"\{.*\}", reply, re.S)
                if not m: 
                    log(f"  段{ci} 未解析出JSON"); continue
                try: data = json.loads(m.group(0))
                except: continue
                # 写入记忆（每本一存，减少并发）
                from memagent.agent import AgentConfig, MemoryAgent
                from memagent.llm import LLMClassifier
                agent = __import__("memagent").MemoryAgent if False else None
                # 延迟导入避免循环，用新实例每段后保存？改为批量每本一实例
                # 为避免重复打开，这里每段后用独立短生命周期agent追加
                from memagent import MemoryAgent as MA
                from memagent.agent import AgentConfig as AC
                from memagent.llm import LLMClassifier as LC
                ag = MA(persist_path=str(PERSIST),
                        cfg=AC(chapter_save_dir=str(WORK/"works"), evolve_on_sleep=False),
                        classifier=LC(api_key=""))
                added = 0
                seen = set(m.content for m in ag.store.all())
                for cat, items in data.items():
                    if not isinstance(items, list): continue
                    for it in items:
                        s = str(it).strip()
                        if len(s) < 8: continue
                        key = f"[{label}/{cat}] {s}"
                        if key in seen: continue
                        if added >= 30: break
                        ag.remember_skill(key, importance=0.55)  # FIX: dedup+0.55 decayable, was 0.75 frozen
                        seen.add(key)
                        added += 1
                    if added >= 30: break
                learned_this += added
                time.sleep(1)
            # FIX: single atomic save per book (was per chunk -> OOM)
            if learned_this:
                for _ in range(3):
                    try: ag.save(); break
                    except Exception as e:
                        if "Concurrent" in type(e).__name__:
                            time.sleep(1)
                            try: ag.store.load()
                            except: pass
                            continue
                        else: raise
            total_learned += learned_this
            log(f"  《{label}》完成，新增技法 {learned_this} 条，累计 {total_learned}")
        except Exception as e:
            log(f"《{label}》异常: {e}")
        # 进度落盘
        done.add(str(path))
        PROGRESS.write_text(json.dumps({"done": sorted(done), "total_learned": total_learned}, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(3)
    log(f"全部完成，共 {len(done)} 本，技法 {total_learned} 条")

if __name__ == "__main__":
    main()
