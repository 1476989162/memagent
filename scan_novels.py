#!/usr/bin/env python3
"""逐段扫描 E:\temp\小说 下所有 .txt，提取【非色情段落】做写作技法分析。

策略：不按文件整体分类。从每部小说中抽段落，过滤掉命中成人关键词的段落，
保留纯叙事段落（战斗、环境、对白、伏笔、心理等），作为范文喂给 agent。
"""
import os, re, json, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(r"E:\temp\小说")
OUT_DIR = Path(r"E:\神经网络\reference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADULT_WORDS = [
    "淫", "操", "屌", "逼", "乳", "舔", "吮", "含吹", "发情", "色诱",
    "御女", "后宫", "NTR", "ntr", "掠夺", "挨操", "绿", "无绿", "加料",
    "女优", "度遍", "蹂躏", "玩弄", "侵犯", "强迫", "迷奸", "胁迫",
    "迷昏", "下药", "体液", "湿润", "挺入", "进出",
]

# 写作技法类型关键词 —— 用于给段落打标签
TECHNIQUE_TAGS = {
    "combat":     ["拳", "掌", "刀", "剑", "斩", "刺", "破", "挡", "身法", "灵力", "内息", "劲"],
    "environment":["山", "崖", "雾", "风", "月", "雪", "雨", "殿", "宫", "殿", "街", "夜", "黎明", "暮"],
    "dialogue":   ["道", "说", "问", "答", "喝", "笑", "叹", "冷声道", "淡淡道", "沉声道"],
    "foreshadow": ["伏笔", "暗", "疑", "蹊跷", "不对劲", "似乎", "莫非", "隐隐"],
    "psychology": ["心", "想", "念", "思", "惊", "惧", "怒", "悲", "怅然", "苦笑", "心中暗"],
    "cultivation":["修炼", "境界", "瓶颈", "突破", "顿悟", "灵根", "资质", "功法", "丹田"],
    "worldbuild": ["世界", "大陆", "宗门", "帝国", "古族", "遗迹", "秘境", "传说"],
}

def read_head(path: Path, max_bytes=30000) -> str:
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return path.read_bytes()[:max_bytes].decode(enc, errors="replace")
        except:
            continue
    return ""

def split_paras(text: str) -> list:
    """按段拆分"""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 80]

def is_adult_segment(seg: str) -> bool:
    return any(w in seg for w in ADULT_WORDS)

def tag_segment(seg: str) -> list:
    tags = []
    for tag, kwlist in TECHNIQUE_TAGS.items():
        if any(kw in seg for kw in kwlist):
            tags.append(tag)
    if not tags:
        tags.append("narrative")
    return tags

def pick_best_paragraphs(paras: list, n: int = 5) -> list:
    """挑最长且信息密度最高的非色情段落"""
    clean = [(p, tag_segment(p)) for p in paras if not is_adult_segment(p)]
    # 按长度降序，过滤后取前 n
    clean.sort(key=lambda x: len(x[0]), reverse=True)
    return clean[:n]

print("=" * 60)
print("逐段扫描 + 提取写作技法范文")
print("=" * 60)

all_refs = []
files_scanned = 0
total_paras = 0
kept_paras = 0
skipped_paras = 0

for root, dirs, files in os.walk(ROOT):
    for fname in files:
        if not fname.lower().endswith(".txt"):
            continue
        fpath = Path(root) / fname
        rel = fpath.relative_to(ROOT)
        files_scanned += 1
        text = read_head(fpath, 50000)
        paras = split_paras(text)
        total_paras += len(paras)
        best = pick_best_paragraphs(paras, n=5)
        kept_paras += len(best)
        skipped_paras += len(paras) - len(best)
        for seg, tags in best:
            all_refs.append({
                "source": str(rel),
                "tags": tags,
                "text": seg[:800],
            })

print(f"扫描 {files_scanned} 个文件，{total_paras} 个段落")
print(f"  保留非色情段落 {kept_paras} 个，跳过 {skipped_paras} 个")
print(f"  提取到 {len(all_refs)} 段范文")

# 按技法类型分组输出
by_tag: dict[str, list] = {}
for ref in all_refs:
    for t in ref["tags"]:
        by_tag.setdefault(t, []).append(ref)

print("\n## 各技法类型范文数量")
for tag in ["combat", "environment", "dialogue", "foreshadow", "psychology", "cultivation", "worldbuild", "narrative"]:
    print(f"  {tag}: {len(by_tag.get(tag,[]))} 段")

# 保存全部范文
report = {
    "scan_time": datetime.now().isoformat(),
    "summary": {
        "files_scanned": files_scanned,
        "total_paras": total_paras,
        "kept_paras": kept_paras,
        "all_refs": len(all_refs),
        "by_tag": {t: len(v) for t, v in by_tag.items()},
    },
    "refs": all_refs,
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "clean_paragraphs.json"
out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n范文库 → {out_path}")

# 同时生成可读版：每类抽 3 段做预览
preview_path = OUT_DIR / "preview_by_technique.txt"
with preview_path.open("w", encoding="utf-8") as f:
    f.write("=== 写作技法范文预览（每类 3 段，共 8 类）===\n\n")
    for tag in ["combat", "environment", "dialogue", "foreshadow", "psychology", "cultivation", "worldbuild", "narrative"]:
        segs = by_tag.get(tag, [])[:3]
        f.write(f"\n--- 【{tag}】---\n")
        for ref in segs:
            f.write(f"[来源: {ref['source']}]\n")
            f.write(ref["text"][:400] + "\n\n")
print(f"技法预览 → {preview_path}")