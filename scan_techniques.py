#!/usr/bin/env python3
"""
从 E:\temp\小说 提取写作技法模式 → 沉淀到 agent 技能记忆。

安全策略：
  - 只读非色情段落做分析
  - LLM 只收到"技法模式"（如"感官-动作-心理三层递进"），不读原文
  - 输出只打印技法规则（干净），不打印任何原文
"""
import os, re, json, sys, traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, r"E:\神经网络")

ROOT = Path(r"E:\temp\小说")
OUT_DIR = Path(r"E:\神经网络\reference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADULT_WORDS = [
    "淫", "操", "屌", "逼", "乳", "舔", "吮", "含吹", "发情", "色诱",
    "御女", "后宫", "NTR", "ntr", "掠夺", "挨操", "绿", "无绿", "加料",
    "女优", "度遍", "蹂躏", "玩弄", "侵犯", "强迫", "迷奸", "胁迫",
    "迷昏", "下药", "体液", "湿润", "挺入", "进出", "穴",
]

# ---------- 1. 切段落 ----------
def read_head(path: Path, max_bytes=50000) -> str:
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return path.read_bytes()[:max_bytes].decode(enc, errors="replace")
        except:
            continue
    return ""

def split_paras(text: str) -> list:
    """切段落：先按连续空行，再按章节标题标记，再按句号+换行"""
    # 按章节标题切
    parts = re.split(r"(?:第\s*\d+\s*章|第[一二三四五六七八九十百]+章|卷[一二三四五六七八九十])\s*", text)
    all_paras = []
    for p in parts:
        # 按连续空行切
        subs = [s.strip() for s in re.split(r"\s*\n\s*\n\s*", p) if s.strip()]
        all_paras.extend(subs)
    return all_paras

# ---------- 2. 过滤 ----------
def is_adult_segment(seg: str) -> bool:
    return any(w in seg for w in ADULT_WORDS)

# ---------- 3. 技法特征提取（启发式，不读原文给 LLM） ----------
def extract_technique_features(seg: str) -> dict:
    """从段落结构中提取技法特征——只产抽象模式，不产原文。"""
    features = {}

    # a) 句式节奏：短长比
    sentences = [s for s in re.split(r"[。！？]", seg) if len(s.strip()) > 3]
    if sentences:
        lens = [len(s.strip()) for s in sentences]
        short = sum(1 for l in lens if l < 12)
        features["sentence_short_ratio"] = round(short / len(lens), 2)
        features["sentence_avg_len"] = round(sum(lens) / len(lens), 1)

    # b) 动作动词密度
    action_verbs = len(re.findall(r"[突跃闪避刺劈挡斩劈冲腾滚扑]", seg))
    features["action_density"] = round(action_verbs / max(len(seg), 1) * 1000, 1)

    # c) 感官描写类型
    senses = {"visual": len(re.findall(r"[望瞥睨睨瞥视看]", seg)),
              "auditory": len(re.findall(r"[听闻声听喝]", seg)),
              "tactile": len(re.findall(r"[触摸握捏抓抚]", seg)),
              "olfactory": len(re.findall(r"[嗅闻气息味]", seg)),
              "thermal": len(re.findall(r"[冷温热灼寒凉寒]", seg))}
    features["senses"] = senses
    features["dominant_sense"] = max(senses, key=senses.get)

    # d) 对话结构
    dialog_count = seg.count("「") + seg.count("『") + seg.count('"') + seg.count('"')
    features["dialog_ratio"] = round(dialog_count / max(len(seg), 1) * 1000, 1)

    # e) 心理描写信号
    psycho_signals = len(re.findall(r"[心想暗想心念念头心生|觉得以为感到]", seg))
    features["psychology_density"] = round(psycho_signals / max(len(seg), 1) * 1000, 1)

    # f) 环境/景物描写信号
    env_signals = len(re.findall(r"[风月云雾雨雪山川殿阁亭桥]", seg))
    features["environment_density"] = round(env_signals / max(len(seg), 1) * 1000, 1)

    # g) 修辞手法
    features["metaphor_signals"] = len(re.findall(r"[如似若恍恰恰比像]", seg))
    features["parallelism_signals"] = len(re.findall(r"[既.{1,4}又|不但.{1,6}而且|不仅.{1,6}还]", seg))

    # h) 伏笔/悬念信号
    features["foreshadow_signals"] = len(re.findall(r"[莫非或许似乎蹊跷隐隐疑]", seg))

    # i) 叙事视角
    first_person = len(re.findall(r"[我自吾]", seg))
    third_person = len(re.findall(r"[他她它]", seg))
    features["perspective"] = "first_person" if first_person > third_person * 3 else "third_person"

    # j) 段落结构：先外后内 / 先内后外 / 混合
    # 简化：动作动词出现位置 vs 心理信号位置
    first_action_pos = float('inf')
    first_psy_pos = float('inf')
    for m in re.finditer(r"[突跃闪避刺劈挡斩劈冲腾滚扑]", seg):
        first_action_pos = min(first_action_pos, m.start())
    for m in re.finditer(r"[心想暗想心念念头心生]", seg):
        first_psy_pos = min(first_psy_pos, m.start())
    if first_action_pos < first_psy_pos:
        features["structure"] = "action_first_then_psycho"
    elif first_psy_pos < first_action_pos:
        features["structure"] = "psycho_first_then_action"
    else:
        features["structure"] = "mixed"

    features["length"] = len(seg)
    return features

# ---------- 4. 聚合 ----------
print("=" * 60)
print("写作技法扫描（仅输出技法规则，不展示原文）")
print("=" * 60)

all_features = []
files_scanned = 0
clean_count = 0

for root, dirs, files in os.walk(ROOT):
    for fname in files:
        if not fname.lower().endswith(".txt"):
            continue
        fpath = Path(root) / fname
        rel = str(fpath.relative_to(ROOT))
        files_scanned += 1
        text = read_head(fpath, 80000)
        paras = split_paras(text)
        for p in paras[:30]:  # 每文件最多 30 段
            if not p or len(p) < 80:
                continue
            if is_adult_segment(p):
                continue
            clean_count += 1
            feats = extract_technique_features(p)
            feats["source"] = rel
            all_features.append(feats)

print(f"扫描 {files_scanned} 个文件，提取 {clean_count} 段干净段落特征")

# 聚合特征分布
import statistics as st
def safe_stats(key):
    vals = [f[key] for f in all_features if key in f]
    if not vals:
        return {}
    return {"mean": round(st.mean(vals), 2),
            "median": round(st.median(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2)}

dist = {
    "sentence_short_ratio": safe_stats("sentence_short_ratio"),
    "sentence_avg_len": safe_stats("sentence_avg_len"),
    "action_density": safe_stats("action_density"),
    "dialog_ratio": safe_stats("dialog_ratio"),
    "psychology_density": safe_stats("psychology_density"),
    "environment_density": safe_stats("environment_density"),
    "metaphor_signals": safe_stats("metaphor_signals"),
    "structure_dist": {},
    "perspective_dist": {},
}

for f in all_features:
    s = f.get("structure", "unknown")
    dist["structure_dist"][s] = dist["structure_dist"].get(s, 0) + 1
    p = f.get("perspective", "unknown")
    dist["perspective_dist"][p] = dist["perspective_dist"].get(p, 0) + 1

print("\n## 技法特征分布")
print(json.dumps(dist, ensure_ascii=False, indent=2))

# 保存特征分布
report = {
    "scan_time": datetime.now().isoformat(),
    "files_scanned": files_scanned,
    "clean_paras": clean_count,
    "distribution": dist,
}
out_path = OUT_DIR / "technique_distribution.json"
out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n特征分布 → {out_path}")