import os, json, re
from pathlib import Path

root = Path(r"E:\自动小说\novels\history")
txts = list(Path(r"E:\自动小说\novels\history").rglob("*.txt"))
clean = []
bad_words = ["操", "肏", "干", "插", "射", "高潮", "淫", "奸", "轮奸", "强奸", "口交", "肛交", "颜射", "吞精", "舔", "舔", "吸吮", "抽插"]

for f in txts[:500]:
    try:
        text = f.read_text(encoding="utf-8", errors="ignore")
        head = text[:5000]
        bad_count = sum(head.count(w) for w in bad_words)
        if bad_count > 10:
            continue
        if len(text) > 100000 and text.count("\"") + text.count("\"") > 5:
            size = f.stat().st_size
            clean.append((f.name, f.stat().st_size, f))
    except Exception as e:
        pass

print(f"通过初筛: {len(clean)} 本")
for n, sz, f in sorted(clean, key=lambda x: -x[1])[:30]:
    print(f"  {n} ({sz/1024:.0f}KB)")