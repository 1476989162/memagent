# -*- coding: utf-8 -*-
"""诊断小说设定密度：统计未解释的专有术语在末 N 章的出现次数。"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent

terms = ['塔纹','淬生气','锈脉','错季','骨契','残蜕','影','裴枕灯','钟无咎',
         '折丹','脊骨','当票','噬账兽','未时','霜意','锁扣','断魂崖','造血深渊',
         '错季城','九川','魇覆','未月','封镇','错季裂隙','骨画','水漏','三魂',
         '命格','当主','错季相','残响','骨片','空壳','错季相的空壳','错季相后',
         '错季相的空壳','错季相','错季之']

chapters = sorted((ROOT / 'works/错季锁星/chapters').glob('第*章.md'))
last10 = chapters[-10:]
nums = [int(re.search(r'第(\d+)章', f.name).group(1)) for f in last10]
print(f'分析最后10章（第{nums[0]}~第{nums[-1]}章）')

explainers = ['是','叫','意为','原来','就是','实为','正是','即为','指','即','即叫','原来叫']
total_unexp = 0
for f in last10:
    txt = f.read_text(encoding='utf-8', errors='replace')
    ch_no = int(re.search(r'第(\d+)章', f.name).group(1))
    hits = [t for t in terms if t in txt]
    unexp = []
    for t in hits:
        found_at = [m.start() for m in re.finditer(re.escape(t), txt)]
        for idx in found_at:
            window = txt[max(0, idx - 30): idx + len(t) + 30]
            if any(e in window for e in explainers):
                break
        else:
            if t not in unexp:
                unexp.append(t)
    total_unexp += len(unexp)
    print(f'第{ch_no:<4}章  术语命中 {len(hits):>3}  未解释 {len(unexp):>3}  {unexp[:8]}')
print(f'\n10 章合计未解释术语引用: {total_unexp}')
print(f'平均每章未解释术语密度: {total_unexp/10:.1f}')