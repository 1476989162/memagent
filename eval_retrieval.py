# -*- coding: utf-8 -*-
"""检索质量基准：标注数据集上测 Recall@k / MRR / P@1。

测的是什么：给定一条已知记忆和一个**换了说法**的查询，检索排序能否把
正例排到前面——衡量"语义匹配 + 同义扩展 + 子串重排"这条链路的综合质量。

设计要点：
- 所有记忆统一 importance=0.5 / SEMANTIC / 同一时刻写入（注入时钟），
  强度项完全同源 → 排序差异纯粹来自相关性，隔离测"找得准不准"；
- 关闭再巩固与 Hot 升级，避免评估过程本身污染被评估对象；
- 干扰项分两层：每个用例自带的近义干扰 + 其他用例的正例（跨题干扰，
  更接近真实记忆库）；
- 数据集全部完整句子——哈希 n-gram 嵌入对措辞敏感，短碎片会产生碰撞噪声
  （见 AGENTS.md 教训）。

用法：
  python eval_retrieval.py                    # 默认配置跑分
  python eval_retrieval.py --ablation         # 附带关闭同义扩展/子串重排的对照
  python eval_retrieval.py --min-r3 0.60     # CI 门禁：Overall Recall@3 低于阈值退出码 1

读数须知：
- 默认 HashEmbedder 只认字面 n-gram 重叠——paraphrase/synonym/temporal 三类
  分数低是**真实能力边界**（换措辞后连 bigram 都不共享，rel≈0），不是评测
  缺陷；换语义嵌入后端（memagent/embedders.py，sentence-transformers 或
  OpenAI 兼容远程）预期显著抬升这三类，本基准即验证工具。
- 实测基线（2026-08，默认配置）：Overall R@1/R@3/R@5/MRR ≈ 0.20/0.34/0.46/0.29；
  近义区分最强（R@3=0.90）。
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MEMAGENT_TEST", "1")  # 不加载真实 .env，离线确定性

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.memory import MemType

FIXED_NOW = 1_700_000_000.0
K_FINAL = 20          # 取这么深只为算 MRR 尾部，报告仍报 @1/@3/@5
CAT_LABELS = {
    "paraphrase": "同义改写",
    "pronoun": "人称互换",
    "synonym": "书面↔口语",
    "temporal": "时间线变化",
    "short": "短查询",
    "discriminate": "近义区分",
}

# ---------- 标注数据集：50 例 ----------
# 每例：q=查询（刻意换措辞）；pos=应命中的记忆；neg=近义干扰项
CASES: list[dict] = [
    # ---- paraphrase 同义改写（12）----
    dict(cat="paraphrase", q="请问我的宠物猫的基本情况？",
         pos="我家猫咪团子今年三岁，是个胆小的橘猫。",
         neg=["小区门口的宠物店每周一洗澡打八折。", "橘猫是比较常见的家猫毛色。",
              "我邻居也养了一只三岁的博美犬。"]),
    dict(cat="paraphrase", q="年会的时间和地点定好了吗？",
         pos="公司年会定在下周五晚上七点，地点在蓝海酒店三楼宴会厅。",
         neg=["部门聚餐上次选的是川菜馆。", "蓝海酒店的停车场每小时收费十元。",
              "下周五下午有个项目评审会。"]),
    dict(cat="paraphrase", q="饮食方面有什么需要注意的忌口？",
         pos="我对花生过敏，吃点心前要看清楚配料表。",
         neg=["最近早餐常吃燕麦粥和鸡蛋。", "坚果类零食我一般放在办公室抽屉里。",
              "配料表上的添加剂种类越来越多。"]),
    dict(cat="paraphrase", q="能说说我家孩子的情况吗？",
         pos="我的女儿今年上小学二年级，最喜欢画画。",
         neg=["我周末会陪家人去美术馆看展览。", "小学门口放学时段交通很拥堵。",
              "画画班的学费每学期三千元。"]),
    dict(cat="paraphrase", q="工作上最近忙什么呢？",
         pos="我负责的项目下个月底要上线，现在天天加班赶进度。",
         neg=["上线仪式只邀请了核心成员参加。", "进度管理用的是看板工具。",
              "天天加班让我的颈椎有点不舒服。"]),
    dict(cat="paraphrase", q="有什么假期的出行安排？",
         pos="我们打算国庆假期去云南自驾，行程大概八天。",
         neg=["云南的紫外线很强要带防晒霜。", "自驾前记得给车做一次保养。",
              "八天的路线包括大理和丽江。"]),
    dict(cat="paraphrase", q="家里老人的健康状况如何？",
         pos="我爷爷有高血压，每天早上都要量血压。",
         neg=["血压计是去年双十一买的电子款。", "早上空腹吃药效果最好。",
              "高血压患者要少盐少油。"]),
    dict(cat="paraphrase", q="你的业余爱好进展怎样？",
         pos="我正在学吉他，目前会弹三首简单的曲子。",
         neg=["吉他是朋友送我的生日礼物。", "简单的和弦转换还需要多练。",
              "曲谱主要在网上找的简谱。"]),
    dict(cat="paraphrase", q="平时住在哪里，周末呢？",
         pos="我家的老房子在郊区，周末才回去住两天。",
         neg=["郊区的空气比市区好很多。", "房子的屋顶去年翻修过一次。",
              "周末高速容易堵车。"]),
    dict(cat="paraphrase", q="讲讲你家里的水族宠物？",
         pos="我养了一缸热带鱼，最漂亮的是那条红色的斗鱼。",
         neg=["鱼缸的过滤器每个月清洗一次。", "斗鱼好斗不能和其他鱼混养。",
              "红色在灯光下特别显眼。"]),
    dict(cat="paraphrase", q="家里近期有什么值得准备的事？",
         pos="我妈的六十岁生日快到了，我想给她办一场家宴。",
         neg=["家宴订在家里附近的私房菜馆。", "六十大寿传统上要吃长寿面。",
              "快递的蛋糕模具明天到货。"]),
    dict(cat="paraphrase", q="你睡前的日常是什么样的？",
         pos="我习惯每天睡前读半小时书，最近在看历史小说。",
         neg=["历史小说的人物关系都很复杂。", "半小时的阅读能帮助入睡。",
              "睡前玩手机反而更精神。"]),

    # ---- pronoun 人称互换（8）：问句用「您」指用户自己 ----
    dict(cat="pronoun", q="您叫什么名字？从事什么职业？",
         pos="我叫王小雅，在南方的城市做护士。",
         neg=["护士站的排班表每月更新。", "南方的冬天潮湿阴冷。",
              "名字只是一个人的代号。"]),
    dict(cat="pronoun", q="您现在的住所情况方便说说吗？",
         pos="我住在城东的老小区，已经住了十年。",
         neg=["老小区正在加装电梯。", "十年前这一带还很荒凉。",
              "住所附近新开了一家超市。"]),
    dict(cat="pronoun", q="您家孩子的学业近况如何？",
         pos="我的孩子明年参加中考，成绩中等偏上。",
         neg=["中考的体育分数占比提高了。", "成绩单下周发给家长。",
              "偏上的名次能上个不错的高中。"]),
    dict(cat="pronoun", q="您的身体有什么旧疾要注意的吗？",
         pos="我的胃不太好，医生嘱咐少吃生冷食物。",
         neg=["生冷的食物包括刺身和冰饮。", "医生的门诊号很难挂。",
              "胃镜检查需要提前预约。"]),
    dict(cat="pronoun", q="您上下班路上要花多长时间？",
         pos="我每天通勤单程要一个半小时，地铁转公交。",
         neg=["地铁早高峰非常拥挤。", "公交接驳站就在小区门口。",
              "一个半小时能听完两集播客。"]),
    dict(cat="pronoun", q="冒昧问下您的出生季节和喜好？",
         pos="我的生日在冬天十二月，最喜欢收围巾当礼物。",
         neg=["十二月份有圣诞节的氛围。", "围巾还是羊毛的最保暖。",
              "礼物的包装我喜欢简约风。"]),
    dict(cat="pronoun", q="您在饭局上喝酒吗？",
         pos="我不会喝酒，聚会时都以茶代酒。",
         neg=["茶文化在中国历史悠久。", "聚会一般选在周五晚上。",
              "敬茶的礼节其实也有讲究。"]),
    dict(cat="pronoun", q="您家里有没有养什么小动物？",
         pos="我养了两只乌龟，养了快八年了。",
         neg=["乌龟冬眠的时候不要喂食。", "八年时间说长也不长。",
              "小动物医院开在城南。"]),

    # ---- synonym 书面↔口语（8）：存书面、问口语 ----
    dict(cat="synonym", q="你每天喝多少水？",
         pos="我每日均会饮用两升白开水。",
         neg=["白开水要烧开晾温再喝。", "两升的水壶刚好装满一杯壶。",
              "运动之后要补充电解质。"]),
    dict(cat="synonym", q="你家住哪儿？",
         pos="我的寓所位于学校旁边。",
         neg=["学校旁边的房租比较贵。", "寓所的物业费按季度交。",
              "位于路口的那家书店关门了。"]),
    dict(cat="synonym", q="那件事情处理好了吗？",
         pos="此事已妥善处置完毕。",
         neg=["处置流程要走三个审批。", "完毕之后请及时归档。",
              "此事的起因是一场误会。"]),
    dict(cat="synonym", q="你买新电器啦？",
         pos="我购置了一台新的洗衣机。",
         neg=["洗衣机的脱水功能有些噪音。", "新款比旧款更省水。",
              "家电以旧换新有补贴。"]),
    dict(cat="synonym", q="你爸爸最近身体咋样？",
         pos="父亲近日身体欠佳，仍在住院观察。",
         neg=["住院部的探视时间是下午。", "观察两天就可以出院。",
              "近日的气温波动比较大。"]),
    dict(cat="synonym", q="你是不是要出趟远门？",
         pos="我拟于下月赴京出差一周。",
         neg=["出差的差旅标准提高了。", "一周的会议安排得很满。",
              "北京下个月有个大型展会。"]),
    dict(cat="synonym", q="你的电脑是不是很卡？",
         pos="这台旧电脑运行迟缓，时常卡顿。",
         neg=["卡顿多半是硬盘老化了。", "电脑清灰之后会好一些。",
              "运行大型软件时尤其明显。"]),
    dict(cat="synonym", q="最近工资涨了吗？",
         pos="我的薪酬上月有所上调。",
         neg=["上个月的绩效评了良好。", "上调幅度大约百分之五。",
              "工资条在系统里可以自查。"]),

    # ---- temporal 时间线变化（6）：问「最近/上次」指向带时间戳的记忆 ----
    dict(cat="temporal", q="我最近发型有变化吗？",
         pos="昨天下午我去理发店剪了头发，短了很多。",
         neg=["理发店的会员卡充五百送五十。", "头发短了显得精神。",
              "下午的预约不用等位。"]),
    dict(cat="temporal", q="我参加过什么跑步比赛？",
         pos="上个月我完成了半程马拉松，成绩两小时十分。",
         neg=["半马的补给站每隔五公里一个。", "十分的提升空间还有很大。",
              "比赛那天的天气很凉爽。"]),
    dict(cat="temporal", q="我明天有什么安排来着？",
         pos="明天上午九点要去机场接多年未见的老同学。",
         neg=["机场大巴的首班是六点半。", "老同学在南方定居多年。",
              "九点之前的高速不堵车。"]),
    dict(cat="temporal", q="关于家庭聚会有什么约定？",
         pos="上周家庭聚餐时定了规矩：每月第一个周日全员回家吃饭。",
         neg=["周日的家常菜由大家轮流掌勺。", "全员到齐拍了合照发家族群。",
              "聚餐的地点大多选在老家。"]),
    dict(cat="temporal", q="我会滑雪吗？是什么时候学的？",
         pos="去年冬天我学会了滑雪，摔了不少跟头。",
         neg=["滑雪场的初级道人最多。", "摔倒时要护住手腕和膝盖。",
              "冬天的雪票提前买更便宜。"]),
    dict(cat="temporal", q="我最近的戒烟进展怎么样？",
         pos="这个月初我把烟戒了，到现在一根没抽。",
         neg=["戒烟糖含片随身带着备用。", "一根烟都不碰才是最有效的。",
              "月初下的决心总是最大的。"]),

    # ---- short 短查询（6）：靠子串优先重排消除泛化命中噪声 ----
    dict(cat="short", q="手机号",
         pos="我的手机号码是13800138000，麻烦存一下。",
         neg=["手机的屏幕之前碎过一次。", "号码归属地显示是本地。",
              "通讯录记得定期做个备份。"]),
    dict(cat="short", q="办公地址",
         pos="办公室在科技园B座1204室。",
         neg=["办公楼的电梯要刷工牌。", "科技园的食堂中午人最多。",
              "会议室改成了临时工位区。"]),
    dict(cat="short", q="车型",
         pos="我的车是白色的丰田卡罗拉轿车。",
         neg=["白色的车漆其实很耐脏。", "丰田的常规保养不算贵。",
              "油耗低是这款车的主打卖点。"]),
    dict(cat="short", q="邮箱",
         pos="我的邮箱是xiaolin@example.com，有事发邮件。",
         neg=["邮箱的签名档还没设置。", "邮件尽量在当天内回复。",
              "示例域名的地址仅用于演示。"]),
    dict(cat="short", q="星座",
         pos="我属虎，狮子座，七月末过生日。",
         neg=["属相要按农历年份来算。", "性格测试说我偏外向。",
              "生日想办个户外派对。"]),
    dict(cat="short", q="wifi密码",
         pos="家里wifi的密码是home8888。",
         neg=["wifi路由器放在客厅电视柜。", "密码应该定期更换。",
              "客厅的信号穿墙后会变弱。"]),

    # ---- discriminate 近义区分（10）：两条相近记忆，问题指向其中一条 ----
    dict(cat="discriminate", q="我去重庆那次吃的火锅是什么样子的？",
         pos="我去重庆旅游时吃了九宫格火锅。",
         neg=["我去成都出差时吃了地道的火锅。", "火锅的蘸料麻酱最经典。",
              "两座城市之间高铁只要两小时。"]),
    dict(cat="discriminate", q="我妹妹在哪个城市工作来着？",
         pos="我妹妹在深圳做产品经理。",
         neg=["我弟弟在上海做程序员。", "产品经理的岗位竞争很激烈。",
              "兄妹俩一年也就见一次面。"]),
    dict(cat="discriminate", q="我家那只黑狗叫什么名字？",
         pos="我家的狗叫煤球，一身黑毛。",
         neg=["我家的猫叫雪球，浑身雪白。", "黑色的狗毛发不耐脏。",
              "宠物的名字都是照着颜色起的。"]),
    dict(cat="discriminate", q="信用卡的尾号是多少？",
         pos="我的信用卡尾号是1024。",
         neg=["我的储蓄卡尾号是6688。", "储蓄卡就是发工资的那张卡。",
              "信用卡的账单日在每月五号。"]),
    dict(cat="discriminate", q="我周四晚上有什么安排？",
         pos="周四晚上我要加班写方案。",
         neg=["周三晚上我有瑜伽课。", "方案的初稿周五要交。",
              "晚上七点开始上课。"]),
    dict(cat="discriminate", q="我爸擅长做什么菜？",
         pos="我爸的拿手菜是清蒸鲈鱼。",
         neg=["我妈的拿手菜是红烧肉。", "清蒸的做法最讲究火候。",
              "家常菜里数红烧的最下饭。"]),
    dict(cat="discriminate", q="我的工资卡是哪家银行？",
         pos="我的工资卡开在工商银行。",
         neg=["我的公积金缴存在建设银行。", "银行的手机客户端要实名认证。",
              "缴存的比例是百分之十二。"]),
    dict(cat="discriminate", q="我老家在江苏哪个城市？",
         pos="我的老家是江苏南通。",
         neg=["我大学就读于江苏无锡。", "无锡的酱排骨很有名气。",
              "南通的基础教育很出名。"]),
    dict(cat="discriminate", q="我说过自己爱吃哪种水果吗？",
         pos="我特别爱吃荔枝，一次能吃一斤。",
         neg=["我对芒果过敏，吃了嘴巴发麻。", "荔枝吃多了容易上火。",
              "过敏体质最好查一下过敏原。"]),
    dict(cat="discriminate", q="我周六的预订是几位？",
         pos="我预订了周六中午的餐厅，四个人。",
         neg=["我订了周日早上的羽毛球场地。", "中午的包间设最低消费。",
              "羽毛球的场地要提前一周抢。"]),
]

assert len(CASES) == 50, f"数据集应为 50 例，当前 {len(CASES)}"


# ---------- 构建 / 评测 ----------

def build_agent(**cfg_overrides) -> tuple[MemoryAgent, dict[str, str]]:
    """构建评测库：全部正例 + 干扰项入库，返回 (agent, 内容→id 映射)。

    强度完全同源（importance/mtype/时刻一致）、关闭再巩固与 Hot 升级，
    让排序差异只来自相关性。
    """
    cfg = AgentConfig(reconsolidate=False, hot_after_access=999, **cfg_overrides)
    agent = MemoryAgent(cfg=cfg, now_fn=lambda: FIXED_NOW)
    ids: dict[str, str] = {}

    def put(content: str) -> str:
        if content not in ids:
            m = agent.store.add(content, importance=0.5,
                                mtype=MemType.SEMANTIC, now=FIXED_NOW)
            ids[content] = m.id
        return ids[content]

    for c in CASES:
        put(c["pos"])
        for n in c["neg"]:
            put(n)
    return agent, ids


def evaluate(agent: MemoryAgent, ids: dict[str, str],
             cases: list[dict] | None = None) -> dict:
    """逐例取排名 → 汇总 Overall 与分类别的 Recall@1/3/5 和 MRR。"""
    cases = cases or CASES
    rows = []
    for c in cases:
        hits = agent.retrieve(c["q"], k=K_FINAL)
        rank = next((i + 1 for i, h in enumerate(hits)
                     if h.memory.id == ids[c["pos"]]), None)
        rows.append({"q": c["q"], "cat": c["cat"], "rank": rank,
                     "rr": 0.0 if rank is None else 1.0 / rank})
    return _aggregate(rows)


def _aggregate(rows: list[dict]) -> dict:
    def met(rws: list[dict]) -> dict:
        n = len(rws)
        ranks = [r["rank"] for r in rws]
        return {
            "n": n,
            "r1": sum(1 for r in ranks if r == 1) / n,
            "r3": sum(1 for r in ranks if r and r <= 3) / n,
            "r5": sum(1 for r in ranks if r and r <= 5) / n,
            "mrr": sum(r["rr"] for r in rws) / n,
        }

    out = {"overall": met(rows),
           "by_cat": {cat: met([r for r in rows if r["cat"] == cat])
                      for cat in CAT_LABELS},
           "misses": [(r["cat"], r["rank"]) for r in rows if r["rank"] is None],
           "rows": rows}
    return out


def _print_report(name: str, rep: dict) -> float:
    o = rep["overall"]
    print(f"\n【{name}】 {o['n']} 例")
    print(f"  {'类别':<8}{'n':>4}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}")
    for cat, label in CAT_LABELS.items():
        m = rep["by_cat"][cat]
        print(f"  {label:<8}{m['n']:>4}{m['r1']:>8.2f}{m['r3']:>8.2f}"
              f"{m['r5']:>8.2f}{m['mrr']:>8.2f}")
    print(f"  {'Overall':<8}{o['n']:>4}{o['r1']:>8.2f}{o['r3']:>8.2f}"
          f"{o['r5']:>8.2f}{o['mrr']:>8.2f}")
    misses = [f"{CAT_LABELS[c]}(rank={r})" for c, r in rep["misses"] if r is None]
    if misses:
        print(f"  未进前{K_FINAL}：{'，'.join(misses)}")
    print()
    return o["r3"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="memagent 检索质量基准")
    ap.add_argument("--ablation", action="store_true",
                    help="附带关闭同义扩展 / 子串重排的对照")
    ap.add_argument("--min-r3", type=float, default=None,
                    help="CI 门禁：Overall Recall@3 低于该值则退出码 1")
    args = ap.parse_args(argv)

    variants = [("默认配置", {})]
    if args.ablation:
        variants += [("关同义扩展", {"query_expansion": False}),
                     ("关子串重排", {"rerank_short_query": False})]

    scores = []
    for name, over in variants:
        agent, ids = build_agent(**over)
        scores.append((name, _print_report(name, evaluate(agent, ids))))

    rc = 0
    if args.min_r3 is not None:
        _, r3 = scores[0]
        if r3 < args.min_r3:
            print(f"[FAIL] Overall Recall@3={r3:.2f} < 门禁 {args.min_r3:.2f}")
            rc = 1
        else:
            print(f"[PASS] Overall Recall@3={r3:.2f} ≥ 门禁 {args.min_r3:.2f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
