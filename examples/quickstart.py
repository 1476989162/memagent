"""把 memagent 当作 SDK 嵌入你的应用 —— 5 分钟上手。

三个方法走天下：
    remember()  写入记忆（自动分类/去重/情绪编码）
    retrieve()  检索记忆（遗忘曲线评分 + 情境加成）
    sleep()     睡眠巩固（回放 + 分级 + 压缩）

运行：python examples/quickstart.py
"""
import time

from memagent import MemoryAgent
from memagent.llm import LLMClassifier


def main() -> None:
    # api_key="" → 离线关键词分类（不调 LLM、零网络依赖）。
    # 生产环境配 OPENAI_* 环境变量即可自动启用 LLM 分类。
    agent = MemoryAgent(classifier=LLMClassifier(api_key=""))

    # 1) 写入：自动识别类型（技能慢忘/语义中忘/情景快忘），自动推断情绪
    agent.remember("用户的时区是 UTC+8，工作日早9晚6", importance=0.8)
    agent.remember("用户偏好简洁回复，不要废话", importance=0.9)   # 高重要性 → 冻结为核心记忆
    agent.remember("上周三聊过他家的猫叫煤球", importance=0.4)

    # 2) 检索：相似度 × 记忆强度排序，越常查越牢（测试效应）
    hits = agent.retrieve("简洁回复 不要废话")
    top = hits[0] if hits else None
    assert top is not None and "简洁" in top.memory.content
    print(f"检索命中: {top.memory.content}  (得分 {top.total:.2f})")

    # 3) 睡眠巩固：回放近期经历 → 低频旧记忆压缩成摘要索引
    report = agent.sleep()
    print(f"睡眠: 回放 {report['replayed_count']} 条, "
          f"分级 高{report['triage_high']}/中{report['triage_medium']}/低{report['triage_low']}")

    # 4) 元认知：系统知道自己记性如何，过度自信会被自动校准
    conf = agent.metacognition.adjusted_confidence(0.9)
    print(f"原始确信度 0.90 → 校准后 {conf:.2f}")

    # 5) 前瞻记忆：记住将来要做的事，到点主动提醒
    agent.prospective.add_task("提醒用户复查项目进度", "event", "站会")
    due = agent.prospective.get_due_tasks(current_activity="明天早上的站会")
    print(f"前瞻触发: {[t.description for t in due]}")

    print("\nOK —— 就这三个方法：remember / retrieve / sleep")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"(耗时 {time.perf_counter() - start:.2f}s)")
