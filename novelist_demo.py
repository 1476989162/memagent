"""最小示例：验证 ① 小说家人设 + 自主演化（设定记忆注入人设档案） ② 429 多模型自动切换。

本脚本完全离线（注入假 HTTP 客户端），验证两条链路：

一、人设与自主演化（memagent.responder.LLMResponder + MemoryAgent.persona）：
    - MemoryAgent(persona=\"novelist\") 自动创建 LLMResponder，内置小说家人设\n
      （长篇玄幻/仙侠 + 成年角色暧昧情欲张力描写，含边界：自愿、成年、克制淡出）；\n
    - remember_setting() 写入作品设定（kind=\"setting\" 记忆）；\n
    - persona_sheet() 按重要性把设定拼成\"演化档案\"，每次回复注入 system prompt——\n
      人设随设定累积自主演化、跨会话保持一致。

二、429 自动切换（ModelPool：分类器与回复生成器共用）：
    - 主模型 429 → 自动切换到备用模型（failover_cooldown 秒内不再碰它）；\n
    - 全部模型都 429 → 等待最早冷却结束再重试整个池（\"一直切换到不限流的模型\"），\n
      直到预算耗尽或出现可用模型。

真实使用（配 key 后零改动）：
    export OPENAI_BASE_URL=https://api.deepseek.com/v1
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_MODEL=deepseek-chat
    export OPENAI_MODELS=deepseek-chat,deepseek-reasoner,glm-4   # 逗号分隔的备用池
    export OPENAI_PERSONA=novelist                                # 或 MemoryAgent(persona=...)
    python -m memagent   # 启动后 /persona 看档案、/models 看模型池
"""

from __future__ import annotations

import json
import time

from memagent import MemoryAgent
from memagent.responder import LLMResponder

_PASS = 0
_FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def demo_persona() -> None:
    print("=" * 62)
    print("一、小说家人设 + 自主演化（设定记忆 → 人设档案 → 注入每次回复）")
    print("=" * 62)

    captured = {"system": None, "user": None, "model": None}

    class CapPost:
        def __call__(self, url, headers, payload, timeout):
            captured["system"] = payload["messages"][0]["content"]
            captured["user"] = payload["messages"][-1]["content"]
            captured["model"] = payload["model"]
            reply = "（mock）夜色渐深，林尘在丹房外停住脚步，指尖还残留着药炉的余温。"
            return 200, json.dumps({"choices": [{"message": {"content": reply}}]})

    r = LLMResponder(api_key="demo-key", base_url="https://example.com/v1",
                     post=CapPost(), persona="novelist")
    agent = MemoryAgent(responder=r, persona="novelist")

    print("\n① 未写入任何设定时，演化档案为空（不注入）：")
    check("persona_sheet() 返回 None", agent.persona_sheet() is None)

    print("\n② 写入作品设定（remember_setting → kind=setting 记忆）：")
    agent.remember_setting("作品：《青州问剑录》，长篇玄幻，已连载 42 章", importance=0.95)
    agent.remember_setting("主角：林尘，青州林氏旁支少年，身负残剑‘听雪’", importance=0.9)
    agent.remember_setting("境界体系：炼气→筑基→金丹→元婴→化神", importance=0.8)
    agent.remember_setting("伏笔：林尘母亲的身份，第 12 章埋下，计划第 80 章揭开", importance=0.85)
    sheet = agent.persona_sheet()
    check("persona_sheet() 已生成 4 条档案", sheet is not None and len(sheet.splitlines()) == 4,
          f"实际 {len(sheet.splitlines()) if sheet else 0} 条")
    check("档案按重要性降序（作品 > 主角 > 伏笔 > 境界）",
          sheet.index("《青州问剑录》") < sheet.index("境界体系"), sheet)

    print("\n③ 对话回复：演化档案被注入 system prompt（人设随记忆自主演化）：")
    agent.respond("继续写下一章")
    sys_prompt = captured["system"] or ""
    check("system prompt 含小说家人设（夜航墨客）", "夜航墨客" in sys_prompt)
    check("system prompt 含演化档案（作品/主角/境界）",
          "《青州问剑录》" in sys_prompt and "林尘" in sys_prompt and "境界体系" in sys_prompt)
    check("基础助手提示仍在", "记忆增强型对话助手" in sys_prompt)
    print("\n  —— 注入后的 system prompt 片段 ——")
    for line in sys_prompt.splitlines()[:7]:
        print(f"    {line}")
    print(f"    ...")
    print(f"    当前模型: {captured['model']}")


def demo_failover() -> None:
    print("\n" + "=" * 62)
    print("二、429 自动切换：主模型限流 → 备用模型 → 全部限流时等待后重试")
    print("=" * 62)

    print("\n① 主模型 429 → 自动切换备用模型：")
    class PoolPost:
        def __init__(self):
            self.calls = []

        def __call__(self, url, headers, payload, timeout):
            self.calls.append(payload["model"])
            if payload["model"] == "deepseek-chat":
                return 429, "{}"
            return 200, json.dumps({"choices": [{"message": {"content": "（备用模型回答）"}}]})

    post = PoolPost()
    r1 = LLMResponder(api_key="demo-key", base_url="https://example.com/v1",
                      model="deepseek-chat", models=["deepseek-reasoner", "glm-4"],
                      post=post, max_retries=0, failover_cooldown=60)
    txt = r1.respond("写一段雨夜的对话")
    check(f"回复来自备用模型: {txt}", txt == "（备用模型回答）")
    check(f"请求序列 {post.calls}（deepseek-chat 429 → deepseek-reasoner 成功）",
          post.calls == ["deepseek-chat", "deepseek-reasoner"])
    st = r1.pool_status()
    check(f"当前模型已切换为 {st['active']}，429 切换次数 {st['failover_count']}",
          st["active"] == "deepseek-reasoner" and st["failover_count"] == 1,
          f"{st}")
    check("最近限流记录含 deepseek-chat", st["recent_429"] and st["recent_429"][0][0] == "deepseek-chat")

    print("\n② 全部模型都 429 → 等待最早冷却结束再重试（一直切换到不限流的模型）：")
    class AllDownPost:
        def __init__(self):
            self.calls = []

        def __call__(self, url, headers, payload, timeout):
            self.calls.append(payload["model"])
            if len(self.calls) <= 2:  # 第一轮两个模型都 429
                return 429, "{}"
            return 200, json.dumps({"choices": [{"message": {"content": "（冷却后成功）"}}]})

    post2 = AllDownPost()
    r2 = LLMResponder(api_key="demo-key", base_url="https://example.com/v1",
                      model="glm-4", models=["kimi"], post=post2,
                      max_retries=0, failover_cooldown=0.1,
                      all_down_retries=3, all_down_wait_cap=0.2)
    t0 = time.time()
    txt2 = r2.respond("再来一段")
    waited = time.time() - t0
    check(f"回复成功: {txt2}", txt2 == "（冷却后成功）")
    check(f"请求序列 {post2.calls}（glm-4、kimi 全 429 → 等冷却 → glm-4 成功）",
          post2.calls == ["glm-4", "kimi", "glm-4"], f"{post2.calls}")
    check(f"确实等待了冷却（{waited:.2f}s ≥ 0.08s）", waited >= 0.08)
    check("全部限流期间的 429 计数 = 2", r2.failover_count == 2, f"{r2.failover_count}")

    print("\n③ 预算耗尽仍无限流恢复 → 明确报错（而不是静默失败）：")
    class Always429:
        def __call__(self, url, headers, payload, timeout):
            return 429, "{}"

    r3 = LLMResponder(api_key="demo-key", base_url="https://example.com/v1",
                      model="a", models=["b"], post=Always429(),
                      max_retries=0, failover_cooldown=100,
                      all_down_retries=1, all_down_wait_cap=0.01)
    try:
        r3.respond("你好")
        check("应抛 RuntimeError", False)
    except RuntimeError as e:
        check(f"报错提示清晰: {e}", "限流" in str(e) and "冷却" in str(e), str(e))


def main() -> int:
    from memagent.cli import enable_utf8

    enable_utf8()
    demo_persona()
    demo_failover()

    print(f"\n结果：{_PASS} 通过 / {_FAIL} 失败")
    if _FAIL == 0:
        print("人设演化 + 429 自动切换链路验证通过 ✅")
        print("\n真实使用：配 OPENAI_API_KEY/OPENAI_MODELS/OPENAI_PERSONA 后\n"
              "    python -m memagent   （/persona 查看档案，/models 查看模型池）")
    else:
        print("存在失败项 ❌")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
