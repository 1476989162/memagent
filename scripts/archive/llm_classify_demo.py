"""最小示例：验证 memagent 的 LLM 分类与回复生成全链路。

memagent 的 LLM 组件（LLMClassifier 类型分类器 + LLMResponder 回复生成器）
兼容任意 OpenAI 风格端点（OpenAI / DeepSeek / Moonshot / Ollama / vLLM ...），
通过三个环境变量配置：

    OPENAI_BASE_URL   端点地址，如 https://api.deepseek.com/v1（默认 api.openai.com/v1）
    OPENAI_API_KEY    密钥
    OPENAI_MODEL      模型名（默认 gpt-4o-mini）

运行本脚本（两种模式自动选择）：

    python llm_classify_demo.py

    1) 已设置 OPENAI_API_KEY → 直接调用真实端点；
    2) 未设置 → 自动启动本地 mock OpenAI 服务（mock_openai_server.py，
       本仓库自带，端口 8765），走**真实 HTTP 传输链路**验证：
       分类（agent → LLMClassifier → urllib POST → mock → 类型/置信度）、
       结果缓存（同内容只调一次）、回复生成（检索结果注入上下文 / 无记忆
       直接回答 / 无 key 回退模板）与关键词离线回退。

本脚本演示三种等价配置方式：
    ① 环境变量（推荐，CLI `python -m memagent` 也用这套）
    ② LLMClassifier / LLMResponder 构造参数
    ③ MemoryAgent(classifier=.../responder=...) 注入
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK_PORT = 8765
MOCK_BASE = f"http://127.0.0.1:{MOCK_PORT}/v1"

# 三个覆盖 skill / semantic / episodic 的样例（mock server 按关键词返回固定结果）
SAMPLES = [
    ("我在练习做饭", "skill", 0.93),
    ("北京是中国的首都", "semantic", 0.72),
    ("昨天去吃了火锅", "episodic", 0.91),
]

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


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_port(port: int, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.1)
    return False


def main() -> int:
    from memagent.cli import enable_utf8

    enable_utf8()

    mock_proc: subprocess.Popen | None = None
    if os.environ.get("OPENAI_API_KEY"):
        mode = f"真实端点 {os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'}（{os.environ.get('OPENAI_MODEL') or 'gpt-4o-mini'}）"
    else:
        # 方式 ①：用环境变量配置（进程内设置；真实使用直接 export 即可）
        if not _port_open(MOCK_PORT):
            server = os.path.join(HERE, "mock_openai_server.py")
            mock_proc = subprocess.Popen([sys.executable, server])
            if not _wait_port(MOCK_PORT):
                print("[FAIL] mock OpenAI 服务启动超时")
                return 1
        os.environ["OPENAI_BASE_URL"] = MOCK_BASE
        os.environ["OPENAI_API_KEY"] = "mock-key"
        os.environ["OPENAI_MODEL"] = "mock-gpt"
        mode = f"本地 mock OpenAI 服务（{MOCK_BASE}，真实 HTTP 传输链路）"

    print(f"=== LLM 类型分类链路验证 ===\n模式：{mode}\n")

    try:
        # ---------- 步骤 1：LLMClassifier 读环境变量（方式 ①）----------
        print("--- 1) 分类三类记忆（来源应为 llm）---")
        from memagent.llm import LLMClassifier
        from memagent.memory import MemType

        clf = LLMClassifier()  # 无参构造 = 读 OPENAI_* 环境变量
        check("分类器可用（有 key）", clf.available)
        for content, want, conf in SAMPLES:
            mt, got_conf, src = clf.classify(content)
            check(
                f"「{content}」→ {mt.value}",
                mt is getattr(MemType, want.upper()),
                f"期望 {want}，得到 {mt.value}（置信 {got_conf:.2f}，来源 {src}）",
            )
            check(
                f"  「{content}」置信度 {got_conf}",
                abs(got_conf - conf) < 0.01 and src == "llm",
                f"期望 {conf}/llm，得到 {got_conf:.2f}/{src}",
            )

        # ---------- 步骤 2：真实 HTTP 调用 + 结果缓存 ----------
        print("\n--- 2) HTTP 传输链路与缓存（同内容只调一次）---")
        from memagent.llm import _default_post

        calls = {"n": 0}

        def counting_post(url, headers, payload, timeout):
            calls["n"] += 1
            return _default_post(url, headers, payload, timeout)

        clf2 = LLMClassifier(post=counting_post)  # 同样的环境变量配置，仅统计 HTTP 调用
        mt, conf, src = clf2.classify("我在练习做饭")
        first_calls = calls["n"]
        mt2, conf2, src2 = clf2.classify("我在练习做饭")  # 缓存命中，不再发请求
        check(
            "首次分类走真实 HTTP 且只发 1 次请求",
            first_calls == 1 and src == "llm",
            f"实际 {first_calls} 次",
        )
        check("重复分类命中缓存（0 次新请求）", calls["n"] == 1 and (mt, conf) == (mt2, conf2))
        check("缓存结果与首次一致", mt is mt2 and abs(conf - conf2) < 0.01)

        # ---------- 步骤 3：MemoryAgent 自动分类入库（方式 ③ 的默认路径）----------
        print("\n--- 3) MemoryAgent 写入时自动分类（remember 不传 mtype）---")
        from memagent import MemoryAgent

        agent = MemoryAgent()  # 默认 classifier = LLMClassifier()（读环境变量）
        for content, want, conf in SAMPLES:
            mem = agent.remember(content)
            check(
                f"「{content}」入库类型 {mem.mtype.value}",
                mem.mtype is getattr(MemType, want.upper()) and mem.mtype_confidence is not None,
                f"mtype={mem.mtype.value} 置信={mem.mtype_confidence}",
            )
        check(
            "CLI 分类入口（agent.classify_text）同样走 LLM",
            agent.classify_text("我在练习做饭")[2] == "llm",
            agent.classify_text("我在练习做饭"),
        )

        # ---------- 步骤 4：LLM 回复生成器（检索结果注入上下文）----------
        print("\n--- 4) 回复生成：有记忆→基于记忆回答，无记忆→直接回答 ---")
        from memagent.responder import LLMResponder

        resp = LLMResponder()  # 同样读 OPENAI_* 环境变量
        check("回复生成器可用（有 key）", resp.available)
        calls2 = {"n": 0}

        def counting_post2(url, headers, payload, timeout):
            calls2["n"] += 1
            return _default_post(url, headers, payload, timeout)

        resp2 = LLMResponder(post=counting_post2)
        txt = resp2.respond("你记得我昨天吃了什么吗", memories=[("我昨天和同事去吃了火锅", "episodic", 0.42)])
        check("有相关记忆 → 基于记忆回答（真实 HTTP 1 次）", "基于记忆" in txt and calls2["n"] == 1, txt)
        txt2 = resp2.respond("天空为什么是蓝色的")
        check("无相关记忆 → LLM 直接回答", "直接回答" in txt2, txt2)

        agent_r = MemoryAgent(responder=LLMResponder())
        reply, hits = agent_r.respond("你记得我昨天吃了什么吗")  # 无旧记忆 → 直接回答模式
        check("Agent 集成：无记忆时 LLM 直接回答（非模板）", "直接回答" in reply, reply[:60])

        # ---------- 步骤 5：未配 key → 自动回退关键词规则 ----------
        print("\n--- 5) 离线回退：无 key 时自动用关键词规则（不依赖网络）---")
        os.environ.pop("OPENAI_API_KEY", None)
        clf_off = LLMClassifier()  # 无 key → available=False
        check("无 key 时分类器不可用（走回退）", not clf_off.available)
        mt, conf, src = clf_off.classify("昨天去吃了火锅")
        check("回退后仍能分类且来源为 keyword", src == "keyword" and mt is MemType.EPISODIC,
              f"得到 {mt.value}/{src}")
        # 回退同时作用于回复生成：无 key 的 agent 走模板
        agent_off = MemoryAgent(responder=LLMResponder())
        reply_off, _ = agent_off.respond("地球是圆的吗")
        check("无 key 时回复生成回退模板", "我还不太了解" in reply_off or "我记得" in reply_off, reply_off[:50])
    finally:
        if mock_proc is not None:
            mock_proc.terminate()

    print(f"\n结果：{_PASS} 通过 / {_FAIL} 失败")
    if _FAIL == 0:
        print("LLM 分类与回复生成链路验证通过 ✅")
    else:
        print("存在失败项 ❌")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
