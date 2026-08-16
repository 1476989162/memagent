"""LLM 分类链路测试：真实 HTTP 传输（线程版 mock server）+ 环境变量配置 + 示例脚本可运行。"""

import os
import subprocess
import sys
import threading
from http.server import HTTPServer

from memagent.llm import LLMClassifier
from memagent.memory import MemType

import mock_openai_server


def _serve_mock() -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), mock_openai_server.Handler)  # 端口 0 → 自动分配
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_real_http_chain():
    """走真实 HTTP 传输链路（urllib → 本地 HTTP server）：三类样例全部正确解析。"""
    server = _serve_mock()
    try:
        port = server.server_address[1]
        clf = LLMClassifier(
            api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1", model="mock-gpt"
        )
        assert clf.classify("我在练习做饭") == (MemType.SKILL, 0.93, "llm")
        assert clf.classify("昨天去吃了火锅") == (MemType.EPISODIC, 0.91, "llm")
        assert clf.classify("北京是中国的首都") == (MemType.SEMANTIC, 0.72, "llm")
    finally:
        server.shutdown()


def test_env_config_equivalence():
    """环境变量配置与构造参数配置等价（LLMClassifier 无参构造读 OPENAI_*）。"""
    os.environ["OPENAI_BASE_URL"] = "https://example.com/v1"
    os.environ["OPENAI_API_KEY"] = "env-key"
    os.environ["OPENAI_MODEL"] = "env-model"
    try:
        from_env = LLMClassifier()
        explicit = LLMClassifier(
            base_url="https://example.com/v1", api_key="env-key", model="env-model"
        )
        assert from_env.base_url == explicit.base_url == "https://example.com/v1"
        assert from_env.api_key == explicit.api_key == "env-key"
        assert from_env.model == explicit.model == "env-model"
        assert from_env.available and explicit.available
    finally:
        for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
            os.environ.pop(k, None)


def test_demo_script_runs_end_to_end():
    """示例脚本整体可运行：子进程显式清空 OPENAI_API_KEY（.env 里可能配了真实
    key，_load_dotenv 的 setdefault 对空串不生效）→ 走本地 mock 分支，退出码 0。"""
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = ""  # 空串阻止 .env 的真实 key 注入，强制 mock 分支
    env.pop("OPENAI_BASE_URL", None)
    env.pop("OPENAI_MODEL", None)
    proc = subprocess.run(
        [sys.executable, "llm_classify_demo.py"],
        capture_output=True, text=True, encoding="utf-8", timeout=120, env=env,
    )
    assert proc.returncode == 0, f"exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert "LLM 分类与回复生成链路验证通过" in proc.stdout
    assert "0 失败" in proc.stdout
    assert "来源应为 llm" in proc.stdout  # 三类样例确实走了 LLM 而非回退
