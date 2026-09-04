"""测试隔离：
- 清除 LLM 环境变量，保证默认分类器走关键词回退，测试可复现；
- MEMAGENT_TIME_SCALE 默认 1（人类尺度）：绝大多数测试用注入时钟、按人类尺度
  编写（τ 以天为单位、时钟增量小），而出厂默认 1/86400 会把 τ 压到秒级，
  导致强度瞬间压到下限、记忆立刻被压缩——测试与默认缩放冲突。
  必须在 memagent 导入前设置（TIME_SCALE 是导入时常量）。
- MEMAGENT_TEST=1（非空，Windows 子进程才能继承）：memagent/llm.py::
  _load_dotenv 检测到即跳过 .env 加载——否则子进程 CLI 测试
  （recall_curve_check / session_memory / remember_agent 等）会重新加载 .env
  里的真实 OPENAI_API_KEY 发起真实 LLM 网络调用（单次可超 60s，慢且不可复现）。
"""

import os

os.environ.setdefault("MEMAGENT_TIME_SCALE", "1")
os.environ["MEMAGENT_TEST"] = "1"

import pytest


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
                "OPENAI_MODELS", "OPENAI_PERSONA", "OPENAI_THINKING"):
        monkeypatch.delenv(key, raising=False)
