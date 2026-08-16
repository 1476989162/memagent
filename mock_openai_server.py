"""临时 mock OpenAI 兼容服务：验证 LLMClassifier 分类与 LLMResponder 回复生成
的真实 HTTP 传输链路。按 user 消息前缀区分两类请求：
- 以「记忆内容：」开头 → 分类请求（返回严格 JSON）；
- 其他 → 回复生成请求（返回文本，回显查询并标注基于记忆/直接回答）。
"""

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        # 按 system 提示区分分类/回复生成：分类提示含「分类」，回复生成提示是助手
        system = next((m.get("content", "") for m in body.get("messages", [])
                       if m.get("role") == "system"), "")
        content = body.get("messages", [{}])[-1].get("content", "")
        if "分类" in system:
            resp = self._classify(content)
        else:
            resp = self._generate(content)
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _classify(self, content):
        # 模拟分类：按关键词返回严格 JSON
        text = content.replace("记忆内容：", "")
        if any(k in text for k in ("学习", "练习", "学会", "做饭", "弹琴")):
            mtype, conf = "skill", 0.93
        elif any(k in text for k in ("昨天", "今天", "吃了", "去了")):
            mtype, conf = "episodic", 0.91
        else:
            mtype, conf = "semantic", 0.72
        return {"choices": [{"message": {"content": json.dumps({"type": mtype, "confidence": conf})}}]}

    def _generate(self, content):
        # 模拟回复生成：回显查询，标注基于记忆 / 直接回答两种模式
        m = re.search(r"用户问题：(.*?)(?:\n|$)", content)
        q = m.group(1).strip() if m else ""
        if "检索到的相关记忆" in content:
            reply = f"（mock 生成·基于记忆）关于「{q}」：根据你的长期记忆回答。"
        else:
            reply = f"（mock 生成·直接回答）关于「{q}」：我凭常识回答你。"
        return {"choices": [{"message": {"content": reply}}]}

    def log_message(self, fmt, *args):  # 静音
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
