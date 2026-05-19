# 导入基础模块
import os
import json
import re
# 导入环境变量加载模块
from dotenv import load_dotenv
# 导入OpenAI客户端
from openai import OpenAI

# 加载.env文件中的环境变量（API密钥、模型地址等）
load_dotenv()
# 初始化OpenAI客户端
client = OpenAI(
    # 从环境变量读取API密钥，避免硬编码
    api_key=os.getenv("OPENAI_API_KEY"),
    # 从环境变量读取大模型接口地址（支持自定义部署的OpenAI兼容接口）
    base_url=os.getenv("OPENAI_BASE_URL")
)


def call_llm_with_json(prompt: str, system_msg: str = " ") -> dict:
    """
    调用大模型并强制返回JSON格式结果
    :param prompt: 用户提示词（核心指令）
    :param system_msg: 系统提示词（定义大模型角色/规则）
    :return: 解析后的JSON字典
    """
    # 从环境变量读取模型名称（如gpt-3.5-turbo、gpt-4）
    model = os.getenv("LLM_MODEL")
    # 构造对话消息列表
    messages = []
    if system_msg:
        # 添加系统角色消息（定义行为准则）
        messages.append({"role": "system", "content": system_msg})
        # 添加用户提示消息
        messages.append({"role": "system", "content": prompt})

    # 调用OpenAI ChatCompletion接口
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},  # 强制返回JSON格式，关键配置
        temperature=0.3,  # 低随机性，保证结果稳定
    )
    # 解析JSON响应内容并返回
    return json.loads(response.choices[0].message.content)


def plan_execute(task: str) -> dict:
    """
    任务规划器：将自然语言任务拆解为结构化执行步骤
    :param task: 原始任务描述（自然语言）
    :return: 包含执行步骤的JSON字典，结构：{"plan": [{"step": 步骤号, "description": 步骤描述, "tool": 工具名, "depends_on": 依赖步骤列表}]}
    """
    # 系统提示词：定义规划器角色和输出格式要求
    system_msg = (
        "你是一个任务规划专家。请严格按照以下要求返回JSON格式，不得遗漏任何字段：\n"
        "JSON结构必须包含：\n"
        '{"plan": [{"step": 整数（步骤编号）, "description": 字符串（步骤描述）, "tool": 字符串（工具名）, "depends_on": 列表（依赖步骤号）}]}\n'
        "要求：\n"
        "1. description 字段必须详细描述该步骤要做什么，不能为空\n"
        "2. tool 字段必须填写具体工具名（如：天气查询API、文本对比工具）\n"
        "3. depends_on 字段即使无依赖也要填空列表 []\n"
        "4. 只返回JSON，不要添加任何额外文字、解释或格式说明"
    )
    # 构造用户提示：拼接任务描述和格式要求
    prompt = f"任务：{task}\n请为该任务制定详细的执行计划，严格遵守上述JSON格式要求。"
    # 调用LLM并返回规划结果
    return call_llm_with_json(prompt, system_msg)


def reflexion_review(original_answer: str, criteria: str) -> dict:
    """
    反思审查器：根据评估标准审查回答质量，指出问题并生成改进版本
    :param original_answer: 原始回答文本
    :param criteria: 评估标准（自然语言）
    :return: 包含评分、问题、改进后回答的JSON字典
    """
    # 系统提示词：定义审查员角色和输出格式
    system_msg = (
        "你是一个严格的质量审查员。请审查以下回答，"
        "指出问题和改进建议。返回严格的 JSON 格式：\n"
        '{"score": 1-10, "issues": ["问题1", "问题2"], '
        '"improved_answer": "改进后的回答"}'
    )

    # 构造用户提示：拼接原始回答和评估标准
    prompt = (
        f"原始回答：\n{original_answer}\n\n"
        f"评估标准：{criteria}\n\n"
        f"请评估并改进。"
    )
    # 调用LLM进行审查和改进
    return call_llm_with_json(prompt, system_msg)


class AgentMemory:
    """
    智能体记忆系统：整合短期记忆、工作记忆、长期记忆
    - 短期记忆：最近的对话记录（可压缩）
    - 工作记忆：任务执行中的中间状态/结果
    - 长期记忆：压缩后的历史对话摘要
    """

    def __init__(self, max_short_term: int = 10):
        """
        初始化记忆系统
        :param max_short_term: 短期记忆最大存储对话对数，超过则触发压缩
        """
        self.short_term = []  # 短期记忆：最近对话 [(user_msg, assistant_msg), ...]
        self.working = {}  # 工作记忆：任务中间状态 {key: value}
        self.long_term = []  # 长期记忆：压缩后的历史对话
        self.max_short_term = max_short_term  # 短期记忆上限
        self.summary = ""  # 早期对话的聚合摘要

    def add_conversation(self, user_msg: str, assistant_msg: str):
        """
        添加新的对话到短期记忆，超过上限则触发压缩
        :param user_msg: 用户消息
        :param assistant_msg: 助手回复消息
        """
        self.short_term.append((user_msg, assistant_msg))
        # 检查短期记忆是否超限
        if len(self.short_term) > self.max_short_term:
            self._compress()

    def _compress(self):
        """
        私有方法：压缩短期记忆，将最早的对话转移到长期记忆并更新摘要
        """
        # 弹出最早的一对对话
        old_pair = self.short_term.pop(0)
        # 格式化对话内容
        combined = f"[用户]: {old_pair[0]}\n[助手]: {old_pair[1]}"
        # 加入长期记忆
        self.long_term.append(combined)

        # 更新对话摘要
        if not self.summary:
            self.summary = f"早期对话摘要：\n{combined}"
        else:
            self.summary += f"\n...\n{combined}"

    def set_working(self, key: str, value: str):
        """设置工作记忆的键值对"""
        self.working[key] = value

    def get_working(self, key: str) -> str:
        """获取工作记忆中指定键的值，默认返回空字符串"""
        return self.working.get(key, "")

    def get_context_for_llm(self) -> str:
        """
        构建供LLM使用的上下文信息（整合摘要、工作记忆、最近对话）
        :return: 格式化的上下文文本
        """
        parts = []
        # 1. 添加历史对话摘要
        if self.summary:
            parts.append(f"## 历史摘要\n{self.summary}\n")
        # 2. 添加当前任务状态（工作记忆）
        if self.working:
            wk_items = "\n".join([f"{k}: {v}" for k, v in self.working.items()])
            parts.append(f"## 当前任务状态\n{wk_items}\n")
        # 3. 添加最近的5轮对话（短期记忆）
        if self.short_term:
            recent = "\n".join(
                f"用户: {u}\n助手: {a}"
                for u, a in self.short_term[-5:]  # 只取最近5轮
            )
            parts.append(f"## 最近对话\n{recent}")
        # 拼接所有部分并返回
        return "\n".join(parts)


def demonstrate_tool_design():
    """演示：高质量工具定义 vs 低质量工具定义的对比"""
    print("=" * 60)
    print("  工具设计对比：好 vs 坏")
    print("=" * 60)

    # 低质量工具定义：描述模糊、参数简单、无示例
    bad_tool = {
        "type": "function",
        "function": {
            "name": "do_stuff",
            "description": "处理数据。",  # 描述过于模糊
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string"}  # 无参数描述、无必选限制
                },
            },
        },
    }

    # 高质量工具定义：描述清晰、参数详细、有示例、有使用限制
    good_tool = {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "向指定邮箱发送邮件。适用于发送通知、报告、提醒等场景。"
                "不支持发送带附件的邮件。收件人地址必须包含 @。"
                "示例：send_email(to='user@example.com', "
                "subject='日报', body='今日数据...')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "收件人邮箱地址，如 user@example.com",
                    },
                    "subject": {
                        "type": "string",
                        "description": "邮件主题，不超过 100 字",
                    },
                    "body": {
                        "type": "string",
                        "description": "邮件正文，纯文本格式",
                    },
                },
                "required": ["to", "subject", "body"],  # 明确必选参数
            },
        },
    }

    # 打印对比结果
    print("\n❌ 低质量工具定义：")
    print(json.dumps(bad_tool, indent=2, ensure_ascii=False))
    print("\n✅ 高质量工具定义：")
    print(json.dumps(good_tool, indent=2, ensure_ascii=False))


def run_full_demo():
    """
    综合演示：Agent三大核心组件（规划器、记忆系统、反思器）协同工作
    演示流程：任务规划 → 模拟工具调用 → 生成回答 → 反思改进 → 记忆存储
    """
    print("\n" + "█" * 60)
    print("█  综合演示：Agent 三大组件协同工作")
    print("█" * 60)

    # 初始化记忆系统（短期记忆上限6条）
    memory = AgentMemory(max_short_term=6)
    # 定义演示任务
    task = "调查北京和上海的天气，写一份简短对比报告"
    print(f"\n📋 任务: {task}")

    # 阶段 1：规划器制定执行计划
    print("\n--- 阶段 1: 规划器制定计划 ---")
    plan = plan_execute(task)
    print(f"计划 {len(plan['plan'])} 个步骤：")
    for step in plan["plan"]:
        # 格式化依赖步骤显示
        deps = f" (依赖步骤 {step['depends_on']})" if step.get("depends_on") else ""
        print(f"  步骤 {step['step']}: {step['description']}{deps}")
        # 将步骤信息存入工作记忆
        memory.set_working(f"step_{step['step']}", step["description"])

    # 阶段 2：模拟工具调用（天气查询）
    print("\n--- 阶段 2: 工具调用（模拟）---")
    weather_data = {
        "北京": "晴天 25°C 湿度40%",
        "上海": "多云 28°C 湿度65%",
    }
    for city in ["北京", "上海"]:
        result = weather_data[city]
        # 将天气数据存入工作记忆
        memory.set_working(f"weather_{city}", result)
        print(f"  get_weather({city}) → {result}")

    # 阶段 3：生成初始回答（模拟）
    print("\n--- 阶段 3: 生成对比报告（模拟）---")
    # 从工作记忆读取天气数据
    beijing = memory.get_working("weather_北京")
    shanghai = memory.get_working("weather_上海")
    # 构造初始回答
    answer = f"北京天气：{beijing}\n上海天气：{shanghai}\n建议：上海温度更高且湿度大，"
    answer += "北京更适合户外活动。"
    print(f"  初版回答:\n{answer}")

    # 阶段 4：反思器自我审查并改进
    print("\n--- 阶段 4: 反思器自我审查 ---")
    review = reflexion_review(
        answer,
        "应包含数值对比（温度差、湿度差）和明确的活动建议"  # 评估标准
    )
    print(f"  评分: {review['score']}/10")
    print(f"  问题: {review['issues']}")
    print(f"\n  改进后回答:\n{review['improved_answer']}")

    # 将最终对话存入记忆系统
    memory.add_conversation(task, review['improved_answer'])

    # 展示记忆系统的最终状态
    print("\n--- 记忆系统状态 ---")
    print(memory.get_context_for_llm())

    print("\n✅ 综合演示完成！")


# 主程序入口
if __name__ == "__main__":
    # 打印标题
    print("╔══════════════════════════════════════════════════════╗")
    print("║       第2章：Agent 核心组件深度解析                    ║")
    print("║       规划器 · 记忆系统 · 工具设计                    ║")
    print("╚══════════════════════════════════════════════════════╝")

    # 演示1：规划器示例
    print("\n▶ 2.1-2.2: 规划器示例（Plan-Execute）")
    plan = plan_execute("分析特斯拉股票是否值得投资")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    # 演示2：工具设计对比
    print("\n▶ 2.3: 工具设计对比演示")
    demonstrate_tool_design()

    # 演示3：综合流程（需配置API Key）
    print("\n▶ 2.4: 综合演示（需要 API 调用）")
    try:
        run_full_demo()
    except Exception as e:
        # 捕获API配置错误并提示
        print(f"\n⚠️ 综合演示需要 API Key，错误信息: {e}")
        print("请确保 .env 文件配置正确后重试。")
