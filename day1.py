# 导入所需内置库
import os          # 读取环境变量配置
import json        # 解析大模型返回的工具调用参数

# 导入第三方库
from dotenv import load_dotenv   # 读取本地.env配置文件
from openai import OpenAI        # 大模型统一调用SDK

# ------------------------------
# 第一步：加载私密配置 + 初始化大模型客户端
# ------------------------------
# 加载项目根目录下的.env文件中的所有配置
load_dotenv()

# 创建大模型请求客户端，兼容所有OpenAI格式接口
client = OpenAI(
    # 从环境变量读取密钥
    api_key=os.getenv("OPENAI_API_KEY"),
    # 从环境变量读取大模型接口地址
    base_url=os.getenv("OPENAI_BASE_URL")
)

# ------------------------------
# 第二步：定义工具清单（给大模型看的工具说明书）
# 格式固定：名称、功能描述、入参结构
# ------------------------------
TOOLS = [
    # 工具1：查询天气
    {
        "type": "function",
        "function": {
            "name": "get_weather",                # 工具函数名(必须和下方函数一致)
            "description": "查询指定城市的天气信息", # 告诉大模型这个工具能干什么
            "parameters": {                       # 定义调用该工具需要什么参数
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海"
                    }
                },
                "required": ["city"]  # 标记必填参数
            }
        }
    },
    # 工具2：网络搜索
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索网络信息，查询知识点、概念",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词、问题"
                    }
                },
                "required": ["query"]
            }
        }
    },
    # 工具3：数学计算
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学四则运算计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：123*456、(10+20)/2"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ------------------------------
# 第三步：实现工具具体功能（真正执行代码逻辑）
# ------------------------------
def get_weather(city: str) -> str:
    """根据城市名返回模拟天气数据"""
    # 模拟本地天气数据库
    weather_data = {
        "北京": "晴天，25°C，湿度40%",
        "上海": "多云，28°C，湿度65%",
        "深圳": "阵雨，30°C，湿度80%"
    }
    # 存在返回天气，不存在返回提示语
    return weather_data.get(city, f"未找到{city}的天气信息")


def search_web(query: str) -> str:
    """模拟联网搜索功能"""
    # 模拟搜索知识库
    data = {
        "langchain": "LangChain 是一个用于开发大语言模型应用的开源框架，支持链式调用、工具调用、记忆等能力。"
    }
    # 简易关键词匹配
    for key in data:
        if key in query.lower():
            return data[key]
    return f"搜索结果：{query}"


def calculate(expression: str) -> str:
    """执行数学表达式计算"""
    try:
        # eval可以直接执行字符串格式数学公式
        return f"计算结果：{eval(expression)}"
    except Exception:
        # 表达式错误捕获异常
        return "计算错误，请检查数学表达式"

# ------------------------------
# 第四步：工具名字映射字典
# 作用：大模型返回工具名 -> 找到对应执行函数
# ------------------------------
TOOL_MAP = {
    "get_weather": get_weather,
    "search_web": search_web,
    "calculate": calculate
}

# ------------------------------
# 第五步：Agent核心循环逻辑（最核心！思考-行动-观察）
# ------------------------------
def run_agent(user_message: str, max_iterations=5):
    """
    运行智能Agent主逻辑
    :param user_message: 用户输入问题
    :param max_iterations: 最大循环轮次，防止死循环
    """
    # 构建对话消息列表，存储全程聊天记录+工具记录
    messages = [
        # 系统提示词：约束大模型行为规则
        {"role": "system", "content": "你是一个智能助手，必须严格使用工具完成任务，不要编造答案。"},
        # 用户第一轮提问
        {"role": "user", "content": user_message}
    ]

    print(f"\n==== 用户：{user_message} ====")

    # 开始Agent多轮循环
    for _ in range(max_iterations):
        # 向大模型发送请求，传入对话历史+工具列表
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),  # 指定使用的大模型名称
            messages=messages,             # 传入完整对话上下文
            tools=TOOLS,                    # 传给大模型所有可用工具
            tool_choice="auto"              # 让大模型自动决定是否调用工具
        )

        # 取出大模型返回的消息内容
        assistant_msg = response.choices[0].message

        # 分支1：大模型不需要调用任何工具，直接输出最终答案
        if not assistant_msg.tool_calls:
            print(f"\n🎯 最终回答：\n{assistant_msg.content}")
            return

        # 分支2：大模型决定调用工具，将模型回复加入对话历史
        messages.append(assistant_msg)

        # 遍历所有需要调用的工具（支持一次调用多个工具）
        for tool_call in assistant_msg.tool_calls:
            # 获取工具名称
            func_name = tool_call.function.name
            # 解析工具调用传入的参数（json字符串转字典）
            args = json.loads(tool_call.function.arguments)

            # 打印日志：可视化调用过程
            print(f"\n🔧 调用工具：{func_name} 参数：{args}")
            # 通过映射表执行对应工具函数，传入参数
            result = TOOL_MAP[func_name](**args)
            # 打印工具执行结果
            print(f"📊 工具返回：{result}")

            # 将工具执行结果传回对话历史，交给大模型继续思考
            messages.append({
                "role": "tool",               # 角色固定为tool
                "tool_call_id": tool_call.id,# 绑定对应工具调用ID
                "content": result            # 工具返回的内容
            })

# ------------------------------
# 第六步：预设测试用例，批量测试Agent能力
# ------------------------------
TEST_CASES = [
    {
        "name": "Test 1 - 简单问答（无需工具）",
        "message": "什么是 Python 编程语言？请用一句话回答。",
        "expected_tools": 0,
    },
    {
        "name": "Test 2 - 天气查询（单工具）",
        "message": "上海今天天气怎么样？",
        "expected_tools": 1,
    },
    {
        "name": "Test 3 - 组合查询（搜索 + 计算）",
        "message": "搜索一下什么是 LangChain，然后帮我算 123 * 456 等于多少。",
        "expected_tools": 2,
    },
    {
        "name": "Test 4 - 需要推理的复杂查询",
        "message": "北京和深圳今天哪个城市更热？温度差多少？",
        "expected_tools": 2,
    },
]

# ------------------------------
# 第七步：程序主入口函数
# ------------------------------
def main():
    # 打印程序标题
    print("╔══════════════════════════════════════════════════════╗")
    print("║          第1章：第一个 Agent - Hello World           ║")
    print("║          理解 Agent 循环的底层原理                    ║")
    print("╚══════════════════════════════════════════════════════╝")

    # 遍历执行所有测试案例
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n{'#'*60}")
        print(f"# {test['name']}")
        print(f"# 预期工具调用数: {test['expected_tools']}")
        print(f"{'#' * 60}")

        # 执行当前测试对话
        try:
            run_agent(test["message"], max_iterations=5)
        except Exception as e:
            print(f"\n❌ 运行出错: {e}")

        # 不是最后一个测试，就暂停等待回车
        if i < len(TEST_CASES):
            print("\n" + "-" * 60)
            input("按 Enter 继续下一个测试...")

# 程序运行入口
if __name__ == "__main__":
    main()
