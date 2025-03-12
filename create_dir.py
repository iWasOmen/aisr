import os
import pathlib
from typing import Dict, List, Tuple

# 项目根目录
ROOT_DIR = "aisr"

# 定义目录结构
DIRECTORIES = [
    "core",
    "memory",
    "agents",
    "workflows",
    "tools",
    "utils",
    "config",
    "config/prompts",
    "tests",
    "tests/test_agents",
    "tests/test_workflows",
    "tests/test_memory",
    "tests/test_tools",
]

# 定义文件结构
FILES = [
    # 根目录文件
    "__init__.py",
    "main.py",
    "requirements.txt",

    # 核心模块
    "core/__init__.py",
    "core/base.py",
    "core/router.py",
    "core/llm_provider.py",

    # 内存系统
    "memory/__init__.py",
    "memory/base.py",
    "memory/global_memory.py",
    "memory/agent_memory.py",
    "memory/workflow_memory.py",
    "memory/manager.py",

    # 智能体
    "agents/__init__.py",
    "agents/base.py",
    "agents/task_plan.py",
    "agents/search_plan.py",
    "agents/sub_answer.py",
    "agents/insight.py",
    "agents/answer_plan.py",
    "agents/answer.py",

    # 工作流
    "workflows/__init__.py",
    "workflows/base.py",
    "workflows/task_planning.py",
    "workflows/sub_answer.py",
    "workflows/research.py",

    # 工具
    "tools/__init__.py",
    "tools/base.py",
    "tools/web_search.py",
    "tools/web_crawler.py",

    # 工具类
    "utils/__init__.py",
    "utils/logging.py",
    "utils/error_handling.py",

    # 配置
    "config/__init__.py",
    "config/settings.py",
    "config/prompts/__init__.py",
    "config/prompts/task_plan.py",
    "config/prompts/search_plan.py",

    # 测试
    "tests/__init__.py",
]

# 定义主要__init__.py文件的内容
INIT_CONTENTS = {
    "__init__.py": """\"\"\"AISR - AI-assisted Search and Research System\"\"\"

from aisr.core.base import Component
from aisr.core.router import Router
""",

    "core/__init__.py": """\"\"\"核心系统组件\"\"\"

from aisr.core.base import Component
from aisr.core.router import Router
""",

    "memory/__init__.py": """\"\"\"记忆系统组件\"\"\"

from aisr.memory.base import Memory
from aisr.memory.global_memory import GlobalMemory
from aisr.memory.agent_memory import AgentMemory
from aisr.memory.workflow_memory import WorkflowMemory
from aisr.memory.manager import MemoryManager
""",

    "agents/__init__.py": """\"\"\"智能体组件\"\"\"

from aisr.agents.base import Agent
from aisr.agents.task_plan import TaskPlanAgent
from aisr.agents.search_plan import SearchPlanAgent
from aisr.agents.sub_answer import SubAnswerAgent
from aisr.agents.insight import InsightAgent
from aisr.agents.answer_plan import AnswerPlanAgent
from aisr.agents.answer import AnswerAgent
""",

    "workflows/__init__.py": """\"\"\"工作流组件\"\"\"

from aisr.workflows.base import Workflow
from aisr.workflows.task_planning import TaskPlanningWorkflow
from aisr.workflows.search_planning import SearchPlanningWorkflow
from aisr.workflows.sub_answer import SubAnswerWorkflow
from aisr.workflows.research import ResearchWorkflow
""",

    "tools/__init__.py": """\"\"\"工具组件\"\"\"

from aisr.tools.base import Tool
from aisr.tools.web_search import WebSearchTool
from aisr.tools.web_crawler import WebCrawlerTool
""",
}

# 定义已实现的抽象基类代码
BASE_CLASS_CONTENTS = {
    "core/base.py": """import abc
from typing import Dict, Any

class Component(abc.ABC):
    \"\"\"
    所有AISR系统组件的抽象基类。

    作为Agent、Workflow和Tool的基础，提供统一执行接口。
    \"\"\"

    @abc.abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        执行组件的功能。

        Args:
            context: 包含执行所需输入数据和参数的字典

        Returns:
            包含执行结果的字典
        \"\"\"
        pass

    def get_id(self) -> str:
        \"\"\"获取组件的唯一标识\"\"\"
        return self.__class__.__name__
""",

    "memory/base.py": """import abc
from typing import Dict, Any

class Memory(abc.ABC):
    \"\"\"
    AISR内存系统的抽象基类。

    负责在研究过程中存储、检索和管理信息状态。
    \"\"\"

    @abc.abstractmethod
    def add(self, entry: Dict[str, Any]) -> None:
        \"\"\"
        向内存添加新条目。

        Args:
            entry: 包含要存储数据的字典
        \"\"\"
        pass

    @abc.abstractmethod
    def get_relevant(self, context: Dict[str, Any]) -> list[Dict[str, Any]]:
        \"\"\"
        基于提供的上下文检索相关内存。

        Args:
            context: 包含指导内存检索的参数的字典

        Returns:
            相关内存条目的列表
        \"\"\"
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        \"\"\"清除所有存储的内存\"\"\"
        pass

    def summarize(self, context: Dict[str, Any] = None) -> str:
        \"\"\"创建相关内存的简洁摘要\"\"\"
        return ""  # 默认实现，子类应重写
""",

    "agents/base.py": """import abc
from typing import Dict, Any
from aisr.core.base import Component

class Agent(Component):
    \"\"\"
    AISR中所有LLM驱动智能体的抽象基类。

    智能体利用LLM能力根据提供的上下文生成决策、洞察和分析。
    \"\"\"

    def __init__(self, llm_provider, memory):
        \"\"\"
        初始化智能体。

        Args:
            llm_provider: LLM服务提供者
            memory: 智能体的内存系统
        \"\"\"
        self.llm = llm_provider
        self.memory = memory

    @abc.abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"执行智能体的推理任务\"\"\"
        pass

    @abc.abstractmethod
    def build_prompt(self, context: Dict[str, Any]) -> str:
        \"\"\"基于上下文和内存为LLM构建提示\"\"\"
        pass

    @abc.abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        \"\"\"将LLM响应解析为结构化输出\"\"\"
        pass
""",

    "workflows/base.py": """import abc
from typing import Dict, Any
from aisr.core.base import Component

class Workflow(Component):
    \"\"\"
    AISR中所有工作流组件的抽象基类。

    工作流负责编排执行流程，协调Agent和Tool完成特定任务。
    \"\"\"

    def __init__(self, router, memory):
        \"\"\"
        初始化工作流。

        Args:
            router: 用于调用其他组件的路由器
            memory: 工作流的内存系统
        \"\"\"
        self.router = router
        self.memory = memory

    @abc.abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"执行工作流逻辑\"\"\"
        pass

    def call_component(self, function: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        通过路由器调用另一个组件。

        Args:
            function: 要调用的函数的字符串标识符
            parameters: 要传递的参数字典

        Returns:
            被调用组件的结果
        \"\"\"
        return self.router.route({"function": function, "parameters": parameters})
""",

    "tools/base.py": """import abc
from typing import Dict, Any
from aisr.core.base import Component

class Tool(Component):
    \"\"\"
    AISR中所有工具的抽象基类。

    工具是执行特定功能的组件，如网页搜索、爬取、数据处理等。
    \"\"\"

    @abc.abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"执行工具的功能\"\"\"
        pass

    @abc.abstractmethod
    def get_description(self) -> str:
        \"\"\"获取此工具在LLM提示中使用的描述\"\"\"
        pass

    def is_available(self) -> bool:
        \"\"\"检查此工具当前是否可用\"\"\"
        return True  # 默认实现，子类应根据需要重写
""",

    "core/router.py": """from typing import Dict, Any

class Router:
    \"\"\"
    组件通信的中央路由系统。

    处理组件间的函数调用路由，提供统一的组件间通信接口。
    \"\"\"

    def __init__(self):
        \"\"\"初始化路由器\"\"\"
        self.components = {}  # 存储注册组件的字典

    def register(self, name: str, component) -> None:
        \"\"\"
        向路由器注册组件。

        Args:
            name: 组件的字符串标识符
            component: 组件实例
        \"\"\"
        self.components[name] = component

    def route(self, function_call: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        将函数调用路由到适当的组件。

        Args:
            function_call: 包含'function'和'parameters'键的字典

        Returns:
            被调用组件的结果

        Raises:
            ValueError: 如果找不到组件或方法
        \"\"\"
        # 解析函数调用
        function_path = function_call.get("function", "")
        parameters = function_call.get("parameters", {})

        if "." not in function_path:
            raise ValueError(f"无效的函数路径: {function_path}。预期格式: 'component.method'")

        component_name, method_name = function_path.split(".", 1)

        # 获取组件
        component = self.components.get(component_name)
        if not component:
            raise ValueError(f"未找到组件: {component_name}")

        # 获取方法
        method = getattr(component, method_name, None)
        if not method or not callable(method):
            raise ValueError(f"未找到方法: {method_name} in component {component_name}")

        # 执行方法
        try:
            return method(**parameters)
        except Exception as e:
            # 实际实现中会包含更复杂的错误处理、日志记录和可能的重试逻辑
            return {
                "error": str(e),
                "component": component_name,
                "method": method_name,
                "status": "failed"
            }
""",
}


def create_project_structure():
    """创建完整项目结构"""
    # 创建根目录
    root = pathlib.Path(ROOT_DIR)
    if not root.exists():
        root.mkdir()
        print(f"创建目录: {ROOT_DIR}")

    # 创建子目录
    for directory in DIRECTORIES:
        dir_path = root / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"创建目录: {ROOT_DIR}/{directory}")

    # 创建文件
    for file in FILES:
        file_path = root / file

        # 如果文件存在，跳过创建
        if file_path.exists():
            print(f"跳过已存在文件: {ROOT_DIR}/{file}")
            continue

        # 检查是否有预定义的内容
        content = ""
        if file in INIT_CONTENTS:
            content = INIT_CONTENTS[file]
        elif file in BASE_CLASS_CONTENTS:
            content = BASE_CLASS_CONTENTS[file]

        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"创建文件: {ROOT_DIR}/{file}")

    print(f"\n项目结构已成功创建在 '{ROOT_DIR}' 目录下！")


if __name__ == "__main__":
    create_project_structure()