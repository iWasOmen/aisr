#!/usr/bin/env python
"""
AISR - AI-assisted Search and Research System
主程序入口点和使用示例
"""

import os
import argparse
import logging
from typing import Dict, Any

from aisr.core.router import Router
from aisr.memory.manager import MemoryManager
from aisr.core.llm_provider import LLMProvider

# 导入工作流
from aisr.workflows.research import ResearchWorkflow
from aisr.workflows.task_planning import TaskPlanningWorkflow
from aisr.workflows.search_planning import SearchPlanningWorkflow
from aisr.workflows.sub_answer import SubAnswerWorkflow

# 导入智能体
from aisr.agents.base import Agent
from aisr.agents import Agent
from aisr.agents.task_plan import TaskPlanAgent
from aisr.agents.search_plan import SearchPlanAgent
from aisr.agents.sub_answer import SubAnswerAgent
from aisr.agents.insight import InsightAgent
from aisr.agents.answer_plan import AnswerPlanAgent
from aisr.agents.answer import AnswerAgent

# 导入工具
from aisr.tools.web_search import WebSearchTool
from aisr.tools.web_crawler import WebCrawlerTool

# 工具类
from aisr.utils.logging import setup_logging


class AISystem:
    """
    AISR系统主类，负责初始化和协调整个系统。
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化AISR系统。

        Args:
            config: 系统配置参数
        """
        self.config = config or {}

        # 设置日志
        setup_logging(self.config.get("log_level", "INFO"))

        # 初始化核心组件
        self.memory_manager = MemoryManager()
        self.router = Router()

        # 初始化LLM提供者
        self.llm_provider = LLMProvider(
            provider=self.config.get("llm_provider", "anthropic"),
            api_key=self.config.get("api_key"),
            model=self.config.get("model")
        )

        # 初始化各组件
        self._initialize_components()

    def _initialize_components(self):
        """初始化并注册所有系统组件。"""
        # 初始化工具
        web_search = WebSearchTool(self.config.get("serpapi_key"))
        web_crawler = WebCrawlerTool(self.config.get("jina_api_key"))

        # 注册工具
        self.router.register("web_search", web_search)
        self.router.register("web_crawler", web_crawler)

        # 初始化智能体
        task_plan_agent = TaskPlanAgent(
            self.llm_provider,
            self.memory_manager.get_memory_view("task_plan_agent")
        )

        search_plan_agent = SearchPlanAgent(
            self.llm_provider,
            self.memory_manager.get_memory_view("search_plan_agent")
        )

        sub_answer_agent = SubAnswerAgent(
            self.llm_provider,
            self.memory_manager.get_memory_view("sub_answer_agent")
        )

        insight_agent = InsightAgent(
            self.llm_provider,
            self.memory_manager.get_memory_view("insight_agent")
        )

        answer_plan_agent = AnswerPlanAgent(
            self.llm_provider,
            self.memory_manager.get_memory_view("answer_plan_agent")
        )

        answer_agent = AnswerAgent(
            self.llm_provider,
            self.memory_manager.get_memory_view("answer_agent")
        )

        # 注册智能体
        self.router.register("task_plan_agent", task_plan_agent)
        self.router.register("search_plan_agent", search_plan_agent)
        self.router.register("sub_answer_agent", sub_answer_agent)
        self.router.register("insight_agent", insight_agent)
        self.router.register("answer_plan_agent", answer_plan_agent)
        self.router.register("answer_agent", answer_agent)

        # 初始化工作流
        task_planning_workflow = TaskPlanningWorkflow(
            self.router,
            self.memory_manager.get_memory_view("task_planning_workflow")
        )

        search_planning_workflow = SearchPlanningWorkflow(
            self.router,
            self.memory_manager.get_memory_view("search_planning_workflow")
        )

        sub_answer_workflow = SubAnswerWorkflow(
            self.router,
            self.memory_manager.get_memory_view("sub_answer_workflow")
        )

        research_workflow = ResearchWorkflow(
            self.router,
            self.memory_manager.get_memory_view("research_workflow")
        )

        # 注册工作流
        self.router.register("task_planning_workflow", task_planning_workflow)
        self.router.register("search_planning_workflow", search_planning_workflow)
        self.router.register("sub_answer_workflow", sub_answer_workflow)
        self.router.register("research_workflow", research_workflow)

        logging.info("AISR系统组件已初始化完成")

    def run_research(self, query: str) -> Dict[str, Any]:
        """
        执行研究任务。

        Args:
            query: 研究查询文本

        Returns:
            研究结果
        """
        logging.info(f"开始执行研究任务: '{query}'")

        # 清除之前的研究状态
        self.memory_manager.clear_research_state()

        # 保存查询到全局内存
        self.memory_manager.save_global_state("query", query)

        # 执行研究工作流
        result = self.router.route({
            "function": "research_workflow.execute",
            "parameters": {"query": query}
        })

        logging.info(f"研究任务完成: '{query}'")
        return result

    def get_research_history(self) -> Dict[str, Any]:
        """获取研究历史记录。"""
        return self.memory_manager.get_global_state("research_history", {})


def main():
    """主程序入口点。"""
    parser = argparse.ArgumentParser(description="AISR - AI-assisted Search and Research System")
    parser.add_argument("--query", "-q", help="研究查询")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic", help="LLM提供者")
    parser.add_argument("--api-key", help="API密钥")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="日志级别")

    args = parser.parse_args()

    # 获取环境变量中的API密钥（如果命令行未提供）
    api_key = args.api_key
    if not api_key:
        provider_env = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
        api_key = os.environ.get(provider_env)

    # 配置系统
    config = {
        "llm_provider": args.provider,
        "api_key": api_key,
        "model": args.model,
        "log_level": args.log_level,
        "serpapi_key": os.environ.get("SERPAPI_KEY"),
        "jina_api_key": os.environ.get("JINA_API_KEY")
    }

    # 初始化系统
    system = AISystem(config)

    # 如果提供了查询，则执行研究
    if args.query:
        result = system.run_research(args.query)
        print("\n=== 研究结果 ===\n")

        # 打印标题和摘要
        print(f"标题: {result.get('title', '无标题')}")
        print(f"摘要: {result.get('summary', '无摘要')}\n")

        # 打印内容部分
        print("=== 内容 ===\n")
        for section in result.get("content", []):
            print(f"## {section.get('section', '无标题部分')}")
            print(f"{section.get('content', '无内容')}\n")

        # 打印元数据
        print("=== 元数据 ===\n")
        metadata = result.get("metadata", {})
        print(f"置信度: {metadata.get('confidence', 'N/A')}")
        print(f"使用源数: {metadata.get('sources_used', 'N/A')}")
        print(f"生成时间: {metadata.get('generated_at', 'N/A')}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()