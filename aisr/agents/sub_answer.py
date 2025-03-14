"""
子答案生成代理模块，负责分析搜索结果并为子任务生成回答。

作为内层工作流的关键组件，负责将原始搜索结果转化为对子任务的直接回答。
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
import json

from aisr.agents.base import Agent
from aisr.utils import logging_utils
logger = logging_utils.get_logger(__name__, color="green")

class SubAnswerAgent(Agent):
    """
    子答案生成代理，分析搜索结果并生成子任务的回答。

    输入：
    - 当前子任务
    - 搜索结果

    输出：
    - 子任务的回答
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行搜索结果分析，生成子任务的回答。

        Args:
            context: 包含执行所需输入的字典
                - task: 当前子任务
                - search_results: 搜索结果

        Returns:
            包含子答案的字典
        """
        logger.info(f"===SubAnswerAgent===")
        task = context.get("task")
        search_results = context.get("search_results")

        if not task:
            raise ValueError("子答案生成必须提供子任务")

        if not search_results:
            logging.warning("未提供搜索结果，将生成基于任务的推测性回答")
            search_results = {"results": []}

        # 构建提示
        prompt = self.build_prompt(context)

        # 调用LLM生成子答案
        functions = [{
            "name": "generate_sub_answer",
            "description": "分析搜索结果并生成对子任务的回答",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "对子任务的综合回答"
                    }
                },
                "required": ["answer"]
            }
        }]

        result = self.llm.generate_with_function_calling(prompt, functions)

        if "name" not in result or result["name"] != "generate_sub_answer":
            # 处理LLM未返回预期函数调用的情况
            logging.error("LLM未返回子答案生成函数调用")
            error_response = {
                "error": "生成子答案失败",
                "answer": "无法基于提供的搜索结果生成回答。"
            }
            return error_response

        logger.info(f"生成sub answer:")
        logger.info(result["arguments"].get("answer", ""))
        # 返回子答案
        return result["arguments"]

    def build_prompt(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        构建LLM提示。

        Args:
            context: 输入上下文

        Returns:
            LLM提示消息列表
        """
        messages = []

        # 系统提示
        system_prompt = self._get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # 构建用户提示
        task = context.get("task", {})
        task_title = task.get("title", "")
        task_description = task.get("description", "")
        search_results = context.get("search_results", {})

        user_prompt = f"我需要基于以下搜索结果，为研究子任务生成回答:\n\n"
        user_prompt += f"## 任务标题\n{task_title}\n\n"
        user_prompt += f"## 任务描述\n{task_description}\n\n"

        # 格式化搜索结果
        user_prompt += self._format_search_results(search_results)

        user_prompt += "\n\n请基于上述搜索结果，生成对该子任务的综合回答。"

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _get_system_prompt(self) -> str:
        """
        获取系统提示。

        Returns:
            系统提示字符串
        """
        formatted_date = datetime.now().strftime("%Y-%m-%d")
        return f"""now date:{formatted_date}\n你是一位资深研究分析师，擅长从搜索结果中提取和综合信息，生成对特定问题的深入回答。

你的职责是：
1. 分析提供的搜索结果
2. 提取与子任务相关的关键信息
3. 综合这些信息生成连贯、全面的回答

分析原则：
- 保持客观，以事实和证据为基础
- 整合多个来源的信息，寻找共识和分歧
- 适当引用信息来源，确保可追溯性
- 区分确定的事实和推测的内容
- 注意信息的时效性和相关性

你应该提供一个综合性的回答，直接针对子任务。你的回答应该全面但简洁，涵盖搜索结果中的主要信息。"""

    def _format_search_results(self, search_results: Dict[str, Any]) -> str:
        """
        格式化搜索结果为可读文本。

        Args:
            search_results: 搜索结果字典

        Returns:
            格式化的搜索结果文本
        """
        formatted_text = "## 搜索结果\n\n"

        results = search_results.get("results", [])
        if not results:
            return formatted_text + "没有提供搜索结果。"

        for i, result in enumerate(results):
            formatted_text += f"### 结果 {i + 1}\n"

            # 添加标题（如果有）
            if "title" in result:
                formatted_text += f"**标题**: {result['title']}\n\n"

            # 添加网址（如果有）
            if "url" in result:
                formatted_text += f"**来源**: {result['url']}\n\n"

            # 添加内容（如果有）
            if "content" in result:
                content = result["content"]
                # 如果内容太长，截断它
                if len(content) > 200:
                    content = content[:200] + "...(内容已截断)"
                formatted_text += f"**内容**:\n{content}\n\n"
            if "snippet" in result:
                formatted_text += f"**摘要**:\n{result['snippet']}\n\n"

            formatted_text += "---\n\n"

        return formatted_text

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应为结构化输出。

        Args:
            response: LLM响应文本

        Returns:
            解析后的结构化输出
        """
        # 由于使用function calling，此方法在当前实现中不会被调用
        # 保留此方法以符合Agent抽象基类要求
        try:
            parsed = json.loads(response)
            return parsed
        except:
            logging.error(f"解析子答案响应失败: {response}")
            return {
                "error": "解析响应失败",
                "answer": ""
            }