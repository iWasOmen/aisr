"""
搜索规划代理模块，负责为特定子任务生成搜索查询列表。

作为中层循环的关键组件，负责确定如何最有效地搜索特定子任务的信息，
可基于前序搜索结果调整策略。
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
import json

from aisr.agents.base import Agent
from aisr.utils import logging_utils
logger = logging_utils.get_logger(__name__, color="green")

class SearchPlanAgent(Agent):
    """
    搜索规划代理，为子任务生成有效的搜索查询。

    输入：
    - 当前子任务
    - 可选的搜索历史（包含前序搜索计划和结果）

    输出：
    - 搜索查询列表
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行搜索规划，为子任务生成搜索查询列表。

        Args:
            context: 包含执行所需输入的字典
                - task: 当前子任务
                - previous_search_plans: 可选，前序搜索计划列表
                - previous_sub_answer: 可选，前序子答案

        Returns:
            包含搜索查询列表的字典
        """
        logger.info(f"===SearchPlanAgent===")
        task = context.get("task")
        if not task:
            raise ValueError("搜索规划必须提供子任务")

        # 构建提示
        prompt = self.build_prompt(context)

        # 调用LLM生成搜索规划
        functions = [{
            "name": "search_planning",
            "description": "为子任务生成有效的搜索查询列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "description": "搜索查询列表",
                        "items": {
                            "type": "string",
                            "description": "搜索查询"
                        }
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "查询生成的推理过程"
                    }
                },
                "required": ["queries", "reasoning"]
            }
        }]

        result = self.llm.generate_with_function_calling(prompt, functions)

        if "name" not in result or result["name"] != "search_planning":
            # 处理LLM未返回预期函数调用的情况
            logging.error("LLM未返回搜索规划函数调用")
            error_response = {
                "error": "生成搜索规划失败",
                "queries": []
            }
            return error_response

        logger.info(f"生成search queries:")
        logger.info(result["arguments"].get("queries", ""))
        logger.info(f"生成reasoning:")
        logger.info(result["arguments"].get("reasoning", ""))
        # 返回搜索规划结果
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

        # 当前任务
        task = context.get("task", {})
        task_title = task.get("title", "")
        task_description = task.get("description", "")

        user_prompt = f"我需要为以下研究子任务生成搜索查询:\n\n"
        user_prompt += f"## 任务标题\n{task_title}\n\n"
        user_prompt += f"## 任务描述\n{task_description}\n\n"
        user_prompt += "请生成有效的搜索查询列表，以帮助我找到相关信息。"

        # 添加历史信息（如果有）
        history_context = self._format_history_context(context)
        if history_context:
            user_prompt += history_context

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _get_system_prompt(self) -> str:
        """
        获取系统提示。

        Returns:
            系统提示字符串
        """
        formatted_date = datetime.now().strftime("%Y-%m-%d")
        return f"""now date:{formatted_date}\n你是一位资深搜索专家，擅长将研究任务转化为有效的搜索查询。

你的职责是：
1. 分析给定的研究子任务
2. 生成1-3个有效的搜索查询
3. 确保查询能够覆盖任务的关键方面
4. 考虑如何最大化相关信息的检索效果

查询设计原则：
- 使用精确的关键词和术语
- 尝试不同的表述方式以覆盖更多相关内容
- 考虑使用专业术语和同义词
- 适当添加限定词以提高精确度
- 从一般到具体，确保覆盖不同深度的信息
- 如果有前序搜索结果，考虑如何改进查询

请注意：
- 生成的查询应该简洁明了，每个查询专注于一个明确的信息需求
- 避免过于宽泛或过于狭窄的查询
- 不要包含特殊搜索运算符，如引号、加号或减号
- 提供详细的推理过程，解释你的查询设计思路

你的输出将直接用于网络搜索，因此请确保查询有效且针对性强。"""

    def _format_history_context(self, context: Dict[str, Any]) -> str:
        """
        格式化历史上下文信息。

        Args:
            context: 上下文

        Returns:
            格式化的历史上下文字符串，如果没有历史信息则返回空字符串
        """
        history_text = ""

        # 检查是否有历史信息
        has_history = (
            context.get("previous_search_plans") or
            context.get("previous_sub_answer")
        )

        if not has_history:
            return ""

        history_text = "\n\n## 历史搜索信息\n"

        # 添加前序搜索计划
        previous_search_plans = context.get("previous_search_plans", [])
        if previous_search_plans:
            history_text += "\n### 前序搜索查询\n"
            for i, plan in enumerate(previous_search_plans):
                history_text += f"\n尝试 {i+1}:\n"

                queries = plan.get("queries", [])
                if queries:
                    for query in queries:
                        history_text += f"- {query}\n"

        # 添加前序子答案
        previous_sub_answer = context.get("previous_sub_answer")
        if previous_sub_answer:
            history_text += "\n### 前序搜索结果\n"
            # 直接使用子答案作为文本，不假设特定的结构
            history_text += f"{str(previous_sub_answer)[:300]}...\n"

        history_text += "\n\n请基于上述历史信息和当前任务，设计更有效的搜索查询。如果前序查询存在不足，请加以改进；如果前序结果已经包含一些有用信息，请设计查询以获取更深入或补充的信息。\n"

        return history_text

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
            logging.error(f"解析搜索规划响应失败: {response}")
            return {
                "error": "解析响应失败",
                "queries": []
            }