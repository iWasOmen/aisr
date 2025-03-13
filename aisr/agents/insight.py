"""
洞察生成代理模块，负责分析已有子答案并生成研究洞察。

作为外层循环的反馈组件，负责评估研究进展并指导后续任务规划。
"""

import logging
from typing import Dict, Any, List
import json

from aisr.agents.base import Agent
from aisr.utils import logging_utils
logger = logging_utils.get_logger(__name__, color="green")

class InsightAgent(Agent):
    """
    洞察生成代理，分析已有子答案并生成研究洞察。

    输入：
    - 用户查询
    - 当前未执行的任务计划
    - 已完成任务的子答案

    输出：
    - 对当前研究进展的洞察
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行子答案分析，生成研究洞察。

        Args:
            context: 包含执行所需输入的字典
                - query: 用户查询
                - unexecuted_plan: 当前未执行的任务计划
                - sub_answers: 已完成任务的子答案字典

        Returns:
            包含研究洞察的字典
        """
        logger.info(f"===InsightAgent===")
        query = context.get("query")
        sub_answers = context.get("sub_answers", {})

        if not query:
            raise ValueError("洞察生成必须提供用户查询")

        if not sub_answers:
            logger.warning("未提供任何子答案，将生成基于查询的初步洞察")

        # 构建提示
        prompt = self.build_prompt(context)

        # 调用LLM生成洞察
        functions = [{
            "name": "generate_insight",
            "description": "分析子答案并生成对研究进展的洞察",
            "parameters": {
                "type": "object",
                "properties": {
                    "insight": {
                        "type": "string",
                        "description": "对当前研究进展的综合洞察"
                    }
                },
                "required": ["insight"]
            }
        }]

        result = self.llm.generate_with_function_calling(prompt, functions)

        if "name" not in result or result["name"] != "generate_insight":
            # 处理LLM未返回预期函数调用的情况
            logger.error("LLM未返回洞察生成函数调用")
            error_response = {
                "error": "生成研究洞察失败",
                "insight": "无法基于提供的子答案生成洞察。"
            }
            return error_response

        logger.info(f"生成insight:")
        logger.info(result["arguments"].get("insight", ""))
        # 返回洞察
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
        query = context.get("query", "")
        sub_answers = context.get("sub_answers", {})
        unexecuted_plan = context.get("unexecuted_plan", {})

        user_prompt = f"我需要基于以下信息，生成研究洞察:\n\n"
        user_prompt += f"## 原始查询\n{query}\n\n"

        # 添加已完成任务的子答案
        if sub_answers:
            user_prompt += "## 已完成的子任务答案\n\n"
            for task_id, answer in sub_answers.items():
                user_prompt += f"### 任务: {task_id}\n"
                user_prompt += f"{str(answer)}\n\n"

        # 添加未执行的任务计划
        if unexecuted_plan:
            user_prompt += "## 未执行的任务计划\n\n"
            sub_tasks = unexecuted_plan.get("sub_tasks", [])
            for task in sub_tasks:
                user_prompt += f"- {task.get('title', '')}: {task.get('description', '')}\n"

        user_prompt += "\n\n请基于上述信息，分析当前研究进展并生成洞察。"

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _get_system_prompt(self) -> str:
        """
        获取系统提示。

        Returns:
            系统提示字符串
        """
        return """你是一位资深研究顾问，擅长分析研究进展并提供战略性洞察。

你的职责是：
1. 分析已完成任务的答案和剩余的研究计划
2. 识别关键的发现、模式和矛盾
3. 评估当前研究进展
4. 提出对研究方向的建议

分析思路：
- 识别已有答案中的共同主题和关键发现
- 注意信息间的冲突或不一致
- 找出研究中的知识缺口
- 考虑原始查询的目标是否正在得到有效解答
- 评估剩余任务计划是否仍然合理，或是否需要调整

你的洞察应该是全面、深入和有建设性的，帮助指导研究的下一步。"""

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
            logger.error(f"解析洞察响应失败: {response}")
            return {
                "error": "解析响应失败",
                "insight": ""
            }