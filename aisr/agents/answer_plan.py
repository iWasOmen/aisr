"""
答案规划代理模块，负责规划最终答案的结构大纲。

作为研究结果综合阶段的组件，负责设计最终答案的结构框架。
"""

import logging
from typing import Dict, Any, List
import json

from aisr.agents.base import Agent
from aisr.utils import logging_utils
logger = logging_utils.get_logger(__name__, color="green")

class AnswerPlanAgent(Agent):
    """
    答案规划代理，为最终答案设计结构大纲。

    输入：
    - 用户查询
    - 所有子任务及其答案

    输出：
    - 最终答案的结构大纲
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行答案规划，生成最终答案的结构大纲。

        Args:
            context: 包含执行所需输入的字典
                - query: 用户查询
                - sub_answers: 所有子答案的字典 {task_id: answer}

        Returns:
            包含答案大纲的字典
        """
        logger.info(f"===AnswerPlanAgent===")
        query = context.get("query")
        sub_answers = context.get("sub_answers", {})

        if not query:
            raise ValueError("答案规划必须提供用户查询")

        if not sub_answers:
            logger.warning("未提供任何子答案，将基于查询生成通用大纲")

        # 构建提示
        prompt = self.build_prompt(context)

        # 调用LLM生成答案大纲
        functions = [{
            "name": "generate_answer_outline",
            "description": "为最终答案生成结构化大纲",
            "parameters": {
                "type": "object",
                "properties": {
                    "outline": {
                        "type": "string",
                        "description": "最终答案的结构大纲"
                    }
                },
                "required": ["outline"]
            }
        }]

        result = self.llm.generate_with_function_calling(prompt, functions)

        if "name" not in result or result["name"] != "generate_answer_outline":
            # 处理LLM未返回预期函数调用的情况
            logger.error("LLM未返回答案大纲生成函数调用")
            error_response = {
                "error": "生成答案大纲失败",
                "outline": "无法基于提供的信息生成答案大纲。"
            }
            return error_response

        # 返回答案大纲
        logger.info(f"生成大纲:")
        logger.info(result["arguments"].get("outline", ""))
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

        user_prompt = f"我需要基于以下信息，为研究问题的最终答案设计结构大纲:\n\n"
        user_prompt += f"## 原始查询\n{query}\n\n"

        # 添加所有子答案
        if sub_answers:
            user_prompt += "## 子任务答案\n\n"
            for task_id, answer in sub_answers.items():
                user_prompt += f"### 任务: {task_id}\n"
                user_prompt += f"{str(answer)}\n\n"

        user_prompt += "\n\n请基于上述信息，设计一个清晰、结构化的大纲，用于组织最终答案。"

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _get_system_prompt(self) -> str:
        """
        获取系统提示。

        Returns:
            系统提示字符串
        """
        return """你是一位资深研究报告编辑，擅长设计清晰、有条理的报告结构。

你的职责是：
1. 分析研究问题和所有已收集的信息
2. 设计一个逻辑清晰的结构大纲
3. 确保大纲能够全面涵盖所有关键发现
4. 为最终答案提供一个易于理解的框架

设计原则：
- 从总体到细节的结构安排
- 逻辑流畅，各部分之间有自然过渡
- 优先处理核心问题和关键发现
- 确保全面性，不遗漏重要信息
- 适当分组相关主题
- 考虑读者视角，确保易于理解

大纲格式：
- 使用标准的大纲格式，包括主要部分和子部分
- 每个部分应有简短描述，说明包含的内容
- 可以使用编号或层级格式

你的大纲将作为生成最终答案的框架，因此应该清晰、全面且结构合理。"""

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
            logger.error(f"解析答案大纲响应失败: {response}")
            return {
                "error": "解析响应失败",
                "outline": ""
            }