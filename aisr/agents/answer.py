"""
答案生成代理模块，负责综合所有子答案，生成最终答案。

作为研究过程的最后一步，负责将所有收集的信息整合为连贯、全面的最终答案。
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
import json

from aisr.agents.base import Agent
from aisr.utils import logging_utils
logger = logging_utils.get_logger(__name__, color="green")

class AnswerAgent(Agent):
    """
    答案生成代理，综合所有子答案，生成最终答案。

    输入：
    - 用户查询
    - 所有子任务及其答案
    - 答案结构大纲

    输出：
    - 最终综合答案
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行答案生成，综合所有信息为最终答案。

        Args:
            context: 包含执行所需输入的字典
                - query: 用户查询
                - sub_answers: 所有子答案的字典 {task_id: answer}
                - outline: 答案结构大纲

        Returns:
            包含最终答案的字典
        """
        logger.info(f"===AnswerAgent===")
        query = context.get("query")
        sub_answers = context.get("sub_answers", {})
        outline = context.get("outline", "")

        if not query:
            raise ValueError("答案生成必须提供用户查询")

        if not sub_answers:
            logging.warning("未提供任何子答案，将基于查询生成简单答案")

        if not outline:
            logging.warning("未提供答案大纲，将自行组织答案结构")

        # 构建提示
        prompt = self.build_prompt(context)

        # 调用LLM生成最终答案
        functions = [{
            "name": "generate_final_answer",
            "description": "综合所有子答案，生成最终答案",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "最终综合答案"
                    }
                },
                "required": ["answer"]
            }
        }]

        result = self.llm.generate_with_function_calling(prompt, functions)

        if "name" not in result or result["name"] != "generate_final_answer":
            # 处理LLM未返回预期函数调用的情况
            logging.error("LLM未返回最终答案生成函数调用")
            error_response = {
                "error": "生成最终答案失败",
                "answer": "无法基于提供的信息生成综合答案。"
            }
            return error_response

        # 返回最终答案
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
        outline = context.get("outline", "")

        user_prompt = f"我需要基于以下信息，生成对原始查询的综合答案:\n\n"
        user_prompt += f"## 原始查询\n{query}\n\n"

        # 添加答案大纲（如果有）
        if outline:
            user_prompt += f"## 答案结构大纲\n{outline}\n\n"

        # 添加所有子答案
        if sub_answers:
            user_prompt += "## 子任务答案\n\n"
            for task_id, answer in sub_answers.items():
                user_prompt += f"### 任务: {task_id}\n"
                user_prompt += f"{str(answer)}\n\n"

        user_prompt += "\n\n请基于上述信息，生成一个全面、连贯的最终答案，回应原始查询。"
        if outline:
            user_prompt += " 请按照提供的大纲组织答案结构。"

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _get_system_prompt(self) -> str:
        """
        获取系统提示。

        Returns:
            系统提示字符串
        """
        formatted_date = datetime.now().strftime("%Y-%m-%d")
        return f"""now date:{formatted_date}\n你是一位资深研究撰写专家，擅长整合多源信息，生成全面、连贯的研究报告。

你的职责是：
1. 分析原始研究问题
2. 综合所有子任务的答案
3. 按照提供的结构大纲(如果有)组织内容
4. 生成一个连贯、全面的最终答案

撰写原则：
- 保持客观，以事实和证据为基础
- 整合所有来源的信息，解决可能存在的矛盾
- 确保最终答案直接回应原始查询
- 重点突出最重要的发现和洞见
- 结构清晰，逻辑连贯，行文流畅
- 适当引用关键信息来源
- 保持适当的详细程度，既全面又简洁

你的最终答案应该是一个完整、独立的文档，能够提供对原始查询的全面解答，无需读者查看中间过程或原始资料。"""

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
            logging.error(f"解析最终答案响应失败: {response}")
            return {
                "error": "解析响应失败",
                "answer": ""
            }