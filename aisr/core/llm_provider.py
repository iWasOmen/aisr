"""
LLM提供者模块，负责与LLM服务提供商集成。
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union

# 可选导入，根据用户配置决定
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from aisr.utils import logging_utils
logger = logging_utils.get_logger(__name__, color="red")

class LLMProvider:
    """
    LLM服务提供者类，统一管理LLM API调用。

    支持Anthropic Claude和OpenAI模型。
    """

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化LLM提供者。

        Args:
            provider: LLM提供者名称，"anthropic"或"openai"
            api_key: API密钥
            model: 模型名称
        """
        self.provider = provider.lower()
        self.api_key = api_key

        # 设置默认模型
        default_models = {
            "anthropic": "claude-3-7-sonnet-20250219",
            "openai": "gpt-4o-2024-08-06"
        }

        # 根据输入或默认值设置模型
        self.model = model if model else default_models.get(self.provider, "claude-3-7-sonnet-20250219")

        # 初始化客户端
        self._initialize_client()

    def _initialize_client(self):
        """初始化LLM客户端。"""
        if self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic 库未安装。请使用 pip install anthropic 安装。")
            if not self.api_key:
                raise ValueError("使用Anthropic需要提供API密钥")

            self.client = anthropic.Anthropic(api_key=self.api_key)

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai 库未安装。请使用 pip install openai 安装。")
            if not self.api_key:
                raise ValueError("使用OpenAI需要提供API密钥")

            self.client = openai.OpenAI(api_key=self.api_key,base_url="http://rerverseapi.workergpt.cn/v1")

        else:
            raise ValueError(f"不支持的提供者: {self.provider}。支持的提供者: anthropic, openai")

    def generate(self, prompt: Union[str, List[Dict[str, Any]]], temperature: float = 0.7, max_tokens: int = 8000) -> str:
        """
        生成LLM响应。

        Args:
            prompt: 提示文本或消息列表
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成的token数

        Returns:
            LLM生成的文本
        """
        logger.info(f"=== API 输入 (generate) ===")
        # 根据prompt类型准备消息
        messages = self._prepare_messages(prompt)
        logger.info(f"消息:{messages}")

        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            logger.info(f"=== API 输出 (generate) ===")
            logger.info(f"原始输出内容: {response}")
            return response.content[0].text

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"=== API 输出 (generate) ===")
            logger.info(f"原始输出内容: {response}")
            return response.choices[0].message.content

        else:
            raise ValueError(f"无法生成: 不支持的提供者 {self.provider}")

    def generate_with_function_calling(self, prompt: Union[str, List[Dict[str, Any]]], functions: List[Dict[str, Any]],
                                       temperature: float = 0.2, max_tokens: int = 8000) -> Dict[str, Any]:
        """
        使用函数调用功能生成结构化输出。

        Args:
            prompt: 提示文本或消息列表
            functions: 函数定义列表（统一使用简化格式，内部会根据提供者转换）
            temperature: 温度参数
            max_tokens: 最大生成的token数

        Returns:
            结构化的函数调用结果
        """
        logger.info(f"=== API 输入 (generate_with_function_calling) ===")
        # 根据prompt类型准备消息
        messages = self._prepare_messages(prompt)
        logger.info(f"消息:{messages}")

        if self.provider == "anthropic":
            # 转换为Anthropic工具格式
            tools = self._convert_to_anthropic_format(functions)
            logger.info(f"工具:{tools}")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                tools=tools
            )

            # 提取工具使用
            for content in response.content:
                if hasattr(content, 'type') and content.type == "tool_use":
                    return {
                        "name": content.name,
                        "arguments": content.input
                    }

            # 如果没有工具使用，使用文本响应
            return {"text": response.content[0].text}

        elif self.provider == "openai":
            # 转换为OpenAI工具格式
            tools = self._convert_to_openai_format(functions)
            logger.info(f"工具:{tools}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice="required"
            )
            logger.info(f"=== API 输出 (generate_with_function_calling) ===")
            logger.info(f"原始输出内容: {response}")

            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                return {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                }
            else:
                # 如果没有工具调用，使用文本响应
                return {"text": response.choices[0].message.content}

        else:
            raise ValueError(f"无法生成: 不支持的提供者 {self.provider}")

    def _prepare_messages(self, prompt: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        将输入提示转换为标准消息格式。

        Args:
            prompt: 提示文本或消息列表

        Returns:
            标准格式的消息列表
        """
        if isinstance(prompt, str):
            # 单条用户消息
            return [{"role": "user", "content": prompt}]

        elif isinstance(prompt, list):
            # 消息列表
            return prompt

        else:
            raise TypeError("prompt必须是字符串或消息列表")

    def _convert_to_anthropic_format(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将统一格式的函数定义转换为Anthropic工具格式。

        Args:
            functions: 统一格式的函数定义列表

        Returns:
            Anthropic格式的工具列表
        """
        tools = []

        for func in functions:
            # 检查是否已经是Anthropic格式
            if "name" in func and "description" in func and "parameters" in func:
                tools.append({
                    "name": func.get("name", "default_tool"),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
            # 如果是OpenAI格式，则提取相关信息
            elif "type" in func and func["type"] == "function" and "function" in func:
                function_data = func["function"]
                tools.append({
                    "name": function_data.get("name", "default_tool"),
                    "description": function_data.get("description", ""),
                    "input_schema": function_data.get("parameters", {})
                })

        return tools

    def _convert_to_openai_format(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将统一格式的函数定义转换为OpenAI工具格式。

        Args:
            functions: 统一格式的函数定义列表

        Returns:
            OpenAI格式的工具列表
        """
        tools = []

        for func in functions:
            # 检查是否已经是OpenAI格式
            if "type" in func and func["type"] == "function" and "function" in func:
                tools.append(func)
            # 如果是简化格式，则转换为OpenAI格式
            elif "name" in func and "description" in func:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                        "strict": True
                    }
                })

        return tools