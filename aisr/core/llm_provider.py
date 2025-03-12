"""
LLM提供者模块，负责与不同的LLM服务提供商集成。
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

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


class LLMProvider:
    """
    LLM服务提供者类，统一管理不同的LLM API调用。

    支持多种LLM提供者（目前支持Anthropic Claude和OpenAI）。
    支持单轮和多轮对话模式。
    """

    def __init__(self, provider: str = "anthropic", api_key: Optional[str] = None, model: Optional[str] = None):
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
        self.models = {
            "anthropic": {
                "default": "claude-3-7-sonnet-20250219",
                "fast": "claude-3-5-sonnet-20240620",
                "powerful": "claude-3-opus-20240229"
            },
            "openai": {
                "default": "gpt-4o-mini",
                "fast": "gpt-3.5-turbo",
                "powerful": "gpt-4o"
            }
        }

        # 根据输入或默认值设置模型
        if model:
            self.model = model
        else:
            self.model = self.models.get(self.provider, {}).get("default")

        # 初始化客户端
        self._initialize_client()

        logging.info(f"已初始化LLM提供者: {self.provider}, 模型: {self.model}")

    def _initialize_client(self):
        """初始化适当的LLM客户端。"""
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

            self.client = openai.OpenAI(api_key=self.api_key)

        else:
            raise ValueError(f"不支持的提供者: {self.provider}。支持的提供者: anthropic, openai")

    def generate(self, prompt: Union[str, List[Dict[str, Any]]], temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """
        生成LLM响应。支持单轮或多轮对话。

        Args:
            prompt: 提示文本或消息列表。
                   如果是字符串，将被视为单轮对话中的用户消息。
                   如果是列表，将被视为多轮对话的完整历史。
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成的token数

        Returns:
            LLM生成的文本
        """
        try:
            # 根据prompt类型准备消息
            messages = self._prepare_messages(prompt)

            message_count = len(messages)
            logging.debug(f"发送{message_count}条消息到{self.provider}")

            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages
                )
                result = response.content[0].text

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                result = response.choices[0].message.content

            else:
                raise ValueError(f"无法生成: 不支持的提供者 {self.provider}")

            logging.debug(f"收到LLM响应，长度: {len(result)}")
            return result

        except Exception as e:
            logging.error(f"LLM生成错误: {str(e)}")
            raise RuntimeError(f"LLM生成失败: {str(e)}")

    def generate_with_function_calling(self, prompt: Union[str, List[Dict[str, Any]]], functions: List[Dict[str, Any]], temperature: float = 0.2, max_tokens: int = 4000) -> Dict[str, Any]:
        """
        使用函数调用功能生成结构化输出。支持单轮或多轮对话。

        Args:
            prompt: 提示文本或消息列表
            functions: 函数定义列表
            temperature: 温度参数
            max_tokens: 最大生成的token数

        Returns:
            结构化的函数调用结果
        """
        try:
            # 根据prompt类型准备消息
            messages = self._prepare_messages(prompt)

            logging.debug(f"发送带函数定义的提示，函数数量: {len(functions)}")

            if self.provider == "anthropic":
                # 转换为Anthropic工具格式
                tools = []
                for func in functions:
                    tools.append({
                        "name": func.get("name", "default_tool"),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {})
                    })

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                    tools=tools
                )

                # 提取工具使用
                result = None
                for content in response.content:
                    if hasattr(content, 'type') and content.type == "tool_use":
                        result = {
                            "name": content.name,
                            "arguments": content.input
                        }
                        break

                if not result:
                    # 如果没有工具使用，使用文本响应
                    return {"text": response.content[0].text}

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=functions,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                if response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    result = {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    }
                else:
                    # 如果没有工具调用，使用文本响应
                    return {"text": response.choices[0].message.content}

            else:
                raise ValueError(f"无法生成: 不支持的提供者 {self.provider}")

            logging.debug(f"收到结构化响应: {result}")
            return result

        except Exception as e:
            logging.error(f"结构化生成错误: {str(e)}")
            return {"error": str(e)}

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
            # 验证消息列表格式
            for msg in prompt:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise ValueError("消息列表格式不正确，每条消息必须包含'role'和'content'字段")

                # 检查role是否有效
                if msg["role"] not in ["user", "assistant", "system"]:
                    raise ValueError(f"无效的消息角色: {msg['role']}，必须是'user'、'assistant'或'system'")

            return prompt

        else:
            raise TypeError("prompt必须是字符串或消息列表")

    def create_conversation(self, system_prompt: Optional[str] = None) -> 'Conversation':
        """
        创建新的对话会话。

        Args:
            system_prompt: 可选的系统提示

        Returns:
            新的对话会话对象
        """
        return Conversation(self, system_prompt)


class Conversation:
    """
    管理与LLM的持续多轮对话。
    """

    def __init__(self, llm_provider: LLMProvider, system_prompt: Optional[str] = None):
        """
        初始化对话。

        Args:
            llm_provider: LLM提供者实例
            system_prompt: 可选的系统提示
        """
        self.llm = llm_provider
        self.messages = []

        # 如果有系统提示，添加到消息中
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 对话元数据
        self.created_at = datetime.now().isoformat()
        self.turn_count = 0

    def add_message(self, role: str, content: str) -> None:
        """
        向对话添加消息。

        Args:
            role: 消息角色 ("user", "assistant", "system")
            content: 消息内容
        """
        if role not in ["user", "assistant", "system"]:
            raise ValueError(f"无效的消息角色: {role}")

        self.messages.append({
            "role": role,
            "content": content
        })

        # 只有用户和助手消息计入对话轮次
        if role in ["user", "assistant"]:
            self.turn_count += 1

    def get_user_input(self, user_input: str) -> str:
        """
        添加用户输入并获取助手回复。

        Args:
            user_input: 用户输入文本

        Returns:
            助手回复文本
        """
        # 添加用户消息
        self.add_message("user", user_input)

        # 获取助手回复
        response = self.llm.generate(self.messages)

        # 添加助手回复到对话历史
        self.add_message("assistant", response)

        return response

    def get_function_call(self, user_input: str, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        添加用户输入并获取函数调用结果。

        Args:
            user_input: 用户输入文本
            functions: 函数定义列表

        Returns:
            函数调用结果
        """
        # 添加用户消息
        self.add_message("user", user_input)

        # 获取函数调用结果
        result = self.llm.generate_with_function_calling(self.messages, functions)

        # 如果是文本响应，添加到对话历史
        if "text" in result:
            self.add_message("assistant", result["text"])

        return result

    def clear_history(self, keep_system_prompt: bool = True) -> None:
        """
        清除对话历史。

        Args:
            keep_system_prompt: 是否保留系统提示
        """
        if keep_system_prompt and self.messages and self.messages[0]["role"] == "system":
            system_message = self.messages[0]
            self.messages = [system_message]
        else:
            self.messages = []

        self.turn_count = 0

    def get_messages(self) -> List[Dict[str, Any]]:
        """获取所有消息。"""
        return self.messages.copy()

    def get_last_turn(self) -> tuple:
        """
        获取最后一轮对话（用户输入和助手回复）。

        Returns:
            (用户输入, 助手回复)元组，如果没有完整的对话轮次则返回(None, None)
        """
        user_msg = None
        assistant_msg = None

        # 从后向前查找
        for i in range(len(self.messages)-1, -1, -1):
            msg = self.messages[i]
            if msg["role"] == "assistant" and assistant_msg is None:
                assistant_msg = msg["content"]
            elif msg["role"] == "user" and assistant_msg is not None and user_msg is None:
                user_msg = msg["content"]
                break

        return (user_msg, assistant_msg)