"""
LLM提供者模块，负责与不同的LLM服务提供商集成。
"""

import json
import logging
from typing import Dict, Any, List, Optional

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

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """
        生成LLM响应。

        Args:
            prompt: 提示文本
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成的token数

        Returns:
            LLM生成的文本
        """
        try:
            logging.debug(f"发送提示到{self.provider}，长度: {len(prompt)}")

            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
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

    def generate_with_function_calling(self, prompt: str, functions: List[Dict[str, Any]], temperature: float = 0.2,
                                       max_tokens: int = 4000) -> Dict[str, Any]:
        """
        使用函数调用功能生成结构化输出。

        Args:
            prompt: 提示文本
            functions: 函数定义列表
            temperature: 温度参数
            max_tokens: 最大生成的token数

        Returns:
            结构化的函数调用结果
        """
        try:
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
                    messages=[{"role": "user", "content": prompt}],
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
                    messages=[{"role": "user", "content": prompt}],
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