"""
LLM提供者模块的测试。
"""

import unittest
import json
from aisr.core.llm_provider import LLMProvider
from tests import API_KEY


class TestLLMProvider(unittest.TestCase):
    """LLMProvider类的测试用例"""

    def setUp(self):
        """测试前设置"""
        #anthropic
        self.llm = LLMProvider(provider="openai", api_key=API_KEY)

    def test_generate(self):
        """测试文本生成功能"""
        prompt = "用一句话解释什么是人工智能"

        print("\n=== 测试LLM文本生成 ===")
        print(f"提示: {prompt}")

        response = self.llm.generate(prompt, temperature=0.7, max_tokens=100)

        print(f"响应: {response}")

        # 验证响应不为空且是字符串
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_function_calling(self):
        """测试函数调用功能"""
        prompt = "北京天气怎么样"

        functions = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": [
                        "location"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]

        print("\n=== 测试LLM函数调用 ===")
        print(f"提示: {prompt}")
        print(f"函数: get_weather")

        result = self.llm.generate_with_function_calling(prompt, functions, temperature=0.2)

        print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

        # 验证返回了函数调用结果
        self.assertIsInstance(result, dict)
        self.assertTrue("name" in result or "text" in result)
        if "name" in result:
            self.assertEqual(result["name"], "get_weather")
            self.assertIn("arguments", result)


if __name__ == '__main__':
    unittest.main()
