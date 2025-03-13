"""
子答案生成代理模块的测试。
"""

import unittest
from aisr.core.llm_provider import LLMProvider
from aisr.agents.sub_answer import SubAnswerAgent
from tests import API_KEY, TEST_TASK, TEST_SEARCH_RESULTS


class TestSubAnswerAgent(unittest.TestCase):
    """SubAnswerAgent类的测试用例"""

    def setUp(self):
        """测试前设置anthropic"""
        llm = LLMProvider(provider="openai", api_key=API_KEY)
        self.agent = SubAnswerAgent(llm, memory=None)

    def test_sub_answer_generation(self):
        """测试子答案生成功能"""
        context = {
            "task": TEST_TASK,
            "search_results": TEST_SEARCH_RESULTS
        }

        print("\n=== 测试子答案生成 ===")
        print(f"任务: {TEST_TASK['title']}")
        print(f"搜索结果数: {TEST_SEARCH_RESULTS['result_count']}")

        result = self.agent.execute(context)

        print(f"生成的子答案:")
        print(result.get("answer", "无答案"))

        # 验证结果包含答案
        self.assertIn("answer", result)
        self.assertIsInstance(result["answer"], str)
        self.assertTrue(len(result["answer"]) > 0)

    def test_sub_answer_generation_with_real_search(self):
        """测试使用真实搜索结果的子答案生成功能"""
        try:
            # 导入web_search函数
            from aisr.tools.search_tools import web_search

            # 执行搜索
            query = "深度学习与传统机器学习的区别"
            print("\n=== 测试使用真实搜索结果的子答案生成 ===")
            print(f"执行搜索，查询: {query}")

            search_results = web_search(query, max_results=2)  # 限制为2个结果以加快测试

            print(f"搜索结果数: {search_results.get('result_count', 0)}")

            # 生成子答案
            context = {
                "task": TEST_TASK,
                "search_results": search_results
            }

            result = self.agent.execute(context)

            print(f"生成的子答案:")
            print(result.get("answer", "无答案"))

            # 验证结果包含答案
            self.assertIn("answer", result)
            self.assertIsInstance(result["answer"], str)
            self.assertTrue(len(result["answer"]) > 0)

        except ImportError as e:
            print("无法导入搜索工具，跳过实际搜索测试")
            print(e)
            self.skipTest("搜索工具不可用")


if __name__ == '__main__':
    unittest.main()