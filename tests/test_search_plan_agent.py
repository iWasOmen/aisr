"""
搜索规划代理模块的测试。
"""

import unittest
from aisr.core.llm_provider import LLMProvider
from aisr.agents.search_plan import SearchPlanAgent
from tests import TEST_TASK
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TestSearchPlanAgent(unittest.TestCase):
    """SearchPlanAgent类的测试用例"""

    def setUp(self):
        """测试前设置anthropic"""
        llm = LLMProvider()
        self.agent = SearchPlanAgent(llm, memory=None)

    def test_search_planning(self):
        """测试搜索规划功能"""
        context = {"task": TEST_TASK}

        print("\n=== 测试搜索规划 ===")
        print(f"任务: {TEST_TASK['title']}")
        print(f"描述: {TEST_TASK['description']}")

        result = self.agent.execute(context)

        print(f"生成的搜索查询:")
        for i, query in enumerate(result.get("queries", [])):
            print(f"{i + 1}. {query}")

        print(f"\n推理过程:")
        print(result.get("reasoning", "无推理过程"))

        # 验证结果包含查询列表
        self.assertIn("queries", result)
        self.assertIsInstance(result["queries"], list)
        self.assertTrue(len(result["queries"]) > 0)

    def test_search_planning_with_previous_results(self):
        """测试带有前序结果的搜索规划功能"""
        # 创建模拟的前序搜索计划
        previous_search_plans = [
            {
                "queries": [
                    "深度学习基础概念",
                    "深度学习工作原理"
                ]
            }
        ]

        # 创建模拟的前序子答案
        previous_sub_answer = "深度学习是机器学习的一个分支，使用多层神经网络进行特征学习。"

        context = {
            "task": TEST_TASK,
            "previous_search_plans": previous_search_plans,
            "previous_sub_answer": previous_sub_answer
        }

        print("\n=== 测试带有前序结果的搜索规划 ===")
        print(f"任务: {TEST_TASK['title']}")
        print(f"前序查询: {previous_search_plans[0]['queries']}")
        print(f"前序子答案: {previous_sub_answer}")

        result = self.agent.execute(context)

        print(f"生成的新搜索查询:")
        for i, query in enumerate(result.get("queries", [])):
            print(f"{i + 1}. {query}")

        # 验证结果包含查询列表
        self.assertIn("queries", result)
        self.assertIsInstance(result["queries"], list)
        self.assertTrue(len(result["queries"]) > 0)


if __name__ == '__main__':
    unittest.main()