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
        """测试带有前序结果和相关任务答案的搜索规划功能"""
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

        # 创建模拟的相关任务答案
        related_tasks_answers = {
            "task_1_机器学习基础": "机器学习是人工智能的一个子领域，专注于通过经验自动改进的算法。基本类型包括监督学习、无监督学习和强化学习。",
            "task_2_神经网络结构": "神经网络由输入层、隐藏层和输出层组成，通过权重连接。深度神经网络包含多个隐藏层，能够学习更复杂的特征表示。"
        }

        # 构建完整的上下文
        context = {
            "task": TEST_TASK,
            "previous_search_plans": previous_search_plans,
            "previous_sub_answer": previous_sub_answer,
            "related_tasks_answers": related_tasks_answers
        }

        print("\n=== 测试带有前序结果和相关任务答案的搜索规划 ===")
        print(f"任务: {TEST_TASK['title']}")
        print(f"前序查询: {previous_search_plans[0]['queries']}")
        print(f"前序子答案: {previous_sub_answer}")
        print(f"相关任务数量: {len(related_tasks_answers)}")

        # 执行代理
        result = self.agent.execute(context)

        print(f"生成的新搜索查询:")
        for i, query in enumerate(result.get("queries", [])):
            print(f"{i + 1}. {query}")

        print(f"推理过程:")
        print(result.get("reasoning", "")[:300] + "..." if len(result.get("reasoning", "")) > 300 else result.get(
            "reasoning", ""))

        # 验证结果包含查询列表
        self.assertIn("queries", result)
        self.assertIsInstance(result["queries"], list)
        self.assertTrue(len(result["queries"]) > 0)

        # 验证结果包含推理过程
        self.assertIn("reasoning", result)

        # 验证查询不与前序查询重复
        for query in result.get("queries", []):
            self.assertNotIn(query, previous_search_plans[0]["queries"])


    def test_search_planning_with_multiple_iterations(self):
        """测试多轮迭代的搜索规划功能"""
        # 创建模拟的多轮搜索计划
        previous_search_plans = [
            {
                "queries": ["深度学习入门概念"]
            },
            {
                "queries": ["深度学习框架比较"]
            }
        ]

        # 多轮子答案
        previous_sub_answers = [
            "深度学习是机器学习的一种方法，使用多层神经网络。",
            "主流深度学习框架包括TensorFlow、PyTorch和Keras，各有优缺点。"
        ]

        # 只使用最后一个子答案作为上下文
        previous_sub_answer = previous_sub_answers[-1]

        context = {
            "task": TEST_TASK,
            "previous_search_plans": previous_search_plans,
            "previous_sub_answer": previous_sub_answer,
            "related_tasks_answers": {}  # 空的相关任务答案
        }

        print("\n=== 测试多轮迭代的搜索规划 ===")
        print(f"任务: {TEST_TASK['title']}")
        print(f"搜索历史: {len(previous_search_plans)} 轮")
        for i, plan in enumerate(previous_search_plans):
            print(f"第 {i + 1} 轮查询: {plan['queries']}")
        print(f"最新子答案: {previous_sub_answer}")

        result = self.agent.execute(context)

        print(f"生成的新搜索查询:")
        for i, query in enumerate(result.get("queries", [])):
            print(f"{i + 1}. {query}")

        # 验证结果包含查询列表
        self.assertIn("queries", result)
        self.assertIsInstance(result["queries"], list)
        self.assertTrue(len(result["queries"]) > 0)

        # 验证新查询与前两轮查询不重复
        all_previous_queries = []
        for plan in previous_search_plans:
            all_previous_queries.extend(plan.get("queries", []))

        for query in result.get("queries", []):
            self.assertNotIn(query, all_previous_queries)


if __name__ == '__main__':
    unittest.main()