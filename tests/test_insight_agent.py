"""
洞察生成代理模块的测试。
"""

import unittest
from aisr.core.llm_provider import LLMProvider
from aisr.agents.insight import InsightAgent
from tests import API_KEY, TEST_QUERY


class TestInsightAgent(unittest.TestCase):
    """InsightAgent类的测试用例"""

    def setUp(self):
        """测试前设置anthropic"""
        llm = LLMProvider(provider="openai", api_key=API_KEY)
        self.agent = InsightAgent(llm, memory=None)

    def test_insight_generation(self):
        """测试洞察生成功能"""
        # 创建模拟的子答案
        sub_answers = {
            "task_1_deep_learning_basics": "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。深度学习模型能够从大量数据中学习特征，无需人工特征工程。",
            "task_2_traditional_ml": "传统机器学习算法通常依赖于手工设计的特征，需要领域专家的知识。常见算法包括决策树、随机森林、SVM等。",
            "task_3_comparison": "深度学习与传统机器学习的主要区别在于特征提取方式。深度学习自动学习特征，而传统机器学习需要手动设计特征。深度学习通常需要更多的数据和计算资源。"
        }

        # 创建模拟的未执行任务计划
        unexecuted_plan = {
            "sub_tasks": [
                {
                    "id": "task_4_applications",
                    "title": "应用领域对比",
                    "description": "研究深度学习和传统机器学习在不同应用领域的优缺点和适用场景。"
                },
                {
                    "id": "task_5_future_trends",
                    "title": "未来发展趋势",
                    "description": "探讨深度学习和传统机器学习的未来发展方向和潜在融合。"
                }
            ]
        }
        unexecuted_plan={}

        context = {
            "query": TEST_QUERY,
            "sub_answers": sub_answers,
            "unexecuted_plan": unexecuted_plan
        }

        print("\n=== 测试洞察生成 ===")
        print(f"查询: {TEST_QUERY}")
        print(f"已完成任务数: {len(sub_answers)}")
        print(f"未执行任务数: {len(unexecuted_plan.get('sub_tasks',[]))}")

        result = self.agent.execute(context)

        print(f"生成的洞察:")
        print(result.get("insight", "无洞察"))

        # 验证结果包含洞察
        self.assertIn("insight", result)
        self.assertIsInstance(result["insight"], str)
        self.assertTrue(len(result["insight"]) > 0)


if __name__ == '__main__':
    unittest.main()