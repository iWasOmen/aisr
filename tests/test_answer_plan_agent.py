"""
答案规划代理模块的测试。
"""

import unittest
from aisr.core.llm_provider import LLMProvider
from aisr.agents.answer_plan import AnswerPlanAgent
from tests import API_KEY, TEST_QUERY


class TestAnswerPlanAgent(unittest.TestCase):
    """AnswerPlanAgent类的测试用例"""

    def setUp(self):
        """测试前设置anthropic"""
        llm = LLMProvider(provider="openai", api_key=API_KEY)
        self.agent = AnswerPlanAgent(llm, memory=None)

    def test_answer_outline_generation(self):
        """测试答案大纲生成功能"""
        # 创建模拟的子答案
        sub_answers = {
            "task_1_deep_learning_basics": "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。深度学习模型能够从大量数据中学习特征，无需人工特征工程。",
            "task_2_traditional_ml": "传统机器学习算法通常依赖于手工设计的特征，需要领域专家的知识。常见算法包括决策树、随机森林、SVM等。",
            "task_3_comparison": "深度学习与传统机器学习的主要区别在于特征提取方式。深度学习自动学习特征，而传统机器学习需要手动设计特征。深度学习通常需要更多的数据和计算资源。",
            "task_4_applications": "深度学习在图像识别、自然语言处理和语音识别等复杂任务上表现更好，而传统机器学习在结构化数据、小数据集和需要可解释性的场景中仍然有优势。"
        }

        context = {
            "query": TEST_QUERY,
            "sub_answers": sub_answers
        }

        print("\n=== 测试答案大纲生成 ===")
        print(f"查询: {TEST_QUERY}")
        print(f"子答案数: {len(sub_answers)}")

        result = self.agent.execute(context)

        print(f"生成的答案大纲:")
        print(result.get("outline", "无大纲"))

        # 验证结果包含大纲
        self.assertIn("outline", result)
        self.assertIsInstance(result["outline"], str)
        self.assertTrue(len(result["outline"]) > 0)


if __name__ == '__main__':
    unittest.main()