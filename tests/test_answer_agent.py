"""
答案生成代理模块的测试。
"""

import unittest
from aisr.core.llm_provider import LLMProvider
from aisr.agents.answer import AnswerAgent
from tests import TEST_QUERY
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TestAnswerAgent(unittest.TestCase):
    """AnswerAgent类的测试用例"""

    def setUp(self):
        """测试前设置anthropic"""
        llm = LLMProvider()
        self.agent = AnswerAgent(llm, memory=None)

    def test_answer_generation(self):
        """测试最终答案生成功能"""
        # 创建模拟的子答案
        sub_answers = {
            "task_1_deep_learning_basics": "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。深度学习模型能够从大量数据中学习特征，无需人工特征工程。",
            "task_2_traditional_ml": "传统机器学习算法通常依赖于手工设计的特征，需要领域专家的知识。常见算法包括决策树、随机森林、SVM等。",
            "task_3_comparison": "深度学习与传统机器学习的主要区别在于特征提取方式。深度学习自动学习特征，而传统机器学习需要手动设计特征。深度学习通常需要更多的数据和计算资源。",
            "task_4_applications": "深度学习在图像识别、自然语言处理和语音识别等复杂任务上表现更好，而传统机器学习在结构化数据、小数据集和需要可解释性的场景中仍然有优势。"
        }

        # 创建模拟的答案大纲
        outline = """
        I. 引言
           A. 深度学习和传统机器学习的概述
           B. 两者在机器学习领域的地位

        II. 深度学习基础
           A. 定义和核心概念
           B. 神经网络结构
           C. 工作原理

        III. 传统机器学习基础
           A. 定义和核心概念
           B. 常见算法类型
           C. 工作原理

        IV. 深度学习与传统机器学习的主要区别
           A. 特征提取方式
           B. 数据需求
           C. 计算资源需求
           D. 模型复杂性

        V. 应用场景对比
           A. 深度学习的优势场景
           B. 传统机器学习的优势场景

        VI. 结论
           A. 两种方法的互补性
           B. 选择方法的考虑因素
        """

        context = {
            "query": TEST_QUERY,
            "sub_answers": sub_answers,
            "outline": outline
        }

        print("\n=== 测试最终答案生成 ===")
        print(f"查询: {TEST_QUERY}")
        print(f"子答案数: {len(sub_answers)}")
        print(f"大纲长度: {len(outline)} 字符")

        result = self.agent.execute(context)

        print(f"最终答案预览 (前300个字符):")
        answer = result.get("answer", "无答案")
        preview = answer[:300] + "..." if len(answer) > 300 else answer
        print(preview)
        print(f"答案总长度: {len(answer)} 字符")

        # 验证结果包含答案
        self.assertIn("answer", result)
        self.assertIsInstance(result["answer"], str)
        self.assertTrue(len(result["answer"]) > 100)  # 答案应该相当长


if __name__ == '__main__':
    unittest.main()