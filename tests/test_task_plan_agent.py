"""
任务规划代理模块的测试。
"""
import os
import unittest
import json
from aisr.core.llm_provider import LLMProvider
from aisr.agents.task_plan import TaskPlanAgent
from tests import TEST_QUERY
from aisr.utils.config import config

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TestTaskPlanAgent(unittest.TestCase):
    """TaskPlanAgent类的测试用例"""

    def setUp(self):
        """测试前设置anthropic"""
        llm = LLMProvider()
        self.agent = TaskPlanAgent(llm, memory=None)

    def test_task_planning(self):
        """测试任务规划功能"""
        context = {"query": TEST_QUERY}

        print("\n=== 测试任务规划 ===")
        print(f"查询: {TEST_QUERY}")

        result = self.agent.execute(context)

        print(f"子任务数: {len(result.get('sub_tasks', []))}")
        print("生成的子任务:")
        for task in result.get("sub_tasks", []):
            print(f"- {task.get('id')}: {task.get('title')}")
            print(f"  描述: {task.get('description')}")
            print()

        # 验证结果包含子任务列表
        self.assertIn("sub_tasks", result)
        self.assertIsInstance(result["sub_tasks"], list)
        self.assertTrue(len(result["sub_tasks"]) > 0)

        # 验证每个子任务有必要的字段
        for task in result["sub_tasks"]:
            self.assertIn("id", task)
            self.assertIn("title", task)
            self.assertIn("description", task)

    def test_task_planning_with_previous_results(self):
        """测试带有前序结果的任务规划功能"""
        # 创建模拟的前序子答案
        previous_sub_answers = {
            "task_1_deep_learning_basics": "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。深度学习模型能够从大量数据中学习特征，无需人工特征工程。"
        }

        context = {
            "query": TEST_QUERY,
            "previous_sub_answers": previous_sub_answers
        }

        print("\n=== 测试带有前序结果的任务规划 ===")
        print(f"查询: {TEST_QUERY}")
        print(f"已有子答案数: {len(previous_sub_answers)}")

        result = self.agent.execute(context)

        print(f"子任务数: {len(result.get('sub_tasks', []))}")
        print("生成的子任务:")
        for task in result.get("sub_tasks", []):
            print(f"- {task.get('id')}: {task.get('title')}")
            print(f"  描述: {task.get('description')}")
            print()

        # 验证结果包含子任务列表
        self.assertIn("sub_tasks", result)
        self.assertIsInstance(result["sub_tasks"], list)
        self.assertTrue(len(result["sub_tasks"]) > 0)


if __name__ == '__main__':
    unittest.main()