"""
主工作流模块的测试。
"""

import unittest
import time
from aisr.workflows.simple_workflow import main_workflow
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TestMainWorkflow(unittest.TestCase):
    """main_workflow函数的测试用例"""

    def test_main_workflow_small_query(self):
        """测试简单查询的完整工作流程"""
        # 使用简单的查询以加快测试速度
        query = "苹果的营养价值是什么？"

        print("\n=== 测试主工作流：简单查询 ===")
        print(f"查询: {query}")
        print("执行中，这可能需要几分钟时间...")

        start_time = time.time()

        # 设置max_iterations=1以加快测试
        result = main_workflow(
            query=query,
            max_iterations=1,
            provider="anthropic"
        )

        duration = time.time() - start_time

        print(f"执行耗时: {duration:.2f}秒")
        print(f"完成的子任务数: {result.get('completed_tasks', 0)}/{result.get('task_count', 0)}")

        print(f"最终答案预览 (前300个字符):")
        answer = result.get("answer", "无答案")
        preview = answer[:300] + "..." if len(answer) > 300 else answer
        print(preview)
        print(f"答案总长度: {len(answer)} 字符")

        # 验证结果包含基本字段
        self.assertIn("query", result)
        self.assertEqual(result["query"], query)
        self.assertIn("answer", result)
        self.assertTrue(len(result["answer"]) > 0)
        self.assertIn("sub_answers", result)
        self.assertIn("execution_time", result)

    @unittest.skip("完整测试较耗时，默认跳过")
    def test_main_workflow_full(self):
        """测试完整的工作流程"""
        query = "深度学习与传统机器学习有什么区别，各有什么优缺点？"

        print("\n=== 测试主工作流：完整查询 ===")
        print(f"查询: {query}")
        print("执行中，这可能需要较长时间...")

        start_time = time.time()

        result = main_workflow(
            query=query,
            max_iterations=2,  # 限制为2次迭代以节省时间
            provider="anthropic"
        )

        duration = time.time() - start_time

        print(f"执行耗时: {duration:.2f}秒")
        print(f"完成的子任务数: {result.get('completed_tasks', 0)}/{result.get('task_count', 0)}")

        print(f"子答案数: {len(result.get('sub_answers', {}))}")

        print(f"最终答案预览 (前300个字符):")
        answer = result.get("answer", "无答案")
        preview = answer[:300] + "..." if len(answer) > 300 else answer
        print(preview)
        print(f"答案总长度: {len(answer)} 字符")

        # 验证结果
        self.assertIn("answer", result)
        self.assertTrue(len(result["answer"]) > 500)  # 应该有相当长的答案
        self.assertTrue(len(result["sub_answers"]) > 0)


if __name__ == '__main__':
    unittest.main()
    #python -m unittest tests.test_task_plan_agent
    #python -m unittest discover tests