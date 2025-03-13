"""
AISR系统测试包。
包含对各个模块的功能测试。
"""

# 设置API密钥（在实际使用中应从环境变量或配置文件获取）
API_KEY = "sk-xx"  # 请替换为实际的API密钥
#API_KEY="sk-xx"
# 通用测试查询
TEST_QUERY = "什么是深度学习，它与传统机器学习有什么区别？"

# 测试子任务
TEST_TASK = {
    "id": "task_1_deep_learning_overview",
    "title": "深度学习概述",
    "description": "研究深度学习的基本概念、工作原理和关键特点。"
}

# 测试搜索结果（极简版，用于那些不想执行实际搜索的测试）
TEST_SEARCH_RESULTS = {
    "results": [
        {
            "title": "深度学习简介 - 维基百科",
            "snippet": "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。",
            "link": "https://example.com/wiki/deep_learning",
            "content": "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。深度学习模型能够从大量数据中学习特征，无需人工特征工程。"
        },
        {
            "title": "深度学习与传统机器学习的区别 - AI研究",
            "snippet": "传统机器学习通常需要手工设计特征，而深度学习能够自动学习特征表示。",
            "link": "https://example.com/ai/deep_vs_traditional",
            "content": "传统机器学习依赖于人工特征工程，而深度学习能够自动从原始数据中学习层次化特征。深度学习模型通常需要更多的数据和计算资源，但在复杂任务上表现更好。"
        }
    ],
    "result_count": 2
}