from typing import Dict, Any

class Router:
    """
    组件通信的中央路由系统。

    处理组件间的函数调用路由，提供统一的组件间通信接口。
    """

    def __init__(self):
        """初始化路由器"""
        self.components = {}  # 存储注册组件的字典

    def register(self, name: str, component) -> None:
        """
        向路由器注册组件。

        Args:
            name: 组件的字符串标识符
            component: 组件实例
        """
        self.components[name] = component

    def route(self, function_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        将函数调用路由到适当的组件。

        Args:
            function_call: 包含'function'和'parameters'键的字典

        Returns:
            被调用组件的结果

        Raises:
            ValueError: 如果找不到组件或方法
        """
        # 解析函数调用
        function_path = function_call.get("function", "")
        parameters = function_call.get("parameters", {})

        if "." not in function_path:
            raise ValueError(f"无效的函数路径: {function_path}。预期格式: 'component.method'")

        component_name, method_name = function_path.split(".", 1)

        # 获取组件
        component = self.components.get(component_name)
        if not component:
            raise ValueError(f"未找到组件: {component_name}")

        # 获取方法
        method = getattr(component, method_name, None)
        if not method or not callable(method):
            raise ValueError(f"未找到方法: {method_name} in component {component_name}")

        # 执行方法
        try:
            return method(**parameters)
        except Exception as e:
            # 实际实现中会包含更复杂的错误处理、日志记录和可能的重试逻辑
            return {
                "error": str(e),
                "component": component_name,
                "method": method_name,
                "status": "failed"
            }
