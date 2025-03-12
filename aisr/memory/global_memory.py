"""
全局内存模块，为整个AISR系统提供共享状态。
"""

import logging
from typing import Dict, Any, List, Optional


class GlobalMemory:
    """
    全局内存类，存储整个研究会话的共享状态。

    不同于其他内存类，GlobalMemory不是从Memory基类继承的，
    因为它有特殊的接口和用途。
    """

    def __init__(self):
        """初始化全局内存。"""
        self._store = {}
        logging.debug("全局内存已初始化")

    def set(self, key: str, value: Any) -> None:
        """
        设置全局状态值。

        Args:
            key: 状态键
            value: 状态值
        """
        self._store[key] = value
        logging.debug(f"全局内存: 已设置 '{key}'")

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取全局状态值。

        Args:
            key: 状态键
            default: 如果键不存在，返回的默认值

        Returns:
            关联的状态值，如果不存在则返回默认值
        """
        return self._store.get(key, default)

    def delete(self, key: str) -> None:
        """
        删除全局状态键。

        Args:
            key: 要删除的状态键
        """
        if key in self._store:
            del self._store[key]
            logging.debug(f"全局内存: 已删除 '{key}'")

    def clear(self) -> None:
        """清除所有全局状态。"""
        self._store.clear()
        logging.debug("全局内存: 已清除所有数据")

    def clear_research_data(self) -> None:
        """清除研究相关数据，但保留系统配置。"""
        # 保留的键列表 - 通常是系统配置
        preserved_keys = ["config", "system_settings"]

        # 创建包含保留键值的新存储
        preserved_data = {k: self._store[k] for k in preserved_keys if k in self._store}

        # 清除所有数据
        self._store.clear()

        # 恢复保留的数据
        self._store.update(preserved_data)

        logging.debug("全局内存: 已清除研究数据，保留系统配置")

    def get_all(self) -> Dict[str, Any]:
        """
        获取所有全局状态。

        Returns:
            包含所有状态键值对的字典
        """
        return self._store.copy()