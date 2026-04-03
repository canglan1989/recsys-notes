# 协同过滤（Collaborative Filtering）

对应小红书笔记：[3 分钟讲清楚协同过滤，小白也能懂](https://www.xiaohongshu.com/explore/69cf6af2000000002202ba9f)

## 📌 代码说明

实现了两种协同过滤算法：
1. **User-based** - 基于用户的协同过滤
2. **Item-based** - 基于物品的协同过滤

## 🚀 快速开始

```bash
# 安装依赖
pip install numpy pandas scikit-learn

# 运行代码
python collaborative_filtering.py
```

## 📊 示例数据

使用 MovieLens 数据集（示例）：
- 用户 - 电影评分矩阵
- 包含 10 个用户，20 部电影

## 💡 核心代码

### User-based 协同过滤

```python
def user_based_cf(user_item_matrix, target_user, k=3):
    """
    基于用户的协同过滤
    
    Args:
        user_item_matrix: 用户 - 物品评分矩阵
        target_user: 目标用户
        k: 选取最相似的 k 个用户
    
    Returns:
        推荐物品列表
    """
    # 1. 计算用户相似度（余弦相似度）
    # 2. 找出最相似的 k 个用户
    # 3. 推荐他们喜欢但目标用户没看过的物品
    pass
```

### Item-based 协同过滤

```python
def item_based_cf(user_item_matrix, target_user, k=3):
    """
    基于物品的协同过滤
    
    Args:
        user_item_matrix: 用户 - 物品评分矩阵
        target_user: 目标用户
        k: 选取最相似的 k 个物品
    
    Returns:
        推荐物品列表
    """
    # 1. 计算物品相似度
    # 2. 找出用户喜欢的物品的相似物品
    # 3. 推荐相似度最高的物品
    pass
```

## 📈 运行结果

```
User-based 推荐结果：
用户 1 的推荐：[电影 A, 电影 B, 电影 C]

Item-based 推荐结果：
用户 1 的推荐：[电影 D, 电影 E, 电影 F]
```

## 🔍 算法对比

| 特性 | User-based | Item-based |
|------|-----------|-----------|
| 相似度计算 | 用户之间 | 物品之间 |
| 实时性 | 差（用户变化需重新计算） | 好（物品相对稳定） |
| 适用场景 | 用户少、物品多 | 用户多、物品少 |
| 推荐效果 | 更个性化 | 更稳定 |

## 📚 参考资料

1. [推荐系统实践 - 项亮](https://book.douban.com/subject/10769813/)
2. [Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121)

## 💬 问题交流

有问题欢迎在 GitHub Issue 或小红书评论区提问！

---

**如果对你有帮助，欢迎 Star ⭐**
