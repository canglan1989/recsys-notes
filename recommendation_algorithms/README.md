# 推荐系统算法对比（Recommendation Algorithms Comparison）

对应小红书笔记：协同过滤之后，推荐系统还有哪些玩法？

## 📌 代码说明

实现了 4 种主流推荐算法并对比效果：

1. **协同过滤（Collaborative Filtering）** - 基于用户/物品相似度
2. **基于内容的推荐（Content-Based）** - 基于物品特征相似度
3. **混合推荐（Hybrid）** - 融合多种算法
4. **神经网络推荐（Neural）** - 简化版深度学习推荐

## 🚀 快速开始

```bash
# 安装依赖
pip install numpy pandas scikit-learn

# 运行代码
python recommendation_algorithms.py
```

## 📊 算法原理对比

### 1. 协同过滤（CF）

**核心思想：** "物以类聚，人以群分"

```python
# User-based: 找相似用户
def user_based_cf(user_item_matrix, target_user, k=3):
    # 1. 计算用户相似度（余弦相似度）
    # 2. 找出最相似的 k 个用户
    # 3. 推荐他们喜欢但目标用户没看过的物品
    pass

# Item-based: 找相似物品
def item_based_cf(user_item_matrix, target_user, k=3):
    # 1. 计算物品相似度
    # 2. 找出用户喜欢的物品的相似物品
    # 3. 推荐相似度最高的物品
    pass
```

### 2. 基于内容的推荐

**核心思想：** "喜欢这个的人，也喜欢相似的"

```python
# 使用 TF-IDF 提取物品特征
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(items_df['description'])

# 计算物品相似度
item_similarity = cosine_similarity(tfidf_matrix)

# 推荐相似物品
def recommend_similar_items(item_id, n_recommend=5):
    # 返回与给定物品最相似的 n 个物品
    pass
```

### 3. 混合推荐

**核心思想：** "博采众长，效果更佳"

```python
# 加权融合协同过滤和基于内容的推荐
def hybrid_recommend(cf_recommender, content_recommender, cf_weight=0.5):
    # 1. 获取 CF 推荐结果
    cf_recs = cf_recommender.user_based_recommend(target_user)
    
    # 2. 获取内容推荐结果
    content_recs = content_recommender.recommend_for_user(user_history)
    
    # 3. 归一化分数
    cf_norm = normalize(cf_recs)
    content_norm = normalize(content_recs)
    
    # 4. 加权融合
    final_score = cf_weight * cf_norm + (1 - cf_weight) * content_norm
    
    return sorted_items_by_final_score
```

### 4. 神经网络推荐（简化版）

**核心思想：** "用深度学习学习用户和物品的嵌入表示"

```python
# 学习用户和物品的低维嵌入
user_embeddings = np.random.randn(n_users, embedding_dim)
item_embeddings = np.random.randn(n_items, embedding_dim)

# 用 SVD 分解模拟训练过程
U, S, Vt = np.linalg.svd(user_item_matrix)
user_embeddings = U[:, :k] * np.sqrt(S[:k])
item_embeddings = Vt[:k, :].T * np.sqrt(S[:k])

# 预测评分：嵌入向量的点积
def predict_rating(user_idx, item_idx):
    return np.dot(user_embeddings[user_idx], item_embeddings[item_idx])
```

## 📈 运行结果示例

```
============================================================
1️⃣ 协同过滤（Collaborative Filtering）
============================================================

为 User1 推荐（User-based）：
  电影 D: 预测评分 4.25
  电影 F: 预测评分 3.80
  电影 C: 预测评分 3.50

============================================================
2️⃣ 基于内容的推荐（Content-Based）
============================================================

与「电影 A」相似的电影：
  电影 E: 相似度 0.42
  电影 C: 相似度 0.28
  电影 B: 相似度 0.15

============================================================
3️⃣ 混合推荐（Hybrid）
============================================================

混合推荐（CF 权重 0.6 + 内容权重 0.4）：
  电影 D: 综合得分 0.85
  电影 F: 综合得分 0.72
  电影 E: 综合得分 0.65

============================================================
4️⃣ 神经网络推荐（Neural，简化版）
============================================================

为 User1 推荐：
  电影 D: 预测评分 2.34
  电影 F: 预测评分 1.89
  电影 C: 预测评分 1.56
```

## 🔍 算法对比总结

| 算法类型 | 优点 | 缺点 | 适用场景 |
|----------|------|------|----------|
| **协同过滤** | 无需内容理解<br>能发现新兴趣 | 冷启动问题<br>数据稀疏问题 | 用户行为数据丰富时 |
| **基于内容** | 无冷启动问题<br>可解释性强 | 推荐多样性低<br>依赖特征质量 | 物品有文本描述时 |
| **混合推荐** | 综合优势<br>效果更稳定 | 实现复杂<br>调参成本高 | 追求最佳效果的场景 |
| **神经网络** | 表达能力强<br>可学习复杂特征交互 | 需要大量数据<br>训练成本高<br>可解释性差 | 大规模推荐系统 |

## 💡 如何选择算法？

```
┌─────────────────────────────────────────────────────┐
│              推荐算法选择决策树                      │
└─────────────────────────────────────────────────────┘

有充足的用户行为数据？
│
├─ 是 → 有物品内容描述？
│       │
│       ├─ 是 → 混合推荐（CF + 内容）⭐ 推荐
│       │
│       └─ 否 → 协同过滤（User-based / Item-based）
│
└─ 否 → 有物品内容描述？
        │
        ├─ 是 → 基于内容的推荐
        │
        └─ 否 → 冷启动问题，考虑：
                - 热门推荐（Top-N）
                - 基于规则推荐
                - 收集更多数据
```

## 🎯 实战建议

### 1. 冷启动阶段
- 新用户：基于内容的推荐 + 热门物品
- 新物品：基于内容相似度推荐

### 2. 数据积累阶段
- 用户行为数据 > 1000 条：可以开始用协同过滤
- 用户行为数据 > 10000 条：可以尝试混合推荐

### 3. 成熟阶段
- 考虑深度学习模型（NCF、Wide&Deep、DeepFM）
- A/B 测试不同算法效果
- 持续优化和迭代

## 📚 进阶学习

### 深度学习推荐模型
1. **NCF (Neural Collaborative Filtering)** - 神经协同过滤
2. **Wide & Deep** - Google 提出的经典模型
3. **DeepFM** - 结合 FM 和深度学习的优势
4. **DIN (Deep Interest Network)** - 阿里提出的深度兴趣网络

### 推荐系统评估指标
- **准确率（Precision）**：推荐物品中用户喜欢的比例
- **召回率（Recall）**：用户喜欢的物品中被推荐的比例
- **NDCG**：考虑排序质量的指标
- **覆盖率（Coverage）**：推荐系统能推荐的物品比例

## 🔗 相关资源

1. [推荐系统实践 - 项亮](https://book.douban.com/subject/10769813/)
2. [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-1-0716-2197-4)
3. [Deep Learning for Recommender Systems](https://dl.acm.org/doi/10.1145/3383313.3385260)

## 💬 问题交流

有问题欢迎在 GitHub Issue 或小红书评论区提问！

---

**代码仓库：[github.com/canglan1989/xiaohongshu-codes](https://github.com/canglan1989/xiaohongshu-codes)**

**如果对你有帮助，欢迎 Star ⭐**

**系列第 1 期：[协同过滤算法](../collaborative_filtering/)**
