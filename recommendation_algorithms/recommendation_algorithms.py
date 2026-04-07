#!/usr/bin/env python3
"""
推荐系统算法对比（Recommendation Algorithms Comparison）
对应小红书笔记：协同过滤之后，推荐系统还有哪些玩法？

作者：熙洲
小红书：https://www.xiaohongshu.com/user/profile/5c8149b8000000001602b584
GitHub: https://github.com/canglan1989/xiaohongshu-codes
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


# ==================== 1. 协同过滤（回顾） ====================

class CollaborativeFiltering:
    """协同过滤推荐算法"""
    
    def __init__(self, user_item_matrix):
        """
        初始化
        
        Args:
            user_item_matrix: 用户 - 物品评分矩阵（DataFrame）
        """
        self.user_item_matrix = user_item_matrix.fillna(0)
        self.user_similarity = None
        self.item_similarity = None
    
    def compute_user_similarity(self):
        """计算用户相似度矩阵"""
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        return self.user_similarity
    
    def compute_item_similarity(self):
        """计算物品相似度矩阵"""
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        return self.item_similarity
    
    def user_based_recommend(self, target_user, n_recommend=5, k=10):
        """
        基于用户的协同过滤推荐
        
        Args:
            target_user: 目标用户 ID
            n_recommend: 推荐物品数量
            k: 选取最相似的 k 个用户
        
        Returns:
            推荐物品列表 [(物品 ID, 预测评分), ...]
        """
        if self.user_similarity is None:
            self.compute_user_similarity()
        
        user_idx = self.user_item_matrix.index.get_loc(target_user)
        similar_users = np.argsort(self.user_similarity[user_idx])[::-1][1:k+1]
        
        predictions = {}
        for item_idx, item_id in enumerate(self.user_item_matrix.columns):
            if self.user_item_matrix.loc[target_user, item_id] == 0:
                numerator = 0
                denominator = 0
                for similar_idx in similar_users:
                    similar_user = self.user_item_matrix.index[similar_idx]
                    rating = self.user_item_matrix.loc[similar_user, item_id]
                    if rating > 0:
                        similarity = self.user_similarity[user_idx, similar_idx]
                        numerator += similarity * rating
                        denominator += abs(similarity)
                
                if denominator > 0:
                    predictions[item_id] = numerator / denominator
        
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommend]
    
    def item_based_recommend(self, target_user, n_recommend=5, k=10):
        """
        基于物品的协同过滤推荐
        
        Args:
            target_user: 目标用户 ID
            n_recommend: 推荐物品数量
            k: 选取最相似的 k 个物品
        
        Returns:
            推荐物品列表 [(物品 ID, 预测评分), ...]
        """
        if self.item_similarity is None:
            self.compute_item_similarity()
        
        user_ratings = self.user_item_matrix.loc[target_user]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        predictions = {}
        for item_idx, item_id in enumerate(self.user_item_matrix.columns):
            if item_id not in rated_items:
                numerator = 0
                denominator = 0
                for rated_item in rated_items:
                    rated_item_idx = self.user_item_matrix.columns.get_loc(rated_item)
                    similarity = self.item_similarity[item_idx, rated_item_idx]
                    rating = user_ratings[rated_item]
                    numerator += similarity * rating
                    denominator += abs(similarity)
                
                if denominator > 0:
                    predictions[item_id] = numerator / denominator
        
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommend]


# ==================== 2. 基于内容的推荐 ====================

class ContentBasedRecommender:
    """基于内容的推荐算法"""
    
    def __init__(self, items_df, content_column='description'):
        """
        初始化
        
        Args:
            items_df: 物品信息 DataFrame
                     包含：item_id, description（文本描述）, 其他特征
            content_column: 内容列名
        """
        self.items_df = items_df
        self.content_column = content_column
        self.tfidf_matrix = None
        self.item_similarity = None
        
        self._compute_tfidf()
    
    def _compute_tfidf(self):
        """计算 TF-IDF 矩阵"""
        # 使用 TF-IDF 将文本内容转换为向量
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = tfidf.fit_transform(self.items_df[self.content_column])
        
        # 计算物品相似度
        self.item_similarity = cosine_similarity(self.tfidf_matrix)
        
        # 将相似度矩阵转换为 DataFrame，方便查询
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.items_df['item_id'],
            columns=self.items_df['item_id']
        )
    
    def recommend_similar_items(self, item_id, n_recommend=5):
        """
        推荐与给定物品相似的物品
        
        Args:
            item_id: 给定物品 ID
            n_recommend: 推荐数量
        
        Returns:
            相似物品列表 [(物品 ID, 相似度), ...]
        """
        if item_id not in self.item_similarity_df.index:
            raise ValueError(f"物品 {item_id} 不存在")
        
        # 获取该物品与其他物品的相似度
        similar_items = self.item_similarity_df[item_id].sort_values(ascending=False)
        
        # 排除自己，返回最相似的 n 个物品
        similar_items = similar_items.iloc[1:n_recommend+1]
        
        return list(zip(similar_items.index, similar_items.values))
    
    def recommend_for_user(self, user_history, n_recommend=5):
        """
        基于用户历史行为推荐
        
        Args:
            user_history: 用户喜欢的物品 ID 列表
            n_recommend: 推荐数量
        
        Returns:
            推荐物品列表 [(物品 ID, 综合相似度), ...]
        """
        # 计算用户历史物品的平均相似度向量
        scores = defaultdict(float)
        
        for item_id in user_history:
            if item_id in self.item_similarity_df.index:
                for other_item, similarity in self.item_similarity_df[item_id].items():
                    if other_item not in user_history:  # 排除已看过的
                        scores[other_item] += similarity
        
        # 归一化（除以被累加的次数）
        for item_id in scores:
            scores[item_id] /= len(user_history)
        
        # 返回得分最高的 n 个物品
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommend]


# ==================== 3. 混合推荐 ====================

class HybridRecommender:
    """混合推荐算法（协同过滤 + 基于内容）"""
    
    def __init__(self, cf_recommender, content_recommender, cf_weight=0.5):
        """
        初始化
        
        Args:
            cf_recommender: 协同过滤推荐器实例
            content_recommender: 基于内容的推荐器实例
            cf_weight: 协同过滤权重（0-1），内容推荐权重 = 1 - cf_weight
        """
        self.cf_recommender = cf_recommender
        self.content_recommender = content_recommender
        self.cf_weight = cf_weight
    
    def recommend(self, target_user, user_history, n_recommend=5):
        """
        混合推荐
        
        Args:
            target_user: 目标用户 ID
            user_history: 用户历史喜欢的物品 ID 列表
            n_recommend: 推荐数量
        
        Returns:
            推荐物品列表 [(物品 ID, 综合得分), ...]
        """
        # 获取协同过滤推荐
        cf_recs = self.cf_recommender.user_based_recommend(target_user, n_recommend=n_recommend*2)
        cf_dict = dict(cf_recs)
        
        # 获取基于内容推荐
        content_recs = self.content_recommender.recommend_for_user(user_history, n_recommend=n_recommend*2)
        content_dict = dict(content_recs)
        
        # 合并所有候选物品
        all_items = set(cf_dict.keys()) | set(content_dict.keys())
        
        # 归一化分数到 0-1 范围
        def normalize(scores):
            if not scores:
                return {}
            min_score = min(scores.values())
            max_score = max(scores.values())
            if max_score - min_score == 0:
                return {k: 0.5 for k in scores}
            return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}
        
        cf_norm = normalize(cf_dict)
        content_norm = normalize(content_dict)
        
        # 加权融合
        final_scores = {}
        for item in all_items:
            cf_score = cf_norm.get(item, 0)
            content_score = content_norm.get(item, 0)
            final_scores[item] = (
                self.cf_weight * cf_score + 
                (1 - self.cf_weight) * content_score
            )
        
        # 返回得分最高的 n 个物品
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommend]


# ==================== 4. 简单的神经网络推荐（简化版） ====================

class NeuralRecommender:
    """
    简化的神经网络推荐（模拟深度学习推荐）
    
    注意：这是教学简化版，实际生产环境会使用：
    - Neural Collaborative Filtering (NCF)
    - Wide & Deep
    - DeepFM
    等更复杂的模型
    """
    
    def __init__(self, user_item_matrix, embedding_dim=10):
        """
        初始化
        
        Args:
            user_item_matrix: 用户 - 物品评分矩阵
            embedding_dim: 嵌入向量维度
        """
        self.user_item_matrix = user_item_matrix.fillna(0)
        self.embedding_dim = embedding_dim
        
        # 初始化用户和物品嵌入（模拟训练后的结果）
        np.random.seed(42)
        n_users = len(self.user_item_matrix)
        n_items = len(self.user_item_matrix.columns)
        
        self.user_embeddings = np.random.randn(n_users, embedding_dim)
        self.item_embeddings = np.random.randn(n_items, embedding_dim)
        
        # 简单的"训练"：用评分矩阵调整嵌入
        self._simple_train()
    
    def _simple_train(self):
        """简化的训练过程（实际应使用梯度下降）"""
        # 这里用 SVD 分解模拟嵌入学习
        U, S, Vt = np.linalg.svd(self.user_item_matrix.values)
        
        # 取前 embedding_dim 个奇异值
        k = min(self.embedding_dim, len(S))
        self.user_embeddings = U[:, :k] * np.sqrt(S[:k])
        self.item_embeddings = Vt[:k, :].T * np.sqrt(S[:k])
    
    def predict_rating(self, user_idx, item_idx):
        """预测用户对物品的评分"""
        # 点积模拟神经网络的输出层
        return np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx])
    
    def recommend(self, target_user, n_recommend=5):
        """
        为用户推荐物品
        
        Args:
            target_user: 目标用户 ID
            n_recommend: 推荐数量
        
        Returns:
            推荐物品列表 [(物品 ID, 预测评分), ...]
        """
        user_idx = self.user_item_matrix.index.get_loc(target_user)
        user_ratings = self.user_item_matrix.loc[target_user]
        rated_items = set(user_ratings[user_ratings > 0].index)
        
        # 预测所有未评分物品的得分
        predictions = {}
        for item_idx, item_id in enumerate(self.user_item_matrix.columns):
            if item_id not in rated_items:
                score = self.predict_rating(user_idx, item_idx)
                predictions[item_id] = score
        
        # 返回得分最高的 n 个物品
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommend]


# ==================== 示例数据生成 ====================

def create_sample_data():
    """创建示例数据"""
    # 用户 - 电影评分矩阵
    rating_data = {
        '电影 A': [5, 4, 0, 0, 3, 0, 0, 5, 0, 4],
        '电影 B': [4, 0, 5, 0, 0, 5, 0, 4, 3, 0],
        '电影 C': [0, 5, 4, 3, 0, 4, 5, 0, 0, 5],
        '电影 D': [0, 0, 0, 5, 4, 0, 4, 0, 5, 0],
        '电影 E': [3, 0, 0, 4, 5, 0, 5, 3, 4, 0],
        '电影 F': [0, 3, 4, 0, 0, 4, 0, 0, 5, 4],
    }
    
    users = [f'User{i}' for i in range(1, 11)]
    rating_df = pd.DataFrame(rating_data, index=users)
    
    # 电影内容描述（用于基于内容的推荐）
    content_data = {
        'item_id': ['电影 A', '电影 B', '电影 C', '电影 D', '电影 E', '电影 F'],
        'description': [
            'action sci-fi adventure space hero',
            'romance drama love story emotional',
            'action thriller crime detective mystery',
            'comedy funny humor family entertaining',
            'sci-fi fantasy magic adventure epic',
            'drama historical biography serious'
        ]
    }
    content_df = pd.DataFrame(content_data)
    
    return rating_df, content_df


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("推荐系统算法对比演示")
    print("=" * 60)
    
    # 创建示例数据
    rating_df, content_df = create_sample_data()
    
    print("\n📊 用户 - 物品评分矩阵：")
    print(rating_df)
    
    print("\n📝 物品内容描述：")
    print(content_df)
    
    # ==================== 1. 协同过滤 ====================
    print("\n" + "=" * 60)
    print("1️⃣ 协同过滤（Collaborative Filtering）")
    print("=" * 60)
    
    cf = CollaborativeFiltering(rating_df)
    target_user = 'User1'
    
    print(f"\n为 {target_user} 推荐（User-based）：")
    cf_recs = cf.user_based_recommend(target_user, n_recommend=3)
    for item, score in cf_recs:
        print(f"  {item}: 预测评分 {score:.2f}")
    
    # ==================== 2. 基于内容的推荐 ====================
    print("\n" + "=" * 60)
    print("2️⃣ 基于内容的推荐（Content-Based）")
    print("=" * 60)
    
    content_rec = ContentBasedRecommender(content_df, content_column='description')
    
    # 推荐与电影 A 相似的电影
    print("\n与「电影 A」相似的电影：")
    similar_items = content_rec.recommend_similar_items('电影 A', n_recommend=3)
    for item, sim in similar_items:
        print(f"  {item}: 相似度 {sim:.2f}")
    
    # 基于用户历史推荐
    user_history = ['电影 A', '电影 B']
    print(f"\n基于用户历史 {user_history} 推荐：")
    content_recs = content_rec.recommend_for_user(user_history, n_recommend=3)
    for item, score in content_recs:
        print(f"  {item}: 综合相似度 {score:.2f}")
    
    # ==================== 3. 混合推荐 ====================
    print("\n" + "=" * 60)
    print("3️⃣ 混合推荐（Hybrid）")
    print("=" * 60)
    
    hybrid_rec = HybridRecommender(cf, content_rec, cf_weight=0.6)
    
    print(f"\n混合推荐（CF 权重 0.6 + 内容权重 0.4）：")
    hybrid_recs = hybrid_rec.recommend(target_user, user_history, n_recommend=3)
    for item, score in hybrid_recs:
        print(f"  {item}: 综合得分 {score:.2f}")
    
    # ==================== 4. 神经网络推荐 ====================
    print("\n" + "=" * 60)
    print("4️⃣ 神经网络推荐（Neural，简化版）")
    print("=" * 60)
    
    neural_rec = NeuralRecommender(rating_df, embedding_dim=5)
    
    print(f"\n为 {target_user} 推荐：")
    neural_recs = neural_rec.recommend(target_user, n_recommend=3)
    for item, score in neural_recs:
        print(f"  {item}: 预测评分 {score:.2f}")
    
    # ==================== 算法对比总结 ====================
    print("\n" + "=" * 60)
    print("📈 算法对比总结")
    print("=" * 60)
    
    comparison = """
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│     算法类型     │    优点      │    缺点      │   适用场景   │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│  协同过滤       │ 无需内容理解 │ 冷启动问题   │ 用户行为数据 │
│  (CF)           │ 能发现新兴趣 │ 数据稀疏问题 │   丰富时     │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│  基于内容       │ 无冷启动问题 │ 推荐多样性低 │ 物品有文本   │
│  (Content)      │ 可解释性强   │ 依赖特征质量 │   描述时     │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│  混合推荐       │ 综合优势     │ 实现复杂     │ 追求最佳效果 │
│  (Hybrid)       │ 效果更稳定   │ 调参成本高   │   的场景     │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│  神经网络       │ 表达能力强   │ 需要大量数据 │ 大规模推荐   │
│  (Neural)       │ 可学习复杂   │ 训练成本高   │   系统       │
│                 │   特征交互   │ 可解释性差   │              │
└─────────────────┴──────────────┴──────────────┴──────────────┘
    """
    print(comparison)
    
    print("=" * 60)
    print("💻 完整代码：https://github.com/canglan1989/xiaohongshu-codes")
    print("📕 小红书：熙洲 | 推荐系统 + AI 创业")
    print("=" * 60)


if __name__ == "__main__":
    main()
