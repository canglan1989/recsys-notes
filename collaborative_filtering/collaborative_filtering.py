#!/usr/bin/env python3
"""
协同过滤算法实现（Collaborative Filtering）
对应小红书笔记：3 分钟讲清楚协同过滤，小白也能懂

作者：熙洲
小红书：https://www.xiaohongshu.com/user/profile/5c8149b8000000001602b584
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFiltering:
    """协同过滤推荐算法"""
    
    def __init__(self, user_item_matrix):
        """
        初始化
        
        Args:
            user_item_matrix: 用户 - 物品评分矩阵（DataFrame）
                             行：用户，列：物品，值：评分
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
        
        # 获取目标用户的索引
        user_idx = self.user_item_matrix.index.get_loc(target_user)
        
        # 获取相似度最高的 k 个用户（排除自己）
        similar_users = np.argsort(self.user_similarity[user_idx])[::-1][1:k+1]
        
        # 计算加权评分
        predictions = {}
        for item_idx, item_id in enumerate(self.user_item_matrix.columns):
            if self.user_item_matrix.loc[target_user, item_id] == 0:  # 只推荐没看过的
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
        
        # 返回评分最高的 n 个物品
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
        
        # 获取目标用户看过的物品
        user_ratings = self.user_item_matrix.loc[target_user]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        # 计算未看过物品的预测评分
        predictions = {}
        for item_idx, item_id in enumerate(self.user_item_matrix.columns):
            if item_id not in rated_items:  # 只推荐没看过的
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
        
        # 返回评分最高的 n 个物品
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommend]


def create_sample_data():
    """创建示例数据"""
    # 模拟用户 - 电影评分数据
    data = {
        '电影 A': [5, 4, 0, 0, 3, 0, 0, 5, 0, 4],
        '电影 B': [4, 0, 5, 0, 0, 5, 0, 4, 3, 0],
        '电影 C': [0, 5, 4, 3, 0, 4, 5, 0, 0, 5],
        '电影 D': [0, 0, 0, 5, 4, 0, 4, 0, 5, 0],
        '电影 E': [3, 0, 0, 4, 5, 0, 5, 3, 4, 0],
        '电影 F': [0, 3, 4, 0, 0, 4, 0, 0, 5, 4],
    }
    
    users = [f'User{i}' for i in range(1, 11)]
    df = pd.DataFrame(data, index=users)
    
    return df


def main():
    """主函数"""
    print("=" * 50)
    print("协同过滤推荐算法演示")
    print("=" * 50)
    
    # 创建示例数据
    print("\n📊 用户 - 物品评分矩阵：")
    user_item_matrix = create_sample_data()
    print(user_item_matrix)
    
    # 创建推荐器
    cf = CollaborativeFiltering(user_item_matrix)
    
    # User-based 推荐
    print("\n" + "=" * 50)
    print("User-based 协同过滤推荐")
    print("=" * 50)
    
    target_user = 'User1'
    recommendations = cf.user_based_recommend(target_user, n_recommend=3)
    
    print(f"\n为 {target_user} 推荐：")
    for item, score in recommendations:
        print(f"  {item}: 预测评分 {score:.2f}")
    
    # Item-based 推荐
    print("\n" + "=" * 50)
    print("Item-based 协同过滤推荐")
    print("=" * 50)
    
    recommendations = cf.item_based_recommend(target_user, n_recommend=3)
    
    print(f"\n为 {target_user} 推荐：")
    for item, score in recommendations:
        print(f"  {item}: 预测评分 {score:.2f}")
    
    print("\n" + "=" * 50)
    print("代码说明详见 GitHub: https://github.com/YOUR_USERNAME/xiaohongshu-codes")
    print("小红书：熙洲")
    print("=" * 50)


if __name__ == "__main__":
    main()
