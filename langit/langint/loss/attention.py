import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
class SimilarityAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, use_normalization=False):
        super(SimilarityAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)
        self.use_normalization = use_normalization  # 是否使用归一化

    def forward(self, global_embedding, local_embedding):
        """
        参数:
            global_embedding: [batch, 77, 768]
            local_embedding: [batch, 77, 768]

        返回:
            fused_embedding: [batch, 77, 768]
        """
        # 定义查询、键、值
        Q = local_embedding        # [batch, 77, 768]
        K = global_embedding          # [batch, 77, 768]
        V = global_embedding          # [batch, 77, 768]

        positions = torch.arange(global_embedding.size(1)).unsqueeze(0).unsqueeze(0)  # [1, 1, 77]
        distance = torch.abs(positions - positions.transpose(-2, -1)).float().to(local_embedding.device)  # [1, 77, 77]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale - distance * 770000  # alpha为调节距离惩罚的超参数
        # 计算缩放点积注意力得分
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, 77, 77]

        # 应用 softmax 获得注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch, 77, 77]

        # 计算注意力输出
        attended_output = torch.matmul(attention_weights, V)  # [batch, 77, 768]

        return attended_output
class SimilarityBasedFusion(nn.Module):
    def __init__(self):
        super(SimilarityBasedFusion, self).__init__()

    def forward(self, global_embedding, local_embedding):
        similarity = F.cosine_similarity(global_embedding, local_embedding, dim=-1)  # [batch, 77]
        # 线性映射，限制在 [0, 1]
        self.alpha = 1.0
        weight = (similarity * self.alpha + 1) / 2  # 假设 similarity 在 [-1, 1]
        weight = weight.clamp(0, 1).unsqueeze(-1)  # [batch, 77, 1]
        fused_embedding = weight * global_embedding + (1 - weight) * local_embedding
        return fused_embedding, weight
 # [batch, 77, 768]
        # return fused_embedding, weight

class SimilaritySelectionFusion(nn.Module):
    def __init__(self, similarity_metric='cosine', dynamic_threshold=True):
        """
        参数:
            similarity_metric (str): 使用的相似度度量方式 ('cosine' 或 'dot')
            dynamic_threshold (bool): 是否使用动态阈值，如果为False，将需要手动指定阈值
        """
        super(SimilaritySelectionFusion, self).__init__()
        if similarity_metric not in ['cosine', 'dot']:
            raise ValueError("similarity_metric must be either 'cosine' or 'dot'")
        self.similarity_metric = similarity_metric
        self.dynamic_threshold = dynamic_threshold
        if not self.dynamic_threshold:
            self.threshold = nn.Parameter(torch.tensor(0.8))  # 默认阈值为0.8，可根据需要调整

    def forward(self, global_embedding, local_embedding):
        """
        参数:
            global_embedding (torch.Tensor): [batch, seq_length, embedding_dim]
            local_embedding (torch.Tensor): [batch, seq_length, embedding_dim]
        
        返回:
            fused_embedding (torch.Tensor): [batch, seq_length, embedding_dim]
            selection_mask (torch.Tensor): [batch, seq_length] (1 表示选择全局嵌入，0 表示选择局部嵌入)
        """
        # 确保输入形状一致
        assert global_embedding.shape == local_embedding.shape, "Embeddings must have the same shape"
        
        if self.similarity_metric == 'cosine':
            # 计算余弦相似度，结果为 [batch, seq_length]
            similarity = F.cosine_similarity(global_embedding, local_embedding, dim=-1)  # [batch, seq_length]
            # 将相似度线性映射到 [0,1]
            similarity = (similarity + 1) / 2  # [batch, seq_length]
        elif self.similarity_metric == 'dot':
            # 计算点积相似度，结果为 [batch, seq_length]
            similarity = torch.sum(global_embedding * local_embedding, dim=-1)  # [batch, seq_length]
            # 使用 sigmoid 函数将相似度归一化到 [0,1]
            similarity = torch.sigmoid(similarity)  # [batch, seq_length]
        
        if self.dynamic_threshold:
            similarity_mean = similarity.mean(dim=1, keepdim=True)  # [batch, 1]
            similarity_std = similarity.std(dim=1, keepdim=True)    # [batch, 1]
            print(similarity_mean)
            print(similarity_std)
            threshold = similarity_mean +  similarity_std  # [batch, 1]
            # 可以限制阈值的最大值不超过1
            threshold = torch.clamp(threshold, max=1.0)
        else:
            # 使用手动指定的阈值
            threshold = self.threshold  # 如果是标量，自动广播到 [batch, seq_length]
        
        if self.dynamic_threshold:
            # 扩展阈值维度以匹配相似度的形状
            threshold = threshold  # [batch, 1] -> [batch, 1]
            # 对于每个 token，比较相似度与阈值
            selection_mask = (similarity >= threshold).float()  # [batch, seq_length]
        else:
            # 使用固定阈值，自动广播比较
            selection_mask = (similarity >= threshold).float()  # [batch, seq_length]
        
        # 扩展掩码维度以匹配嵌入向量
        selection_mask = selection_mask.unsqueeze(-1)  # [batch, seq_length, 1]
        
        # 选择融合嵌入
        fused_embedding = selection_mask * global_embedding + (1 - selection_mask) * local_embedding  # [batch, seq_length, embedding_dim]
        print("fusion",fused_embedding.shape)
        return fused_embedding, selection_mask.squeeze(-1)
class BidirectionalSimilarityAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, similarity_bias=0.5):
        super(BidirectionalSimilarityAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.similarity_bias = similarity_bias  # 固定的相似度偏置
        
    def forward(self, global_embedding, local_embedding):
        # 计算相似度
        global_proj = global_embedding
        local_proj = local_embedding
        similarity_g2l = torch.matmul(global_proj, local_proj.transpose(-1, -2))
        similarity_l2g = torch.matmul(local_proj, global_proj.transpose(-1, -2))
        
        # 增加相似度偏置
        similarity_l2g = similarity_l2g + self.similarity_bias  # 增加全局嵌入的相似度
        
        # 计算注意力权重
        attention_weights_g2l = F.softmax(similarity_g2l, dim=-1)
        attention_weights_l2g = F.softmax(similarity_l2g, dim=-1)
        
        # 加权嵌入
        attended_local = torch.matmul(attention_weights_g2l, local_embedding)
        attended_global = torch.matmul(attention_weights_l2g, global_embedding)
        
        # 双向融合（例如，拼接）
        fused_embedding = torch.cat([attended_local, attended_global], dim=1)
        return fused_embedding

class BidirectionalSimilarityAttentionFusion_(nn.Module):
    def __init__(self, embed_dim=768):
        super(BidirectionalSimilarityAttentionFusion_, self).__init__()
        self.embed_dim = embed_dim
        
    
    def forward(self, global_embedding, local_embedding):
        # 可选：通过线性层投影
        # global_proj = self.linear(global_embedding)
        # local_proj = self.linear(local_embedding)
        global_proj = global_embedding
        local_proj = local_embedding
        
        # 计算相似度
        similarity_g2l = torch.matmul(global_proj, local_proj.transpose(-1, -2))  # [batch, 77, 77]
        similarity_l2g = torch.matmul(local_proj, global_proj.transpose(-1, -2))  # [batch, 77, 77]
        
        # 注意力权重
        attention_weights_g2l = F.softmax(similarity_g2l, dim=-1)  # [batch, 77, 77]
        # print(attention_weights_g2l)
        attention_weights_l2g = F.softmax(similarity_l2g, dim=-1)  # [batch, 77, 77]
        
        # 加权嵌入
        attended_local = torch.matmul(attention_weights_g2l, local_embedding)  # [batch, 77, 768]
        attended_global = torch.matmul(attention_weights_l2g, global_embedding)  # [batch, 77, 768]
        
        # 双向融合
        # fused_embedding = attended_local
        # fused_embedding = (0.2*attended_local +  0.8*attended_global)  # [batch, 77, 768]
        fused_embedding = torch.cat([attended_local, attended_global], dim=1)
        return fused_embedding
class ConcatFusion(nn.Module):
    def __init__(self):
        super(ConcatFusion, self).__init__()           
    def forward(self, global_embedding, local_embedding):
        fused_embedding = torch.cat([global_embedding, local_embedding], dim=1)
        return fused_embedding
class ConcatAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768):
        super(ConcatAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        # Optional: Add a linear layer for projection if needed
        # self.linear = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, global_embedding, local_embedding):
        """
        Args:
            global_embedding: Tensor of shape [batch_size, seq_len, embed_dim]
            local_embedding: Tensor of shape [batch_size, seq_len, embed_dim]
        
        Returns:
            fused_embedding: Tensor of shape [batch_size, 2 * seq_len, embed_dim]
        """
        # Step 1: Concatenate embeddings along the sequence dimension
        # Resulting shape: [batch_size, 2 * seq_len, embed_dim]
        combined = torch.cat([global_embedding, local_embedding], dim=1)
        
        # Optional: Project the combined embeddings if a linear transformation is desired
        # combined = self.linear(combined)  # Shape: [batch_size, 2 * seq_len, embed_dim]
        
        # Step 2: Compute similarity scores for attention
        # Similarity shape: [batch_size, 2 * seq_len, 2 * seq_len]
        similarity = torch.matmul(combined, combined.transpose(-1, -2))
        
        # Step 3: Apply softmax to obtain attention weights
        attention_weights = F.softmax(similarity, dim=-1)
        
        # Step 4: Apply attention weights to the combined embeddings
        # Fused embedding shape: [batch_size, 2 * seq_len, embed_dim]
        fused_embedding = torch.matmul(attention_weights, combined)
        
        return fused_embedding
