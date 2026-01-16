import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768):
        super(SimilarityAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        
        # 线性层用于投影（可选）
        self.linear = nn.Linear(embed_dim, embed_dim)
        
        # 初始化权重并冻结参数（如果使用线性层）
        self._init_weights()
        self._freeze_parameters()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def _freeze_parameters(self):
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, global_embedding, local_embedding):
        """
        参数:
            global_embedding: [batch, 77, 768]
            local_embedding: [batch, 77, 768]
        
        返回:
            fused_embedding: [batch, 77, 768]
        """
        # 可选：通过线性层投影
        # global_proj = self.linear(global_embedding)  # [batch, 77, 768]
        # local_proj = self.linear(local_embedding)    # [batch, 77, 768]
        global_proj = global_embedding
        local_proj = local_embedding
        # 计算点积相似度
        similarity = torch.matmul(global_proj, local_proj.transpose(-1, -2))  # [batch, 77, 77]
        # similarity = torch.matmul(local_proj, local_proj.transpose(-1, -2))
        # 计算注意力权重（按行softmax）
        attention_weights = F.softmax(similarity, dim=-1)  # [batch, 77, 77]
        
        # 使用注意力权重对局部嵌入进行加权
        attended_local = torch.matmul(attention_weights, local_embedding)  # [batch, 77, 768]
        # attended_global = torch.matmul(attention_weights, global_embedding)
        # 融合全局和注意后的局部嵌入
        # fused_embedding = attended_global 

        fused_embedding = (global_embedding + attended_local) / 2# [batch, 77, 768]
        # fused_embedding = concatenated = torch.cat([global_embedding, local_embedding], dim=-1)  # 形状: [batch_size, 77, 2*embed_dim]
        
        # 2. 在通道维度上进行平均池化
        # 通过对最后一个维度进行池化，将通道数从 2*embed_dim 压缩回 embed_dim
        # fused_embedding = F.adaptive_avg_pool1d(concatenated, 768)
        print(fused_embedding.shape)

        # fused_embedding = (local_embedding + attended_global)/2
        return fused_embedding
    

class BidirectionalSimilarityAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768):
        super(BidirectionalSimilarityAttentionFusion, self).__init__()
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
        attention_weights_l2g = F.softmax(similarity_l2g, dim=-1)  # [batch, 77, 77]
        
        # 加权嵌入
        attended_local = torch.matmul(attention_weights_g2l, local_embedding)  # [batch, 77, 768]
        attended_global = torch.matmul(attention_weights_l2g, global_embedding)  # [batch, 77, 768]
        
        # 双向融合
        # fused_embedding = (attended_local +  attended_global) /2  # [batch, 77, 768]
        fused_embedding = torch.cat([attended_local, attended_global], dim=1)
        return fused_embedding
class ConcatFusion(nn.Module):
    def __init__(self):
        super(ConcatFusion, self).__init__()           
    def forward(self, global_embedding, local_embedding):
        fused_embedding = torch.cat([global_embedding, local_embedding], dim=1)
        return fused_embedding
# 示例用法
if __name__ == "__main__":
    batch_size = 16
    global_embedding = torch.randn(batch_size, 77, 768)
    local_embedding = torch.randn(batch_size, 77, 768)
    
    fusion_layer = SimilarityAttentionFusion()
    fused_embedding = fusion_layer(global_embedding, local_embedding)
    print(fused_embedding.shape)  # 应输出: torch.Size([16, 77, 768])
