import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

class LeastConfidentQuant():
    name = "least_confident"
    
    def quantify(self, X_eval, model, **kwargs):
        probs = model.predict_proba(X_eval)
        max_probs = np.max(probs, axis=1)
        return -max_probs
        
        
class EntropyQuant():
    name = "entropy"
    
    def quantify(self, X_eval, model, **kwargs):
        probs = model.predict_proba(X_eval)
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)
        return entropies
    
    
class BLOODQuant():
    name = "BLOOD"
    
    def quantify(self, X_eval, model, batch_size=64, path=False, estimator=True, **kwargs):
        norms = model.get_grad_layers(X_eval, batch_size=batch_size, estimator=estimator)  # (num_layers-1, batch_size)
        
        if path:
            return norms
        else:
            return norms.mean(axis=0)
        

class PCAResidualQuant():
    name = "PCA_Residual"

    def quantify(self, X_eval, X_anchor, model, batch_size=64, pca_variance_ratio=0.95, **kwargs):
        # 提取训练数据和测试数据的特征向量
        with torch.no_grad():
            # 获取训练数据的特征向量
            anchor_features = model.get_encoded(X_anchor, batch_size=batch_size).cpu().numpy()  # (N_train, feature_dim)
            # 获取测试数据的特征向量
            eval_features = model.get_encoded(X_eval, batch_size=batch_size).cpu().numpy()  # (N_test, feature_dim)
        
        # 对训练数据的特征向量进行PCA
        pca = PCA(n_components=pca_variance_ratio, svd_solver='full')
        pca.fit(anchor_features)
        
        # 重构测试数据的特征向量
        eval_features_reconstructed = pca.inverse_transform(pca.transform(eval_features))
        # 计算残差（原始特征 - 重构特征）
        residuals = eval_features - eval_features_reconstructed
        # 计算残差的范数作为OOD得分
        residual_scores = np.linalg.norm(residuals, axis=1)
        
        return residual_scores


class CombinedBLOODPCAQuant():
    name = "Combined_BLOOD_PCA"

    def quantify(self, X_eval, X_anchor, model, batch_size=64, pca_variance_ratio=0.95, estimator=True, **kwargs):
        # 不要在这里使用 torch.no_grad()
        model.eval()
        # 获取训练数据的梯度变化特征
        anchor_norms = model.get_grad_layers(X_anchor, batch_size=batch_size, estimator=estimator)  # (num_layers-1, N_train)
        anchor_features = anchor_norms.cpu().numpy().T  # 转置为(N_train, num_layers-1)

        # 获取测试数据的梯度变化特征
        eval_norms = model.get_grad_layers(X_eval, batch_size=batch_size, estimator=estimator)  # (num_layers-1, N_eval)
        eval_features = eval_norms.cpu().numpy().T  # 转置为(N_eval, num_layers-1)

        # 对训练数据的梯度特征进行PCA
        pca = PCA(n_components=pca_variance_ratio, svd_solver='full')
        pca.fit(anchor_features)

        # 重构测试数据的梯度特征
        eval_features_reconstructed = pca.inverse_transform(pca.transform(eval_features))
        # 计算残差（原始梯度特征 - 重构梯度特征）
        residuals = eval_features - eval_features_reconstructed
        # 计算残差的范数作为OOD得分
        residual_scores = np.linalg.norm(residuals, axis=1)

        return residual_scores

