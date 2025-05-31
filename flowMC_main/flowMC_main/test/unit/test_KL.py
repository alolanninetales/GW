from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.special import kl_div
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp  # JAX NumPy
def compute_model_kl(true_data, model_samples, model):
    """
    计算真实数据分布与模型分布之间的KL散度
    
    参数:
        true_data: 真实数据样本 (jax.Array)
        model_samples: 模型生成的样本 (jax.Array)
        model: 训练好的流模型
        
    返回:
        kl_divergence: KL散度估计值
    """
    # 转换为NumPy数组
    true_data_np = np.array(true_data)
    model_samples_np = np.array(model_samples)
    
    # 使用KDE估计真实分布
    kde_true = KernelDensity(kernel='gaussian', bandwidth='scott')
    kde_true.fit(true_data_np)
    
    # 使用KDE估计模型分布
    kde_model = KernelDensity(kernel='gaussian', bandwidth='scott')
    kde_model.fit(model_samples_np)
    
    # 准备评估点 - 组合真实数据和模型样本
    eval_points = np.vstack([true_data_np, model_samples_np])
    
    # 计算真实分布的对数概率密度 (KDE估计)
    log_p_true = kde_true.score_samples(eval_points)
    p_true = np.exp(log_p_true)
    
    # 计算模型分布的对数概率密度 (KDE估计)
    log_q_model = kde_model.score_samples(eval_points)
    q_model = np.exp(log_q_model)
    
    # 避免零概率问题
    epsilon = 1e-10
    p_true = np.clip(p_true, epsilon, None)
    q_model = np.clip(q_model, epsilon, None)
    
    # 计算KL散度
    kl_values = kl_div(p_true, q_model)
    kl_divergence = np.mean(kl_values)
    
    return kl_divergence
# def calculate_kl(data, samples, model, n_dim):
#     """计算高维连续分布的KL散度"""
#     # 将数据转换为numpy数组
#     data_np = np.array(data)
#     samples_np = np.array(samples)
    
#     # 创建核密度估计（适用于低维数据）
#     if n_dim <= 4:
#         kde = gaussian_kde(data_np.T)
#         log_p = kde.logpdf(samples_np.T)
#     else:  # 高维时使用流模型估计真实分布
#         raise NotImplementedError("高维数据建议使用归一化流模型估计真实分布")
    
#     # 获取生成样本的概率
#     log_q = model.log_prob(samples)
    
#     # 计算KL散度（D_KL(P||Q) = E_P[log(p/q)] ≈ mean(log_p - log_q))
#     kl_divergence = np.mean(log_p - log_q)
#     return kl_divergence
def test_kl_computation():
    """验证KL散度计算函数的正确性"""
    print("=== 验证KL计算函数 ===")
    
    # 测试1: 相同分布 - KL应接近0
    print("\n测试1: 相同分布 (应接近0)")
    key = jr.key(42)
    data = jr.normal(key, (10000, 4))  # 标准正态分布
    
    # 创建"伪模型"
    class MockModel:
        def log_prob(self, x):
            return jax.scipy.stats.norm.logpdf(x).sum(axis=1)
    
    model = MockModel()
    
    # 生成模型样本（与真实数据相同）
    kl_same = compute_model_kl(data, data, model)
    print(f"相同分布的KL散度: {kl_same:.6f} (应接近0)")
    
    # 测试2: 不同分布 - 应有明显差异
    print("\n测试2: 不同分布 (应有明显差异)")
    # 创建明显不同的分布（均值和方差都不同）
    shifted_data = data * 1.5 + 2.0
    kl_diff = compute_model_kl(data, shifted_data, model)
    print(f"不同分布的KL散度: {kl_diff:.6f} (应显著大于0)")
    
    # 测试3: 已知KL值的分布
    print("\n测试3: 已知KL值的分布")
    # 创建两个高斯混合模型
    def create_gmm(means, covs, weights, n_samples):
        gmm = GaussianMixture(
            n_components=len(means),
            covariance_type='full',
            means_init=means,
            weights_init=weights
        )
        # 设置协方差矩阵
        gmm.covariances_ = covs
        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
        samples, _ = gmm.sample(n_samples)
        return jnp.array(samples)
    
    # 定义两个不同的GMM
    means1 = np.array([[0, 0, 0, 0], [2, 2, 2, 2]])
    covs1 = np.array([np.eye(4), np.eye(4)])
    weights1 = np.array([0.7, 0.3])
    
    means2 = np.array([[0, 0, 0, 0], [3, 3, 3, 3]])
    covs2 = np.array([np.eye(4) * 0.5, np.eye(4) * 1.5])
    weights2 = np.array([0.5, 0.5])
    
    data1 = create_gmm(means1, covs1, weights1, 10000)
    data2 = create_gmm(means2, covs2, weights2, 10000)
    
    # 计算KL散度
    kl_gmm = compute_model_kl(data1, data2, model)
    print(f"不同GMM分布的KL散度: {kl_gmm:.6f} (应大于0.5)")
    
    # 测试4: 使用数值积分验证一维情况
    print("\n测试4: 一维数值积分验证")
    # 定义两个一维高斯分布
    mean1, std1 = 0.0, 1.0
    mean2, std2 = 1.0, 1.0
    
    # 理论KL值: KL(N(0,1)||N(1,1)) = 0.5
    x = np.linspace(-5, 5, 10000).reshape(-1, 1)
    p = np.exp(jax.scipy.stats.norm.logpdf(x, mean1, std1))
    q = np.exp(jax.scipy.stats.norm.logpdf(x, mean2, std2))
    
    # 数值积分计算KL散度
    kl_integral = np.trapz(p * np.log(p / q), x.flatten())
    
    # 使用我们的函数计算
    samples1 = jr.normal(key, (10000, 1)) * std1 + mean1
    samples2 = jr.normal(key, (10000, 1)) * std2 + mean2
    kl_our = compute_model_kl(samples1, samples2, model)
    
    print(f"数值积分KL值: {kl_integral:.6f}")
    print(f"我们的函数计算值: {kl_our:.6f}")
    print(f"理论值: 0.500000")
    print(f"差异: {abs(kl_our - 0.5):.6f}")

# 运行验证测试
test_kl_computation()