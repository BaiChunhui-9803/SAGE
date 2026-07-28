import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# --- 全局设置字体为 Times New Roman ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix' # 让数学公式也接近 Times 风格

def load_and_agg(directory, pattern):
    """搜索匹配模式的所有文件并计算均值和标准差 """
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    df_list = [pd.read_csv(f) for f in files]
    combined = pd.concat(df_list)
    # 按 Episode 分组计算统计指标
    stats = combined.groupby('Episode')['Score'].agg(['mean', 'std']).reset_index()
    return stats

def plot_decision_results(directory='output/data/decision'):
    # 1. 读取 Loss 数据 (单样本)
    df_loss_with = pd.read_csv(os.path.join(directory, 'train_loss_with_cause_mask.txt'))
    df_loss_without = pd.read_csv(os.path.join(directory, 'train_loss_without_cause_mask.txt'))

    # 2. 读取并聚合 Score 数据 (多样本)
    # 匹配 result_with_cause_mask_1.txt, _2.txt 等
    stats_with = load_and_agg(directory, 'result_with_cause_mask_*.txt')
    stats_without = load_and_agg(directory, 'result_without_cause_mask_*.txt')

    # 3. 创建画布，使用 constrained_layout 自动管理边距，解决留白过大问题
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # --- 左侧图：Loss 曲线 ---
    l1, = ax1.plot(df_loss_with['Epoch'], df_loss_with['Average_Loss'],
                   label='With Cause Mask', color='#1f77b4', linewidth=1.5)
    l2, = ax1.plot(df_loss_without['Epoch'], df_loss_without['Average_Loss'],
                   label='Without Cause Mask', color='#ff7f0e', linewidth=1.5)
    ax1.set_title('Training Loss', pad=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- 右侧图：Score 曲线 (带误差带) ---
    if stats_with is not None:
        # 绘制均值实线
        ax2.plot(stats_with['Episode'], stats_with['mean'], color='#1f77b4', linewidth=1.5)
        # 绘制标准差填充区域
        ax2.fill_between(stats_with['Episode'],
                         stats_with['mean'] - stats_with['std'],
                         stats_with['mean'] + stats_with['std'],
                         color='#1f77b4', alpha=0.2)

    if stats_without is not None:
        ax2.plot(stats_without['Episode'], stats_without['mean'], color='#ff7f0e', linewidth=1.5)
        ax2.fill_between(stats_without['Episode'],
                         stats_without['mean'] - stats_without['std'],
                         stats_without['mean'] + stats_without['std'],
                         color='#ff7f0e', alpha=0.2)

    ax2.set_title('Episode Scores (Mean ± Std)', pad=10)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 4. 设置共用图例 (置于顶部，不遮挡标题)
    # 使用 bbox_to_anchor 定位在画布顶端中央
    fig.legend(handles=[l1, l2], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize='large')

    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.84, wspace=0.1)

    plt.show()
    print('Finished')

# 调用函数
# plot_decision_results()