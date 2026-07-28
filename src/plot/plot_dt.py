import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_dt_evaluation_results(stats_results, save_path="output/dt_eval_plot.png"):
    """
    根据统计结果绘制 Fitness vs Accuracy 的分析图
    """
    df = pd.DataFrame(stats_results)

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 图 1：Fitness 与 Accuracy 的关系
    sns.regplot(data=df, x='fitness', y='accuracy', ax=ax1,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    ax1.set_title('Correlation: Fitness vs Action Accuracy', fontsize=14)
    ax1.set_xlabel('Trajectory Fitness', fontsize=12)
    ax1.set_ylabel('Prediction Accuracy (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 图 2：Accuracy 的分布情况
    sns.histplot(df['accuracy'], bins=20, kde=True, ax=ax2, color='skyblue')
    ax2.set_title('Distribution of Prediction Accuracy', fontsize=14)
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"统计图表已保存至: {save_path}")
    plt.show()