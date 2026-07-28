from utils.calculate_utils import *
from scipy.interpolate import griddata

import json, math, os, re, time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"plotly\.io\._kaleido")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="plotly")

import plotly
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_color_from_value(value, min_v, max_v):
    """
    根据给定的数值、最大最小值，按照自定义的蓝-白-红比例返回 Hex 颜色
    """
    # 颜色定义 (与你的 custom_colorscale 保持一致)
    color_min = '#5c7ee6'  # 蓝色
    color_zero = '#ebebeb'  # 白色 (中性)
    color_max = '#b62d0a'  # 红色

    # 边界检查
    if value <= min_v: return color_min
    if value >= max_v: return color_max

    # 辅助函数：执行两个 Hex 颜色之间的线性插值
    def lerp_hex(c1, c2, f):
        rgb1 = np.array(mcolors.to_rgb(c1))
        rgb2 = np.array(mcolors.to_rgb(c2))
        res_rgb = rgb1 * (1 - f) + rgb2 * f
        return mcolors.to_hex(res_rgb)

    # 1. 全正区间：白 -> 红
    if min_v >= 0:
        frac = (value - min_v) / (max_v - min_v)
        return lerp_hex(color_zero, color_max, frac)

    # 2. 全负区间：蓝 -> 白
    if max_v <= 0:
        frac = (value - min_v) / (max_v - min_v)
        return lerp_hex(color_min, color_zero, frac)

    # 3. 跨越 0 点的双分段插值逻辑
    if value < 0:
        # 在 [min_v, 0] 区间内插值 (蓝 -> 白)
        frac = (value - min_v) / (0 - min_v)
        return lerp_hex(color_min, color_zero, frac)
    else:
        # 在 [0, max_v] 区间内插值 (白 -> 红)
        frac = (value - 0) / (max_v - 0)
        return lerp_hex(color_zero, color_max, frac)

def plot_single_trajectory_pdf(idx, fitness_color, sample, coords, xi, yi, grid_z, custom_colorscale, save_dir):
    """
    封装函数：绘制单条轨迹对比图并保存为 PDF
    """
    fig = go.Figure()

    # 1. 绘制背景地形等高线
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=grid_z,
        colorscale=custom_colorscale,
        showscale=False,
        # contours=dict(coloring='fill'),
        contours=dict(coloring='heatmap'),
        # contours=dict(coloring='lines'),
        connectgaps=False,
        opacity=0.7,
        line=dict(width=0.00),
        hoverinfo='skip'
    ))

    # 坐标提取辅助闭包
    def get_coords(s_list):
        pts = coords[s_list]
        return pts[:, 0], pts[:, 1]

    # 2. 绘制三条核心轨迹线
    # A. 历史输入轨迹 (History)
    in_x, in_y = get_coords(sample['input'])
    fig.add_trace(go.Scatter(
        x=in_x, y=in_y, mode='markers+lines',
        name='History',
        line=dict(color=f'{fitness_color}', width=2),
        marker=dict(size=6, color=f'{fitness_color}', symbol='circle')
    ))

    # B. 真实未来轨迹 (Actual) - 连接历史终点
    act_full_x, act_full_y = get_coords([sample['input'][-1]] + list(sample['actual']))
    fig.add_trace(go.Scatter(
        x=act_full_x, y=act_full_y, mode='lines+markers',
        name='Actual',
        line=dict(color='#16a085', width=3, dash='dot'),
        marker=dict(size=8, color='#16a085', symbol='circle',
                    line=dict(width=0.5, color='white'))
    ))

    # C. 模型预测轨迹 (Prediction) - 连接历史终点
    pred_full_x, pred_full_y = get_coords([sample['input'][-1]] + list(sample['preds']))
    fig.add_trace(go.Scatter(
        x=pred_full_x, y=pred_full_y, mode='lines+markers',
        name='Prediction',
        line=dict(color='#8e44ad', width=3),
        marker=dict(size=8, color='#8e44ad', symbol='diamond',
                    line=dict(width=0.5, color='white'))
    ))

    # 3. 布局设置 (针对学术 PDF 优化)
    x_min, x_max, y_min, y_max = xi.min(), xi.max(), yi.min(), yi.max()
    pad = 0.05
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Times New Roman', size=18, color='black'),
        xaxis=dict(range=[x_min-pad, x_max+pad], visible=False, constrain='domain'),
        yaxis=dict(range=[y_min-pad, y_max+pad], visible=False, constrain='domain'),
        margin=dict(l=5, r=5, t=5, b=5),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.98, xanchor="left", x=0.02,
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="Black", borderwidth=1
        ),
        width=600, height=600
    )

    # 4. 导出 PDF
    save_path = os.path.join(save_dir, f"traj_analysis_{idx}.pdf")
    try:
        fig.write_image(save_path, format="pdf", scale=3)
        return True
    except Exception as e:
        logger.error(f"导出 Traj_{idx} 失败: {e}")
        return False

def batch_plot_pred_FL(pred_results_dict, state_distance_matrix, state_value, save_dir="cache/pdf_plots"):
    """
    主入口：计算公共背景数据并批量调用封装好的绘图函数
    """
    # 1. 计算所有图共用的 MDS 坐标和地形背景
    coords = get_or_create_mds_coords(state_distance_matrix)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    xi = np.linspace(x_min, x_max, 250)
    yi = np.linspace(y_min, y_max, 250)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata(points=coords, values=state_value, xi=(grid_x, grid_y), method='cubic')

    # 计算公共颜色映射
    min_v, max_v = np.min(state_value), np.max(state_value)
    zero_pos = 0.5 if (min_v >= 0 or max_v <= 0) else (0 - min_v) / (max_v - min_v)
    custom_colorscale = [[0, '#5c7ee6'], [zero_pos, '#ebebeb'], [1, '#b62d0a']]
    # grid_z = np.nan_to_num(grid_z, nan=zero_pos)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 批量调用
    count = 0
    for idx, sample in pred_results_dict.items():
        color = get_color_from_value(pred_results_dict[idx]['actual_fitness'], min_v, max_v)
        success = plot_single_trajectory_pdf(
            idx, color, sample, coords, xi, yi, grid_z, custom_colorscale, save_dir
        )
        if success: count += 1

    logger.info(f"批量处理完成：成功导出 {count}/{len(pred_results_dict)} 个 PDF 文件至 {save_dir}")

def plot_prediction_results(save_path=f"../output/figures/comparison/prediction_results/prediction_results.png"):
    files = {
        "Vanilla": f"../output/data/trajTransformer/prediction_results_N{N}_K{K}.xlsx",
        # "W/ SP": f"../output/data/{model_name}/prediction_results_sp.xlsx",
        "W/ IEF": f"../output/data/trajTransformer/prediction_results_embedF_N{N}_K{K}.xlsx",
        "W/ IET": f"../output/data/trajTransformer/prediction_results_embedT_N{N}_K{K}.xlsx",
        "W/ SG IEF": f"../output/data/sgTransformer/prediction_results_embedF_N{N}_K{K}.xlsx",
        "W/ SG IET": f"../output/data/sgTransformer/prediction_results_embedT_N{N}_K{K}.xlsx",
        # "W/ SP+IEF": f"../output/data/{model_name}/prediction_results_sp_embedF.xlsx",
        # "W/ SP+IET": f"../output/data/{model_name}/prediction_results_sp_embedT.xlsx",
    }

    plt.figure(figsize=(6, 6))

    for label, path in files.items():
        try:
            # 只读取 'Index' 用于排序，和 'Avg_Error' 用于绘图
            # 这样即使其他列（如路径列）格式混乱，也不会干扰读取
            df = pd.read_excel(path, usecols=['Avg_Error'], engine='openpyxl')

            # 排序以确保折线连续
            # df = df.sort_values(by='Avg_Error')

            avg_err = df['Avg_Error'].tolist()
            sort_avg_err = sorted(df['Avg_Error'].tolist())
            plt.plot(sort_avg_err, label=label, linewidth=1.5)
            print(f"Success: {label} loaded.")

        except Exception as e:
            print(f"Error loading {label}: {e}")

    # 图表精修
    plt.title('Ablation Study: Average Error Comparison', fontsize=12)
    plt.xlabel('Sample Index', fontsize=10)
    plt.ylabel('Avg Error', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, dpi=300)

def plot_training_metrics(file_configs, save_path=f"../output/figures/comparison/training_metrics/training_comparison.png"):
    """
    批量读取指标文件并绘制对比图
    :param file_configs: 字典列表，包含文件名和对应的显示标签
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for config in file_configs:
        file_name = config['file']
        label = config['label']

        if not os.path.exists(file_name):
            print(f"警告: 文件 {file_name} 不存在，已跳过。")
            continue

        # 读取数据
        df = pd.read_csv(file_name)
        epochs = df['Epoch']

        # 绘制左图：Train Loss (反映模型拟合速度)
        ax1.plot(epochs, df['Train_Loss'], label=f'{label}', marker='o', markersize=4)

        # 绘制右图：Total Mean Err (反映物理空间预测精度)
        ax2.plot(epochs, df['Total_Mean_Err'], label=f'{label}', marker='s', markersize=4)

    # 图表细节配置
    ax1.set_title("Training Loss Comparison", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.set_title("Total Mean Physical Error Comparison", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Distance Error")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()
    print(f"对比图已保存至: {save_path}")

def plot_all_matching_trajectories(res, msk, k, pred_seq, state_distance_matrix, state_value, save_path):
    """
    绘制多条匹配轨迹（Mask 0 & 1）以及模型给出的预测序列（pred_seq）。
    """
    # 1. 准备坐标与地形数据
    coords = get_or_create_mds_coords(state_distance_matrix)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    xi = np.linspace(x_min, x_max, 250)
    yi = np.linspace(y_min, y_max, 250)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata(points=coords, values=state_value, xi=(grid_x, grid_y), method='cubic')

    min_v, max_v = np.min(state_value), np.max(state_value)
    zero_pos = 0.5 if (min_v >= 0 or max_v <= 0) else (0 - min_v) / (max_v - min_v)
    custom_colorscale = [[0, '#5c7ee6'], [zero_pos, '#ebebeb'], [1, '#b62d0a']]

    fig = go.Figure()

    # 2. 绘制背景地形
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=grid_z,
        colorscale=custom_colorscale,
        showscale=True,
        contours=dict(coloring='heatmap'),
        opacity=0.6,
        line=dict(width=0),
        hoverinfo='skip'
    ))

    def get_coords(s_list):
        if not s_list: return [], []
        # 防止 pred_seq 包含 PAD_ID
        valid_list = [s for s in s_list if s < len(coords)]
        pts = coords[valid_list]
        return pts[:, 0], pts[:, 1]

    # 3. 循环绘制真实轨迹 (Ground Truth)
    last_query_end_coord = None  # 用于定位预测路径的起点

    for i, (traj, msk_list) in enumerate(zip(res, msk)):
        query_part = [s for s, m in zip(traj, msk_list) if m == 0]
        full_suffix = [s for s, m in zip(traj, msk_list) if m == 1]
        suffix_part = full_suffix[:k]

        # 绘制 Mask 0 (匹配序列)
        q_x, q_y = get_coords(query_part)
        fig.add_trace(go.Scatter(
            x=q_x, y=q_y,
            mode='lines',
            line=dict(color='rgba(160, 160, 160, 0.5)', width=2),  # 弱化处理真实 query 部分
            showlegend=True if i == 0 else False,
            name='Matched Query (GT)',
            legendgroup='query'
        ))

        # 记录最后一个 Query 的坐标作为预测起点（假设所有 query 相同）
        if i == 0 and query_part:
            last_query_end_coord = query_part[-1]

        # 绘制前 k 步 Suffix Flow (实际发生)
        if suffix_part:
            flow_seq = [query_part[-1]] + suffix_part
            s_x, s_y = get_coords(flow_seq)
            fig.add_trace(go.Scatter(
                x=s_x, y=s_y,
                mode='lines+markers',
                marker=dict(
                    size=8,  # 统一标记大小
                    symbol='circle',  # 真实轨迹使用圆形
                    color=f'rgba(22, 160, 133, 0.3)',
                    # line=dict(width=0.5, color='white')
                ),
                line=dict(
                    dash='dash',
                    color=f'rgba(22, 160, 133, 0.3)',
                    width=3  # 统一线宽
                ),
                showlegend=True if i == 0 else False,
                name=f'Actual Flow (Next {k})',
                legendgroup='actual'
            ))

    # 4. 绘制模型预测路径 (Prediction)
    if pred_seq and last_query_end_coord is not None:
        # 连接 Query 的末尾和预测序列
        full_pred_path = [last_query_end_coord] + list(pred_seq[:k])
        p_x, p_y = get_coords(full_pred_path)

        fig.add_trace(go.Scatter(
            x=p_x, y=p_y,
            mode='lines+markers',  # 统一为点线模式
            marker=dict(
                size=8,  # 统一标记大小
                symbol='diamond',  # 预测轨迹使用菱形
                color=f'rgba(142, 68, 173, 0.5)',
                opacity=0.5,
                # line=dict(width=0.5, color='white')
            ),
            line=dict(
                color=f'rgba(142, 68, 173, 0.5)',
                width=3  # 统一线宽
            ),
            name='Model Prediction',
        ))

    # 5. 布局优化
    fig.update_layout(
        title=f"Prediction vs Reality Analysis (K={k}, N_Samples={len(res)})",
        template='plotly_white',
        font=dict(family='Times New Roman', size=16),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(
            yanchor="top", y=0.98, xanchor="left", x=0.02,
            bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="Black", borderwidth=1
        ),
        width=900, height=800
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_image(save_path, format="pdf", scale=2)
    print(f"预测对比图已保存至: {save_path}")






if __name__ == "__main__":
    plot_prediction_results()

    configs = [
        {
            'file': f'../output/data/trajTransformer/test_metrics_step_wise_N{N}_K{K}.txt',
            'label': 'Baseline (Vanilla)'
        },
        {
            'file': f'../output/data/trajTransformer/test_metrics_step_wise_embedF_N{N}_K{K}.txt',
            'label': 'Init Embedding (Freeze)'
        },
        {
            'file': f'../output/data/trajTransformer/test_metrics_step_wise_embedT_N{N}_K{K}.txt',
            'label': 'Init Embedding (Train)'
        },
        {
            'file': f'../output/data/sgTransformer/test_metrics_step_wise_N{N}_K{K}.txt',
            'label': 'SG Init Embedding (Freeze)'
        },
        {
            'file': f'../output/data/sgTransformer/test_metrics_step_wise_embedT_N{N}_K{K}.txt',
            'label': 'SG Init Embedding (Train)'
        },
    ]
    plot_training_metrics(configs)