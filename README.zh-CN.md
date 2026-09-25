# SAGE

AIIDE 2026 论文 **Switch-Aware Graph Planning Integrated with Gated Exploration on Unseen States for RTS Micromanagement** 的代码、数据和交互式面板。

[English README](README.md) · [完整操作与实验指南](ARTIFACT.md) · [论文图表输入索引](paper_assets/aiide26_sage/README.md)

项目重点是 **Graph-only SAGE** 和 **Full SAGE**：前者在经验转移图上进行允许备选分支切换的规划，后者加入面向不确定/未见状态的门控探索及局部动作价值模型。其他方法的论文报告值保存在对应的图表数据目录中。

## 论文配图与 Web

使用 **Python 3.12**，无需 SC2、GPU 或 PyTorch。从包含全部 Git LFS 对象的源码 checkout 运行；单独安装 wheel 不会包含论文数据与运行资产。

```bash
git lfs install
git clone https://github.com/BaiChunhui-9803/SAGE.git
cd SAGE
# Check out the artifact revision supplied with the submission, then:
git lfs pull
python -m venv .venv-artifact
# Windows: .venv-artifact\Scripts\activate
# Linux/macOS: source .venv-artifact/bin/activate
python -m pip install -r requirements-artifact.txt
python scripts/reproduce_paper.py --figure all
python -m streamlit run scripts/visualize_etg_web_en.py
```

图表写入 `output/paper_reproduction/figures/`。面板提供论文结果、配图、数据目录树、各场景经验转移图、束搜索与图上推演。图视图默认一跳、每个状态最多六个邻居，最多显示 24 个节点、60 条边。

## 新的 SAGE 对局

使用独立的 **Python 3.8.10 + requirements-sc2.txt** 环境，并安装本地 StarCraft II。启动器直接读取 `assets/maps/`，无需修改 PySC2 包或注册自定义地图。

```bash
python scripts/run_paper_sage.py --scenario sce-1 --method graph-only --episodes 2 --run
python scripts/run_paper_sage.py --scenario sce-1 --method full --episodes 2 --run
```

去掉 `--run` 只准备配置；使用 `--episodes 300` 开展新的完整评估。六场景为 `sce-1`、`sce-1m`、`sce-2`、`sce-2m`、`sce-3`、`sce-3m`。Graph-only 与 Full 分别读取其完整历史参数；Full 在初始模型副本上在线更新。输出包含逐局记录、参数、manifest 和 `completion.json`。新运行受游戏随机性影响，不等于论文冻结的 100 局样本。

完整安装、逐图复现、不同实验参数及输出核对方式见 [ARTIFACT.md](ARTIFACT.md)。冻结清单约 **1.98 GB**，四份距离矩阵首次展开增加约 **2.26 GB** 本地缓存；建议预留至少 **8 GB** 磁盘空间及 **8 GB RAM**，SC2 和 Python 环境另计。

代码许可证为 [MIT](LICENSE)。引用时注明 SAGE 论文和所用 artifact revision。
