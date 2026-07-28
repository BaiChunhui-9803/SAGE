def _build_live_game_html(port: int = 8000, host: str = "localhost") -> str:
    _html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
:root{--bg:#0e1117;--sf:#262730;--sf2:#1e1f26;--bd:#3d3f4a;--tx:#fafafa;--tx2:#a3a8b8;
--ac:#4fc3f7;--grn:#4caf50;--ylw:#ffc107;--red:#f44336;--rd:8px}
.light{--bg:#f5f5f5;--sf:#ffffff;--sf2:#eeeeee;--bd:#d0d0d0;--tx:#1a1a1a;--tx2:#666666;
--ac:#0288d1;--grn:#2e7d32;--ylw:#f57f17;--red:#d32f2f;--rd:8px}
.light .console-body{background:#fff;color:#333}
.light .log-line{border-bottom-color:rgba(0,0,0,.08)}
.light .log-ts{color:#999}
.light .lv-info{color:#666}.light .lv-success{color:#2e7d32}.light .lv-warn{color:#e65100}.light .lv-error{color:#c62828}
.light .src-api{background:#e3f2fd;color:#1565c0}
.light .src-game{background:#eceff1;color:#546e7a}
.light .badge-gray{background:#e0e0e0;color:#616161}
.light .badge-green{background:#e8f5e9;color:#2e7d32}
.light .badge-yellow{background:#fff8e1;color:#f57f17}
.light .btn-primary{background:#0288d1;color:#fff}
.light .btn-ghost{background:#eee;color:#333;border-color:#ccc}
.light .toggle .slider{background:#bbb}
.light .toast-ok{background:#e8f5e9;color:#1b5e20}
.light .lv-debug{color:#00796b}
.light .src-debug{background:#00796b;color:#b2dfdb}
.light .toast-err{background:#ffebee;color:#b71c1c}
.toast-wrap{position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:9999;display:flex;flex-direction:column;gap:6px;align-items:center;pointer-events:none}
.toast{padding:8px 20px;border-radius:6px;font-size:13px;font-weight:600;pointer-events:auto;animation:toast-in .2s ease;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.toast-ok{background:#1b5e20;color:#a5d6a7}
.toast-err{background:#b71c1c;color:#ffcdd2}
@keyframes toast-in{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Source Sans Pro',-apple-system,sans-serif;background:var(--bg);color:var(--tx);font-size:14px;line-height:1.6;overflow:auto;min-height:100vh}
.root{display:flex;min-height:100vh}
.left{flex:1;padding:16px;min-width:0;border-right:1px solid var(--bd)}
.right{width:480px;display:flex;flex-direction:column;flex-shrink:0;position:sticky;top:0;height:100vh}
.console-header{padding:10px 14px;background:var(--sf);border-bottom:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;flex-shrink:0;position:relative}
.console-header span{font-size:13px;font-weight:600;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px}
.filter-panel{position:absolute;top:100%;right:0;z-index:100;background:var(--sf);border:1px solid var(--bd);border-radius:var(--rd);padding:12px;min-width:200px;display:none;box-shadow:0 4px 12px rgba(0,0,0,.3)}
.filter-panel.show{display:block}
.filter-panel h4{font-size:12px;color:var(--tx2);margin:0 0 8px;font-weight:600}
.filter-group{margin-bottom:10px}
.filter-group:last-child{margin-bottom:0}
.filter-group label{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--tx);padding:2px 0;cursor:pointer}
.filter-group label:hover{color:var(--ac)}
.filter-group input[type=checkbox]{accent-color:var(--ac)}
.console-body{flex:1;overflow-y:auto;padding:8px 10px;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:12px;line-height:1.8}
.log-line{display:flex;gap:8px;padding:1px 0;border-bottom:1px solid rgba(61,63,74,.3)}
.log-ts{color:#666;flex-shrink:0;width:60px;text-align:right}
.log-src{flex-shrink:0;min-width:36px;padding:0 8px;font-weight:700;text-align:center;border-radius:3px;white-space:nowrap}
.src-api{background:#1565c0;color:#90caf9}
.src-game{background:#37474f;color:#b0bec5}
.log-msg{flex:1;word-break:break-all}
.lv-info{color:var(--tx2)}.lv-success{color:#a5d6a7}.lv-warn{color:#fff9c4}.lv-error{color:#ef9a9a}.lv-debug{color:#80cbc4}
.console-footer{padding:6px 14px;background:var(--sf);border-top:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;flex-shrink:0;font-size:12px;color:var(--tx2)}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:var(--rd);padding:14px;margin-bottom:10px}
.card-title{font-size:13px;font-weight:600;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px}
.metrics{display:flex;gap:10px;flex-wrap:wrap}
.metric{flex:1;min-width:70px;background:var(--sf2);border-radius:6px;padding:8px 10px}
.metric-label{font-size:11px;color:var(--tx2);margin-bottom:2px}
.metric-value{font-size:18px;font-weight:700}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600}
.badge-green{background:#1b5e20;color:#a5d6a7}
.badge-yellow{background:#f57f17;color:#fff9c4}
.badge-gray{background:#424242;color:#9e9e9e}
.btn{border:none;border-radius:6px;padding:7px 14px;font-size:13px;font-weight:600;cursor:pointer;transition:opacity .15s}
.btn:hover{opacity:.85}.btn:active{opacity:.7}
.btn-primary{background:#4fc3f7;color:#fff}
.btn-blue-dark{background:#0288d1;color:#fff}
.btn-step{background:#4fc3f7;color:#fff}
.btn-danger{background:#f44336;color:#fff}
.btn-warn{background:#ff9800;color:#fff}
.btn-ok{background:#4caf50;color:#fff}
.btn-ghost{background:var(--sf2);color:var(--tx);border:1px solid var(--bd)}
.btn-sm{padding:5px 16px;font-size:12px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.row-between{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.card{background:var(--sf2);border-radius:8px;padding:12px;margin-bottom:10px}
.card-title{font-size:13px;font-weight:600;color:var(--tx2);margin-bottom:8px}
.toggle-wrap{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--tx2)}
.toggle{position:relative;width:36px;height:20px;cursor:pointer}
.toggle input{opacity:0;width:0;height:0}
.toggle .slider{position:absolute;inset:0;background:#424242;border-radius:10px;transition:.2s}
.toggle .slider:before{content:"";position:absolute;width:16px;height:16px;left:2px;top:2px;background:#fff;border-radius:50%;transition:.2s}
.toggle input:checked+.slider{background:var(--ac)}
.toggle input:checked+.slider:before{transform:translateX(16px)}
.ep-item{border:1px solid var(--bd);border-radius:6px;padding:8px 10px;margin-bottom:6px;background:var(--sf)}
.ep-item:hover{border-color:var(--ac)}
.ep-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.ep-title{font-size:13px;font-weight:600;color:var(--tx)}
.ep-badge{padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600}
.ep-badge-win{background:#1b5e20;color:#a5d6a7}
.ep-badge-loss{background:#b71c1c;color:#ef9a9a}
.ep-badge-interrupted{background:#424242;color:#9e9e9e}
.ep-badge-dogfall{background:#e65100;color:#ffe0b2}
.ep-meta{font-size:12px;color:var(--tx2);margin-bottom:4px}
.ep-export-btns{margin-left:auto;display:flex;gap:3px;flex-shrink:0}
.ep-export-btns button{font-size:10px;padding:1px 6px;border-radius:3px;border:1px solid var(--bd);background:var(--sf2);color:var(--tx2);cursor:pointer;font-weight:600;line-height:1.4}
.ep-export-btns button:hover{background:var(--ac);color:#fff;border-color:var(--ac)}
.ep-flow{display:flex;flex-wrap:wrap;gap:2px;font-size:11px;margin-top:4px}
.ep-flow-item{padding:2px 5px;border-radius:3px;background:var(--sf2);color:var(--tx2);white-space:nowrap;font-size:11px;cursor:pointer}
.ep-flow-item:hover{border-color:var(--ac);color:var(--tx)}
#frame-popup{display:none;position:fixed;z-index:200;width:260px;padding:10px;background:var(--sf);border:1px solid var(--bd);border-radius:6px;box-shadow:0 4px 16px rgba(0,0,0,.4);font-size:11px;overflow-y:auto}
#frame-popup.open{display:block}
#frame-popup .fp-title{font-size:14px;font-weight:700;color:var(--tx);margin-bottom:6px}
#frame-popup .fp-row{display:flex;justify-content:space-between;align-items:center;padding:2px 0}
#frame-popup .fp-label{color:var(--tx2);font-size:10px}
#frame-popup .fp-val{font-weight:600;color:var(--tx);font-size:11px}
#frame-popup .fp-bar-wrap{height:14px;background:var(--bd);border-radius:3px;flex:1;margin:0 8px;overflow:hidden;position:relative}
#frame-popup .fp-bar{height:100%;border-radius:3px;min-width:1px;display:block}
#frame-popup .fp-bar-text{position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:600;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.6);line-height:1;pointer-events:none}
#frame-popup .fp-divider{height:1px;background:var(--bd);margin:6px 0}
.fp-badge{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600}
.fp-badge-kg_plan{background:rgba(13,71,161,.25);color:#64b5f6}
.fp-badge-kg_follow{background:rgba(27,94,32,.2);color:#81c784}
.fp-badge-diverge{background:rgba(245,127,23,.25);color:#ffb74d}
.fp-badge-fallback{background:rgba(183,28,28,.25);color:#ef9a9a}
#frame-popup .fp-scatter{margin-top:6px;border:1px solid var(--bd);border-radius:4px;background:var(--sf2);overflow:hidden}
#frame-popup .fp-scatter-legend{display:flex;justify-content:center;gap:12px;padding:3px 6px;font-size:9px;color:var(--tx2)}
#frame-popup .fp-scatter-legend span::before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:3px;vertical-align:middle}
#frame-popup .fp-scatter-legend .leg-my::before{background:#f44336}
#frame-popup .fp-scatter-legend .leg-en::before{background:#2196f3}
.fp-badge-backup_switch_exact{background:rgba(123,31,162,.25);color:#ce93d8}
.fp-badge-backup_switch_fuzzy{background:rgba(123,31,162,.2);color:#b39ddb}
.fp-badge-external{background:rgba(33,150,243,.2);color:#90caf9}
.ep-flow-arrow{color:var(--ac);padding:0 1px}
.ep-events{font-size:10px;color:var(--tx2);margin-top:3px;max-height:60px;overflow-y:auto}
.ep-page-btn{padding:2px 8px;border-radius:4px;border:1px solid var(--bd);background:var(--sf);color:var(--tx);cursor:pointer;font-size:11px}
.ep-page-btn:hover{border-color:var(--ac)}
.ep-page-btn.active{background:var(--ac);color:#fff;border-color:var(--ac)}
.ep-details{margin-top:6px;border:1px solid var(--bd);border-radius:4px;overflow:hidden}
.ep-details summary{padding:5px 8px;font-size:11px;color:var(--tx2);cursor:pointer;background:var(--sf);user-select:none}
.ep-details summary:hover{color:var(--ac)}
.ep-plan-block{padding:6px 8px;border-bottom:1px solid var(--bd);font-size:11px}
.ep-plan-block:last-child{border-bottom:none}
.ep-plan-title{font-weight:600;color:var(--ac);margin-bottom:4px}
.ep-plan-details{border-bottom:1px solid var(--bd)}
.ep-plan-details:last-child{border-bottom:none}
.ep-plan-details summary{padding:5px 8px;font-size:11px;color:var(--ac);cursor:pointer;background:var(--sf);user-select:none;font-weight:600}
.ep-plan-details summary:hover{opacity:.85}
.ep-plan-details .ep-plan-content{padding:6px 8px}
.ep-beam-path{margin-bottom:3px;padding:3px 5px;border-radius:3px;border:1px solid var(--bd);font-size:10px}
.ep-beam-path.chosen{border-color:rgba(13,71,161,.45);background:rgba(13,71,161,.06)}
.ep-beam-path .path-label{font-weight:600;margin-right:4px}
.ep-beam-path.chosen .path-label{color:#1565c0}
.ep-beam-path:not(.chosen) .path-label{color:var(--tx2)}
.ep-beam-path .path-metrics{float:right;color:var(--tx2);font-size:9px}
.ep-beam-path .path-metrics span{margin-left:6px}
.ep-beam-step{display:inline-block;padding:0 3px;border-radius:2px;background:var(--sf2);color:var(--tx2);white-space:nowrap}
.ep-beam-path .path-arrow{color:var(--ac);opacity:.6;margin:0 1px}
.ep-beam-table{width:100%;border-collapse:collapse;font-size:10px;margin-top:4px}
.ep-beam-table th{text-align:left;padding:2px 4px;color:var(--tx2);font-weight:600;border-bottom:1px solid var(--bd)}
.ep-beam-table td{padding:2px 4px;color:var(--tx)}
.ep-plan-inline{display:inline-block;margin:2px 4px;padding:2px 6px;border-left:3px solid var(--ac);background:rgba(79,195,247,.06);border-radius:0 4px 4px 0;font-size:10px;vertical-align:top}
.ep-plan-inline .plan-label{font-weight:600;color:var(--ac);cursor:pointer;font-size:10px;user-select:none}
.ep-plan-inline .plan-label:hover{opacity:.8}
.ep-plan-inline .plan-content{display:none;position:fixed;z-index:200;min-width:320px;max-width:500px;max-height:350px;overflow-y:auto;padding:8px;background:var(--sf);border:1px solid var(--bd);border-radius:6px;box-shadow:0 4px 16px rgba(0,0,0,.4)}
.ep-plan-inline .plan-content.open{display:block}
.ep-dev-area{margin:6px 0;border:1px solid var(--bd);border-radius:var(--rd);overflow:hidden;display:none}
.ep-dev-area.open{display:block}
.ep-dev-toggle{padding:5px 8px;font-size:11px;color:var(--tx2);cursor:pointer;background:var(--sf);user-select:none;display:flex;align-items:center;gap:6px}
.ep-dev-toggle:hover{color:var(--ac)}
.ep-dev-toggle .arrow{transition:transform .15s;font-size:9px}
.ep-dev-area.open .ep-dev-toggle .arrow{transform:rotate(90deg)}
.ep-dev-body{padding:8px;overflow-x:auto}
.ep-dev-body svg{display:block}
.ep-dev-section-title{font-size:10px;color:var(--tx2);font-weight:600;margin-bottom:4px;text-transform:uppercase;letter-spacing:.3px}
.ep-dev-chart-wrap{margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid var(--bd)}
.ep-dev-tree-wrap{}
.dev-tooltip{position:fixed;z-index:300;padding:4px 8px;background:var(--sf);border:1px solid var(--bd);border-radius:4px;font-size:10px;color:var(--tx);pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,.3);white-space:nowrap}
.chart-zoom-btn{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border:1px solid var(--bd);border-radius:3px;background:var(--sf2);color:var(--tx2);font-size:10px;cursor:pointer;line-height:1;margin-left:6px;vertical-align:middle;transition:color .15s,border-color .15s}
.chart-zoom-btn:hover{color:var(--ac);border-color:var(--ac)}
#chart-modal{display:none;position:fixed;inset:0;z-index:400;background:rgba(0,0,0,.65);align-items:center;justify-content:center}
#chart-modal.open{display:flex}
.chart-modal-box{position:relative;background:var(--sf);border:1px solid var(--bd);border-radius:var(--rd);max-width:90vw;max-height:90vh;overflow:auto;box-shadow:0 8px 32px rgba(0,0,0,.5)}
.chart-modal-close{position:absolute;top:6px;right:8px;background:none;border:none;color:var(--tx2);font-size:20px;cursor:pointer;line-height:1;padding:2px 6px;z-index:1}
.chart-modal-close:hover{color:var(--red)}
.chart-modal-title{padding:10px 36px 6px 14px;font-size:13px;font-weight:600;color:var(--tx2)}
.chart-modal-body{padding:8px 14px 14px}
.chart-modal-body svg{width:auto;height:auto;max-width:100%;max-height:calc(90vh - 60px)}
</style></head>
<body>

<div id="toast-container" class="toast-wrap"></div>
<div id="frame-popup"></div>
<div id="dev-tooltip" class="dev-tooltip" style="display:none"></div>
<div id="chart-modal"><div class="chart-modal-box"><button class="chart-modal-close" onclick="closeChartModal()">&times;</button><div class="chart-modal-title"></div><div class="chart-modal-body"></div></div></div>

<div class="root">
<div class="left">

<div style="display:flex;align-items:stretch;gap:10px;margin-bottom:10px">
  <button class="btn btn-ghost btn-sm" style="padding:5px 16px;font-size:15px;border-radius:var(--rd)" onclick="toggleTheme()" id="theme-btn" title="切换主题">&#9728;</button>
  <div class="card" style="margin-bottom:0;display:flex;align-items:center;gap:4px;padding:4px 8px">
    <span style="font-size:10px;color:var(--tx2);white-space:nowrap">窗口</span>
    <input type="number" id="win-x" value="2600" style="width:50px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="X">
    <input type="number" id="win-y" value="50" style="width:48px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="Y">
    <span style="font-size:10px;color:var(--tx2)">&times;</span>
    <input type="number" id="win-w" value="640" style="width:52px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="宽">
    <input type="number" id="win-h" value="480" style="width:52px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="高">
    <button class="btn btn-ghost btn-sm" style="padding:2px 8px;font-size:10px;margin-left:2px" onclick="applyWindowPos()">应用</button>
  </div>
  <div class="card" style="margin-bottom:0;flex:1">
    <div class="row-between">
      <div style="display:flex;align-items:center;gap:12px">
        <span class="conn-dot" id="conn-dot"></span>
        <span style="font-size:12px;color:var(--tx2)" id="conn-text">检测中...</span>
        <span style="font-size:11px;color:var(--tx2)">|</span>
        <span style="font-size:12px;color:var(--tx2)">端口: __PORT__</span>
      </div>
      <div class="toggle-wrap">
        <label class="toggle"><input type="checkbox" id="auto-refresh" checked><span class="slider"></span></label>
        <select class="input" id="refresh-interval" style="width:55px;padding:3px 5px;font-size:11px">
          <option value="1000">1s</option>
          <option value="2000" selected>2s</option>
          <option value="3000">3s</option>
          <option value="5000">5s</option>
        </select>
      </div>
    </div>
  </div>
  <div class="card" style="margin-bottom:0">
    <div style="display:flex;align-items:center;gap:6px;justify-content:center">
      <button class="btn btn-warn btn-sm" onclick="ctrl('pause')">暂停</button>
      <button class="btn btn-ok btn-sm" onclick="ctrl('resume')">恢复</button>
      <span style="color:var(--bd);font-size:16px;margin:0 2px">|</span>
      <button class="btn btn-step btn-sm" onclick="ctrl('step')">步进</button>
      <button class="btn btn-blue-dark btn-sm" onclick="ctrl('run_episode')">单局运行</button>
      <span style="color:var(--bd);font-size:16px;margin:0 2px">|</span>
      <button class="btn btn-danger btn-sm" onclick="shutdownService()">停止服务</button>
    </div>
  </div>
</div>

<div class="card">
  <div class="card-title">游戏状态</div>
  <div class="metrics" style="margin-bottom:10px">
    <div class="metric"><div class="metric-label">状态</div><div class="metric-value"><span class="badge badge-gray" id="state">--</span></div></div>
    <div class="metric"><div class="metric-label">帧</div><div class="metric-value" id="frame">--</div></div>
    <div class="metric"><div class="metric-label">Episode</div><div class="metric-value" id="episode">--</div></div>
    <div class="metric"><div class="metric-label">我方</div><div class="metric-value" id="my-count">--</div></div>
    <div class="metric"><div class="metric-label">敌方</div><div class="metric-value" id="enemy-count">--</div></div>
    <div class="metric"><div class="metric-label">聚类</div><div class="metric-value" id="cluster" style="font-size:13px">--</div></div>
  </div>
  <div class="metrics">
    <div class="metric"><div class="metric-label">我方 HP</div><div class="metric-value" id="my-hp">--</div></div>
    <div class="metric"><div class="metric-label">敌方 HP</div><div class="metric-value" id="enemy-hp">--</div></div>
    <div class="metric" style="flex:2"><div class="metric-label">上一步动作</div><div class="metric-value" id="last-action" style="font-size:13px;color:var(--ac)">--</div></div>
    <div class="metric"><div class="metric-label">KG</div><div class="metric-value" id="kg-status" style="font-size:11px">--</div></div>
    <div class="metric"><div class="metric-label">Buffer</div><div class="metric-value" id="history-buffer" style="font-size:11px;color:var(--ac)">--</div></div>
  </div>
</div>

<div class="card" id="episodes-card">
  <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
    <span>对局记录</span>
    <div style="display:flex;gap:6px;align-items:center">
      <select id="ep-sort" class="input" style="padding:3px 6px;font-size:11px;border-radius:4px;border:1px solid var(--bd);background:var(--sf);color:var(--tx)" onchange="epCurrentPage=1;renderLocalEpisodes()">
        <option value="id_desc">最新优先</option>
        <option value="id_asc">最早优先</option>
        <option value="score_desc">得分降序</option>
        <option value="score_asc">得分升序</option>
      </select>
      <input id="ep-search" type="text" placeholder="搜索..." style="padding:3px 8px;font-size:11px;border-radius:4px;border:1px solid var(--bd);background:var(--sf);color:var(--tx);width:80px" oninput="debounceSearch()">
      <button class="btn btn-ghost btn-sm" style="padding:3px 8px;font-size:11px" onclick="loadEpisodes()">刷新</button>
      <input id="save-count" type="number" value="100" min="1" title="保存条数" style="width:45px;padding:3px 4px;font-size:11px;border-radius:4px;border:1px solid var(--bd);background:var(--sf);color:var(--tx);text-align:center">
      <button class="btn btn-ghost btn-sm" style="padding:3px 8px;font-size:11px;color:#4CAF50" onclick="saveResults()">保存</button>
      <button class="btn btn-ghost btn-sm" style="padding:3px 8px;font-size:11px;color:#f44336" onclick="clearEpisodes()">清空</button>
    </div>
  </div>
  <div id="episodes-list" style="max-height:840px;overflow-y:auto"></div>
  <div id="ep-pagination" style="display:flex;justify-content:center;gap:4px;margin-top:8px;font-size:12px"></div>
  <div style="text-align:center;font-size:11px;color:var(--tx2);margin-top:4px"><span id="ep-count">0</span> 条记录</div>
</div>

</div>

<div class="right">
  <div class="console-header">
    <span>控制台日志</span>
    <div style="display:flex;gap:6px">
      <div style="position:relative">
        <button class="btn btn-ghost btn-sm" onclick="toggleFilter()">过滤</button>
        <div class="filter-panel" id="filter-panel">
          <div class="filter-group">
            <h4>日志级别</h4>
            <label><input type="checkbox" data-filter="level" value="info" checked onchange="applyFilters()"> Info</label>
            <label><input type="checkbox" data-filter="level" value="success" checked onchange="applyFilters()"> Success</label>
            <label><input type="checkbox" data-filter="level" value="warn" checked onchange="applyFilters()"> Warn</label>
            <label><input type="checkbox" data-filter="level" value="error" checked onchange="applyFilters()"> Error</label>
            <label><input type="checkbox" data-filter="level" value="debug" onchange="applyFilters()"> Debug</label>
          </div>
          <div class="filter-group">
            <h4>来源</h4>
            <label><input type="checkbox" data-filter="source" value="game" checked onchange="applyFilters()"> GAME</label>
            <label><input type="checkbox" data-filter="source" value="api" checked onchange="applyFilters()"> API</label>
          </div>
          <div class="filter-group">
            <h4>消息类型</h4>
            <label><input type="checkbox" data-filter="type" value="info" checked onchange="applyFilters()"> Info</label>
            <label><input type="checkbox" data-filter="type" value="action" checked onchange="applyFilters()"> Action</label>
            <label><input type="checkbox" data-filter="type" value="fallback" checked onchange="applyFilters()"> Fallback</label>
            <label><input type="checkbox" data-filter="type" value="result" checked onchange="applyFilters()"> Result</label>
            <label><input type="checkbox" data-filter="type" value="episode" checked onchange="applyFilters()"> Episode</label>
            <label><input type="checkbox" data-filter="type" value="control" checked onchange="applyFilters()"> Control</label>
          </div>
          <div style="display:flex;gap:6px;margin-top:10px;border-top:1px solid var(--bd);padding-top:8px">
            <button class="btn btn-primary btn-sm" style="flex:1;padding:4px 0;font-size:11px" onclick="saveFilterConfig()">保存配置</button>
            <button class="btn btn-ghost btn-sm" style="flex:1;padding:4px 0;font-size:11px" onclick="resetFilterConfig()">重置默认</button>
          </div>
        </div>
      </div>
      <button class="btn btn-ghost btn-sm" onclick="clearLogs()">清空</button>
    </div>
  </div>
  <div class="console-body" id="console-body">
    <div style="text-align:center;color:#666;padding:40px 0">等待日志...</div>
  </div>
  <div class="console-footer">
    <span id="log-count">0 条</span>
    <span id="log-scroll-hint" style="cursor:pointer;color:var(--ac)" onclick="scrollConsoleBottom()">↓ 滚动到底部</span>
  </div>
</div>
</div>

<script>
const API='http://__HOST__:__PORT__';
let timer=null, logTimer=null, latestSeq=0, renderedMaxSeq=0, recommendedAction='';
let userScrolled=false;

const LV_COLORS={info:'lv-info',success:'lv-success',warn:'lv-warn',error:'lv-error',debug:'lv-debug'};
const SRC_CLS={api:'src-api',game:'src-game'};
let _filterLevels=new Set(['info','success','warn','error']);
let _filterSources=new Set(['game','api']);
let _filterTypes=new Set(['info','action','fallback','result','episode','control']);
function _extractType(msg){
  if(!msg)return 'info';
  if(msg.startsWith('执行动作'))return 'action';
  if(msg.startsWith('回退策略'))return 'fallback';
  if(msg.startsWith('判定'))return 'result';
  if(msg.startsWith('Episode'))return 'episode';
  if(msg.startsWith('步进')||msg.startsWith('游戏暂停')||msg.startsWith('游戏恢复')||msg.startsWith('游戏停止'))return 'control';
  return 'info';
}
function _isFiltered(log){if(log.level==='debug')return !_filterLevels.has('debug');return !_filterLevels.has(log.level||'info')||!_filterSources.has(log.source||'game')||!_filterTypes.has(_extractType(log.message))}
function toggleFilter(){document.getElementById('filter-panel').classList.toggle('show')}
function _syncCheckboxesFromSets(){
  document.querySelectorAll('#filter-panel input[type=checkbox]').forEach(function(cb){
    var t=cb.getAttribute('data-filter'),v=cb.value;
    if(t==='level')cb.checked=_filterLevels.has(v);
    else if(t==='source')cb.checked=_filterSources.has(v);
    else cb.checked=_filterTypes.has(v);
  });
}
function _syncSetsFromCheckboxes(){
  _filterLevels=new Set();_filterSources=new Set();_filterTypes=new Set();
  document.querySelectorAll('#filter-panel input[type=checkbox]').forEach(function(cb){
    if(cb.checked){var t=cb.getAttribute('data-filter'),v=cb.value;if(t==='level')_filterLevels.add(v);else if(t==='source')_filterSources.add(v);else _filterTypes.add(v)}
  });
}
var FILTER_STORAGE_KEY='live_filter_cfg';
var FILTER_DEFAULTS={levels:['info','success','warn','error'],sources:['game','api'],types:['info','action','fallback','result','episode','control']};
function loadFilterConfig(){
  var raw=localStorage.getItem(FILTER_STORAGE_KEY);
  if(!raw)return;
  try{
    var cfg=JSON.parse(raw);
    _filterLevels=new Set(cfg.levels||FILTER_DEFAULTS.levels);
    _filterSources=new Set(cfg.sources||FILTER_DEFAULTS.sources);
    _filterTypes=new Set(cfg.types||FILTER_DEFAULTS.types);
    _syncCheckboxesFromSets();
    document.querySelectorAll('#console-body .log-line').forEach(function(el){
      el.style.display=(_filterLevels.has(el.getAttribute('data-level'))&&_filterSources.has(el.getAttribute('data-source'))&&_filterTypes.has(el.getAttribute('data-type')))?'':'none';
    });
  }catch(e){}
}
function saveFilterConfig(){
  _syncSetsFromCheckboxes();
  localStorage.setItem(FILTER_STORAGE_KEY,JSON.stringify({levels:Array.from(_filterLevels),sources:Array.from(_filterSources),types:Array.from(_filterTypes)}));
  toast('过滤配置已保存');
}
function resetFilterConfig(){
  localStorage.removeItem(FILTER_STORAGE_KEY);
  _filterLevels=new Set(FILTER_DEFAULTS.levels);
  _filterSources=new Set(FILTER_DEFAULTS.sources);
  _filterTypes=new Set(FILTER_DEFAULTS.types);
  _syncCheckboxesFromSets();
  document.querySelectorAll('#console-body .log-line').forEach(function(el){el.style.display=''});
  toast('已重置为默认');
}
function applyFilters(){
  _syncSetsFromCheckboxes();
  document.querySelectorAll('#console-body .log-line').forEach(function(el){
    el.style.display=(_filterLevels.has(el.getAttribute('data-level'))&&_filterSources.has(el.getAttribute('data-source'))&&_filterTypes.has(el.getAttribute('data-type')))?'':'none';
  });
}
function _positionPopup(triggerEl,popupEl,minW){
  var r=triggerEl.getBoundingClientRect();
  var vh=window.innerHeight,vw=window.innerWidth;
  var spaceBelow=vh-r.bottom-8,spaceAbove=r.top-8;
  popupEl.style.bottom='';
  popupEl.style.right='';
  if(spaceBelow>=200){
    popupEl.style.top=(r.bottom+2)+'px';
    popupEl.style.maxHeight=Math.min(spaceBelow,500)+'px';
  }else if(spaceAbove>=200){
    popupEl.style.top='';
    popupEl.style.bottom=(vh-r.top+2)+'px';
    popupEl.style.maxHeight=Math.min(spaceAbove,500)+'px';
  }else{
    popupEl.style.top='10px';
    popupEl.style.maxHeight=(vh-20)+'px';
  }
  popupEl.style.left=r.left+'px';
  if(r.left+minW>vw){popupEl.style.left='';popupEl.style.right=(vw-r.right)+'px'}
}
function closeFramePopup(){var p=document.getElementById('frame-popup');if(p){p.classList.remove('open');p.style.top='';p.style.bottom='';p.style.maxHeight=''}}
function showFramePopup(ep,frIdx,el){
  closeFramePopup();
  document.querySelectorAll('.ep-plan-inline .plan-content.open').forEach(function(c){c.style.top='';c.style.left='';c.style.right='';c.classList.remove('open')});
  if(!ep||!ep.events||!ep.events[frIdx])return;
  var ev=ep.events[frIdx];
  var sid=ev.state_id!=null?(typeof ev.state_id==='number'?'S'+ev.state_id:''+ev.state_id):'?';
  var et=ev.event_type||'-';
  var etLbl={kg_plan:'规划',kg_follow:'跟随',diverge:'偏离',backup_switch_exact:'备选(精准)',backup_switch_fuzzy:'备选(模糊)',backup_switch:'备选',fallback:'回退',replay:'回放',external:'外部',no_action:'无动作',manual:'手动'}[et]||et;
  var badgeCls='fp-badge fp-badge-'+et;
  var myHp=ev.hp_my||0,enHp=ev.hp_enemy||0;
  var maxHp=Math.max(myHp,enHp,1);
  var myW=Math.round(myHp/maxHp*100),enW=Math.round(enHp/maxHp*100);
  var h='<div class="fp-title">S'+sid+'</div>';
  h+='<div class="fp-row"><span class="fp-label">来源</span><span class="'+badgeCls+'">'+etLbl+'</span></div>';
  h+='<div class="fp-row"><span class="fp-label">动作</span><span class="fp-val">'+escHtml(ev.action||'-')+'</span></div>';
  if(ev.action_code){
    var code=ev.action_code;
    h+='<div class="fp-row"><span class="fp-label">编码</span><span class="fp-val" style="font-family:monospace">'+escHtml(code)+'</span></div>';
    if(code.length===2&&code[0]>='0'&&code[0]<='9'&&code[1]>='a'&&code[1]<='z'){
      var ci=parseInt(code[0]),ai=code.charCodeAt(1)-97;
      var _CL=['k_means_000','k_means_025','k_means_050','k_means_075','k_means_100'];
      var _AL=['action_ATK_nearest','action_ATK_clu_nearest','action_ATK_nearest_weakest','action_ATK_clu_nearest_weakest','action_ATK_threatening','action_DEF_clu_nearest','action_MIX_gather','action_MIX_lure','action_MIX_sacrifice_lure','do_randomly','do_nothing'];
      if(ci<_CL.length)h+='<div class="fp-row"><span class="fp-label">聚类粒度</span><span class="fp-val">'+_CL[ci]+' ('+ci+')</span></div>';
      if(ai<_AL.length)h+='<div class="fp-row"><span class="fp-label">映射动作</span><span class="fp-val" style="font-size:10px">'+_AL[ai]+'</span></div>';
    }
  }
  if(ev.state_cluster&&ev.state_cluster.length===2)h+='<div class="fp-row"><span class="fp-label">聚类</span><span class="fp-val" style="font-family:monospace">P('+ev.state_cluster[0]+','+ev.state_cluster[1]+')</span></div>';
  h+='<div class="fp-row"><span class="fp-label">游戏帧</span><span class="fp-val">'+(ev.game_loop!=null?ev.game_loop:'-')+'</span></div>';
  h+='<div class="fp-divider"></div>';
  var myCnt=ev.my_count||0,enCnt=ev.enemy_count||0;
  var myHp=ev.hp_my||0,enHp=ev.hp_enemy||0;
  var e0=ep.events&&ep.events[0]?ep.events[0]:null;
  var myBase=e0?Math.max(e0.hp_my||1,1):Math.max(myHp,1);
  var enBase=e0?Math.max(e0.hp_enemy||1,1):Math.max(enHp,1);
  var myPct=Math.round(myHp/myBase*100),enPct=Math.round(enHp/enBase*100);
  h+='<div class="fp-row"><span class="fp-label">我方</span><span class="fp-val">'+myCnt+' 单位</span><span class="fp-bar-wrap"><span class="fp-bar" style="width:'+myPct+'%;background:#f44336"></span><span class="fp-bar-text">'+myHp+'/'+myBase+' '+myPct+'%</span></span></div>';
  h+='<div class="fp-row"><span class="fp-label">敌方</span><span class="fp-val">'+enCnt+' 单位</span><span class="fp-bar-wrap"><span class="fp-bar" style="width:'+enPct+'%;background:#2196f3"></span><span class="fp-bar-text">'+enHp+'/'+enBase+' '+enPct+'%</span></span></div>';
  var myUp=ev.my_units_pos||[],enUp=ev.enemy_units_pos||[];
  if(myUp.length>0||enUp.length>0){
    h+='<div class="fp-divider"></div>';
    h+='<div class="fp-scatter"><div class="fp-scatter-legend"><span class="leg-my">我方</span><span class="leg-en">敌方</span></div>';
    h+='<svg width="100%" height="120" viewBox="0 0 240 120" preserveAspectRatio="xMidYMid meet" style="display:block;background:rgba(0,0,0,.06)">';
    var allPts=myUp.concat(enUp);
    var minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
    for(var pi=0;pi<allPts.length;pi++){var pt=allPts[pi];if(pt.x<minX)minX=pt.x;if(pt.y<minY)minY=pt.y;if(pt.x>maxX)maxX=pt.x;if(pt.y>maxY)maxY=pt.y}
    if(minX>=maxX){minX-=1;maxX+=1}if(minY>=maxY){minY-=1;maxY+=1}
    var pad=10,svgW=240,padH=120;
    for(var mi=0;mi<myUp.length;mi++){var mu=myUp[mi];var sx=pad+(mu.x-minX)/(maxX-minX)*(svgW-2*pad);var sy=pad+(mu.y-minY)/(maxY-minY)*(padH-2*pad);var ratio=mu.hp/45;if(ratio>1)ratio=1;if(ratio<0.3)ratio=0.3;h+='<circle cx="'+sx.toFixed(1)+'" cy="'+sy.toFixed(1)+'" r="3.5" fill="rgba(244,67,54,'+ratio.toFixed(2)+')" stroke="rgba(244,67,54,0.8)" stroke-width="0.5"/>'}
    for(var ei=0;ei<enUp.length;ei++){var eu=enUp[ei];var ex=pad+(eu.x-minX)/(maxX-minX)*(svgW-2*pad);var ey=pad+(eu.y-minY)/(maxY-minY)*(padH-2*pad);var eRatio=eu.hp/45;if(eRatio>1)eRatio=1;if(eRatio<0.3)eRatio=0.3;h+='<circle cx="'+ex.toFixed(1)+'" cy="'+ey.toFixed(1)+'" r="3.5" fill="rgba(33,150,243,'+eRatio.toFixed(2)+')" stroke="rgba(33,150,243,0.8)" stroke-width="0.5"/>'}
    h+='</svg></div>';
  }
  if(ev.end_game_flag)h+='<div class="fp-divider"></div><div style="color:#ff9800;font-weight:600;font-size:10px;text-align:center">对局结束帧</div>';
  if(ev.plan)h+='<div class="fp-divider"></div><div style="color:var(--ac);font-size:10px;text-align:center">有推演规划 → 点击规划标签查看</div>';
  var popup=document.getElementById('frame-popup');
  popup.innerHTML=h;
  popup.classList.add('open');
  _positionPopup(el,popup,260);
}
document.addEventListener('click',function(e){
if(!e.target.closest('.filter-panel')&&!e.target.closest('[onclick*="toggleFilter"]')){document.getElementById('filter-panel').classList.remove('show')}
if(e.target.classList&&e.target.classList.contains('plan-label')){closeFramePopup();var box=e.target.closest('.ep-plan-inline');if(!box)return;var content=box.querySelector('.plan-content');if(!content)return;var isOpen=content.classList.contains('open');document.querySelectorAll('.ep-plan-inline .plan-content.open').forEach(function(c){c.style.top='';c.style.left='';c.style.right='';c.style.bottom='';c.style.maxHeight='';c.classList.remove('open')});if(!isOpen){content.classList.add('open');_positionPopup(e.target,content,320)}e.stopPropagation();return}
if(e.target.classList&&e.target.classList.contains('ep-flow-item')){var fi=e.target.closest('.ep-flow-item');if(!fi)return;var epIdx=fi.getAttribute('data-ep');var frIdx=fi.getAttribute('data-fr');if(epIdx!=null&&frIdx!=null){showFramePopup(allEpisodes.find(function(e){return e.id===parseInt(epIdx)}),parseInt(frIdx),fi)}e.stopPropagation();return}
if(!e.target.closest('.ep-plan-inline')&&!e.target.closest('#frame-popup')){document.querySelectorAll('.ep-plan-inline .plan-content.open').forEach(function(c){c.style.top='';c.style.left='';c.style.right='';c.classList.remove('open')});closeFramePopup()}});
document.addEventListener('mouseover',function(e){var el=e.target.closest('[data-tip]');if(el){_devShowTip(e,el.getAttribute('data-tip').replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"'))}});
document.addEventListener('mouseout',function(e){if(e.target.closest('[data-tip]'))_devHideTip()});

async function apiGet(p){const r=await fetch(API+p);if(!r.ok)throw new Error(r.status);return r.json()}
async function apiPost(p,b){const r=await fetch(API+p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});if(!r.ok){const t=await r.text().catch(()=>'');throw new Error(t||r.status)}return r.json()}

function toast(m,ok=true){const c=document.getElementById('toast-container'),d=document.createElement('div');d.className='toast '+(ok?'toast-ok':'toast-err');d.textContent=m;c.appendChild(d);setTimeout(()=>d.remove(),3000)}

function setConn(ok){const d=document.getElementById('conn-dot'),t=document.getElementById('conn-text');d.className='conn-dot '+(ok?'conn-ok':'conn-err');t.textContent=ok?'已连接':'连接失败'}
function setStatus(s){const el=document.getElementById('state');if(s.running&&!s.paused){el.textContent='运行中';el.className='badge badge-green'}else if(s.paused){el.textContent='暂停';el.className='badge badge-yellow'}else{el.textContent='已停止';el.className='badge badge-gray'}document.getElementById('frame').textContent=s.frame||0;document.getElementById('episode').textContent=s.episode||0;document.getElementById('my-count').textContent=s.my_count!=null?s.my_count:'-';document.getElementById('enemy-count').textContent=s.enemy_count!=null?s.enemy_count:'-';document.getElementById('cluster').textContent=s.state_cluster||'-';const kgEl=document.getElementById('kg-status');if(s.kg_loaded){const f=s.kg_file||'';kgEl.textContent=f.split('/').pop().replace('.pkl','');kgEl.style.color='#4caf50'}else{kgEl.textContent='未加载';kgEl.style.color='#f44336'}const bufEl=document.getElementById('history-buffer');if(s.history_episodes!=null){bufEl.textContent=s.history_episodes+' eps / '+s.history_frames+' frames / '+s.history_capacity+' cap'}else{bufEl.textContent='--'}}
function setObs(o){if(o.error)return;document.getElementById('my-hp').textContent=o.my_total_hp||0;document.getElementById('enemy-hp').textContent=o.enemy_total_hp||0;document.getElementById('last-action').textContent=o.last_action||'-'}

async function refresh(){try{const s=await apiGet('/game/status');setStatus(s);setConn(true);if(!logTimer)startLogTimer();if(s.running){try{const o=await apiGet('/game/observation');setObs(o)}catch(e){}}}catch(e){setConn(false)}}

function startTimer(){stopTimer();const ms=parseInt(document.getElementById('refresh-interval').value)||2000;timer=setInterval(refresh,ms)}
function stopTimer(){if(timer){clearInterval(timer);timer=null}}
document.getElementById('auto-refresh').addEventListener('change',function(){if(this.checked)startTimer();else stopTimer()});
document.getElementById('refresh-interval').addEventListener('change',function(){if(document.getElementById('auto-refresh').checked)startTimer()});

const consoleBody=document.getElementById('console-body');
consoleBody.addEventListener('scroll',function(){const el=this;userScrolled=(el.scrollTop+el.clientHeight)<el.scrollHeight-30});
function scrollConsoleBottom(){consoleBody.scrollTop=consoleBody.scrollHeight;userScrolled=false}

function appendLogLine(log){
  if(log.seq!==undefined&&log.seq<=renderedMaxSeq)return;
  if(log.seq!==undefined&&log.seq>renderedMaxSeq)renderedMaxSeq=log.seq;
  const div=document.createElement('div');
  div.className='log-line';
  div.setAttribute('data-level',log.level||'info');
  div.setAttribute('data-source',log.source||'game');
  div.setAttribute('data-type',_extractType(log.message));
  if(_isFiltered(log)){div.style.display='none'}
  const lvCls=LV_COLORS[log.level]||'lv-info';
  const srcCls=SRC_CLS[log.source]||'src-game';
  div.innerHTML=`<span class="log-ts">${log.ts||''}</span><span class="log-src ${srcCls}">${(log.source||'').toUpperCase()}</span><span class="log-msg ${lvCls}">${escHtml(log.message||'')}</span>`;
  consoleBody.appendChild(div);
  while(consoleBody.childElementCount>200){
    var first=consoleBody.firstChild;
    if(first.style.display==='none'){consoleBody.removeChild(first)}
    else{var vc=0;for(var c=consoleBody.firstChild;c;c=c.nextSibling){if(c.style.display!=='none')vc++}if(vc<=200)break;consoleBody.removeChild(first)}
  }
  if(!userScrolled)scrollConsoleBottom();
}
function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function escapeAttr(s){return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

async function fetchLogs(){
  try{
    const r=await apiGet('/game/logs?after_seq='+latestSeq);
    const logs=r.logs||[];
    if(logs.length===0)return;
    if(consoleBody.querySelector('[style*="text-align:center"]'))consoleBody.innerHTML='';
    logs.forEach(l=>appendLogLine(l));
    if(r.latest_seq>latestSeq)latestSeq=r.latest_seq;
    document.getElementById('log-count').textContent=consoleBody.childElementCount+' 条';
  }catch(e){}
}

function startLogTimer(){stopLogTimer();logTimer=setInterval(fetchLogs,1500)}
function stopLogTimer(){if(logTimer){clearInterval(logTimer);logTimer=null}}

async function clearLogs(){try{await apiPost('/game/logs/clear',{})}catch(e){}consoleBody.innerHTML='<div style="text-align:center;color:#666;padding:40px 0">已清空</div>';latestSeq=0;renderedMaxSeq=0;document.getElementById('log-count').textContent='0 条'}

async function ctrl(cmd){try{await apiPost('/game/control',{command:cmd});toast('已发送: '+cmd);await refresh()}catch(e){toast('失败: '+e.message,false)}}
function toggleTheme(){const d=document.documentElement;d.classList.toggle('light');const isLight=d.classList.contains('light');document.getElementById('theme-btn').innerHTML=isLight?'&#9728;':'&#9790;';localStorage.setItem('live_theme',isLight?'light':'dark')}
function loadWindowPos(){var s=localStorage.getItem('live_win_pos');if(!s)return;try{var p=JSON.parse(s);document.getElementById('win-x').value=p.x||2600;document.getElementById('win-y').value=p.y||50;document.getElementById('win-w').value=p.w||640;document.getElementById('win-h').value=p.h||480}catch(e){}}
async function applyWindowPos(){var x=parseInt(document.getElementById('win-x').value)||50;var y=parseInt(document.getElementById('win-y').value)||50;var w=parseInt(document.getElementById('win-w').value)||640;var h=parseInt(document.getElementById('win-h').value)||480;localStorage.setItem('live_win_pos',JSON.stringify({x:x,y:y,w:w,h:h}));try{var r=await apiPost('/game/window_pos',{x:x,y:y,w:w,h:h});if(r.ok)toast('窗口位置已更新');else toast('更新失败: '+(r.error||''),false)}catch(e){toast('请求失败: '+e.message,false)}}
async function shutdownService(){stopTimer();stopLogTimer();setConn(false);latestSeq=0;renderedMaxSeq=0;consoleBody.innerHTML='<div style="text-align:center;color:#666;padding:40px 0">等待日志...</div>';document.getElementById('log-count').textContent='0 条';document.getElementById('conn-text').textContent='已停止';try{await fetch(API+'/game/shutdown',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})}catch(e){}toast('服务已停止')}

var epCurrentPage=1,epPerPage=10,epSearchTimer=null,allEpisodes=[];
function debounceSearch(){clearTimeout(epSearchTimer);epSearchTimer=setTimeout(function(){epCurrentPage=1;renderLocalEpisodes()},400)}
async function loadEpisodes(){
  try{
    var data=await apiGet('/game/episodes?page=1&per_page=50');
    var newEps=data.episodes||[];
    newEps.forEach(function(ep){
      var idx=allEpisodes.findIndex(function(e){return e.id===ep.id});
      if(idx>=0){allEpisodes[idx]=ep}else{allEpisodes.push(ep)}
    });
    renderLocalEpisodes();
  }catch(e){}
}
function exportEpData(epId,type){
  var ep=allEpisodes.find(function(e){return e.id===epId});
  if(!ep){toast('未找到对局 #'+epId,false);return}
  var data,filename;
  if(type==='frames'){
    data={id:ep.id,result:ep.result,score:ep.score,steps:ep.steps,timestamp:ep.timestamp,markov_flow:ep.markov_flow,events:ep.events};
    filename='episode_'+epId+'_frames.json';
  }else if(type==='plans'){
    var plans=(ep.events||[]).map(function(e){return e.plan}).filter(function(p){return p!=null});
    data={id:ep.id,result:ep.result,steps:ep.steps,plans:plans};
    filename='episode_'+epId+'_plans.json';
  }else{
    data=ep;filename='episode_'+epId+'_full.json';
  }
  var blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=filename;a.click();URL.revokeObjectURL(a.href);
  toast('已导出 '+filename);
}
function renderLocalEpisodes(){
  var sort=document.getElementById('ep-sort').value;
  var search=document.getElementById('ep-search').value.trim().toLowerCase();
  var filtered=allEpisodes.slice();
  if(search){filtered=filtered.filter(function(ep){return ep.result.toLowerCase().indexOf(search)>=0||ep.mode.toLowerCase().indexOf(search)>=0})}
  if(sort==='id_desc')filtered.sort(function(a,b){return b.id-a.id});
  else if(sort==='id_asc')filtered.sort(function(a,b){return a.id-b.id});
  else if(sort==='score_desc')filtered.sort(function(a,b){return(b.score||0)-(a.score||0)});
  else if(sort==='score_asc')filtered.sort(function(a,b){return(a.score||0)-(b.score||0)});
  var total=filtered.length;
  var start=(epCurrentPage-1)*epPerPage;
  var pageItems=filtered.slice(start,start+epPerPage);
  renderEpisodes({episodes:pageItems,total:total,page:epCurrentPage,per_page:epPerPage,total_pages:Math.max(1,Math.ceil(total/epPerPage))});
}
var epAutoTimer=null;
function startEpAutoRefresh(){stopEpAutoRefresh();epAutoTimer=setInterval(loadEpisodes,15000)}
function stopEpAutoRefresh(){if(epAutoTimer){clearInterval(epAutoTimer);epAutoTimer=null}}
async function clearEpisodes(){try{await apiPost('/game/episodes/clear');allEpisodes=[];renderEpisodes({episodes:[],total:0,page:1,per_page:epPerPage,total_pages:1});toast('对局记录已清空')}catch(e){toast('清空失败: '+e.message,false)}}
async function saveResults(){var n=parseInt(document.getElementById('save-count').value)||100;try{var r=await apiPost('/game/results/save',{max_count:n});if(r.ok){toast('已保存 '+r.num_episodes+' 局 → '+r.filename)}else{toast('保存失败',false)}}catch(e){toast('保存失败: '+e.message,false)}}

function renderEpisodes(data){
  var list=document.getElementById('episodes-list');
  var pgn=document.getElementById('ep-pagination');
  var cnt=document.getElementById('ep-count');
  var eps=data.episodes||[];
  cnt.textContent=data.total||0;
  if(eps.length===0){list.innerHTML='<div style="text-align:center;color:var(--tx2);padding:20px 0;font-size:12px">暂无对局记录</div>';pgn.innerHTML='';return}
  var html='';
  for(var i=0;i<eps.length;i++){
    var ep=eps[i];
    var isMulti=ep.mode==='multi_step';
    var badgeCls='ep-badge-interrupted';
    if(ep.result==='Win')badgeCls='ep-badge-win';
    else if(ep.result==='Loss')badgeCls='ep-badge-loss';
    else if(ep.result==='Dogfall')badgeCls='ep-badge-dogfall';
    var flowHtml='';
    if(ep.markov_flow&&ep.markov_flow.length>0){
      var maxShow=Math.min(ep.markov_flow.length,20);
      for(var j=0;j<maxShow;j++){
        var item=ep.markov_flow[j];
        var ev=(ep.events&&ep.events[j])?ep.events[j]:{event_type:'no_action'};
        var et=ev.event_type||'no_action';
        var bg,bd,tc;
        if(et==='kg_plan'){
          if(isMulti){bg='rgba(13,71,161,.18)';bd='2px dashed rgba(13,71,161,.5)';tc='#42a5f5'}
          else{bg='rgba(13,71,161,.12)';bd='1px solid rgba(13,71,161,.3)';tc='#64b5f6'}
        }else if(et==='kg_follow'){
          bg='rgba(27,94,32,.10)';bd='1px solid rgba(27,94,32,.25)';tc='#81c784';
        }else if(et==='diverge'){
          bg='rgba(245,127,23,.15)';bd='1px solid rgba(245,127,23,.3)';tc='#ffb74d';
        }else if(et==='fallback'){
          bg='rgba(183,28,28,.15)';bd='1px solid rgba(183,28,28,.3)';tc='#ef9a9a';
        }else if(et==='backup_switch'||et==='backup_switch_exact'){
          bg='rgba(123,31,162,.15)';bd='1px solid rgba(123,31,162,.3)';tc='#ce93d8';
        }else if(et==='backup_switch_fuzzy'){
          bg='rgba(123,31,162,.12)';bd='1px solid rgba(123,31,162,.25)';tc='#b39ddb';
        }else if(et==='replay'){
          bg='rgba(0,150,136,.12)';bd='1px solid rgba(0,150,136,.3)';tc='#4db6ac';
        }else{
          bg='var(--sf2)';bd='1px solid var(--bd)';tc='var(--tx2)';
        }
        var lbl={kg_plan:'规划',kg_follow:'跟随',diverge:'偏离',backup_switch_exact:'备选(精准)',backup_switch_fuzzy:'备选(模糊)',backup_switch:'备选',fallback:'回退',replay:'回放',external:'外部',no_action:'-',manual:'手动'}[et]||et;
        if(j>0)flowHtml+='<span class="ep-flow-arrow">&rarr;</span>';
        var sid=item[0]!=null?(typeof item[0]==='number'?'S'+item[0]:''+item[0]):'S?';
        var act=item[1]||'';
        if(act.startsWith('action_'))act=act.replace(/^action_/,'').substring(0,6);
        flowHtml+='<span class="ep-flow-item" data-ep="'+ep.id+'" data-fr="'+j+'" style="background:'+bg+';border:'+bd+';color:'+tc+'" title="'+lbl+'">'+sid+'<span style="opacity:.7;margin-left:2px">'+escHtml(act)+'</span></span>';
        if(ev.plan){
          flowHtml+=renderPlanInline(ev.plan,j+1);
        }
      }
      if(ep.markov_flow.length>20)flowHtml+='<span class="ep-flow-item" style="color:var(--ylw);background:none;border:none">+'+(ep.markov_flow.length-20)+'</span>';
    }
    var scoreStr=ep.score!==undefined&&ep.score!==0?(ep.score>0?'+':'')+ep.score.toFixed(0):'-';
    html+='<div class="ep-item">';
    html+='<div class="ep-header"><span class="ep-title">#'+ep.id+'</span><span class="ep-badge '+badgeCls+'">'+(ep.result||'?')+'</span>';
    html+='<span class="ep-export-btns">';
    html+='<button onclick="exportEpData('+ep.id+',&#39;frames&#39;)" title="导出帧流">F</button>';
    html+='<button onclick="exportEpData('+ep.id+',&#39;plans&#39;)" title="导出推演详情">P</button>';
    html+='<button onclick="exportEpData('+ep.id+',&#39;all&#39;)" title="导出全部">A</button>';
    html+='<button onclick="toggleDevArea('+ep.id+')" title="规划偏差">D</button>';
    html+='</span></div>';
    html+='<div class="ep-meta">得分: '+scoreStr+' | 步数: '+ep.steps+' | '+(ep.mode==='multi_step'?'多步':'单步')+' '+(ep.match_mode?'('+ep.match_mode+')':'')+' | '+(ep.timestamp||'')+'</div>';
    if(flowHtml)html+='<div class="ep-flow" style="flex-wrap:wrap">'+flowHtml+'</div>';
    html+=renderDeviationSection(ep);
    html+='</div>';
  }
  list.innerHTML=html;
  var totalPages=data.total_pages||1;
  if(totalPages<=1){pgn.innerHTML='';return}
  var phtml='';
  for(var p=1;p<=totalPages;p++){
    if(p===epCurrentPage)phtml+='<button class="ep-page-btn active">'+p+'</button>';
    else phtml+='<button class="ep-page-btn" onclick="epCurrentPage='+p+';renderLocalEpisodes()">'+p+'</button>';
  }
  pgn.innerHTML=phtml;
}
var _devTooltip=document.getElementById('dev-tooltip');
function _devShowTip(e,html){if(!_devTooltip)return;_devTooltip.innerHTML=html;_devTooltip.style.display='block';_devTooltip.style.left=Math.min(e.clientX+10,window.innerWidth-200)+'px';_devTooltip.style.top=(e.clientY+10)+'px'}
function _devHideTip(){if(_devTooltip)_devTooltip.style.display='none'}
function toggleDevArea(epId){var el=document.getElementById('dev-area-'+epId);if(el)el.classList.toggle('open')}
function showChartModal(epId,type){
  var ep=allEpisodes.find(function(e){return e.id===epId});
  if(!ep){toast('未找到对局 #'+epId,false);return}
  var modal=document.getElementById('chart-modal');
  var title=modal.querySelector('.chart-modal-title');
  var body=modal.querySelector('.chart-modal-body');
  var scale=2.5;
  if(type==='deviation'){
    title.textContent='偏差曲线 — Episode #'+epId;
    body.innerHTML=renderDeviationChart(ep);
  }else{
    title.textContent='路径分叉图 — Episode #'+epId;
    body.innerHTML=renderForkTree(ep);
  }
  var svg=body.querySelector('svg');
  if(svg){
    var ow=parseFloat(svg.getAttribute('width'))||300;
    var oh=parseFloat(svg.getAttribute('height'))||150;
    svg.setAttribute('width',Math.round(ow*scale));
    svg.setAttribute('height',Math.round(oh*scale));
    svg.setAttribute('preserveAspectRatio','xMidYMid meet');
  }
  modal.classList.add('open');
}
function closeChartModal(){document.getElementById('chart-modal').classList.remove('open')}
document.getElementById('chart-modal').addEventListener('click',function(e){if(e.target===this)closeChartModal()});
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeChartModal()});
function _planColor(pid,trigger){
  if(trigger==='diverge'||trigger==='偏离触发')return{bg:'rgba(245,127,23,.12)',bd:'rgba(245,127,23,.35)'};
  if(trigger==='backup_switch')return{bg:'rgba(123,31,162,.10)',bd:'rgba(123,31,162,.3)'};
  return{bg:'rgba(13,71,161,.08)',bd:'rgba(13,71,161,.25)'};
}
function _etDotColor(et){
  var m={kg_plan:'#42a5f5',kg_follow:'#81c784',diverge:'#ffb74d',fallback:'#ef9a9a',backup_switch:'#ce93d8',backup_switch_exact:'#ce93d8',backup_switch_fuzzy:'#b39ddb',no_action:'#888',manual:'#90caf9'};
  return m[et]||'#888';
}
function _forkStateColor(et){
  var m={kg_plan:'#1565c0',kg_follow:'#2e7d32',diverge:'#e65100',fallback:'#b71c1c',backup_switch:'#7b1fa2',backup_switch_exact:'#7b1fa2',backup_switch_fuzzy:'#9575cd',no_action:'#616161',manual:'#1565c0'};
  return m[et]||'#616161';
}
function renderDeviationChart(ep){
  var evts=ep.events;if(!evts||evts.length<2)return'<div style="font-size:10px;color:var(--tx2);text-align:center;padding:8px">帧数据不足</div>';
  var n=Math.min(evts.length,40);
  var W=Math.max(n*28+60,300),H=170;
  var pad={t:20,r:20,b:30,l:45};
  var cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  var maxDev=0;
  var devs=[];
  for(var i=0;i<n;i++){
    var d=evts[i].deviation;
    devs.push(d!=null&&d!==undefined?d:null);
    if(d!=null&&d>maxDev)maxDev=d;
  }
  maxDev=Math.max(maxDev,0.1);
  var yScale=ch/maxDev;
  var xStep=cw/(n-1);
  var h='<svg width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'" xmlns="http://www.w3.org/2000/svg">';
  h+='<rect width="'+W+'" height="'+H+'" fill="var(--sf2)" rx="4"/>';
  h+='<line x1="'+pad.l+'" y1="'+pad.t+'" x2="'+pad.l+'" y2="'+(pad.t+ch)+'" stroke="var(--bd)" stroke-width="1"/>';
  h+='<line x1="'+pad.l+'" y1="'+(pad.t+ch)+'" x2="'+(pad.l+cw)+'" y2="'+(pad.t+ch)+'" stroke="var(--bd)" stroke-width="1"/>';
  h+='<text x="'+(pad.l-4)+'" y="'+pad.t+'" text-anchor="end" font-size="9" fill="var(--tx2)">'+maxDev.toFixed(2)+'</text>';
  h+='<text x="'+(pad.l-4)+'" y="'+(pad.t+ch)+'" text-anchor="end" font-size="9" fill="var(--tx2)">0</text>';
  for(var gi=0;gi<Math.min(5,n);gi++){
    var gx=pad.l+gi*xStep;
    h+='<text x="'+gx+'" y="'+(pad.t+ch+14)+'" text-anchor="middle" font-size="8" fill="var(--tx2)">'+gi+'</text>';
  }
  if(n>5){var gx2=pad.l+(n-1)*xStep;h+='<text x="'+gx2+'" y="'+(pad.t+ch+14)+'" text-anchor="middle" font-size="8" fill="var(--tx2)">'+(n-1)+'</text>'}
  var plans={};
  for(var i=0;i<n;i++){
    var pid=evts[i].plan_id||0;
    if(evts[i].plan)plans[pid]={start:i,trigger:evts[i].plan.trigger||evts[i].plan.mode||'',end:i};
    else if(plans[pid])plans[pid].end=i;
  }
  for(var pk in plans){
    var p=plans[pk],pc=_planColor(parseInt(pk),p.trigger);
    var x1=pad.l+p.start*xStep-2,x2=pad.l+p.end*xStep+2;
    h+='<rect x="'+x1+'" y="'+pad.t+'" width="'+(x2-x1)+'" height="'+ch+'" fill="'+pc.bg+'" opacity="0.6"/>';
    h+='<rect x="'+x1+'" y="'+pad.t+'" width="'+(x2-x1)+'" height="'+ch+'" fill="none" stroke="'+pc.bd+'" stroke-width="1" stroke-dasharray="3,2"/>';
  }
  for(var i=0;i<n;i++){
    var x=pad.l+i*xStep;
    if(i>0){
      var x0=pad.l+(i-1)*xStep;
      h+='<line x1="'+x0+'" y1="'+(pad.t+ch)+'" x2="'+x+'" y2="'+(pad.t+ch)+'" stroke="rgba(255,255,255,.06)" stroke-width="1"/>';
    }
  }
  var actualPts=[],plannedPts=[];
  for(var i=0;i<n;i++){
    var x=pad.l+i*xStep;
    var ay=pad.t+ch;
    var et=evts[i].event_type||'no_action';
    actualPts.push({x:x,y:ay,et:et,i:i});
    if(evts[i].planned_state!=null){
      var d=devs[i];
      if(d!=null){
        var py=pad.t+ch-d*yScale;
        plannedPts.push({x:x,y:py,i:i});
      }
    }
  }
  var aPath='';for(var i=0;i<actualPts.length;i++){aPath+=(i===0?'M':'L')+actualPts[i].x+' '+actualPts[i].y}
  h+='<path d="'+aPath+'" fill="none" stroke="var(--ac)" stroke-width="1.5"/>';
  var pPath='';for(var i=0;i<plannedPts.length;i++){pPath+=(i===0?'M':'L')+plannedPts[i].x+' '+plannedPts[i].y}
  if(pPath)h+='<path d="'+pPath+'" fill="none" stroke="#ce93d8" stroke-width="1.5" stroke-dasharray="4,3"/>';
  for(var i=0;i<n;i++){
    var x=pad.l+i*xStep;
    var d=devs[i];
    if(d!=null&&d>0.001){
      var py=pad.t+ch-d*yScale;
      var ay=pad.t+ch;
      var col=d<0.05?'#4caf50':d<0.15?'#ffc107':'#f44336';
      h+='<line x1="'+x+'" y1="'+ay+'" x2="'+x+'" y2="'+py+'" stroke="'+col+'" stroke-width="1" opacity="0.5"/>';
    }
  }
  for(var i=0;i<actualPts.length;i++){
    var p=actualPts[i],c=_etDotColor(p.et);
    var tip=escapeAttr('Step '+p.i+': S'+(evts[p.i].state_id!=null?evts[p.i].state_id:'?')+' (actual) | '+evts[p.i].event_type);
    h+='<circle cx="'+p.x+'" cy="'+p.y+'" r="3" fill="'+c+'" stroke="var(--sf2)" stroke-width="1" data-tip="'+tip+'"/>';
  }
  for(var i=0;i<plannedPts.length;i++){
    var p=plannedPts[i];
    var tip=escapeAttr('Step '+p.i+': S'+(evts[p.i].planned_state!=null?evts[p.i].planned_state:'?')+' (planned) | dev: '+devs[p.i]);
    h+='<rect x="'+(p.x-2.5)+'" y="'+(p.y-2.5)+'" width="5" height="5" fill="#ce93d8" stroke="var(--sf2)" stroke-width="1" data-tip="'+tip+'"/>';
  }
  var lx=pad.l+cw-90,ly=pad.t+6;
  h+='<rect x="'+lx+'" y="'+ly+'" width="88" height="30" fill="var(--sf2)" rx="3" opacity="0.85"/>';
  h+='<line x1="'+(lx+4)+'" y1="'+(ly+10)+'" x2="'+(lx+20)+'" y2="'+(ly+10)+'" stroke="var(--ac)" stroke-width="1.5"/>';
  h+='<text x="'+(lx+24)+'" y="'+(ly+13)+'" font-size="9" fill="var(--tx2)">实际路径</text>';
  h+='<line x1="'+(lx+4)+'" y1="'+(ly+22)+'" x2="'+(lx+20)+'" y2="'+(ly+22)+'" stroke="#ce93d8" stroke-width="1.5" stroke-dasharray="4,3"/>';
  h+='<text x="'+(lx+24)+'" y="'+(ly+25)+'" font-size="9" fill="var(--tx2)">规划路径</text>';
  h+='</svg>';
  return h;
}
function _distColor(d){
  if(d==null)return '#888';
  if(d<0.05)return '#4caf50';
  if(d<0.15)return '#ffc107';
  return '#f44336';
}
function renderForkTree(ep){
  var tree=ep.fork_tree;
  if(!tree||!tree.nodes||tree.nodes.length<2)return'<div style="font-size:10px;color:var(--tx2);text-align:center;padding:8px">无分叉树数据</div>';
  var nodes=tree.nodes,edges=tree.edges,coords=tree.coords;
  if(!coords||Object.keys(coords).length===0)return'<div style="font-size:10px;color:var(--tx2);text-align:center;padding:8px">布局数据缺失</div>';
  var maxX=0,maxY=0;
  for(var nid in coords){
    if(coords[nid][0]>maxX)maxX=coords[nid][0];
    if(coords[nid][1]>maxY)maxY=coords[nid][1];
  }
  var padR=50,padB=40;
  var W=Math.max(maxX+padR,200),H=Math.max(maxY+padB,150);
  var nodeMap={};
  for(var i=0;i<nodes.length;i++)nodeMap[nodes[i].id]=nodes[i];
  var h='<svg width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'" xmlns="http://www.w3.org/2000/svg">';
  h+='<rect width="'+W+'" height="'+H+'" fill="var(--sf2)" rx="4"/>';
  for(var i=0;i<edges.length;i++){
    var e=edges[i];
    var pf=coords[e.from],pt=coords[e.to];
    if(!pf||!pt)continue;
    if(e.type==='actual'){
      var c=_etDotColor(e.et);
      h+='<line x1="'+pf[0]+'" y1="'+pf[1]+'" x2="'+pt[0]+'" y2="'+pt[1]+'" stroke="'+c+'" stroke-width="2.5"/>';
      if(e.dist!=null){
        var mx=(pf[0]+pt[0])/2,my=(pf[1]+pt[1])/2;
        h+='<text x="'+mx+'" y="'+(my-4)+'" text-anchor="middle" font-size="6" fill="'+c+'" opacity="0.7">'+e.dist.toFixed(2)+'</text>';
      }
    }
  }
  for(var i=0;i<edges.length;i++){
    var e=edges[i];
    if(e.type!=='beam')continue;
    var pf=coords[e.from],pt=coords[e.to];
    if(!pf||!pt)continue;
    var sw=e.chosen?2:1;
    var op=e.chosen?1:0.45;
    var dc=_distColor(e.dist);
    var dash=e.chosen?'':'stroke-dasharray="5,3" ';
    h+='<line x1="'+pf[0]+'" y1="'+pf[1]+'" x2="'+pt[0]+'" y2="'+pt[1]+'" stroke="'+dc+'" stroke-width="'+sw+'" opacity="'+op+'" '+dash+'/>';
    if(e.dist!=null&&e.dist>0.001){
      var mx2=(pf[0]+pt[0])/2,my2=(pf[1]+pt[1])/2;
      h+='<text x="'+mx2+'" y="'+(my2-3)+'" text-anchor="middle" font-size="5" fill="'+dc+'" opacity="0.8">'+e.dist.toFixed(2)+'</text>';
    }
  }
  for(var i=0;i<nodes.length;i++){
    var nd=nodes[i];
    var p=coords[nd.id];
    if(!p)continue;
    if(nd.type==='actual'){
      var c=_etDotColor(nd.et);
      var tip=escapeAttr('Step '+nd.frame+': '+nd.label+' (实际) | '+nd.et+' | plan#'+nd.planId);
      h+='<circle cx="'+p[0]+'" cy="'+p[1]+'" r="6" fill="'+c+'" stroke="var(--sf2)" stroke-width="1.5" data-tip="'+tip+'"/>';
      h+='<text x="'+p[0]+'" y="'+(p[1]+15)+'" text-anchor="middle" font-size="8" fill="var(--tx)" font-weight="600">'+nd.label+'</text>';
    }else{
      var fill=nd.chosen?'#ce93d8':'none';
      var stroke=nd.chosen?'#ce93d8':'#b39ddb';
      var pathsStr=(nd.pathIndices||[]).join(',');
      var tip=escapeAttr(nd.label+' | paths:['+pathsStr+']'+(nd.chosen?' [选中]':'')+' | step '+nd.stepIdx+' @frame '+nd.frame);
      var dx=5,dy=5;
      h+='<polygon points="'+p[0]+','+(p[1]-dy)+' '+(p[0]+dx)+','+p[1]+' '+p[0]+','+(p[1]+dy)+' '+(p[0]-dx)+','+p[1]+'" fill="'+fill+'" stroke="'+stroke+'" stroke-width="1" data-tip="'+tip+'"/>';
      h+='<text x="'+(p[0]+7)+'" y="'+(p[1]+3)+'" font-size="7" fill="var(--tx2)">'+nd.label+'</text>';
    }
  }
  var lx=4,ly=4;
  h+='<rect x="'+lx+'" y="'+ly+'" width="260" height="20" fill="var(--sf2)" rx="3" opacity="0.9"/>';
  h+='<circle cx="'+(lx+8)+'" cy="'+(ly+10)+'" r="4" fill="var(--ac)"/>';
  h+='<text x="'+(lx+16)+'" y="'+(ly+13)+'" font-size="7" fill="var(--tx2)">实际决策</text>';
  h+='<polygon points="'+(lx+75)+','+(ly+6)+' '+(lx+79)+','+(ly+10)+' '+(lx+75)+','+(ly+14)+' '+(lx+71)+','+(ly+10)+'" fill="#ce93d8" stroke="#ce93d8" stroke-width="1"/>';
  h+='<text x="'+(lx+84)+'" y="'+(ly+13)+'" font-size="7" fill="var(--tx2)">选中规划</text>';
  h+='<polygon points="'+(lx+143)+','+(ly+6)+' '+(lx+147)+','+(ly+10)+' '+(lx+143)+','+(ly+14)+' '+(lx+139)+','+(ly+10)+'" fill="none" stroke="#b39ddb" stroke-width="1"/>';
  h+='<text x="'+(lx+152)+'" y="'+(ly+13)+'" font-size="7" fill="var(--tx2)">备选路径</text>';
  h+='<line x1="'+(lx+205)+'" y1="'+(ly+6)+'" x2="'+(lx+205)+'" y2="'+(ly+14)+'" stroke="#4caf50" stroke-width="1"/>';
  h+='<text x="'+(lx+210)+'" y="'+(ly+13)+'" font-size="7" fill="var(--tx2)">近</text>';
  h+='<line x1="'+(lx+228)+'" y1="'+(ly+6)+'" x2="'+(lx+228)+'" y2="'+(ly+14)+'" stroke="#f44336" stroke-width="1"/>';
  h+='<text x="'+(lx+233)+'" y="'+(ly+13)+'" font-size="7" fill="var(--tx2)">远</text>';
  h+='</svg>';
  return h;
}
function renderDeviationSection(ep){
  var h='<div class="ep-dev-area" id="dev-area-'+ep.id+'">';
  h+='<div class="ep-dev-toggle" onclick="toggleDevArea('+ep.id+')">';
  h+='<span class="arrow">&#9654;</span><span>规划偏差分析</span>';
  h+='</div>';
  h+='<div class="ep-dev-body">';
  h+='<div class="ep-dev-chart-wrap"><div class="ep-dev-section-title">偏差曲线 <button class="chart-zoom-btn" onclick="showChartModal('+ep.id+',&#39;deviation&#39;)" title="放大显示">&#x2922;</button></div>'+renderDeviationChart(ep)+'</div>';
  h+='<div class="ep-dev-tree-wrap"><div class="ep-dev-section-title">路径分叉图 <button class="chart-zoom-btn" onclick="showChartModal('+ep.id+',&#39;fork&#39;)" title="放大显示">&#x2922;</button></div>'+renderForkTree(ep)+'</div>';
  h+='</div></div>';
  return h;
}
function renderPlanInline(pl,idx){
  var h='<div class="ep-plan-inline">';
  h+='<span class="plan-label">规划 #'+idx+' — S'+pl.state_id;
  var trigger=pl.trigger||'';
  var trigLbl={diverge:'偏离触发',exhausted:'用尽重规划',single_step:'单步规划'}[trigger]||trigger||pl.mode;
  h+=' ('+trigLbl+')</span>';
  h+='<div class="plan-content">';
  if(pl.beam_paths&&pl.beam_paths.length>0){
    for(var pi=0;pi<pl.beam_paths.length;pi++){
      var path=pl.beam_paths[pi];
      var chosen=path.chosen?'chosen':'';
      h+='<div class="ep-beam-path '+chosen+'">';
      h+='<span class="path-label">'+(path.chosen?'[选中] ':'')+path.rank+'</span>';
      h+='<span class="path-metrics"><span>CumP:'+(path.cum_prob*100).toFixed(1)+'%</span><span>'+(path.steps.length-1)+'步</span></span>';
      h+='<br>';
      for(var si=0;si<path.steps.length;si++){
        var st=path.steps[si];
        if(si===0){h+='<span class="ep-beam-step" style="font-weight:600">S'+st.state+'</span>'}
        else{h+='<span class="path-arrow">&rarr;</span>';if(st.action&&st.action!=='')h+='<span class="ep-beam-step" style="color:var(--ac)">'+st.action+'</span>';h+='<span class="ep-beam-step">S'+st.state+'</span>'}
      }
      h+='</div>';
    }
  }else if(pl.action_plan&&pl.action_plan.length>0){
    h+='<div class="ep-beam-path chosen"><span class="path-label">[选中] 1</span><br>';
    for(var a=0;a<pl.action_plan.length;a++){
      if(a===0){h+='<span class="ep-beam-step" style="font-weight:600">S'+pl.planned_states[a]+'</span>'}
      else{h+='<span class="path-arrow">&rarr;</span><span class="ep-beam-step" style="color:var(--ac)">'+pl.action_plan[a]+'</span><span class="ep-beam-step">S'+pl.planned_states[a]+'</span>'}
    }
    h+='</div>';
  }
  if(pl.beam_results&&pl.beam_results.length>0){
    h+='<details style="margin-top:4px"><summary style="font-size:10px;color:var(--tx2);cursor:pointer">Beam ('+pl.beam_results.length+' 节点)</summary>';
    h+='<table class="ep-beam-table"><tr><th>Step</th><th>State</th><th>Action</th><th>Beam</th><th>WR</th><th>Q</th><th>CumP</th></tr>';
    for(var b=0;b<pl.beam_results.length;b++){
      var br=pl.beam_results[b];
      h+='<tr><td>'+br.step+'</td><td>S'+br.state+'</td><td>'+(br.action||'-')+'</td><td>B'+br.beam_id+'</td><td>'+(br.win_rate*100).toFixed(1)+'%</td><td>'+br.quality_score.toFixed(1)+'</td><td>'+br.cumulative_probability.toFixed(4)+'</td></tr>';
    }
    h+='</table></details>';
  }
  h+='</div></div>';
  return h;
}

async function init(){
  if(localStorage.getItem('live_theme')!=='dark'){document.documentElement.classList.add('light');document.getElementById('theme-btn').innerHTML='&#9728;'}
  loadFilterConfig();
  loadWindowPos();
  await refresh();
  startTimer();
  startLogTimer();
  fetchLogs();
  loadEpisodes();
  startEpAutoRefresh();
}
init();
</script>
</body></html>"""
    return _html.replace("__HOST__", host).replace("__PORT__", str(port))
