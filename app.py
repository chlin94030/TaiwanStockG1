"""
Taiwan Alpha Radar V12.0 Nordic Design Research App.
Enlarged Typography, Plain Language, Full Technical K-Line Indicators (MA/Bollinger/KD/RSI), Clear Form Units.
Run: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import html
import os
import gc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import radar_service as service
from market_data import DailyPriceStore, _taipei_timestamp
from trading_calendar import calendar_reference, daily_freshness, entry_review_allowed
from presentation import plain_summary
from policy_engine import HORIZONS
from return_first_model import holding_review, finite_scalar

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ALPHA_RADAR_DATA_DIR", str(ROOT / "data")))
VIEW_LABELS = ["⭐ 極選", "⚡ 短線", "📈 中線", "🧭 長波段", "🔎 個股診斷"]
HORIZON_LABELS = {"short": "短線 · 10交易日", "mid": "中線 · 40交易日", "long": "長波段 · 120交易日"}
FAMILY_LABELS = {
    "價量研究｜無財報／法人代理": "price_only",
    "基本面確認｜需要歷史公告資料": "business_confirmed",
    "法人確認｜需要歷史籌碼資料": "flow_confirmed",
    "完整證據｜基本面＋法人": "full",
}
SETUP_LABELS = {
    "BREAKOUT": "平台突破", "PULLBACK": "趨勢回測", "RECLAIM": "重新站回",
    "TREND": "趨勢延續", "BASE": "整理觀察", "DRYUP": "低量沉寂",
}
STATE_LABELS = {
    "CONDITIONS_MET_NOT_FILLED": "收盤符合條件 · 次日開盤確認",
    "WAIT_ENTRY_ZONE": "等待回到最佳布局區",
    "WAIT_BREAKOUT": "等待突破關鍵價",
    "WAIT_CONFIRMATION": "量價確認未成立",
    "DO_NOT_CHASE": "超出追價上限 · 靜待回測",
    "INVALIDATED": "原結構已失效",
    "DATA_UNVERIFIED": "資料待核對",
    "NO_RETURN_ESTIMATE": "無可用報酬估算",
}
HOLD_LABELS = {
    "DATA_UNVERIFIED": "價格資料未核對，暫不作留／賣判斷",
    "ORIGINAL_STRUCTURE_INVALIDATED": "⚠️️ 結構失效：已跌破防守價，建議優先考慮出場",
    "ORIGINAL_THESIS_INVALIDATED": "⚠️ 買進理由失效：原看多條件不存在，應重新審視",
    "PROTECTION_TRIGGER_REVIEW_EXECUTION": "🔔 觸及獲利保護價：建議核對報價準備落袋為安",
    "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW": "缺少原始停損價或買進理由，需手動評估",
    "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE": "✅ 尚未破壞防守結構；請持續依紀律觀察",
}

CSS = """
<style>
:root { --slate-900:#0f172a; --slate-800:#1e293b; --slate-600:#475569; --slate-100:#f1f5f9; --blue-600:#2563eb; --emerald-600:#059669; --rose-600:#e11d48; }
.stApp { background:#f8fafc; color:#0f172a; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.block-container { max-width:1080px; padding-top:1.5rem; padding-bottom:5rem; }

/* 北歐極簡 Hero 區塊 */
.hero { padding:32px 28px; border-radius:28px; background:linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #1e3a8a 100%); color:white; margin-bottom:24px; box-shadow:0 20px 25px -5px rgba(15,23,42,0.15); }
.hero .eyebrow { letter-spacing:.15em; font-size:.9rem; font-weight:800; color:#93c5fd; text-transform:uppercase; }
.hero h1 { font-size:2.4rem; font-weight:900; line-height:1.2; color:white; margin:.5rem 0; letter-spacing:-.02em; }
.hero p { font-size:1.1rem; opacity:.9; margin:.5rem 0 0; line-height:1.6; color:#e2e8f0; }

.statusline { font-size:1rem; color:#475569; margin:12px 0 20px; font-weight:600; background:#f1f5f9; padding:10px 16px; border-radius:12px; display:inline-block; }

/* 北歐風卡片設計 */
.card { background:white; border:1px solid #e2e8f0; border-radius:24px; padding:28px; margin:20px 0; box-shadow:0 10px 30px -5px rgba(0,0,0,0.05); }
.card-head { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
.card-title { font-size:1.85rem; font-weight:900; line-height:1.2; color:#0f172a; letter-spacing:-.01em; }
.card-code { font-size:1.05rem; font-weight:600; color:#64748b; margin-top:6px; }
.card-price { font-size:1.75rem; font-weight:900; text-align:right; color:#0f172a; white-space:nowrap; }

.badge { display:inline-block; font-size:.92rem; font-weight:800; padding:7px 14px; border-radius:12px; background:#f1f5f9; color:#475569; margin:12px 8px 10px 0; }
.badge-blue { color:#1d4ed8; background:#dbeafe; }
.badge-emerald { color:#047857; background:#d1fae5; }

.topnote { padding:16px 20px; border:1px solid #bfdbfe; background:#eff6ff; border-radius:18px; margin:16px 0; color:#1e40af; font-size:1.05rem; font-weight:600; line-height:1.6; }

/* 大字級 EV 區塊 */
.return-box { background:linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border:1px solid #bae6fd; border-radius:20px; padding:22px 24px; margin:16px 0; }
.return-k { font-size:1.1rem; font-weight:800; color:#0369a1; }
.return-v { font-size:3.2rem; font-weight:900; letter-spacing:-.04em; line-height:1.1; color:#0284c7; margin:6px 0; }
.return-desc { font-size:1rem; color:#0369a1; font-weight:700; margin-top:4px; }

/* 白話數據網格 */
.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0; }
.stat { border:1px solid #f1f5f9; background:#fafafa; padding:14px 16px; border-radius:16px; }
.stat .k { font-size:.9rem; font-weight:700; color:#64748b; }
.stat .v { font-size:1.35rem; font-weight:900; color:#0f172a; margin-top:6px; }

.decision { padding:16px 20px; border-radius:16px; background:#fef3c7; border:1px solid #fde68a; color:#92400e; font-weight:800; font-size:1.15rem; margin-top:12px; }
.decision-ok { background:#d1fae5; border-color:#a7f3d0; color:#065f46; }

.levels { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:12px; }
.level { background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:14px; }
.level .k { font-size:.9rem; font-weight:700; color:#64748b; }
.level .v { font-weight:900; font-size:1.25rem; color:#0f172a; margin-top:4px; }

@media(max-width:650px){
 .block-container{ padding:1rem 1rem 4rem; }
 .hero{ padding:24px 20px; border-radius:22px; }
 .hero h1{ font-size:1.8rem; }
 .card{ padding:20px 16px; border-radius:20px; }
 .card-title{ font-size:1.5rem; }
 .card-price{ font-size:1.4rem; }
 .return-v{ font-size:2.5rem; }
 .stats{ grid-template-columns:repeat(2,1fr); }
 .levels{ grid-template-columns:repeat(2,1fr); }
}
</style>
"""

def esc(value): return html.escape(str(value))

def percent(value, signed=True):
    v = finite_scalar(value)
    return "—" if not np.isfinite(v) else f"{v*100:{'+' if signed else ''}.2f}%"

def money(value):
    v = finite_scalar(value)
    return "—" if not np.isfinite(v) else f"{v:,.2f}".rstrip("0").rstrip(".")

def render_chart(chart, plan, key):
    if not chart or "ohlcv" not in chart:
        st.caption("這檔股票 K 線未在快取中，診斷時將自動連線載入。")
        return
    
    df = pd.DataFrame(chart["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"])
    dates = chart.get("dates", [])
    if df.empty: return

    # 計算全套技術指標
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()
    
    std20 = close.rolling(20).std()
    df["Boll_Upper"] = df["MA20"] + 2 * std20
    df["Boll_Lower"] = df["MA20"] - 2 * std20
    
    # KD 指標
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9 + 1e-6) * 100
    df["K"] = rsv.ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()
    
    # RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df["RSI"] = 100 - (100 / (1 + rs))

    # 建立多分頁/多子圖主圖 (K線 + 成交量 + KD + RSI)
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.50, 0.16, 0.17, 0.17],
        subplot_titles=("K線 / 均線 / 布林通道", "成交量 (張)", "KD 指標 (9,3,3)", "RSI 指標 (14)")
    )

    # 1. 主 K 線 (台股台標準：紅 K 上漲 #ef4444，綠 K 下跌 #10b981)
    fig.add_trace(go.Candlestick(
        x=dates, open=df.Open, high=df.High, low=df.Low, close=df.Close,
        increasing_line_color="#ef4444", increasing_fillcolor="#ef4444",
        decreasing_line_color="#10b981", decreasing_fillcolor="#10b981",
        name="K線"
    ), row=1, col=1)

    # 均線
    fig.add_trace(go.Scatter(x=dates, y=df.MA5, line=dict(color="#f59e0b", width=1.2), name="5日線(周)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MA20, line=dict(color="#2563eb", width=1.5), name="20日線(月)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MA60, line=dict(color="#9333ea", width=1.5), name="60日線(季)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MA120, line=dict(color="#ec4899", width=1.5), name="120日線(半年)"), row=1, col=1)

    # 布林通道
    fig.add_trace(go.Scatter(x=dates, y=df.Boll_Upper, line=dict(color="#94a3b8", width=1, dash="dot"), name="布林上軌"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.Boll_Lower, line=dict(color="#94a3b8", width=1, dash="dot"), fill='tonexty', fillcolor='rgba(148,163,184,0.08)', name="布林下軌"), row=1, col=1)

    # 關鍵進出場價位框
    if plan:
        fig.add_hrect(y0=plan.get("zone_low", 0), y1=plan.get("zone_high", 0), line_width=0, fillcolor="rgba(37,99,235,0.12)", row=1, col=1)
        fig.add_hline(y=plan.get("trigger", 0), line_color="#d97706", line_dash="dot", row=1, col=1)
        fig.add_hline(y=plan.get("invalidation", 0), line_color="#e11d48", line_dash="dash", row=1, col=1)

    # 2. 成交量
    vol_colors = np.where(df.Close >= df.Open, "#ef4444", "#10b981")
    fig.add_trace(go.Bar(x=dates, y=df.Volume, marker_color=vol_colors, name="成交量"), row=2, col=1)

    # 3. KD 指標
    fig.add_trace(go.Scatter(x=dates, y=df.K, line=dict(color="#2563eb", width=1.5), name="K值"), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.D, line=dict(color="#f59e0b", width=1.5), name="D值"), row=3, col=1)
    fig.add_hline(y=80, line_color="#cbd5e1", line_dash="dash", row=3, col=1)
    fig.add_hline(y=20, line_color="#cbd5e1", line_dash="dash", row=3, col=1)

    # 4. RSI 指標
    fig.add_trace(go.Scatter(x=dates, y=df.RSI, line=dict(color="#059669", width=1.5), name="RSI(14)"), row=4, col=1)
    fig.add_hline(y=70, line_color="#cbd5e1", line_dash="dash", row=4, col=1)
    fig.add_hline(y=30, line_color="#cbd5e1", line_dash="dash", row=4, col=1)

    fig.update_layout(height=650, margin=dict(l=10, r=10, t=25, b=10), showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    fig.update_xaxes(type="category", nticks=6, fixedrange=True, showgrid=False)
    fig.update_yaxes(fixedrange=True, gridcolor="#f1f5f9")
    
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"displayModeBar": False, "displaylogo": False, "scrollZoom": False})

def card(obj, h, snap, view, chart=None, calendar=None, rank_idx=1):
    if not isinstance(obj, dict): return
    block = obj.get("horizons", {}).get(h, {})
    f = block.get("forecast") or {}
    plan = block.get("plan")
    summary = f.get("strategy") or {}
    condition = block.get("entry_state", "NO_RETURN_ESTIMATE")
    state_label = STATE_LABELS.get(condition, condition)
    
    calendar = calendar or calendar_reference(DATA_DIR, now=_taipei_timestamp())
    fresh = daily_freshness(obj.get("price_date", ""), calendar)
    review_ok = entry_review_allowed(obj.get("price_date", ""), plan, calendar)
    
    headline = f"TOP {rank_idx} 建議標的｜{state_label}"
    if not fresh.get("current_daily", False): headline = "請點擊上方更新最新日線行情"
    
    ev = percent(summary.get("mean"))
    conf_score = f.get("confidence_score", 75.0)
    factor_score = f.get("composite_factor_score", 70.0)
    summary_sentence = plain_summary(f)
    setup = SETUP_LABELS.get(obj.get("setup"), obj.get("setup", ""))

    st.markdown(f"""
<div class="card">
 <div class="card-head"><div><div class="card-title">#{rank_idx} {esc(obj.get('name', obj.get('ticker', '')))}</div>
 <div class="card-code">{esc(obj.get('ticker', ''))} · {esc(obj.get('industry',''))}</div></div>
 <div class="card-price">{money(obj.get('price', 0))} 元<div class="card-code">{esc(obj.get('price_date', ''))} 最新日線</div></div></div>
 <span class="badge badge-blue">{esc(HORIZON_LABELS.get(h, h))}</span><span class="badge">{esc(setup)}</span>
 <span class="badge badge-emerald">動能總分：{factor_score:.1f} 分</span>
 <div class="topnote"><b>量化診斷白話解析：</b>{esc(summary_sentence)}</div>
 <div class="return-box">
   <div class="return-k">策略預期淨報酬率</div>
   <div class="return-v">{ev}</div>
   <div class="return-desc">綜合動能評估：<b>{factor_score:.1f} 分</b>｜模型評估勝率信心：<b>{conf_score:.1f}%</b></div>
 </div>
 <div class="stats">
  <div class="stat"><div class="k">領先大盤幅度</div><div class="v">{percent(f.get('alpha_mean'))}</div></div>
  <div class="stat"><div class="k">平穩行情預估 (中位數)</div><div class="v">{percent(summary.get('median'))}</div></div>
  <div class="stat"><div class="k">強勢行情預估 (前25%)</div><div class="v">{percent(summary.get('p75'))}</div></div>
  <div class="stat"><div class="k">極端拉回風險 (最大風險)</div><div class="v">{percent(summary.get('expected_shortfall10_loss'), False)}</div></div>
 </div>
 <div class="decision {'decision-ok' if review_ok and condition=='CONDITIONS_MET_NOT_FILLED' else ''}">{esc(headline)}</div>
</div>""", unsafe_allow_html=True)
    
    with st.expander("📊 點此展開「技術分析 K 線圖」與「進出場價格規劃」", expanded=False):
        if plan:
            st.markdown(f"""<div class="levels">
<div class="level"><div class="k">建議布局區</div><div class="v">{money(plan.get('zone_low'))}–{money(plan.get('zone_high'))} 元</div></div>
<div class="level"><div class="k">突破確認價</div><div class="v">{money(plan.get('trigger'))} 元</div></div>
<div class="level"><div class="k">不追價上限</div><div class="v">{money(plan.get('chase_limit'))} 元</div></div>
<div class="level"><div class="k">結構失效停損價</div><div class="v">{money(plan.get('invalidation'))} 元</div></div>
</div>""", unsafe_allow_html=True)
        
        snap_id = snap.get("snapshot_id", "default") if isinstance(snap, dict) else "default"
        chart = chart or (snap.get("charts", {}).get(obj.get("ticker")) if isinstance(snap, dict) else None)
        if chart is None and isinstance(snap, dict):
            try: chart = service.chart_on_demand(snap, obj.get("ticker", ""), DATA_DIR, allow_fetch=False)
            except Exception: pass
        render_chart(chart, plan, f"chart_{view}_{h}_{obj.get('ticker')}_{snap_id}")

def render_horizon(snap, h, calendar=None):
    st.subheader(HORIZON_LABELS.get(h, h))
    if not snap or not isinstance(snap, dict):
        st.info("尚無收益快照，請點擊上方『⚡ 更新市場與報酬研究』。")
        return

    # 強制推薦 5 檔
    picked = service.select_market_best(snap, h, n=5)
        
    if picked:
        st.caption(f"依據多因子量化矩陣（RS大盤強度＋多頭結構＋攻擊量）為您嚴選 TOP {len(picked)} 強勢標的：")
        for idx, obj in enumerate(picked, 1):
            card(obj, h, snap, h, calendar=calendar, rank_idx=idx)
    else:
        st.info("目前市場環境下無滿足過濾條件之標的。")

def main():
    st.set_page_config(page_title="Alpha Radar · Nordic UI V12.0", page_icon="📈", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="hero"><div class="eyebrow">TAIWAN ALPHA RADAR · V12.0 NORDIC</div>
<h1>全台股收益導向量化選股與個股診斷</h1>
<p>2,000+ 檔上市櫃開放資料動態母池 × 多因子動能打分 × 完整指標 K 線圖</p></div>""", unsafe_allow_html=True)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if st.session_state.get("v8_version") != service.OPERATIONS_VERSION:
        for key in ("v8_snapshot", "v8_doctor", "v8_error"):
            st.session_state.pop(key, None)
        st.session_state["v8_version"] = service.OPERATIONS_VERSION
        try:
            previous = service.load_dashboard(DATA_DIR / "dashboard_snapshot.json", include_features=False)
            if previous:
                st.session_state["v8_snapshot"] = service.compact_session_dashboard(previous)
        except Exception: pass

    with st.sidebar:
        st.markdown("### 資料與研究設定")
        family_label = st.selectbox("報酬模型的資料範圍", list(FAMILY_LABELS), key="family_v12")
        refs = st.selectbox("歷史參考股票數", [160, 300, 600], key="reference_v12")
        period = st.selectbox("歷史研究長度", ["5y", "8y", "10y", "3y"], key="period_v12")
        
        with st.expander("維護與快取", expanded=False):
            if st.button("強制清除行情與舊快照", key="clear_prices_v12"):
                DailyPriceStore(DATA_DIR / "daily_prices.sqlite").clear()
                service.remove_saved_dashboard(DATA_DIR / "dashboard_snapshot.json")
                st.session_state.pop("v8_snapshot", None)
                st.success("行情與舊快照已完全重置！")

    settings = service.RunSettings(
        reference_size=int(refs), candidate_size=300, history_period=period,
        model_family=FAMILY_LABELS[family_label]
    )

    if st.button("⚡ 更新市場與報酬研究（即時連線全台股 Open Data）", type="primary", use_container_width=True, key="run_scan_v12"):
        progress = st.progress(0, text="準備資料")
        try:
            def update(stage, value):
                progress.progress(min(1., max(0., value)), text="連線抓取盤面與計算多因子：" + stage)
            snap = service.run_scan(DATA_DIR, settings, progress=update)
            compact = service.compact_session_dashboard(snap)
            st.session_state["v8_snapshot"] = compact
            st.session_state.pop("v8_doctor", None)
            st.session_state.pop("v8_error", None)
            del snap, compact
            gc.collect()
        except Exception as exc:
            st.session_state["v8_error"] = f"{type(exc).__name__}: {exc}"
            st.error("掃描未完成，已保留上一份成功快照。")
        finally:
            progress.empty()

    snap = st.session_state.get("v8_snapshot")
    calendar = calendar_reference(DATA_DIR, now=_taipei_timestamp())

    if snap and isinstance(snap, dict):
        st.markdown(f"""<div class="statusline">截至 <b>{esc(snap.get('price_date',''))}</b> · 即時母池 {snap.get('coverage',{}).get('requested',0):,} 檔 · 深度過濾 {snap.get('candidate_n',0):,} 檔</div>""", unsafe_allow_html=True)

    view = st.radio("功能", VIEW_LABELS, horizontal=True, label_visibility="collapsed", key="view_v12")

    if view == VIEW_LABELS[0]:
        st.subheader("⭐ 各週期代表標的 (自動跨週期去重)")
        if not snap or not isinstance(snap, dict):
            st.info("尚無收益快照，請點擊上方『⚡ 更新市場與報酬研究』進行連線掃描。")
        else:
            used_tickers = []
            for h in HORIZONS:
                picks = service.select_market_best(snap, h, n=5)
                valid_picks = [p for p in picks if p["ticker"] not in used_tickers]
                if valid_picks:
                    obj = valid_picks[0]
                    used_tickers.append(obj["ticker"])
                    card(obj, h, snap, "prime", calendar=calendar)
    elif view == VIEW_LABELS[4]:
        st.subheader("🔎 個股診斷與持股檢視")
        st.caption("填寫您的持股資料，系統將協助比對結構是否健全與關鍵防守價位。")
        
        with st.form("doctor_form_v12"):
            code = st.text_input("股票代碼", value="2330", key="doctor_code_v12", help="請填寫 4 位數字代碼，如：2330 或 6187")
            horizon_text = st.selectbox("評估週期", list(HORIZON_LABELS.values()), index=1, key="doctor_h_v12")
            own = st.checkbox("已有持股（勾選後開啟部位風控計算）", key="doctor_own_v12")
            
            c1, c2 = st.columns(2)
            with c1:
                cost = st.number_input("持股買進成本（單位：元/股）", min_value=0., value=0., key="doctor_cost_v12", help="例如 580.5 元，若無填 0")
                invalid = st.number_input("原始停損價（單位：元/股）", min_value=0., value=0., key="doctor_stop_v12", help="若無預設停損填 0")
            with c2:
                shares = st.number_input("持股數量（單位：股）", min_value=0, value=0, step=1000, key="doctor_qty_v12", help="請填寫股數，例如 1 張請填 1000 股")
                trail = st.number_input("移動獲利保護價（單位：元/股）", min_value=0., value=0., key="doctor_trail_v12", help="若無填 0")
            
            thesis = st.selectbox("原買進理由是否仍成立？", ["尚未確認", "看多理由仍成立", "看多理由已不成立"], key="doctor_thesis_v12")
            submitted = st.form_submit_button("立即診斷標的", type="primary", use_container_width=True)
            
        if submitted and snap:
            dr = service.diagnose(code, snap, DATA_DIR)
            dr["h"] = next(k for k, v in HORIZON_LABELS.items() if v == horizon_text)
            st.session_state["v8_doctor"] = service.compact_doctor_result(dr)
            
        dr = st.session_state.get("v8_doctor")
        if dr and snap and "stock" in dr:
            card(dr["stock"], dr.get("h", "mid"), snap, "doctor", dr.get("chart"), calendar=calendar)
            if own:
                result = holding_review(dr["stock"]["price"], original_invalidation=invalid, trailing_protection=trail, thesis_broken=(thesis == "看多理由已不成立"))
                st.markdown("#### 既有持股紀律檢查報告")
                st.info(HOLD_LABELS.get(result, result))
    else:
        render_horizon(snap, {VIEW_LABELS[1]:"short", VIEW_LABELS[2]:"mid", VIEW_LABELS[3]:"long"}[view], calendar=calendar)

if __name__ == "__main__":
    main()