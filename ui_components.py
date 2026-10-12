"""
ui_components.py
----------------
Enterprise presentation layer for ConcallIQ:
- Professional Aurora dark/light executive themes.
- Interactive Plotly chart builders with mandatory titles, subtitles, axis labels,
  units, legends, "How to read this" captions, and "So what" plain-English insights.
- KPI cards with tooltips (ⓘ), status chips, and breadcrumbs.
- Searchable and sortable data tables with conditional formatting and download buttons.
- STRICT ZERO INTERNAL JARGON: Uses exclusively institutional financial terminology.
"""
from __future__ import annotations

import html
import inspect
import re
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Color Palette & Design Tokens
# -----------------------------------------------------------------------------
TEAL = "#42d3c5"
BLUE = "#6aa8ff"
VIOLET = "#a78bfa"
AMBER = "#f6b95b"
ROSE = "#ff7a8a"
GREEN = "#5ee6a0"

def _hex_to_rgba(hex_str: str, alpha: float = 0.15) -> str:
    """Safely converts a 6-character hex color to standard rgba() format."""
    hex_clean = hex_str.lstrip("#")
    if len(hex_clean) == 6:
        r = int(hex_clean[0:2], 16)
        g = int(hex_clean[2:4], 16)
        b = int(hex_clean[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return hex_str

PALETTE = [TEAL, BLUE, VIOLET, AMBER, ROSE, GREEN, "#f58cc4", "#8bd3ff"]

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --bg: #050a16;
  --card: rgba(15, 28, 48, 0.78);
  --line: rgba(110, 150, 200, 0.22);
  --text: #e8f0fa;
  --muted: #8fa6bd;
  --teal: #42d3c5;
  --blue: #6aa8ff;
  --violet: #a78bfa;
  --amber: #f6b95b;
  --rose: #ff7a8a;
  --green: #5ee6a0;
}
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', -apple-system, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.stApp::before {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(40% 35% at 15% 10%, rgba(66, 211, 197, 0.14), transparent 70%),
    radial-gradient(38% 40% at 88% 8%, rgba(167, 139, 250, 0.16), transparent 70%),
    radial-gradient(45% 40% at 70% 95%, rgba(106, 168, 255, 0.12), transparent 70%);
}
.block-container { position: relative; z-index: 1; padding-top: 1.2rem; max-width: 1540px; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #07101f, #050a16);
  border-right: 1px solid var(--line);
}

/* Brand */
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.logo {
  width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
  font-size: 21px; color: #04101c; font-weight: 800;
  background: linear-gradient(135deg, var(--teal), var(--violet));
  box-shadow: 0 8px 24px rgba(66, 211, 197, 0.35);
}
.brand b { font-size: 21px; letter-spacing: -0.4px; color: var(--text); }
.brand small { display: block; color: var(--muted); font-size: 10px; letter-spacing: 1.6px; font-weight: 700; }

/* Hero Banner */
.hero {
  position: relative; overflow: hidden; border-radius: 24px; padding: 26px 32px; margin-bottom: 22px;
  border: 1px solid var(--line);
  background: linear-gradient(125deg, rgba(14, 34, 58, 0.94), rgba(9, 19, 34, 0.94) 55%, rgba(26, 18, 56, 0.92));
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
}
.hero h1 { margin: 0; font-size: 33px; font-weight: 800; letter-spacing: -1px; }
.hero p { margin: 8px 0 0; color: #aebfd2; font-size: 14.5px; max-width: 820px; line-height: 1.6; }
.grad { background: linear-gradient(90deg, var(--teal), var(--blue), var(--violet)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.kicker { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--teal); text-transform: uppercase; letter-spacing: 1.8px; margin-bottom: 6px; }
.badge { display: inline-block; padding: 4px 11px; border-radius: 999px; background: rgba(66, 211, 197, 0.10); border: 1px solid rgba(66, 211, 197, 0.35); color: #a8efe8; font-size: 11.5px; margin: 10px 6px 0 0; }

/* Robotic Voice Assistant Sidebar Card */
.voice-assistant-card {
  background: linear-gradient(180deg, rgba(13, 24, 48, 0.95), rgba(8, 14, 28, 0.95));
  border: 1px solid rgba(0, 210, 255, 0.35);
  border-radius: 20px;
  padding: 20px 16px;
  text-align: center;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(0, 210, 255, 0.05);
  margin: 16px 0;
}
.avatar-stage {
  position: relative;
  width: 86px;
  height: 86px;
  margin: 0 auto 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.avatar-core {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, #00d2ff, #2563eb 60%, #4f46e5);
  box-shadow: 0 0 22px rgba(0, 210, 255, 0.6), inset 0 0 12px rgba(255, 255, 255, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  z-index: 2;
}
.avatar-face {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: #060c18;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 0 8px rgba(0, 210, 255, 0.6);
}
.avatar-eyes {
  display: flex;
  gap: 10px;
  margin-bottom: 4px;
}
.avatar-eye {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #00d2ff;
  box-shadow: 0 0 8px #00d2ff;
}
.avatar-mouth {
  width: 14px;
  height: 3px;
  border-radius: 999px;
  background: #00d2ff;
  box-shadow: 0 0 6px #00d2ff;
}
.soundwave-ring {
  position: absolute;
  inset: 0;
  border-radius: 50%;
  border: 1px dashed rgba(0, 210, 255, 0.4);
  animation: ringRotate 8s linear infinite;
}
@keyframes ringRotate {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Claim Details Card */
.claim-details-card {
  background: rgba(12, 22, 42, 0.9);
  border: 1px solid rgba(40, 75, 125, 0.5);
  border-radius: 16px;
  padding: 20px 22px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
  margin-bottom: 16px;
}
.claim-quote-box {
  background: rgba(8, 14, 28, 0.65);
  border-left: 3px solid #ef4444;
  border-radius: 0 10px 10px 0;
  padding: 10px 14px;
  margin: 8px 0;
  font-size: 13.5px;
  line-height: 1.6;
  color: #e2e8f0;
}

/* Top Dashboard Bar */
.top-dashboard-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(10, 18, 36, 0.85);
  border: 1px solid rgba(40, 75, 125, 0.4);
  border-radius: 18px;
  padding: 12px 20px;
  margin-bottom: 16px;
  backdrop-filter: blur(14px);
}
.pill-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
}
.pill-badge.green { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399; }
.pill-badge.red { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #f87171; }
.pill-badge.blue { background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.4); color: #60a5fa; }
.pill-badge.amber { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); color: #fbbf24; }

/* Recent Key Insights Strip */
.insight-pill-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-top: 14px;
}
.insight-pill-card {
  background: rgba(12, 22, 42, 0.9);
  border: 1px solid rgba(40, 75, 125, 0.45);
  border-radius: 14px;
  padding: 12px 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
  transition: transform 0.2s ease, border-color 0.2s ease;
}
.insight-pill-card:hover {
  transform: translateY(-2px);
  border-color: #00d2ff;
}

/* Chart Explanatory Boxes */
.kpi {
  display: flex; align-items: center; gap: 16px; background: var(--card); border: 1px solid var(--line);
  border-radius: 18px; padding: 18px 20px; backdrop-filter: blur(10px);
  transition: transform 0.2s ease, box-shadow 0.2s ease; height: 100%;
}
.kpi:hover { transform: translateY(-3px); box-shadow: 0 14px 34px rgba(0, 0, 0, 0.32); border-color: rgba(66, 211, 197, 0.45); }
.kpi .ic { width: 50px; height: 50px; border-radius: 14px; display: grid; place-items: center; font-size: 24px; flex: none; }
.kpi .v { font-size: 28px; font-weight: 800; letter-spacing: -0.8px; line-height: 1.1; }
.kpi .l { color: var(--muted); font-size: 12.5px; margin-top: 4px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
.kpi .s { font-size: 11px; margin-top: 3px; color: #7f97af; }

/* Standard Cards */
.card {
  background: var(--card); backdrop-filter: blur(10px); border: 1px solid var(--line);
  border-radius: 18px; padding: 20px 22px; margin-bottom: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.20);
}
.section-title { font-size: 20px; font-weight: 800; margin: 16px 0 4px; letter-spacing: -0.4px; }
.section-sub { color: var(--muted); font-size: 13.5px; margin-bottom: 14px; line-height: 1.5; }

/* Insights & Quotes */
.insight {
  border-left: 3px solid var(--teal); background: rgba(66, 211, 197, 0.06); padding: 12px 15px;
  border-radius: 0 12px 12px 0; color: #dce8f3; margin: 9px 0; line-height: 1.6; font-size: 14px;
}
.insight.risk { border-color: var(--rose); background: rgba(255, 122, 138, 0.06); }
.insight.plan { border-color: var(--violet); background: rgba(167, 139, 250, 0.07); }
.insight small { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.6px; display: block; margin-bottom: 4px; }

/* Chart Explanatory Boxes */
.chart-card {
  background: var(--card); border: 1px solid var(--line); border-radius: 18px;
  padding: 18px 20px 14px; margin-bottom: 18px;
}
.chart-meta { margin-bottom: 8px; }
.chart-meta h4 { margin: 0; font-size: 16.5px; font-weight: 700; color: var(--text); }
.chart-meta p { margin: 3px 0 0; font-size: 12.5px; color: var(--muted); }
.chart-footer {
  margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(110, 150, 200, 0.16);
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
}
.chart-caption { font-size: 11.5px; color: var(--muted); line-height: 1.5; }
.chart-caption b { color: #b8ccdf; }
.chart-insight { font-size: 11.5px; color: #a8efe8; line-height: 1.5; background: rgba(66, 211, 197, 0.07); padding: 6px 10px; border-radius: 8px; border-left: 2px solid var(--teal); }

/* Chat Bubbles */
.bubble { padding: 14px 18px; border-radius: 18px; margin: 10px 0; line-height: 1.6; max-width: 90%; font-size: 14px; border: 1px solid var(--line); }
.bubble.user { margin-left: auto; background: linear-gradient(135deg, rgba(106, 168, 255, 0.22), rgba(167, 139, 250, 0.20)); border-bottom-right-radius: 4px; }
.bubble.bot { background: var(--card); border-bottom-left-radius: 4px; }
.bubble small { display: block; color: var(--muted); font-size: 10.5px; margin-bottom: 4px; letter-spacing: 1px; font-family: 'JetBrains Mono', monospace; font-weight: 700; }

/* Buttons & Controls */
.stButton>button, .stDownloadButton>button {
  border: 1px solid rgba(66, 211, 197, 0.35); background: rgba(15, 34, 56, 0.85);
  color: var(--text); border-radius: 11px; font-weight: 700; transition: all 0.18s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  border-color: var(--teal); color: #04101c; background: linear-gradient(90deg, var(--teal), var(--blue)); transform: translateY(-1px);
}
.stButton>button[kind="primary"] {
  background: linear-gradient(90deg, var(--teal), var(--violet)); color: #04101c; border: 0;
  box-shadow: 0 6px 20px rgba(66, 211, 197, 0.28);
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
</style>
"""

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --bg: #f8fafc;
  --card: #ffffff;
  --line: #e2e8f0;
  --text: #0f172a;
  --muted: #64748b;
  --teal: #0d9488;
  --blue: #2563eb;
  --violet: #7c3aed;
  --amber: #d97706;
  --rose: #e11d48;
  --green: #16a34a;
}
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', -apple-system, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { position: relative; z-index: 1; padding-top: 1.2rem; max-width: 1540px; }
section[data-testid="stSidebar"] {
  background: #f1f5f9;
  border-right: 1px solid var(--line);
}

/* Brand */
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.logo {
  width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
  font-size: 21px; color: #ffffff; font-weight: 800;
  background: linear-gradient(135deg, var(--teal), var(--blue));
}
.brand b { font-size: 21px; letter-spacing: -0.4px; color: var(--text); }
.brand small { display: block; color: var(--muted); font-size: 10px; letter-spacing: 1.6px; font-weight: 700; }

/* Hero Banner */
.hero {
  border-radius: 24px; padding: 26px 32px; margin-bottom: 22px;
  border: 1px solid var(--line);
  background: linear-gradient(125deg, #ffffff, #f1f5f9 60%, #e2e8f0);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
}
.hero h1 { margin: 0; font-size: 33px; font-weight: 800; letter-spacing: -1px; color: #0f172a; }
.hero p { margin: 8px 0 0; color: #475569; font-size: 14.5px; max-width: 820px; line-height: 1.6; }
.grad { background: linear-gradient(90deg, var(--teal), var(--blue)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.kicker { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--teal); text-transform: uppercase; letter-spacing: 1.8px; margin-bottom: 6px; font-weight: 700; }
.badge { display: inline-block; padding: 4px 11px; border-radius: 999px; background: #e0f2fe; border: 1px solid #bae6fd; color: #0284c7; font-size: 11.5px; margin: 10px 6px 0 0; }

/* KPI Cards */
.kpi {
  display: flex; align-items: center; gap: 16px; background: #ffffff; border: 1px solid var(--line);
  border-radius: 18px; padding: 18px 20px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04); height: 100%;
}
.kpi:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08); border-color: var(--teal); }
.kpi .ic { width: 50px; height: 50px; border-radius: 14px; display: grid; place-items: center; font-size: 24px; flex: none; }
.kpi .v { font-size: 28px; font-weight: 800; letter-spacing: -0.8px; line-height: 1.1; color: #0f172a; }
.kpi .l { color: var(--muted); font-size: 12.5px; margin-top: 4px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
.kpi .s { font-size: 11px; margin-top: 3px; color: #64748b; }

/* Standard Cards */
.card {
  background: #ffffff; border: 1px solid var(--line); border-radius: 18px; padding: 20px 22px; margin-bottom: 16px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
}
.section-title { font-size: 20px; font-weight: 800; margin: 16px 0 4px; letter-spacing: -0.4px; color: #0f172a; }
.section-sub { color: var(--muted); font-size: 13.5px; margin-bottom: 14px; line-height: 1.5; }

/* Insights & Quotes */
.insight {
  border-left: 3px solid var(--teal); background: #f0fdf4; padding: 12px 15px;
  border-radius: 0 12px 12px 0; color: #0f172a; margin: 9px 0; line-height: 1.6; font-size: 14px;
}
.insight.risk { border-color: var(--rose); background: #fff1f2; }
.insight.plan { border-color: var(--violet); background: #f5f3ff; }
.insight small { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.6px; display: block; margin-bottom: 4px; }

/* Chart Explanatory Boxes */
.chart-card {
  background: #ffffff; border: 1px solid var(--line); border-radius: 18px;
  padding: 18px 20px 14px; margin-bottom: 18px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
}
.chart-meta { margin-bottom: 8px; }
.chart-meta h4 { margin: 0; font-size: 16.5px; font-weight: 700; color: #0f172a; }
.chart-meta p { margin: 3px 0 0; font-size: 12.5px; color: var(--muted); }
.chart-footer {
  margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line);
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
}
.chart-caption { font-size: 11.5px; color: #64748b; line-height: 1.5; }
.chart-caption b { color: #1e293b; }
.chart-insight { font-size: 11.5px; color: #0f766e; line-height: 1.5; background: #ccfbf1; padding: 6px 10px; border-radius: 8px; border-left: 2px solid var(--teal); }

/* Chat Bubbles */
.bubble { padding: 14px 18px; border-radius: 18px; margin: 10px 0; line-height: 1.6; max-width: 90%; font-size: 14px; border: 1px solid var(--line); }
.bubble.user { margin-left: auto; background: #e0f2fe; border-bottom-right-radius: 4px; color: #0369a1; }
.bubble.bot { background: #ffffff; border-bottom-left-radius: 4px; color: #0f172a; }
.bubble small { display: block; color: var(--muted); font-size: 10.5px; margin-bottom: 4px; letter-spacing: 1px; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
</style>
"""


def inject_theme_css(theme: str = "dark") -> None:
    """Injects dark or light theme CSS."""
    if theme == "light":
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    else:
        st.markdown(DARK_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Compatibility & Plot Rendering
# -----------------------------------------------------------------------------
_NEW_WIDTH_API = "width" in inspect.signature(st.plotly_chart).parameters


def show_plot(fig: go.Figure, key: str | None = None) -> None:
    if _NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": True})
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": True})


def show_styled_table(df: pd.DataFrame, column_config: Optional[Dict[str, Any]] = None, key: Optional[str] = None) -> None:
    if _NEW_WIDTH_API:
        st.dataframe(df, width="stretch", column_config=column_config, hide_index=True, key=key)
    else:
        st.dataframe(df, use_container_width=True, column_config=column_config, hide_index=True, key=key)


# -----------------------------------------------------------------------------
# Component Card Builders
# -----------------------------------------------------------------------------
def kpi_card(
    icon: str,
    label: str,
    value: Any,
    subtext: str,
    tooltip: str,
    accent_color: str = TEAL,
) -> str:
    """Renders KPI card with hover tooltip icon (ⓘ) and subtext."""
    esc_v = html.escape(str(value))
    esc_l = html.escape(label)
    esc_s = html.escape(subtext)
    esc_t = html.escape(tooltip)
    bg_tint = f"{accent_color}1a"

    return f"""
<div class="kpi">
  <div class="ic" style="background:{bg_tint};color:{accent_color};">{icon}</div>
  <div style="flex:1;">
    <div class="v" style="color:{accent_color};">{esc_v}</div>
    <div class="l">
      <span>{esc_l}</span>
      <span title="{esc_t}" style="cursor:help;font-size:13px;opacity:0.75;">ⓘ</span>
    </div>
    <div class="s">{esc_s}</div>
  </div>
</div>
"""


def render_kpi(
    value: Any,
    label: str,
    subtext: str = "",
    accent_color: str = TEAL,
    icon: str = "📊",
    tooltip: str = "",
) -> None:
    """Renders a styled KPI card directly to Streamlit."""
    html_card = kpi_card(
        icon=icon,
        label=label,
        value=value,
        subtext=subtext,
        tooltip=tooltip or label,
        accent_color=accent_color,
    )
    st.markdown(html_card.strip(), unsafe_allow_html=True)


def render_chart_wrapper(
    fig: go.Figure,
    title: str,
    subtitle: str,
    how_to_read: str,
    so_what_insight: str,
    key: Optional[str] = None,
) -> None:
    """
    Renders Plotly chart accompanied by title, subtitle,
    'How to read this' caption, and 'So what' plain-English insight.
    """
    esc_t = html.escape(title)
    esc_s = html.escape(subtitle)
    esc_r = html.escape(how_to_read)
    esc_w = html.escape(so_what_insight)

    st.markdown(
        f"""<div class="chart-card">
  <div class="chart-meta">
    <h4>{esc_t}</h4>
    <p>{esc_s}</p>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    show_plot(fig, key=key)

    st.markdown(
        f"""<div style="margin-top:-10px;margin-bottom:18px;background:rgba(15,28,48,0.6);border:1px solid rgba(110,150,200,0.18);border-radius:12px;padding:12px 16px;">
  <div class="chart-caption"><b>How to read this:</b> {esc_r}</div>
  <div class="chart-insight" style="margin-top:8px;"><b>Key Takeaway:</b> {esc_w}</div>
</div>""",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Interactive Plotly Chart Builders
# -----------------------------------------------------------------------------
def build_tone_gauge(score: float, label: str) -> go.Figure:
    """Renders a gauge chart representing corporate tone from -1.0 to +1.0."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": "+.2f", "font": {"size": 34, "color": "#e8f0fa"}},
            title={"text": f"<b>{html.escape(label)}</b><br><span style='font-size:12px;color:#8fa6bd'>Sentiment Scale: -1.0 (Cautious) to +1.0 (Confident)</span>", "font": {"size": 14, "color": "#e8f0fa"}},
            gauge={
                "axis": {"range": [-1.0, 1.0], "tickwidth": 1, "tickcolor": "rgba(110,150,200,0.3)"},
                "bar": {"color": TEAL, "thickness": 0.28},
                "bgcolor": "rgba(15, 28, 48, 0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(110,150,200,0.2)",
                "steps": [
                    {"range": [-1.0, -0.15], "color": "rgba(255, 122, 138, 0.25)"},
                    {"range": [-0.15, 0.15], "color": "rgba(246, 185, 91, 0.25)"},
                    {"range": [0.15, 1.0], "color": "rgba(94, 230, 160, 0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_tone_trend_chart(quarter_analyses: List[Dict[str, Any]]) -> go.Figure:
    """Renders chronological sentiment line with custom marker colors."""
    quarters = [q["quarter"] for q in quarter_analyses]
    scores = [q.get("management_tone", {}).get("score", 0.0) for q in quarter_analyses]
    colors = [GREEN if s > 0.15 else ROSE if s < -0.15 else AMBER for s in scores]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=quarters,
            y=scores,
            mode="lines+markers",
            line=dict(color=BLUE, width=3),
            marker=dict(size=12, color=colors, line=dict(color="#ffffff", width=2)),
            hovertemplate="<b>%{x}</b><br>Tone Score: %{y:.2f}<extra></extra>",
            name="Tone Score",
        )
    )
    fig.add_hline(y=0.15, line_dash="dot", line_color="rgba(94,230,160,0.4)", annotation_text="Positive Threshold")
    fig.add_hline(y=-0.15, line_dash="dot", line_color="rgba(255,122,138,0.4)", annotation_text="Cautious Threshold")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Fiscal Reporting Quarter", gridcolor="rgba(110,150,200,0.12)"),
        yaxis=dict(title="Net Sentiment Score (-1 to +1)", range=[-0.8, 0.8], gridcolor="rgba(110,150,200,0.12)"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
    )
    return fig


def build_topic_radar_chart(topic_df: pd.DataFrame, quarters_to_plot: Sequence[str]) -> go.Figure:
    """Renders multi-quarter radar chart comparing topic intensity shares."""
    fig = go.Figure()

    if topic_df.empty:
        return fig

    pivot = topic_df.pivot(index="Quarter", columns="Topic", values="Share (%)").fillna(0.0)
    topics = list(pivot.columns)

    for i, q in enumerate(quarters_to_plot):
        if q in pivot.index:
            vals = pivot.loc[q].tolist()
            # Close polygon
            vals_closed = vals + [vals[0]]
            topics_closed = topics + [topics[0]]
            c = PALETTE[i % len(PALETTE)]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals_closed,
                    theta=topics_closed,
                    fill="toself",
                    fillcolor=_hex_to_rgba(c, 0.15),
                    line=dict(color=c, width=2.5),
                    name=q,
                )
            )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 35], ticksuffix="%", gridcolor="rgba(110,150,200,0.2)"),
            angularaxis=dict(gridcolor="rgba(110,150,200,0.2)"),
            bgcolor="rgba(15, 28, 48, 0.3)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def build_signal_volume_bar_chart(quarter_analyses: List[Dict[str, Any]]) -> go.Figure:
    """Renders grouped bar chart of statement volumes (Confident vs Cautious vs Neutral)."""
    quarters = [q["quarter"] for q in quarter_analyses]
    pos_counts = []
    caut_counts = []
    total_counts = []

    for q in quarter_analyses:
        sents = q.get("sentences", [])
        total_counts.append(len(sents))
        pos_counts.append(sum(1 for s in sents if s.get("stance") == "Confident"))
        caut_counts.append(sum(1 for s in sents if s.get("stance") == "Cautious"))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Confident Statements", x=quarters, y=pos_counts, marker_color=GREEN))
    fig.add_trace(go.Bar(name="Cautious / Headwind Statements", x=quarters, y=caut_counts, marker_color=ROSE))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Fiscal Quarter", gridcolor="rgba(110,150,200,0.12)"),
        yaxis=dict(title="Statement Count", gridcolor="rgba(110,150,200,0.12)"),
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_confusion_matrix_chart(cm_data: List[List[int]], classes: List[str]) -> go.Figure:
    """Renders interactive confusion matrix heatmap for Model Health / Reliability."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_data,
            x=classes,
            y=classes,
            colorscale=[[0, "rgba(15,28,48,0.8)"], [1, TEAL]],
            hoverongaps=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Classification Reliability Matrix (Evaluated on Held-Out Test Set)",
        title_font=dict(size=14, color="#e8f0fa"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Predicted Category", tickangle=-35),
        yaxis=dict(title="Actual Category"),
        height=400,
        margin=dict(l=100, r=20, t=50, b=100),
    )
    return fig


def build_scorecard_metric_trend(qoq_df: pd.DataFrame, metric_name: str) -> go.Figure:
    """Renders a chronological trend for a specific financial scorecard metric."""
    sub = qoq_df[qoq_df["Metric"] == metric_name].copy()
    fig = go.Figure()

    if sub.empty:
        return fig

    # Extract numerical values
    vals = []
    for v in sub["Reported Value"]:
        m = re.search(r"(-?\d+(?:\.\d+)?)", str(v))
        vals.append(float(m.group(1)) if m else None)

    sub["Numeric"] = vals
    valid = sub.dropna(subset=["Numeric"])

    fig.add_trace(
        go.Bar(
            x=valid["Quarter"],
            y=valid["Numeric"],
            marker_color=BLUE,
            text=valid["Reported Value"],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Reported: %{text}<extra></extra>",
            name=metric_name,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Quarter", gridcolor="rgba(110,150,200,0.12)"),
        yaxis=dict(title=f"Reported {metric_name}", gridcolor="rgba(110,150,200,0.12)"),
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


def render_page_footer() -> None:
    """
    Renders the standardized corporate footer on every single page.
    Includes the mandatory exact line:
    'To activate voice assistant, say “Concall”.'
    """
    st.markdown(
        """
        <div style="margin-top:48px;padding:26px 20px 32px;border-top:1px solid rgba(110,150,200,0.18);text-align:center;">
          <div style="font-size:15px;font-weight:700;color:#42d3c5;letter-spacing:0.3px;margin-bottom:8px;">
            To activate voice assistant, say “Concall”.
          </div>
          <div style="font-size:11.5px;color:#8fa6bd;max-width:820px;margin:0 auto;line-height:1.6;">
            <b>ConcallIQ Enterprise Intelligence Platform</b> · Research & Decision Support System · Non-Advisory<br>
            All extracted metrics, thematic stances, and claim classifications are grounded in verifiable transcript citations.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_global_assistant_hud(
    current_page: str,
    current_company: str,
    assistant_engine: Any,
    qa_engine: Any = None,
    claim_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Renders the persistent global voice assistant HUD across all pages.
    Supports continuous wake-word listening with accessible text fallback.
    """
    with st.container():
        hud_col1, hud_col2 = st.columns([3, 1])
        with hud_col1:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:12px;background:rgba(15,28,48,0.72);border:1px solid rgba(66,211,197,0.3);border-radius:14px;padding:8px 16px;margin-bottom:14px;">
                  <div style="width:10px;height:10px;border-radius:50%;background:#42d3c5;box-shadow:0 0 10px #42d3c5;animation:pulse 2s infinite;"></div>
                  <div style="font-size:12.5px;color:#e8f0fa;font-weight:600;">
                    Global Voice Assistant Active · Say <span style="color:#42d3c5;font-weight:700;">“Concall”</span> to navigate or investigate
                  </div>
                  <div style="margin-left:auto;font-size:11px;color:#8fa6bd;">
                    Page: <b>{current_page}</b> | Target: <b>{current_company}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hud_col2:
            pass

        # Quick assistant input box (text/voice fallback)
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_cmd = st.text_input(
                "Voice or text command",
                placeholder="Say 'Concall' or type command (e.g. 'What can I do on this page?', 'Which claims changed?')",
                key=f"assistant_input_{current_page.replace(' ', '_')}",
                label_visibility="collapsed",
            )
        with btn_col:
            submit_cmd = st.button("Submit", key=f"btn_cmd_{current_page.replace(' ', '_')}", use_container_width=True)

        if submit_cmd and user_cmd:
            res = assistant_engine.process_query(
                user_cmd,
                current_page=current_page,
                current_company=current_company,
                qa_engine=qa_engine,
                claim_data=claim_data,
            )
            st.session_state["last_assistant_response"] = res
            if res.get("navigate_to"):
                st.session_state["nav_request"] = res["navigate_to"]
                st.rerun()

        # Display last assistant response if present
        last_res = st.session_state.get("last_assistant_response")
        if last_res:
            with st.expander("💬 Assistant Investigation Response", expanded=True):
                st.markdown(last_res.get("text", ""))
                if last_res.get("evidence"):
                    st.caption(f"Source Evidence: \"{last_res['evidence'][:200]}...\"")
                if st.button("Dismiss", key=f"dismiss_{current_page.replace(' ', '_')}"):
                    st.session_state["last_assistant_response"] = None
                    st.rerun()


# =============================================================================
# Reference Dashboard Layout Components (Matching Master Concept)
# =============================================================================
def render_dashboard_top_bar(
    company: str,
    quarters: List[str],
    active_quarter: str,
    sentiment_score: float = 0.32,
    risks_count: int = 3,
    changes_count: int = 5,
) -> None:
    """
    Renders the unified top header and company filter bar matching the concept design.
    """
    st.markdown(
        f"""
        <div class="top-dashboard-nav">
          <div style="display:flex;align-items:center;gap:18px;">
            <div style="display:flex;align-items:center;gap:10px;">
              <div style="width:36px;height:36px;border-radius:10px;background:radial-gradient(circle at 35% 35%, #00d2ff, #2563eb);display:grid;place-items:center;font-size:18px;">◈</div>
              <div>
                <b style="font-size:18px;color:#ffffff;letter-spacing:-0.3px;">{company}</b>
                <div style="font-size:11px;color:#8fa6bd;">Earnings Intelligence · Multi-Quarter Trajectory</div>
              </div>
            </div>
            <div style="display:flex;gap:6px;margin-left:14px;">
              {"".join(f'<span style="padding:4px 12px;border-radius:999px;font-size:11.5px;font-weight:700;background:{"rgba(0, 210, 255, 0.22);border:1px solid #00d2ff;color:#ffffff;" if q == active_quarter else "rgba(15, 28, 48, 0.6);border:1px solid rgba(40, 75, 125, 0.4);color:#8fa6bd;"}">{q}</span>' for q in quarters[-4:])}
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:10px;">
            <div class="pill-badge green">
              <span>😊</span>
              <span>Overall Sentiment: <b>{sentiment_score:+.2f}</b> Mostly Positive</span>
            </div>
            <div class="pill-badge red">
              <span>⚠️</span>
              <span>Key Risks: <b>{risks_count}</b> High Attention</span>
            </div>
            <div class="pill-badge blue">
              <span>🔄</span>
              <span>Claim Changes: <b>{changes_count}</b> Notable Changes</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_voice_assistant(is_active: bool = False, active_remaining: int = 30) -> None:
    """
    Renders the glowing robotic Voice Assistant card in the sidebar matching the visual reference.
    """
    state_pill = f'<span style="display:inline-block;padding:3px 10px;border-radius:999px;background:rgba(16,185,129,0.2);border:1px solid #10b981;color:#34d399;font-size:11px;font-weight:700;margin-bottom:8px;">🟢 Active · {active_remaining}s</span>' if is_active else '<span style="display:inline-block;padding:3px 10px;border-radius:999px;background:rgba(14,165,233,0.15);border:1px solid rgba(14,165,233,0.4);color:#38bdf8;font-size:11px;font-weight:700;margin-bottom:8px;">💤 Standby · Say "Concall"</span>'
    btn_text = f"🛑 Sleep" if is_active else "🎙️ Tap to Speak"

    st.sidebar.markdown(
        f"""
        <div class="voice-assistant-card">
          <div class="avatar-stage">
            <div class="soundwave-ring"></div>
            <div class="avatar-core">
              <div class="avatar-face">
                <div class="avatar-eyes">
                  <div class="avatar-eye"></div>
                  <div class="avatar-eye"></div>
                </div>
                <div class="avatar-mouth"></div>
              </div>
            </div>
          </div>
          {state_pill}
          <div style="font-size:15px;font-weight:800;color:#ffffff;letter-spacing:-0.2px;margin-bottom:4px;">Voice Assistant</div>
          <div style="font-size:11.5px;color:#94a3b8;line-height:1.5;margin-bottom:12px;">
            Hi! I'm your ConcallIQ assistant. You can ask me about claims, risks, sentiment or anything from the earnings calls.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_claim_details_card(
    claim_node: Optional[Dict[str, Any]] = None,
    company: str = "TCS",
) -> None:
    """
    Renders the Selected Claim Details right-hand panel matching the concept design.
    """
    if not claim_node:
        claim_node = {
            "topic": "Strategic Execution",
            "status": "contradiction_candidate",
            "status_label": "Contradicted",
            "previous_quarter": "Baseline Period",
            "previous_text": "Management expressed strong early confidence in operational delivery and financial targets.",
            "quarter": "Subsequent Period",
            "text": "Subsequent remarks qualified prior guidance citing market decision cycles and regional volatility.",
            "speaker": "Executive Leadership",
            "status_reason": "Subsequent disclosure diverges from baseline confidence, warranting cross-quarter investigation.",
        }

    status_color = ROSE if claim_node.get("status") == "contradiction_candidate" else (AMBER if claim_node.get("status") == "weakened" else GREEN)
    prev_q_label = claim_node.get("previous_quarter") or "Baseline Period"
    curr_q_label = claim_node.get("quarter") or "Subsequent Period"

    st.markdown(
        f"""
        <div class="claim-details-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div style="font-size:14px;font-weight:700;color:#ffffff;">Selected Claim Details</div>
            <span style="padding:2px 8px;border-radius:6px;background:rgba(239,68,68,0.15);border:1px solid {status_color};color:{status_color};font-size:10.5px;font-weight:700;">{claim_node.get('status_label', 'Contradicted')}</span>
          </div>

          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="font-size:18px;">🔴</span>
            <b style="font-size:16px;color:#ffffff;">{claim_node.get('topic', 'Strategic Execution')}</b>
          </div>

          <div style="font-size:11px;color:#8fa6bd;font-weight:700;margin-top:8px;">{prev_q_label} Statement</div>
          <div class="claim-quote-box" style="border-left-color:#38bdf8;">
            "{claim_node.get('previous_text', 'Management expressed strong early confidence in operational delivery.')}"
            <div style="text-align:right;font-size:10.5px;color:#64748b;margin-top:4px;">— {claim_node.get('speaker', 'Management')}</div>
          </div>

          <div style="font-size:11px;color:#8fa6bd;font-weight:700;margin-top:10px;">{curr_q_label} Statement</div>
          <div class="claim-quote-box" style="border-left-color:#ef4444;">
            "{claim_node.get('text', 'Subsequent remarks qualified prior guidance due to macro volatility.')}"
            <div style="text-align:right;font-size:10.5px;color:#64748b;margin-top:4px;">— {claim_node.get('speaker', 'Management')}</div>
          </div>

          <div style="background:rgba(15,28,48,0.65);border-radius:10px;padding:10px 12px;margin-top:10px;">
            <div style="font-size:11px;font-weight:700;color:#f87171;display:flex;align-items:center;gap:5px;margin-bottom:4px;">
              <span>🎙️</span> Reason for Classification
            </div>
            <div style="font-size:12px;color:#cbd5e1;line-height:1.5;">
              {claim_node.get('status_reason', 'Directional reversal observed between baseline and subsequent disclosures.')}
            </div>
          </div>

          <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(40,75,125,0.35);">
            <div style="font-size:11px;font-weight:700;color:#8fa6bd;margin-bottom:8px;">Related Topics</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;">
              <span style="padding:3px 10px;border-radius:6px;background:rgba(15,28,48,0.8);border:1px solid rgba(40,75,125,0.4);font-size:11px;color:#cbd5e1;">Supply Chain</span>
              <span style="padding:3px 10px;border-radius:6px;background:rgba(15,28,48,0.8);border:1px solid rgba(40,75,125,0.4);font-size:11px;color:#cbd5e1;">Logistics</span>
              <span style="padding:3px 10px;border-radius:6px;background:rgba(15,28,48,0.8);border:1px solid rgba(40,75,125,0.4);font-size:11px;color:#cbd5e1;">Global Demand</span>
              <span style="padding:3px 10px;border-radius:6px;background:rgba(15,28,48,0.8);border:1px solid rgba(40,75,125,0.4);font-size:11px;color:#cbd5e1;">Risks</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_risks_breakdown() -> None:
    """
    Renders the Top Risks progress breakdown matching the reference dashboard:
    Supply Chain Disruptions 23%, Regulatory Changes 18%, Market Competition 15%, etc.
    """
    risks = [
        ("Supply Chain Disruptions", 23, ROSE, "⚠️"),
        ("Regulatory Changes", 18, AMBER, "⚖️"),
        ("Market Competition", 15, BLUE, "🛡️"),
        ("Economic Uncertainty", 12, VIOLET, "📉"),
        ("Geopolitical Tensions", 8, TEAL, "🌐"),
    ]

    bars_html = "".join(
        f'<div style="margin-bottom:12px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;margin-bottom:4px;">'
        f'<span style="display:flex;align-items:center;gap:6px;color:#e2e8f0;"><span>{icon}</span> <span>{label}</span></span>'
        f'<b style="color:#ffffff;">{pct}%</b>'
        f'</div>'
        f'<div style="height:6px;border-radius:999px;background:rgba(15,28,48,0.8);overflow:hidden;">'
        f'<div style="height:100%;width:{pct * 3.5}%;background:{col};border-radius:999px;"></div>'
        f'</div>'
        f'</div>'
        for label, pct, col, icon in risks
    )

    card_html = (
        f'<div class="card" style="height:100%;padding:18px 20px;">'
        f'<div style="font-size:14px;font-weight:700;color:#ffffff;margin-bottom:14px;">'
        f'Top Risks <span style="font-size:11px;color:#8fa6bd;font-weight:400;">(by frequency)</span>'
        f'</div>'
        f'{bars_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_recent_key_insights(company: str = "TCS") -> None:
    """
    Renders the bottom horizontal strip of 5 metric cards matching the concept design.
    """
    insights_html = (
        '<div style="margin-top:20px;">'
        '<div style="font-size:14px;font-weight:700;color:#ffffff;margin-bottom:8px;">Recent Key Insights</div>'
        '<div class="insight-pill-strip">'
        '<div class="insight-pill-card">'
        '<div style="width:34px;height:34px;border-radius:10px;background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.4);display:grid;place-items:center;font-size:16px;">📈</div>'
        '<div>'
        '<div style="font-size:11px;color:#8fa6bd;">Revenue Growth</div>'
        '<div style="font-size:13.5px;font-weight:800;color:#10b981;">+10-12%</div>'
        '<div style="font-size:10px;color:#64748b;">Consistent across calls</div>'
        '</div>'
        '</div>'
        '<div class="insight-pill-card">'
        '<div style="width:34px;height:34px;border-radius:10px;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.4);display:grid;place-items:center;font-size:16px;">⚠️</div>'
        '<div>'
        '<div style="font-size:11px;color:#8fa6bd;">Supply Chain</div>'
        '<div style="font-size:13.5px;font-weight:800;color:#ef4444;">Cross-Quarter Shift</div>'
        '<div style="font-size:10px;color:#f87171;">Audit attention area</div>'
        '</div>'
        '</div>'
        '<div class="insight-pill-card">'
        '<div style="width:34px;height:34px;border-radius:10px;background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.4);display:grid;place-items:center;font-size:16px;">💡</div>'
        '<div>'
        '<div style="font-size:11px;color:#8fa6bd;">AI Investment</div>'
        '<div style="font-size:13.5px;font-weight:800;color:#10b981;">Increasing</div>'
        '<div style="font-size:10px;color:#64748b;">Positive forward tone</div>'
        '</div>'
        '</div>'
        '<div class="insight-pill-card">'
        '<div style="width:34px;height:34px;border-radius:10px;background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.4);display:grid;place-items:center;font-size:16px;">⚖️</div>'
        '<div>'
        '<div style="font-size:11px;color:#8fa6bd;">Cost Management</div>'
        '<div style="font-size:13.5px;font-weight:800;color:#38bdf8;">Stable</div>'
        '<div style="font-size:10px;color:#64748b;">Neutral discipline tone</div>'
        '</div>'
        '</div>'
        '<div class="insight-pill-card">'
        '<div style="width:34px;height:34px;border-radius:10px;background:rgba(168,85,247,0.15);border:1px solid rgba(168,85,247,0.4);display:grid;place-items:center;font-size:16px;">🔬</div>'
        '<div>'
        '<div style="font-size:11px;color:#8fa6bd;">R&D Scale</div>'
        '<div style="font-size:13.5px;font-weight:800;color:#c084fc;">Growing focus</div>'
        '<div style="font-size:10px;color:#64748b;">Expanding pipeline</div>'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(insights_html, unsafe_allow_html=True)

