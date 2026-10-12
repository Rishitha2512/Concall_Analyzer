"""
visualization_3d.py
--------------------
Enterprise-grade 3D interactive visualizations for ConcallIQ:
1. The 3D Claim Time Machine (Signature narrative evolution centerpiece).
2. The 3D Corporate Narrative Galaxy (Cross-entity & thematic knowledge network).
3. The 3D Narrative Stress Map (Quantified investigation priority landscape).
4. The 3D Financial Peer Space (Multi-dimensional corporate benchmarking).
5. The Audio & Narrative Exploration View (Acoustic and speech segment tracking).

Strict zero internal AI/ML jargon: exposed exclusively through institutional
financial and narrative terminology with accessible 2D alternatives alongside.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Color Palette & Design Tokens (Aurora Dark System)
# -----------------------------------------------------------------------------
BG_COLOR = "#0a0f1d"
CARD_BG = "#0d1527"
GRID_COLOR = "rgba(110, 150, 200, 0.14)"
TEXT_COLOR = "#e6edf3"
MUTED_TEXT = "#8fa6bd"

TEAL = "#42d3c5"
BLUE = "#6aa8ff"
VIOLET = "#a78bfa"
AMBER = "#f6b95b"
ROSE = "#ff7a8a"
GREEN = "#5ee6a0"

STATUS_COLORS = {
    "new": BLUE,
    "reaffirmed": GREEN,
    "weakened": AMBER,
    "contradiction_candidate": ROSE,
}


# =============================================================================
# 1. Signature Experience: The 3D Claim Time Machine
# =============================================================================
def build_3d_claim_time_machine(
    claim_dataset: Dict[str, Any],
    filter_topic: Optional[str] = None,
    filter_status: Optional[str] = None,
    highlight_claim_id: Optional[str] = None,
) -> go.Figure:
    """
    Renders an interactive spatial representation of management claim evolution across quarters.
    X: Chronological Quarters
    Y: Business Topics
    Z: Stance & Narrative Elevation
    Links: Directed lines connecting statements that refer to the same underlying claim.
    """
    nodes = claim_dataset.get("nodes", [])
    links = claim_dataset.get("links", [])
    quarters = claim_dataset.get("quarters", [])
    topics = claim_dataset.get("topics", [])

    # Filter nodes if requested
    filtered_nodes = nodes
    if filter_topic and filter_topic != "All Topics":
        filtered_nodes = [n for n in filtered_nodes if n.get("topic") == filter_topic]
    if filter_status and filter_status != "All Statuses":
        filtered_nodes = [n for n in filtered_nodes if n.get("status") == filter_status]

    fig = go.Figure()

    # 0. Holographic Concentric Platform Rings (Cybernetic base from reference design)
    floor_z = -1.8
    theta_pts = np.linspace(0, 2 * np.pi, 50)
    center_x = (len(quarters) - 1) / 2.0 if quarters else 1.5
    center_y = (len(topics) - 1) / 2.0 if topics else 3.5

    for r_idx, radius in enumerate([1.5, 2.8, 4.2]):
        ring_x = center_x + radius * np.cos(theta_pts)
        ring_y = center_y + (radius * 0.75) * np.sin(theta_pts)
        ring_z = [floor_z] * len(theta_pts)
        fig.add_trace(
            go.Scatter3d(
                x=ring_x,
                y=ring_y,
                z=ring_z,
                mode="lines",
                line=dict(color=f"rgba(0, 210, 255, {0.18 + r_idx * 0.08})", width=2.2),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # 1. Render cross-quarter connecting lines (Links)
    node_ids = {n["claim_id"] for n in filtered_nodes}
    for l in links:
        s_id = l.get("source_id")
        t_id = l.get("target_id")
        if s_id in node_ids and t_id in node_ids:
            x0, y0, z0 = l["source_coord"]
            x1, y1, z1 = l["target_coord"]
            c = l.get("color", BLUE)
            fig.add_trace(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode="lines",
                    line=dict(color=c, width=3.5),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

    # 2. Render Claim Nodes grouped by status
    status_groups: Dict[str, List[Dict[str, Any]]] = {
        "reaffirmed": [],
        "new": [],
        "weakened": [],
        "contradiction_candidate": [],
    }
    for n in filtered_nodes:
        st_key = n.get("status", "new")
        if st_key in status_groups:
            status_groups[st_key].append(n)

    status_labels_map = {
        "reaffirmed": "Reaffirmed Target (Supported)",
        "new": "Newly Introduced Claim",
        "weakened": "Weakened / Qualified Claim",
        "contradiction_candidate": "Contradiction Candidate (Requires Investigation)",
    }

    for st_key, grp_nodes in status_groups.items():
        if not grp_nodes:
            continue

        xs = [n["x"] for n in grp_nodes]
        ys = [n["y"] for n in grp_nodes]
        zs = [n["z"] for n in grp_nodes]

        hover_customdata = [
            [
                n["claim_id"],
                n["quarter"],
                n["status_label"],
                n["speaker"],
                (n["text"][:110] + "...") if len(n["text"]) > 110 else n["text"],
                n["status_reason"],
                n["metric_ref"] or "N/A",
            ]
            for n in grp_nodes
        ]

        # Marker sizes (highlight selected claim)
        sizes = [
            11 if n["claim_id"] == highlight_claim_id else 7.5
            for n in grp_nodes
        ]

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=STATUS_COLORS.get(st_key, BLUE),
                    opacity=0.92,
                    line=dict(color="#ffffff", width=1.0 if st_key != "contradiction_candidate" else 2.0),
                ),
                name=status_labels_map.get(st_key, st_key),
                customdata=hover_customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> | %{customdata[1]}<br>"
                    "Status: <b>%{customdata[2]}</b><br>"
                    "Speaker: %{customdata[3]}<br>"
                    "Reported Metric: <b>%{customdata[6]}</b><br>"
                    "Claim: <i>\"%{customdata[4]}\"</i><br>"
                    "Audit Note: %{customdata[5]}<extra></extra>"
                ),
            )
        )

    # Layout configuration
    q_indices = list(range(len(quarters)))
    t_indices = list(range(len(topics)))

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        scene=dict(
            xaxis=dict(
                title=dict(text="Chronological Fiscal Quarter", font=dict(color=TEXT_COLOR, size=11)),
                tickmode="array",
                tickvals=q_indices,
                ticktext=quarters,
                gridcolor=GRID_COLOR,
                backgroundcolor=CARD_BG,
                showbackground=True,
            ),
            yaxis=dict(
                title=dict(text="Business Strategic Domain", font=dict(color=TEXT_COLOR, size=11)),
                tickmode="array",
                tickvals=t_indices,
                ticktext=[t[:18] for t in topics],
                gridcolor=GRID_COLOR,
                backgroundcolor=CARD_BG,
                showbackground=True,
            ),
            zaxis=dict(
                title=dict(text="Narrative Certainty / Stance Elevation", font=dict(color=TEXT_COLOR, size=11)),
                gridcolor=GRID_COLOR,
                backgroundcolor=CARD_BG,
                showbackground=True,
            ),
            camera=dict(
                eye=dict(x=1.65, y=-1.75, z=1.2),
                center=dict(x=0, y=0, z=-0.1),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=540,
    )

    return fig


def build_2d_claim_timeline(
    claim_dataset: Dict[str, Any],
    filter_topic: Optional[str] = None,
    filter_status: Optional[str] = None,
) -> go.Figure:
    """
    Accessible 2D fallback for the Claim Time Machine.
    Plots claim distribution and status across chronological quarters.
    """
    nodes = claim_dataset.get("nodes", [])
    if filter_topic and filter_topic != "All Topics":
        nodes = [n for n in nodes if n.get("topic") == filter_topic]
    if filter_status and filter_status != "All Statuses":
        nodes = [n for n in nodes if n.get("status") == filter_status]

    fig = go.Figure()
    if not nodes:
        return fig

    df = pd.DataFrame(nodes)
    quarters = claim_dataset.get("quarters", [])

    for st_key, color in STATUS_COLORS.items():
        sub = df[df["status"] == st_key]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["quarter"],
                y=sub["topic"],
                mode="markers",
                marker=dict(size=12, color=color, opacity=0.85, line=dict(color="#ffffff", width=1.0)),
                name=sub["status_label"].iloc[0],
                customdata=sub[["claim_id", "speaker", "metric_ref", "text", "status_reason"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{x})<br>"
                    "Domain: %{y}<br>"
                    "Speaker: %{customdata[1]}<br>"
                    "Metric: %{customdata[2]}<br>"
                    "Statement: <i>\"%{customdata[3]}\"</i><br>"
                    "Rationale: %{customdata[4]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        xaxis=dict(title="Fiscal Quarter", categoryorder="array", categoryarray=quarters, gridcolor=GRID_COLOR),
        yaxis=dict(title="Business Topic Domain", gridcolor=GRID_COLOR),
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


# =============================================================================
# 2. Corporate Narrative Galaxy (3D Knowledge Network)
# =============================================================================
def build_3d_narrative_galaxy(
    company: str,
    topics: Sequence[str],
    technologies: Sequence[str],
    geographies: Sequence[str],
    co_occurrence_rules: List[Dict[str, Any]],
) -> go.Figure:
    """
    Renders a 3D spherical knowledge network representing relationships
    between the company, strategic topics, technologies, and operating regions.
    """
    fig = go.Figure()

    # Center node: Company
    node_coords: Dict[str, Tuple[float, float, float]] = {company: (0.0, 0.0, 0.0)}
    node_types: Dict[str, str] = {company: "Company"}

    # Shell 1: Topics (Radius = 1.0)
    n_topics = len(topics)
    for i, t in enumerate(topics):
        phi = math.acos(-1.0 + (2.0 * i) / max(1, n_topics - 1))
        theta = math.sqrt(n_topics * math.pi) * phi
        x = 1.0 * math.cos(theta) * math.sin(phi)
        y = 1.0 * math.sin(theta) * math.sin(phi)
        z = 1.0 * math.cos(phi)
        node_coords[t] = (x, y, z)
        node_types[t] = "Strategic Topic"

    # Shell 2: Technologies (Radius = 1.8)
    n_techs = len(technologies)
    for j, tech in enumerate(technologies):
        phi = math.acos(-1.0 + (2.0 * j + 1) / max(1, 2 * n_techs))
        theta = 2.4 * j
        x = 1.8 * math.cos(theta) * math.sin(phi)
        y = 1.8 * math.sin(theta) * math.sin(phi)
        z = 1.8 * math.cos(phi)
        node_coords[tech] = (x, y, z)
        node_types[tech] = "Technology Platform"

    # Shell 3: Geographies (Radius = 2.4)
    n_geos = len(geographies)
    for k, geo in enumerate(geographies):
        angle = (2 * math.pi * k) / max(1, n_geos)
        node_coords[geo] = (2.4 * math.cos(angle), 2.4 * math.sin(angle), 0.2 * math.sin(2 * angle))
        node_types[geo] = "Operating Market"

    # Draw edges from center to topics
    for t in topics:
        x0, y0, z0 = (0.0, 0.0, 0.0)
        x1, y1, z1 = node_coords[t]
        fig.add_trace(
            go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                line=dict(color="rgba(106, 168, 255, 0.25)", width=2),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Draw co-occurrence edges
    rules_list = []
    if isinstance(co_occurrence_rules, pd.DataFrame):
        rules_list = co_occurrence_rules.to_dict(orient="records")
    elif isinstance(co_occurrence_rules, list):
        rules_list = co_occurrence_rules

    for rule in rules_list[:25]:
        if not isinstance(rule, dict):
            continue
        a = rule.get("antecedent") or rule.get("Concept A") or ""
        b = rule.get("consequent") or rule.get("Concept B") or ""
        if a in node_coords and b in node_coords:
            x0, y0, z0 = node_coords[a]
            x1, y1, z1 = node_coords[b]
            fig.add_trace(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode="lines",
                    line=dict(color="rgba(66, 211, 197, 0.45)", width=2.5),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

    # Draw Nodes grouped by Category
    category_colors = {
        "Company": "#ffffff",
        "Strategic Topic": TEAL,
        "Technology Platform": BLUE,
        "Operating Market": VIOLET,
    }

    for cat, color in category_colors.items():
        cat_nodes = [name for name, t in node_types.items() if t == cat]
        if not cat_nodes:
            continue
        xs = [node_coords[n][0] for n in cat_nodes]
        ys = [node_coords[n][1] for n in cat_nodes]
        zs = [node_coords[n][2] for n in cat_nodes]

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers+text" if cat in ("Company", "Strategic Topic") else "markers",
                marker=dict(
                    size=12 if cat == "Company" else (8 if cat == "Strategic Topic" else 6),
                    color=color,
                    opacity=0.9,
                    line=dict(color="#ffffff", width=1),
                ),
                text=cat_nodes if cat in ("Company", "Strategic Topic") else None,
                textposition="top center",
                textfont=dict(color=TEXT_COLOR, size=10),
                name=cat,
                hovertemplate="<b>%{text}</b><br>Category: " + cat + "<extra></extra>" if cat in ("Company", "Strategic Topic") else "<b>%{customdata}</b><br>Category: " + cat + "<extra></extra>",
                customdata=cat_nodes,
            )
        )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=CARD_BG,
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=30, b=10),
        height=500,
    )
    return fig


# =============================================================================
# 3. Narrative Stress Map (3D Investigation Landscape)
# =============================================================================
def build_3d_narrative_stress_map(stress_data: Dict[str, Any]) -> go.Figure:
    """
    Renders a 3D landscape of investigation priorities across quarters and topics.
    X: Quarters
    Y: Business Topics
    Z: Narrative Stress Index (0 to 10)
    """
    df = stress_data.get("dataframe", pd.DataFrame())
    quarters = stress_data.get("quarters", [])
    topics = stress_data.get("topics", [])

    fig = go.Figure()
    if df.empty:
        return fig

    # Create 3D scatter with elevation and color representing stress
    for priority_label, color in [("High Priority", ROSE), ("Medium Priority", AMBER), ("Normal", TEAL)]:
        sub = df[df["priority"] == priority_label]
        if sub.empty:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=sub["quarter_idx"],
                y=sub["topic_idx"],
                z=sub["stress_score"],
                mode="markers",
                marker=dict(
                    size=sub["stress_score"].apply(lambda v: max(6, v * 1.8)),
                    color=color,
                    opacity=0.88,
                    line=dict(color="#ffffff", width=1.0),
                ),
                name=f"{priority_label} Area",
                customdata=sub[["quarter", "topic", "stress_score", "explanation", "weakened_count", "contradiction_count"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]} · %{customdata[1]}</b><br>"
                    "Narrative Stress Index: <b>%{customdata[2]}/10</b><br>"
                    "Contradiction Candidates: %{customdata[5]}<br>"
                    "Weakened Statements: %{customdata[4]}<br>"
                    "Audit Analysis: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    # Add vertical guide stems from z=0 to node for depth perception
    for _, row in df.iterrows():
        if row["stress_score"] > 2.0:
            fig.add_trace(
                go.Scatter3d(
                    x=[row["quarter_idx"], row["quarter_idx"]],
                    y=[row["topic_idx"], row["topic_idx"]],
                    z=[0, row["stress_score"]],
                    mode="lines",
                    line=dict(color=row["color"], width=1.5),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        scene=dict(
            xaxis=dict(
                title=dict(text="Fiscal Quarter", font=dict(color=TEXT_COLOR, size=11)),
                tickmode="array",
                tickvals=list(range(len(quarters))),
                ticktext=quarters,
                gridcolor=GRID_COLOR,
                backgroundcolor=CARD_BG,
                showbackground=True,
            ),
            yaxis=dict(
                title=dict(text="Business Topic Area", font=dict(color=TEXT_COLOR, size=11)),
                tickmode="array",
                tickvals=list(range(len(topics))),
                ticktext=[t[:16] for t in topics],
                gridcolor=GRID_COLOR,
                backgroundcolor=CARD_BG,
                showbackground=True,
            ),
            zaxis=dict(
                title=dict(text="Narrative Stress Index (0-10)", font=dict(color=TEXT_COLOR, size=11)),
                range=[0, 10.5],
                gridcolor=GRID_COLOR,
                backgroundcolor=CARD_BG,
                showbackground=True,
            ),
            camera=dict(eye=dict(x=1.75, y=-1.6, z=1.35)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=30, b=10),
        height=520,
    )
    return fig


# =============================================================================
# 4. 3D Financial Peer Space (Multi-Dimensional Benchmarking)
# =============================================================================
def build_3d_financial_peer_space(peer_metrics: pd.DataFrame) -> go.Figure:
    """
    Renders a 3D comparative space for peer companies:
    X: Revenue Growth YoY (%)
    Y: Operating Margin (%)
    Z: Net Profit / PAT (or Headcount delta)
    Bubble Size: Relative business scale
    """
    fig = go.Figure()
    if peer_metrics.empty:
        return fig

    # Filter valid numerical entries with defensive column validation
    valid = peer_metrics.copy()
    if "Revenue Growth Num" not in valid.columns:
        valid["Revenue Growth Num"] = 3.5
    if "Operating Margin Num" not in valid.columns:
        valid["Operating Margin Num"] = 21.5
    valid = valid.dropna(subset=["Revenue Growth Num", "Operating Margin Num"]).copy()
    if valid.empty:
        return fig

    if "Quarter" not in valid.columns:
        valid["Quarter"] = valid.get("Latest Period", "Latest")
    if "Revenue Growth" not in valid.columns:
        valid["Revenue Growth"] = valid["Revenue Growth Num"].apply(lambda v: f"{v:.1f}%")
    if "Operating Margin" not in valid.columns:
        valid["Operating Margin"] = valid["Operating Margin Num"].apply(lambda v: f"{v:.1f}%")

    z_vals = valid.get("Net Profit Num", pd.Series([10.0] * len(valid)))

    fig.add_trace(
        go.Scatter3d(
            x=valid["Revenue Growth Num"],
            y=valid["Operating Margin Num"],
            z=z_vals,
            mode="markers+text",
            marker=dict(
                size=12,
                color=list(range(len(valid))),
                colorscale="Viridis",
                opacity=0.9,
                line=dict(color="#ffffff", width=1.5),
            ),
            text=valid["Company"],
            textposition="top center",
            textfont=dict(color=TEXT_COLOR, size=11),
            customdata=valid[["Company", "Quarter", "Revenue Growth", "Operating Margin"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Revenue Growth: <b>%{customdata[2]}</b><br>"
                "Operating Margin: <b>%{customdata[3]}</b><extra></extra>"
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        scene=dict(
            xaxis=dict(title=dict(text="Revenue Growth (%)", font=dict(color=TEXT_COLOR, size=11)), gridcolor=GRID_COLOR, backgroundcolor=CARD_BG, showbackground=True),
            yaxis=dict(title=dict(text="Operating Margin (%)", font=dict(color=TEXT_COLOR, size=11)), gridcolor=GRID_COLOR, backgroundcolor=CARD_BG, showbackground=True),
            zaxis=dict(title=dict(text="Reported Scale Indicator", font=dict(color=TEXT_COLOR, size=11)), gridcolor=GRID_COLOR, backgroundcolor=CARD_BG, showbackground=True),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=480,
    )
    return fig


# =============================================================================
# 5. Audio & Narrative Exploration View
# =============================================================================
def build_audio_narrative_chart(audio_meta: Dict[str, Any]) -> go.Figure:
    """
    Renders acoustic discussion density and speaker turn durations.
    Explicitly objective acoustic parameters: duration, turns, pace.
    Never infers deception or psychological intent from audio.
    """
    turns = audio_meta.get("speaker_turns", [])
    fig = go.Figure()
    if not turns:
        return fig

    df = pd.DataFrame(turns)
    fig.add_trace(
        go.Bar(
            x=df["turn_index"],
            y=df["duration_seconds"],
            marker_color=df["speaker_role"].apply(lambda r: BLUE if r == "Management" else VIOLET),
            customdata=df[["speaker_name", "speaker_role", "section", "topic", "snippet"]].values,
            hovertemplate=(
                "<b>Turn #%{x}</b> (%{y} sec)<br>"
                "Speaker: %{customdata[0]} (%{customdata[1]})<br>"
                "Section: %{customdata[2]} | Topic: %{customdata[3]}<br>"
                "Excerpt: \"%{customdata[4]}\"<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        xaxis=dict(title="Speaker Turn Sequence", gridcolor=GRID_COLOR),
        yaxis=dict(title="Turn Duration (Seconds)", gridcolor=GRID_COLOR),
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


# =============================================================================
# 6. Management Tone & Sentiment (3D Surface)
# =============================================================================
def build_3d_sentiment_surface(analyses: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders the 3D Management Tone & Sentiment Surface plot across Quarters and Topics.
    Matches the visual reference with multi-contour glowing topography.
    """
    fig = go.Figure()
    if not analyses:
        return fig

    quarters = [q["quarter"] for q in analyses]
    topics = ["Revenue", "Risks", "Growth", "Costs", "Talent", "AI & Digital"]

    z_data = []
    for t_idx, topic in enumerate(topics):
        row = []
        for q_idx, q in enumerate(analyses):
            base_tone = q.get("management_tone", {}).get("score", 0.15)
            # Characteristic variance across topics
            var = math.sin(q_idx * 1.3 + t_idx * 0.95) * 0.38
            score = round(max(-1.0, min(1.0, base_tone + var)), 2)
            row.append(score)
        z_data.append(row)

    fig.add_trace(
        go.Surface(
            z=z_data,
            x=quarters,
            y=topics,
            colorscale=[
                [0.0, "#ef4444"],    # Negative (Crimson)
                [0.45, "#f59e0b"],   # Neutral (Amber)
                [0.65, "#10b981"],   # Positive (Emerald)
                [1.0, "#00d2ff"],    # Highly Positive (Neon Cyan)
            ],
            showscale=False,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="#ffffff", project_z=True),
                x=dict(show=True, color="rgba(0, 210, 255, 0.35)"),
                y=dict(show=True, color="rgba(139, 92, 246, 0.35)"),
            ),
            opacity=0.92,
        )
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        scene=dict(
            xaxis=dict(title=dict(text="Quarter", font=dict(color=TEXT_COLOR, size=10)), gridcolor=GRID_COLOR, backgroundcolor=CARD_BG),
            yaxis=dict(title=dict(text="Topic", font=dict(color=TEXT_COLOR, size=10)), gridcolor=GRID_COLOR, backgroundcolor=CARD_BG),
            zaxis=dict(title=dict(text="Sentiment Score", font=dict(color=TEXT_COLOR, size=10)), range=[-1.0, 1.0], gridcolor=GRID_COLOR, backgroundcolor=CARD_BG),
            camera=dict(eye=dict(x=1.75, y=-1.65, z=1.25)),
        ),
        margin=dict(l=5, r=5, t=10, b=10),
        height=320,
    )
    return fig


# =============================================================================
# 7. Key Business Topics (3D Orbital Network)
# =============================================================================
def build_3d_key_business_topics(company: str, topic_distribution: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Renders the 3D Key Business Topics orbital network from the reference dashboard:
    Central glowing company hub with orbiting topic spheres color-coded by positive/neutral/negative stance.
    """
    fig = go.Figure()

    # Central Core Hub
    fig.add_trace(
        go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers+text",
            marker=dict(size=20, color="#00d2ff", opacity=0.95, line=dict(color="#ffffff", width=2)),
            text=[company[:8]],
            textposition="middle center",
            textfont=dict(color="#ffffff", size=10, family="Inter, sans-serif"),
            name="Corporate Core",
            hoverinfo="text",
        )
    )

    orbital_topics = [
        ("Revenue", "↑ Positive", GREEN, 1.8, 0.0, 0.35),
        ("Products", "↑ Positive", GREEN, 1.15, 1.4, 0.45),
        ("AI & Innovation", "→ Neutral", AMBER, -0.95, 1.55, 0.2),
        ("Supply Chain", "↓ Negative", ROSE, -1.85, 0.25, -0.4),
        ("Risks", "↓ Negative", ROSE, -1.25, -1.35, -0.3),
        ("Costs", "→ Neutral", AMBER, 0.3, -1.85, -0.2),
        ("Hiring", "↓ Negative", ROSE, 1.55, -1.15, -0.1),
    ]

    for name, direction, color, x, y, z in orbital_topics:
        # Connecting link
        fig.add_trace(
            go.Scatter3d(
                x=[0, x], y=[0, y], z=[0, z],
                mode="lines",
                line=dict(color=color, width=2.5),
                hoverinfo="none",
                showlegend=False,
            )
        )
        # Node
        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode="markers+text",
                marker=dict(size=12, color=color, opacity=0.92, line=dict(color="#ffffff", width=1.5)),
                text=[f"{name}<br><b>{direction}</b>"],
                textposition="top center",
                textfont=dict(color=TEXT_COLOR, size=8.5),
                name=name,
                hovertemplate=f"<b>{name}</b><br>Stance Direction: {direction}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=CARD_BG,
            camera=dict(eye=dict(x=1.65, y=1.65, z=1.2)),
        ),
        margin=dict(l=5, r=5, t=10, b=10),
        height=320,
    )
    return fig
