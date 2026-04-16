import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #0f1117);
        border: 1px solid #2d3150; border-radius: 12px;
        padding: 20px; text-align: center; margin: 8px 0;
    }
    .metric-card .label {
        font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
        color: #6b7db3; font-family: 'IBM Plex Mono', monospace; margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 28px; font-weight: 700; color: #e8eaf6;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-card .delta { font-size: 12px; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }
    .section-header {
        font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
        color: #4a5568; font-family: 'IBM Plex Mono', monospace;
        margin: 24px 0 10px 0; padding-bottom: 8px; border-bottom: 1px solid #1e2235;
    }
    .page-title { font-size: 24px; font-weight: 700; color: #e8eaf6; letter-spacing: -0.5px; margin-bottom: 4px; }
    .page-subtitle { font-size: 12px; color: #6b7db3; margin-bottom: 24px; font-family: 'IBM Plex Mono', monospace; }
    div[data-testid="stSidebar"] { background: #0a0c14; border-right: 1px solid #1e2235; }
</style>
""", unsafe_allow_html=True)

# ─── COLORS ────────────────────────────────────────────────────────────────────
PROMPT_COLORS = {
    "Baseline(Prompt_v1)": "#6b7db3",
    "Prompt_v2":           "#818cf8",
    "Prompt_v3":           "#34d399",
    "Prompt_v4":           "#f472b6",
}
FINETUNE_COLORS = {
    "Baseline(Prompt_v1)":  "#6b7db3",
    "Finetune_Prompt_v1":   "#818cf8",
    "Finetune_Prompt_v2":   "#34d399",
    "Finetune_Prompt_v3":   "#f472b6",
    "Finetune_Prompt_v4":   "#fb923c",
}
METRIC_COLORS = {"CER": "#818cf8", "WER": "#34d399", "CLS": "#f472b6"}

# ─── DATA LOADING ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    prompts  = pd.read_csv(f"{DATA_DIR}/prompts_comparison.csv")
    fields   = pd.read_csv(f"{DATA_DIR}/field_scores.csv")
    cls_sc   = pd.read_csv(f"{DATA_DIR}/classification_scores.csv")
    finetune = pd.read_csv(f"{DATA_DIR}/finetune_comparison.csv")
    loss     = pd.read_csv(f"{DATA_DIR}/training_loss.csv")
    return prompts, fields, cls_sc, finetune, loss

df_prompts, df_fields, df_cls, df_finetune, df_loss = load_data()

# categories used in diff table / CLS bar (no yukang/jingjie)
MAIN_CATS = ['all(20)', 'tidy', 'messy', 'edge_cases']

# ─── LAYOUT BASE ───────────────────────────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    font=dict(color="#6b7db3", family="IBM Plex Mono", size=11),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor="#1e2235", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#1e2235", tickfont=dict(size=11)),
)

# ─── PLOT HELPERS ──────────────────────────────────────────────────────────────

def grouped_metric_bar(df_in, x_col, title, height=420):
    """Grouped (non-stacked) bar for CER, WER, CLS error — no overlap."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="CER", x=df_in[x_col], y=df_in["CER"],
        marker_color=METRIC_COLORS["CER"],
        text=df_in["CER"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside", textfont=dict(size=9),
    ))
    fig.add_trace(go.Bar(
        name="WER", x=df_in[x_col], y=df_in["WER"],
        marker_color=METRIC_COLORS["WER"],
        text=df_in["WER"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside", textfont=dict(size=9),
    ))
    fig.add_trace(go.Bar(
        name="CLS Error (100−CLS)", x=df_in[x_col], y=100 - df_in["CLS"],
        marker_color=METRIC_COLORS["CLS"],
        text=(100 - df_in["CLS"]).apply(lambda v: f"{v:.1f}%"),
        textposition="outside", textfont=dict(size=9),
    ))
    fig.update_layout(**{**LAYOUT_BASE, "barmode": "group", "height": height,
        "title": dict(text=title, font=dict(size=13, color="#e8eaf6")),
        "yaxis": {**LAYOUT_BASE["yaxis"], "title": "Score (%)"}})
    return fig


def line_graph_selected(df_in, y_col, title, selected_prompts, color_map, height=380):
    """Line graph — only selected prompts shown."""
    fig = go.Figure()
    for prompt in selected_prompts:
        color = color_map.get(prompt, "#999")
        sub = df_in[df_in["prompt"] == prompt]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["category"], y=sub[y_col], name=prompt,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7, color=color),
        ))
    fig.update_layout(**{**LAYOUT_BASE, "height": height,
        "title": dict(text=title, font=dict(size=13, color="#e8eaf6")),
        "yaxis": {**LAYOUT_BASE["yaxis"], "title": f"{y_col} (%)"}})
    return fig


def cls_grouped_bar(df_in, selected_cats, title, height=400):
    """Grouped bar for CLS per field per prompt — only main cats."""
    sub = df_in[df_in["category"].isin(selected_cats)]
    fig = go.Figure()
    for prompt, color in PROMPT_COLORS.items():
        p = sub[sub["prompt"] == prompt]
        if p.empty:
            continue
        # aggregate across cats
        p_avg = p.groupby("field")["value"].mean().reset_index()
        fig.add_trace(go.Bar(
            name=prompt, x=p_avg["field"], y=(p_avg["value"] * 100).round(1),
            marker_color=color,
            text=(p_avg["value"] * 100).round(1).apply(lambda v: f"{v:.1f}%"),
            textposition="outside", textfont=dict(size=9),
        ))
    fig.update_layout(**{**LAYOUT_BASE, "barmode": "group", "height": height,
        "title": dict(text=title, font=dict(size=13, color="#e8eaf6")),
        "yaxis": {**LAYOUT_BASE["yaxis"], "title": "Accuracy (%)"}})
    return fig


def field_bar_selected(df_in, metric, selected_fields, selected_cat, title, height=440):
    """Grouped bar for selected fields/category — tick-box driven."""
    sub = df_in[(df_in["metric"] == metric) & (df_in["field"].isin(selected_fields)) &
                (df_in["category"] == selected_cat)]
    fig = go.Figure()
    for prompt, color in PROMPT_COLORS.items():
        p = sub[sub["prompt"] == prompt]
        if p.empty:
            continue
        fig.add_trace(go.Bar(
            name=prompt, x=p["field"], y=p["value"],
            marker_color=color,
            text=p["value"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside", textfont=dict(size=9),
        ))
    fig.update_layout(**{**LAYOUT_BASE, "barmode": "group", "height": height,
        "title": dict(text=title, font=dict(size=13, color="#e8eaf6")),
        "xaxis": {**LAYOUT_BASE["xaxis"], "tickangle": -25, "tickfont": dict(size=10)},
        "yaxis": {**LAYOUT_BASE["yaxis"], "title": f"{metric} (%)"}})
    return fig


def loss_line_single(df_in, prompt):
    """Loss curve for one prompt."""
    sub = df_in[df_in["prompt"] == prompt]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["step"], y=sub["train_loss"], name="Train Loss",
        mode="lines+markers", line=dict(color="#818cf8", width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=sub["step"], y=sub["val_loss"], name="Val Loss",
        mode="lines+markers", line=dict(color="#f472b6", width=2, dash="dot"),
        marker=dict(size=6, symbol="square")))
    fig.update_layout(**{**LAYOUT_BASE, "height": 360,
        "title": dict(text=f"Loss — {prompt}", font=dict(size=13, color="#e8eaf6")),
        "xaxis": {**LAYOUT_BASE["xaxis"], "title": "Step"},
        "yaxis": {**LAYOUT_BASE["yaxis"], "title": "Loss"}})
    return fig


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 28px 0;'>
        <div style='font-size:18px;font-weight:700;color:#e8eaf6;'>OCR Benchmark</div>
        <div style='font-size:10px;color:#4a5568;font-family:IBM Plex Mono;letter-spacing:2px;margin-top:4px;'>
            QWEN3-VL · 2B INSTRUCT
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigate",
        ["📈 Finetuning Difference", "🔍 Prompts vs Baseline"],
        label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#4a5568;font-family:IBM Plex Mono;line-height:2;'>
        METRICS<br>
        <span style='color:#818cf8;'>■</span> CER — Character Error Rate<br>
        <span style='color:#34d399;'>■</span> WER — Word Error Rate<br>
        <span style='color:#f472b6;'>■</span> CLS — Classification Accuracy
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#4a5568;font-family:IBM Plex Mono;line-height:1.8;'>
        CSV FILES<br>
        prompts_comparison.csv<br>
        field_scores.csv<br>
        classification_scores.csv<br>
        finetune_comparison.csv<br>
        training_loss.csv
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FINETUNING DIFFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📈 Finetuning Difference":

    st.markdown('<div class="page-title">Finetuning Difference</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">BASELINE vs FINETUNED MODELS — CER · WER · CLS</div>', unsafe_allow_html=True)

    base_row = df_finetune[df_finetune["model"] == "Baseline(Prompt_v1)"].iloc[0]
    best_row = df_finetune[df_finetune["model"] != "Baseline(Prompt_v1)"].sort_values("CER").iloc[0]

    # ── METRIC CARDS ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Best Model Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, base_val, lower_better in zip(
        [c1, c2, c3, c4],
        ["Best CER", "Best WER", "Best CLS", "Best Model"],
        [best_row["CER"], best_row["WER"], best_row["CLS"], best_row["model"]],
        [base_row["CER"], base_row["WER"], base_row["CLS"], None],
        [True, True, False, None],
    ):
        with col:
            if label == "Best Model":
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value" style="font-size:15px;padding-top:10px;">{val}</div>
                    <div class="delta" style="color:#818cf8;">top performer</div>
                </div>""", unsafe_allow_html=True)
            else:
                delta = val - base_val
                good = (delta < 0 and lower_better) or (delta > 0 and not lower_better)
                color = "#34d399" if good else "#f87171"
                fmt = f"{val:.2f}%" if "CLS" not in label else f"{val:.1f}%"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{fmt}</div>
                    <div class="delta" style="color:{color};">{delta:+.2f}% vs baseline</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── RESULTS TABLE ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Results Table — CER · WER · CLS</div>', unsafe_allow_html=True)

    df_display = df_finetune.copy()
    df_display["CER Δ"] = df_display["CER"] - base_row["CER"]
    df_display["WER Δ"] = df_display["WER"] - base_row["WER"]
    df_display["CLS Δ"] = df_display["CLS"] - base_row["CLS"]

    st.dataframe(
        df_display.style
        .format({"CER": "{:.2f}%", "WER": "{:.2f}%", "CLS": "{:.1f}%",
                 "CER Δ": "{:+.2f}%", "WER Δ": "{:+.2f}%", "CLS Δ": "{:+.1f}%"})
        .map(lambda v: "color:#34d399" if isinstance(v, (int, float)) and v < 0 else
                            "color:#f87171" if isinstance(v, (int, float)) and v > 0 else "",
                  subset=["CER Δ", "WER Δ"])
        .map(lambda v: "color:#34d399" if isinstance(v, (int, float)) and v > 0 else
                            "color:#f87171" if isinstance(v, (int, float)) and v < 0 else "",
                  subset=["CLS Δ"])
        .set_properties(**{"background-color": "#0f1117", "color": "#e8eaf6"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("")

    # ── GROUPED BAR — no overlapping ──────────────────────────────────────────
    st.markdown('<div class="section-header">CER · WER · CLS Comparison (Grouped)</div>', unsafe_allow_html=True)
    st.plotly_chart(
        grouped_metric_bar(df_finetune, "model", "CER / WER / CLS Error — Finetuned Models"),
        use_container_width=True
    )

    # ── TRAINING LOSS — DROPDOWN ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Training & Validation Loss</div>', unsafe_allow_html=True)

    loss_prompts = df_loss["prompt"].unique().tolist()
    selected_loss_prompt = st.selectbox(
        "Select prompt to view loss curve:",
        options=loss_prompts,
        index=0,
    )

    col_loss, col_table = st.columns([3, 2])
    with col_loss:
        st.plotly_chart(loss_line_single(df_loss, selected_loss_prompt), use_container_width=True)

    with col_table:
        st.markdown("**Prompt Summary (No Finetune vs Finetuned)**")
        # build comparison table: baseline prompt vs finetuned version
        prompt_map = {
            "Prompt_v1": ("Baseline(Prompt_v1)", "Finetune_Prompt_v1"),
            "Prompt_v2": ("Prompt_v2",           "Finetune_Prompt_v2"),
            "Prompt_v3": ("Prompt_v3",           "Finetune_Prompt_v3"),
            "Prompt_v4": ("Prompt_v4",            "Finetune_Prompt_v4"),
        }
        pname = selected_loss_prompt
        if pname in prompt_map:
            base_name, ft_name = prompt_map[pname]

            # get no-finetune from prompts_comparison all(20)
            p_noFT = df_prompts[(df_prompts["prompt"] == base_name) & (df_prompts["category"] == "all(20)")]
            p_FT   = df_finetune[df_finetune["model"] == ft_name]

            rows = []
            if not p_noFT.empty:
                r = p_noFT.iloc[0]
                rows.append({"": "No Finetune", "CER": f"{r['CER']:.2f}%", "WER": f"{r['WER']:.2f}%", "CLS": f"{r['CLS']:.1f}%"})
            if not p_FT.empty:
                r = p_FT.iloc[0]
                rows.append({"": "Finetuned", "CER": f"{r['CER']:.2f}%", "WER": f"{r['WER']:.2f}%", "CLS": f"{r['CLS']:.1f}%"})

            if rows:
                st.dataframe(pd.DataFrame(rows).set_index(""),
                             use_container_width=True)
        else:
            st.info("No matching finetune data found for this prompt.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PROMPTS VS BASELINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Prompts vs Baseline":

    st.markdown('<div class="page-title">Prompts vs Baseline</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">BASELINE(PROMPT_V1) → PROMPT_V2 → PROMPT_V3 → PROMPT_V4</div>', unsafe_allow_html=True)

    prompts_available = list(PROMPT_COLORS.keys())
    df_main = df_prompts[df_prompts["category"].isin(MAIN_CATS)]

    # ── DROPDOWN: select single prompt to inspect ──────────────────────────────
    st.markdown('<div class="section-header">Select Prompt to Inspect</div>', unsafe_allow_html=True)
    selected_prompt = st.selectbox("Choose a prompt:", options=prompts_available)
    df_selected = df_prompts[df_prompts["prompt"] == selected_prompt]

    # ── TABLE 1: CER / WER / CLS for selected prompt ──────────────────────────
    st.markdown('<div class="section-header">CER · WER · CLS by Category</div>', unsafe_allow_html=True)

    df_t1 = df_selected[df_selected["category"].isin(MAIN_CATS)][["category", "CER", "WER", "CLS"]].copy()
    df_t1.columns = ["Category", "CER (%)", "WER (%)", "CLS (%)"]
    st.dataframe(
        df_t1.style
        .format({"CER (%)": "{:.2f}%", "WER (%)": "{:.2f}%", "CLS (%)": "{:.1f}%"})
        .background_gradient(subset=["CER (%)"], cmap="RdYlGn_r")
        .background_gradient(subset=["WER (%)"], cmap="RdYlGn_r")
        .background_gradient(subset=["CLS (%)"], cmap="RdYlGn")
        .set_properties(**{"background-color": "#0f1117", "color": "#e8eaf6"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("")

    # ── GROUPED BAR for selected prompt ───────────────────────────────────────
    st.markdown('<div class="section-header">Error Distribution (Grouped)</div>', unsafe_allow_html=True)
    df_bar_sel = df_selected[df_selected["category"].isin(MAIN_CATS)]
    st.plotly_chart(
        grouped_metric_bar(df_bar_sel, "category", f"CER / WER / CLS — {selected_prompt}", height=400),
        use_container_width=True
    )

    st.markdown("")

# ── TABLE 2: RAW VALUES — 2x2 GRID ───────────────────────────────────────
    st.markdown('<div class="section-header">Prompt Results — All(20) · Tidy · Messy · Edge Cases</div>', unsafe_allow_html=True)

    # build base dataframe
    raw_rows = []
    for prompt in prompts_available:
        p_df = df_prompts[df_prompts["prompt"] == prompt].set_index("category")
        row  = {"Prompt": prompt}
        for cat in MAIN_CATS:
            if cat in p_df.index:
                row[f"CER_{cat}"] = round(p_df.loc[cat, "CER"], 2)
                row[f"WER_{cat}"] = round(p_df.loc[cat, "WER"], 2)
                row[f"CLS_{cat}"] = round(p_df.loc[cat, "CLS"], 1)
        raw_rows.append(row)

    df_raw = pd.DataFrame(raw_rows)

    # ── ROW 1 ─────────────────────────────────────────────────────────────────
    col_tl, col_tr = st.columns(2)

    # TOP LEFT — Overall (avg across MAIN_CATS)
    with col_tl:
        st.markdown("**Overall (avg across categories)**")
        df_overall = df_raw[["Prompt"]].copy()
        df_overall["Avg CER (%)"] = df_raw[[f"CER_{c}" for c in MAIN_CATS]].mean(axis=1).round(2)
        df_overall["Avg WER (%)"] = df_raw[[f"WER_{c}" for c in MAIN_CATS]].mean(axis=1).round(2)
        df_overall["Avg CLS (%)"] = df_raw[[f"CLS_{c}" for c in MAIN_CATS]].mean(axis=1).round(1)
        st.dataframe(
            df_overall.style
            .format({"Avg CER (%)": "{:.2f}%", "Avg WER (%)": "{:.2f}%", "Avg CLS (%)": "{:.1f}%"})
            .background_gradient(subset=["Avg CER (%)"], cmap="RdYlGn_r")
            .background_gradient(subset=["Avg WER (%)"], cmap="RdYlGn_r")
            .background_gradient(subset=["Avg CLS (%)"], cmap="RdYlGn")
            .set_properties(**{"background-color": "#0f1117", "color": "#e8eaf6"}),
            use_container_width=True, hide_index=True,
        )

    # TOP RIGHT — CER
    with col_tr:
        st.markdown("**CER (%) per Category**")
        cer_cols = [f"CER_{c}" for c in MAIN_CATS]
        df_cer = df_raw[["Prompt"] + cer_cols].copy()
        df_cer.columns = ["Prompt"] + MAIN_CATS
        st.dataframe(
            df_cer.style
            .format({c: "{:.2f}%" for c in MAIN_CATS})
            .background_gradient(subset=MAIN_CATS, cmap="RdYlGn_r")
            .set_properties(**{"background-color": "#0f1117", "color": "#e8eaf6"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown("")

    # ── ROW 2 ─────────────────────────────────────────────────────────────────
    col_bl, col_br = st.columns(2)

    # BOTTOM LEFT — WER
    with col_bl:
        st.markdown("**WER (%) per Category**")
        wer_cols = [f"WER_{c}" for c in MAIN_CATS]
        df_wer = df_raw[["Prompt"] + wer_cols].copy()
        df_wer.columns = ["Prompt"] + MAIN_CATS
        st.dataframe(
            df_wer.style
            .format({c: "{:.2f}%" for c in MAIN_CATS})
            .background_gradient(subset=MAIN_CATS, cmap="RdYlGn_r")
            .set_properties(**{"background-color": "#0f1117", "color": "#e8eaf6"}),
            use_container_width=True, hide_index=True,
        )

    # BOTTOM RIGHT — CLS
    with col_br:
        st.markdown("**CLS (%) per Category**")
        cls_cols = [f"CLS_{c}" for c in MAIN_CATS]
        df_cls_t = df_raw[["Prompt"] + cls_cols].copy()
        df_cls_t.columns = ["Prompt"] + MAIN_CATS
        st.dataframe(
            df_cls_t.style
            .format({c: "{:.1f}%" for c in MAIN_CATS})
            .background_gradient(subset=MAIN_CATS, cmap="RdYlGn")
            .set_properties(**{"background-color": "#0f1117", "color": "#e8eaf6"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown("")
    # ── LINE GRAPHS — TICK-BOX MULTISELECT ────────────────────────────────────
    st.markdown('<div class="section-header">CER & WER Line Graphs — Select Prompts to Compare</div>', unsafe_allow_html=True)

    selected_for_line = st.multiselect(
        "Tick prompts to compare:",
        options=prompts_available,
        default=prompts_available,
    )

    df_line_data = df_prompts[df_prompts["category"].isin(MAIN_CATS)]
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            line_graph_selected(df_line_data, "CER", "CER (%) per Category",
                                selected_for_line, PROMPT_COLORS),
            use_container_width=True
        )
    with col_b:
        st.plotly_chart(
            line_graph_selected(df_line_data, "WER", "WER (%) per Category",
                                selected_for_line, PROMPT_COLORS),
            use_container_width=True
        )

    # ── CLS GROUPED BAR — only all(20), tidy, messy, edge_cases ──────────────
    st.markdown('<div class="section-header">Classification Accuracy — All Prompts</div>', unsafe_allow_html=True)
    st.plotly_chart(
        cls_grouped_bar(df_cls, MAIN_CATS, "CLS Accuracy (%) per Field — All Prompts"),
        use_container_width=True
    )

    # ── FIELD-LEVEL BREAKDOWN — TICK-BOX ─────────────────────────────────────
    st.markdown('<div class="section-header">Field-Level Breakdown</div>', unsafe_allow_html=True)

    tab_cer, tab_wer = st.tabs(["CER per Field", "WER per Field"])

    with tab_cer:
        cer_fields_all = sorted(df_fields[df_fields["metric"] == "CER"]["field"].unique().tolist())
        cer_cats_all   = sorted(df_fields[df_fields["metric"] == "CER"]["category"].unique().tolist())

        col_ctrl, col_chart = st.columns([1, 3])
        with col_ctrl:
            st.markdown("**Fields**")
            sel_cer_fields = []
            for f in cer_fields_all:
                if st.checkbox(f, value=True, key=f"cer_{f}"):
                    sel_cer_fields.append(f)
            st.markdown("**Category**")
            sel_cer_cat = st.selectbox("Category", cer_cats_all, key="cer_cat")

        with col_chart:
            if sel_cer_fields:
                st.plotly_chart(
                    field_bar_selected(df_fields, "CER", sel_cer_fields, sel_cer_cat,
                                       f"CER (%) — {sel_cer_cat}", height=480),
                    use_container_width=True
                )
            else:
                st.info("Select at least one field.")

    with tab_wer:
        wer_fields_all = sorted(df_fields[df_fields["metric"] == "WER"]["field"].unique().tolist())
        wer_cats_all   = sorted(df_fields[df_fields["metric"] == "WER"]["category"].unique().tolist())

        col_ctrl2, col_chart2 = st.columns([1, 3])
        with col_ctrl2:
            st.markdown("**Fields**")
            sel_wer_fields = []
            for f in wer_fields_all:
                if st.checkbox(f, value=True, key=f"wer_{f}"):
                    sel_wer_fields.append(f)
            st.markdown("**Category**")
            sel_wer_cat = st.selectbox("Category", wer_cats_all, key="wer_cat")

        with col_chart2:
            if sel_wer_fields:
                st.plotly_chart(
                    field_bar_selected(df_fields, "WER", sel_wer_fields, sel_wer_cat,
                                       f"WER (%) — {sel_wer_cat}", height=480),
                    use_container_width=True
                )
            else:
                st.info("Select at least one field.")
