import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def render_analytics_dashboard():

    st.markdown("""
    <style>
    .kpi-container {
        background: #1e2a3a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .kpi-label {
        font-size: 12px;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    .kpi-sub {
        font-size: 12px;
        color: #8899aa;
        margin-top: 4px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #e2e8f0;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Page Header ──────────────────────────────────────────────────────
    st.markdown("## Analytics Dashboard")
    st.markdown(
        "<p style='color:#8899aa;margin-top:-12px;margin-bottom:24px'>"
        "Platform-wide performance metrics · IntelliRisk AI</p>",
        unsafe_allow_html=True
    )

    # ── Module Tabs ───────────────────────────────────────────────────────
    tab_overview, tab_loan, tab_fraud = st.tabs([
        "Overview", "Loan module", "Fraud module"
    ])

    # ════════════════════════════════════════════════════════════════
    #  TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════
    with tab_overview:

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Total records</div>
                <div class="kpi-value">322,931</div>
                <div class="kpi-sub">307K loan + 15K fraud</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Models trained</div>
                <div class="kpi-value">6</div>
                <div class="kpi-sub">3 loan + 3 fraud models</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Best AUC</div>
                <div class="kpi-value">0.8115</div>
                <div class="kpi-sub">Fraud XGBoost</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Fraud caught</div>
                <div class="kpi-value">923</div>
                <div class="kpi-sub">6.0% of 15,420 claims</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

        # Row 1: AUC comparison + data split
        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown("<div class='section-header'>Model AUC comparison — all 6 models</div>", unsafe_allow_html=True)
            auc_df = pd.DataFrame({
                "Model": [
                    "XGB Loan", "RF Loan", "LR Loan",
                    "XGB Fraud", "RF Fraud", "LR Fraud"
                ],
                "AUC": [0.7656, 0.74, 0.70, 0.8115, 0.7965, 0.7915],
                "Module": ["Loan","Loan","Loan","Fraud","Fraud","Fraud"]
            })
            color_map = {"Loan": "#1D9E75", "Fraud": "#534AB7"}
            fig_auc = px.bar(
                auc_df, x="AUC", y="Model", orientation="h",
                color="Module", color_discrete_map=color_map,
                range_x=[0.55, 0.85]
            )
            fig_auc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c2c0b6",
                height=260,
                margin=dict(l=10, r=20, t=10, b=10),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1
                )
            )
            fig_auc.update_xaxes(
                gridcolor="rgba(255,255,255,0.06)",
                tickformat=".2f"
            )
            fig_auc.update_yaxes(gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_auc, use_container_width=True)

        with col_right:
            st.markdown("<div class='section-header'>Platform data split</div>", unsafe_allow_html=True)
            fig_split = go.Figure(go.Pie(
                labels=["Loan module", "Fraud module"],
                values=[307511, 15420],
                hole=0.65,
                marker_colors=["#1D9E75", "#534AB7"],
                textinfo="label+percent",
                textfont_size=12,
            ))
            fig_split.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c2c0b6",
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_split, use_container_width=True)

        # Row 2: SHAP features + tech stack
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='section-header'>Explainability — SHAP features</div>", unsafe_allow_html=True)
            fig_shap = px.bar(
                x=["Loan module", "Fraud module"],
                y=[122, 33],
                color=["Loan module", "Fraud module"],
                color_discrete_map={"Loan module": "#1D9E75", "Fraud module": "#534AB7"},
                labels={"x": "Module", "y": "Feature count"}
            )
            fig_shap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c2c0b6",
                height=200,
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig_shap.update_xaxes(gridcolor="rgba(0,0,0,0)")
            fig_shap.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_shap, use_container_width=True)

        with col_b:
            st.markdown("<div class='section-header'>Tech stack</div>", unsafe_allow_html=True)
            techs = [
                ("XGBoost", "Primary classifier"),
                ("SHAP", "Explainability layer"),
                ("Isolation Forest", "Anomaly detection"),
                ("SMOTE", "Class balancing"),
                ("Streamlit", "Frontend"),
                ("Plotly", "Visualisation"),
            ]
            t1, t2 = st.columns(2)
            for i, (name, desc) in enumerate(techs):
                col = t1 if i % 2 == 0 else t2
                col.markdown(
                    f"<div style='background:rgba(255,255,255,0.04);border-radius:8px;"
                    f"padding:10px 12px;margin-bottom:8px;font-size:13px;color:#8899aa'>"
                    f"<span style='font-weight:600;color:#e2e8f0'>{name}</span><br>{desc}</div>",
                    unsafe_allow_html=True
                )

    # ════════════════════════════════════════════════════════════════
    #  TAB 2 — LOAN MODULE
    # ════════════════════════════════════════════════════════════════
    with tab_loan:

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Borrowers</div>
                <div class="kpi-value">307,511</div>
                <div class="kpi-sub">Home Credit dataset</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Best AUC</div>
                <div class="kpi-value">0.7656</div>
                <div class="kpi-sub">XGBoost model</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Features</div>
                <div class="kpi-value">122</div>
                <div class="kpi-sub">After engineering</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Default rate</div>
                <div class="kpi-value">8.1%</div>
                <div class="kpi-sub">24,825 defaults</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown("<div class='section-header'>Model performance comparison</div>", unsafe_allow_html=True)
            loan_models = pd.DataFrame({
                "Model": ["XGBoost", "Random Forest", "Logistic Reg."],
                "AUC": [0.7656, 0.74, 0.70]
            })
            fig_lm = px.bar(
                loan_models, x="Model", y="AUC",
                color="AUC",
                color_continuous_scale=[[0, "#9FE1CB"], [0.5, "#5DCAA5"], [1, "#1D9E75"]],
                range_y=[0.55, 0.80]
            )
            fig_lm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c2c0b6", height=240,
                margin=dict(l=10,r=10,t=10,b=10),
                coloraxis_showscale=False
            )
            fig_lm.update_xaxes(gridcolor="rgba(0,0,0,0)")
            fig_lm.update_yaxes(
                gridcolor="rgba(255,255,255,0.06)",
                tickformat=".2f"
            )
            st.plotly_chart(fig_lm, use_container_width=True)

        with col_right:
            st.markdown("<div class='section-header'>Default vs non-default</div>", unsafe_allow_html=True)
            fig_bal = go.Figure(go.Pie(
                labels=["No default", "Default"],
                values=[282686, 24825],
                hole=0.65,
                marker_colors=["#1D9E75", "#D85A30"],
                textinfo="label+percent",
                textfont_size=12
            ))
            fig_bal.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c2c0b6", height=240,
                margin=dict(l=10,r=10,t=10,b=10),
                showlegend=False
            )
            st.plotly_chart(fig_bal, use_container_width=True)

        st.markdown("<div class='section-header'>Top SHAP features — key drivers of loan decisions</div>", unsafe_allow_html=True)
        shap_loan = pd.DataFrame({
            "Feature": [
                "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH",
                "DAYS_EMPLOYED", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"
            ],
            "Importance": [0.18, 0.15, 0.11, 0.10, 0.08, 0.07, 0.06]
        }).sort_values("Importance")
        fig_shap_l = px.bar(
            shap_loan, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#9FE1CB"], [1, "#1D9E75"]]
        )
        fig_shap_l.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c2c0b6", height=260,
            margin=dict(l=10,r=10,t=10,b=10),
            coloraxis_showscale=False
        )
        fig_shap_l.update_xaxes(
            gridcolor="rgba(255,255,255,0.06)",
            tickformat=".2f"
        )
        fig_shap_l.update_yaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_shap_l, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    #  TAB 3 — FRAUD MODULE
    # ════════════════════════════════════════════════════════════════
    with tab_fraud:

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Total claims</div>
                <div class="kpi-value">15,420</div>
                <div class="kpi-sub">Oracle ML dataset</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Best AUC</div>
                <div class="kpi-value">0.8115</div>
                <div class="kpi-sub">XGBoost model</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Fraud cases</div>
                <div class="kpi-value">923</div>
                <div class="kpi-sub">6.0% fraud rate</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown("""<div class="kpi-container">
                <div class="kpi-label">Features</div>
                <div class="kpi-value">33</div>
                <div class="kpi-sub">After preprocessing</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown("<div class='section-header'>Fraud model performance comparison</div>", unsafe_allow_html=True)
            fraud_models = pd.DataFrame({
                "Model": ["XGBoost", "Random Forest", "Logistic Reg."],
                "AUC": [0.8115, 0.7965, 0.7915]
            })
            fig_fm = px.bar(
                fraud_models, x="Model", y="AUC",
                color="AUC",
                color_continuous_scale=[[0, "#AFA9EC"], [0.5, "#7F77DD"], [1, "#534AB7"]],
                range_y=[0.70, 0.83]
            )
            fig_fm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c2c0b6", height=240,
                margin=dict(l=10,r=10,t=10,b=10),
                coloraxis_showscale=False
            )
            fig_fm.update_xaxes(gridcolor="rgba(0,0,0,0)")
            fig_fm.update_yaxes(
                gridcolor="rgba(255,255,255,0.06)",
                tickformat=".2f"
            )
            st.plotly_chart(fig_fm, use_container_width=True)

        with col_right:
            st.markdown("<div class='section-header'>Fraud vs legitimate claims</div>", unsafe_allow_html=True)
            fig_fb = go.Figure(go.Pie(
                labels=["Legitimate", "Fraud"],
                values=[14497, 923],
                hole=0.65,
                marker_colors=["#534AB7", "#D85A30"],
                textinfo="label+percent",
                textfont_size=12
            ))
            fig_fb.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c2c0b6", height=240,
                margin=dict(l=10,r=10,t=10,b=10),
                showlegend=False
            )
            st.plotly_chart(fig_fb, use_container_width=True)

        st.markdown("<div class='section-header'>Fraud rate by fault type — policy holder 8x higher risk</div>", unsafe_allow_html=True)
        fault_df = pd.DataFrame({
            "Fault type": ["Policy holder", "Third party vehicle", "Third party"],
            "Fraud rate %": [18.4, 9.2, 2.3]
        })
        fig_fault = px.bar(
            fault_df, x="Fault type", y="Fraud rate %",
            color="Fault type",
            color_discrete_map={
                "Policy holder": "#D85A30",
                "Third party vehicle": "#EF9F27",
                "Third party": "#534AB7"
            },
            text="Fraud rate %"
        )
        fig_fault.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_fault.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c2c0b6", height=240,
            margin=dict(l=10,r=10,t=30,b=10),
            showlegend=False
        )
        fig_fault.update_xaxes(gridcolor="rgba(0,0,0,0)")
        fig_fault.update_yaxes(
            gridcolor="rgba(255,255,255,0.06)",
            ticksuffix="%"
        )
        st.plotly_chart(fig_fault, use_container_width=True)
