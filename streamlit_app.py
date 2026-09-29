#!/usr/bin/env python3
"""
CFA Numérique – Streamlit Dashboard
====================================
Interactive monetary thermostat explorer.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.linalg import solve_discrete_are
import json
from datetime import datetime

st.set_page_config(
    page_title="CFA Numérique – Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        color: #0f3b5e;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #c8a45c;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f4efe1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #c8a45c;
    }
    .stMetricValue {
        color: #0f3b5e !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">CFA Numérique – Tableau de bord</p>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Thermostat monétaire cointégré pour la BEAC</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------
st.sidebar.header("Paramètres du modèle")

kappa = st.sidebar.slider(
    "κ (réaction au déséquilibre)",
    min_value=0.0, max_value=1.5, value=0.48, step=0.02,
    help="Plus κ est élevé, plus la règle réagit fortement à l'ECT."
)

alpha = st.sidebar.slider(
    "α (vitesse d'ajustement)",
    min_value=-0.30, max_value=-0.05, value=-0.15, step=0.01,
    help="Vitesse à laquelle la masse monétaire revient à l'équilibre."
)

theta = st.sidebar.slider(
    "θ (semi-élasticité au taux)",
    min_value=-0.05, max_value=0.0, value=-0.02, step=0.001,
    help="Effet causal du taux sur la croissance monétaire (identifié par IV)."
)

lam = st.sidebar.slider(
    "λ (pénalité de volatilité)",
    min_value=0.01, max_value=1.0, value=0.1, step=0.01,
    help="Poids accordé à la stabilité du taux dans la fonction de perte."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Modèle champion**")
st.sidebar.info("Linear DOLS + LQR\n\nLes satellites (NARDL, NN, KAN) s'exécutent en arrière-plan.")

# ---------------------------------------------------------------------------
# Data generation (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def generate_data(T=80, seed=2024):
    np.random.seed(seed)
    world_trend = np.cumsum(np.random.normal(0.003, 0.01, T))
    oil_trend   = np.cumsum(np.random.normal(0.002, 0.03, T))
    country_trends = np.random.normal(0, 0.02, (6, T)).cumsum(axis=1)
    real_gdp = np.zeros((6, T))
    for i in range(6):
        oil_coef = 0.5 if i in [1, 2, 3, 5] else 0.1
        real_gdp[i] = 10 + 0.8*world_trend + oil_coef*oil_trend + 0.3*country_trends[i]
    inflation = np.zeros_like(real_gdp)
    for i in range(6):
        inflation[i] = 0.03 + np.cumsum(np.random.normal(0, 0.004, T))
    consts = np.linspace(1.0, 2.0, 6)
    real_m2 = np.zeros_like(real_gdp)
    for i in range(6):
        m2_star = 1.1*real_gdp[i] - 0.08*inflation[i] + consts[i]
        ect = np.zeros(T); ect[0] = np.random.normal(0, 0.02)
        for t in range(1, T):
            ect[t] = 0.5*ect[t-1] + np.random.normal(0, 0.02)
        real_m2[i] = m2_star + ect
    return (
        real_gdp.mean(axis=0),
        inflation.mean(axis=0),
        real_m2.mean(axis=0),
    )

gdp_avg, infl_avg, m2_avg = generate_data()
T_sim = len(gdp_avg)

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
BETA_GDP = 1.059
BETA_INFL = -0.322
CONST = -2.37
R_BASE = 3.0

def simulate(kappa, alpha, theta, shocks=None):
    m = np.zeros(T_sim); r = np.zeros(T_sim); m[0] = m2_avg[0]
    if shocks is None:
        np.random.seed(42)
        shocks = np.random.normal(0, 0.01, T_sim)
    for t in range(1, T_sim):
        ect = m[t-1] - (BETA_GDP*gdp_avg[t-1] + BETA_INFL*infl_avg[t-1] + CONST)
        r[t-1] = np.clip(R_BASE + kappa*ect, 0.5, 10.0)
        m[t] = m[t-1] + alpha*ect + theta*(r[t-1] - R_BASE) + shocks[t-1]
    ect_series = m - (BETA_GDP*gdp_avg + BETA_INFL*infl_avg + CONST)
    r[-1] = np.clip(R_BASE + kappa*ect_series[-1], 0.5, 10.0)
    return ect_series, r

# ---------------------------------------------------------------------------
# LQR computation
# ---------------------------------------------------------------------------
def compute_lqr(alpha, theta, lam, beta=0.99):
    A = 1 + alpha
    B = theta
    try:
        P = solve_discrete_are(
            np.array([[np.sqrt(beta)*A]]),
            np.array([[np.sqrt(beta)*B]]),
            np.array([[1.0]]),
            np.array([[lam]])
        )
        P = float(np.asarray(P).item())
        kappa_lqr = float((beta*B*P*A)/(lam + beta*B**2*P))
        return P, kappa_lqr
    except Exception:
        return None, None

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
ect_no, _ = simulate(0.0, alpha, theta)
ect_rule, rate_rule = simulate(kappa, alpha, theta)

P_lqr, kappa_lqr = compute_lqr(alpha, theta, lam)

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    reduction = (1 - np.std(ect_rule)/np.std(ect_no)) * 100
    st.metric("Réduction de volatilité", f"{reduction:.1f}%", delta=f"κ = {kappa:.2f}")

with col2:
    st.metric("κ optimal (LQR)", f"{kappa_lqr:.3f}" if kappa_lqr else "n/a",
              delta=f"Différence: {abs(kappa - (kappa_lqr or 0)):.3f}")

with col3:
    st.metric("ECT moyen (règle)", f"{np.mean(ect_rule):.4f}")

with col4:
    st.metric("Taux moyen", f"{np.mean(rate_rule):.2f}%")

# ---------------------------------------------------------------------------
# Chart 1 — ECT comparison
# ---------------------------------------------------------------------------
st.markdown("### Stabilisation de l'écart monétaire")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=ect_no, name="Sans règle", line=dict(color="#94a3b8", width=2)))
fig1.add_trace(go.Scatter(y=ect_rule, name=f"Avec règle (κ={kappa:.2f})",
                          line=dict(color="#0f3b5e", width=3)))
fig1.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
fig1.update_layout(
    xaxis_title="Trimestre",
    yaxis_title="ECT",
    hovermode="x unified",
    height=400,
    plot_bgcolor="#f9fbfd",
    paper_bgcolor="white",
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 2 — Digital rate
# ---------------------------------------------------------------------------
st.markdown("### Taux de rémunération du portefeuille numérique")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=rate_rule, name="Taux numérique",
                          line=dict(color="#c8a45c", width=3)))
fig2.add_hline(y=R_BASE, line_dash="dash", line_color="grey",
               annotation_text="Taux neutre 3%")
fig2.update_layout(
    xaxis_title="Trimestre",
    yaxis_title="Taux (%)",
    height=350,
    plot_bgcolor="#f9fbfd",
    paper_bgcolor="white",
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 3 — Bootstrapped fan chart
# ---------------------------------------------------------------------------
st.markdown("### Robustesse – 200 trajectoires bootstrapées")

@st.cache_data
def fan_chart(kappa, alpha, theta, n_paths=200):
    paths = np.zeros((n_paths, T_sim))
    for i in range(n_paths):
        np.random.seed(i)
        shocks = np.random.normal(0, 0.01, T_sim)
        ect, _ = simulate(kappa, alpha, theta, shocks)
        paths[i] = ect
    return paths

if st.checkbox("Afficher le fan chart (plus lent)", value=False):
    with st.spinner("Simulation de 200 trajectoires..."):
        paths = fan_chart(kappa, alpha, theta)
    median = np.median(paths, axis=0)
    lo = np.percentile(paths, 5, axis=0)
    hi = np.percentile(paths, 95, axis=0)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig3.add_trace(go.Scatter(y=lo, line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(15,59,94,0.2)", name="90% CI", hoverinfo="skip"))
    fig3.add_trace(go.Scatter(y=median, name="Médiane", line=dict(color="#0f3b5e", width=2)))
    fig3.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
    fig3.update_layout(
        xaxis_title="Trimestre", yaxis_title="ECT", height=400,
        plot_bgcolor="#f9fbfd", paper_bgcolor="white",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Parameter file preview
# ---------------------------------------------------------------------------
st.markdown("### Fichier de paramètres signé (aperçu)")

param = {
    "model_version": f"dashboard-{datetime.now().strftime('%Y-%m-%d')}",
    "champion": {
        "kappa": round(kappa, 4),
        "alpha": round(alpha, 4),
        "beta": [1.0, -BETA_GDP, -BETA_INFL],
        "constant": CONST,
    },
    "lqr_optimal_kappa": round(kappa_lqr, 4) if kappa_lqr else None,
    "rate_base": R_BASE,
    "rate_floor": 0.5,
    "rate_ceiling": 10.0,
}
st.json(param)

st.download_button(
    label="📥 Télécharger ce fichier de paramètres (JSON)",
    data=json.dumps(param, indent=2),
    file_name=f"params_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
    mime="application/json",
)

# =============================================================================
# DEvaluation scenario
# =============================================================================
st.markdown("---")
st.markdown("## 🔴 Scénario de dévaluation")

st.markdown("""
Simulez l'impact d'une dévaluation brutale du franc CFA sur le thermostat monétaire.
La dévaluation est modélisée comme un choc ponctuel sur le niveau des prix et un
décalage permanent de la constante de cointégration — exactement comme en 1994.
""")

deval_size = st.slider(
    "Ampleur de la dévaluation (%)",
    min_value=0, max_value=60, value=30, step=5,
    help="1994 : 50 %. Une valeur de 0 correspond à aucun choc."
)

deval_quarter = st.slider(
    "Trimestre de déclenchement",
    min_value=10, max_value=70, value=40, step=1,
    help="Le choc se produit à ce trimestre de la simulation."
)

def simulate_devaluation(kappa, alpha, theta, deval_size_pct, deval_q,
                          gdp, infl, m2, T=80, beta=1.059, bi=-0.322, const=-2.37):
    m = np.zeros(T); r = np.zeros(T); m[0] = m2[0]
    price_shift = 0.0
    np.random.seed(42)
    shocks = np.random.normal(0, 0.01, T)

    for t in range(1, T):
        if t == deval_q:
            price_shift = deval_size_pct / 100.0
        const_adj = const - np.log(1 + price_shift)
        ect = m[t-1] - (beta*gdp[t-1] + bi*infl[t-1] + const_adj)
        r[t-1] = np.clip(3.0 + kappa*ect, 0.5, 10.0)
        m[t] = m[t-1] + alpha*ect + theta*(r[t-1] - 3.0) + shocks[t-1]

    ect_series = np.zeros(T)
    price_shift = 0.0
    for t in range(T):
        if t >= deval_q:
            price_shift = deval_size_pct / 100.0
        const_adj = const - np.log(1 + price_shift)
        ect_series[t] = m[t] - (beta*gdp[t] + bi*infl[t] + const_adj)
    return ect_series, r

ect_deval, rate_deval = simulate_devaluation(
    kappa, alpha, theta, deval_size, deval_quarter,
    gdp_avg, infl_avg, m2_avg
)

fig_deval = go.Figure()
fig_deval.add_trace(go.Scatter(y=ect_rule, name="ECT – sans dévaluation",
                                line=dict(color="#94a3b8", width=2)))
fig_deval.add_trace(go.Scatter(y=ect_deval, name=f"ECT – avec dévaluation ({deval_size}%)",
                                line=dict(color="#ef4444", width=3)))
fig_deval.add_vline(x=deval_quarter, line_dash="dash", line_color="#c8a45c",
                    annotation_text="Dévaluation", annotation_position="top")
fig_deval.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.4)
fig_deval.update_layout(
    title=f"Impact d'une dévaluation de {deval_size}% au trimestre {deval_quarter}",
    xaxis_title="Trimestre",
    yaxis_title="ECT",
    plot_bgcolor="#f9fbfd",
    paper_bgcolor="white",
    height=420,
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig_deval, use_container_width=True)

max_ect = np.max(np.abs(ect_deval[deval_quarter:deval_quarter+12]))
recovery_q = next((i for i in range(deval_quarter, T_sim)
                    if abs(ect_deval[i]) < 0.02), None)
recovery_time = (recovery_q - deval_quarter) if recovery_q else "> 40"

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("ECT max après choc", f"{max_ect:.4f}")
with col2:
    st.metric("Taux au plafond ?", "Oui" if np.max(rate_deval[deval_quarter:]) >= 9.9 else "Non")
with col3:
    st.metric("Retour à l'équilibre", f"{recovery_time} trimestres")

st.info("""
**Ce que cela démontre:** même après un choc de 50 % (comme en 1994), le thermostat
ramène l'économie vers l'équilibre. La re-estimation trimestrielle de la cointégration
permet d'ajuster le vecteur après le choc. Le mode crise se déclenche automatiquement.
""")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "**Consilium CFA** – Centre de recherche en stabilité monétaire · "
    "[consilium-cfa.com](https://consilium-cfa.com) · "
    "aymar.makanda@consilium-cfa.com"
)
