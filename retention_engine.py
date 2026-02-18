"""
=======================================================
  THE RETENTION ENGINE
  Predictive Customer Churn & Lifetime Value Dashboard
=======================================================
Author  : Data Analytics Project
Version : 1.0
Python  : 3.8+
Packages: pandas, numpy, scikit-learn, matplotlib, seaborn

HOW TO RUN:
    python retention_engine.py

OUTPUT:
    - Console summary & model metrics
    - retention_dashboard.png  (visual dashboard)
    - churn_report.csv         (at-risk customers)
=======================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, recall_score,
                             precision_score, f1_score, accuracy_score)

# ─────────────────────────────────────────────
#  STEP 1: GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────

def generate_dataset(n=1500, seed=42):
    """Generate realistic synthetic Telco-style customer data."""
    np.random.seed(seed)
    n = n

    # Demographics
    age           = np.random.randint(18, 70, n)
    gender        = np.random.choice(['Male', 'Female'], n)
    location      = np.random.choice(['Mumbai', 'Delhi', 'Bengaluru', 'Pune', 'Chennai'], n)

    # Subscription
    contract      = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                      n, p=[0.55, 0.25, 0.20])
    tenure        = np.where(contract == 'Month-to-month',
                             np.random.randint(1, 24, n),
                             np.where(contract == 'One year',
                                      np.random.randint(6, 36, n),
                                      np.random.randint(12, 72, n)))
    monthly_charge = np.round(
        np.random.normal(65, 20, n).clip(20, 120) +
        (contract == 'Month-to-month') * 10, 2)
    internet_type  = np.random.choice(['Fiber optic', 'DSL', 'No internet'], n, p=[0.45, 0.40, 0.15])
    support_tickets = np.random.poisson(1.5, n)
    payment_method  = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n)

    # Churn logic (realistic rules)
    churn_prob = (
        0.05
        + 0.30 * (contract == 'Month-to-month')
        + 0.10 * (internet_type == 'Fiber optic')
        + 0.008 * support_tickets
        - 0.004 * tenure
        + 0.15 * (payment_method == 'Electronic check')
        + 0.01 * np.random.randn(n)
    ).clip(0, 1)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    # Customer Lifetime Value (LTV)
    ltv = np.round(
        monthly_charge * tenure * (1 - 0.3 * churn) +
        np.random.normal(0, 50, n), 2).clip(0)

    df = pd.DataFrame({
        'CustomerID'     : [f'CUST{str(i).zfill(4)}' for i in range(n)],
        'Age'            : age,
        'Gender'         : gender,
        'Location'       : location,
        'Contract'       : contract,
        'Tenure'         : tenure,
        'MonthlyCharge'  : monthly_charge,
        'InternetType'   : internet_type,
        'SupportTickets' : support_tickets,
        'PaymentMethod'  : payment_method,
        'LTV'            : ltv,
        'Churn'          : churn
    })
    return df


# ─────────────────────────────────────────────
#  STEP 2: EDA HELPER
# ─────────────────────────────────────────────

def run_eda(df):
    print("\n" + "="*55)
    print("   EXPLORATORY DATA ANALYSIS")
    print("="*55)
    print(f"  Total Customers   : {len(df):,}")
    print(f"  Churned           : {df['Churn'].sum():,}  ({df['Churn'].mean()*100:.1f}%)")
    print(f"  Retained          : {(df['Churn']==0).sum():,}  ({(1-df['Churn'].mean())*100:.1f}%)")
    print(f"  Avg Monthly Charge: ₹{df['MonthlyCharge'].mean():.2f}")
    print(f"  Avg Tenure        : {df['Tenure'].mean():.1f} months")
    print(f"  Avg LTV           : ₹{df['LTV'].mean():,.2f}")

    print("\n  Churn by Contract Type:")
    ct = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
    for k, v in ct.items():
        bar = "█" * int(v * 40)
        print(f"    {k:<20} {bar} {v*100:.1f}%")

    print("\n  Churn by Internet Type:")
    it = df.groupby('InternetType')['Churn'].mean().sort_values(ascending=False)
    for k, v in it.items():
        bar = "█" * int(v * 40)
        print(f"    {k:<20} {bar} {v*100:.1f}%")


# ─────────────────────────────────────────────
#  STEP 3: PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    df2 = df.copy()
    cat_cols = ['Gender', 'Location', 'Contract', 'InternetType', 'PaymentMethod']
    le = LabelEncoder()
    for col in cat_cols:
        df2[col] = le.fit_transform(df2[col])

    feature_cols = ['Age', 'Gender', 'Location', 'Contract', 'Tenure',
                    'MonthlyCharge', 'InternetType', 'SupportTickets', 'PaymentMethod']
    X = df2[feature_cols]
    y = df2['Churn']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    return X_scaled, y, feature_cols


# ─────────────────────────────────────────────
#  STEP 4: MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',   # handles class imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
#  STEP 5: EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    rec     = recall_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)

    print("\n" + "="*55)
    print("   MODEL EVALUATION  (Random Forest)")
    print("="*55)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%   ← Most important for churn!")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

    return y_pred, y_proba, auc


# ─────────────────────────────────────────────
#  STEP 6: AT-RISK REPORT
# ─────────────────────────────────────────────

def generate_report(df, X_scaled, model):
    proba = model.predict_proba(X_scaled)[:, 1]
    df2 = df.copy()
    df2['ChurnProbability'] = np.round(proba * 100, 2)
    df2['RiskSegment'] = pd.cut(
        df2['ChurnProbability'],
        bins=[0, 30, 60, 100],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    at_risk = df2[df2['RiskSegment'] == 'High Risk'].sort_values(
        'ChurnProbability', ascending=False)

    report_path = 'churn_report.csv'
    at_risk[['CustomerID', 'Contract', 'Tenure', 'MonthlyCharge',
             'LTV', 'ChurnProbability', 'RiskSegment']].to_csv(report_path, index=False)

    print("\n" + "="*55)
    print("   AT-RISK CUSTOMER REPORT")
    print("="*55)
    print(f"  High Risk Customers : {len(at_risk):,}")
    print(f"  Revenue at Risk     : ₹{at_risk['LTV'].sum():,.2f}")
    print(f"  Avg Churn Prob      : {at_risk['ChurnProbability'].mean():.1f}%")
    print(f"\n  Top 5 Highest-Risk Customers:")
    print(at_risk[['CustomerID', 'Contract', 'MonthlyCharge',
                   'LTV', 'ChurnProbability']].head().to_string(index=False))
    print(f"\n  Report saved → churn_report.csv")

    return df2


# ─────────────────────────────────────────────
#  STEP 7: DASHBOARD (6-PANEL VISUALIZATION)
# ─────────────────────────────────────────────

def build_dashboard(df, df_scored, model, feature_cols, X_test, y_test, y_pred, y_proba, auc):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0F1117')

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    ACCENT   = '#00D4FF'
    RED      = '#FF4B4B'
    GREEN    = '#00C48C'
    YELLOW   = '#FFD166'
    BG_PANEL = '#1A1D2E'
    TEXT     = '#E0E0E0'

    panel_style = dict(facecolor=BG_PANEL)

    # ── Title bar ──────────────────────────────
    fig.text(0.5, 0.97, '🔮  THE RETENTION ENGINE',
             ha='center', va='top', fontsize=22, fontweight='bold',
             color=ACCENT, fontfamily='monospace')
    fig.text(0.5, 0.945, 'Predictive Customer Churn & Lifetime Value Dashboard',
             ha='center', va='top', fontsize=12, color=TEXT)

    # ── Panel 1: Churn by Contract ──────────────
    ax1 = fig.add_subplot(gs[0, 0], **panel_style)
    contract_churn = df.groupby('Contract')['Churn'].mean() * 100
    colors = [RED if v > 30 else YELLOW if v > 15 else GREEN
              for v in contract_churn.values]
    bars = ax1.bar(contract_churn.index, contract_churn.values, color=colors, edgecolor='none', width=0.55)
    ax1.set_facecolor(BG_PANEL)
    ax1.set_title('Churn Rate by Contract Type', color=TEXT, fontsize=11, pad=10)
    ax1.set_ylabel('Churn Rate (%)', color=TEXT, fontsize=9)
    ax1.tick_params(colors=TEXT, labelsize=8)
    ax1.spines[:].set_visible(False)
    for bar, val in zip(bars, contract_churn.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', color=TEXT, fontsize=9, fontweight='bold')

    # ── Panel 2: Feature Importance ────────────
    ax2 = fig.add_subplot(gs[0, 1], **panel_style)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    colors_fi = [ACCENT if i >= len(importances)-3 else '#5A5F7D' for i in range(len(importances))]
    importances.plot(kind='barh', ax=ax2, color=colors_fi, edgecolor='none')
    ax2.set_facecolor(BG_PANEL)
    ax2.set_title('Feature Importance (Why Customers Leave)', color=TEXT, fontsize=11, pad=10)
    ax2.set_xlabel('Importance Score', color=TEXT, fontsize=9)
    ax2.tick_params(colors=TEXT, labelsize=8)
    ax2.spines[:].set_visible(False)

    # ── Panel 3: ROC Curve ─────────────────────
    ax3 = fig.add_subplot(gs[0, 2], **panel_style)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax3.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f'AUC = {auc:.3f}')
    ax3.plot([0, 1], [0, 1], color='#5A5F7D', lw=1, linestyle='--', label='Random')
    ax3.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
    ax3.set_facecolor(BG_PANEL)
    ax3.set_title('ROC Curve – Model Performance', color=TEXT, fontsize=11, pad=10)
    ax3.set_xlabel('False Positive Rate', color=TEXT, fontsize=9)
    ax3.set_ylabel('True Positive Rate', color=TEXT, fontsize=9)
    ax3.tick_params(colors=TEXT, labelsize=8)
    ax3.spines[:].set_visible(False)
    ax3.legend(facecolor=BG_PANEL, labelcolor=TEXT, fontsize=9)

    # ── Panel 4: Churn Probability Distribution ─
    ax4 = fig.add_subplot(gs[1, 0], **panel_style)
    churn_probs = df_scored['ChurnProbability']
    ax4.hist(churn_probs[df_scored['Churn']==0], bins=30, alpha=0.7, color=GREEN, label='Retained')
    ax4.hist(churn_probs[df_scored['Churn']==1], bins=30, alpha=0.7, color=RED, label='Churned')
    ax4.set_facecolor(BG_PANEL)
    ax4.set_title('Predicted Churn Probability Distribution', color=TEXT, fontsize=11, pad=10)
    ax4.set_xlabel('Churn Probability (%)', color=TEXT, fontsize=9)
    ax4.set_ylabel('Count', color=TEXT, fontsize=9)
    ax4.tick_params(colors=TEXT, labelsize=8)
    ax4.spines[:].set_visible(False)
    ax4.legend(facecolor=BG_PANEL, labelcolor=TEXT, fontsize=9)

    # ── Panel 5: LTV by Risk Segment ───────────
    ax5 = fig.add_subplot(gs[1, 1], **panel_style)
    risk_ltv = df_scored.groupby('RiskSegment')['LTV'].mean()
    seg_colors = [GREEN, YELLOW, RED]
    bars5 = ax5.bar(risk_ltv.index, risk_ltv.values, color=seg_colors, edgecolor='none', width=0.55)
    ax5.set_facecolor(BG_PANEL)
    ax5.set_title('Avg Customer LTV by Risk Segment', color=TEXT, fontsize=11, pad=10)
    ax5.set_ylabel('Avg LTV (₹)', color=TEXT, fontsize=9)
    ax5.tick_params(colors=TEXT, labelsize=8)
    ax5.spines[:].set_visible(False)
    for bar, val in zip(bars5, risk_ltv.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'₹{val:,.0f}', ha='center', color=TEXT, fontsize=9, fontweight='bold')

    # ── Panel 6: Confusion Matrix ───────────────
    ax6 = fig.add_subplot(gs[1, 2], **panel_style)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'],
                linewidths=1, linecolor=BG_PANEL,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax6.set_facecolor(BG_PANEL)
    ax6.set_title('Confusion Matrix', color=TEXT, fontsize=11, pad=10)
    ax6.set_xlabel('Predicted', color=TEXT, fontsize=9)
    ax6.set_ylabel('Actual', color=TEXT, fontsize=9)
    ax6.tick_params(colors=TEXT, labelsize=9)

    # ── Panel 7 (wide): Tenure vs Monthly Charge scatter ──
    ax7 = fig.add_subplot(gs[2, :2], **panel_style)
    sample = df_scored.sample(500, random_state=1)
    scatter = ax7.scatter(
        sample['Tenure'], sample['MonthlyCharge'],
        c=sample['ChurnProbability'], cmap='RdYlGn_r',
        alpha=0.7, s=40, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('Churn Probability (%)', color=TEXT, fontsize=8)
    cbar.ax.tick_params(colors=TEXT)
    ax7.set_facecolor(BG_PANEL)
    ax7.set_title('Tenure vs Monthly Charge — Colored by Churn Risk', color=TEXT, fontsize=11, pad=10)
    ax7.set_xlabel('Tenure (months)', color=TEXT, fontsize=9)
    ax7.set_ylabel('Monthly Charge (₹)', color=TEXT, fontsize=9)
    ax7.tick_params(colors=TEXT, labelsize=8)
    ax7.spines[:].set_visible(False)

    # ── Panel 8: KPI Summary Box ────────────────
    ax8 = fig.add_subplot(gs[2, 2], **panel_style)
    ax8.set_facecolor(BG_PANEL)
    ax8.axis('off')
    ax8.set_title('Key Metrics Summary', color=TEXT, fontsize=11, pad=10)

    high_risk = df_scored[df_scored['RiskSegment'] == 'High Risk']
    kpis = [
        ("Total Customers",     f"{len(df_scored):,}",                         ACCENT),
        ("Churn Rate",          f"{df_scored['Churn'].mean()*100:.1f}%",        RED),
        ("High Risk Count",     f"{len(high_risk):,}",                          YELLOW),
        ("Revenue at Risk",     f"₹{high_risk['LTV'].sum():,.0f}",              RED),
        ("Model Recall",        f"{recall_score(y_test, y_pred)*100:.1f}%",     GREEN),
        ("ROC-AUC Score",       f"{auc:.3f}",                                   ACCENT),
    ]
    for i, (label, value, color) in enumerate(kpis):
        y_pos = 0.88 - i * 0.155
        ax8.text(0.05, y_pos, label, transform=ax8.transAxes,
                 color='#AAAAAA', fontsize=9)
        ax8.text(0.95, y_pos - 0.05, value, transform=ax8.transAxes,
                 color=color, fontsize=13, fontweight='bold', ha='right')

    out_path = 'retention_dashboard.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print(f"\n  Dashboard saved → retention_dashboard.png")
    return out_path


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔮  THE RETENTION ENGINE — Starting Pipeline...\n")

    # 1. Data
    print("  [1/5] Generating synthetic customer dataset...")
    df = generate_dataset(n=1500)

    # 2. EDA
    print("  [2/5] Running Exploratory Data Analysis...")
    run_eda(df)

    # 3. Preprocess
    print("\n  [3/5] Preprocessing data...")
    X, y, feature_cols = preprocess(df)
    print(f"        Features: {feature_cols}")
    print(f"        Samples : {len(X):,}")

    # 4. Train
    print("  [4/5] Training Random Forest Classifier...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    y_pred, y_proba, auc = evaluate(model, X_test, y_test)

    # 5. Report
    print("  [5/5] Generating at-risk report & dashboard...")
    df_scored = generate_report(df, X, model)
    build_dashboard(df, df_scored, model, feature_cols,
                    X_test, y_test, y_pred, y_proba, auc)

    print("\n" + "="*55)
    print("  ✅  Pipeline Complete! Outputs:")
    print("      • retention_dashboard.png")
    print("      • churn_report.csv")
    print("="*55 + "\n")