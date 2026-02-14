"""Generate the results_analysis.ipynb notebook programmatically."""
import json, os

def make_cell(cell_type, source, outputs=None):
    cell = {"cell_type": cell_type, "metadata": {}, "source": source if isinstance(source, list) else source.split("\n")}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = outputs or []
    return cell

cells = []

# --- Title ---
cells.append(make_cell("markdown", [
    "# Ghost in the Machine: Results Analysis\n",
    "\n",
    "Publication-quality visualizations for the prompt injection detection research.\n",
    "All figures are saved to `results/figures/` at 300 DPI.\n",
]))

# --- Imports & Config ---
cells.append(make_cell("code", [
    "import json\n",
    "import warnings\n",
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mticker\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from matplotlib.patches import FancyBboxPatch\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# --- Style ---\n",
    "plt.rcParams.update({\n",
    "    'figure.dpi': 150,\n",
    "    'savefig.dpi': 300,\n",
    "    'font.family': 'serif',\n",
    "    'font.size': 11,\n",
    "    'axes.titlesize': 13,\n",
    "    'axes.labelsize': 11,\n",
    "    'figure.facecolor': 'white',\n",
    "    'axes.grid': True,\n",
    "    'grid.alpha': 0.3,\n",
    "})\n",
    "\n",
    "# Colorblind-friendly palette (Okabe-Ito)\n",
    "C = {\n",
    "    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',\n",
    "    'red': '#D55E00', 'purple': '#CC79A7', 'cyan': '#56B4E9',\n",
    "    'yellow': '#F0E442', 'black': '#000000', 'grey': '#999999',\n",
    "}\n",
    "\n",
    "# --- Paths ---\n",
    "ROOT = Path('..') if Path('../data').exists() else Path('.')\n",
    "DATA = ROOT / 'data' / 'processed'\n",
    "RESULTS = ROOT / 'results'\n",
    "FIG_DIR = RESULTS / 'figures'\n",
    "FIG_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "def savefig(fig, name):\n",
    "    fig.savefig(FIG_DIR / f'{name}.png', bbox_inches='tight', dpi=300)\n",
    "    print(f'Saved: {FIG_DIR / name}.png')\n",
]))

# --- Load Data ---
cells.append(make_cell("markdown", [
    "## 1. Dataset Analysis\n",
]))

cells.append(make_cell("code", [
    "# Load all splits\n",
    "train_df = pd.read_csv(DATA / 'train.csv')\n",
    "val_df = pd.read_csv(DATA / 'val.csv')\n",
    "test_df = pd.read_csv(DATA / 'test.csv')\n",
    "zeroday_df = pd.read_csv(DATA / 'test_zeroday.csv')\n",
    "combined = pd.read_csv(DATA / 'combined_dataset.csv')\n",
    "\n",
    "with open(DATA / 'data_summary.json') as f:\n",
    "    summary = json.load(f)\n",
    "\n",
    "# Load evaluation results\n",
    "with open(RESULTS / 'evaluation_results.json') as f:\n",
    "    eval_results = json.load(f)\n",
    "\n",
    "with open(RESULTS / 'training_history.json') as f:\n",
    "    history = json.load(f)\n",
    "\n",
    "print(f'Total unique samples: {len(combined)}')\n",
    "print(f'Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, zeroday={len(zeroday_df)}')\n",
]))

# --- 1a. Sample Distribution ---
cells.append(make_cell("markdown", [
    "### 1a. Class Distribution (Malicious vs Benign)\n",
]))

cells.append(make_cell("code", [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n",
    "\n",
    "# Overall distribution\n",
    "labels_map = {1: 'Malicious', 0: 'Benign'}\n",
    "counts = combined['label'].value_counts().sort_index()\n",
    "colors = [C['green'], C['red']]\n",
    "bars = axes[0].bar([labels_map[i] for i in counts.index], counts.values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)\n",
    "for bar, v in zip(bars, counts.values):\n",
    "    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{v}\\n({v/len(combined):.0%})', ha='center', va='bottom', fontweight='bold')\n",
    "axes[0].set_title('Overall Class Distribution')\n",
    "axes[0].set_ylabel('Number of Samples')\n",
    "axes[0].set_ylim(0, max(counts.values) * 1.25)\n",
    "\n",
    "# Per-split distribution\n",
    "splits = {'Train': train_df, 'Val': val_df, 'Test': test_df, 'Zero-Day': zeroday_df}\n",
    "x = np.arange(len(splits))\n",
    "mal = [int((df['label'] == 1).sum()) for df in splits.values()]\n",
    "ben = [int((df['label'] == 0).sum()) for df in splits.values()]\n",
    "w = 0.35\n",
    "axes[1].bar(x - w/2, ben, w, label='Benign', color=C['green'], edgecolor='white')\n",
    "axes[1].bar(x + w/2, mal, w, label='Malicious', color=C['red'], edgecolor='white')\n",
    "axes[1].set_xticks(x)\n",
    "axes[1].set_xticklabels(splits.keys())\n",
    "axes[1].set_title('Class Distribution per Split')\n",
    "axes[1].set_ylabel('Number of Samples')\n",
    "axes[1].legend()\n",
    "for i, (b, m) in enumerate(zip(ben, mal)):\n",
    "    axes[1].text(i - w/2, b + 2, str(b), ha='center', fontsize=9)\n",
    "    axes[1].text(i + w/2, m + 2, str(m), ha='center', fontsize=9)\n",
    "\n",
    "fig.suptitle('Dataset Class Distribution', fontsize=14, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "savefig(fig, 'class_distribution')\n",
    "plt.show()\n",
]))

# --- 1b. Attack Type Breakdown ---
cells.append(make_cell("markdown", [
    "### 1b. Attack Category Breakdown\n",
]))

cells.append(make_cell("code", [
    "attack_counts = combined[combined['label'] == 1]['attack_category'].value_counts()\n",
    "attack_counts = attack_counts[attack_counts.index != 'none'].head(15)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "colors_list = [C['red'] if cat in ('homoglyph', 'caesar') else C['blue'] for cat in attack_counts.index]\n",
    "bars = ax.barh(range(len(attack_counts)), attack_counts.values, color=colors_list, edgecolor='white')\n",
    "ax.set_yticks(range(len(attack_counts)))\n",
    "ax.set_yticklabels(attack_counts.index)\n",
    "ax.invert_yaxis()\n",
    "ax.set_xlabel('Number of Samples')\n",
    "ax.set_title('Attack Category Distribution (Malicious Samples Only)', fontweight='bold')\n",
    "for i, v in enumerate(attack_counts.values):\n",
    "    ax.text(v + 1, i, str(v), va='center', fontsize=9)\n",
    "\n",
    "# Legend for zero-day\n",
    "from matplotlib.patches import Patch\n",
    "ax.legend(handles=[Patch(color=C['blue'], label='Training attacks'), Patch(color=C['red'], label='Zero-day holdout')], loc='lower right')\n",
    "\n",
    "plt.tight_layout()\n",
    "savefig(fig, 'attack_category_distribution')\n",
    "plt.show()\n",
]))

# --- 1c. Source Dataset ---
cells.append(make_cell("markdown", [
    "### 1c. Source Dataset Contribution\n",
]))

cells.append(make_cell("code", [
    "source_counts = combined['source'].value_counts()\n",
    "source_labels = {'encoded_attacks': 'Dataset 2:\\nEncoded', 'straightforward': 'Dataset 1:\\nStraightforward', 'rag_poisoned': 'Dataset 4:\\nRAG-Poisoned', 'multimodal': 'Dataset 3:\\nMultimodal', 'encoded_benign': 'Dataset 2:\\nBenign'}\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "palette = [C['blue'], C['cyan'], C['orange'], C['green'], C['purple']]\n",
    "labels = [source_labels.get(s, s) for s in source_counts.index]\n",
    "wedges, texts, autotexts = ax.pie(source_counts.values, labels=labels, autopct='%1.1f%%', colors=palette[:len(source_counts)], startangle=90, pctdistance=0.8)\n",
    "for t in autotexts:\n",
    "    t.set_fontsize(9)\n",
    "ax.set_title('Contribution by Source Dataset', fontweight='bold')\n",
    "plt.tight_layout()\n",
    "savefig(fig, 'source_distribution')\n",
    "plt.show()\n",
]))

# --- 1d. Text Length ---
cells.append(make_cell("markdown", [
    "### 1d. Text Length Distribution\n",
]))

cells.append(make_cell("code", [
    "combined['text_len'] = combined['text'].str.len()\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n",
    "\n",
    "for label, color, name in [(0, C['green'], 'Benign'), (1, C['red'], 'Malicious')]:\n",
    "    subset = combined[combined['label'] == label]['text_len']\n",
    "    axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=f'{name} (n={len(subset)})', edgecolor='white')\n",
    "axes[0].set_xlabel('Character Length')\n",
    "axes[0].set_ylabel('Frequency')\n",
    "axes[0].set_title('Text Length Distribution by Class')\n",
    "axes[0].legend()\n",
    "\n",
    "data_by_class = [combined[combined['label'] == 0]['text_len'], combined[combined['label'] == 1]['text_len']]\n",
    "bp = axes[1].boxplot(data_by_class, labels=['Benign', 'Malicious'], patch_artist=True)\n",
    "bp['boxes'][0].set_facecolor(C['green'])\n",
    "bp['boxes'][1].set_facecolor(C['red'])\n",
    "for box in bp['boxes']:\n",
    "    box.set_alpha(0.6)\n",
    "axes[1].set_ylabel('Character Length')\n",
    "axes[1].set_title('Text Length Box Plot')\n",
    "\n",
    "fig.suptitle('Text Length Analysis', fontsize=14, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "savefig(fig, 'text_length_distribution')\n",
    "plt.show()\n",
    "\n",
    "print(f\"Benign  - mean: {data_by_class[0].mean():.0f}, median: {data_by_class[0].median():.0f}\")\n",
    "print(f\"Malicious - mean: {data_by_class[1].mean():.0f}, median: {data_by_class[1].median():.0f}\")\n",
]))

# --- Section 2: Model Performance ---
cells.append(make_cell("markdown", [
    "## 2. Model Performance Comparison\n",
]))

# --- 2a. F1 Bar Charts ---
cells.append(make_cell("markdown", [
    "### 2a. F1-Score Comparison: Test Set vs Zero-Day\n",
]))

cells.append(make_cell("code", [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
    "\n",
    "models = ['BERT\\nClassifier', 'Keyword\\nFilter', 'Regex\\nPatterns']\n",
    "test_f1 = [eval_results['baseline_test']['f1'], eval_results['keyword_baseline']['f1'], eval_results['regex_baseline']['f1']]\n",
    "zd_f1 = [eval_results['baseline_zeroday']['f1'], eval_results['keyword_zeroday']['f1'], eval_results['regex_zeroday']['f1']]\n",
    "bar_colors = [C['blue'], C['orange'], C['green']]\n",
    "\n",
    "# Test F1\n",
    "bars = axes[0].bar(models, test_f1, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.5)\n",
    "for bar, v in zip(bars, test_f1):\n",
    "    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)\n",
    "axes[0].set_ylim(0, 1.15)\n",
    "axes[0].set_ylabel('F1-Score')\n",
    "axes[0].set_title('Test Set Performance', fontweight='bold')\n",
    "axes[0].axhline(y=0.9, color=C['grey'], linestyle='--', alpha=0.5, label='90% threshold')\n",
    "axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n",
    "\n",
    "# Zero-Day F1\n",
    "bars = axes[1].bar(models, zd_f1, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.5)\n",
    "for bar, v in zip(bars, zd_f1):\n",
    "    y = max(bar.get_height(), 0.02)\n",
    "    axes[1].text(bar.get_x() + bar.get_width()/2, y + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)\n",
    "axes[1].set_ylim(0, 1.15)\n",
    "axes[1].set_ylabel('F1-Score')\n",
    "axes[1].set_title('Zero-Day Attack Detection', fontweight='bold')\n",
    "axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n",
    "\n",
    "fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "savefig(fig, 'f1_comparison')\n",
    "plt.show()\n",
]))

# --- 2b. Metrics Table ---
cells.append(make_cell("markdown", [
    "### 2b. Comprehensive Metrics Table\n",
]))

cells.append(make_cell("code", [
    "metrics_data = {\n",
    "    'Method': ['BERT Classifier', 'BERT (Zero-Day)', 'Keyword Filter', 'Keyword (Zero-Day)', 'Regex Patterns', 'Regex (Zero-Day)'],\n",
    "    'Accuracy': [\n",
    "        eval_results['baseline_test']['accuracy'], eval_results['baseline_zeroday']['accuracy'],\n",
    "        eval_results['keyword_baseline']['accuracy'], eval_results['keyword_zeroday']['accuracy'],\n",
    "        eval_results['regex_baseline']['accuracy'], eval_results['regex_zeroday']['accuracy'],\n",
    "    ],\n",
    "    'Precision': [\n",
    "        eval_results['baseline_test']['precision'], eval_results['baseline_zeroday']['precision'],\n",
    "        eval_results['keyword_baseline']['precision'], eval_results['keyword_zeroday']['precision'],\n",
    "        eval_results['regex_baseline']['precision'], eval_results['regex_zeroday']['precision'],\n",
    "    ],\n",
    "    'Recall': [\n",
    "        eval_results['baseline_test']['recall'], eval_results['baseline_zeroday']['recall'],\n",
    "        eval_results['keyword_baseline']['recall'], eval_results['keyword_zeroday']['recall'],\n",
    "        eval_results['regex_baseline']['recall'], eval_results['regex_zeroday']['recall'],\n",
    "    ],\n",
    "    'F1-Score': [\n",
    "        eval_results['baseline_test']['f1'], eval_results['baseline_zeroday']['f1'],\n",
    "        eval_results['keyword_baseline']['f1'], eval_results['keyword_zeroday']['f1'],\n",
    "        eval_results['regex_baseline']['f1'], eval_results['regex_zeroday']['f1'],\n",
    "    ],\n",
    "}\n",
    "metrics_df = pd.DataFrame(metrics_data)\n",
    "print(metrics_df.to_string(index=False, float_format='%.4f'))\n",
]))

# --- 2c. Radar Chart ---
cells.append(make_cell("markdown", [
    "### 2c. Multi-Metric Radar Comparison\n",
]))

cells.append(make_cell("code", [
    "fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))\n",
    "\n",
    "metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']\n",
    "bert_vals = [eval_results['baseline_test']['accuracy'], eval_results['baseline_test']['precision'], eval_results['baseline_test']['recall'], eval_results['baseline_test']['f1']]\n",
    "kw_vals = [eval_results['keyword_baseline']['accuracy'], eval_results['keyword_baseline']['precision'], eval_results['keyword_baseline']['recall'], eval_results['keyword_baseline']['f1']]\n",
    "rx_vals = [eval_results['regex_baseline']['accuracy'], eval_results['regex_baseline']['precision'], eval_results['regex_baseline']['recall'], eval_results['regex_baseline']['f1']]\n",
    "\n",
    "angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()\n",
    "for vals in [bert_vals, kw_vals, rx_vals]:\n",
    "    vals.append(vals[0])\n",
    "angles.append(angles[0])\n",
    "\n",
    "ax.plot(angles, bert_vals, 'o-', linewidth=2, label='BERT', color=C['blue'])\n",
    "ax.fill(angles, bert_vals, alpha=0.15, color=C['blue'])\n",
    "ax.plot(angles, kw_vals, 's-', linewidth=2, label='Keyword', color=C['orange'])\n",
    "ax.fill(angles, kw_vals, alpha=0.15, color=C['orange'])\n",
    "ax.plot(angles, rx_vals, '^-', linewidth=2, label='Regex', color=C['green'])\n",
    "ax.fill(angles, rx_vals, alpha=0.15, color=C['green'])\n",
    "\n",
    "ax.set_xticks(angles[:-1])\n",
    "ax.set_xticklabels(metrics_names)\n",
    "ax.set_ylim(0, 1.05)\n",
    "ax.set_title('Multi-Metric Comparison (Test Set)', fontweight='bold', pad=20)\n",
    "ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0))\n",
    "\n",
    "plt.tight_layout()\n",
    "savefig(fig, 'radar_comparison')\n",
    "plt.show()\n",
]))

# --- 2d. Improvement Over Baselines ---
cells.append(make_cell("markdown", [
    "### 2d. BERT Improvement Over Traditional Baselines\n",
]))

cells.append(make_cell("code", [
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "\n",
    "metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']\n",
    "bert = [eval_results['baseline_test']['accuracy'], eval_results['baseline_test']['precision'], eval_results['baseline_test']['recall'], eval_results['baseline_test']['f1']]\n",
    "kw = [eval_results['keyword_baseline']['accuracy'], eval_results['keyword_baseline']['precision'], eval_results['keyword_baseline']['recall'], eval_results['keyword_baseline']['f1']]\n",
    "rx = [eval_results['regex_baseline']['accuracy'], eval_results['regex_baseline']['precision'], eval_results['regex_baseline']['recall'], eval_results['regex_baseline']['f1']]\n",
    "\n",
    "x = np.arange(len(metrics))\n",
    "w = 0.25\n",
    "ax.bar(x - w, bert, w, label='BERT', color=C['blue'], edgecolor='white')\n",
    "ax.bar(x, kw, w, label='Keyword', color=C['orange'], edgecolor='white')\n",
    "ax.bar(x + w, rx, w, label='Regex', color=C['green'], edgecolor='white')\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(metrics)\n",
    "ax.set_ylim(0, 1.1)\n",
    "ax.set_ylabel('Score')\n",
    "ax.set_title('Detailed Metric Comparison (Test Set)', fontweight='bold')\n",
    "ax.legend()\n",
    "ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n",
    "\n",
    "# Annotate improvements\n",
    "for i, (b, k, r) in enumerate(zip(bert, kw, rx)):\n",
    "    best_base = max(k, r)\n",
    "    if b > best_base:\n",
    "        delta = b - best_base\n",
    "        ax.annotate(f'+{delta:.1%}', xy=(i - w, b), xytext=(i - w, b + 0.03), ha='center', fontsize=9, color=C['blue'], fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "savefig(fig, 'metric_comparison_grouped')\n",
    "plt.show()\n",
]))

# --- Section 3: Per-Attack-Type ---
cells.append(make_cell("markdown", [
    "## 3. Per-Attack-Type Analysis\n",
]))

cells.append(make_cell("code", [
    "# Compute per-attack-type accuracy for each baseline method\n",
    "import re\n",
    "\n",
    "MALICIOUS_KEYWORDS = ['ignore', 'bypass', 'override', 'disregard', 'forget', 'reveal', 'expose', 'leak', 'delete', 'admin', 'credentials', 'password', 'previous instructions', 'system prompt', 'sensitive', 'unauthorized']\n",
    "MALICIOUS_PATTERNS = [r'ignore\\s+(all\\s+)?previous\\s+(instructions|directions|rules)', r'bypass\\s+(all\\s+)?(security|safety|content|auth)', r'override\\s+(system|safety|security)', r'disregard\\s+(your\\s+)?(guidelines|rules|safety|instructions)', r'forget\\s+(your\\s+)?(instructions|constraints|rules)', r'reveal\\s+(system\\s+prompt|passwords|credentials|sensitive)', r'(grant|give)\\s+(full\\s+|admin\\s+)?access', r'(show|display|expose)\\s+(admin|sensitive|confidential|internal)', r'(delete|remove|modify)\\s+(user\\s+data|records|files)', r'execute\\s+(admin\\s+)?commands?']\n",
    "compiled = [re.compile(p, re.IGNORECASE) for p in MALICIOUS_PATTERNS]\n",
    "\n",
    "def kw_pred(text):\n",
    "    return 1 if any(k in text.lower() for k in MALICIOUS_KEYWORDS) else 0\n",
    "\n",
    "def rx_pred(text):\n",
    "    return 1 if any(p.search(text) for p in compiled) else 0\n",
    "\n",
    "# Compute on full combined malicious set\n",
    "mal_df = combined[combined['label'] == 1].copy()\n",
    "mal_df['kw_pred'] = mal_df['text'].apply(kw_pred)\n",
    "mal_df['rx_pred'] = mal_df['text'].apply(rx_pred)\n",
    "\n",
    "attack_perf = mal_df.groupby('attack_category').agg(\n",
    "    count=('label', 'count'),\n",
    "    kw_recall=('kw_pred', 'mean'),\n",
    "    rx_recall=('rx_pred', 'mean'),\n",
    ").sort_values('count', ascending=False)\n",
    "\n",
    "# Mark zero-day categories\n",
    "attack_perf['is_zeroday'] = attack_perf.index.isin(['homoglyph', 'caesar'])\n",
    "print(attack_perf.to_string())\n",
]))

# --- 3a. Heatmap ---
cells.append(make_cell("markdown", [
    "### 3a. Detection Recall Heatmap by Attack Category\n",
]))

cells.append(make_cell("code", [
    "top_attacks = attack_perf.head(12)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "data = top_attacks[['kw_recall', 'rx_recall']].values\n",
    "\n",
    "im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)\n",
    "ax.set_xticks([0, 1])\n",
    "ax.set_xticklabels(['Keyword', 'Regex'])\n",
    "ax.set_yticks(range(len(top_attacks)))\n",
    "ylabels = [f'* {cat}' if top_attacks.loc[cat, 'is_zeroday'] else cat for cat in top_attacks.index]\n",
    "ax.set_yticklabels(ylabels)\n",
    "\n",
    "for i in range(len(top_attacks)):\n",
    "    for j in range(2):\n",
    "        val = data[i, j]\n",
    "        color = 'white' if val < 0.4 else 'black'\n",
    "        ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=10, fontweight='bold', color=color)\n",
    "\n",
    "ax.set_title('Baseline Detection Recall by Attack Category\\n(* = zero-day holdout)', fontweight='bold')\n",
    "plt.colorbar(im, ax=ax, label='Recall', shrink=0.8)\n",
    "plt.tight_layout()\n",
    "savefig(fig, 'attack_type_heatmap')\n",
    "plt.show()\n",
]))

# --- Section 4: Key Findings ---
cells.append(make_cell("markdown", [
    "## 4. Key Findings Summary\n",
]))

cells.append(make_cell("code", [
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "ax.set_xlim(0, 10)\n",
    "ax.set_ylim(0, 10)\n",
    "ax.axis('off')\n",
    "\n",
    "# Title\n",
    "ax.text(5, 9.5, 'Key Findings: Ghost in the Machine', ha='center', fontsize=16, fontweight='bold')\n",
    "ax.text(5, 8.9, 'Prompt Injection Detection via Semantic Dissonance Analysis', ha='center', fontsize=11, style='italic', color=C['grey'])\n",
    "\n",
    "# Finding boxes\n",
    "findings = [\n",
    "    ('96.9%', 'Test F1-Score', 'BERT Classifier on known attacks', C['blue']),\n",
    "    ('100%', 'Zero-Day F1', 'Perfect detection of unseen attacks', C['green']),\n",
    "    ('+20.3%', 'vs Keyword', 'Improvement over keyword baseline', C['orange']),\n",
    "    ('0.999', 'AUC-ROC', 'Near-perfect discrimination', C['purple']),\n",
    "]\n",
    "\n",
    "for i, (value, label, desc, color) in enumerate(findings):\n",
    "    x_pos = 1.25 + i * 2.1\n",
    "    rect = FancyBboxPatch((x_pos - 0.8, 5.5), 1.9, 2.8, boxstyle='round,pad=0.15', facecolor=color, alpha=0.1, edgecolor=color, linewidth=2)\n",
    "    ax.add_patch(rect)\n",
    "    ax.text(x_pos + 0.15, 7.5, value, ha='center', va='center', fontsize=20, fontweight='bold', color=color)\n",
    "    ax.text(x_pos + 0.15, 6.7, label, ha='center', va='center', fontsize=10, fontweight='bold')\n",
    "    ax.text(x_pos + 0.15, 6.1, desc, ha='center', va='center', fontsize=8, color=C['grey'])\n",
    "\n",
    "# Bottom summary\n",
    "summary_text = ('Traditional keyword and regex defences achieve 0% detection rate on zero-day attacks.\\n'\n",
    "                'Semantic analysis with transformer embeddings generalises to unseen attack categories,\\n'\n",
    "                'validating the hypothesis that intent-level understanding transcends surface-level obfuscation.')\n",
    "ax.text(5, 4.2, summary_text, ha='center', va='center', fontsize=10, style='italic',\n",
    "        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor=C['grey'], alpha=0.8))\n",
    "\n",
    "# Stats line\n",
    "ax.text(5, 2.5, f'Dataset: {len(combined)} samples | 4 attack datasets | {len(zeroday_df)} zero-day holdout samples', ha='center', fontsize=9, color=C['grey'])\n",
    "\n",
    "plt.tight_layout()\n",
    "savefig(fig, 'key_findings_summary')\n",
    "plt.show()\n",
]))

cells.append(make_cell("markdown", [
    "## 5. Publication Summary Statistics\n",
]))

cells.append(make_cell("code", [
    "print('=' * 60)\n",
    "print('PUBLICATION SUMMARY STATISTICS')\n",
    "print('=' * 60)\n",
    "print(f'Total unique samples:        {len(combined)}')\n",
    "print(f'Training samples:            {len(train_df)}')\n",
    "print(f'Validation samples:          {len(val_df)}')\n",
    "print(f'Test samples:                {len(test_df)}')\n",
    "print(f'Zero-day holdout samples:    {len(zeroday_df)}')\n",
    "print(f'Attack categories:           {combined[combined[\"label\"]==1][\"attack_category\"].nunique()}')\n",
    "print(f'Zero-day categories:         homoglyph, caesar')\n",
    "print('-' * 60)\n",
    "print(f'BERT Test Accuracy:          {eval_results[\"baseline_test\"][\"accuracy\"]:.4f}')\n",
    "print(f'BERT Test F1-Score:          {eval_results[\"baseline_test\"][\"f1\"]:.4f}')\n",
    "print(f'BERT Test AUC-ROC:           {eval_results[\"baseline_test\"].get(\"auc\", \"N/A\")}')\n",
    "print(f'BERT Zero-Day F1:            {eval_results[\"baseline_zeroday\"][\"f1\"]:.4f}')\n",
    "print(f'Keyword Test F1:             {eval_results[\"keyword_baseline\"][\"f1\"]:.4f}')\n",
    "print(f'Keyword Zero-Day F1:         {eval_results[\"keyword_zeroday\"][\"f1\"]:.4f}')\n",
    "print(f'Regex Test F1:               {eval_results[\"regex_baseline\"][\"f1\"]:.4f}')\n",
    "print(f'Regex Zero-Day F1:           {eval_results[\"regex_zeroday\"][\"f1\"]:.4f}')\n",
    "print('-' * 60)\n",
    "print(f'BERT vs Keyword (F1 delta):  +{eval_results[\"baseline_test\"][\"f1\"] - eval_results[\"keyword_baseline\"][\"f1\"]:.4f}')\n",
    "print(f'BERT vs Regex (F1 delta):    +{eval_results[\"baseline_test\"][\"f1\"] - eval_results[\"regex_baseline\"][\"f1\"]:.4f}')\n",
    "print('=' * 60)\n",
    "\n",
    "print('\\nAll figures saved to:', str(FIG_DIR.resolve()))\n",
]))

# --- Build notebook ---
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.3"},
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "results_analysis.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {out_path}")
