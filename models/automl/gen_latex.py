"""生成 LaTeX 对比表，避免 shell 转义干扰。"""
import json

with open('outputs/automl/evaluation.json') as f:
    minimal = json.load(f)
with open('outputs/automl_efficient/evaluation.json') as f:
    efficient = json.load(f)

hand = minimal['handcrafted_46d']
m_tsf = minimal['tsfresh']
e_tsf = efficient['tsfresh']

def fmt(name, sub):
    return f"${sub[name + '_mean']:.4f} \\pm {sub[name + '_std']:.4f}$"

latex = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    "\\caption{Comparison of handcrafted 46-dim features vs TSFRESH (AutoML) baselines.\n"
    "All results are 5-fold stratified cross-validation (mean $\\pm$ std) on the\n"
    "473-student IDE log dataset. Raw feature counts: TSFRESH (minimal) = 70,\n"
    "TSFRESH (efficient) = 4863. After FDR selection ($\\alpha=0.05$):\n"
    "minimal = 8, efficient = 102 features.}\n"
    "\\label{tab:automl_baseline}\n"
    "\\begin{tabular}{lccc}\n"
    "\\hline\n"
    "\\textbf{Metric} & \\textbf{Handcrafted 46d} & "
    "\\textbf{TSFRESH (minimal)} & \\textbf{TSFRESH (efficient)} \\\\\n"
    "\\hline\n"
    f"ACCURACY   & {fmt('accuracy', hand)}   & {fmt('accuracy', m_tsf)}   & {fmt('accuracy', e_tsf)}   \\\\\n"
    f"PRECISION  & {fmt('precision', hand)}  & {fmt('precision', m_tsf)}  & {fmt('precision', e_tsf)}  \\\\\n"
    f"RECALL     & {fmt('recall', hand)}     & {fmt('recall', m_tsf)}     & {fmt('recall', e_tsf)}     \\\\\n"
    f"F1         & $\\mathbf{{{hand['f1_mean']:.4f} \\pm {hand['f1_std']:.4f}}}$ & "
    f"{fmt('f1', m_tsf)} & {fmt('f1', e_tsf)} \\\\\n"
    f"AUC        & {fmt('auc', hand)}        & {fmt('auc', m_tsf)}        & {fmt('auc', e_tsf)}        \\\\\n"
    "\\hline\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
)

with open('outputs/automl/latex_table.tex', 'w') as f:
    f.write(latex)
print('LaTeX saved')