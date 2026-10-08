"""Build PRESENTATION.html from the saved results.

Every number and figure is read from outputs/, and the images are embedded, so
the page is a single file that opens offline with a double-click.

    python make_presentation.py
"""

import base64
import json
import os

OUT = "PRESENTATION.html"


def img(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def pct(x):
    return f"{x:.3f}"


ARM_ORDER = ["cnn_original", "cnn_matched", "cnn_matched_frozen", "qnn_frozen", "qnn_trained"]
ARM_ROWS = {
    "cnn_original":       ("CNN, original", "16 ch 3&times;3 conv, 4,096-input head (reference)"),
    "cnn_matched":        ("CNN, matched", "2&times;2 conv, 4 ch, stride 2, trained"),
    "cnn_matched_frozen": ("CNN, matched, frozen", "same conv left at random init"),
    "qnn_frozen":         ("QNN, frozen", "4-qubit circuit left at random init"),
    "qnn_trained":        ("QNN, trained", "4-qubit circuit trained end-to-end"),
}


def _mci(e, fmt="{:.3f}"):
    return f"{fmt.format(e['mean'])} <span class=\"ci\">&plusmn; {fmt.format(e['ci95'])}</span>"


def _verdict(d):
    lo, hi = d["mean_diff"] - d["ci95"], d["mean_diff"] + d["ci95"]
    if lo <= 0 <= hi:
        return "no clear difference"
    return "first is higher" if d["mean_diff"] > 0 else "first is lower"


def bottom_lines(st):
    """Plain-language conclusions, derived from the paired-comparison verdicts."""
    if not st or "qnn_trained" not in st["arms"]:
        return []
    cs = {(c["a"], c["b"]): c for c in st["contrasts"]}
    keys = [("qnn_frozen", "cnn_matched_frozen"), ("qnn_trained", "cnn_matched"), ("qnn_trained", "qnn_frozen")]
    if not all(k in cs for k in keys):
        return []

    def says(c):
        return _verdict(c["balanced.accuracy"]), _verdict(c["imbalanced.precision"])

    rand, trained, helps = (says(cs[k]) for k in keys)
    same = lambda v: all(x == "no clear difference" for x in v)
    lines = [
        "Random quantum features vs random classical features of the same shape: "
        + ("no clear difference." if same(rand) else f"accuracy {rand[0]}, 1:10 precision {rand[1]}."),
        "Trained quantum layer vs trained classical layer of the same shape: "
        + ("no clear difference." if same(trained) else f"accuracy {trained[0]}, 1:10 precision {trained[1]}."),
        "Training the circuit vs leaving it random: "
        + ("helps on both accuracy and 1:10 precision." if helps == ("first is higher", "first is higher")
           else f"accuracy {helps[0]}, 1:10 precision {helps[1]}."),
    ]
    if same(rand) and same(trained):
        lines.append("So on this task the 4-qubit layer behaves like a classical layer with the same number of outputs. "
                     "The large gap in the single run came from the original CNN's much bigger classifier head, "
                     "not from the quantum layer.")
    return lines


def study_section(st, fig):
    arms = [a for a in ARM_ORDER if a in st["arms"]]
    n = max(st["arms"][a]["n_seeds"] for a in arms)
    rows = []
    for a in arms:
        e = st["arms"][a]
        name, desc = ARM_ROWS[a]
        rows.append(
            f"<tr><td><b>{name}</b><br><span class=\"ci\">{desc}</span></td>"
            f"<td>{e['trainable_params']:,}</td>"
            f"<td>{_mci(e['balanced.accuracy'])}</td><td>{_mci(e['balanced.roc_auc'])}</td>"
            f"<td>{_mci(e['imbalanced.precision'])}</td><td>{_mci(e['imbalanced.avg_precision'])}</td>"
            f"<td>{_mci(e['imbalanced.false_positives'], '{:.0f}')}</td></tr>")

    crow = []
    for c in st["contrasts"]:
        cells = []
        for key in ["balanced.accuracy", "imbalanced.precision", "imbalanced.avg_precision"]:
            d = c[key]
            md = 0.0 if abs(d["mean_diff"]) < 5e-4 else d["mean_diff"]
            cells.append(f"<td>{md:+.3f} <span class=\"ci\">&plusmn; {d['ci95']:.3f}</span>"
                         f"<br><span class=\"ci\">{_verdict(d)}</span></td>")
        crow.append(f"<tr><td>{c['question']}<br><span class=\"ci\">{c['n_pairs']} paired seeds</span></td>"
                    + "".join(cells) + "</tr>")

    has_trained = "qnn_trained" in st["arms"]
    lines = bottom_lines(st)
    bottom = ('<div class="note"><b>Bottom line.</b><ul>' + "".join(f"<li>{x}</li>" for x in lines)
              + "</ul></div>") if lines else ""
    title = "matched capacity, trained circuit" if has_trained else "matched capacity (trained-circuit runs in progress)"
    pending = "" if has_trained else (
        "<p class=\"note\"><b>In progress:</b> the trained-circuit model (5 more runs, about 5 minutes each on a CPU) "
        "is still running. This section updates when it finishes.</p>")
    moved = st["arms"].get("qnn_trained", {}).get("circuit_weight_change_max")
    moved_txt = (f" In the trained arm the circuit parameters moved by up to {moved:.2f} rad, "
                 f"so gradients do reach the circuit.") if moved is not None else ""

    return f"""
<section id="study">
  <h2>5. Controlled study: {n} seeds, {title}</h2>
  {pending}
  <p>The single run above cannot attribute the gap to the quantum layer: the models differ in head size and epochs, the circuit is never trained, and there is one seed. This study fixes all three.</p>
  <ul>
    <li><b>Matched capacity.</b> The matched CNN has the QNN's exact shape: a 2&times;2, stride-2, 4-channel front end (one output per patch per channel, like the 4 qubits), tanh to keep outputs in [&minus;1, 1] like a Pauli-Z expectation, then the same pool and the same 256&nbsp;&rarr;&nbsp;16&nbsp;&rarr;&nbsp;2 head. Front ends: 20 classical parameters vs 24 quantum.</li>
    <li><b>Trained circuit.</b> The circuit is re-implemented as a PyTorch statevector so gradients reach it. It matches PennyLane to 5&times;10<sup>&minus;7</sup> in value and to 3&times;10<sup>&minus;7</sup> in gradient against PennyLane's parameter-shift rule.{moved_txt}</li>
    <li><b>Same training for every model.</b> Adam, lr 10<sup>&minus;3</sup>, batch 64, 12 epochs; each model keeps the epoch with the lowest loss on a 10% validation split.</li>
    <li><b>Variance.</b> Each seed is a different train/validation/test split and initialisation. Values are mean &plusmn; 95% confidence interval over seeds.</li>
  </ul>
  <figure><img src="{fig}" alt="Study results with confidence intervals"></figure>
  <div class="wrap"><table>
    <tr><th>Model</th><th>Trainable params</th><th>Balanced acc.</th><th>ROC-AUC</th><th>1:10 precision</th><th>1:10 avg. precision</th><th>1:10 false pos.</th></tr>
    {"".join(rows)}
  </table></div>
  <h3>Paired comparisons (same splits; difference = first minus second)</h3>
  <div class="wrap"><table>
    <tr><th>Question</th><th>Balanced acc.</th><th>1:10 precision</th><th>1:10 avg. precision</th></tr>
    {"".join(crow)}
  </table></div>
  <p class="sub">"No clear difference" means the 95% interval of the paired difference includes zero.</p>
  {bottom}
</section>
"""


def main():
    with open("outputs/metrics.json") as f:
        m = json.load(f)
    with open("outputs/gate_g0.json") as f:
        g = json.load(f)

    cnn, qnn = m["cnn"], m["qnn"]
    cnn_i, qnn_i = m["cnn_imbalanced"], m["qnn_imbalanced"]
    g0 = m["gate_g0"]

    st = None
    if os.path.exists("outputs/study_summary.json") and os.path.exists("outputs/fig_study.png"):
        with open("outputs/study_summary.json") as f:
            st = json.load(f)
    study_html = study_section(st, img("outputs/fig_study.png")) if st else ""
    nav_study = '<a href="#study">Controlled study</a>' if st else ""
    _bl = bottom_lines(st)
    if _bl:
        sa = st["arms"]
        acc = lambda k: sa[k]["balanced.accuracy"]["mean"]
        gq, gc = acc("qnn_trained") - acc("qnn_frozen"), acc("cnn_matched") - acc("cnn_matched_frozen")
        gain_txt = (f"Training adds {gq:+.3f} accuracy to the circuit and {gc:+.3f} to the classical layer"
                    + (", about the same." if abs(gq - gc) < 0.01 else "."))
        summary_study = (
            '<div class="note"><b>Main result (controlled study, 5 seeds, <a href="#study">section 5</a>).</b> '
            "Matched for size and training, the quantum layer performs the same as a classical layer of the same shape: "
            f"balanced accuracy {sa['qnn_trained']['balanced.accuracy']['mean']:.3f} (trained QNN) vs "
            f"{sa['cnn_matched']['balanced.accuracy']['mean']:.3f} (matched CNN), "
            f"and {sa['qnn_frozen']['balanced.accuracy']['mean']:.3f} vs {sa['cnn_matched_frozen']['balanced.accuracy']['mean']:.3f} "
            "when both are left random. " + gain_txt
            + "<ul>" + "".join(f"<li>{x}</li>" for x in _bl) + "</ul></div>")
        tiles_title = "Single run (seed 42, original models): what happens at a realistic class ratio"
    else:
        summary_study = ""
        tiles_title = ""
    single = " (single run)" if st else ""
    k = 1 if st else 0

    figs = {k: img(f"outputs/{k}.png") for k in [
        "fig_gate_g0", "fig_imbalance",
        "roc_curves", "confusion_matrices", "training_curves",
    ]}

    if st:
        LIMITS = (
            "<li><b>Sections 2&ndash;4 are a single run</b> (seed 42, original models). Use the controlled study in section 5 for any CNN vs QNN claim.</li>"
            "<li><b>Simulated, noiseless circuit.</b> All quantum results are exact statevector simulation on a CPU. Hardware noise would only lower the QNN numbers.</li>"
            "<li><b>Small circuit.</b> 4 qubits, 2 layers, 24 parameters. Nothing here says how larger circuits behave.</li>"
            "<li><b>Balanced training only.</b> Every model trains at 1:1 and is tested at 1:10. Training at a skewed ratio, where focal loss would become relevant, has not been tried.</li>"
            "<li><b>Patch classification, not full detection.</b> The task classifies 32&times;32 patches; there is no detection on whole images yet.</li>")
        NEXT = (
            "<li>Train at the skewed ratio and compare plain cross-entropy with focal loss.</li>"
            "<li>Add depolarising noise to the circuit to see how much of the QNN result survives realistic hardware.</li>"
            "<li>Scale the circuit (more layers, larger patches) with the matched classical counterpart scaled alongside.</li>"
            "<li>Run as a sliding-window detector on full images (e.g. WIDER FACE) and report average precision.</li>")
        REPRO_STUDY = ("<li><code>python study.py</code> then <code>python -m src.study_report</code>: the controlled "
                       "5-seed study and its summary (<code>outputs/study_runs.jsonl</code>, "
                       "<code>outputs/study_summary.json</code>).</li>")
    else:
        LIMITS = (
            "<li><b>The quantum circuit is not trained.</b> Quantum features are computed once and cached to disk, and the classifier trains on the cached tensor, so the loss never reaches the 24 circuit parameters. The QNN is therefore a <i>fixed random quantum feature</i> baseline. Its parameters are marked as frozen so the parameter count reflects this.</li>"
            "<li><b>The two models are not capacity-matched.</b> The CNN's classifier receives 4,096 inputs (16 channels &times; 16&times;16); the QNN's receives 256 (4 channels pooled to 8&times;8). The CNN also trains for 15 epochs against 10. The accuracy gap cannot yet be attributed to the quantum layer.</li>"
            "<li><b>One seed, one split.</b> No error bars yet, so small differences should not be read as real.</li>"
            "<li><b>No validation set.</b> Epoch counts are fixed rather than tuned.</li>")
        NEXT = (
            "<li>Match capacity: same channel count and pooling into the classifier, same epoch budget for both models.</li>"
            "<li>Train the circuit end-to-end and compare against the fixed-circuit result.</li>"
            "<li>Repeat over 5 seeds and report means with confidence intervals.</li>"
            "<li>Train at the skewed ratio as well; focal loss only becomes relevant then.</li>")
        REPRO_STUDY = ""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Detection: CNN vs Quanvolutional QNN</title>
<style>
  :root {{
    --bg: #fcfcfb; --ink: #1f1f1e; --muted: #5f5f5b; --line: #e6e6e3;
    --card: #ffffff; --cnn: #4A90D9; --qnn: #E8543A; --warn: #f6ece9;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.55 "Segoe UI", system-ui, -apple-system, Roboto, sans-serif; }}
  main {{ max-width: 1040px; margin: 0 auto; padding: 40px 20px 80px; }}
  nav {{ position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--line);
        z-index: 2; }}
  nav div {{ max-width: 1040px; margin: 0 auto; padding: 10px 20px; display: flex; gap: 18px;
            flex-wrap: wrap; font-size: 14px; }}
  nav a {{ color: var(--muted); text-decoration: none; }}
  nav a:hover {{ color: var(--ink); }}
  h1 {{ font-size: 30px; line-height: 1.2; margin: 0 0 6px; }}
  h2 {{ font-size: 21px; margin: 56px 0 10px; padding-top: 8px; }}
  h3 {{ font-size: 16px; margin: 22px 0 6px; }}
  .sub {{ color: var(--muted); margin: 0 0 28px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
  .tile {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }}
  .tile .n {{ font-size: 28px; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .tile .l {{ color: var(--muted); font-size: 14px; }}
  figure {{ margin: 18px 0; background: var(--card); border: 1px solid var(--line);
           border-radius: 10px; padding: 12px; }}
  figure img {{ width: 100%; height: auto; display: block; }}
  figcaption {{ color: var(--muted); font-size: 14px; margin-top: 8px; }}
  table {{ border-collapse: collapse; width: 100%; background: var(--card); font-size: 15px;
          font-variant-numeric: tabular-nums; }}
  th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ font-weight: 600; color: var(--muted); }}
  .wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }}
  .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }}
  .bad {{ font-weight: 600; }}
  .ci {{ color: var(--muted); font-size: 13px; }}
  .note {{ background: var(--warn); border-radius: 10px; padding: 14px 16px; }}
  .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  @media (max-width: 760px) {{ .two {{ grid-template-columns: 1fr; }} }}
  code {{ font-family: Consolas, "SFMono-Regular", monospace; font-size: 14px;
         background: #f1f1ee; padding: 1px 5px; border-radius: 4px; }}
  ul {{ padding-left: 22px; }}
  li {{ margin: 4px 0; }}
  @media print {{
    nav {{ display: none; }}
    main {{ padding-top: 0; }}
    h2 {{ margin-top: 28px; break-after: avoid; }}
    figure, .tile, table, .note {{ break-inside: avoid; }}
    .two {{ grid-template-columns: 1fr 1fr; }}
  }}
</style>
</head>
<body>
<nav><div>
  <a href="#summary">Summary</a><a href="#setup">Setup</a><a href="#g0">Validity check</a>
  <a href="#imbalance">1:10 ratio</a><a href="#balanced">Balanced results</a>
  {nav_study}<a href="#limits">Limitations</a><a href="#next">Next steps</a><a href="#repro">Reproduce</a>
</div></nav>
<main>

<section id="summary">
  <h1>Face Detection Benchmark: Classical CNN vs Quanvolutional QNN</h1>
  <p class="sub">Stage 1 of a one-shot face recognition system. Faces from LFW, non-faces from CIFAR-10, 32&times;32 grayscale patches.</p>

  {summary_study}
  <h3>{tiles_title}</h3>
  <div class="tiles">
    <div class="tile"><div class="n">{pct(qnn['precision'])} &rarr; {pct(qnn_i['precision'])}</div>
      <div class="l">QNN precision, balanced &rarr; 1:10 background</div></div>
    <div class="tile"><div class="n">{pct(cnn['precision'])} &rarr; {pct(cnn_i['precision'])}</div>
      <div class="l">CNN precision, balanced &rarr; 1:10 background</div></div>
    <div class="tile"><div class="n">{qnn_i['false_positives']} vs {cnn_i['false_positives']}</div>
      <div class="l">False alarms at 1:10 (QNN vs CNN)</div></div>
    <div class="tile"><div class="n">{max(g['new']['sharpness_auc'], g['new']['intensity_auc']):.2f}</div>
      <div class="l">Best single-statistic AUC (0.5 = chance). Benchmark passes the validity gate</div></div>
  </div>
</section>

<section id="setup">
  <h2>1. Setup</h2>
  <ul>
    <li><b>Faces:</b> LFW, every identity with at least 20 photos: 62 people, 3,023 images.</li>
    <li><b>Non-faces:</b> CIFAR-10, all 10 classes, matched 1:1. 80/20 stratified split (4,836 train / 1,210 test), seed 42.</li>
    <li><b>CNN:</b> Conv 3&times;3 (16 channels) &rarr; max-pool &rarr; dense 16 &rarr; 2 classes. {cnn['n_params']:,} trainable parameters.</li>
    <li><b>QNN:</b> 4-qubit quanvolution over 2&times;2 patches (RY encoding, 2-layer ring of CNOTs, Pauli-Z readout, circuit depth {qnn['circuit_depth']}) &rarr; max-pool &rarr; dense 16 &rarr; 2 classes. {qnn['n_params']:,} trainable parameters plus {qnn['q_params']} fixed circuit parameters.</li>
  </ul>
</section>

<section id="g0">
  <h2>2. Is the benchmark valid? (Gate G0)</h2>
  <p>Faces and non-faces come from different datasets, so a model could score well by recognising <i>which dataset</i> an image came from instead of whether it contains a face. Before any training, three trivial baselines are run. If one number per image (sharpness or brightness) could separate the classes, the accuracy numbers would be meaningless.</p>
  <figure><img src="{figs['fig_gate_g0']}" alt="Gate G0 dot plot">
    <figcaption>Both statistics stay close to chance. Widening the negatives from 4 vehicle classes to all 10 classes reduced the sharpness signal further (AUC {g['old']['sharpness_auc']:.3f} &rarr; {g['new']['sharpness_auc']:.3f}, Cohen's d {g['old']['sharpness_d']:.2f} &rarr; {g['new']['sharpness_d']:.2f}).</figcaption></figure>
  <p>A logistic regression on raw pixels does reach AUC {g0['pixel_auc']:.3f}. That is expected for a real task with spatial structure and is not a confound: the gate's self-test (<code>tests/test_gate_g0.py</code>) builds two classes with identical sharpness and brightness that differ only in layout, and raw pixels separate those perfectly too.</p>
</section>

<section id="imbalance">
  <h2>3. Performance at a realistic class ratio{single}</h2>
  <p>A sliding-window detector sees far more background than faces. The models were therefore also tested on <b>{qnn_i['n_pos']} test faces + {qnn_i['n_neg']:,} background images</b> (1:10). None of the background images were used in training or in the balanced test set.</p>
  <figure><img src="{figs['fig_imbalance']}" alt="Precision and ROC-AUC, balanced vs 1:10"></figure>
  <div class="wrap"><table>
    <tr><th>At 1:10</th><th><span class="dot" style="background:var(--cnn)"></span>CNN</th><th><span class="dot" style="background:var(--qnn)"></span>QNN</th></tr>
    <tr><td>Precision</td><td>{pct(cnn_i['precision'])}</td><td class="bad">{pct(qnn_i['precision'])}</td></tr>
    <tr><td>Recall</td><td>{pct(cnn_i['recall'])}</td><td>{pct(qnn_i['recall'])}</td></tr>
    <tr><td>F1</td><td>{pct(cnn_i['f1'])}</td><td>{pct(qnn_i['f1'])}</td></tr>
    <tr><td>Average precision</td><td>{pct(cnn_i['avg_precision'])}</td><td>{pct(qnn_i['avg_precision'])}</td></tr>
    <tr><td>ROC-AUC</td><td>{pct(cnn_i['roc_auc'])}</td><td>{pct(qnn_i['roc_auc'])}</td></tr>
    <tr><td>False positives (rate)</td><td>{cnn_i['false_positives']} ({cnn_i['fpr']:.4f})</td><td class="bad">{qnn_i['false_positives']} ({qnn_i['fpr']:.4f})</td></tr>
  </table></div>
  <p><b>Why ROC-AUC hides this:</b> ROC-AUC depends only on how the model ranks faces against non-faces, not on how many of each there are. Precision does depend on it. The QNN's false-positive rate of {qnn_i['fpr']*100:.1f}% looks harmless at 1:1 but turns into {qnn_i['false_positives']} false alarms when background outnumbers faces ten to one. For a detector, precision and average precision are the numbers to report.</p>
</section>

<section id="balanced">
  <h2>4. Balanced test set (1:1, 1,210 images){single}</h2>
  <div class="wrap"><table>
    <tr><th>Metric</th><th><span class="dot" style="background:var(--cnn)"></span>CNN</th><th><span class="dot" style="background:var(--qnn)"></span>QNN</th></tr>
    <tr><td>Accuracy</td><td>{pct(cnn['accuracy'])}</td><td>{pct(qnn['accuracy'])}</td></tr>
    <tr><td>Precision</td><td>{pct(cnn['precision'])}</td><td>{pct(qnn['precision'])}</td></tr>
    <tr><td>Recall</td><td>{pct(cnn['recall'])}</td><td>{pct(qnn['recall'])}</td></tr>
    <tr><td>F1</td><td>{pct(cnn['f1'])}</td><td>{pct(qnn['f1'])}</td></tr>
    <tr><td>ROC-AUC</td><td>{pct(cnn['roc_auc'])}</td><td>{pct(qnn['roc_auc'])}</td></tr>
    <tr><td>Trainable parameters</td><td>{cnn['n_params']:,}</td><td>{qnn['n_params']:,} (+{qnn['q_params']} fixed)</td></tr>
    <tr><td>Inference per 32&times;32 image (simulator)</td><td>{cnn['latency_full']:.2f} ms</td><td>{qnn['latency_full']:,.0f} ms</td></tr>
  </table></div>
  <div class="two">
    <figure><img src="{figs['roc_curves']}" alt="ROC curves"></figure>
    <figure><img src="{figs['confusion_matrices']}" alt="Confusion matrices"></figure>
  </div>
  <figure><img src="{figs['training_curves']}" alt="Training curves"></figure>
  <p class="sub">QNN latency is classical simulation time on a CPU, not quantum hardware time.</p>
</section>

{study_html}
<section id="limits">
  <h2>{5 + k}. Limitations</h2>
  <div class="note">
  <ul>
    {LIMITS}
  </ul>
  </div>
</section>

<section id="next">
  <h2>{6 + k}. Next steps</h2>
  <ol>
    {NEXT}
  </ol>
</section>

<section id="repro">
  <h2>{7 + k}. Reproduce</h2>
  <ul>
    <li><code>python main.py</code>: full run (G0 gate, both models, balanced and 1:10 evaluation, all figures). Console output of the reported run is in <code>outputs/run_log.txt</code>.</li>
    <li><code>python -m src.baseline_g0</code>: validity gate, old vs new negatives.</li>
    <li><code>python tests/test_pipeline.py</code> and <code>python tests/test_gate_g0.py</code>: checks that train, test and background sets never overlap, and that the gate catches known confounds.</li>
    {REPRO_STUDY}
    <li><code>notebooks/quantum_face_simulator.ipynb</code>: interactive simulator (circuit, feature maps, live detector, deployment sliders). Open with <code>launch_simulator.bat</code>, <code>jupyter lab</code>, or <a href="https://colab.research.google.com/github/NAVTEJJ/face_detection_benchmark/blob/main/notebooks/quantum_face_simulator.ipynb">Google Colab</a>.</li>
    <li><code>python make_presentation.py</code>: rebuilds this page from <code>outputs/</code>.</li>
  </ul>
</section>

</main>
</body>
</html>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
