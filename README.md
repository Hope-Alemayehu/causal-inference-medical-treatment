# Causal Inference & Bayesian Networks in Medical AI

Two notebooks demonstrating the difference between correlation,
Bayesian belief updating, and causal inference — using simulated
medical treatment data.

Built while reading Judea Pearl's *The Book of Why*.

---

## The Core Problem

In medicine, doctors don't assign treatments randomly. They use
judgment — often giving treatments to patients they believe will
respond. This creates **confounding bias**:

- Less severe patients → more likely to receive treatment
- Less severe patients → more likely to recover anyway
- ML model sees this and concludes: treatment is very effective
- Reality: ML is measuring doctor behavior, not drug effectiveness

Bayesian networks update beliefs correctly — but still can't tell
you what happens when you *intervene*. For that you need causal
inference.

This repo demonstrates both, and shows exactly where each breaks down.

---

## Notebooks

### 1. `causal_model.ipynb` — Causal Inference with DoWhy

**Question answered:** Does this treatment actually cause recovery,
or do healthier patients just happen to receive it?

**What it demonstrates:**
- Confounding bias deliberately baked into simulated patient data
- Why ML overestimates treatment effects by 3x on the same data
- Causal graph (DAG) construction from domain knowledge
- DoWhy's four-step pipeline: Model → Identify → Estimate → Refute
- Pearl's Rung 2 — Intervention

**Key result:**

| Model | Estimate | Ground Truth | Error |
|---|---|---|---|
| Raw Data (naive) | 0.228 | 0.100 | 128% too high |
| ML Model (Logistic Regression) | 0.301 | 0.100 | 201% too high |
| Causal Model (DoWhy) | 0.070 | 0.100 | 30% off |

The causal model isn't perfect — but it's in the right neighborhood.
The ML model is in a completely different city.

**The causal graph:**
Age      ──→ Treatment
Age      ──→ Recovery
Severity ──→ Treatment  ← confounders block this path
Severity ──→ Recovery
Treatment──→ Recovery   ← this is what we want to measure

Age and severity are **confounders** — they influence both who gets
treatment and who recovers. DoWhy blocks these backdoor paths and
isolates the true treatment effect.

**How it works — Pearl's framework:**
1. **Model** — declare the causal graph (what causes what)
2. **Identify** — find a valid estimation strategy (backdoor criterion)
3. **Estimate** — calculate the actual causal effect
4. **Refute** — stress test the result with random noise

---

### 2. `bayesian_network_medical.ipynb` — Bayesian Network

**Question answered:** How do beliefs about a patient update
in real time as clinical evidence arrives?

**What it demonstrates:**
- Fork, Chain, and Collider structures in one network
- Live belief updating using Variable Elimination
- Explaining away — ruling out one cause amplifies belief in another
- Berkson's paradox — collider bias from conditioning on hospital data
- Pearl's Rung 1 — Association

**Key result:**

| Evidence | P(Recovery) | P(Discharged) |
|---|---|---|
| Baseline | 0.605 | 0.641 |
| Old + Severe | 0.360 | 0.476 |
| Old + Severe + Treated | 0.500 | 0.700 |
| Old + Severe + Treated + Recovered | 1.000 | 0.900 |

**The causal structures:**
FORK (Confounder):
Age ──→ Recovery
Age ──→ Severity
→ Age is a common cause — must control for it
CHAIN (Mediation):
Age ──→ Severity ──→ Treatment ──→ Recovery
→ Information flows through the chain
COLLIDER (Berkson's Paradox):
Treatment ──→ Discharged ←── Recovery
→ Never condition on Discharged
or you create spurious correlations that
don't exist in the real population

---

## The Core Insight Across Both Notebooks

Standard ML and Bayesian networks operate on Pearl's **Rung 1**
— they update beliefs based on observed patterns in data.

Causal models operate on **Rung 2** — they answer what actually
happens when you intervene and change something regardless of
who normally receives it.
Rung 1 — Association (Bayesian Network):
P(Recovery | Treatment)
"Patients who received treatment tend to recover more"
→ Includes confounding bias
→ Cannot tell you what happens if you change anything
Rung 2 — Intervention (DoWhy):
P(Recovery | do(Treatment))
"If we force treatment on everyone regardless of severity,
what actually happens?"
→ Removes doctor selection behavior
→ Isolates the real drug effect
Rung 3 — Counterfactual (not built yet):
"Would this specific patient have recovered
without treatment?"
→ Requires individual-level causal reasoning

These are mathematically different questions.
Confusing Rung 1 with Rung 2 is the most common and
dangerous mistake in medical AI.

---

## Why This Matters in Real Medical AI

Real hospital data has this problem everywhere. Patients aren't
randomly assigned treatments. Sicker patients get different drugs,
different dosages, different care pathways. A model trained on
this data without causal reasoning will:

- Overstate effectiveness of treatments given to healthier patients
- Understate effectiveness of treatments given to sicker patients
- Make confident, wrong recommendations that affect real people
- Fail to generalize across hospitals because it learned
  institutional behavior, not biological mechanisms

Additionally — studies that use only **discharged patients**
(because that's who has complete records) are conditioning on
a collider. This is Berkson's paradox. The correlations they
find inside that population don't exist in the real world.

---

## Tools

- [DoWhy](https://py-why.github.io/dowhy) — Causal inference
  (Microsoft Research)
- [pgmpy](https://pgmpy.org) — Probabilistic graphical models
  and Bayesian networks
- scikit-learn — ML baseline (Logistic Regression)
- pandas / numpy — Data simulation
- networkx + matplotlib — Graph visualization

---

## Running the Notebooks

Open directly in Google Colab:

Notebook 1 — Causal Inference:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Rdsq15UoMLDIJqwCP4WR1BYdRg6UHH14?usp=sharing)

Notebook 2 — Bayesian Network:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]((https://colab.research.google.com/drive/1Z0Y4HJCAR6d95TtnTTTuwY61zjussYWv#scrollTo=3tZEgPQwLROb))

Or clone and run locally:

```bash
git clone https://github.com/Hope-Alemayehu/causal-inference-medical-treatment
pip install dowhy pgmpy scikit-learn pandas matplotlib numpy networkx
jupyter notebook
```

---

## Key Concepts Demonstrated

| Concept | Notebook | Description |
|---|---|---|
| Confounding bias | 1 | Hidden variable corrupting ML estimates |
| Backdoor criterion | 1 | Mathematical strategy to block confounding |
| Treatment effect estimation | 1 | Core problem in medical AI |
| Causal graphs (DAGs) | Both | Formalizing domain knowledge as structure |
| Refutation testing | 1 | Stress testing causal estimates |
| Fork structure | 2 | Common cause — confounder |
| Chain structure | 2 | Sequential causation |
| Collider structure | 2 | Common effect — Berkson's paradox |
| Explaining away | 2 | Ruling out one cause amplifies another |
| Belief updating | 2 | Real-time probability revision |

---

## Background & Motivation

Built as preparation for medical AI research — specifically
treatment effect estimation and causal reasoning in clinical
imaging and patient data.

The problems demonstrated here are not theoretical. They appear
in published medical AI research:

- Models trained on chest X-rays that learn hospital equipment
  signatures instead of disease patterns
- Drug effectiveness studies that overstate results because
  healthier patients were selectively enrolled
- Hospital studies that find spurious correlations because they
  condition on discharged patients (collider bias)

Understanding the difference between correlation and causation —
and having the tools to measure it — is foundational to building
medical AI that actually helps patients.

---

## Further Reading

- Judea Pearl — *The Book of Why* (non-technical, essential)
- Bernhard Schölkopf — *Towards Causal Representation Learning*
  (connecting causality to deep learning)
- DoWhy documentation — https://py-why.github.io/dowhy
- pgmpy documentation — https://pgmpy.org
