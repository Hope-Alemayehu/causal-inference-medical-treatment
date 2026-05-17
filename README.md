markdown# Causal Inference vs ML in Medical Treatment Data

A demonstration of why standard machine learning models produce 
dangerously wrong conclusions in medical data — and how causal 
inference corrects them.

## The Core Problem

In medicine, doctors don't assign treatments randomly. They use 
judgment — often giving treatments to patients they believe will 
respond. This creates **confounding bias**:

- Less severe patients → more likely to receive treatment
- Less severe patients → more likely to recover anyway
- ML model sees this and concludes: treatment is very effective
- Reality: ML is measuring doctor behavior, not drug effectiveness

## What This Project Shows

| Model | Treatment Effect Estimate | Error |
|---|---|---|
| Raw Data (naive) | 0.228 | 128% too high |
| ML Model (Logistic Regression) | 0.301 | 201% too high |
| Causal Model (DoWhy) | 0.070 | 30% off |
| **Real Effect (ground truth)** | **0.100** | **—** |

The causal model isn't perfect — but it's in the right neighborhood. 
The ML model is in a completely different city.

## Why This Matters in Real Medical AI

Real hospital data has this problem everywhere. Patients aren't 
randomly assigned treatments. Sicker patients get different drugs, 
different dosages, different care pathways. A model trained on this 
data without causal reasoning will:

- Overstate effectiveness of treatments given to healthier patients
- Understate effectiveness of treatments given to sicker patients
- Make confident, wrong recommendations that affect real people

## The Causal Graph
age      ──→ treatment
age      ──→ recovery
severity ──→ treatment
severity ──→ recovery
treatment──→ recovery  ← this is what we want to measure

Age and severity are **confounders** — they influence both who gets 
treatment and who recovers. DoWhy blocks these backdoor paths and 
isolates the true treatment effect.

## How It Works — Pearl's Framework

This project implements Judea Pearl's causal inference pipeline:

1. **Model** — declare the causal graph (what causes what)
2. **Identify** — find a valid estimation strategy (backdoor criterion)
3. **Estimate** — calculate the actual causal effect
4. **Refute** — stress test the result with random noise

## Tools

- [DoWhy](https://github.com/py-why/dowhy) — Microsoft's causal inference library
- scikit-learn — naive ML baseline
- pandas / numpy — data simulation
- matplotlib — visualization

## Running the Project

Open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

Or clone and run locally:

```bash
git clone https://github.com/Hope-Alemayehu/causal-inference-medical-treatment
pip install dowhy scikit-learn pandas matplotlib numpy
jupyter notebook causal_model.ipynb
```

## Key Concepts Demonstrated

- **Confounding bias** — hidden variables that corrupt ML estimates
- **Backdoor criterion** — mathematical strategy to block confounding
- **Treatment effect estimation** — core problem in medical AI
- **Causal graphs (DAGs)** — formalizing domain knowledge as structure
- **Refutation testing** — stress testing causal estimates

## Background

This project was built as an introduction to causal inference 
in medical AI contexts. The concepts here — confounding, 
treatment effects, causal graphs — are foundational to how 
modern medical AI research evaluates interventions.

For deeper reading:
- Judea Pearl — *The Book of Why*
- DoWhy documentation — https://py-why.github.io/dowhy
