# Paper 1 — Phase 3: Intervention Experiment + Calibration Literature

**Paper**: `paper1/paper1_v2.tex`  
**Goal**: Add two targeted improvements to lift the paper from "good observation" to "actionable insight + grounded in theory"  
**Output**: Updated `paper1/paper1_v2.tex` with new section, updated Related Work, and updated Discussion. New analysis script + results JSON.

---

## ⚠️ IMPORTANT CONTEXT

- `paper1/paper1.tex` is the ORIGINAL — **DO NOT MODIFY**.
- All edits go into `paper1/paper1_v2.tex` (already exists, already has Phase 1+2 updates).
- All new analysis scripts go into `paper1/analysis/`.
- **No new LLM compute needed.** Both tasks below use existing data (10 runs per question per model).
- The existing data directories are:
  - `hotpotqa/results_llama/` (200 questions, 10 runs each)
  - `hotpotqa/results_claude/` (200 questions, 10 runs each)
  - `hotpotqa/results_gpt5/` (200 questions, 10 runs each)
  - `hotpotqa/results_gpt4o/` (100 questions, 10 runs each)
  - `hotpotqa/results_gemini/` or `hotpotqa/results_cortex_gemini/` (100-102 questions, 10 runs each)
- Use `common/analysis.py` for `is_correct()` and other metric functions.
- The paper is targeting an **ICML 2026 Workshop** (6-page limit + appendix).

---

## Task A: Majority-Vote Intervention ("Run 3× and Vote")

### Why
The paper currently observes that consistency correlates with correctness and proposes consistency monitoring as a "practical intervention" — but never actually *demonstrates* the intervention working. Reviewers will ask: "You say run multiple times and check agreement — does this actually help?" This task provides the concrete answer.

### What
Use the existing 10 runs per question to simulate a **majority-vote** intervention at various budgets (k=3, 5, 7). For each question, subsample k runs, take the majority answer, and check correctness. Compare against single-run accuracy. Also evaluate a **selective prediction** variant: only answer when ≥(k/2+1) runs agree; abstain otherwise.

### Steps

1. **Create `paper1/analysis/task_majority_vote.py`**:

2. **Majority-vote accuracy** — for each model, for k ∈ {1, 3, 5, 7, 10}:
   - For k=1: sample 1 run per question, 1000 bootstrap iterations. Report mean and 95% CI of accuracy.
   - For k=3,5,7: subsample k runs (without replacement) per question, 500 bootstrap iterations. In each iteration:
     - Take the majority answer (most common answer among the k runs).
     - Check correctness against gold using `is_correct()` from `common/analysis.py`.
     - Report mean accuracy and 95% CI across bootstrap iterations.
   - For k=10: use all 10 runs, take majority answer. This is a single deterministic value per question.
   - For each model, produce a table: k | accuracy | 95% CI | gain over k=1.

3. **Selective prediction (consistency-filtered)** — for k=3:
   - **Unanimous agreement**: all 3 runs give the same answer → answer with that. Otherwise → abstain.
   - **Majority threshold**: ≥2 of 3 agree → answer with majority. Otherwise → abstain.
   - Report:
     - **Coverage**: fraction of questions where the system gives an answer (doesn't abstain).
     - **Accuracy on answered questions**: correctness among non-abstained.
     - **Accuracy × Coverage** (effective accuracy): to compare against always-answer baselines.
   - Also do this for k=5 with thresholds: ≥5/5 (unanimous), ≥4/5, ≥3/5.

4. **Per-difficulty-stratum analysis** — using the difficulty strata from Task 1 (Easy/Medium/Hard):
   - Compute majority-vote gain for k=3 within each stratum.
   - Key question: does voting help MORE on medium/hard questions (where there's more variance)?

5. **Save all results** to `paper1/analysis/task_majority_vote_results.json`.

6. **Print a summary** showing:
   ```
   === Majority Vote Results ===
   Model: Llama 3.1 70B
   k=1: XX.X% [CI]
   k=3: XX.X% [CI] (+X.Xpp)
   k=5: XX.X% [CI] (+X.Xpp)
   
   === Selective Prediction (k=3) ===
   Model: Llama 3.1 70B
   Unanimous (3/3): accuracy=XX.X%, coverage=XX.X%
   Majority  (2/3): accuracy=XX.X%, coverage=XX.X%
   
   === Gain by Difficulty (k=3) ===
   Easy: +X.Xpp | Medium: +X.Xpp | Hard: +X.Xpp
   ```

### How to interpret
- If k=3 majority vote improves accuracy by ≥2pp, this is a concrete, actionable finding: "running 3× and voting improves accuracy by Xpp at 3× compute cost."
- If selective prediction (unanimous k=3) achieves >90% accuracy with reasonable coverage, this validates the "consistency as confidence" thesis.
- If the gain is larger on hard questions, this connects to the difficulty stratification analysis.

### Paper integration

After running the analysis:

1. **Add a new subsection** in Section 4 (Results), after the failure detection section (§4.3):

   ```latex
   \subsection{Majority Voting as a Practical Intervention}
   \label{sec:intervention}
   
   Our analysis suggests consistency signals correctness, but can this be 
   exploited? We simulate a majority-vote intervention using our existing 
   multi-run data. For budget $k \in \{1, 3, 5\}$, we subsample $k$ runs 
   per question, take the majority answer, and evaluate accuracy (500 
   bootstrap iterations).
   
   [INSERT TABLE/INLINE NUMBERS HERE]
   
   Running three parallel executions and voting improves accuracy by 
   [X.X]pp on average across models (Table~\ref{tab:vote}). The gain 
   is concentrated on [easy/medium/hard] questions ([X]pp vs.\ [Y]pp), 
   where behavioral variance provides the most signal. Under selective 
   prediction---answering only when all three runs agree---accuracy 
   reaches [X]\% with [Y]\% coverage, a [Z]pp accuracy gain at the 
   cost of abstaining on [W]\% of questions.
   ```

2. **Add a compact table** (keep it small for page budget):

   ```latex
   \begin{table}[h]
   \centering
   \small
   \caption{Majority-vote accuracy (\%) by compute budget $k$. 
   Gain = improvement over single-run ($k\!=\!1$).}
   \label{tab:vote}
   \begin{tabular}{lccc}
   \toprule
   \textbf{Model} & $k\!=\!1$ & $k\!=\!3$ & \textbf{Gain} \\
   \midrule
   [Fill from results] \\
   \bottomrule
   \end{tabular}
   \end{table}
   ```

3. **Update the abstract**: Add one sentence about the intervention:
   > "A simple 3-run majority vote improves accuracy by [X]pp on average, 
   > validating consistency monitoring as a practical reliability intervention."

4. **Update the contributions list** (Section 1): Add:
   > "\item \textbf{Practical intervention}: Majority voting over 3 parallel 
   > runs improves accuracy by [X]pp, with selective prediction (unanimous 
   > agreement) achieving [Y]\% accuracy at [Z]\% coverage."

5. **Update the Discussion** paragraph on "Practical implications": Replace the speculative "could flag likely failures" with the concrete finding.

6. **Update the Conclusion**: Replace speculative language about monitoring with the concrete result.

7. **Move full results to appendix**: The full k={1,3,5,7,10} × 5 models table and the per-stratum breakdown go in a new Appendix section.

---

## Task B: Connect to Calibration and Uncertainty Literature

### Why
The paper currently has a thin "LLM Calibration and Uncertainty" paragraph in Related Work (one sentence). Reviewers at ML venues will expect the consistency-as-uncertainty-signal thesis to be situated within the broader calibration/uncertainty/selective prediction literature. This is pure writing — no new compute.

### What
Expand the Related Work and Discussion to explicitly connect behavioral consistency to:
1. **Ensemble disagreement** as uncertainty (classic ML)
2. **Self-consistency / majority voting** (Wang et al. 2022 — already cited)
3. **Semantic uncertainty / entropy** (Kuhn et al. 2023 — already cited)
4. **Selective prediction / "I don't know"** (Geifman & El-Yaniv 2017, Kamath et al. 2020)
5. **Conformal prediction for LLMs** (recent work)
6. **Verbalized uncertainty** (Xiong et al. 2024, Lin et al. 2023, Tian et al. 2023)

### Steps

1. **Add BibTeX entries** to `paper1/references.bib` for these papers:

   ```bibtex
   @article{lakshminarayanan2017simple,
     title={Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles},
     author={Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
     journal={NeurIPS},
     year={2017}
   }
   
   @article{geifman2017selective,
     title={Selective Classification for Deep Neural Networks},
     author={Geifman, Yonatan and El-Yaniv, Ran},
     journal={NeurIPS},
     year={2017}
   }
   
   @article{xiong2024llms,
     title={Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs},
     author={Xiong, Miao and Hu, Zhiyuan and Lu, Xinyang and Li, Yifei and Fu, Jie and He, Junxian and Hooi, Bryan},
     journal={ICLR},
     year={2024}
   }
   
   @article{lin2022teaching,
     title={Teaching Models to Express Their Uncertainty in Words},
     author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
     journal={TMLR},
     year={2022}
   }
   
   @article{tian2023just,
     title={Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback},
     author={Tian, Katherine and Mitchell, Eric and Yao, Huaxiu and Manning, Christopher D and Finn, Chelsea},
     journal={EMNLP},
     year={2023}
   }
   
   @article{kamath2020selective,
     title={Selective Question Answering under Domain Shift},
     author={Kamath, Amita and Jia, Robin and Liang, Percy},
     journal={ACL},
     year={2020}
   }
   ```
   
   Verify the citation keys don't conflict with existing entries.

2. **Rewrite the Related Work "Self-Consistency and Uncertainty" paragraph**. Currently it says:

   > "\citet{wang2022self} use majority voting over sampled chains to boost accuracy. We study consistency as a diagnostic signal rather than an ensembling strategy. Prior work on LLM calibration \citep{kadavath2022language} and semantic uncertainty \citep{kuhn2023semantic} focuses on single-turn outputs; we extend this to multi-step agentic settings."

   Replace with a more structured subsection (can still be a `\paragraph` to save space):

   ```latex
   \paragraph{Uncertainty and Selective Prediction.}
   Our work connects to three threads. First, \emph{ensemble disagreement}: 
   deep ensembles use prediction variance as an uncertainty estimate 
   \citep{lakshminarayanan2017simple}; behavioral consistency across agent 
   runs is an analogous signal, where each run acts as an implicit ensemble 
   member. Second, \emph{self-consistency}: \citet{wang2022self} show that 
   majority voting over sampled reasoning chains improves accuracy. We 
   extend this from single-turn CoT to multi-step agentic settings and 
   additionally study consistency as a \emph{diagnostic} signal, not just 
   an ensembling strategy. Third, \emph{selective prediction}: 
   \citet{geifman2017selective} and \citet{kamath2020selective} show that 
   abstaining on uncertain inputs improves precision; our consistency-filtered 
   results (Section~\ref{sec:intervention}) instantiate this for agents. 
   Unlike verbalized confidence \citep{xiong2024llms, tian2023just}, 
   behavioral consistency requires no model modification---it emerges 
   naturally from repeated execution.
   ```

3. **Add a paragraph in Discussion** explicitly connecting findings to calibration:

   ```latex
   \paragraph{Consistency as implicit calibration.}
   Traditional calibration asks whether a model's stated confidence matches 
   its accuracy \citep{kadavath2022language}. Behavioral consistency provides 
   an alternative, \emph{behavioral} notion of calibration: a well-calibrated 
   agent should be uncertain (produce diverse trajectories) precisely when it 
   is likely to err. Our data supports this: answer entropy achieves 
   AUROC~0.67--0.69 for failure detection (Table~\ref{tab:detection}), and 
   majority voting yields concrete accuracy gains 
   (Section~\ref{sec:intervention}). This is analogous to how ensemble 
   disagreement serves as uncertainty in supervised learning 
   \citep{lakshminarayanan2017simple}, but applied to multi-step agentic 
   behavior. Notably, this requires no access to model internals or explicit 
   uncertainty elicitation \citep{xiong2024llms}---it is purely behavioral.
   ```

4. **Minor touch**: In the abstract or contributions, consider adding:
   > "connecting behavioral consistency to the broader uncertainty estimation 
   > and selective prediction literature"

   Only if this fits within the page budget. If tight, skip this.

---

## EXECUTION ORDER

```
1. Task A — Analysis script:
   - Create paper1/analysis/task_majority_vote.py
   - Run it: cd hotpotqa && python ../paper1/analysis/task_majority_vote.py
   - Verify output and save results JSON

2. Task B — References:
   - Add new BibTeX entries to paper1/references.bib
   
3. Task A+B — Paper integration:
   - Update paper1/paper1_v2.tex:
     a. Add §4.X "Majority Voting as a Practical Intervention" (new subsection)
     b. Rewrite Related Work paragraph on uncertainty
     c. Add Discussion paragraph on implicit calibration
     d. Update abstract with intervention result
     e. Update contributions list
     f. Update Discussion "Practical implications"
     g. Update Conclusion
     h. Add appendix section with full vote results
   
4. Page budget check:
   - Task A adds ~0.3 pages (compact table + 1 paragraph)
   - Task B is mostly a rewrite (space-neutral) + 1 new Discussion paragraph
   - If over 6 pages, move full vote table to appendix (keep only k=1 vs k=3 inline)
   
5. Compile and verify:
   - pdflatex paper1_v2.tex compiles cleanly
   - All new \ref and \citep resolve
   - Page count ≤ 6 main body
```

---

## SUCCESS CRITERIA

- [ ] `paper1/analysis/task_majority_vote.py` runs without errors
- [ ] `paper1/analysis/task_majority_vote_results.json` contains all results
- [ ] Majority-vote gain (k=3 vs k=1) is reported for all 5 models with CIs
- [ ] Selective prediction (k=3, unanimous + majority) accuracy and coverage reported
- [ ] Per-stratum vote gain computed
- [ ] New BibTeX entries added to `references.bib` (6 new references)
- [ ] Related Work paragraph rewritten with proper citations
- [ ] Discussion paragraph on "implicit calibration" added
- [ ] New subsection §4.X in Results with compact vote table
- [ ] Abstract updated with concrete intervention number
- [ ] Contributions list updated
- [ ] All numbers in abstract/contributions match the analysis output
- [ ] `paper1.tex` is unchanged
- [ ] Paper compiles cleanly at ≤ 6 pages + appendix
