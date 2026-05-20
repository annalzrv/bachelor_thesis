# Submission plan — LC-PINN paper

Strategic timeline for the LC-PINN paper across May 2026 → early 2027. Today is **2026-05-02**. The work is currently workshop-strength on Burgers (clean win vs SA-PINN, ReLoBRaLo, Causal-PINN; λ-invariance result; LC-conditioned eigenvalue extension on Laplace 1D); the BL benchmark fails uniformly across methods and is filed as a caveat. The main gaps for full-conference quality are an operator-learning baseline (FNO / DeepONet) and 2–3 additional PDEs where LC-PINN clearly wins.

## Pipeline

| When | Venue | Deadline | Page limit | Status | Version |
|------|-------|----------|------------|--------|---------|
| **May 4** abs / **May 6** full (AoE) | NeurIPS 2026 main | Hard | 9 pp | **Lottery ticket** | v1: current evidence base + FNO baseline if it fits in 4 days |
| **May 7** (AoE) | AI4Physics @ ICML 2026 | Hard | up to 8 pp | **Primary near-term target** | Same paper as v1, possibly trimmed |
| **Aug–Sep** (TBA when workshop is accepted by NeurIPS) | NeurIPS 2026 ML4PS | Soft (estimated) | 4–8 pp | **Primary workshop target** | v2: + FNO baseline (if not in v1), 2 more PDEs, tightened framing |
| **Oct 2026 onwards** (after NeurIPS main decision ≈ late Sep) | TMLR | Rolling | flexible | **Long-form home** | v3: full extended journal version, theory section |
| **Late Sep / early Oct 2026** | ICLR 2027 main | Hard (date not yet posted) | 9–10 pp | **Stretch target if v3 ready in time** | v3 trimmed to conference length |

## Why this combo

- **AI4Physics @ ICML** is the highest-fit topical venue you can still hit — directly on PINNs / scientific ML, non-archival, dual-sub friendly. This is the primary near-term goal.
- **NeurIPS main** in 4 days is a low-probability lottery ticket (~5–10% odds without FNO baseline, somewhat better with one). The cost is 4 days of writing for the chance of a NeurIPS line on the CV. Same paper as AI4Physics — workshop is non-archival, so concurrent submission is allowed.
- **NeurIPS ML4PS** in August–September gives ~3–4 months of slack to add the FNO baseline, two more PDEs (candidates: Helmholtz with parametric wavenumber, parametric heat, KdV soliton family), and one crisp framing. Highest-fit workshop venue at NeurIPS.
- **TMLR** is the long-form journal home for the extended version — no time pressure, rolling submission, accepts work-in-progress that's been polished. Has to wait until after the NeurIPS main decision because of TMLR's "no concurrent overlap" rule.
- **ICLR 2027** main is the realistic full-conference shot once v3 is mature. Deadline lands right after NeurIPS workshop notification.

## Policy constraints (what's allowed, what isn't)

- **NeurIPS main + AI4Physics @ ICML simultaneously**: ALLOWED. NeurIPS CFP explicitly says workshops do not count as concurrent archival submissions. Standard practice.
- **NeurIPS main rejection → ML4PS later**: ALLOWED. Workshops accept prior-rejected work.
- **NeurIPS main rejection → TMLR**: ALLOWED, but only sequentially. TMLR forbids submission while substantially-overlapping work is under review elsewhere. Wait for the NeurIPS decision (~late September) before submitting to TMLR.
- **Workshop-to-workshop incrementalism**: workshops want "original or substantially extended" work. The May AI4Physics version and the August ML4PS version cannot be byte-identical — by August we should have the FNO baseline and at least one more PDE. This happens naturally if the work keeps moving.
- **Happy-problem caveat**: if NeurIPS main accepts (~7%), the paper becomes archival. Workshop versions of the same archival paper are then constrained to "presentation only" rather than full workshop papers. Low-probability scenario; address only if it happens.

## Off-the-table workshops (rejected as off-topic, do NOT submit)

- **CoLoRAI** — Connecting Low-rank Representations. No low-rank angle in LC-PINN.
- **GenBio** — generative biology / biomedicine. Wrong domain.
- **FMSD** — Foundation Models for Structured Data (tables, graphs). Wrong domain.
- **Graph FM** — graph foundation models. Wrong domain.
- **Weight-Space Symmetries** — too tangential; LC's λ-conditioning is not a weight-space symmetry result.
- **AI for Math** — eigenvalue extension fits weakly; would require a paper substantially different from the LC-PINN main pitch.

Submitting off-topic papers in parallel is reputationally negative in a small subfield and almost always desk-rejected. Avoid.

## Author-side decisions

- **Affiliation.** Use **HSE Moscow** (current 4th-year student status). "Independent Researcher" is a backup if HSE affiliation becomes inappropriate. Unincorporated company cannot appear as an affiliation (not a legal entity yet).
- **Authorship.** If the advisor (научник) has been substantively involved, clarify co-authorship before the May 6/7 submissions, not after. Default position: solo first-author unless the advisor explicitly contributed ideas/results.
- **IP.** Copyright on the paper itself is licensed (not transferred) to the conference under standard NeurIPS / ICML / ICLR / TMLR licenses; the author retains ownership. Code and methods ownership defaults to the author unless HSE's IP policy says otherwise — worth checking before any commercial use, but does not block submission.
- **arXiv preprint.** All four conferences explicitly allow concurrent arXiv preprints during review. Posting to arXiv around the AI4Physics submission date is fine and increases visibility for downstream citations.

## Per-version content checklist

**v1 (May 4–7) — NeurIPS main + AI4Physics**

Scope expanded on 2026-04-27 from baseline-only to full v1+ push. See `V1_PUSH_PLAN.md` for task-level detail.

- Burgers headline: SA-PINN / ReLoBRaLo / Causal-PINN baselines + LC-PINN at 50k and 200k.
- **FNO + DeepONet** operator-learning baselines on Burgers (no longer "if achievable" — required).
- **Helmholtz with parametric wavenumber** as second PDE win (SA + ReLo + LC + FNO/DeepONet; Causal-PINN N/A for elliptic).
- **Viscous-regularised BL** replacing the inviscid caveat (s_t + f'(s)·s_x − ε·s_xx = 0; full method comparison).
- λ-invariance result + **formal proposition + sketch proof** (Methods).
- Eigenvalue extension (Laplace 1D LC).
- Ablation study: K_eval sweep, n_λ_samples sweep, sampling distribution.
- All tables with mean±std (4 seeds).
- 8 pages (AI4Physics), trim or extend to 9 (NeurIPS main).

Drop-list if behind: ablations beyond K_eval → DeepONet → viscous-BL → Helmholtz baselines (keep LC-only). Hard floor: FNO Burgers + error bars + theorem.

**v2 (Aug–Sep) — NeurIPS ML4PS**
- v1 contents +
- FNO + DeepONet baselines on Burgers (head-to-head amortisation comparison).
- Two more parametric PDEs where LC-PINN wins (Helmholtz with parametric wavenumber, parametric heat, or KdV soliton family).
- One crisp framing (likely the **λ-invariance** finding as the scientific spine).
- 4–8 pages.

**v3 (Oct onwards) — TMLR + ICLR 2027 (if pace allows)**
- v2 contents +
- Light theoretical content: a proposition that at the optimum of the LC-PINN objective the conditional network parameterises a residual-minimising solution for the support of λ, with a sketch proof.
- Full ablations (sampling mode, hidden dims scaling, K_eval sensitivity).
- Possibly viscous-regularised BL (s_t + f'(s)·s_x = ε·s_xx) to recover the BL story.
- ICLR-trim version: 9–10 pages.
- TMLR-extended version: no fixed limit, ≈18–25 pages typical.

## Calendar

```
2026
May  2  ← today
May  4  NeurIPS abstract  ──┐  same paper
May  6  NeurIPS full      ──┤
May  7  AI4Physics ICML   ──┘
        ─── decision wait ───
Jun  6  NeurIPS workshop proposals deadline (ML4PS will likely be selected)
Jul    ICML 2026 conference (AI4Physics if accepted, decision before then)
Aug    NeurIPS main reviewer feedback / discussion
Aug–Sep NeurIPS ML4PS paper deadline (TBA)
Sep    NeurIPS main decision
Sep–Oct ICLR 2027 main deadline (date TBA)
Oct    TMLR submission of v3 (after NeurIPS decision)
Dec    NeurIPS 2026 conference (ML4PS workshop)

2027
Jan    ICML 2027 main deadline (fallback if ICLR rejects)
Feb–Mar ICLR 2027 decision
```

## Decision log

- **2026-05-02**: combo decided after deadline-research pass over ICML 2026 workshops, NeurIPS 2026 main + workshop track, ICLR 2027, AISTATS, AAAI, TMLR, JMLR, JCP. Rejected four off-topic ICML workshops (CoLoRAI, GenBio, FMSD, Graph FM). Accepted that NeurIPS main is a low-probability shot worth taking because (a) workshop is non-archival so concurrent submission costs nothing, (b) same paper writing serves both, (c) NeurIPS main decision feeds the TMLR sequence cleanly. ICLR 2027 deferred to v3 to avoid burning the slot on an underdeveloped paper.
