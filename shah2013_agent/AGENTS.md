# AGENTS.md

## Purpose

This directory is an agent-oriented transcription of the article **“Ultrafast reversal of a Fano resonance in a plasmon-exciton system”** (Physical Review B 88, 075411, 2013).

## Source priority

1. Read `article.md` first for searchable, single-column text and LaTeX mathematics.
2. Treat `source.pdf` as the source of truth whenever any symbol, equation, punctuation, citation, unit, or figure detail is ambiguous.
3. Use `images/figure-1.png` through `images/figure-4.png` for plot axes, legends, curves, annotations, and visual interpretation. The figure captions are in `article.md`.
4. Do not silently correct apparent typos or awkward wording in the paper. Preserve the published source unless the task explicitly asks for an editorial correction.

## Stable anchors

Equations are anchored as `#eq-1` through `#eq-8` and retain their published equation numbers `(1)` through `(8)`. Figures are anchored as `#fig-1` through `#fig-4`. Prefer these anchors when citing or discussing the paper internally.

## Mathematical fidelity

- Preserve hats, subscripts, superscripts, operator order, signs, factors of 2, and `\hbar` exactly when implementing the models.
- Distinguish the CQED model, the naïve semiclassical (SC) model, and the corrected SC model. In particular, do not substitute `\gamma_1` for `\gamma_1^{\mathrm{eff}}` where the corrected SC model is intended.
- Do not infer missing equations from general knowledge. If a formula appears questionable, verify it against `source.pdf`.
- The Supplemental Material referenced by the paper is **not included** in this directory. Do not assume its contents are available.

## Figures

The PNG files are direct high-resolution crops rendered from the corresponding figure image regions in `source.pdf`; captions remain text in `article.md`. When reproducing numerical values from a plot, state that they are read from the figure unless the same value is explicitly given in the article text.

## Implementation guidance

When asked to reproduce the paper computationally, first extract a parameter table from the article, then implement Eqs. (1)–(8) without changing notation or approximations. Separate source-stated assumptions from implementation choices. Any numerical convention not specified in the article or supplied Supplemental Material should be marked as an implementation assumption rather than attributed to the authors.
