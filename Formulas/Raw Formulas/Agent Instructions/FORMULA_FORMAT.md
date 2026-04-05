# Formula Documentation Format Guide

> **Audience:** AI agents tasked with adding formulas to the Formula Bank.
> Read this file before writing any formula entry. Follow the structure exactly.

---

## Structure Overview

Each formula entry is a self-contained section with five parts in this exact order:

1. **Heading** — Name, source link, and citation
2. **Context** — What problem the formula addresses and why it exists
3. **Formula** — The LaTeX expression in display math
4. **Where** — A table decomposing every symbol
5. **Usage & Context** — Detailed explanation of purpose, properties, and connections

Separate each formula entry with a horizontal rule (`---`).

---

## Heading Format

Use an H2 (`##`) heading. The formula name is wrapped in an Obsidian internal link pointing to the NOTES file it was extracted from. Follow the link with the author citation in parentheses.

```markdown
## [Formula Name](obsidian://open?vault=undergrad-thesis&file=NOTES%20-%20Paper%20Title%20Here) (Author(s), Year)
```

**Rules:**
- The link text is the formula's descriptive name (e.g., "Price Relative Vector", "Transaction Remainder Factor").
- The Obsidian URI must point to the NOTES file the formula was sourced from. Encode spaces as `%20`.
- The citation uses the format `(Last Name et al., Year)` or `(Last Name, Year)`.
- Do NOT use numbered headings (e.g., "## 1. Formula Name"). The ordering is implicit from document position.

---

## Context Section

One to two paragraphs immediately below the heading. No sub-heading needed — this flows directly after the H2.

**Rules:**
- Explain *what* the formula captures and *why* it is needed.
- Ground it in the broader framework or problem domain.
- Use italics for emphasis on key conceptual terms (e.g., *ratio*, *natural drift*).
- Use bold for framework-specific terminology (e.g., **multiplicative dynamics**).
- Keep LaTeX inline here: use `$...$` for variable references within prose.
- Do NOT use display math (`$$...$$`) in this section.

---

## Formula Section

The LaTeX expression in display math, placed between blank lines.

```markdown
$$
\mathbf{y}_t := \mathbf{v}_t \oslash \mathbf{v}_{t-1}
$$
```

**Rules:**
- Always use `$$ ... $$` (display math) for the formula itself.
- Place a blank line before and after the `$$` delimiters.
- If a formula has multiple equivalent forms (e.g., a definition and an expanded version), show them in a single display block separated by `=`.
- Do NOT add a label, equation number, or caption — the heading serves as the label.

---

## Where Section

A header-less markdown table introduced by the word "Where," on its own line. Each row maps one symbol to its meaning.

```markdown
Where,

| $\mathbf{y}_t$ | Price relative vector at period $t$. Each element is the ratio of ... |
| -------------- | --------------------------------------------------------------------- |
| $\mathbf{v}_t$ | Vector of closing prices for all $m$ assets at the end of period $t$. |
| $\oslash$      | Element-wise (Hadamard) division operator.                            |
```

**Rules:**
- Start with `Where,` (capital W, followed by a comma) on its own line, then a blank line, then the table.
- The table has **no header row**. The first row of data doubles as the visual header via the separator line beneath it.
- Column 1: the symbol in inline LaTeX `$...$`.
- Column 2: plain-English meaning. Start with a noun phrase or short sentence.
- List every symbol, subscript, operator, and special notation that appears in the formula.
- If a symbol was already defined in a previous formula entry, include it anyway but add a parenthetical cross-reference: `(see Price Relative Vector)`.
- Order the symbols in the same order they appear in the formula, left to right, top to bottom.
- Align the pipe characters for readability.

---

## Usage & Context Section

One to two dense paragraphs following the Where table. No sub-heading needed — this flows directly after the table.

**Rules:**
- Explain *how* the formula is used in the framework or algorithm.
- Describe any notable mathematical properties (e.g., self-referential, non-analytic, additive vs. multiplicative).
- Bold key phrases that name important properties or roles (e.g., **multiplicative transition operator**, **self-referential**).
- Cross-reference related formulas by their name (e.g., "see Transaction Remainder Factor"), not by number.
- Explain the practical or economic intuition — why does this formula matter for the agent or the portfolio?
- If the formula has known limitations or idealisations, state them here.
- Use inline LaTeX `$...$` for variable references in prose.
- Do NOT use display math in this section.

---

## LaTeX Conventions

| Situation | Format | Example |
|---|---|---|
| Variable reference in prose | Inline: `$...$` | The weight vector $\mathbf{w}_t$ is... |
| Standalone formula | Display: `$$...$$` | (see Formula Section above) |
| Vectors | Bold Roman: `\mathbf{x}` | $\mathbf{y}_t$, $\mathbf{w}_t$ |
| Scalars | Italic (default): `x` | $p_t$, $\mu_t$, $r_t$ |
| Operators (named) | Upright: `\ln`, `\max`, `\sum`, `\prod` | $\ln(\cdot)$ |
| Element-wise ops | Standard symbols: `\odot`, `\oslash` | $\mathbf{y}_t \odot \mathbf{w}_{t-1}$ |
| Dot product | Centered dot: `\cdot` | $\mathbf{y}_t \cdot \mathbf{w}_{t-1}$ |
| Transpose | Superscript top: `^\top` | $(\cdot)^\top$ |
| ReLU / positive part | Subscript plus: `(\cdot)_+` | $(x)_+$ |
| Fractions in display | `\frac{a}{b}` | $\frac{p_t}{p_{t-1}}$ |
| Fractions inline | `a/b` or small `\tfrac` | $p_t / p_{t-1}$ |

---

## Complete Template

Copy and fill in the template below for each new formula entry:

```markdown
## [FORMULA_NAME](obsidian://open?vault=undergrad-thesis&file=NOTES%20-%20PAPER_TITLE) (AUTHOR, YEAR)

CONTEXT_PARAGRAPH_1

CONTEXT_PARAGRAPH_2 (optional)

$$
LATEX_FORMULA
$$

Where,

| $SYMBOL_1$  | MEANING_1  |
| ----------- | ---------- |
| $SYMBOL_2$  | MEANING_2  |
| $SYMBOL_3$  | MEANING_3  |

USAGE_AND_CONTEXT_PARAGRAPH_1

USAGE_AND_CONTEXT_PARAGRAPH_2 (optional)

---
```

---

## Checklist Before Submitting

- [ ] Heading uses an Obsidian link to the source NOTES file with author/year citation.
- [ ] Context section uses no display math — only inline `$...$`.
- [ ] Formula uses `$$...$$` with blank lines above and below.
- [ ] Where table has no header row and covers every symbol in the formula.
- [ ] Symbols in the Where table appear in the same order as in the formula.
- [ ] Cross-references to other formulas use names, not numbers.
- [ ] Usage section explains both the mathematical role and the practical/economic intuition.
- [ ] Entry ends with a horizontal rule (`---`).
