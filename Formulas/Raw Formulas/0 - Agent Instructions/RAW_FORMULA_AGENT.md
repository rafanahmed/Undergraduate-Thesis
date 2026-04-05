# Raw Formula Notes — Agent Instructions

> **Audience:** AI agents tasked with creating or editing standalone formula notes in the `Raw Formulas/` directory.
> Read this file in full before writing any raw formula note. Follow every convention exactly.

---

## How Raw Formulas Differ from the Formula Bank

The `FORMULA_FORMAT.md` file in this directory governs the **Formula Bank** — a single consolidated document where formulas are sections separated by `---`. Raw Formula notes are a **different system**:

| | Formula Bank | Raw Formulas |
|---|---|---|
| Structure | Sections in one `.md` file | Each formula is its own standalone `.md` file |
| Organization | Flat list of entries | Grouped into subdirectories by source paper |
| Headings | `##` with Obsidian URI links | No heading — the filename is the title |
| Cross-references | Plain-text names (e.g., "see Transaction Remainder Factor") | Obsidian wiki-links: `[[Note Name]]` |
| Navigation | Scroll within one document | `0 - Reading Order.md` dependency tree per paper |

Do NOT mix these formats. When working in `Raw Formulas/`, follow this file. When working in the Formula Bank, follow `FORMULA_FORMAT.md`.

---

## Directory Structure

```
Raw Formulas/
├── RAW_FORMULA_AGENT.md              ← this file
├── FORMULA_FORMAT.md                  ← Formula Bank format (separate system)
├── Formula Bank (Use When Paper Reading).md
│
├── Full Paper Title Here/
│   ├── 0 - Reading Order.md
│   ├── Formula Name (Author, Year).md
│   ├── Another Formula (Author, Year).md
│   └── ...
│
└── Another Paper Title/
    ├── 0 - Reading Order.md
    └── ...
```

Each source paper gets its own subdirectory. The directory name is the **full, unabbreviated title** of the paper.

---

## File Naming

Every formula file follows this pattern:

```
Formula Name (First Author, Year).md
```

**Rules:**
- **Formula Name** is a descriptive noun phrase (e.g., "Price Relative Vector", "Transaction Remainder Factor"). Choose the clearest conceptual name, not the paper's numbering.
- **Citation** uses `(First Initial. Last Name et al., Year)` for multi-author papers, or `(Last Name, Year)` for single-author papers.
- If the formula is a general concept not uniquely attributed to one paper's authors (e.g., a standard compounding identity), omit the citation: `Final Portfolio Value (Compounding Objective).md`.
- Do NOT prefix filenames with numbers. Reading order is handled by `0 - Reading Order.md`.

---

## Formula File Structure

Each raw formula `.md` file has **four sections** in this exact order. There is **no heading** — the filename serves as the title.

### 1. Context (opening paragraphs)

One to two paragraphs of plain prose. No heading tag, no display math.

**Rules:**
- Explain *what* the formula captures and *why* it exists within the paper's framework.
- Use *italics* for conceptual emphasis and **bold** for framework-specific terminology.
- Use inline LaTeX `$...$` for variable references in prose.
- Do NOT use display math (`$$...$$`) in this section.

### 2. Display Formula

The LaTeX expression in display math, placed between blank lines.

```
$$
\mathbf{y}_t := \mathbf{v}_t \oslash \mathbf{v}_{t-1}
$$
```

**Rules:**
- Always use `$$...$$` with blank lines before and after.
- If a formula has multiple equivalent forms, show them in one block separated by `=`.
- No equation numbers, labels, or captions.

### 3. Where Table

A table decomposing every symbol, introduced by `Where,` on its own line.

```
Where,

| $\mathbf{y}_t$ | Price relative vector at period $t$. |
| -------------- | ------------------------------------ |
| $\mathbf{v}_t$ | Vector of closing prices for ...     |
```

**Rules:**
- Start with `Where,` (capital W, comma) on its own line, followed by a blank line, then the table.
- The table has **no header row**. The first data row sits above the separator line.
- Column 1: the symbol in inline LaTeX `$...$`.
- Column 2: plain-English definition. Start with a noun phrase or short sentence.
- List **every** symbol, subscript, operator, and special notation in the formula.
- Order symbols as they appear in the formula, left to right.
- When a symbol was defined in another formula note in the same paper directory, add a wiki-link cross-reference: `(see [[Price Relative Vector (Z. Jiang et al., 2017)]])`.
- Align pipe characters for readability.

### 4. Discussion (Usage & Context)

One to three paragraphs and/or structured bullet lists after the Where table. No sub-heading needed.

**Rules:**
- Explain the formula's role in the paper's algorithm or framework.
- Describe notable mathematical properties (e.g., self-referential, non-analytic, scale-invariant).
- **Bold** key property names and roles.
- *Italicize* conceptual emphasis.
- Cross-reference related formulas with wiki-links: `(see [[Other Formula (Author, Year)]])`.
- Use inline LaTeX `$...$` for variable references. No display math.
- Bullet lists with tab indentation can be used to show logical chains or implications. Indent sub-bullets with one tab.

---

## References Section

Every formula file ends with a `**References**:` block. This section uses Obsidian `[[...]]` wiki-links exclusively.

```
**References**:
- [[NOTES - Paper Title Here]]
- [[Paper Title Here.pdf]]
- [[Related Formula 1 (Author, Year)]]
- [[Related Formula 2 (Author, Year)]]
```

**Rules:**
- Use `**References**:` (bold, with colon) as the section marker.
- List external sources first:
  1. The NOTES file for the source paper: `[[NOTES - Full Paper Title]]`
  2. The source PDF: `[[Full Paper Title.pdf]]`
- Then list all internal formula notes that are referenced in-text within this file (either in the Where table or the Discussion section).
- Every wiki-link target must **exactly match** the target filename without the `.md` extension.
- Separate the References from the Discussion with **two blank lines**.

---

## Reading Order File

Each paper subdirectory must contain a `0 - Reading Order.md` file that maps the conceptual dependency tree.

**Format:**
- Each entry is a wiki-link to a formula file: `[[Formula Name (Author, Year)]]`
- Top-level (unindented) entries are root concepts or convergence points.
- Indented entries (tab + `-`) under a link signify that the parent concept **directly leads to** the child concept.
- Multiple levels of indentation represent deeper dependency chains.

**Example:**
```
[[Root Concept (Author, Year)]]
- [[Depends on Root A (Author, Year)]]
	- [[Depends on A (Author, Year)]]
	- [[Also Depends on A (Author, Year)]]
- [[Depends on Root B (Author, Year)]]
	- [[Depends on B (Author, Year)]]
[[Convergence Point (Author, Year)]]
- [[Final Result (Author, Year)]]
```

**Rules:**
- Build the tree by tracing formula dependencies: if Formula B uses a quantity defined in Formula A, then B is indented under A.
- When two branches converge into one formula (e.g., a cost-adjusted return merging an idealized return with a cost factor), place the convergence formula as a new top-level entry below both branches.
- Use tabs (not spaces) for indentation.
- No prose, no headings — only wiki-links and structure.

---

## Linking Conventions

All cross-references between raw formula notes use Obsidian wiki-links: `[[Target Note Name]]`.

| Context | Format | Example |
|---|---|---|
| In-text reference | `(see [[Note Name]])` | `(see [[Price Relative Vector (Z. Jiang et al., 2017)]])` |
| Where table cross-ref | Append to definition | `Price relative vector at period $t$ (see [[Price Relative Vector (Z. Jiang et al., 2017)]]).` |
| References list | Bare wiki-link with `- ` prefix | `- [[Portfolio Value Transition (Z. Jiang et al., 2017)]]` |
| Reading Order entries | Bare wiki-link | `[[Formula Name (Author, Year)]]` |

**Critical rule:** The text inside `[[...]]` must be the **exact filename** of the target note (without `.md`). A misspelled or abbreviated link will not resolve in Obsidian.

Do NOT use:
- Plain-text references like "see Formula 1" or "see the price relative vector".
- Obsidian URI links (`obsidian://open?vault=...`) — those are reserved for the Formula Bank.
- Markdown hyperlinks (`[text](url)`).

---

## LaTeX Conventions

Follow the LaTeX conventions table defined in `FORMULA_FORMAT.md`. The key rules are reproduced here:

| Situation | Format | Example |
|---|---|---|
| Vectors | Bold roman: `\mathbf{x}` | $\mathbf{y}_t$, $\mathbf{w}_t$ |
| Scalars | Default italic: `x` | $p_t$, $\mu_t$, $r_t$ |
| Named operators | Upright: `\ln`, `\max`, `\sum`, `\prod` | $\ln(\cdot)$ |
| Element-wise ops | `\odot` (multiply), `\oslash` (divide) | $\mathbf{y}_t \odot \mathbf{w}_{t-1}$ |
| Dot product | Centered dot: `\cdot` | $\mathbf{y}_t \cdot \mathbf{w}_{t-1}$ |
| Transpose | Superscript: `^\top` | $(\cdot)^\top$ |
| ReLU / positive part | Subscript plus: `(\cdot)_+` | $(x)_+$ |
| Display fractions | `\frac{a}{b}` | $\frac{p_t}{p_{t-1}}$ |
| Inline fractions | `a/b` or `\tfrac` | $p_t / p_{t-1}$ |

---

## Workflow: Adding a New Paper

1. **Create the subdirectory** — name it the full paper title.
2. **Create each formula as a standalone `.md` file** — follow File Naming and Formula File Structure above.
3. **Add wiki-links** — as you write each file, cross-reference related formulas in the Where table, Discussion, and References sections.
4. **Create `0 - Reading Order.md`** — once all formulas are written, trace the dependency graph and build the reading order tree.
5. **Verify link consistency** — ensure every `[[...]]` target exactly matches an existing filename (without `.md`).

---

## Checklist Before Submitting a New Formula Note

- [ ] Filename follows `Formula Name (Author, Year).md` pattern.
- [ ] File has no heading — opens directly with the context paragraph.
- [ ] Context section uses only inline `$...$`, no display math.
- [ ] Formula uses `$$...$$` with blank lines above and below.
- [ ] Where table starts with `Where,` on its own line and covers every symbol.
- [ ] Symbols in the Where table are ordered as they appear in the formula.
- [ ] Cross-references use `[[...]]` wiki-links with exact filenames.
- [ ] No plain-text references like "Formula 1" or "see the price vector".
- [ ] Discussion section uses bold for key properties and italics for emphasis.
- [ ] `**References**:` section is present with NOTES, PDF, and related formula links.
- [ ] `0 - Reading Order.md` is updated to include the new formula in the dependency tree.
