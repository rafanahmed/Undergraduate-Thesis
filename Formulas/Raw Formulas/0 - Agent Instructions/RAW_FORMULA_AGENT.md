# Raw Formula Notes — Agent Instructions

> **Audience:** AI agents tasked with parsing the Formula Bank and creating standalone formula notes in the `Raw Formulas/` directory.
> Read this file in full before writing any raw formula note. Follow every convention exactly.

---

## How Raw Formulas Differ from the Formula Bank

The `FORMULA_FORMAT.md` file in this directory governs the **Formula Bank** — a single consolidated document where formulas are sections separated by `---`. Raw Formula notes are a **different system**:

| | Formula Bank | Raw Formulas |
|---|---|---|
| Structure | Sections in one `.md` file | Each formula is its own standalone `.md` file |
| Organization | Flat list of entries | Grouped into subdirectories by source paper |
| Headings | `##` with formula name and citation | No heading — the filename is the title |
| Cross-references | Plain-text names (e.g., "see Transaction Remainder Factor") | Obsidian wiki-links: `[[Note Name]]` |
| Navigation | Scroll within one document | `0 - Reading Order.md` dependency tree per paper |

Do NOT mix these formats. When working in `Raw Formulas/`, follow this file. When working in the Formula Bank, follow `FORMULA_FORMAT.md`.

---

## Directory Structure

```
Raw Formulas/
├── 0 - Agent Instructions/
│   ├── RAW_FORMULA_AGENT.md              ← this file
│   └── FORMULA_FORMAT.md                 ← Formula Bank format (separate system)
│
├── Formula Bank (Use When Paper Reading).md
│
├── Full Paper Title Here/
│   ├── 0 - Reading Order.md
│   ├── 1.0 - Formula Name (Author, Year).md
│   ├── 1.1 - Child Formula (Author, Year).md
│   ├── 1.1.1 - Grandchild Formula (Author, Year).md
│   ├── 2.0 - Second Branch Root (Author, Year).md
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
X.Y - Formula Name (First Author, Year).md
```

**Rules:**
- **Number prefix** (`X.Y -`) encodes the formula's position in the dependency tree. The numbering scheme works as follows:
	- Top-level root concepts start at `1.0`, `2.0`, `3.0`, etc.
	- Direct children append a sub-number: `1.1`, `1.2`, `2.1`, etc.
	- Deeper dependencies add another level: `1.1.1`, `1.1.2`, `1.2.1`, etc.
	- A new top-level number (e.g., `2.0`) indicates a **convergence point** — a formula that draws on concepts from multiple earlier branches or introduces a new independent thread.
- **Formula Name** is preserved exactly from the Formula Bank heading (the `## Heading` text in the Formula Bank, minus the citation). If the Formula Bank heading is `## Log Return with Transaction Costs (Z. Jiang et al., 2017)`, the formula name portion is `Log Return with Transaction Costs`.
- **Citation** uses `(First Initial. Last Name et al., Year)` for multi-author papers, or `(Last Name, Year)` for single-author papers.
- If the formula is a general concept not uniquely attributed to one paper's authors (e.g., a standard compounding identity), include a parenthetical descriptor instead: `1.1.1 - Final Portfolio Value (Compounding Objective) (Z. Jiang et al., 2017).md`.

---

## Formula File Structure

Each raw formula `.md` file has **five sections** in this exact order. There is **no heading** — the filename serves as the title.

### 1. Context (Opening Paragraphs)

One or more paragraphs of plain prose. No heading tag, no display math.

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
- If a formula has multiple distinct expressions (e.g., a definition and its logarithmic form), use a `\begin{aligned}...\end{aligned}` block.
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
- When a symbol was defined in another formula note in the same paper directory, add a wiki-link cross-reference in the definition: `(see [[1.0 - Price Relative Vector (Z. Jiang et al., 2017)]])`.
- Alternatively, when a symbol IS the concept from another note, use a bare wiki-link as the definition: `[[1.2.1 - Transaction Remainder Factor (Z. Jiang et al., 2017)]]. Scales the gross return down to reflect commission losses.`
- Align pipe characters for readability.

### 4. Discussion (Usage & Context)

One or more paragraphs and/or structured bullet lists after the Where table. Separated from the Where table by a blank line.

**Rules:**
- Explain the formula's role in the paper's algorithm or framework.
- Describe notable mathematical properties (e.g., self-referential, non-analytic, scale-invariant).
- **Bold** key property names and roles.
- *Italicize* conceptual emphasis.
- Cross-reference related formulas with wiki-links: `(see [[1.1 - Portfolio Value Transition (Z. Jiang et al., 2017)]])`.
- Use inline LaTeX `$...$` for variable references. No display math.
- Bullet lists can be used to show logical chains or implications. Indent sub-bullets with one tab.

### 5. References

Every formula file ends with a `**References**:` block. This section uses Obsidian `[[...]]` wiki-links exclusively.

```

**References**:
- [[NOTES - Paper Title Here]]
- [[Paper Title Here.pdf]]
- [[1.0 - Related Formula (Author, Year)]]
- [[1.1 - Another Related Formula (Author, Year)]]
```

**Rules:**
- Use `**References**:` (bold, with colon) as the section marker.
- Separate the References block from the Discussion section with **two blank lines**.
- List external sources first:
	1. The NOTES file for the source paper: `[[NOTES - Full Paper Title]]`
	2. The source PDF: `[[Full Paper Title.pdf]]`
- Then list all internal formula notes that are referenced in-text within this file (in the Where table or the Discussion section).
- Every wiki-link target must **exactly match** the target filename without the `.md` extension.

---

## Reading Order File

Each paper subdirectory must contain a `0 - Reading Order.md` file that maps the conceptual dependency tree.

**Format:**
- Each entry is a wiki-link to a formula file: `[[1.0 - Formula Name (Author, Year)]]`
- Top-level (unindented) entries are root concepts or convergence points — written as bare wiki-links with no bullet.
- Child entries are indented with `- ` (dash + space) under their parent.
- Deeper dependencies use tab indentation: one tab per nesting level, followed by `- `.

**Example:**
```
[[1.0 - Root Concept (Author, Year)]]
- [[1.1 - Depends on Root (Author, Year)]]
	- [[1.1.1 - Depends on 1.1 (Author, Year)]]
	- [[1.1.2 - Also Depends on 1.1 (Author, Year)]]
- [[1.2 - Another Child of Root (Author, Year)]]
	- [[1.2.1 - Depends on 1.2 (Author, Year)]]
[[2.0 - Convergence Point (Author, Year)]]
- [[2.1 - Final Result (Author, Year)]]
```

**Rules:**
- Build the tree by tracing formula dependencies: if Formula B uses a quantity defined in Formula A, then B is indented under A.
- When two branches converge into one formula (e.g., a cost-adjusted return merging an idealized return with a cost factor), place the convergence formula as a new top-level entry (`2.0`, `3.0`, etc.) below both branches.
- Top-level entries have no bullet — just the bare `[[...]]` wiki-link.
- Child entries use `- ` with tab indentation.
- Use tabs (not spaces) for indentation.
- No prose, no headings — only wiki-links and structure.

---

## Linking Conventions

All cross-references between raw formula notes use Obsidian wiki-links: `[[Target Note Name]]`.

| Context | Format | Example |
|---|---|---|
| In-text reference | `(see [[Note Name]])` | `(see [[1.0 - Price Relative Vector (Z. Jiang et al., 2017)]])` |
| Where table cross-ref | Append to definition | `Price relative vector at period $t$ (see [[1.0 - Price Relative Vector (Z. Jiang et al., 2017)]]).` |
| Where table bare link | Link IS the definition | `[[1.2.1 - Transaction Remainder Factor (Z. Jiang et al., 2017)]]. Scales the gross return...` |
| References list | Bare wiki-link with `- ` prefix | `- [[1.1 - Portfolio Value Transition (Z. Jiang et al., 2017)]]` |
| Reading Order entries | Bare wiki-link (root) or `- [[...]]` (child) | `[[1.0 - Formula Name (Author, Year)]]` |

**Critical rule:** The text inside `[[...]]` must be the **exact filename** of the target note (without `.md`). A misspelled or abbreviated link will not resolve in Obsidian.

Do NOT use:
- Plain-text references like "Formula 1" or "see the price relative vector".
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

## Parsing the Formula Bank

The Formula Bank (`Formula Bank (Use When Paper Reading).md`) is the **input source** for creating raw formula files. When the Formula Bank contains content for a paper, follow this procedure to parse it into individual files.

### Formula Bank Structure

The Formula Bank for a given paper follows this layout:

```
# Document Title

[[Paper Title.pdf]]

---

## Formula Name A (Citation)

Context paragraphs...

$$
LaTeX...
$$

Where,

| Symbol | Definition |
| ------ | ---------- |
| ...    | ...        |

Discussion paragraphs...

---

## Formula Name B (Citation)

...

---
```

**Key elements to identify:**
1. **Document title** — the `# Heading` on the first line. This is descriptive and may differ from the paper title.
2. **PDF link** — the `[[Paper Title.pdf]]` on the second content line. Extract the paper title from this link (strip `.pdf`).
3. **Formula entries** — each `## Heading` starts a new formula. Everything between two `---` separators belongs to one formula.
4. **Formula name** — the `## Heading` text. The citation is in parentheses at the end: `## Formula Name (Author, Year)`. The formula name is everything before the final parenthetical citation.

### Parsing Procedure

**Step 1 — Create the subdirectory.** Use the full paper title extracted from the `[[Paper Title.pdf]]` link. Example: if the link is `[[A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem.pdf]]`, the directory name is `A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem`.

**Step 2 — Inventory all formulas.** Read through every `## Heading` in the Formula Bank and list them in order. Each heading becomes one raw formula file.

**Step 3 — Map the dependency tree.** Before writing any files, trace the dependencies between formulas:
- Read the context, Where table, and discussion of each formula.
- Identify which formulas reference or depend on quantities defined in other formulas.
- Assign number prefixes based on the dependency structure:
	- Independent root concepts → `1.0`, `2.0`, `3.0`, etc.
	- Direct dependents → `1.1`, `1.2`, `2.1`, etc.
	- Deeper dependents → `1.1.1`, `1.1.2`, `1.2.1`, etc.
	- Convergence points (formulas that merge multiple branches) → new top-level number.

**Step 4 — Create each formula file.** For each Formula Bank entry:
1. Extract the formula name from the `## Heading` — preserve this name exactly.
2. Determine the citation format from the heading's parenthetical.
3. Combine the number prefix, formula name, and citation into the filename: `X.Y - Formula Name (Author, Year).md`.
4. Write the file following the Formula File Structure (Section above):
	- Copy the context paragraphs (everything between the `## Heading` and the `$$` block). Remove any leading line like "The core identity that anchors this entire framework is:" — the context should flow directly into the formula without a transitional sentence pointing to the display math.
	- Copy the display formula.
	- Copy the Where table.
	- Copy the discussion paragraphs.
5. **Convert all plain-text cross-references to wiki-links.** The Formula Bank uses plain-text references like "(see Logarithmic Reward)" or "(see Cumulative Wealth Transition)". Replace every one of these with the full wiki-link to the corresponding raw formula file: `(see [[X.Y - Formula Name (Author, Year)]])`.
6. Add the `**References**:` block at the bottom.

**Step 5 — Create `0 - Reading Order.md`.** Build the reading order tree from the dependency map created in Step 3.

**Step 6 — Clear the Formula Bank.** Once all formulas have been successfully parsed into individual files, remove the processed entries from the Formula Bank. Leave the Formula Bank file in place (it will be reused for the next paper).

**Step 7 — Verify link consistency.** Ensure every `[[...]]` wiki-link target in every file exactly matches an existing filename (without `.md`).

---

## Workflow Summary

```
Formula Bank has content
        │
        ▼
Step 1: Create paper subdirectory
        │
        ▼
Step 2: Inventory all ## headings
        │
        ▼
Step 3: Map dependency tree → assign number prefixes
        │
        ▼
Step 4: Create each formula .md file
        │   • Preserve formula names from bank headings
        │   • Convert plain-text refs → wiki-links
        │   • Add References block
        │
        ▼
Step 5: Create 0 - Reading Order.md
        │
        ▼
Step 6: Clear processed entries from Formula Bank
        │
        ▼
Step 7: Verify all [[...]] links resolve
```

---

## Checklist Before Submitting a New Formula Note

- [ ] Filename follows `X.Y - Formula Name (Author, Year).md` pattern.
- [ ] Number prefix correctly reflects position in dependency tree.
- [ ] File has no heading — opens directly with the context paragraph.
- [ ] Context section uses only inline `$...$`, no display math.
- [ ] Formula uses `$$...$$` with blank lines above and below.
- [ ] Where table starts with `Where,` on its own line and covers every symbol.
- [ ] Symbols in the Where table are ordered as they appear in the formula.
- [ ] Cross-references use `[[...]]` wiki-links with exact filenames (including number prefix).
- [ ] No plain-text references like "see Formula 1" or "see the price vector".
- [ ] Discussion section uses bold for key properties and italics for emphasis.
- [ ] `**References**:` section is present with NOTES, PDF, and related formula links.
- [ ] `0 - Reading Order.md` is updated to include the new formula in the dependency tree.
- [ ] All `[[...]]` link targets exactly match existing filenames (without `.md`).
