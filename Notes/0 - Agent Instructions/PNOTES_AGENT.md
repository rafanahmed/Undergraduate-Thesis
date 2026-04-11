# Paper Notes (PNOTES) — Agent Instructions

> **Audience:** AI agents tasked with producing or completing paper notes in the `Notes/Paper Notes/` directory.
> Read this file in full before writing any PNOTES file. Follow every convention exactly.

---

## What PNOTES Are

A PNOTES file is a **single standalone Markdown document** that distills an academic paper into structured, annotated notes. The notes serve two roles simultaneously:

1. **Comprehension artifact** — a section-by-section walkthrough that tracks the paper's own narrative arc, enriched with *Context* annotations that unpack non-obvious ideas.
2. **Reference document** — a self-contained resource a reader can revisit months later and quickly recall the paper's problem, method, results, mathematics, and relevance.

PNOTES are **not summaries**. They preserve the logical structure of the source paper, including its mathematical machinery, while adding interpretive scaffolding that makes the content accessible on re-read.

---

## Directory Structure

```
Notes/
├── 0 - Agent Instructions/
│   └── PNOTES_AGENT.md                      ← this file
│
├── Paper Notes/
│   ├── 1 - Thematic Cluster Name/
│   │   ├── PNOTES - Full Paper Title A.md
│   │   └── PNOTES - Full Paper Title B.md
│   │
│   ├── 2 - Another Cluster Name/
│   │   └── PNOTES - Full Paper Title C.md
│   │
│   └── ...
│
└── Relevant Misc. Notes/
```

Papers are grouped into **thematic subdirectories** (e.g., `1 - Log Utility & Portfolio RL`). The directory name is a short descriptive label, not a paper title. When a PNOTES file is created, it is placed in the thematic subdirectory that best fits the paper's topic. If no existing subdirectory fits, create a new one with the next sequential number.

---

## File Naming

Every PNOTES file follows this pattern:

```
PNOTES - Full Paper Title.md
```

**Rules:**
- The prefix is always `PNOTES - ` (capital P, followed by a space-dash-space).
- **Full Paper Title** is the complete, unabbreviated title of the paper as it appears on the first page of the PDF.
- Do not include author names, year, or venue in the filename.

---

## Inputs

The agent may receive up to three types of input. The PDF is always required. The other two are optional and may appear in any combination.

| Input | Required? | What It Is |
|---|---|---|
| **PDF** | Always | The source paper. The primary authority on all claims, equations, and structure. |
| **NotebookLM Summary** | Optional | An AI-generated overview of the paper produced by Google NotebookLM. A high-level map of the paper's terrain — useful for orientation, not for detail. |
| **Partial PNOTES** | Optional | An existing, incomplete PNOTES file that needs to be finished. |

### How to Use a NotebookLM Summary

A NotebookLM summary is a **pre-reading scaffold**, not a substitute for the PDF. It typically provides:
- A plain-language overview of the paper's purpose and contributions.
- A condensed walkthrough of the main sections and arguments.
- Simplified explanations of key concepts and results.

**Rules for incorporating a NotebookLM summary:**
- **Read the summary first.** Use it to build a mental map of the paper before diving into the PDF. It tells you what the paper is about, where the key ideas live, and how the argument flows.
- **Then read the PDF.** The summary orients; the PDF is the ground truth. Every claim, equation, and detail in the PNOTES must be verified against the PDF. Do not propagate errors, oversimplifications, or omissions from the summary into the notes.
- **Use the summary to calibrate depth.** If the summary highlights a section as central to the paper's contribution, that section should receive deeper treatment (more *Context* blocks, more detailed bullet decomposition) in the PNOTES. If the summary treats something as background, it is likely background — but verify against the PDF.
- **Use the summary to identify structure.** NotebookLM summaries often surface the paper's logical skeleton (problem → method → key result → limitation) more clearly than the paper itself. Use this to plan the PNOTES structure before writing.
- **Do not quote or cite the summary.** The summary is a tool for the agent, not a source for the reader. PNOTES should read as if the agent engaged directly with the PDF. No references to "the summary says..." or "according to NotebookLM...".
- **Resolve conflicts in favor of the PDF.** If the summary and the PDF disagree on a claim, a definition, or a result, the PDF wins. Always.

### Input Modes

The combination of inputs determines the agent's operating mode.

### Mode A — From Scratch

**Input:** PDF + optionally a NotebookLM summary.
**Output:** A complete PNOTES file with all sections described below.

**Procedure:**
1. If a NotebookLM summary is provided, read it first to orient yourself — identify the paper's problem, method, key results, and structural arc.
2. Read the PDF in full. Use the summary's map to navigate efficiently, but do not skip any section of the PDF.
3. Identify the paper's structural sections (Abstract, Introduction, Methods, Results, Discussion, Conclusion — or whatever headings the paper uses).
4. Build the PNOTES file following the File Structure below.

### Mode B — Completing Partial Notes

**Input:** An existing, incomplete PNOTES file + the source PDF + optionally a NotebookLM summary.
**Output:** The same file, completed.

**Procedure:**
1. Read the existing PNOTES to determine what has already been written and where the notes stop.
2. If a NotebookLM summary is provided, read it to understand the full scope of the paper — especially the sections not yet covered by the partial notes.
3. Read the PDF from the point where the existing notes leave off (though you may need to re-read earlier sections for context).
4. Continue writing in the same voice, style, and level of detail established by the existing content.
5. Do not rewrite or restructure sections that are already complete unless they contain clear errors.
6. Add any missing structural sections (MATHEMATICS, Structured Analysis, Integration) if absent.

---

## File Structure

Every complete PNOTES file has **four parts** in this exact order:

```
┌──────────────────────────────────────┐
│  I.   PDF Link                       │
│  II.  Main Body                      │
│  III. MATHEMATICS                    │
│  IV.  Structured Analysis            │
└──────────────────────────────────────┘
```

Each part is separated from the next by a horizontal rule (`---`).

---

### Part I — PDF Link

The file opens with an Obsidian wiki-link to the source PDF on its own line:

```
[[Full Paper Title.pdf]]
```

**Rules:**
- The link text must match the PDF filename exactly (including `.pdf`).
- This is the first line of the file. No heading precedes it.
- Leave one blank line after the link before Part II begins.

---

### Part II — Main Body

The main body is a **section-by-section walkthrough** of the paper. It follows the paper's own organizational structure rather than a rigid prescribed template. Different papers have different sections — a methods-heavy empirical paper will look different from a theory paper or a survey.

#### Heading Conventions

- Use `#` for major paper sections (Abstract, Introduction, Methods, Results, Conclusion, etc.).
- Use `##` for subsections within those (e.g., `## Dataset and Experiments` under `# Methods`).
- Use `###` for named concepts, named equations, or sub-subsections within those.
- Bold the heading text when it is a thematic label you are assigning rather than a heading lifted verbatim from the paper (e.g., `# **Abstract**:`).

#### Content Conventions

Within each section, follow these formatting rules:

| Element | Convention | Example |
|---|---|---|
| Key terms / framework terminology | **Bold** | **EIIE topology**, **Kelly Criterion** |
| Conceptual emphasis | *Italics* | *risk-aware*, *model-free* |
| Mathematical symbols in prose | Inline LaTeX `$...$` | $\mathbf{w}_t$, $\pi^*$ |
| Key equations in the body | Display LaTeX `$$...$$` with blank lines | See below |
| Cross-references to other PNOTES | Obsidian wiki-links | `[[PNOTES - Other Paper Title]]` |
| Cross-references within the file | Bold section pointers | **See MATHEMATICS §2** |

#### Bullet Structure

The main body is **bullet-heavy**. Use bullets and sub-bullets to decompose arguments, enumerate components, and layer detail:

- Top-level bullets capture the main claim or component.
	- Indented sub-bullets add detail, nuance, or implication.
		- Deeper indentation adds further specificity or worked examples.

Use tab indentation (not spaces) for each nesting level.

#### Display Math in the Main Body

When a key equation appears as part of the paper's narrative, include it inline using display math:

```
$$
\pi^* = \frac{\mu - r_f}{R_a \sigma^2}
$$
```

Always surround `$$...$$` with blank lines. Equation tags `\tag{N}` are optional in the main body — use them only if the paper itself numbers the equations and you need to reference them later (e.g., in MATHEMATICS or *Context* blocks).

---

### *Context* Blocks

*Context* blocks are the defining feature of PNOTES. They are **inline annotations** that unpack a concept, connect it to broader knowledge, or explain why something matters. They appear wherever a claim, term, or equation would benefit from elaboration.

**Format:**

```
- *Context*:
	- Explanation paragraph or bullets here.
```

**Rules:**
- Always begin with `- *Context*:` (italicized, with colon) on its own bullet line.
- The explanation follows as indented sub-bullets or prose beneath.
- *Context* blocks may contain inline LaTeX, bold terms, cross-references, and nested bullets.
- Write *Context* blocks in a clear, explanatory voice — as if a knowledgeable colleague is walking you through the concept at a whiteboard.
- *Context* blocks should illuminate, not just restate. They exist to:
	- Translate formal notation into intuition.
	- Connect an isolated claim to the broader landscape.
	- Expose hidden assumptions or subtle implications.
	- Provide concrete examples that ground abstract ideas.
	- Highlight *why* something matters, not just *what* it is.
- Include *Context* blocks liberally. When in doubt, add one.

---

### Part III — MATHEMATICS

After the main body and a `---` separator, every PNOTES file includes a standalone `# MATHEMATICS` section. This section formalizes the paper's core mathematical machinery in a self-contained, pedagogical walkthrough.

#### Purpose

The MATHEMATICS section is **not a repetition** of the main body. Where the main body tracks the paper's narrative and intersperses equations as they arise, the MATHEMATICS section presents the paper's formal framework as a **unified, dependency-ordered derivation chain**. It answers: *"If I only read this section, would I understand the mathematical scaffolding of the paper?"*

#### Structure

```
# MATHEMATICS

Opening paragraph framing the mathematical landscape of the paper.

### 1. Section Title

Prose introducing the first concept or derivation...

**Named Equation ($symbol$):**
$$
LaTeX
$$

Prose continuing the derivation or connecting to the next concept...

### 2. Section Title

...
```

**Rules:**
- Open with a paragraph (no heading tag) that frames the mathematical narrative — what the derivation chain establishes and why it matters.
- Number subsections sequentially: `### 1.`, `### 2.`, etc.
- Each subsection covers one logical unit: a definition, a derivation step, a key result, or a transformation.
- Name key equations in bold before their display math: `**Named Equation ($symbol$):**`
- Use display math `$$...$$` for all formal equations, with blank lines before and after.
- Equation tags `\tag{N}` are encouraged here for cross-referencing within the MATHEMATICS section.
- Prose between equations should explain the *transition* — why the next step follows, what assumption is invoked, what changes.
- Use inline LaTeX `$...$` for variable references in prose.
- The ordering of subsections should follow the **logical dependency** of the mathematics (definitions before derivations, primitives before composites), which may differ from the order of appearance in the paper.
- Cross-reference the main body where appropriate: *"As discussed in the Methods section above, ..."*

#### LaTeX Conventions

| Situation | Format | Example |
|---|---|---|
| Vectors | Bold roman: `\mathbf{x}` | $\mathbf{y}_t$, $\mathbf{w}_t$ |
| Matrices | Bold capital: `\mathbf{X}` | $\mathbf{A}$, $\mathbf{\Sigma}$ |
| Scalars | Default italic | $p_t$, $\mu$, $r_t$ |
| Named operators | Upright: `\ln`, `\max`, `\sum`, `\prod` | $\ln(\cdot)$ |
| Sets | Calligraphic or blackboard bold | $\mathcal{S}$, $\mathbb{R}$ |
| Expectations | `E[\cdot]` or `\mathbb{E}[\cdot]` | $E[\ln W]$ |
| Element-wise ops | `\odot` (multiply), `\oslash` (divide) | $\mathbf{y}_t \odot \mathbf{w}_{t-1}$ |
| Dot product | Centered dot: `\cdot` | $\mathbf{y}_t \cdot \mathbf{w}_{t-1}$ |
| Display fractions | `\frac{a}{b}` | $\frac{p_t}{p_{t-1}}$ |
| Inline fractions | `a/b` or `\tfrac` | $p_t / p_{t-1}$ |

Preserve the paper's own notation wherever possible. When the paper's notation is ambiguous or inconsistent, standardize it and note the change.

---

### Part IV — Structured Analysis

After the MATHEMATICS section and a `---` separator, every PNOTES file ends with a structured analysis that distills the paper into a fixed set of evaluative dimensions. This section is written **after** the main body and MATHEMATICS are complete, as a reflective synthesis.

#### Format

Each dimension is a `###` heading with bold numbering. The content beneath each heading is a mix of prose and bullets.

```
### **1. Problem:**
What problem does this paper address? What gap does it fill?

### **2. Setup:**
What kind of dynamics, action spaces, or environments does the method assume?
Characterize the mathematical and structural setup.

### **3. Key Idea:**
What is the core methodological or conceptual contribution?
State it in 2-3 sentences.

### **4. Assumptions:**
What does the method take for granted?
List both explicit assumptions and implicit ones you identified.

### **5. Limitation:**
Where does the method break down?
What does it not address?

### **6. Relevance & Open Questions:**
How does this paper connect to your broader research agenda?
What questions does it raise? What gaps remain?

---
### Integration:

* **Problem:** One-paragraph synthesis of how this paper's problem framing connects to your line of inquiry.

* **Limitation:** One-paragraph synthesis of the limitations that matter most for your purposes.
```

**Rules:**
- The six numbered dimensions (`Problem` through `Relevance & Open Questions`) are always present.
- The `Integration` section is always present and always comes last, after a `---` separator.
- `### **6. Relevance & Open Questions:**` is a general-purpose reflection point. Use it to connect the paper to your research, note open problems it suggests, identify gaps it leaves, or flag ideas worth pursuing. This dimension adapts to whatever you are currently working on.
- `Integration` synthesizes the most important takeaways from dimensions 1–6 into a compact reference. It always has at least `Problem` and `Limitation` sub-entries.
- Use **bold** for key terms, *italics* for emphasis, and inline LaTeX `$...$` for mathematical references.
- Bullet points with `*` prefix for sub-entries under `Integration`.

---

## Adaptability Across Paper Types

PNOTES are designed to accommodate any academic paper. The main body (Part II) adapts to the paper's own structure. Here is guidance for common paper types:

| Paper Type | Main Body Emphasis | MATHEMATICS Emphasis |
|---|---|---|
| **Empirical / Methods paper** | Dataset, architecture, experimental setup, results tables, ablation studies | Core objective function, loss derivations, update rules |
| **Theory paper** | Theorem statements, proof sketches, assumptions, corollaries | Full derivation chains, proof structure, key lemmas |
| **Survey / Review** | Taxonomy of approaches, comparison tables, evolution of ideas | Unifying formalism (if the survey provides one), key equations from representative papers |
| **Position / Perspective paper** | Central argument, supporting evidence, counterarguments | Formalize the argument's quantitative claims if any; otherwise, MATHEMATICS may be brief |

The MATHEMATICS section is always present. For papers with minimal formal content, it may be shorter but should still capture whatever quantitative scaffolding exists — even if that is a single equation or a formalized problem statement.

---

## Writing Voice and Style

### Prose Style
- Write in **third-person analytical voice** for the main body ("The paper proposes...", "This framework assumes...").
- *Context* blocks may use second-person or first-person plural for a more direct tone ("Think of it as...", "We need both to compute...").
- Avoid filler phrases ("It is worth noting that...", "Interestingly,..."). Be direct.
- Prefer concrete statements over vague ones. Instead of "This is important," explain *why* it is important.

### Density and Depth
- PNOTES are **dense**. Every line should carry information.
- Prefer structured bullets over long paragraphs when enumerating components, comparisons, or logical chains.
- Use paragraphs for narrative flow, conceptual framing, and connecting ideas across subsections.

### Notation References
- When an equation first appears, briefly state what each symbol represents.
- For equations that reuse symbols defined earlier, reference the earlier definition rather than re-defining.
- In the MATHEMATICS section, every symbol must be defined at the point of first use.

---

## Linking Conventions

All cross-references use Obsidian-compatible links.

| Context | Format | Example |
|---|---|---|
| Source PDF | Wiki-link | `[[Paper Title.pdf]]` |
| Other PNOTES | Wiki-link | `[[PNOTES - Other Paper Title]]` |
| Raw Formula notes | Wiki-link | `[[1.0 - Formula Name (Author, Year)]]` |
| Internal section reference | Bold pointer | **See MATHEMATICS §3** |
| External URL | Markdown link | `[Author](https://example.com)` |

**Rules:**
- Wiki-link text must exactly match the target filename without the `.md` extension.
- Do not use Obsidian URI links (`obsidian://open?vault=...`) for PDF references. Use wiki-links: `[[Paper Title.pdf]]`.
- Cross-reference other PNOTES files when the current paper cites or builds on a paper you have already taken notes on.

---

## Workflow Summary

```
Input: PDF + optionally NotebookLM summary + optionally partial PNOTES
         │
         ▼
Step 0:  (If NotebookLM summary provided)
         Read summary for orientation
         │   • Map the paper's problem, method, results, arc
         │   • Identify which sections are central vs. background
         │   • Do NOT treat summary as ground truth
         │
         ▼
Step 1:  Read the PDF in full
         │   • Use the summary's map (if available) to navigate
         │   • Verify all claims against the PDF
         │
         ▼
Step 2:  Identify the paper's section structure
         │
         ▼
Step 3:  Write Part I — PDF Link
         │
         ▼
Step 4:  Write Part II — Main Body
         │   • Follow the paper's own sections
         │   • Add *Context* blocks throughout
         │   • Include display math for key equations
         │
         ▼
Step 5:  Write Part III — MATHEMATICS
         │   • Unified derivation chain
         │   • Dependency-ordered subsections
         │   • Every symbol defined at first use
         │
         ▼
Step 6:  Write Part IV — Structured Analysis
         │   • Problem → Setup → Key Idea
         │   • Assumptions → Limitation
         │   • Relevance & Open Questions
         │   • Integration
         │
         ▼
Step 7:  Verify link consistency
         │   • All [[...]] targets match real filenames
         │   • All PDF links resolve
         │   • All internal §N references match
         │
         ▼
Done
```

For **Mode B** (completing partial notes), enter the workflow at the appropriate step based on where the existing notes leave off. Do not repeat completed steps unless correcting errors. Step 0 (reading the NotebookLM summary) is always worth doing in Mode B if a summary is provided, since it reveals the full scope of the paper including sections not yet covered.

---

## Checklist Before Submitting a PNOTES File

- [ ] Filename follows `PNOTES - Full Paper Title.md` pattern.
- [ ] File is placed in the correct thematic subdirectory under `Paper Notes/`.
- [ ] First line is a wiki-link to the source PDF: `[[Paper Title.pdf]]`.
- [ ] Main body sections mirror the paper's own organizational structure.
- [ ] *Context* blocks appear wherever a concept benefits from unpacking.
- [ ] **Bold** is used for key terms; *italics* for conceptual emphasis.
- [ ] Inline LaTeX `$...$` for symbols in prose; display `$$...$$` with blank lines for equations.
- [ ] `# MATHEMATICS` section is present with numbered subsections and named equations.
- [ ] Every symbol in MATHEMATICS is defined at first use.
- [ ] Structured Analysis section is present with all six numbered dimensions.
- [ ] `Integration` section is present after a `---` separator.
- [ ] All `[[...]]` wiki-links resolve to real filenames.
- [ ] No orphaned sections, no placeholder text, no `TODO` markers.
