# Hierarchical Context Management for Agents

## Core Observation

An LLM agent's context window is a scarce resource. Most information an agent
might need -- file contents, filesystem structure, process output, conversation
history, domain knowledge -- is hierarchically structured. Today, agents
interact with this information through flat tool-call results appended linearly
to the conversation history, forcing the model to mentally reassemble structure
from fragments.

The central idea here is that **expand/collapse is a universal interaction
primitive for context management**. Files, filesystems, running processes,
knowledge bases, and even the conversation history itself can all be modeled as
hierarchical documents that the agent navigates by selectively expanding and
collapsing regions. This unifying abstraction could replace the ad hoc
collection of `read_file` / `list_directory` / `search` tools that most agents
use today.


## Idea 1: Content-Aware File Reading (Outlining)

### The Problem

When an agent reads a file that exceeds what fits comfortably in context, the
typical approach is to truncate to some fixed number of lines and note how many
were elided. This gives the model an arbitrary prefix of the file, which is
rarely the most useful slice.

### The Idea

Instead of blind truncation, return a **structural outline** of the file --
analogous to the "outlining" or "code folding" view in an IDE. Top-level
declarations (classes, functions, headings) are shown with their signatures and
boundaries, while their bodies are collapsed:

```
  1| #include <string>
  2| #include <vector>
  3|
  4| enum class TokenKind {
   |   ... (14 lines)
 18| };
 19|
 20| struct Token {
   |   ... (8 lines)
 28| };
 29|
 30| class Lexer {
   |   ... (42 lines)
 72| };
```

The agent sees the full top-level structure and can then request specific
regions by line number.

### Approaches to Generating Outlines

- **Language-specific parsers**: tree-sitter, Python's `ast` module, or LSP
  symbols can identify syntactic structure precisely.
- **Indentation heuristic**: for unknown file types, treat indentation level as
  a proxy for nesting depth. Collapse everything beyond depth 0 or 1.
- **Markdown headings**: for `.md` files, collapse to heading hierarchy with
  the first line of each section as a preview.

### Prior Art

- **Aider's "repo map"**: Uses tree-sitter to extract a structural map of the
  entire repository (class names, function signatures) and includes it in the
  agent's context. This is one of Aider's most impactful features for coding
  performance. See: https://aider.chat/docs/repomap.html
- **Cursor's file reading**: Returns line-numbered output and supports offset/
  limit parameters, with semantic search returning "chunk signatures" (function
  and class signatures) as an outlining mechanism.

### Assessment

| Dimension | Rating |
|-----------|--------|
| Proven in practice? | **Yes** -- Aider's repo-map demonstrates clear value |
| Implementation difficulty | **Low** -- self-contained change to the file reading tool |
| Value for Thorn | **High** -- immediately helps with evaluation scenarios |


## Idea 2: Open Files with Expand/Collapse

### The Problem

After an initial read of a file, an agent that needs more detail makes
additional `read_file` calls for specific line ranges. Each call appends a new
tool-result to the conversation history. The model must mentally stitch these
overlapping fragments together, and stale/redundant fragments consume context.

### The Idea

Replace `read_file` with a richer set of tools that manage a **persistent view
state** for each "open" file:

- `open_file(path)` -- adds the file to the agent's set of open files; the
  initial view is an outline (as in Idea 1).
- `expand_region(path, line)` -- expands the collapsed region containing
  `line`, revealing its contents in the view.
- `show_line(path, line)` -- ensures `line` and its immediate context are
  visible (expanding as needed).
- `collapse_region(path, line)` -- re-collapses a region the agent no longer
  needs.
- `close_file(path)` -- removes the file from the open set entirely.

The key difference from sequential `read_file` calls is that the **current
view** of each open file is a single, coherent, up-to-date representation,
rather than a chain of partial snapshots scattered through the history.

### Where Does the View Live in Context?

Three options, in rough order of practicality:

1. **Pinned to the most recent tool call**: The tool-result for the most recent
   operation on `foo.h` contains the current view. Earlier tool-results for
   `foo.h` are replaced with a short note like `"(see current view of foo.h
   below)"`. This is the most natural fit for existing message formats.

2. **Injected as a preamble**: An `OpenFileManager` maintains view state, and
   before each LLM call, the current views are injected as a system-prompt
   section or synthetic user message. Past tool-results become short
   acknowledgments. Simpler to implement than (1) since it avoids mutating
   past messages.

3. **Pinned to system prompt**: All open file views live in the system prompt.
   Simplest but most rigid, and impractical if many files are open.

### Prior Art

- **MemGPT / Letta** (Packer et al., 2023, "MemGPT: Towards LLMs as Operating
  Systems"): Draws the explicit analogy between the context window and physical
  RAM. The agent has tools to page information in from "archival storage"
  (long-term, searchable) and "recall storage" (conversation history), and to
  page information out when context is full. This is the same core idea but at
  a coarser granularity -- whole documents in/out rather than regions within
  documents. https://arxiv.org/abs/2310.08560

### Assessment

| Dimension | Rating |
|-----------|--------|
| Proven in practice? | **Partially** -- MemGPT shows agents can self-manage context, but overhead from frequent paging decisions is a real cost |
| Implementation difficulty | **Medium** -- requires per-file view state and a strategy for presenting it in context |
| Value for Thorn | **Medium-High** -- reduces redundant content in context, gives agent finer-grained control |

### Design Refinement: Expand-Only Tools with Automatic Eviction

A key concern with giving the agent both expand and collapse tools is
**thrashing**: MemGPT found that agents sometimes enter paging loops (bringing
something in, forgetting why, paging it out, then needing it again). This
happens because the MemGPT design asks the agent to be both the application
(doing the actual task) and the kernel (managing its own memory).

A better separation of concerns: **give the agent only tools that bring
information into context** (`open_file`, `expand_region`, `show_line`), plus a
voluntary `close_file` for explicit "I'm done with this." The framework handles
eviction automatically when context pressure exceeds a threshold.

This is more akin to how a user-space application relates to virtual memory: the
application allocates and frees what it needs, but the OS handles paging
decisions. Asking an agent to simultaneously do its primary task and act as its
own virtual memory manager is a heavy cognitive burden, and one that
current-generation models don't handle reliably.

The analogy to human cognition reinforces this: we don't typically make active
choices about what to drop from working memory. Things decay, get displaced, or
lose salience. The deliberate "I should stop thinking about X" decision is
unusual for humans and apparently also unnatural for LLMs.

For the automatic eviction mechanism:

- **LRU** is a natural baseline heuristic: evict (collapse) the content the
  agent has interacted with least recently. This works well for files and
  filesystem views, where recency is a strong signal of relevance.
- **Hysteresis** should be applied to the compaction trigger to avoid
  oscillation when the agent's working set is close to the context budget.
  A standard high-water/low-water approach (e.g., trigger compaction at 80%
  of budget, compact down to 60%, don't compact again until 80%) prevents
  the framework from repeatedly compacting and the agent from repeatedly
  re-expanding.
- More sophisticated eviction (using a model to assess what's currently most
  relevant) is possible but puts us back in the territory of model-guided
  compaction. Still, having the framework make these decisions in a batch
  -- rather than asking the agent to interleave eviction decisions with its
  primary task -- preserves the separation of concerns.


## Idea 3: Filesystem Tree as Expandable Hierarchy

### The Idea

The filesystem has the same hierarchical expand/collapse structure as file
contents. Instead of flat `list_directory` calls, the agent maintains an
expandable tree view of the workspace:

- `expand_directory(path)` -- shows the children of a directory.
- `collapse_directory(path)` -- hides the children again.

The tree view could be presented alongside open files in a "workspace context"
preamble. This maps directly to what IDE file explorers show to humans.

### Prior Art

- Every IDE's file tree is this.
- Aider's repo-map implicitly provides a flattened version.

### Assessment

| Dimension | Rating |
|-----------|--------|
| Proven in practice? | **Yes** (conceptually -- it's what every IDE does for humans) |
| Implementation difficulty | **Low-Medium** -- similar state-management pattern as open files |
| Value for Thorn | **Medium** -- helpful for orientation in unfamiliar codebases |


## Idea 4: Process Output as a Managed View

### The Idea

A running process that produces output (a build, a test suite, a server) is
essentially a file that grows over time. The same expand/collapse metaphor
applies: the agent might want to see the last N lines, or expand a region
around an error, or collapse a wall of passing-test output.

The interesting wrinkle is that the content changes behind the agent's back.
But this is also true of files on disk -- we're just accustomed to tool-call
results reflecting a point-in-time snapshot. A "managed view" of a process
could show a snapshot that updates on the agent's next interaction with it.

### Assessment

| Dimension | Rating |
|-----------|--------|
| Proven in practice? | **Yes** (conceptually -- this is Unix's "everything is a file") |
| Implementation difficulty | **Low** (given Idea 2 exists, this is a specialization of it) |
| Value for Thorn | **Medium** -- useful for build/test workflows |


## Idea 5: Knowledge Bases as Hierarchical Documents

### The Idea

A knowledge base -- whether it's API documentation, project conventions, or
accumulated findings from prior tasks -- can be modeled as a hierarchical
document (or a tree of documents) with the same expand/collapse navigation.

This goes beyond typical RAG (retrieval-augmented generation), where a
knowledge base is a flat search index. The hierarchical model lets the agent:

- Browse the knowledge base's structure (see what topics exist)
- Expand topics of interest
- Perform search/query operations that expand and highlight matching regions
- Create new documents and organize findings into them

This last point is important: a general-purpose agent should be able to
**author** knowledge-base entries, not just consume them. If the agent
encounters a new API, it could create a knowledge-base document capturing what
it learns, organized hierarchically, for future reference by itself or other
agents.

### Prior Art

- **Agent skills / plugins**: Many agent frameworks (Cursor, Copilot, custom
  agents) support "skills" that are presented to the agent as a brief
  description up front, with the full skill content living in separate files
  that the agent reads on demand. This is already the expand pattern in
  practice: the collapsed view is the one-line skill description in the system
  prompt, and expanding means reading the `SKILL.md` and any referenced
  documents. The hierarchy is clear -- skill list → skill document → supporting
  files -- and the agent decides when to drill down based on the task at hand.

  What existing skill systems generally *don't* do is handle the reverse
  direction: once a skill's content has been paged into context, it typically
  stays there for the remainder of the conversation, even after the relevant
  task is complete. The expand-only + automatic eviction model described under
  Idea 2 would handle this naturally: skill content that hasn't been referenced
  recently would be evicted by LRU when context pressure hits, without
  requiring the agent to decide "I'm done with this skill."

- **RAG systems** (vector-store-backed retrieval): well-established, but treat
  knowledge as a flat search space rather than navigable hierarchy.
- **Reflexion** (Shinn et al., 2023): agents that maintain persistent memory
  across episodes, though the memory model is simpler than a full hierarchical
  knowledge base. https://arxiv.org/abs/2303.11366

### Assessment

| Dimension | Rating |
|-----------|--------|
| Proven in practice? | **Partially** -- skills demonstrate the expand pattern for knowledge in production; RAG is proven for flat retrieval; hierarchical navigation as a general mechanism is underexplored |
| Implementation difficulty | **Medium** -- the document model is straightforward; search/indexing adds complexity |
| Value for Thorn | **Medium-High** -- especially for multi-step tasks that build up domain knowledge |


## Idea 6: Non-Lossy Conversation Compaction

### The Problem

As conversations grow, agents must eventually compact the history. Today's
approaches are lossy:

- **Summarization**: replace old messages with a summary (loses detail).
- **Sliding window**: drop old messages entirely.
- **Prompt compression** (e.g., LLMLingua): compress token sequences while
  preserving semantics (still lossy, and the original is gone).

### The Idea

Model the conversation history itself as a collapsible document. "Compaction"
means **collapsing regions** of the history rather than destroying them. Each
collapsed region gets a short label/summary, and the agent retains the ability
to **re-expand** any region to recover the full detail.

This makes compaction **reversible**. If the agent later realizes it needs
details from an earlier exchange, it can expand that region rather than relying
on a potentially inadequate summary.

The process would work roughly as:

1. **Segment** the conversation into collapsible regions (per-turn,
   per-subtask, or discovered via an LLM call).
2. **Label** each region with a short summary (much cheaper than full
   summarization -- just a phrase or sentence).
3. **Collapse** regions when context pressure demands it, replacing them with
   their labels.
4. **Re-expand** on demand, restoring the original messages to context.

When presenting the partially-collapsed history to the LLM, the message list
is reconstructed with collapsed regions replaced by their labels, and expanded
regions showing the original messages.

### Structural Markers from Agent Planning

An important observation that substantially reduces the difficulty of steps (1)
and (2) above: when an agent works with a TODO list or other plan
representation, it is already annotating its own history with hierarchical
structure as a side effect of normal planning behavior. Consider:

```
Turn 12: "I'll work on these items: A, B, C"
Turn 13-24: [work on A]
Turn 25: "A is done. Starting B."
Turn 26-40: [work on B]
Turn 41: "B is done. Starting C."
...
```

The span from "here's my plan" to "plan complete" is a natural top-level
collapsible region. Each individual TODO item's work (from "starting item B"
to "item B complete") is a natural sub-region. And the TODO list itself serves
as the summary label for the collapsed region -- no LLM call is needed to
figure out the hierarchy or generate labels.

This generalizes beyond TODO lists: any structured agent behavior with explicit
start/end markers creates natural collapsible regions. Tool calls have this
property (call start → result). Error-handling sequences (error → diagnosis →
fix → verify) do too. The more structured the agent's behavior, the easier its
history is to compress without model intervention.

This means that for the common case of an agent working through a plan, the
"Hard" difficulty rating below may be closer to "Medium" -- the expensive parts
(segmentation and labeling) come largely for free from the agent's own
planning structure.

### Prior Art

- **MemGPT's recall storage**: archived conversation history is searchable and
  retrievable, but presented as flat retrieval results, not as a hierarchical
  view with collapse/expand.
- **LLMLingua** (Jiang et al., 2023): prompt compression that removes
  low-information tokens. Different mechanism (token-level rather than
  message-level), and lossy. https://arxiv.org/abs/2310.05736
- No direct prior art for the "collapsible conversation" model as described
  here.

### Assessment

| Dimension | Rating |
|-----------|--------|
| Proven in practice? | **No** -- novel in this formulation |
| Implementation difficulty | **High** in general, but **Medium** when the agent uses structured planning (see above) -- requires view reconstruction and interaction with provider message format expectations, but segmentation and labeling can be derived from the agent's own plan structure |
| Value for Thorn | **High** (if it works) -- eliminates the lossy-compaction tradeoff |


## The Unifying Abstraction

All six ideas share a common model:

> **Information is hierarchical. The agent's context is a view into that
> hierarchy. The agent navigates by expanding regions to see more detail and
> collapsing regions to reclaim context space.**

| Information source | Hierarchy | Expand | Collapse |
|--------------------|-----------|--------|----------|
| File contents | Syntactic nesting (classes, functions, blocks) | Show body of a declaration | Hide body, show only signature |
| Filesystem | Directory tree | List children of a directory | Hide children |
| Process output | Temporal / structural (test suites, build phases) | Show region around an error | Hide passing output |
| Knowledge base | Topic hierarchy | Show details on a topic | Show only topic heading |
| Conversation history | Subtask / turn boundaries | Restore original messages | Replace with summary label |

A single well-designed set of primitives -- `open`, `close`, `expand`,
`search` (which expands matching regions) -- could in principle serve all of
these, with the hierarchy source being the only thing that varies. Per the
design refinement discussed under Idea 2, `collapse` is notably absent from
the agent-facing tools: collapsing is handled automatically by the framework
when context pressure demands it, keeping the agent focused on its primary
task rather than on memory management.


## Dual Use: Agent Context Management and Human-Facing UI

The hierarchical view that an agent maintains for its own context management
is also a natural organizing principle for the human-facing UI of an agent
conversation. Current agent UIs (ChatGPT, Claude, Cursor) present conversation
history as a flat stream of messages. For long-lived conversations, this makes
it difficult for a human to find a specific earlier topic or understand what
the agent currently "has in mind."

If the same hierarchical structure serves both purposes:

- **Browsing**: A human can navigate the collapsed conversation hierarchy to
  find old topics, much as they would browse a table of contents, rather than
  scrolling through a flat message stream.

- **Steering**: A human who expands a collapsed region signals to the agent
  "I want you to reconsider this." This is a much more natural interaction
  than typing "remember what we discussed about X" -- the human is directly
  manipulating the agent's context through the same hierarchy the agent uses
  internally.

- **Transparency**: The set of currently-expanded hierarchy nodes represents
  what the agent "has in mind" at a given point. Showing this to the human
  creates a shared mental model of what's relevant, making the agent's
  behavior more legible and predictable.

This dual-use property is an additional argument for the hierarchical approach
over flat summarization: a summary is useful to the agent but offers the human
no interactive affordance. A collapsible hierarchy is both a compression
mechanism and a navigation interface.


## Suggested Implementation Path for Thorn

### Phase 1: Content-Aware File Reading

- Modify `read_file` in `_tools.py` to accept an optional line limit
  (defaulting to ~200 lines).
- When the file exceeds the limit and no line range is specified, return an
  outline view using indentation-based heuristics as a baseline.
- Optionally integrate tree-sitter for language-aware outlining of common file
  types (Python, C/C++, JavaScript/TypeScript).
- Include line numbers in output so the model can request specific ranges.
- **Estimated effort**: A few days.

### Phase 2: Open File Manager

- Introduce an `OpenFileManager` that tracks open files and their current
  expand/collapse state.
- Add agent-facing tools: `open_file`, `expand_region`, `show_line`,
  `close_file`. Collapse is handled automatically by the framework (see the
  expand-only design refinement under Idea 2).
- Present current file views as a preamble injected before each LLM call
  (option 2 from the discussion above).
- Replace past file-operation tool-results with short notes.
- Implement automatic eviction with LRU and hysteresis when total open-file
  content exceeds a context budget.
- **Estimated effort**: 1-2 weeks.

### Phase 3: Filesystem Tree View

- Extend `OpenFileManager` (or a sibling `WorkspaceNavigator`) to track
  expanded directories.
- Present the workspace tree alongside open files in the context preamble.
- **Estimated effort**: A few days (given Phase 2 infrastructure).

### Phase 4: Non-Lossy Conversation Compaction

- Start with plan-derived segmentation: when the agent works through a TODO
  list or similar plan, use its planning transitions as natural region
  boundaries with the plan items as labels. This avoids the need for
  LLM-based segmentation in the common case.
- Build the "collapsed history" view that reconstructs the message list with
  collapsed/expanded regions.
- Implement automatic collapse with LRU and hysteresis (same pattern as
  Phase 2) when the history exceeds a context budget.
- Add a tool for the agent to re-expand collapsed regions on demand.
- Later: add LLM-based segmentation for history that lacks explicit planning
  structure.
- **Estimated effort**: 1-2 weeks for the plan-derived case; several more
  weeks for the general case.


## Key References

- **Aider repo-map**: https://aider.chat/docs/repomap.html
  Practical demonstration that structural outlines of code significantly
  improve agent coding performance.

- **MemGPT / Letta** (Packer et al., 2023): https://arxiv.org/abs/2310.08560
  "MemGPT: Towards LLMs as Operating Systems." The foundational work on agents
  managing their own context via explicit paging tools. Establishes the
  context-window-as-RAM analogy.

- **LLMLingua** (Jiang et al., 2023): https://arxiv.org/abs/2310.05736
  Prompt compression via token removal. A different (lossy) approach to the
  context pressure problem.

- **Reflexion** (Shinn et al., 2023): https://arxiv.org/abs/2303.11366
  Agents with persistent episodic memory. Relevant to the knowledge-base idea.
