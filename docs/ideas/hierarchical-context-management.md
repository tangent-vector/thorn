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

### Practical Considerations

- Every expand/collapse is a tool call, costing tokens and latency. The agent
  may not always make optimal decisions about what to expand. Keeping the tools
  cheap (fast round-trip, minimal output) helps.
- MemGPT found that agents sometimes enter loops: paging something in,
  forgetting why, paging it out, needing it again. The hierarchical
  expand/collapse model may mitigate this since the agent can see *what's
  collapsed* (structure is always visible, only detail is hidden).


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

- **RAG systems** (vector-store-backed retrieval): well-established, but treat
  knowledge as a flat search space rather than navigable hierarchy.
- **Reflexion** (Shinn et al., 2023): agents that maintain persistent memory
  across episodes, though the memory model is simpler than a full hierarchical
  knowledge base. https://arxiv.org/abs/2303.11366

### Assessment

| Dimension | Rating |
|-----------|--------|
| Proven in practice? | **Partially** -- RAG is proven; hierarchical navigation of knowledge bases is underexplored |
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
| Implementation difficulty | **High** -- requires conversation segmentation, label generation, view reconstruction, and interaction with provider message format expectations |
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
`collapse`, `search` (which expands matching regions) -- could in principle
serve all of these, with the hierarchy source being the only thing that varies.


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
- Add `open_file`, `expand_region`, `collapse_region`, `close_file` tools.
- Present current file views as a preamble injected before each LLM call
  (option 2 from the discussion above).
- Replace past file-operation tool-results with short notes.
- **Estimated effort**: 1-2 weeks.

### Phase 3: Filesystem Tree View

- Extend `OpenFileManager` (or a sibling `WorkspaceNavigator`) to track
  expanded directories.
- Present the workspace tree alongside open files in the context preamble.
- **Estimated effort**: A few days (given Phase 2 infrastructure).

### Phase 4: Non-Lossy Conversation Compaction

- Implement conversation segmentation (start with per-turn granularity).
- Add a labeling pass that generates short summaries for each segment.
- Build the "collapsed history" view that reconstructs the message list with
  collapsed/expanded regions.
- Add tools for the agent to re-expand collapsed regions.
- **Estimated effort**: Several weeks of design and iteration.


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
