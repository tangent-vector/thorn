# Thorn Workflow Efficiency Improvements

Analysis and proposed mitigations based on the `thorn_cpp` workflow's
`calc` scenario, trial 001
(`evaluation/trials/thorn_cpp/calc/trial_001/trace.jsonl`).

---

## 1. Problem Statement

A side-by-side comparison of the Thorn coordinator-based workflow and the
Claude CLI single-agent workflow on the same `calc` scenario reveals an
enormous efficiency gap:

| Metric | Thorn (`thorn_cpp`) | Claude CLI |
|---|---|---|
| Total tokens | **~1.7 M** | **~10 K** |
| Wall-clock time | **~35 min** | **~2 min** |
| LLM completions | **163** | ~10–15 (est.) |
| Prompt : Completion ratio | **17 : 1** (94.6% prompt) | ~3 : 1 (est.) |

The token budget is dominated by prompt tokens — the system repeatedly
re-processes large contexts across many short-lived agent conversations.
Execution is almost entirely sequential despite the multi-agent hierarchy
(sum of LLM durations ≈ wall-clock time, with only ~53 s spent outside
LLM calls).

Even if the Thorn approach ultimately produces higher-quality, more
modular code, a ~150× token multiplier is unsustainable.  The sections
below identify root causes and propose concrete mitigations.

---

## 2. Root Causes

### 2.1 Recursive Coordinator Nesting

The delegation tree reaches **4 levels deep**:

```
Root → Coordinator@main → Coordinator@parser → Coordinator@parser.lexer → Implementer
```

Every `delegate_to_child` call spawns a new Coordinator that must
orient itself (call `module_status`, `list_submodules`, `read_file`)
before it can delegate further.  Coordinators produce **zero code** —
their entire job is reading state and re-delegating.

**Trace evidence:**
- Coordinator-scoped completions accounted for **~855 K tokens (40%
  of total)** and **115 tool calls (44.7%)**, none of which were
  writes, builds, or tests.
- Average prompt size at depth 4 (13,333 tokens) was ~5× the root
  level.

### 2.2 Rigid 4-Phase Pipeline

Every module — regardless of complexity — passes through the same
sequence:

1. `architect` → decompose, create children
2. `api_designer` → write the header
3. `stub_implementer` → write throwing stubs
4. `test_engineer` → write tests
5. `implementer` → write real code

This means a minimum of 4–5 delegation round-trips per module.  For the
5 modules plus main, that produced **32 sub-agent invocations** in this
trial.  A single-agent approach writes all files in one conversation.

### 2.3 Context Accumulation in Late-Stage Calls

Because each agent conversation accumulates messages (system prompts +
tool calls + tool results + validation retries), prompt tokens grow with
every turn within a single agent session.

**Trace evidence — worst offenders:**

| Agent | Scope Depth | Prompt Tokens | Cause |
|---|---|---|---|
| TestEngineer@parser.lexer | 4 | **129,522** | Read `doctest.h` (7,134 lines) |
| Implementer@parser.lexer | 4 | **40 K–45 K** (×10 calls) | Validation retry loop + file reads |

The top 10 most expensive individual calls consumed **~600 K tokens** —
over a third of the total budget.

### 2.4 No Context Sharing Between Agents

Every new sub-agent starts with a blank conversation and re-reads the
same files from scratch:

| File | Times Read | Distinct Agents |
|---|---|---|
| `src/parser/lexer.h` | 16 | 7 |
| `src/parser/ast.h` | 14 | 8 |
| `src/parser.h` | 12 | 7 |
| `src/repl.h` | 8 | 7 |

`module_status` was called **30 times** total, with individual modules
checked 4–7 times each by different Coordinators re-confirming state
that had not changed.

### 2.5 Error Cascades from Cross-Boundary Issues

The `parser.lexer` module consumed **15.5 minutes** (out of ~35 total)
due to cascading design errors that required cross-role fixes:

1. `api_designer` chose `TokenKind::ERROR` — clashes with a Windows
   macro.
2. `test_engineer` tried to build → failed → `raise_error` (140 s +
   130 K tokens wasted on reading `doctest.h`).
3. Coordinator re-delegated to `api_designer` → renamed to `INVALID`
   (29 s).
4. `api_designer` raised error: cannot fix test references (only has
   header write access).
5. Coordinator re-delegated to `test_engineer` → updated references
   (98 s).
6. `implementer` stored `const std::string&` (dangling reference to
   temporaries) → build/test failed → `raise_error` (275 s).
7. Coordinator re-delegated to `api_designer` → changed to store by
   value (26 s).
8. Coordinator re-delegated to `implementer` → re-implemented (51 s).

**8 delegation round-trips, 619 seconds of rework** — for one leaf
module.  A single agent would have caught `ERROR` immediately and
fixed the dangling reference in-place.

### 2.6 Prescriptive Coordinator Delegation

Coordinators routinely passed **300–500 token task descriptions**
prescribing exact API designs, test scenarios, and implementation
details — information they were not well-positioned to specify because
they hadn't read dependency headers in detail.

Example — delegation to `api_designer@parser.ast`:
> "Define BinaryOp: Add, Subtract, Multiply, Divide...  NumberNode
>  stores double... clone() returns unique_ptr..."

Example — delegation to `stub_implementer`:
> "For clone() methods: return nullptr... For getter methods: return
>  the appropriate member variable..."

This directly contradicts `StubImplementer`'s system prompts (which
mandate throwing on everything) and removes the value of having
specialized roles make local decisions.  The indirect cost is that
coordinators spend extra turns formulating these specs, growing their
own conversation contexts.

### 2.7 Unbounded File Reads

The `read_file` tool returns entire file contents with no size limit.
When the TestEngineer read `tests/doctest.h` (7,134 lines ≈ 100 K
tokens) to investigate a build error, it created the single most
expensive completion in the trace (129,522 prompt tokens).

---

## 3. Proposed Mitigations

### A. Context Injection for Agent Bootstrapping

**Problem addressed:** 2.1, 2.4 — Coordinators and workers waste
multiple LLM turns discovering information that is predictable from
their role and module.

**Approach:** Before the first `agent.prompt()` call, inject synthetic
tool-call/result pairs into the agent's message history corresponding
to the standard discovery operations each role performs:

- **Coordinators:** `module_status(module)` result,
  `list_submodules(module)` result, and the module's own header
  content.
- **api_designer / stub_implementer / implementer:** The module's
  header file, plus headers of direct dependency modules.
- **test_engineer:** The module's header file (to know what to test).

This eliminates 3–5 discovery round-trips per agent, each of which
currently pays the full (and growing) conversation-context cost.

**Estimated savings:**
- ~13,500 tokens per Coordinator startup × ~10 Coordinators =
  ~135 K tokens
- ~4,000 tokens per worker startup × ~26 workers = ~104 K tokens
- **Total: ~200 K – 250 K tokens (12–15% of 1.7 M)**

More importantly, this also eliminates ~50–80 LLM round-trips,
directly reducing wall-clock time.

**Implementation notes:**
- The injection point is in `_run_with_validation`
  ([orchestration.py](workflows/thorn_cpp/template/.thorn/orchestration.py),
  around the `agent.prompt(task, messages=messages)` call).
  Pre-populate `messages` with synthetic assistant-tool-call /
  tool-result pairs before the first prompt.
- Determine which files to inject by inspecting the module's
  `#include` directives (the `dependency_order` tool already
  parses these).
- Inject only what's genuinely needed for the role; avoid
  over-injection, which would recreate the context bloat at turn 0.

### B. Agent Pooling by (Role, Module) Pair

**Problem addressed:** 2.5 — Re-delegation to the same (role, module)
pair discards all prior context and forces a full re-bootstrap.

**Approach:** Maintain a cache of agent instances keyed by
`(role_name, module_name)`.  When `delegate_to_role` is called for a
pair that already exists in the pool, resume the existing agent's
conversation instead of creating a fresh one.

**Estimated savings:**
- In the `parser.lexer` cascade alone, 4 extra delegations started
  fresh agents.  Pooling could save **~100 K tokens** there.
- Across the full run: **~100 K – 150 K tokens**.

**Context growth challenge:**
Over multiple rounds, a pooled agent's conversation grows.  Mitigation
strategies:

1. **Soft restart:** When re-activating a pooled agent, truncate old
   messages and inject a brief summary of prior work (e.g., "You
   previously wrote the lexer header with 6 token types.  The
   following issue was found: ...").
2. **Hard cap:** If the conversation exceeds a threshold (e.g.,
   30 K tokens), discard the pool entry and create a fresh agent
   with context injection (mitigation A).

**Implementation notes:**
- The pool could live as a `dict[tuple[str, str], Agent]` managed by
  the Coordinator or by the orchestration module.
- `delegate_to_role` and `delegate_to_child` in
  [orchestration.py](workflows/thorn_cpp/template/.thorn/orchestration.py)
  would check the pool before constructing a new agent.
- The pool should be scoped to the lifetime of a single top-level
  `coordinate` call.

### C. Bounded File Reads and Search Tools

**Problem addressed:** 2.7, 2.3 — Unbounded `read_file` results
can inject enormous amounts of irrelevant content into agent contexts.

**Approach — three complementary changes:**

1. **Cap `read_file` output.**  Return at most N lines (suggested:
   200).  If the file is longer, return the first N lines plus a
   truncation notice:
   ```
   [File truncated: showing 200 of 7134 lines.
    Use search_file to find specific content.]
   ```

2. **Add a `search_file` tool.**  A grep-style tool that searches a
   file by regex pattern and returns matching lines with context:
   ```python
   @tool
   def search_file(path: str, pattern: str, context_lines: int = 3) -> str:
       ...
   ```
   This lets agents inspect large files surgically without reading
   them wholesale.

3. **Hide vendored files.**  Add `doctest.h` (and similar vendored
   headers) to the `FileAccessLevel.HIDDEN` rules in the `Developer`
   base class in
   [roles.py](workflows/thorn_cpp/template/.thorn/roles.py):
   ```python
   FileAccessRule("tests/doctest.h", FileAccessLevel.HIDDEN),
   ```

**Estimated savings:**
- Prevents the ~100 K token `doctest.h` incident entirely.
- In general, prevents any pathological single-file read from
  dominating the token budget.

### D. Less Prescriptive Coordinator Delegation

**Problem addressed:** 2.6 — Coordinators spend turns formulating
detailed specs and pass long task strings that undermine role
specialization.

**Approach:** Strengthen the Coordinator's system prompts to enforce
concise delegation.  Suggested additions to the Coordinator prompt in
[roles.py](workflows/thorn_cpp/template/.thorn/roles.py):

```
DELEGATION STYLE:
When delegating, describe WHAT needs to be done, not HOW.  Do not
prescribe API designs, enum values, class hierarchies, test
scenarios, or implementation algorithms — those are the sub-agent's
responsibility based on its own expertise and the module's header
comments.

Keep delegation task descriptions to 2-3 sentences focused on the
goal and any constraints.  Example:

  GOOD: "Design the public API for parser.lexer.  This module
  tokenizes arithmetic expressions into a stream of tokens.  See the
  header comments for responsibilities."

  BAD: "Design the API with TokenKind enum having values NUMBER,
  PLUS, MINUS... Define a Lexer class with next_token() and
  peek_token()... Store token values as std::string..."

Similarly, when delegating to stub_implementer, say WHAT to stub
(which module), not HOW to stub (what values to return).
stub_implementer has its own strict rules.
```

**Estimated savings:**
- Direct: ~5 K – 10 K tokens (shorter task strings).
- Indirect: ~50 K – 100 K tokens (fewer coordinator turns spent
  "designing" before delegating, letting workers make faster local
  decisions).

### E. Trace Visibility for Validation

**Problem addressed:** Validation (build/test) runs after every
delegation but is invisible in the trace, making analysis difficult.

**Evidence that validation IS running:** Comparing `scope_exit`
durations to `delegate_to_role` `tool_end` durations shows consistent
2–4 second gaps:

| Delegation | Agent Duration | Tool Duration | Gap |
|---|---|---|---|
| api_designer@parser.ast | 38.9 s | 42.9 s | 4.0 s |
| stub_implementer@parser.ast | 26.6 s | 28.8 s | 2.2 s |
| test_engineer@parser.ast | 262.7 s | 265.3 s | 2.5 s |
| implementer@parser.ast | 58.0 s | 60.9 s | 2.9 s |

**Approach:** Add explicit trace events around the `_run_validation`
call in [orchestration.py](workflows/thorn_cpp/template/.thorn/orchestration.py):

```python
trace_event("validation_start", rules=sorted(rules))
failures = await _run_validation(rules)
trace_event("validation_end", passed=not failures,
            failures=[(name, detail) for name, detail in failures])
```

This makes traces interpretable and also helps spot cases where
validation is slow or failing silently.

### F. System Prompt Restructuring for Prefix Caching

**Problem addressed:** 2.1, 2.4, and a structural concern — the
current system prompts use `{module}` string templates that are
rendered early in the MRO-collected prompt sequence, breaking the
shared prefix between agents of the same role class and limiting the
effectiveness of LLM prompt caching.

**Background — how prompt caching works:**
LLM APIs (notably Anthropic's) can cache the prefix of a request.
If two requests share the same first N tokens, those N tokens are
served from cache at reduced cost (~90% discount) and reduced latency
(~5× faster).  The cacheable region is the system prompt + tool
schemas + the beginning of the message history — in that order.

**Current behavior:**
`_render_system_prompts` in
[_agent.py](../src/thorn/_agent.py) walks the MRO outermost-first,
collecting `system_prompts` from each class, and renders `{module}`
templates immediately.  For an `Implementer@parser.lexer`, the
rendered prompt list is:

1. `"You are working on a C++ project."` (Developer)
2. `_COMMENT_CONVENTION` (Developer)
3. `_FILESYSTEM_CONVENTION` (Developer)
4. `_SUMMARY_GUIDANCE` (Developer)
5. `"You are ONLY responsible for module `parser.lexer` itself..."` (WorkflowRole) — **diverges here**
6. `_MANDATE_WARNING` (WorkflowRole)
7. `"You are implementer@parser.lexer. You write the implementation..."` (Implementer)

In `OpenAIProvider.complete`, these are joined into a single system
message via `"\n\n".join(system_prompts)`.  The `{module}` substitution
at item 5 means the shared prefix between any two agents — even two
`Implementer` agents for different modules — is only ~450–500 tokens
(the Developer base prompts).

**Proposed change:**
Eliminate `{module}` templates from all system prompt strings.  Move
the module identity to a single trailing entry — either a callable
`(self) -> str` (already supported by `_render_system_prompts`) or an
explicit final prompt.

After the change, `WorkflowRole` would look like:

```python
system_prompts = [
    "You are ONLY responsible for the module you are assigned to "
    "-- not its parent, not its children. Each module has its own "
    "responsible agent.",
    _MANDATE_WARNING,
]
```

And each concrete role would strip `{module}` from its instructions:

```python
class Implementer(WorkflowRole):
    system_prompts = [
        "You are an implementer. You write the implementation in "
        "the module's .cpp SOURCE file to satisfy the declarations "
        "in its header.\n\n"
        "FILE ACCESS: You have write access ONLY to the module's "
        ".cpp source file. All other files are read-only.\n\n"
        # ... (rest of generic instructions, no {module}) ...
    ]
```

The identity is set via a callable at the tail of the `WorkflowRole`
prompts (inherited by all roles):

```python
class WorkflowRole(Developer, abstract=True):
    system_prompts = [
        "You are ONLY responsible for the module you are assigned "
        "to -- not its parent, not its children...",
        _MANDATE_WARNING,
        lambda self: (
            f"AGENT IDENTITY: You are "
            f"{type(self).__name__.lower()}@{self.module}. "
            f"Your module is `{self.module}`."
        ),
    ]
```

**Result — shared prefix tiers:**

| Agents compared | Shared prefix (current) | Shared prefix (proposed) |
|---|---|---|
| Two Implementers (different modules) | ~450 tokens | ~1,500–2,000 tokens |
| Implementer vs TestEngineer | ~450 tokens | ~600–800 tokens |
| Two Coordinators (different modules) | ~450 tokens | ~1,800–2,200 tokens |

Tool schemas (which follow the system message in the API request)
extend the cacheable prefix further — agents of the same role class
have identical tools, adding ~500–1,000 tokens to the shared region.

**Estimated impact:**
- For early-turn calls (~2,500 prompt tokens), a 1,500-token cached
  prefix covers ~60% of the prompt.
- For mid-conversation calls (~10,000 prompt tokens), ~15%.
- For the bloated late-stage calls (~45,000), ~3%.
- Across 163 completions, the aggregate cached token volume is
  roughly 163 × 1,500 = ~244 K tokens.  At a 90% cache discount,
  that's equivalent to saving ~220 K tokens in cost.
- Latency improvement: cached tokens process ~5× faster, reducing
  wall time for the system-prompt portion of each call.

**Why it's worth doing even without confirmed caching support:**
1. **Low effort** — restructuring where `{module}` appears in prompt
   strings; the existing callable support provides a natural hook.
2. **Enables future benefit** — if Thorn adds a native Anthropic
   provider or the OpenAI-compatible endpoint gains caching, the
   benefit is automatic.
3. **Compounds with agent pooling (mitigation B)** — for a pooled
   agent, the entire conversation history becomes a cacheable prefix
   for subsequent turns.
4. **Cleaner architecture** — separating "what this role does"
   (generic) from "which module this agent owns" (instance-specific)
   makes prompts easier to reason about, test, and evolve.

**Key files:**
- [_agent.py](../src/thorn/_agent.py): `_render_system_prompts`,
  `_collect_system_prompts`
- [roles.py](workflows/thorn_cpp/template/.thorn/roles.py): all
  `system_prompts` class variables
- [_provider.py](../src/thorn/_provider.py): `OpenAIProvider.complete`
  (may need changes if switching to Anthropic's native API format
  with explicit cache breakpoints)

---

## 4. Expected Combined Impact

| Mitigation | Estimated Token Savings | Primary Mechanism |
|---|---|---|
| A. Context injection | ~200 K – 250 K | Fewer discovery round-trips |
| B. Agent pooling | ~100 K – 150 K | Avoid re-bootstrap on re-delegation |
| C. Bounded file reads | ~100 K | Prevent pathological single-file reads |
| D. Less prescriptive delegation | ~50 K – 100 K | Fewer coordinator turns, shorter tasks |
| E. Trace visibility | (diagnostic only) | Enables future analysis |
| F. System prompt restructuring | ~220 K (cost-equivalent) | Prefix caching across same-role agents |
| **Combined** | **~450 K – 600 K direct + ~220 K cached** | **~30–35% reduction + caching gains** |

With all mitigations applied, the estimated run would drop from ~1.7 M
to ~1.1 M – 1.3 M tokens (direct savings), with an additional ~220 K
tokens of cost-equivalent savings from prompt caching.

### Structural Gap

Even after these improvements, the Thorn approach will remain
significantly more expensive than a single-agent approach for a task of
`calc`'s complexity (~500 lines of C++).  The remaining cost comes from
inherent multi-agent overhead:

- 5 modules × 4 roles = 20 sub-agent conversations minimum, each
  paying system prompt + tool schema + injected context.
- 5+ coordinator conversations orchestrating them.
- Every conversation pays the full prompt on every turn.

The hierarchical design should break even against a single agent when
the codebase is large enough that no single agent can hold the relevant
context in one conversation window.  For a 500-line calculator, that
crossover point is not reached.

---

## 5. Open Questions

1. **Crossover point.** At what codebase size / module complexity does
   the Thorn coordination overhead become worthwhile compared to a
   single-agent approach?  The `calc` scenario is useful as a
   lower-bound stress test but is below the intended operating point.

2. **Adaptive pipeline depth.** Should the number of phases
   (architect → api_designer → stub → test → implement) be adaptive?
   For trivial leaf modules, collapsing api_designer + stub +
   implement into a single "developer" role could cut delegation
   round-trips by 3×.

3. **Context compaction aggressiveness.** For pooled agents, how much
   history should be retained vs. summarized?  Too aggressive
   compaction risks losing important context; too little risks
   unbounded growth.

4. **Parallelism opportunities.** The current execution is almost
   entirely sequential.  Could sibling modules (e.g., `parser.ast`
   and `parser.lexer` when they don't depend on each other) be
   developed in parallel?  This would not reduce tokens but could
   significantly reduce wall-clock time.

5. **Prompt caching availability.** Mitigation F restructures system
   prompts to maximize cacheable prefixes, but the actual benefit
   depends on whether the underlying LLM API supports prefix caching.
   If the current OpenAI-compatible endpoint does not, this is an
   argument for adding a native Anthropic provider (which has explicit
   cache-control support) or for prioritizing agent pooling (which
   creates longer-lived conversations with naturally cacheable
   prefixes).

---

## Appendix: Token Usage by Scope Depth

From the trial 001 trace (163 completions, 1.7 M total tokens):

| Depth | Completions | Total Tokens | Avg Prompt | Avg Completion | Duration |
|---|---|---|---|---|---|
| 0 (root) | 3 | 8,747 | 2,702 | 214 | 27 s |
| 1 (Coordinator) | 6 | 30,482 | 4,742 | 338 | 70 s |
| 2 (depth-2 agents) | 33 | 284,092 | 8,304 | 305 | 277 s |
| 3 (depth-3 agents) | 55 | 446,098 | 7,586 | 525 | 664 s |
| 4 (depth-4 leaf agents) | 66 | 930,734 | 13,333 | 769 | 1,040 s |

Depth 4 accounts for **54.7% of all tokens** — the deepest agents,
running with the longest accumulated contexts, dominate the budget.
