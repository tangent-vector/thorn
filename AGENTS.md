## Development Workflow

This project uses **`uv`** as its package manager. Do not use `pip install`, `python -m pytest`, or other ad-hoc commands.

- **Set up / sync the environment:** `uv sync --all-extras`
  This creates (or updates) the `.venv/` virtual environment and installs all dependencies, including every optional-dependency group (test, eval, mcp, gitlab, github).

- **Run tests:** `uv run pytest`
  To run a single test file: `uv run pytest tests/test_foo.py`

- **Run any project command:** `uv run <command>`
  For example: `uv run thorn ...`

---

## What counts as a priority

- Thorn is developed as an end-user application.
  The priorities, in order, are:
  1. The `thorn` CLI (`thorn run`, `thorn chat`, and related subcommands).
  2. The `thorn serve` gateway and the agency / runtime infrastructure it depends on.
  3. Everything else.

- The library-style public API (`Agent`, `prompt()`, `@skill`, `@tool`, `wrap_function`, `ALL_BUILTIN_TOOLS`, etc.) exists to serve the CLI and the gateway.
  It is **not** a stable contract.
  When a legacy library-API shape comes into tension with a forward-looking plan (sandboxing, coordination, skills-as-the-successor-to-user-tools, etc.), the library API is the thing that gives.

- Concretely: do not preserve library-API surface for its own sake, do not add compatibility shims to keep external embedders working, and do not treat "someone might be calling this as a library" as a reason to hold back on a design change.
  If in doubt about whether something is load-bearing, check whether the CLI or gateway depends on it; if they don't, it's safe to change or remove.

---

- Don't guess or presume. If you aren't exceptionally confident that you understand the situation, the user's intent, etc. then you should ask clarifying questions.
  We are collaborators, and you should leverage the things that the user is good at and the knowledge they have that you may lack.

  DO interrupt coding work to ask questions if you run into an unresolved question around design or policy.

- Make sure any code you add builds, has no linter issues, and passes all tests.

- If you see build/lint/test failures that don't seem related to your own work, you are still responsible for addressing them or (if you cannot fix them) bringing them to the attention of the user.
  Do NOT shrug off any kind of issues as not your problem.

- Make sure to add adequate test coverage for code you introduce or change.
  Ensure that you are testing the functionality a user of your code/API would actually care about, rather than just adding fluff.

- Do not add unit tests whose primary purpose is to police exact wording,
  headings, examples, or narrative coverage in `README.md`, files under
  `docs/`, comments, or docstrings. Documentation correctness is handled
  through development review, release rehearsals, and tests of the executable
  behavior the docs describe.

  Tests may use documentation-shaped files as inputs when that file handling is
  product behavior, such as Markdown outline parsing, project-root detection, or
  `AGENTS.md` loading. Those tests should assert the behavior of the code, not
  freeze prose that humans need to edit freely.

- When working on code, follow this general sequence, and perform a self-review of your work after each step:
  - Make any necessary changes to the architecture and decomposition of modules
  - Make any necessary changes the public API surface area of modules.
    Ensure documentation comments / docstrings properly reflect the intended behavior
  - Author/update black-box tests related to the new/changed API surface area, based on their documented behavioral contract
  - Author/update the implementations of the new/changed API surface area (and supporting non-public code), based on documented behavior and testing against the previously-authored tests

- Flat code is better than deeply nested code.
  Handle-early-out cases in functions and loops first, so that the main or most complicated path can remain less nested.

- Define explicit types for things rather than just using strings, integers, etc.
  For example, if you have a function that takes `user_id: str`, then that should almost certainly be `user_id: UserID`.

  Anything that could count as "stringly typed" programming is forbidden.

- Prefer actual class hierarchies over ad hoc tagged union types unless there is a clear reason why you need to use a tagged union.

- Scrutinize every boolean-typed field/parameter you add.
  Is it really just two states, and will it realistically always remain that way?
  Should you be defining a new subtype rather than adding yet another flag to an existing one?
  Does the receiver of this boolean value actually have all the info they need, or is there other associated data (in which case you probably wanted an `Optional[T]` or `T | None`)? 

- Comments should be used to explain *why* you are doing something in a particular way, or using a particular design/architecture approach. They should discuss alternatives considered, where appropriate.

  Comments that just state *what* the code is doing are a code smell.
  If code is complicated enough that you need a comment to explain what it's doing, then you should be defining cleanly named helper routines, temporaries, or whatever it takes to make the code more obvious to somebody reading the code itself.

- Names should favor clarity and accuracy over brevity.
  Variables/attributes/parameters should accurately name what they hold/are, functions should accurately describe what they do or compute, etc.
  When clear and accurate names are verbose or ugly, they can help us identify where better designs are called for.

- If something would typically be rendered in all caps in English (ID, HTML, API, etc.) then it should either appear in all caps or all lowercase in names (`user_id`, `UserID`, `html_node`, `HTMLNode`, etc.).
  Pascal-case conventions that mix this up (`HtmlNode`) are ugly and bad.
