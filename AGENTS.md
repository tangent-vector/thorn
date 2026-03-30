- Don't guess or presume. If you aren't exceptionally confident that you understand the situation, the user's intent, etc. then you should ask clarifying questions.
  We are collaborators, and you should leverage the things that the user is good at and the knowledge they have that you may lack.

- Make sure any code you add builds, has no linter issues, and passes all tests.

- If you see build/lint/test failures that don't seem related to your own work, you are still responsible for addressing them or (if you cannot fix them) bringing them to the attention of the user.
  Do NOT shrug off any kind of issues as not your problem.

- Make sure to add tests for code you introduce, whenever possible.
  Ensure that you are only adding tests for the functionality a user of your code/API would actually care about; we don't need to pad out test counts with fluff.

- Flat code is better than deeply nested code.
  Handle-early-out cases in functions and loops first, so that the main or most complicated path can remain less nested.

- Define explicit types for things rather than just using strings, integers, etc.
  For example, if you have a function that takes `user_id: str`, then that should almost certainly be `user_id: UserID`.

  Anything that could count as "stringly typed" programming is forbidden.

- Prefer actual class hierarchies over ad hoc tagged union types unless there is a clear reason why you need to use a tagged union.

- Scrutinize every boolean field/parameter you add. Is it really just two states, and will it realistically always remain that way? Should you be defining a new subtype rather than adding yet another flag to an existing one? Does the receiver of this boolean value actually have all the info they need, or is there other associated data (in which case you probably wanted an `Optional[T]` or `T | None`)? 

- Comments should be used to explain *why* you are doing something in a particular way, or using a particular design/architecture approach. They should discuss alternatives considered, where appropriate.

  Comments that just state *what* the code is doing are a code smell.
  If code is complicated enough that you need a comment to explain what it's doing, then you should be defining cleanly named helper routines, temporaries, or whatever it takes to make the code more obvious to somebody reading the code itself.
  