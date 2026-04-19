# Notes on `coordination.md`

## Session Key Format/Shape

One of the biggest things I think needs to be worked out here is whether we continue to use the current `projects/foo/forks/bar/issues/XYZ` style where we can easily map a session key to a directory path such that the nesting is clean and logical (would work well for things like memory organization, etc.), or if we move to use something like the "typed" `project:foo/fork:bar/issue:XYZ` style and take that notation literally (embedding it into things like filesystem paths for workspaces, memory, etc.).
I'm strongly inclined to think that we're better off striving to keep the current cleaner shape for keys, just with the session templates telling us how to handle things... but I'm open to having my mind changed.

