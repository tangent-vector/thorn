# Notes on `coordination.md`

## Session Key Format/Shape

One of the biggest things I think needs to be worked out here is whether we continue to use the current `projects/foo/forks/bar/issues/XYZ` style where we can easily map a session key to a directory path such that the nesting is clean and logical (would work well for things like memory organization, etc.), or if we move to use something like the "typed" `project:foo/fork:bar/issue:XYZ` style and take that notation literally (embedding it into things like filesystem paths for workspaces, memory, etc.).
I'm strongly inclined to think that we're better off striving to keep the current cleaner shape for keys, just with the session templates telling us how to handle things... but I'm open to having my mind changed.

## On the Core Model

I think the most important thing to get right here is the shape of what a "session template" (currently referred to as a "routing template" or just "template" in the document) looks like, and how it functions.

Your breakdown has most of the important details, but I want us to think of this as more than just a routing thing, but rather our way of enumerating what sessions are allowed to exist and what their purpose is. My gut instinct is that the session templates should be defined on a per-agent basis.

Most of the following will match the breakdown in the document, but I want to walk through my mental model to make sure I know how well they align.

- In my mind, the session-key template would describe a fragment of a filesystem path with "holes", and its a matter of policy (how the templates get set up) whether holes are entire path segments or sub-strings of them. A valid session key template could still be `projects/{p}/issues/{f}-{n}`, if the user wanted that.

  The key thing with the session-key template is that it should ideally be reversible. That is, given a session with key `foo/bar/baz`, we should be able to match it against the session-key template for a given session template and, in doing so, know whether the session is an "instance" of that template, and also what the values are for any vital metadata/keys of that template.

  If the session-key mapping is reversible, then it is easy for inter-session notifications to just use the string form of keys, which would already be what gets surfaced to agents.
  I recognize that the `project:foo/fork:bar/...` style makes it easier to extract the key-value information than the simpler paths I like aesthetically, but one key is that in order for extraction from the `k:v` style to be complete (give us all the data that matters about the session), it would be necessary that any relevant tags also appear in the path (e.g., `dms` in `peer:tess/dms/service:telegram` needs to be interpreted as a tag, and thus every segment of the session key needs to be semantically significant)... it's not necessarily a Bad Thing to do it that way (and maybe I'll be convinced its the only/best option), but it constrains the shapes of session keys a lot.

- The "match shape" part is important since that is what defines the routing from an arbitrary notification with tags + k-v pairs to a single "most specific" session template that can then derive the session key.

  One important benefit of the "strongly typed" session key templates like `peer:{p}/dms/service:telegram` is that they actually serve to define the necessary match shape as well. That is, every segment either specifies a required tag to match (`dms`) a required key and specific value (`service:telegram`) or a required key with wildcard value (`peer:{p}`). For the wildcard case it also gives us exactly the information we need to validate whatever the value is (e.g., we know to validate `p` as a `peer` by looking it up in the appropriate registry... once we have one).

  The more simple by also more aesthetic path style for session key templates, like `peers/{p}/dms/telegram` or `projects/{p}/issues/{f}-{i}`, would basically require a reverse mapping that defines the equivalent match shape like `tags: [ "dms" ] pairs: { "peer":"{p}", "service":"telegram" }`

- I believe that by default "parent" session templates shold be defined implicitly just based on their match shape (whether the match shape is explicitly declared or inferred from a "typed" session key template).

  A session template that is "more general" than another is a kind of de facto ancestor. "More general" means requiring a subset of tags and a subset of keys, and for each key requiring either the same value as the more-specific template, or accepting a wildcard.

  Notably, the actual shape of the session key might not matter for more-general-ness. A `service:{s}` session template might naturally be a parent of `peer:{p}/dms/service:{s}` even if its session keys aren't a prefix.

  A session template should ideally inherit configuration state from its ancestors. Similarly, a particular session should inherit state from *its* ancestors. The details of how session-template ancestry/inheritance does or does not align with session ancestry/inheritance is something I'll be interested to see worked out.

- For things like heartbeat policy and cross-tree send policy... I think I'm okay with those being defined on a session template, but they aren't the only important policy stuff that should be defined as part of a template.

  One big thing I haven't see yet is the acknowledgement that we really need a way to associate an `AGENTS.md`-style system prompt with each session template, potentially along with other comparable information like skills, tools, etc.

  We could of course add such things to the JSON config (although as soon as we're talking about embedding multi-line text like a system prompt, I think we're probably in YAML territory, rather than just JSON).

  A more powerful (but complicated) alternative would be to have each session template map to a directory path in the agent's home directory, by replacing the wildcards in the session key template with a fixed `_` (e.g., `~/peer:_/dms/service:_/`). That directory could then be used to store an `AGENTS.md` that would apply to all sessions that are instances of the corresponding template, as well as possibly/eventually something like a `.agents/` directory with skills, etc.
  If we wanted to get really fancy, we could allow for an `AGENTS.md.jinja` instead, so that the content of that file could use `{{peer}}` or `{{service}}` to fill in parts of the system prompt a given session sees.

The key obersvation I'm trying to make here is that this "session template" concept is actually what the gateway system wants in place of a notion like `Agent` sub-classes. Each session template represents a known specialization of context information, and provides an opportunity to aggregate policy guidance, skills, etc. from across many different places, thus allowing the same underlying "agent" to act appropriately to the context of an individual session, effectively "wearing many hats" as a natural consequence of the multi-session architecture, so that in ordinary operation neither the users nor the agent should need to think about it (but when it wants to, or at the user's discretion, the agent can tweak its session-template-specific policies/guidance to better perform the appropriate tasks).

The discussion of "prompt conventions" later in the document invites the question of where the prompt text in question lives, and the idea of having the prompt text attached to the session templates seems like the obvious answer for per-session-template role guidance.

## Heartbeats

This is definitely something that should belong to the per-session-template configuration (and/or be allowed per-session).
Heartbeats as a feature are probably big enough in scope that they need to be considered as a major push of their own, if we want to get them right.

I think heartbeats are probably just one specific case of scheduled tasks, where we'd want to have infrastructure like:

- every session, and every session template can have a configured list of "scheduled tasks" (terminology TBD)

- Each scheduled task would either specify a single exact time (UTC) when it should fire, or a `cron`-style specification of how it should fire in a recurring fashion

- Each scheduled task should include a prompt string, which will be included as the content of the relevant notification

- The configuration of these tasks should be handled as something like `timers.yaml` stored in a place accessible to the agent and its sessions. The agent should be able to find the `timers.yaml` for a given session, or for a session template, and add an entry to it easily.

- The runtime system would either load timer info on startup and then use something like a filesystem watcher to notice updates/changes, or just rely on periodically scanning for relevant timer files in the memory filesystem hierarchy.

- When a timer fires, a notification is placed in the corresponding session's inbox (for a single-session timer), or in the inbox of every existing session that is an instance of the given template. A one-time-only timer gets deleted from the corresponding YAML file once notifications have been posted

The key idea here is to make timers in general (and heartbeats in particular) something that an agent can set up for itself and its own sessions.

