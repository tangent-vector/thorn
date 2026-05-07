Thorn System Vision
===================

This document describes the architecture and design of the Thorn system as we imagine it should look.
The descriptions here may not match what is actually implemented at present, both in details and in the broad strokes.
The goal of this document is to help provide a "point on the horizon" that can guide incremental implementation choices toward the eventual goal.

Concepts
--------

### Agencies

An **agency** is a collection of agents, their sessions, and supporting services.
An agency can be persisted to disk and version-controlled as a single repository, and it can be loaded into memory by a runtime gateway/daemon process so that its agents can take action, services can send/receive events, etc.

An agency's durable state is peristed as a directory containing files and subdirectories for the state of its agents, services, etc.
We often refer to that directory as the **agency root** directory.

We can broadly distinguish two main cases of agencies that get created:

- **Local** agencies are typically created for use by a single user. By convention, these will often be persisted as `~username/.thorn/`

- **Server**-based agencies are typically hosted on their own machine (or inside a container/VM, etc.), in which case they might have any directory name/path (TODO: define a conventional path to have be the default).

Throughout this document we will make note of important differences in how local vs. server agencies are typically configured or run.

### Agents

An **agent** is an AI individual, having its own memory, accounts on various services, permissions, etc.
Distinct agents should typically use distinct accounts on things like messaging services, just as distinct humans typically should.

A single agent can sustain multiple conversational sessions at once, and can be performing multiple tasks, of different kinds, at once.
Thus it is not necessary to define new agents to enable concurrency, or to perform different kinds of tasks.
The main reason to define a new agent rather than use an existing one should be because of differences in the permissions that are appropriate to each agent: permissions to perform actions and access services/information.

The state of an agent can be divided into a few key pieces:

- The static *configuration* of the agent: e.g., what tools it has access to, what user accounts it has on what services
- The persistent/durable state of the agent's *memory*.
- The various *sessions* of conversation/activity that the agent is (or has been) engaged in.
- The contents of a filesystem *workspace* where the agent authors and manipulates data.

The memory and sessions of an agent together define its durable state, and these are persisted as a subdirectory under a given agency.
If `helper` is an agent and the agency root is `.../.thorn`, then the state of the agent would be persisted as the directory `.../.thorn/agents/helper/`.
Under the agent's directory there are its *home* directory (`.../.thorn/agents/helper/home/`) and a directory for its sessions (`.../.thorn/agents/helper/sessions/`).

The static state of an agent (the ports of itself it shouldn't be allowed to modify freely) are stored under the agent directory as a simple configuration file (`.../.thorn/agents/helper/agent.json`).

The workspace of an agent is more like the desk or computer of a human: a place for in-progress work that, while important, is not as vital to save as the agent's own memory.
The runtime system for an agency should strive not to throw away an agent's workspace, but at the same time agents are expected to be able to recover from loss of workspace data in much the same way a human would: if something you're working on is truly important, it should be backed up, in source control, etc.

### Sessions

A **session** is single logical "thread" of conversation and/or action that an agent is engaged in.
All of the sessions for a given agent share the same memory, but each session has a distinct sequence of messages that are used when requesting completions from an LLM.

Each session also has a logical **inbox** of **notifications**.
Notifications can represent incoming events from chat services, other agents, etc.
A session is active/runnable when it has unhandled notifications in its inbox; the system will schedule runnable sessions, prompting an LLM to read and handle the remaining notifications.

#### Session Keys

Every session of an agent has a unique **session key**.

When a notification is to be sent to an agent, an appropriate session key is determined and then the notification is placed in the session of that agent corresponding to the key (potentially creating such a session if it did not already exist).

At the simplest, session keys are just strings, but in practice they use a path-like notation that ends up being reflected in how sessiosn are actually stored.
For example for a session with a key like `a/b/c` on an agent `helper`, the state of the session would be stored at `.thorn/agents/helper/sessions/a/b/c/`.

#### Session Workspaces

Each session has a workspace within the overall workspace of the containing agent, with a relative path based on the session key.
For example, if the workspace of agent `helper` is at `/workspace/helper/` then the workspace for a session with key `a/b/c` would be `/workspace/helper/a/b/c/`.

Note that all sessions of an agent share the same home directory (that of the agent), but distinct sessions always have distinct session workspaces.

The workspace of a session is effectively just the "current working directory" for the purpose of paths used in tool calls made for that session.
Tool calls made by the session may still access/modify files outside of its specific workspace, unless other measures are taken to prevent such operations.

### Services

A **service** represents an messaging platform, server, etc. that is external to the agency.
An example of a service would be Telegram or a GitLab server.
An agent can have an **account** (including the relevant **credentials**) on one or more services.

An agency is explicitly configured with a list of services it should be aware of, and the static configuration of an agent includes its accounts/credentials on the various services it can receive events from, send messages/requests to, etc.

### Peers

A **peer** is a logical individual (typically a person, but not always) that agents in a given agency might have reason to communicate or interact with.

The configuration of an agency includes a list of known peers.
For each peer, the agency includes information on their account names/numbers on various services, allowing the agency and the agents in it to identify when two accounts (potentially on different services) represent the same individual.

### Routing of Events

When a running agency detects a potentially-interesting **event** coming from a service (e.g., by polling for notifications on GitHub or GitLab), it will route that event to the correct session inbox for handling.

#### Routing to Agents

Most events will clearly indicate the right agent to handle them.
For example, if an event shows a direct message was received for a Discord account, then the correct agent to handle it is the one associated with that account (if any).
When events aren't clearly directed to a single agent, the given service will have to use other criteria to determin which agent(s) should be notified.

#### Routing to Sessions

Once the runtime has determined that a given agent should potentially be notified of an event, it must decide which session on the agent should receive the notificatin in its inbox.

In order to guide routing, we think of each event as producing a combination of:

- A set of string tags defining simple attributes of the event's context (e.g., `direct_message` for events that relate to a direct-messaging channel/conversation).

- A set of key-value pairs (string to string) defining properties of the events context (e.g., mapping the key `peer` to the name of a known peer, or from the key `service` to the name of a service)

As part of the static configuration of an agent, there are **routing rules** that determine where an event should go.
Each routing rule defines:

- A set of tags to look for. The rule matches events that have all the tags, and otherwise doesn't match.

- A set of keys to look for, and either a corresponding value to look for or a wildcard `*`.
  The rule matches on events that have the key, if it has an identical value, or if the rule had a wildcard (`*`) value.

- A template for a session key to use, if this rule matches and is chosen.
  For example, a template might take the form `peers/{peer}/dms/{service}` for a rule that required the `direct_message` tag, and matched both the keys `peer` and `service` with wildcards.

  The `{}`-enclosed names in the session-key template must be keys that the rule matched with a wildcard.
  Not every key that was matched must be present in the session key template.

The runtime matches an event against the declared rules and then picks the *most specific* rule that matched.
A rule that matches on more tags is more specific than one that matches on a subset of those tags.
A rule that matches on more keys is more specific than one that matches on fewer keys.
A rule that matches on a specific value for a key is more specific than one that does a wildcard match.
If no single most-specific rule matches, then an error is logged (the situation represents a configuration problem).

### Memory

The memory of an agent is stored as ordinary files in the agent's home directory.
The agent is given flexibility to read and write those files as it sees fit, but it is also informed of conventions that it should follow and that the runtime system will strive to support and reinforce.

#### `MEMORY.md`

The top-level `MEMORY.md` file in an agent's home diretory is meant to serve as a high-level overview of how the rest of the memory is organized.
The `MEMORY.md` file is authored and maintained by the agent, for its own benefit.
The runtime system treats this file specially, injecting it in as part of the system prompt for the agent.

#### `AGENTS.md`

While a session may also derive `AGENTS.md` and related information (e.g., a `.agents/skills/` directory) from the session's workspace path, an `AGENTS.md` file in the agent's home directory will also be treated as system-prompt guidance for the agent.

Under this convention, an agent in Thorn is allowed to write to its own behavioral instructions.
This is by design, to allow conversations with user's to guide the agent in shaping its own behavior.

#### Session-Aligned Memory Scopes

While not something directly enforced or required by the runtime, conventions and policies will encourage agents to record important information at paths in their memory that correspond to the shape of session keys.
E.g., if a session key like `peers/tess/dms/telegram` is used for Telegram direct messages between the agent and the peer/user `tess`, then the following paths under the agent's home directory are reasonable places to put notes:

- `peers/tess/`: information pertaining to the agent's understanding of the user `tess` and what it likes/expects/etc.
- `peers/tess/dms`: information pertaining to DM conversations that have been had with `tess`
- `peers/`: overview information pertaining to all of the agent's peers, perhaps noting groups or organizations that are relevant

The runtime system will not only encourage agents (via their system prompt) to write down important information to paths following such shapes.
In addition, if extends the policy of automatically loading any `MEMORY.md` and/or `AGENTS.md` files from the agent's home directory to also apply to subdirectories derived from a session's key.
For example, in a session with key `peers/tess/dms/telegram`, if there was a file `peers/tess/MEMORY.md` under the agent's home directory, it would automatically be part of the system prompt for that session.

### Memory Keys

A **memory key** is a collection of information akin to that used when routing events to sessions:

- A set of string tags
- A dictionary of key-value pairs (string-to-string)

Similar to the routing rules, the static configuration of an agent includes a set of rules for mapping a session key to a memory key (almost the inverse of the event-to-session-key mapping operation).
For example:

- A session key of the form `peers/{p}/dms/{s}` might map to a memory key with tag `direct_message` and key-value pairs `peer={p}` and `service={s}`

- A session key of the form `projects/{p}/issues/{f}-{i}` might map to a memory key with key-value pairs `project={p}`, `fork={f}`, `issue={i}`, and `service={f.forge}`

We can define additional paths that even if not used as session keys can be turned into memory keys when referring to filesystem paths in the agent's home directory.
E.g., if there is a session `projects/foo/issues/gitlab-123`, then a directory `~/projects/foo/forks/gitlab/` in the agent's home directory would also be relevant, even if it isn't a strict prefix match on the session key.

(This is all basically trying to add semantic meaning to the directory hierarchy under the agent's home directory, in a way that we can do the mapping deterministically.. which is maybe not even worth it in practice...)

### Journaling

Every agent has access to temporal memory in the form of a **journal**.
The journal for an agent is stored in its home directory as files for the form `journal/YYY/MM/DD.md`.

While an agent's journal is stored as ordinary files, and can be manipulated as such, agents will typically write journal entries with a dedicated tool, which automatically appends a timestamped entry to the current day's journal.
Each journal entry is tagged with both a timestamp, and also the session key of the session that posted the entry.

When prompting an LLM, the system prompt generated for a session will include recent journal entries for the agent, prioritizing entries that are from the same or related sessions.
"Related" here can be identified by translating the session key of the session A being process and the session key B of a session that wrote a journal entry, translating both of those session keys over to memory keys, and then comparing the similarity of the memory keys (how many tags or key/value pairs do they have in common vs. how many differ).

CLI Sessions
------------

The `thorn chat` and `thorn run` tools by default create sessions that run inside the default agent of the local agency for the current user (that is, whatever is in `~<username>/.thorn/`).

The `--agent <name>` option can be used to identify the specific agent to open a session with (not necessarily the default).
The `--model <name>` option can be used to specify a specific LLM model to use for the session; otherwise the model to use will be chosen based on the agent's configuration.

Each invocation of `thorn chat` or `thorn run` creates a fresh session by default, with a key that depends on the current working directory when the command was run, along with a unique ID.
When the CLI app exits, the session is always prompted to perform housekeeping and migrate all relevant information out to the journal or durable memory, since there is no guarantee that the session will ever be revisited.

(slash commands could be added to `thorn chat` to allow a user to re-connect to old sessions, or to explicitly ask that the current session be saved)

When `thorn` CLI sessions connect to the default local agency, they check for a singleton daemon process serving that agency and start it if necessary. All currently-active CLI sessions connect to the same daemon process. The daemon will automatically shut down when no CLI connections have been made for a certain duration (around 1 minute). For scheduled actions (e.g., "dreaming" to better organize memory, etc.), a cron job is used to wake the local agency so it can process any time-based updates it needs on a semi-regular basis.

If the `--server <URL>` option is passed to `thorn run` or `thorn chat` they will not connect to the default local agency, and will instead connect to the agency running on the identified server.

The more detailed control-plane design for local daemons, remote SSH connections, and monitoring clients lives in `agency-control-plane.md`.
