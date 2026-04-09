# Idea: A Language for Describing Multi-Agent systems

The idea discussed here arose from the Thorn system and in particular the recent work to extend it with support for a "gateway" daemon that manages a runtime "agency" of multiple distinct agents, each of which might have multiple different sessions, and with sessions potentially also spawning sub-sessions (aka "sub-agents") or even, potentially, entire new agents (with their own memory/workspace).
During discussion of how different agent systems/harnesses model the concept of a "sub-agent," and trying to organize those into a coherent unified model, I ended up using some concepts and terminology that is more typical of the game programming space than AI/LLM programming (borrowing some ideas for entity component systems (ECS) to help clarify concepts).

The idea I want to explore here is that of taking the idea of an entity "template" or "prefab" as it might be described in a game engine, and applying it as a mental model for how to describe an instantiable complex of entities/components/etc. that together make up a complex agent or multi-agent system.

> Note: While I'm going to mention the concept of ECSes here and
> there, I want to be clear that I am not referring to the particular
> brand of dogmatic ECS designs that are a dime a dozen in the
> hobbyist game programming space (especially among Rust users).
>
> The original Unity engine is an example of a component-based game
> entity system that does not adhere to ECS dogma in terms of how
> it structurally organizes the storage of components, or sequences
> the operations of systems in time. Think more in those terms than
> a dogmatic "data-oriented" ECS design.
>
> The key point is the "composition over inheritance" paradigm.

Since I'm a programmign language designer at heart, I'll present a lot of the ideas here in terms of a hypothetical language.
Whether or not these ideas would be realized as a language or not is something we'd have to decide later, if we wanted to implement any of this.

## The Basic Idea

... (Getting distracted by other work, so I'll have to get back to this) ...
