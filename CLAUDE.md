# OXIGRAPH TRIVYN

This is a fork of oxigraph with TRIVYN and MOOSE specific enhancements

---

## Core Pricnciples

1.  Maintain compatibility with main oxigraph.  We want to be able to take changes from the main oxigraph repo.  Do not change any interfaces unless absolutely necessary. Always ask permission first.

2.  Minimal viable changes.  To reduce merge conflicts when taking changes from the main oxigraph repo, do not make unnecessary code modifications.  We are not here to make nitpicking improvements to oxigraph, we just want to make the minimal changes necessary to support features in Trivyn and MOOSE.

## Development practices

### 1.  Plan Mode Default
- Enter plan mode for ANY non-trivial task (#+ steps or architectural decisions)
- If something goes wrong, STOP and re-plan immediately - don't keep pressing on.
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the Pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution."
- Skip this for simple, obvious fixes -- don't over-engineer
- Challenge your own work before presenting it.

### 6. Autonomous Bug fixing
- When given a bug report: Just fix it.  Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task management

1. **Plan First**: Write plan to 'tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles
- **simplicity first**: Make every change as simple as possible.  Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes.  Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
~
