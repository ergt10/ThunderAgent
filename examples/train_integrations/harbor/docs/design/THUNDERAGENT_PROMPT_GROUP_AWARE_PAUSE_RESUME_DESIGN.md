# ThunderAgent Prompt-Group-Aware Pause/Resume Design

Date: 2026-03-19

## Problem

SkyRL trainer expands each prompt into `n_samples_per_prompt` samples (e.g. n=5). Each sample becomes an independent `Program` in ThunderAgent. The trainer waits for **all n samples of a prompt** to complete before using that prompt for training (`fully_async_trainer.py:566-570` — `generator.generate()` blocks until all samples finish).

ThunderAgent's pause/resume is purely capacity-driven and treats each program independently. This means it can pause 3 out of 5 samples from the same prompt while letting the other 2 complete. The 2 completed samples then sit idle, wasting the GPU time already spent on them, until the paused 3 are eventually resumed and finish.

## Current Data Flow

```
prompt (instance_id=abc)
  -> prepare_generator_input(n_samples_per_prompt=5)
  -> 5 TrajectoryIDs: abc_0, abc_1, abc_2, abc_3, abc_4
  -> 5 Programs in TA, each with program_id = "{instance_id}_{repetition_id}"
  -> generator.generate() blocks until all 5 complete
  -> GeneratedOutputGroup enqueued to training buffer
```

TA knows: program_id, token count, status (REASONING/ACTING), state (ACTIVE/PAUSED/TERMINATED).
TA does NOT know: which programs share a prompt, how many siblings exist, or that partial completion is wasteful.

## Current Pause/Resume Logic

**Pause** (`_pause_until_safe`): When a backend exceeds capacity, pause ACTING programs (smallest tokens first), then mark REASONING programs. No group awareness.

**Resume** (`_greedy_resume`): BFD bin-packing on individual programs. Priority: REASONING (step>1) > NEW (step=1) > ACTING. No group awareness.

## Proposed Design: Group-Atomic Pause/Resume

### 1. Propagate Group Info (Zero Protocol Change)

Current `program_id` format is already `"{instance_id}_{repetition_id}"`. TA can derive the group key:

```python
prompt_group_id = program_id.rsplit("_", 1)[0]  # "abc"
```

Add to `Program`:
```python
@dataclass
class Program:
    # ... existing fields ...
    prompt_group_id: Optional[str] = None
```

Maintain on router:
```python
self.prompt_groups: Dict[str, Set[str]] = {}  # group_id -> {program_ids}
```

Populated in `get_or_create_program`, cleaned up in `release_program`.

### 2. Group-Atomic Pause

When `_pause_until_safe` decides to pause a program, pause all **active siblings** in the same prompt group:

```python
def _pause_program_group(self, trigger_program_id: str, state: Program):
    group_id = state.prompt_group_id
    if not group_id:
        self._pause_program(trigger_program_id, state)
        return

    siblings = self.prompt_groups.get(group_id, set())
    for pid in siblings:
        sib_state = self.programs.get(pid)
        if sib_state and sib_state.state == ProgramState.ACTIVE:
            if sib_state.status == ProgramStatus.ACTING:
                self._pause_program(pid, sib_state)
            else:  # REASONING — mark for deferred pause
                self._mark_program_for_pause(pid, sib_state)
```

This prevents half-finished groups from blocking the trainer.

### 3. Group-Atomic Resume

In `_greedy_resume`, select and place groups as units:

- Aggregate paused programs by `prompt_group_id`
- For each group, compute total required tokens = sum of member tokens + BUFFER_PER_PROGRAM * count
- Only resume a group if capacity fits **all** its paused members
- BFD placement at group level

Fallback: if a group has only 1 surviving paused member (others already terminated), treat it as an individual program (degrades to current behavior).

### 4. Edge Cases

| Case | Handling |
|------|----------|
| `n_samples_per_prompt=1` | Group size = 1, degrades to current behavior |
| Some siblings already terminated | Group = only living (non-terminated) members |
| Siblings on different backends | Pause collects across backends; resume can scatter across backends (fine since trainer only cares about completion, not co-location) |
| Group too large to fit anywhere | Skip entire group, resume smaller groups first. Avoids partial resume that creates the same deadlock |
| `program_id` doesn't follow `{id}_{rep}` format | `prompt_group_id = None`, falls back to individual behavior |

### 5. Implementation Path

1. Add `prompt_group_id` field to `Program` dataclass
2. Add `prompt_groups` dict to `MultiBackendRouter.__init__`
3. Parse group from `program_id` in `get_or_create_program`; register in `prompt_groups`
4. Clean up `prompt_groups` in `release_program` when last member leaves
5. Modify `_pause_until_safe` to use `_pause_program_group`
6. Modify `_greedy_resume` to operate on groups
7. Add tests for group-atomic behavior

### 6. Open Questions

- Should there be a configurable toggle (`group_aware_pause=True`) for backward compat?
- Should TA also expose a `/programs/group_status` endpoint so the trainer can query group-level completion?
- If the prompt_group_id convention (`rsplit("_", 1)`) is too fragile, should we instead have the caller pass `prompt_group_id` explicitly in the request payload?
