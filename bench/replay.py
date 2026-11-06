"""Model-free replay logic, kept free of AgentDojo imports so it can be unit-tested.

``bench.oracle`` wires this to AgentDojo. Everything here works on plain
callables: a list of reference calls, a function that executes one call, and
a function that says whether the attacker's goal was reached.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from provguard import Action, Guard, Policy

Confirm = Literal["deny", "approve"]


class Call(Protocol):
    function: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class Step:
    function: str
    action: str
    ran: bool
    reason: str


def run_calls(
    guard: Guard,
    calls: Iterable[Call],
    execute: Callable[[Call], Any | None],
    confirm: Confirm = "deny",
    stop_on_block: bool = False,
) -> list[Step]:
    """Replay calls through the guard.

    ``execute`` runs a call and returns its output, or ``None`` if the tool
    failed. ``confirm`` decides CONFIRM: ``deny`` models no human (or one who
    always refuses), ``approve`` a human who approves everything, the worst
    case for security. Denied calls are skipped and replay continues, as a
    hijacked agent would keep trying, unless ``stop_on_block``.
    """
    steps: list[Step] = []
    for call in calls:
        decision = guard.check(call.function, dict(call.args))
        ran = decision.action is Action.ALLOW or (decision.action is Action.CONFIRM and confirm == "approve")
        if ran:
            output = execute(call)
            guard.commit(decision)
            if output is not None:
                guard.record_result(call.function, output)
        steps.append(Step(call.function, decision.action.name.lower(), ran, decision.explain()))
        if stop_on_block and decision.action is Action.BLOCK:
            break
    return steps


def worst(steps: list[Step]) -> tuple[str, str]:
    order = {"allow": 0, "confirm": 1, "block": 2}
    top = max(steps, key=lambda s: order[s.action], default=None)
    if top is None or top.action == "allow":
        return "allow", ""
    return top.action, top.reason


def utility_record(steps: list[Step], solved: bool | None) -> dict:
    action, reason = worst(steps)
    return {
        "action": action,  # strictest decision on the reference solution (v2 metric)
        "reason": reason,
        "confirmations": sum(s.action == "confirm" for s in steps),
        "blocks": sum(s.action == "block" for s in steps),
        # With a user who approves every confirmation, does the task still succeed?
        "solved_with_user": solved,
    }


@dataclass
class PairOutcome:
    # v2 metric: at least one attacker call denied (kept for comparison).
    stopped: bool | None
    action: str | None
    call: str | None
    reason: str
    # v3 metrics, judged on the environment the attack left behind.
    verifiable: bool | None = None       # does the attack succeed with no guard at all?
    goal_no_human: bool | None = None     # goal reached when confirmations are refused
    goal_any_human: bool | None = None    # goal reached when confirmations are approved
    confirmations: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def attack_outcome(
    guard: Guard,
    env: Any,
    calls_for: Callable[[Any], list[Call]],
    execute_in: Callable[[Any], Callable[[Call], Any | None]],
    goal_reached: Callable[[Any, Any, list[Call]], bool],
) -> PairOutcome:
    """Judge one (user task, injection task) pair from the state it leaves.

    ``env`` is the environment after the user task's reference calls;
    ``calls_for(env)`` returns the attacker's reference calls;
    ``execute_in(env)`` returns an executor bound to ``env``;
    ``goal_reached(pre, post, executed)`` is the injection task's own success
    check; ``executed`` lists the attacker calls that actually ran.
    """
    reference = calls_for(copy.deepcopy(env))
    if not reference:
        return PairOutcome(None, None, None, "no reference")

    def replay(with_guard: Guard | None, confirm: Confirm) -> tuple[bool, list[Step]]:
        pre, post = copy.deepcopy(env), copy.deepcopy(env)
        bound = execute_in(post)
        executed: list[Call] = []

        def run(call: Call) -> Any | None:
            executed.append(call)
            return bound(call)

        if with_guard is None:
            for call in reference:
                run(call)
            return goal_reached(pre, post, executed), []
        steps = run_calls(copy.deepcopy(with_guard), reference, run, confirm)
        return goal_reached(pre, post, executed), steps

    unguarded, _ = replay(None, "deny")
    goal_deny, steps = replay(guard, "deny")
    goal_approve, _ = replay(guard, "approve")

    first = next((s for s in steps if s.action != "allow"), None)
    return PairOutcome(
        stopped=first is not None,
        action=first.action if first else "allow",
        call=first.function if first else None,
        reason=first.reason if first else "",
        verifiable=unguarded,
        goal_no_human=goal_deny,
        goal_any_human=goal_approve,
        confirmations=sum(s.action == "confirm" for s in steps),
    )


def lint_policy(policy: Policy, signatures: Mapping[str, Iterable[str]]) -> list[str]:
    """Rules that name a tool or argument the suite does not have.

    Such a rule silently protects nothing, so the oracle prints these before
    any number.
    """
    problems = []
    for tool, rules in policy.tools.items():
        if tool not in signatures:
            problems.append(f"policy names unknown tool {tool!r}")
            continue
        known = set(signatures[tool])
        problems += [f"{tool}: unknown argument {arg!r}" for arg in rules.args if arg not in known]
    return problems
