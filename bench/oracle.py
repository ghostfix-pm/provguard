"""Model-free evaluation: replay AgentDojo's ground-truth trajectories through ProvGuard.

    python -m bench.oracle                                  # v2 policies -> results/oracle/
    python -m bench.oracle --policies bench/policies-v3 --tag v3   # -> results/oracle-v3/

Questions answered without any LLM in the loop:

* Utility cost: for each user task, which calls of the reference solution does
  ProvGuard confirm or block, and, with a user who approves every
  confirmation, does the task still succeed by AgentDojo's own check?
* Attack coverage: each (user task, injection task) pair runs the user task's
  reference calls in the injected environment, then the injection task's
  reference calls as if the agent were fully hijacked. A hijacked agent keeps
  going after a refusal, so denied calls are skipped, not fatal. The attack
  is then judged by AgentDojo's own ``security`` check on the final state:

  - goal_no_human: reached when nobody approves confirmations;
  - goal_any_human: reached even when every confirmation is approved, so only
    BLOCK decisions protect;
  - verifiable: the attack does reach its goal with no guard at all. Pairs
    where it does not cannot be judged by replay and are reported apart.

  The v2 metric ("at least one attacker call denied") is kept as ``stopped``.

This measures the policy, not a model. A real agent may take other paths; the
LLM runs in ``bench.run`` measure that.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from provguard import Guard, Policy

from . import attacks  # noqa: F401  (registers the adaptive attack)
from .agentdojo_guard import structured
from .replay import attack_outcome, lint_policy, run_calls, utility_record
from .run import BENCHMARK_VERSION, POLICIES, ROOT, SUITES
from .stats import fmt_rate


class _NamedPipeline:
    # Attacks only read the pipeline's name, to address the model in the injection text.
    name = "local"


def _executor(runtime: FunctionsRuntime, env):
    def execute(call) -> Any | None:
        result, error = runtime.run_function(env, call.function, dict(call.args))
        return None if error is not None else structured(tool_result_to_str(result))

    return execute


def _signatures(suite) -> dict[str, list[str]]:
    tools = suite.tools.values() if isinstance(suite.tools, dict) else suite.tools
    return {fn.name: list(fn.parameters.model_fields) for fn in tools}


def _task_succeeded(task, pre, post, executed) -> bool:
    # AgentDojo's own order: a trace-based check when the task defines one,
    # the state-based check otherwise (some tasks implement only one of them).
    by_trace = task.utility_from_traces("", pre, post, executed)
    if by_trace is not None:
        return bool(by_trace)
    return bool(task.utility("", pre, post))


def _goal(inj):
    def reached(pre, post, executed) -> bool:
        # Some injection tasks judge the trace rather than the final state.
        from_traces = getattr(inj, "security_from_traces", None)
        by_trace = from_traces("", pre, post, executed) if from_traces else None
        return bool(by_trace) if by_trace is not None else bool(inj.security("", pre, post))

    return reached


def evaluate_suite(name: str, attack_name: str, policies: Path) -> dict:
    suite = get_suite(BENCHMARK_VERSION, name)
    policy = Policy.from_yaml(policies / f"{name}.yaml")
    problems = lint_policy(policy, _signatures(suite))
    for p in problems:
        print(f"  policy warning [{name}]: {p}", flush=True)
    runtime = FunctionsRuntime(suite.tools)
    attack = load_attack(attack_name, suite, _NamedPipeline())

    utility: dict[str, dict] = {}
    for task_id, task in suite.user_tasks.items():
        calls = task.ground_truth(suite.load_and_inject_default_environment({}))

        def solved(confirm: str | None) -> tuple[bool, list]:
            env = suite.load_and_inject_default_environment({})
            pre = env.model_copy(deep=True)
            run = _executor(runtime, env)
            executed: list = []

            def execute(call):
                executed.append(call)
                return run(call)

            if confirm is None:
                for call in calls:
                    execute(call)
                steps = []
            else:
                guard = Guard(policy)
                guard.user(task.PROMPT)
                steps = run_calls(guard, calls, execute, confirm)
            return _task_succeeded(task, pre, env, executed), steps

        baseline, _ = solved(None)
        with_user, steps = solved("approve")
        record = utility_record(steps, with_user if baseline else None)
        record["solvable_by_replay"] = baseline  # False when the check needs the model's answer text
        utility[task_id] = record

    security: dict[str, dict] = {}
    for task_id, task in suite.user_tasks.items():
        for inj_id, inj in suite.injection_tasks.items():
            env = suite.load_and_inject_default_environment(attack.attack(task, inj))
            guard = Guard(policy)
            guard.user(task.PROMPT)
            run_calls(guard, task.ground_truth(env.model_copy(deep=True)), _executor(runtime, env), "deny")
            outcome = attack_outcome(
                guard,
                env,
                calls_for=inj.ground_truth,
                execute_in=lambda e: _executor(runtime, e),
                goal_reached=_goal(inj),
            )
            security[f"{task_id}|{inj_id}"] = outcome.as_dict()

    return {
        "suite": name,
        "attack": attack_name,
        "policies": str(policies.relative_to(ROOT)) if policies.is_relative_to(ROOT) else str(policies),
        "policy_warnings": problems,
        "utility": utility,
        "security": security,
    }


def summarize(result: dict) -> str:
    u = result["utility"]
    acts = Counter(v["action"] for v in u.values())
    judged = [v for v in u.values() if v.get("solved_with_user") is not None]
    solved = sum(v["solved_with_user"] for v in judged)
    confirms = sum(v.get("confirmations", 0) for v in u.values())

    s = [v for v in result["security"].values() if v["stopped"] is not None]
    v2 = sum(v["stopped"] for v in s)
    ver = [v for v in s if v.get("verifiable")]
    no_human = sum(not v["goal_no_human"] for v in ver)
    any_human = sum(not v["goal_any_human"] for v in ver)
    lines = [
        f"{result['suite']:9} user tasks {len(u):3}: allowed {acts['allow']:3}  confirm {acts['confirm']:3}  "
        f"blocked {acts['block']:3} | {confirms} confirmations in total | "
        f"solved with an approving user {solved}/{len(judged)}",
        f"          v2 metric, a call denied: {v2}/{len(s)}",
        f"          attack goal prevented, confirmations refused: {fmt_rate(no_human, len(ver))}",
        f"          attack goal prevented, confirmations approved: {fmt_rate(any_human, len(ver))}",
        f"          pairs the replay cannot judge: {len(s) - len(ver)} (attack fails even unguarded), "
        f"{len(result['security']) - len(s)} (no reference calls)",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", nargs="+", choices=SUITES, default=SUITES)
    parser.add_argument("--attack", default="important_instructions")
    parser.add_argument("--policies", type=Path, default=POLICIES)
    parser.add_argument("--tag", default="", help="results go to results/oracle[-TAG]/")
    args = parser.parse_args()

    out_dir = ROOT / "results" / ("oracle-" + args.tag if args.tag else "oracle")
    if args.attack != "important_instructions":
        out_dir = out_dir.with_name(out_dir.name + "-" + args.attack)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.suite:
        result = evaluate_suite(name, args.attack, args.policies.resolve())
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(summarize(result), flush=True)


if __name__ == "__main__":
    main()
