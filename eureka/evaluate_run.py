"""
evaluate_cli.py

Standalone command-line tool for evaluating ANY trained EUREKA candidate on
demand, independent of eureka/loop.py's search. Useful for:
    - re-checking a candidate on more episodes than the search used
    - comparing a hand-picked shortlist side by side
    - sanity-checking a checkpoint after the fact without rerunning the loop
    - watching a policy drive, or saving a GIF of one episode

Reuses the exact same deterministic evaluation as the search
(eureka.evaluate_candidate.evaluate_candidate) and the exact same diagnostic
fitness formula (eureka.fitness.compute_fitness), so numbers here are
directly comparable to what's in eureka_log.json / the console's final
archive table (see eureka/logging_utils.py::print_final_archive_table).

Usage:
    python -m eureka.evaluate_cli gen2_cand5
    python -m eureka.evaluate_cli gen2_cand5 --episodes 100
    python -m eureka.evaluate_cli gen2_cand5 gen0_cand5 gen1_cand5   # compare several
    python -m eureka.evaluate_cli --all                              # every trained candidate
    python -m eureka.evaluate_cli --list                             # what's available, without running anything
    python -m eureka.evaluate_cli gen2_cand5 --checkpoint some/other.pt

    # Rendering (in addition to the numeric metrics above):
    python -m eureka.evaluate_cli gen2_cand5 --render                     # save GIF(s) to eureka/renders/
    python -m eureka.evaluate_cli gen2_cand5 --render --render-episodes 3
    python -m eureka.evaluate_cli gen2_cand5 --render-live                # open a live window instead (needs a local display)
    python -m eureka.evaluate_cli gen2_cand5 --render-live --render-only  # skip the numeric eval, just watch it drive

Candidate names accept either the short form ("gen2_cand5") or the full
dotted module path ("eureka.candidates.gen2_cand5") - both resolve to the
same thing.

Rendering saves frames via env.render(render_mode="rgb_array") to a GIF
using imageio (pip install imageio if not already present). --render-live
instead opens an actual pygame window via render_mode="human" - only makes
sense with exactly one candidate and a local display (works fine running
directly on Windows/macOS/Linux desktop; will not work over a headless SSH
session without X forwarding - use --render for that case instead).

Note on --render-live pacing: highway-env's viewer only paces frames to
wall-clock time (see EnvViewer.display() in highway_env/envs/common/
graphics.py) when the env's own config sets "real_time_rendering": True.
This project's ENV_CONFIG (config.py) never sets that key, and the
default is False - so a live render's pacing would otherwise depend
entirely on incidental per-step render cost happening to roughly match
simulation_frequency, rather than anything guaranteed. This module
explicitly requests real-time pacing for the live path only (see
RenderJob.env_overrides / _build_render_env below), so a fast machine
can't make the window flash open and close faster than a human can see.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eureka.eureka_config import FITNESS_WEIGHTS, N_EVAL_EPISODES
from eureka.evaluate_candidate import evaluate_candidate
from eureka.fitness import compute_fitness
from eureka.logging_utils import get_logger, print_final_archive_table

logger = get_logger(__name__)

CANDIDATES_DIR = Path("eureka") / "candidates"
REJECTED_DIR = CANDIDATES_DIR / "rejected"
CHECKPOINTS_DIR = Path("eureka") / "checkpoints"
DEFAULT_RENDER_DIR = Path("eureka") / "renders"

# Distinct from evaluate_candidate.py's own eval seeds (2000 + episode) so a
# render never silently reruns the exact same rollout already scored in the
# numeric metrics - it's a fresh, separately-seeded look at the same policy.
RENDER_SEED_BASE = 9000


# --------------------------------------------------------------------------- #
# Candidate resolution / discovery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CandidateRef:
    """
    Everything needed to locate one candidate's source/checkpoint, resolved
    once from whatever form the user typed it in (short name or dotted
    module path, with or without a trailing ".py").
    """

    short_name: str
    module_path: str
    source_path: Path
    rejected_path: Path
    default_checkpoint: Path

    @classmethod
    def from_raw(cls, raw: str) -> "CandidateRef":
        name = raw.strip()
        if name.endswith(".py"):
            name = name[: -len(".py")]
        if name.startswith("eureka.candidates."):
            module_path = name
            short = name.rsplit(".", 1)[-1]
        else:
            short = name
            module_path = f"eureka.candidates.{short}"
        return cls(
            short_name=short,
            module_path=module_path,
            source_path=CANDIDATES_DIR / f"{short}.py",
            rejected_path=REJECTED_DIR / f"{short}.py",
            default_checkpoint=CHECKPOINTS_DIR / f"{short}.pt",
        )

    def resolve_checkpoint(self, override: Optional[str]) -> Path:
        return Path(override) if override else self.default_checkpoint


def discover_trained_candidates() -> list[str]:
    """Every gen*.pt under eureka/checkpoints/ that also has a matching
    (non-rejected) source file - i.e. actually loadable and evaluable."""
    if not CHECKPOINTS_DIR.is_dir():
        return []
    return sorted(
        path.stem
        for path in CHECKPOINTS_DIR.glob("*.pt")
        if (CANDIDATES_DIR / f"{path.stem}.py").is_file()
    )


def discover_all_candidate_names() -> tuple[set[str], set[str]]:
    """Returns (all_saved_source_names, rejected_names)."""
    saved = {p.stem for p in CANDIDATES_DIR.glob("*.py")} if CANDIDATES_DIR.is_dir() else set()
    rejected = {p.stem for p in REJECTED_DIR.glob("*.py")} if REJECTED_DIR.is_dir() else set()
    return saved, rejected


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RenderJob:
    """Everything one render pass (GIF or live) needs, bundled so it isn't
    threaded through four separate positional parameters."""

    episodes: int
    live: bool
    render_dir: Path
    fps: int


def _build_render_env(module_path: str, live: bool):
    """Constructs the wrapped highway-env instance used for one rendered
    episode. Split out of _run_render_episode purely so the (fairly heavy)
    import list and env-construction logic reads as one clear step."""
    import gymnasium as gym
    import highway_env  # noqa: F401

    from config import ENV_CONFIG, ENV_ID
    from eureka.candidate_wrapper import CandidateRewardWrapper
    from eureka.sandbox import load_shaping_reward_from_module_path

    shaping_fn = load_shaping_reward_from_module_path(module_path)

    render_mode = "human" if live else "rgb_array"
    env = gym.make(ENV_ID, render_mode=render_mode)

    render_config = dict(ENV_CONFIG)
    # offscreen_rendering=False is required for an actual visible window
    # (human mode); True is what lets rgb_array capture work on a machine
    # with no display attached (e.g. over SSH) while still returning frames.
    render_config["offscreen_rendering"] = not live
    if live:
        # See the module docstring's "Note on --render-live pacing" - this
        # is required for the window to actually run at a watchable speed
        # instead of finishing in a fraction of a second.
        render_config["real_time_rendering"] = True
    env.unwrapped.configure(render_config)

    return CandidateRewardWrapper(env, shaping_fn)


def _load_policy(checkpoint_path: Path, obs_dim: int, n_actions: int):
    import torch

    from config import NET_ARCH
    from networks import ActorCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, n_actions, hidden_sizes=NET_ARCH).to(device)
    model.load_state_dict(
        torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    )
    model.eval()
    return model, device


def _run_render_episode(
    module_path: str, checkpoint_path: Path, seed: int, live: bool
) -> Optional[list]:
    """
    Runs ONE deterministic episode (same argmax policy as evaluate_candidate.py)
    with rendering enabled, and returns the list of captured RGB frames
    (None when live=True, since a "human" render window displays itself and
    there is nothing to save).

    Deliberately a self-contained duplicate of evaluate_candidate.py's model-
    loading logic rather than a shared refactor: this keeps the two code
    paths (numeric eval vs. visual render) easy to verify independently by
    reading either file top to bottom, matching this project's existing
    style (e.g. smoke_test.py's runtime probe duplicates rather than
    importing evaluate_candidate.py's env construction).

    The env is always closed via `finally`, even if the policy, the env, or
    a candidate's shaping function raises mid-episode - a bare `env.close()`
    after the loop would otherwise skip on exception and leak the pygame
    window / worker resources.
    """
    import numpy as np
    import torch

    env = _build_render_env(module_path, live)
    try:
        obs, info = env.reset(seed=seed)
        obs_dim = int(np.prod(obs.shape))
        n_actions = env.action_space.n
        model, device = _load_policy(checkpoint_path, obs_dim, n_actions)

        frames: Optional[list] = None if live else []

        def _capture() -> None:
            frame = env.render()
            if frames is not None and frame is not None:
                frames.append(frame)

        _capture()
        done = False
        while not done:
            obs_t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = model.forward(obs_t)
                action = torch.argmax(logits, dim=-1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            _capture()

        return frames
    finally:
        env.close()


def _save_frames_as_gif(frames: list, out_path: Path, fps: int) -> None:
    try:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio
    except ImportError as e:
        raise RuntimeError(
            "Saving renders needs the 'imageio' package. Install it with: "
            "pip install imageio"
        ) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(str(out_path), frames, duration=duration)


def render_candidate(ref: CandidateRef, checkpoint_path: Path, job: RenderJob) -> None:
    """
    Renders `job.episodes` deterministic rollouts for one candidate. In file
    mode, saves one GIF per episode under render_dir/{short}_epN.gif and
    prints the path. In live mode, opens a window per episode (blocks until
    the episode ends) and saves nothing.

    Each episode is independent and failures are isolated: one episode
    raising (e.g. a transient rendering error) is reported and skipped
    rather than aborting every remaining episode/candidate.
    """
    for ep in range(job.episodes):
        seed = RENDER_SEED_BASE + ep
        try:
            frames = _run_render_episode(ref.module_path, checkpoint_path, seed, job.live)
        except Exception as e:
            logger.warning(
                "render episode failed",
                extra={"event": "render_episode_failed", "candidate": ref.short_name, "episode": ep, "reason": str(e)},
            )
            print(f"  RENDER FAILED for {ref.short_name} (episode {ep}): {e}")
            continue

        if job.live:
            print(f"  {ref.short_name}: rendered episode {ep} live (seed={seed})")
            continue

        out_path = job.render_dir / f"{ref.short_name}_ep{ep}.gif"
        try:
            _save_frames_as_gif(frames, out_path, job.fps)
        except RuntimeError as e:
            print(f"  {e}")
            return
        print(f"  {ref.short_name}: saved render -> {out_path} ({len(frames)} frames @ {job.fps}fps)")


# --------------------------------------------------------------------------- #
# Numeric evaluation
# --------------------------------------------------------------------------- #


@dataclass
class EvalOutcome:
    ref: CandidateRef
    metrics: dict
    fitness: float

    def as_archive_row(self) -> dict:
        return {
            "module_path": self.ref.module_path,
            "candidate_id": self.ref.short_name,
            "pareto_rank": "-",
            "legacy_fitness": self.fitness,
            "metrics": self.metrics,
        }


def print_single_result(short: str, metrics: dict, fitness: float, episodes: int) -> None:
    header = f"  {short}  ({episodes} deterministic episodes)"
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")
    print(f"  crash_rate      {metrics['crash_rate']:.2%}")
    print(f"  mean_speed      {metrics['mean_speed']:.2f} m/s")
    print(f"  mean_overtakes  {metrics['mean_overtakes']:.2f} per episode")
    print(f"  mean_raw_return {metrics['mean_raw_return']:.2f}")
    print(f"  legacy_fitness  {fitness:.3f}  (diagnostic only - see eureka/fitness.py)")

    component_means = metrics.get("component_means") or {}
    if component_means:
        print("  reward components (mean per episode):")
        for key in sorted(component_means):
            print(f"    {key}: {component_means[key]:.4f}")
    print()


def print_available_candidates() -> None:
    trained = set(discover_trained_candidates())
    all_sources, rejected = discover_all_candidate_names()
    untrained = sorted(all_sources - trained)

    print("\nTrained candidates (source + checkpoint present - ready to evaluate):")
    if trained:
        for name in sorted(trained):
            print(f"  {name}")
    else:
        print("  (none)")

    if untrained:
        print("\nSaved but never trained (source present, no checkpoint found):")
        for name in untrained:
            print(f"  {name}  ->  train first: eureka.train_candidate.train_candidate('eureka.candidates.{name}', ...)")

    if rejected:
        print("\nRejected by the sandbox (cannot be evaluated - see docs/SECURITY.md):")
        for name in sorted(rejected):
            print(f"  {name}")
    print()


# --------------------------------------------------------------------------- #
# Per-candidate orchestration
# --------------------------------------------------------------------------- #


@dataclass
class RunPlan:
    """Fully-validated, ready-to-execute description of one CLI invocation.
    Building this up front (in build_run_plan) keeps every validation error
    in one place, before any potentially slow work (evaluation/rendering)
    starts."""

    names: list[str]
    episodes: int
    checkpoint_override: Optional[str]
    render_only: bool
    render_job: Optional[RenderJob]


class CandidateSkipped(Exception):
    """Raised internally to short-circuit one candidate's processing with a
    user-facing message already printed - not a real error, just a clean
    way to express "move on to the next candidate" without nested ifs."""


def _resolve_and_check(ref: CandidateRef, checkpoint: Path) -> None:
    if not ref.source_path.is_file():
        if ref.rejected_path.is_file():
            raise CandidateSkipped(
                f"this candidate was rejected by the sandbox and never trained "
                f"(see {ref.rejected_path} for why)."
            )
        raise CandidateSkipped(f"no source file at {ref.source_path}.")

    if not checkpoint.is_file():
        raise CandidateSkipped(
            f"no checkpoint at {checkpoint}. Has this candidate actually been trained yet?"
        )


@dataclass
class ProcessResult:
    """Distinguishes 'this candidate was skipped' from 'this candidate
    succeeded but produced no EvalOutcome' (the --render-only case) -
    collapsing both into a single Optional[EvalOutcome] would make a fully
    skipped --render-only run look identical to a fully successful one."""

    succeeded: bool
    outcome: Optional[EvalOutcome] = None


def process_candidate(raw_name: str, plan: RunPlan) -> ProcessResult:
    """
    Runs the (numeric-eval, render) pipeline for one candidate name. Never
    raises: all expected failure modes are caught and reported, matching
    this CLI's "one bad candidate shouldn't stop the rest" philosophy (same
    as loop.py's candidate rejection handling).
    """
    ref = CandidateRef.from_raw(raw_name)
    checkpoint = ref.resolve_checkpoint(plan.checkpoint_override)

    try:
        _resolve_and_check(ref, checkpoint)
    except CandidateSkipped as e:
        print(f"  SKIP {ref.short_name}: {e}")
        return ProcessResult(succeeded=False)

    outcome: Optional[EvalOutcome] = None

    if plan.render_only:
        print(f"  Skipping numeric evaluation for {ref.short_name} (--render-only).")
    else:
        print(f"  Evaluating {ref.short_name} ({plan.episodes} episodes)...")
        try:
            metrics = evaluate_candidate(str(checkpoint), ref.module_path, n_episodes=plan.episodes)
        except Exception as e:
            logger.warning(
                "evaluation failed",
                extra={"event": "cli_eval_failed", "candidate": ref.short_name, "reason": str(e)},
            )
            print(f"  FAILED to evaluate {ref.short_name}: {e}")
            return ProcessResult(succeeded=False)

        fitness = compute_fitness(metrics, FITNESS_WEIGHTS)
        print_single_result(ref.short_name, metrics, fitness, plan.episodes)
        outcome = EvalOutcome(ref=ref, metrics=metrics, fitness=fitness)

    if plan.render_job is not None:
        render_candidate(ref, checkpoint, plan.render_job)

    return ProcessResult(succeeded=True, outcome=outcome)


# --------------------------------------------------------------------------- #
# CLI parsing / validation
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eureka.evaluate_cli",
        description="Run a deterministic evaluation on any trained EUREKA candidate, "
                     "outside the search loop.",
    )
    parser.add_argument(
        "candidates", nargs="*",
        help="Candidate name(s), e.g. gen2_cand5 or eureka.candidates.gen2_cand5. "
             "Omit to evaluate every trained candidate found (same as --all).",
    )

    selection = parser.add_argument_group("candidate selection")
    selection.add_argument(
        "--all", action="store_true",
        help="Evaluate every trained candidate found under eureka/checkpoints/.",
    )
    selection.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override the checkpoint path. Only valid together with exactly one candidate name.",
    )
    selection.add_argument(
        "--list", action="store_true",
        help="List available/trained/rejected candidates and exit without evaluating anything.",
    )

    evaluation = parser.add_argument_group("numeric evaluation")
    evaluation.add_argument(
        "--episodes", type=int, default=N_EVAL_EPISODES,
        help=f"Deterministic eval episodes per candidate (default: {N_EVAL_EPISODES}, "
             "same as the search loop's current N_EVAL_EPISODES).",
    )

    rendering = parser.add_argument_group("rendering")
    rendering.add_argument(
        "--render", action="store_true",
        help="Also save a GIF of one deterministic episode per candidate to eureka/renders/ "
             "(needs the 'imageio' package: pip install imageio).",
    )
    rendering.add_argument(
        "--render-only", action="store_true",
        help="Skip the numeric evaluate_candidate() pass entirely and only render. "
             f"Useful with --render-live: the default numeric eval runs {N_EVAL_EPISODES} "
             "full episodes before rendering even starts, which can take minutes and "
             "looks like nothing is happening if all you wanted was to watch the policy "
             "drive. Implies --render.",
    )
    rendering.add_argument(
        "--render-live", action="store_true",
        help="Instead of saving a GIF, open a live window (render_mode='human'). "
             "Needs a local display; only valid with exactly one candidate. Implies --render.",
    )
    rendering.add_argument(
        "--render-episodes", type=int, default=1,
        help="Number of episodes to render per candidate (default: 1).",
    )
    rendering.add_argument(
        "--render-dir", type=str, default=str(DEFAULT_RENDER_DIR),
        help=f"Output directory for saved GIFs (default: {DEFAULT_RENDER_DIR}).",
    )
    rendering.add_argument(
        "--render-fps", type=int, default=5,
        help="Frames per second for saved GIFs (default: 5, matching policy_frequency).",
    )
    return parser


def build_run_plan(parser: argparse.ArgumentParser, args: argparse.Namespace) -> RunPlan:
    """
    Validates the raw argparse namespace and turns it into a RunPlan.
    All cross-argument validation (things argparse itself can't express,
    like "--checkpoint only with one candidate") lives here, in one place,
    and fails via parser.error() before any work starts.
    """
    if args.checkpoint and (args.all or len(args.candidates) != 1):
        parser.error("--checkpoint can only be used together with exactly one candidate name")

    render = args.render or args.render_only or args.render_live

    if args.render_live and (args.all or len(args.candidates) != 1):
        parser.error("--render-live can only be used with exactly one candidate name")

    if args.render_episodes < 1:
        parser.error("--render-episodes must be >= 1")

    names = list(args.candidates)
    if args.all or not names:
        names = discover_trained_candidates()

    render_job = (
        RenderJob(
            episodes=args.render_episodes,
            live=args.render_live,
            render_dir=Path(args.render_dir),
            fps=args.render_fps,
        )
        if render
        else None
    )

    return RunPlan(
        names=names,
        episodes=args.episodes,
        checkpoint_override=args.checkpoint,
        render_only=args.render_only,
        render_job=render_job,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_available_candidates()
        return 0

    plan = build_run_plan(parser, args)

    if not plan.names:
        print(
            "No trained candidates found under eureka/checkpoints/. "
            "Run `python -m eureka.loop` first, or pass a candidate name "
            "explicitly with --checkpoint. Use --list to see what's on disk."
        )
        return 1

    if plan.render_job is not None and not plan.render_job.live:
        plan.render_job.render_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[EvalOutcome] = []
    processed_count = 0
    for raw_name in plan.names:
        result = process_candidate(raw_name, plan)
        if result.succeeded:
            processed_count += 1
        if result.outcome is not None:
            outcomes.append(result.outcome)

    if len(outcomes) > 1:
        best = max(outcomes, key=lambda o: o.fitness)
        print_final_archive_table(
            [o.as_archive_row() for o in outcomes],
            representative_id=best.ref.short_name,
        )

    return 0 if processed_count else 1


if __name__ == "__main__":
    sys.exit(main())
