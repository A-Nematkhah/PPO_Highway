"""
evaluate_cli.py

Standalone command-line tool for evaluating ANY trained EUREKA candidate on
demand, independent of eureka/loop.py's search. Useful for:
    - re-checking a candidate on more episodes than the search used
    - comparing a hand-picked shortlist side by side
    - sanity-checking a checkpoint after the fact without rerunning the loop

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
    python -m eureka.evaluate_cli gen2_cand5 --render                # save GIF(s) to eureka/renders/
    python -m eureka.evaluate_cli gen2_cand5 --render --render-episodes 3
    python -m eureka.evaluate_cli gen2_cand5 --render-live            # open a live window instead (needs a local display)

Candidate names accept either the short form ("gen2_cand5") or the full
dotted module path ("eureka.candidates.gen2_cand5") - both resolve to the
same thing.

Rendering saves frames via env.render(render_mode="rgb_array") to a GIF
using imageio (pip install imageio if not already present). --render-live
instead opens an actual pygame window via render_mode="human" - only makes
sense with exactly one candidate and a local display (works fine running
directly on Windows/macOS/Linux desktop; will not work over a headless SSH
session without X forwarding - use --render for that case instead).
"""

from __future__ import annotations

import argparse
import os
import sys

from eureka.eureka_config import FITNESS_WEIGHTS, N_EVAL_EPISODES
from eureka.evaluate_candidate import evaluate_candidate
from eureka.fitness import compute_fitness
from eureka.logging_utils import print_final_archive_table

CANDIDATES_DIR = os.path.join("eureka", "candidates")
REJECTED_DIR = os.path.join(CANDIDATES_DIR, "rejected")
CHECKPOINTS_DIR = os.path.join("eureka", "checkpoints")
DEFAULT_RENDER_DIR = os.path.join("eureka", "renders")


def _normalize(raw: str) -> tuple[str, str]:
    """
    Accepts either "gen2_cand5" or "eureka.candidates.gen2_cand5" (with or
    without a trailing ".py"). Returns (short_name, dotted_module_path).
    """
    name = raw.strip()
    if name.endswith(".py"):
        name = name[: -len(".py")]
    if name.startswith("eureka.candidates."):
        module_path = name
        short = name.rsplit(".", 1)[-1]
    else:
        short = name
        module_path = f"eureka.candidates.{short}"
    return short, module_path


def _default_checkpoint_path(short: str) -> str:
    return os.path.join(CHECKPOINTS_DIR, f"{short}.pt")


def _candidate_source_path(short: str) -> str:
    return os.path.join(CANDIDATES_DIR, f"{short}.py")


def _rejected_source_path(short: str) -> str:
    return os.path.join(REJECTED_DIR, f"{short}.py")


def _run_render_episode(module_path: str, checkpoint_path: str, seed: int, live: bool):
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
    """
    import gymnasium as gym
    import highway_env  # noqa: F401
    import numpy as np
    import torch

    from config import ENV_CONFIG, ENV_ID, NET_ARCH
    from networks import ActorCritic
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
        # highway-env's EnvViewer only paces frames to wall-clock time
        # (clock.tick(simulation_frequency) before pygame.display.flip(),
        # see highway_env/envs/common/graphics.py::EnvViewer.display) when
        # config["real_time_rendering"] is True. It defaults to False, and
        # this project's own ENV_CONFIG (config.py) never sets it - so a
        # live render's actual pacing depended entirely on incidental
        # per-step render compute cost happening to roughly match
        # simulation_frequency, rather than anything guaranteed. On a
        # faster machine/GPU that incidental cost can drop well below the
        # target frame time, and the window can open, run, and close well
        # under a second - too fast for a human to perceive anything,
        # which looks exactly like "rendering doesn't work" even though no
        # exception is ever raised. Explicitly requesting real-time pacing
        # here removes that dependency for the one mode (live) where a
        # human is actually watching.
        render_config["real_time_rendering"] = True
    env.unwrapped.configure(render_config)
    env = CandidateRewardWrapper(env, shaping_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs, info = env.reset(seed=seed)
    obs_dim = int(np.prod(obs.shape))
    n_actions = env.action_space.n

    model = ActorCritic(obs_dim, n_actions, hidden_sizes=NET_ARCH).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    frames = None if live else []

    def _capture():
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

    env.close()
    return frames


def _save_frames_as_gif(frames: list, out_path: str, fps: int) -> None:
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

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(out_path, frames, duration=duration)


def _render_candidate(
    short: str, module_path: str, checkpoint_path: str,
    episodes: int, live: bool, render_dir: str, fps: int,
) -> None:
    """
    Renders `episodes` deterministic rollouts for one candidate. In file
    mode, saves one GIF per episode under render_dir/{short}_epN.gif and
    prints the path. In live mode, opens a window per episode (blocks until
    the episode ends) and saves nothing.
    """
    # Distinct seed range from evaluate_candidate.py's own eval seeds
    # (2000 + episode) so a render never silently reruns the exact same
    # rollout already scored in the numeric metrics above - it's a fresh,
    # separately-seeded look at the same policy.
    render_seed_base = 9000
    for ep in range(episodes):
        seed = render_seed_base + ep
        try:
            frames = _run_render_episode(module_path, checkpoint_path, seed, live)
        except Exception as e:
            print(f"  RENDER FAILED for {short} (episode {ep}): {e}")
            continue

        if live:
            print(f"  {short}: rendered episode {ep} live (seed={seed})")
            continue

        out_path = os.path.join(render_dir, f"{short}_ep{ep}.gif")
        try:
            _save_frames_as_gif(frames, out_path, fps)
        except RuntimeError as e:
            print(f"  {e}")
            return
        print(f"  {short}: saved render -> {out_path} ({len(frames)} frames @ {fps}fps)")


def _discover_trained_short_names() -> list[str]:
    """Every gen*.pt under eureka/checkpoints/ that also has a matching
    (non-rejected) source file - i.e. actually loadable and evaluable."""
    if not os.path.isdir(CHECKPOINTS_DIR):
        return []
    names = []
    for filename in sorted(os.listdir(CHECKPOINTS_DIR)):
        if not filename.endswith(".pt"):
            continue
        short = filename[: -len(".pt")]
        if os.path.isfile(_candidate_source_path(short)):
            names.append(short)
    return names


def _print_available() -> None:
    trained = set(_discover_trained_short_names())
    all_sources = set()
    if os.path.isdir(CANDIDATES_DIR):
        all_sources = {
            f[: -len(".py")]
            for f in os.listdir(CANDIDATES_DIR)
            if f.endswith(".py")
        }
    rejected = set()
    if os.path.isdir(REJECTED_DIR):
        rejected = {
            f[: -len(".py")]
            for f in os.listdir(REJECTED_DIR)
            if f.endswith(".py")
        }

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


def _print_single_result(short: str, metrics: dict, fitness: float, episodes: int) -> None:
    print(f"\n  {short}  ({episodes} deterministic episodes)")
    print(f"  {'-' * (len(short) + 2 + len(str(episodes)) + 24)}")
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


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate every trained candidate found under eureka/checkpoints/.",
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EVAL_EPISODES,
        help=f"Deterministic eval episodes per candidate (default: {N_EVAL_EPISODES}, "
             "same as the search loop's current N_EVAL_EPISODES).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override the checkpoint path. Only valid together with exactly one candidate name.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available/trained/rejected candidates and exit without evaluating anything.",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Also save a GIF of one deterministic episode per candidate to eureka/renders/ "
             "(needs the 'imageio' package: pip install imageio).",
    )
    parser.add_argument(
        "--render-only", action="store_true",
        help="Skip the numeric evaluate_candidate() pass entirely and only render. "
             "Useful with --render-live: the default numeric eval runs "
             f"{N_EVAL_EPISODES} full episodes before rendering even starts, which "
             "can take minutes and looks like nothing is happening if all you "
             "wanted was to watch the policy drive. Implies --render.",
    )
    parser.add_argument(
        "--render-live", action="store_true",
        help="Instead of saving a GIF, open a live window (render_mode='human'). "
             "Needs a local display; only valid with exactly one candidate.",
    )
    parser.add_argument(
        "--render-episodes", type=int, default=1,
        help="Number of episodes to render per candidate (default: 1).",
    )
    parser.add_argument(
        "--render-dir", type=str, default=DEFAULT_RENDER_DIR,
        help=f"Output directory for saved GIFs (default: {DEFAULT_RENDER_DIR}).",
    )
    parser.add_argument(
        "--render-fps", type=int, default=5,
        help="Frames per second for saved GIFs (default: 5, matching policy_frequency).",
    )
    args = parser.parse_args(argv)

    if args.list:
        _print_available()
        return 0

    if args.checkpoint and (args.all or len(args.candidates) != 1):
        parser.error("--checkpoint can only be used together with exactly one candidate name")

    # --render-live opens a real (or fake, over an unconfigured display)
    # window per rendered episode - that only makes sense pinned to exactly
    # one candidate, and we want to fail BEFORE running any training-free
    # but still nontrivial evaluation work, not partway through a multi-
    # candidate loop.
    if args.render_only:
        args.render = True
    if args.render_live:
        if not args.render:
            args.render = True
        if args.all or len(args.candidates) != 1:
            parser.error("--render-live can only be used with exactly one candidate name")

    if args.render_episodes < 1:
        parser.error("--render-episodes must be >= 1")

    names = list(args.candidates)
    if args.all or not names:
        names = _discover_trained_short_names()
        if not names:
            print(
                "No trained candidates found under eureka/checkpoints/. "
                "Run `python -m eureka.loop` first, or pass a candidate name "
                "explicitly with --checkpoint. Use --list to see what's on disk."
            )
            return 1

    if args.render and not args.render_live:
        os.makedirs(args.render_dir, exist_ok=True)

    results = []
    processed_count = 0
    for raw in names:
        short, module_path = _normalize(raw)
        checkpoint = args.checkpoint or _default_checkpoint_path(short)

        if not os.path.isfile(_candidate_source_path(short)):
            if os.path.isfile(_rejected_source_path(short)):
                print(
                    f"  SKIP {short}: this candidate was rejected by the sandbox "
                    f"and never trained (see {_rejected_source_path(short)} for why)."
                )
            else:
                print(f"  SKIP {short}: no source file at {_candidate_source_path(short)}.")
            continue

        if not os.path.isfile(checkpoint):
            print(
                f"  SKIP {short}: no checkpoint at {checkpoint}. "
                "Has this candidate actually been trained yet?"
            )
            continue

        if args.render_only:
            print(f"  Skipping numeric evaluation for {short} (--render-only).")
            processed_count += 1
        else:
            print(f"  Evaluating {short} ({args.episodes} episodes)...")
            try:
                metrics = evaluate_candidate(checkpoint, module_path, n_episodes=args.episodes)
            except Exception as e:
                print(f"  FAILED to evaluate {short}: {e}")
                continue

            fitness = compute_fitness(metrics, FITNESS_WEIGHTS)
            _print_single_result(short, metrics, fitness, args.episodes)
            processed_count += 1

            results.append({
                "module_path": module_path,
                "candidate_id": short,
                "pareto_rank": "-",
                "legacy_fitness": fitness,
                "metrics": metrics,
            })

        if args.render:
            _render_candidate(
                short, module_path, checkpoint,
                episodes=args.render_episodes,
                live=args.render_live,
                render_dir=args.render_dir,
                fps=args.render_fps,
            )

    if len(results) > 1:
        best = max(results, key=lambda r: r["legacy_fitness"])
        print_final_archive_table(results, representative_id=best["candidate_id"])

    return 0 if processed_count or args.list else 1


if __name__ == "__main__":
    sys.exit(main())