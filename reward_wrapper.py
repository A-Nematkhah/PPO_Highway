"""
reward_wrapper.py

Adds two custom shaping terms on top of highway-env's built-in reward
(which is just a fixed weighted sum of collision/right_lane/high_speed/
lane_change terms - see ENV_CONFIG in config.py):

    1. TTC penalty  - a *continuous* penalty based on time-to-collision to
       the nearest vehicle ahead in the same lane, rather than only
       penalizing an actual collision after the fact. This gives the agent
       a gradient pointing away from dangerous situations before they
       become crashes.

    2. Overtake bonus - an explicit reward the instant the ego vehicle
       passes another vehicle, computed by comparing each vehicle's
       longitudinal position relative to ego between consecutive steps.

Vehicle identity across steps is tracked via id(vehicle) (a "fingerprint"
based on the actual Python object), NOT by row index in the observation.
Row order changes every step because observations are sorted by distance
("order": "sorted" in ENV_CONFIG) - indexing by row would silently compare
different vehicles across steps.
"""

import gymnasium as gym


def compute_overtakes(road, ego, prev_relative_x: dict):
    """
    Compares each nearby vehicle's relative longitudinal position to last
    step. A vehicle that was ahead of ego (relative_x > 0) and is now
    behind (relative_x <= 0) has just been overtaken.

    Identity is tracked via id(vehicle) - a "fingerprint" based on the
    actual Python object, NOT row index in the observation (row order
    changes every step because observations are sorted by distance).

    Returns (n_overtakes, current_relative_x) - the caller should store
    current_relative_x and pass it back in as prev_relative_x next step.
    """
    current_relative_x = {}
    n_overtakes = 0

    for vehicle in road.vehicles:
        if vehicle is ego:
            continue

        fingerprint = id(vehicle)
        relative_x = vehicle.position[0] - ego.position[0]
        current_relative_x[fingerprint] = relative_x

        prev_relative_x_value = prev_relative_x.get(fingerprint)
        if prev_relative_x_value is not None and prev_relative_x_value > 0 >= relative_x:
            n_overtakes += 1

    return n_overtakes, current_relative_x


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, ttc_threshold: float = 3.0, ttc_weight: float = 0.1,
                 overtake_bonus: float = 0.2, llm_judge_fn=None,
                 llm_judge_weight: float = 0.5, llm_judge_every_n_episodes: int = 1):
        """
        ttc_threshold: TTC (seconds) below which the penalty kicks in.
                       3.0s is a common driving-safety rule of thumb.
        ttc_weight:    max penalty magnitude, applied when TTC -> 0.
        overtake_bonus: flat reward added per vehicle overtaken in a step.

        llm_judge_fn: optional callable(stats_dict) -> 0 or 1. ORPHAN HOOK:
                      `llm_judge.py` was removed; env_utils only wires this
                      when USE_LLM_JUDGE=True, which currently ImportErrors.
                      Leave None (normal / EUREKA path).
        llm_judge_weight: multiplier applied to the judge's 0/1 score
                          before adding it to the terminal step's reward.
        llm_judge_every_n_episodes: only call llm_judge_fn on every Nth
                          finished episode (per env instance), to control
                          API cost/latency. 1 = judge every episode.
        """
        super().__init__(env)
        self.ttc_threshold = ttc_threshold
        self.ttc_weight = ttc_weight
        self.overtake_bonus = overtake_bonus

        self.llm_judge_fn = llm_judge_fn
        self.llm_judge_weight = llm_judge_weight
        self.llm_judge_every_n_episodes = max(1, llm_judge_every_n_episodes)

        # fingerprint (id(vehicle)) -> relative longitudinal position last step
        self._prev_relative_x = {}

        # per-episode accumulators, used to build the summary passed to the
        # LLM judge at episode end
        self._episode_speed_sum = 0.0
        self._episode_overtake_count = 0
        self._episode_steps = 0
        self._episode_count = 0  # how many episodes finished so far (for the N-episode sampling)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_relative_x = {}
        self._episode_speed_sum = 0.0
        self._episode_overtake_count = 0
        self._episode_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ttc_penalty = self._compute_ttc_penalty()
        overtake_bonus, n_overtakes = self._compute_overtake_bonus()

        shaped_reward = reward + ttc_penalty + overtake_bonus

        # exposed for logging/diagnostics — does not affect the
        # env's own info keys (e.g. "crashed", "speed")
        info["ttc_penalty"] = ttc_penalty
        info["overtake_bonus"] = overtake_bonus
        info["n_overtakes"] = n_overtakes
        info["raw_reward"] = reward  # the original, unshaped reward

        # track running episode stats regardless of whether the judge will
        # actually be called this episode (cheap, and keeps counters correct)
        self._episode_speed_sum += info.get("speed", 0.0)
        self._episode_overtake_count += n_overtakes
        self._episode_steps += 1

        done = terminated or truncated
        info["llm_judge_score"] = None
        info["llm_judge_bonus"] = 0.0

        if done and self.llm_judge_fn is not None:
            self._episode_count += 1
            if self._episode_count % self.llm_judge_every_n_episodes == 0:
                episode_stats = {
                    "crashed": bool(info.get("crashed", False)),
                    "mean_speed": self._episode_speed_sum / max(self._episode_steps, 1),
                    "overtakes": self._episode_overtake_count,
                    "length": self._episode_steps,
                }
                score = self.llm_judge_fn(episode_stats)  # blocking call if judge is wired
                llm_bonus = self.llm_judge_weight * score

                shaped_reward += llm_bonus
                info["llm_judge_score"] = score
                info["llm_judge_bonus"] = llm_bonus

        return obs, shaped_reward, terminated, truncated, info

    def _compute_ttc_penalty(self) -> float:
        """
        Finds the smallest time-to-collision among vehicles ahead of ego in
        the same lane that ego is closing in on, and converts it into a
        penalty that grows as TTC shrinks toward zero. Vehicles in other
        lanes, vehicles behind ego, or vehicles ego isn't gaining on are
        ignored (TTC is undefined/infinite for them).
        """
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        if ego is None:
            return 0.0

        min_ttc = float("inf")

        for vehicle in unwrapped.road.vehicles:
            if vehicle is ego:
                continue

            # only consider vehicles in the same lane
            if vehicle.lane_index[2] != ego.lane_index[2]:
                continue

            dx = vehicle.position[0] - ego.position[0]
            if dx <= 0:
                continue  # not ahead of ego

            closing_speed = ego.speed - vehicle.speed
            if closing_speed <= 0:
                continue  # ego isn't gaining on this vehicle

            ttc = dx / closing_speed
            min_ttc = min(min_ttc, ttc)

        if min_ttc < self.ttc_threshold:
            severity = (self.ttc_threshold - min_ttc) / self.ttc_threshold  # in (0, 1]
            return -self.ttc_weight * severity

        return 0.0

    def _compute_overtake_bonus(self):
        """
        Delegates overtake counting to compute_overtakes() so baseline and
        EUREKA paths share one implementation. Returns (total_bonus,
        n_overtakes) for this step.
        """
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        n_overtakes, self._prev_relative_x = compute_overtakes(
            unwrapped.road, ego, self._prev_relative_x
        )
        return self.overtake_bonus * n_overtakes, n_overtakes
