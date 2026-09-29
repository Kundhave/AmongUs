"""Frozen simulation configuration; every tunable lives here (SPEC §3)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Every knob the simulation reads; scenarios override via dataclasses.replace."""

    seed: int = 0
    n_players: int = 8
    n_impostors: int = 2
    colors: tuple[str, ...] = (
        "red",
        "blue",
        "green",
        "pink",
        "orange",
        "yellow",
        "black",
        "white",
    )
    max_ticks: int = 400
    tasks_per_crewmate: int = 4
    task_duration_noise: int = 1
    kill_cooldown: int = 20
    button_room: str = "cafeteria"
    p_miss: float = 0.02
    p_miss_lights: float = 0.50
    p_body_visible_lights: float = 0.70
    alpha_risk: float = 2.0
    last_seen_decay: int = 15
    theta_vote: float = 0.35
    theta_shadow: float = 0.50
    meeting_rounds: int = 2
    notes_in_prompt: int = 12
    ejection_reveals_role: bool = True
    sabotage_cooldown: int = 30
    p_sabotage: float = 0.08
    reactor_timer: int = 30
    reactor_panels: tuple[str, ...] = ("reactor", "o2")
    panel_hold_ticks: int = 2
    eta_grace: int = 3
    lights_fix_room: str = "electrical"
    doors_duration: int = 10
    p_defect: float = 0.5
    impostor_underbid: float = 0.5
    deadlock_window: int = 20
    shadow_cooldown: int = 15
    commit_protocol: bool = True
    deadlock_protocol: bool = True
    planner: str = "astar"
    deliberator: str = "gemini"
    llm_model: str = "gemini-3.8-flash"
    llm_cache: str = "runs/llm_cache.jsonl"
    llm_max_attempts: int = 4
    llm_backoff_base: float = 1.5
    llm_max_workers: int = 4
    verbosity: int = 1
