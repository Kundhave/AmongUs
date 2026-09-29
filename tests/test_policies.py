"""Tests for CrewmatePolicy and ImpostorPolicy: §11's strict priority orders."""

from dataclasses import replace

from amongus.agents.crewmate import CrewmatePolicy
from amongus.agents.impostor import ImpostorPolicy
from amongus.agents.reactor import Commitment, ReactorBoard
from amongus.config import SimConfig
from amongus.types import (
    AgentPhys,
    DoTask,
    HoldPanel,
    Kill,
    Move,
    Note,
    PressButton,
    Report,
    Role,
    Sabotage,
    TaskInstance,
    Vent,
    Wait,
)
from amongus.world.observation import Observation, SabotageAlarm

_CFG = SimConfig()


def _phys(
    agent_id: str,
    role: Role,
    room: str,
    tasks: list[TaskInstance] | None = None,
    kill_cooldown: int = 0,
) -> AgentPhys:
    """Build a minimal AgentPhys for an Observation's self_phys field."""
    return AgentPhys(
        id=agent_id, role=role, alive=True, room=room, transit=None,
        tasks=tasks or [], kill_cooldown=kill_cooldown, button_used=False,
    )


def _obs(
    self_id: str,
    self_phys: AgentPhys,
    room: str,
    tick: int = 10,
    occupants: list[str] | None = None,
    bodies_here: list[str] | None = None,
    local_events: list[dict] | None = None,
    partner_id: str | None = None,
    sabotage_alarm=None,
    death_notices: list[str] | None = None,
) -> Observation:
    """Build an Observation with the fields CrewmatePolicy/ImpostorPolicy actually read."""
    return Observation(
        self_id=self_id, tick=tick, room=room, self_phys=self_phys,
        occupants=occupants or [], bodies_here=bodies_here or [], local_events=local_events or [],
        task_bar=0.0, sabotage_alarm=sabotage_alarm, closed_doors=frozenset(), radio=[],
        stall_flag=False, death_notices=death_notices or [], partner_id=partner_id,
        partner_room=None,
    )


def test_crewmate_reports_body_over_pending_tasks() -> None:
    """Step 1 beats step 5: a body in the room is reported even with tasks pending."""
    policy = CrewmatePolicy("red", _CFG)
    tasks = [TaskInstance(task_id="t01", room="weapons", duration=4)]
    phys = _phys("red", Role.CREWMATE, "electrical", tasks=tasks)
    obs = _obs("red", phys, "electrical", bodies_here=["blue"])
    assert policy.decide(obs) == Report()


def test_crewmate_does_not_rereport_an_already_reported_body() -> None:
    """§7/§11.1 step 1: a body already in death_notices is not re-reported; falls to its task."""
    policy = CrewmatePolicy("red", _CFG)
    tasks = [TaskInstance(task_id="t01", room="electrical", duration=4)]
    phys = _phys("red", Role.CREWMATE, "electrical", tasks=tasks)
    obs = _obs("red", phys, "electrical", bodies_here=["blue"], death_notices=["blue"])
    action = policy.decide(obs)
    assert action != Report()
    assert action == DoTask(task_id="t01")


def test_crewmate_still_reports_an_unreported_body_alongside_a_reported_one() -> None:
    """A room with one reported and one still-unreported body is still Reported."""
    policy = CrewmatePolicy("red", _CFG)
    phys = _phys("red", Role.CREWMATE, "electrical")
    obs = _obs(
        "red", phys, "electrical", bodies_here=["blue", "green"], death_notices=["blue"]
    )
    assert policy.decide(obs) == Report()


def test_crewmate_reports_witnessed_kill() -> None:
    """Step 1: a kill witnessed this tick is reported."""
    policy = CrewmatePolicy("red", _CFG)
    phys = _phys("red", Role.CREWMATE, "electrical")
    events = [{"kind": "kill", "killer": "blue", "victim": "green", "room": "electrical"}]
    obs = _obs("red", phys, "electrical", local_events=events)
    assert policy.decide(obs) == Report()


def test_crewmate_shadows_suspect_over_tasks() -> None:
    """Step 4 beats step 5: a highly suspicious agent is shadowed even with tasks pending."""
    policy = CrewmatePolicy("red", _CFG)
    policy.memory.suspicion["blue"] = 0.9
    policy.memory.last_seen["blue"] = ("electrical", 5)
    tasks = [TaskInstance(task_id="t01", room="weapons", duration=4)]
    phys = _phys("red", Role.CREWMATE, "cafeteria", tasks=tasks)
    obs = _obs("red", phys, "cafeteria", tick=6)
    action = policy.decide(obs)
    assert isinstance(action, Move)


def test_crewmate_shadow_waits_when_colocated() -> None:
    """Shadowing a suspect already in the same room waits instead of moving."""
    policy = CrewmatePolicy("red", _CFG)
    policy.memory.suspicion["blue"] = 0.9
    policy.memory.last_seen["blue"] = ("electrical", 5)
    phys = _phys("red", Role.CREWMATE, "electrical")
    obs = _obs("red", phys, "electrical", tick=6)
    assert policy.decide(obs) == Wait()


def test_crewmate_below_theta_shadow_does_tasks_instead() -> None:
    """Low suspicion does not trigger shadowing; the agent proceeds to its tasks."""
    policy = CrewmatePolicy("red", _CFG)
    policy.memory.suspicion["blue"] = 0.1
    policy.memory.last_seen["blue"] = ("electrical", 5)
    tasks = [TaskInstance(task_id="t01", room="cafeteria", duration=4)]
    phys = _phys("red", Role.CREWMATE, "cafeteria", tasks=tasks)
    obs = _obs("red", phys, "cafeteria", tick=6)
    assert policy.decide(obs) == DoTask(task_id="t01")


def test_crewmate_does_task_on_arrival() -> None:
    """Standing in a task's room with that task undone does the task."""
    policy = CrewmatePolicy("red", _CFG)
    tasks = [TaskInstance(task_id="t01", room="electrical", duration=4)]
    phys = _phys("red", Role.CREWMATE, "electrical", tasks=tasks)
    obs = _obs("red", phys, "electrical")
    assert policy.decide(obs) == DoTask(task_id="t01")


def test_crewmate_picks_nearest_task_room() -> None:
    """Task selection is nearest-first by A* distance from the current room."""
    policy = CrewmatePolicy("red", _CFG)
    tasks = [
        TaskInstance(task_id="near", room="weapons", duration=4),  # cafeteria->weapons = 3
        TaskInstance(task_id="far", room="communications", duration=4),  # much further
    ]
    phys = _phys("red", Role.CREWMATE, "cafeteria", tasks=tasks)
    obs = _obs("red", phys, "cafeteria")
    policy.decide(obs)
    assert policy.memory.goal == "weapons"


def test_crewmate_patrol_avoids_risky_neighbor() -> None:
    """With no tasks left, the agent steps toward the lowest-risk neighboring room."""
    policy = CrewmatePolicy("red", _CFG)
    tasks = [TaskInstance(task_id="t01", room="cafeteria", duration=1, progress=1)]
    phys = _phys("red", Role.CREWMATE, "cafeteria", tasks=tasks)
    # cafeteria's neighbors: weapons, medbay, upper_engine, storage, admin.
    # Suspicion stays below theta_shadow so step 4 does not preempt patrol (step 6).
    policy.memory.suspicion["blue"] = 0.3
    policy.memory.last_seen["blue"] = ("weapons", 9)
    obs = _obs("red", phys, "cafeteria", tick=10)
    action = policy.decide(obs)
    assert isinstance(action, Move)
    assert action.to != "weapons"


def test_alpha_risk_changes_the_chosen_route() -> None:
    """Belief becomes geometry: alpha_risk > 0 changes the first hop versus alpha_risk = 0."""
    task = [TaskInstance(task_id="t11", room="shields", duration=4)]

    def first_hop(alpha_risk: float) -> str:
        cfg = replace(SimConfig(), alpha_risk=alpha_risk)
        policy = CrewmatePolicy("red", cfg)
        # Suspicion stays below theta_shadow so step 4 (shadow the sighting directly)
        # never preempts step 5's A*-routed task trip, isolating the risk-cost effect.
        policy.memory.suspicion["blue"] = 0.45
        policy.memory.last_seen["blue"] = ("o2", 9)  # o2 sits on the cheap weapons->shields leg
        phys = _phys("red", Role.CREWMATE, "weapons", tasks=task)
        obs = _obs("red", phys, "weapons", tick=10)
        action = policy.decide(obs)
        assert isinstance(action, Move)
        return action.to

    assert first_hop(0.0) == "o2"  # unweighted shortest path: weapons->o2->shields (6)
    assert first_hop(5.0) == "navigation"  # risk on o2 flips it to weapons->navigation->shields


def test_planner_astar_no_risk_ignores_suspicion_and_matches_plain_distance() -> None:
    """§8.2 ablation: `planner=astar_no_risk` zeroes the risk term even with alpha_risk > 0."""
    task = [TaskInstance(task_id="t11", room="shields", duration=4)]

    def first_hop(planner: str) -> str:
        cfg = replace(SimConfig(), alpha_risk=5.0, planner=planner)
        policy = CrewmatePolicy("red", cfg)
        policy.memory.suspicion["blue"] = 0.9
        policy.memory.last_seen["blue"] = ("o2", 9)  # o2 sits on the cheap weapons->shields leg
        phys = _phys("red", Role.CREWMATE, "weapons", tasks=task)
        obs = _obs("red", phys, "weapons", tick=10)
        action = policy.decide(obs)
        assert isinstance(action, Move)
        return action.to

    # With risk on, the same seen-blue-at-o2 signal that flips the route in
    # test_alpha_risk_changes_the_chosen_route also flips it here (astar plans around it)...
    assert first_hop("astar") == "navigation"
    # ...but astar_no_risk ignores suspicion entirely, reproducing the plain-distance route.
    assert first_hop("astar_no_risk") == "o2"


def test_crewmate_routes_to_button_after_witnessing_kill() -> None:
    """Step 4: an unreported kill note routes the crewmate toward button_room."""
    policy = CrewmatePolicy("red", _CFG)
    policy.memory.notes.append(Note(tick=5, kind="kill", text="t=5 I saw blue kill green."))
    tasks = [TaskInstance(task_id="t01", room="electrical", duration=4)]
    phys = _phys("red", Role.CREWMATE, "electrical", tasks=tasks)
    obs = _obs("red", phys, "electrical", tick=10)
    action = policy.decide(obs)
    assert isinstance(action, Move)  # not co-located with cafeteria (button_room): heads there


def test_crewmate_presses_button_on_arrival_with_witness_note() -> None:
    """Step 4: once at button_room with an unreported witness note, PressButton fires."""
    policy = CrewmatePolicy("red", _CFG)
    policy.memory.notes.append(Note(tick=5, kind="vent", text="t=5 I saw blue vent."))
    phys = _phys("red", Role.CREWMATE, "cafeteria")
    obs = _obs("red", phys, "cafeteria", tick=10)
    assert policy.decide(obs) == PressButton()


def test_crewmate_without_witness_note_never_presses() -> None:
    """No kill/vent note at all: the crewmate never presses, even standing at button_room."""
    policy = CrewmatePolicy("red", _CFG)
    phys = _phys("red", Role.CREWMATE, "cafeteria")
    obs = _obs("red", phys, "cafeteria", tick=10)
    assert policy.decide(obs) != PressButton()


def test_crewmate_witness_note_before_last_meeting_never_presses() -> None:
    """A kill/vent note that predates the most recent meeting note does not trigger a press."""
    policy = CrewmatePolicy("red", _CFG)
    policy.memory.notes.append(Note(tick=3, kind="kill", text="t=3 I saw blue kill green."))
    policy.memory.notes.append(Note(tick=6, kind="meeting", text="Meeting 1: blue was ejected."))
    phys = _phys("red", Role.CREWMATE, "cafeteria")
    obs = _obs("red", phys, "cafeteria", tick=10)
    assert policy.decide(obs) != PressButton()


def test_crewmate_button_used_never_presses_again() -> None:
    """A crewmate whose button is already used skips step 4 even with a fresh witness note."""
    policy = CrewmatePolicy("red", _CFG)
    policy.memory.notes.append(Note(tick=9, kind="kill", text="t=9 I saw blue kill green."))
    phys = _phys("red", Role.CREWMATE, "cafeteria")
    phys.button_used = True
    obs = _obs("red", phys, "cafeteria", tick=10)
    assert policy.decide(obs) != PressButton()


def test_crewmate_reactor_active_never_presses() -> None:
    """No meetings while a reactor sabotage is active: step 4 does not fire."""
    from amongus.world.observation import SabotageAlarm

    policy = CrewmatePolicy("red", _CFG)
    policy.memory.notes.append(Note(tick=9, kind="kill", text="t=9 I saw blue kill green."))
    phys = _phys("red", Role.CREWMATE, "cafeteria")
    alarm = SabotageAlarm(kind="reactor", panels=("reactor", "o2"), timer=20, panel_done={})
    obs = _obs("red", phys, "cafeteria", tick=10, sabotage_alarm=alarm)
    assert policy.decide(obs) != PressButton()


def test_impostor_kills_lone_crewmate() -> None:
    """Step 1: a single non-partner occupant with cooldown 0 is killed."""
    policy = ImpostorPolicy("black", _CFG)
    phys = _phys("black", Role.IMPOSTOR, "electrical", kill_cooldown=0)
    obs = _obs("black", phys, "electrical", occupants=["red"])
    assert policy.decide(obs) == Kill(target="red")


def test_impostor_does_not_kill_with_extra_witness() -> None:
    """No kill when more than one non-partner occupant is observed."""
    policy = ImpostorPolicy("black", _CFG)
    phys = _phys("black", Role.IMPOSTOR, "electrical", kill_cooldown=0)
    obs = _obs("black", phys, "electrical", occupants=["red", "green"])
    action = policy.decide(obs)
    assert not isinstance(action, Kill)


def test_impostor_does_not_kill_on_cooldown() -> None:
    """No kill while the kill cooldown is nonzero."""
    policy = ImpostorPolicy("black", _CFG)
    phys = _phys("black", Role.IMPOSTOR, "electrical", kill_cooldown=3)
    obs = _obs("black", phys, "electrical", occupants=["red"])
    action = policy.decide(obs)
    assert not isinstance(action, Kill)


def test_impostor_vents_out_the_tick_after_a_kill() -> None:
    """The tick after a kill, the impostor vents out if its room has a vent."""
    policy = ImpostorPolicy("black", _CFG)
    phys = _phys("black", Role.IMPOSTOR, "electrical", kill_cooldown=0)
    obs = _obs("black", phys, "electrical", occupants=["red"], tick=10)
    policy.decide(obs)  # tick 10: kills, schedules escape for tick 11

    phys2 = _phys("black", Role.IMPOSTOR, "electrical", kill_cooldown=20)
    obs2 = _obs("black", phys2, "electrical", tick=11)
    action = policy.decide(obs2)
    assert isinstance(action, Vent)  # electrical vents to medbay/security


def test_impostor_never_reports_a_body() -> None:
    """Impostors never self-report, even when a body is in their own room."""
    policy = ImpostorPolicy("black", _CFG)
    phys = _phys("black", Role.IMPOSTOR, "electrical", kill_cooldown=5)
    obs = _obs("black", phys, "electrical", bodies_here=["red"])
    action = policy.decide(obs)
    assert not isinstance(action, Report)


def test_impostor_hunts_last_seen_crewmate() -> None:
    """Step 4: with no kill available, the impostor routes toward a last-seen crewmate."""
    policy = ImpostorPolicy("black", _CFG)
    policy.memory.last_seen["red"] = ("electrical", 9)
    phys = _phys("black", Role.IMPOSTOR, "cafeteria", kill_cooldown=5)
    obs = _obs("black", phys, "cafeteria", tick=10)
    action = policy.decide(obs)
    assert isinstance(action, Move)


def _committed_board(agent_id: str, target: str, eta: int = 999) -> ReactorBoard:
    """A ReactorBoard where agent_id already holds a live, unrevoked commitment on target."""
    board = ReactorBoard(is_open=True, panels=("reactor", "o2"))
    board.commits[target] = Commitment(agent=agent_id, target=target, eta=eta)
    return board


def test_crewmate_holds_committed_reactor_panel_on_arrival() -> None:
    """Step 2: standing at a committed, undone reactor panel holds it."""
    board = _committed_board("red", "reactor")
    policy = CrewmatePolicy("red", _CFG, reactor_board=board)
    phys = _phys("red", Role.CREWMATE, "reactor")
    alarm = SabotageAlarm(kind="reactor", panels=("reactor", "o2"), timer=20, panel_done={})
    obs = _obs("red", phys, "reactor", tick=10, sabotage_alarm=alarm)
    assert policy.decide(obs) == HoldPanel(room="reactor")


def test_crewmate_routes_to_committed_reactor_panel_over_tasks() -> None:
    """Step 2 beats step 6: a committed panel is routed to even with tasks pending."""
    board = _committed_board("red", "reactor")
    policy = CrewmatePolicy("red", _CFG, reactor_board=board)
    tasks = [TaskInstance(task_id="t01", room="weapons", duration=4)]
    phys = _phys("red", Role.CREWMATE, "cafeteria", tasks=tasks)
    alarm = SabotageAlarm(kind="reactor", panels=("reactor", "o2"), timer=20, panel_done={})
    obs = _obs("red", phys, "cafeteria", tick=10, sabotage_alarm=alarm)
    action = policy.decide(obs)
    assert isinstance(action, Move)


def test_crewmate_ignores_done_reactor_commitment() -> None:
    """Step 2 does not fire once the committed panel is already marked done."""
    board = _committed_board("red", "reactor")
    policy = CrewmatePolicy("red", _CFG, reactor_board=board)
    phys = _phys("red", Role.CREWMATE, "reactor")
    alarm = SabotageAlarm(
        kind="reactor", panels=("reactor", "o2"), timer=20, panel_done={"reactor": True}
    )
    obs = _obs("red", phys, "reactor", tick=10, sabotage_alarm=alarm)
    assert policy.decide(obs) != HoldPanel(room="reactor")


def test_crewmate_commit_protocol_off_rushes_nearest_panel() -> None:
    """Ablation: with commit_protocol off, an uncommitted crewmate still routes to a panel."""
    cfg = replace(SimConfig(), commit_protocol=False)
    policy = CrewmatePolicy("red", cfg)
    phys = _phys("red", Role.CREWMATE, "cafeteria")
    alarm = SabotageAlarm(kind="reactor", panels=("reactor", "o2"), timer=20, panel_done={})
    obs = _obs("red", phys, "cafeteria", tick=10, sabotage_alarm=alarm)
    action = policy.decide(obs)
    assert isinstance(action, Move)


def test_crewmate_holds_lights_panel_within_six_ticks() -> None:
    """Step 3: lights active and within 6 ticks of electrical routes there and holds it."""
    policy = CrewmatePolicy("red", _CFG)
    phys = _phys("red", Role.CREWMATE, "electrical")
    alarm = SabotageAlarm(kind="lights", panels=("electrical",), timer=0, panel_done={})
    obs = _obs("red", phys, "electrical", tick=10, sabotage_alarm=alarm)
    assert policy.decide(obs) == HoldPanel(room="electrical")


def test_crewmate_skips_lights_panel_beyond_six_ticks() -> None:
    """Step 3 does not fire when electrical is more than 6 plain-cost ticks away."""
    policy = CrewmatePolicy("red", _CFG)
    tasks = [TaskInstance(task_id="t01", room="navigation", duration=4)]
    phys = _phys("red", Role.CREWMATE, "navigation", tasks=tasks)
    alarm = SabotageAlarm(kind="lights", panels=("electrical",), timer=0, panel_done={})
    obs = _obs("red", phys, "navigation", tick=10, sabotage_alarm=alarm)
    action = policy.decide(obs)
    assert action != HoldPanel(room="electrical")


class _FixedRNG:
    """A stub RNG: `.random()` replays a fixed queue, `.choice()` always returns one value."""

    def __init__(self, random_values: list[float], choice_value: str = "lights") -> None:
        """Store the scripted `.random()` return sequence and the fixed `.choice()` result."""
        self._values = list(random_values)
        self._choice_value = choice_value

    def random(self) -> float:
        """Pop and return the next scripted value."""
        return self._values.pop(0)

    def choice(self, _options, p=None):  # noqa: ANN001 — mirrors numpy.random.Generator.choice
        """Always return the scripted sabotage kind."""
        return self._choice_value


def test_impostor_sabotages_when_eligible_and_roll_hits() -> None:
    """Step 2: off cooldown, >=3 crewmates alive, roll under p_sabotage -> Sabotage(kind)."""
    cfg = replace(SimConfig(), p_sabotage=1.0)
    rng = _FixedRNG(random_values=[0.0], choice_value="lights")
    policy = ImpostorPolicy("black", cfg, rng=rng)
    phys = _phys("black", Role.IMPOSTOR, "cafeteria", kill_cooldown=5)
    obs = _obs("black", phys, "cafeteria", tick=10)
    assert policy.decide(obs) == Sabotage(kind="lights")


def test_impostor_does_not_sabotage_below_three_crewmates() -> None:
    """Step 2's headcount gate: fewer than 3 known-alive crewmates blocks sabotage."""
    cfg = replace(SimConfig(), n_players=4, n_impostors=1, p_sabotage=1.0)
    rng = _FixedRNG(random_values=[0.0], choice_value="lights")
    policy = ImpostorPolicy("black", cfg, rng=rng)
    policy._known_dead = {"red", "blue"}  # 3 crewmates total - 2 known dead = 1 alive
    phys = _phys("black", Role.IMPOSTOR, "cafeteria", kill_cooldown=5)
    obs = _obs("black", phys, "cafeteria", tick=10)
    assert policy.decide(obs) != Sabotage(kind="lights")


def test_impostor_does_not_sabotage_while_one_is_already_active() -> None:
    """Step 2 never fires a second sabotage while one is already active."""
    rng = _FixedRNG(random_values=[0.0], choice_value="lights")
    policy = ImpostorPolicy("black", replace(SimConfig(), p_sabotage=1.0), rng=rng)
    phys = _phys("black", Role.IMPOSTOR, "cafeteria", kill_cooldown=5)
    alarm = SabotageAlarm(kind="doors", panels=(), timer=5, panel_done={})
    obs = _obs("black", phys, "cafeteria", tick=10, sabotage_alarm=alarm)
    action = policy.decide(obs)
    assert not isinstance(action, Sabotage)


def test_impostor_defects_by_waiting_on_a_won_commitment() -> None:
    """Step 3: a defecting impostor Waits forever on its own won reactor commitment."""
    board = ReactorBoard(is_open=True, panels=("reactor", "o2"))
    board.commits["reactor"] = Commitment(agent="black", target="reactor", eta=999)
    policy = ImpostorPolicy("black", _CFG, reactor_board=board)
    policy._defect_intent = True
    phys = _phys("black", Role.IMPOSTOR, "cafeteria", kill_cooldown=5)
    alarm = SabotageAlarm(kind="reactor", panels=("reactor", "o2"), timer=20, panel_done={})
    obs = _obs("black", phys, "cafeteria", tick=10, sabotage_alarm=alarm)
    assert policy.decide(obs) == Wait()


def test_impostor_fixes_honestly_when_not_defecting() -> None:
    """Step 3: a non-defecting impostor routes to and holds its own won commitment."""
    board = ReactorBoard(is_open=True, panels=("reactor", "o2"))
    board.commits["reactor"] = Commitment(agent="black", target="reactor", eta=999)
    policy = ImpostorPolicy("black", _CFG, reactor_board=board)
    policy._defect_intent = False
    phys = _phys("black", Role.IMPOSTOR, "reactor", kill_cooldown=5)
    alarm = SabotageAlarm(kind="reactor", panels=("reactor", "o2"), timer=20, panel_done={})
    obs = _obs("black", phys, "reactor", tick=10, sabotage_alarm=alarm)
    assert policy.decide(obs) == HoldPanel(room="reactor")
