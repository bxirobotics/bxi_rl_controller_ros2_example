from bxi_example_py_elf3.utils.robot_state_base import RobotControlState


def _state_behavior_classes():
    def walk_subclasses(base_class):
        for subclass in base_class.__subclasses__():
            yield subclass
            yield from walk_subclasses(subclass)

    return {cls.__name__: cls for cls in walk_subclasses(RobotControlState)}


def _allocate_state_id(state_name, state_config, used_ids, next_id):
    configured_id = (state_config or {}).get("id")
    if configured_id is not None:
        state_id = int(configured_id)
        if state_id in used_ids:
            raise ValueError(f"duplicate state id {state_id} for state: {state_name}")
        used_ids.add(state_id)
        next_id = max(next_id, state_id + 1)
        return state_id, next_id

    while next_id in used_ids:
        next_id += 1
    state_id = next_id
    used_ids.add(state_id)
    return state_id, next_id + 1


def build_robot_states(config):
    states_config = config.get("states", {})
    if not states_config:
        raise ValueError("state machine config must define states")

    behavior_classes = _state_behavior_classes()
    states = {}
    used_ids = set()
    next_id = 0

    for state_name, state_config in states_config.items():
        state_config = state_config or {}
        behavior_name = state_config.get("behavior")
        if not behavior_name:
            raise ValueError(f"state '{state_name}' must define behavior")

        behavior_class = behavior_classes.get(behavior_name)
        if behavior_class is None:
            raise ValueError(
                f"unknown state behavior '{behavior_name}' for state '{state_name}'"
            )

        state_id, next_id = _allocate_state_id(
            state_name, state_config, used_ids, next_id
        )
        params = state_config.get("params", {}) or {}
        state = behavior_class(state_name, state_id, **params)
        state.speed_profile_name = state_config.get("speed_profile")
        states[state_name] = state

    return states
