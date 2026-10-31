"""Hash external inputs that affect simulation prompt construction."""

from pathlib import Path

from .simulation_matrix import digest, load_json, resolve


def dependency_digests(scenario_path, config_path):
    scenario_path = Path(scenario_path).resolve()
    config_path = Path(config_path).resolve()

    scenario = load_json(scenario_path)
    config = load_json(config_path)

    files = {}

    def add(label, base, reference):
        path = resolve(base, reference)
        files[label] = digest(path)

    for index, reference in enumerate(scenario.get("character_cards", [])):
        add(
            f"character_card:{index}",
            scenario_path.parent,
            reference,
        )

    for key in ("prompt_profile", "completion_preset"):
        if key in config and config[key] is not None:
            add(key, config_path.parent, config[key])

    runtime = config.get("runtime", {})
    st = runtime.get("st_chat_npc")

    if st is not None:
        add(
            "st_chat_npc:preset",
            config_path.parent,
            st["preset"],
        )

    arfyjb = runtime.get("arfyjb_npc")
    if arfyjb is not None:
        add(
            "arfyjb_npc:source",
            config_path.parent,
            arfyjb["source"],
        )

    megumin = runtime.get("megumin_npc")
    if megumin is not None:
        profile_path = resolve(
            config_path.parent, megumin["config"]
        )
        engine_dir = (
            config_path.parent / megumin["engine_directory"]
        ).resolve()
        profile = load_json(profile_path)

        files["megumin:profile"] = digest(profile_path)
        files["megumin:source"] = digest(
            resolve(Path.cwd(), profile["source"]["preset"])
        )
        for name in ("engine.json", "cot.json"):
            engine_file = engine_dir / name
            if not engine_file.is_file():
                raise ValueError(f"Missing Megumin engine file: {engine_file}")
            files[f"megumin:{name}"] = digest(engine_file)

    return files
