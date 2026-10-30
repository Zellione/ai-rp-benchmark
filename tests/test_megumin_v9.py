import pytest

from rpbench.megumin_v9 import (
    MeguminResolver,
    MeguminSlotError,
    make_engine_slots,
)


def test_recursive_expansion():
    resolver = MeguminResolver({
        "[[a]]": "Hello [[b]]",
        "[[b]]": "world",
    })
    assert resolver.render("[[a]]!") == "Hello world!"


def test_unknown_slot_rejected():
    resolver = MeguminResolver({})
    with pytest.raises(MeguminSlotError, match="undefined slot"):
        resolver.render("[[missing]]", identifier="main")


def test_cycle_rejected():
    resolver = MeguminResolver({
        "[[a]]": "[[b]]",
        "[[b]]": "[[a]]",
    })
    with pytest.raises(MeguminSlotError, match="Cyclic"):
        resolver.render("[[a]]")


def test_historical_engine_and_cot():
    resolver = make_engine_slots(
        {
            "id": "v9-core",
            "p1": "Engine one",
            "p4": "Engine four",
        },
        {
            "id": "cot-v9-english",
            "content": "Reason carefully.",
        },
    )

    assert resolver.render("[[prompt1]]") == "Engine one"
    assert resolver.render("[[prompt4]]") == "Engine four"
    assert resolver.render("[[THINK]]") == (
        "<think>\nReason carefully.\n</think>"
    )
    assert resolver.render("[[COT]]") == ""


def test_profile_resolver_rejects_unknown_engine():
    from rpbench.megumin_v9 import make_profile_resolver

    config = {
        "engine": "v10-core",
        "cot": "cot-v9-english",
    }
    with pytest.raises(ValueError, match="Unsupported Megumin engine"):
        make_profile_resolver({}, {}, config)


def test_profile_resolver_uses_historical_v9_suppression():
    from rpbench.megumin_v9 import make_profile_resolver

    config = {
        "engine": "v9-core",
        "cot": "cot-v9-english",
        "profile": {
            "cotEnabled": True,
            "thinkingV2": False,
            "userPronouns": "off",
            "addons": [],
            "blocks": [],
            "storyPlan": {"enabled": False},
            "imageGen": {"enabled": False},
            "dnRatio": {"enabled": False},
            "onomatopoeia": {"enabled": False},
            "toggles": {"ooc": False, "control": False},
            "userLanguage": "German",
            "v9Limits": {
                "leanMin": 300,
                "leanMax": 400,
                "fullMin": 700,
                "fullMax": 1200,
            },
        },
    }
    resolver = make_profile_resolver(
        {"id": "v9-core", "p1": "Hello"},
        {"id": "cot-v9-english", "content": "Think"},
        config,
    )
    assert resolver.render("[[main]][[OOC]][[control]]") == ""
    assert resolver.render("[[prompt1]]") == "Hello"
    assert "GERMAN ONLY" in resolver.render("[[Language]]")
    assert resolver.render("[[v9_full_max]]") == "1200"
