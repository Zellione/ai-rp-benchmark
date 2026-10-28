import pytest

from rpbench.st_template_engine import (
    TemplateContext,
    TemplateError,
    render_template,
)


def test_assignment_persists():
    ctx = TemplateContext()
    assert render_template(
        "{{ .feature = 1 }}", ctx
    ) == ""
    assert render_template(
        "{{ if .feature }}ON{{ /if }}", ctx
    ) == "ON"


def test_condition_false():
    ctx = TemplateContext()
    render_template("{{ .feature = 0 }}", ctx)

    assert render_template(
        "before{{ if .feature }}hidden{{ /if }}after",
        ctx,
    ) == "beforeafter"


def test_else():
    ctx = TemplateContext(
        variables={".feature": False}
    )

    assert render_template(
        "{{ if .feature }}yes{{ else }}no{{ /if }}",
        ctx,
    ) == "no"


def test_or_and_negation():
    ctx = TemplateContext(
        variables={
            ".a": False,
            ".b": True,
        }
    )

    assert render_template(
        "{{ if !.a || .b }}yes{{ /if }}", ctx
    ) == "yes"


def test_inactive_assignment_ignored():
    ctx = TemplateContext(
        variables={".feature": False}
    )

    render_template(
        "{{ if .feature }}"
        "{{ .other = 1 }}"
        "{{ /if }}",
        ctx,
    )

    assert ".other" not in ctx.variables


def test_comment_and_user():
    ctx = TemplateContext(user_name="Player")

    assert render_template(
        "{{// ignored }}Hello {{ user }}",
        ctx,
    ) == "Hello Player"


def test_undefined_variable_rejected():
    with pytest.raises(TemplateError, match="Undefined"):
        render_template(
            "{{ if .missing }}yes{{ /if }}",
            TemplateContext(),
        )


def test_unknown_expression_rejected():
    with pytest.raises(TemplateError, match="Unsupported"):
        render_template(
            "{{ execute something }}",
            TemplateContext(),
        )


def test_unclosed_condition_rejected():
    with pytest.raises(TemplateError, match="Unclosed"):
        render_template(
            "{{ .x = 1 }}{{ if .x }}test",
            TemplateContext(),
        )


def test_nested_condition():
    ctx = TemplateContext(
        variables={".a": True, ".b": False}
    )
    assert render_template(
        "{{ if .a }}A"
        "{{ if .b }}B{{ else }}C{{ /if }}"
        "{{ /if }}",
        ctx,
    ) == "AC"



def test_voyage_known_defect_repaired():
    from rpbench.st_template_engine import (
        VOYAGE_LOCATION_ID,
        VOYAGE_V6_SHA256,
        repair_known_template_defect,
    )

    original = "{{ trim }{{ if .setting_tracker_location }}OK{{ /if }}"

    repaired, fixes = repair_known_template_defect(
        original,
        source_sha256=VOYAGE_V6_SHA256,
        identifier=VOYAGE_LOCATION_ID,
    )

    assert fixes == ["voyage-v6-location-sheet-missing-brace"]
    assert repaired == (
        "{{ trim }}{{ if .setting_tracker_location }}OK{{ /if }}"
    )

    ctx = TemplateContext(
        variables={".setting_tracker_location": True}
    )
    assert render_template(repaired, ctx) == "OK"


def test_voyage_defect_not_repaired_for_other_source():
    from rpbench.st_template_engine import (
        VOYAGE_LOCATION_ID,
        repair_known_template_defect,
    )

    original = "{{ trim }{{ if .setting_tracker_location }}"

    repaired, fixes = repair_known_template_defect(
        original,
        source_sha256="different-source",
        identifier=VOYAGE_LOCATION_ID,
    )

    assert repaired == original
    assert fixes == []
