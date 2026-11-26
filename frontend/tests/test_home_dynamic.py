"""Suggested questions for the Home page: built from real column names, never invented values."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from home_dynamic import GENERIC_QUESTIONS, dataset_questions, has_active_run, suggestions  # noqa: E402


def _ds(cols, rows=500, id=1, at="2026-01-01"):
    return {"id": id, "row_count": rows, "uploaded_at": at, "profile": {"columns": cols}}


SALES = [
    {"name": "order_id", "kind": "numeric", "unique_count": 500},
    {"name": "order_date", "kind": "datetime", "unique_count": 300},
    {"name": "region", "kind": "categorical", "unique_count": 4},
    {"name": "total_revenue", "kind": "numeric", "unique_count": 480},
    {"name": "units_sold", "kind": "numeric", "unique_count": 40},
]


def test_questions_use_real_columns_and_skip_identifiers():
    qs = dataset_questions(_ds(SALES))
    assert "Which region has the highest total revenue?" in qs
    assert "How has total revenue changed over time?" in qs
    assert not any("order id" in q for q in qs), "an id column is not something to total"


def test_no_usable_columns_gives_no_questions_and_falls_back_to_generic():
    assert dataset_questions(_ds([{"name": "notes", "kind": "text", "unique_count": 500}])) == []
    got = suggestions([_ds([{"name": "notes", "kind": "text", "unique_count": 500}])])
    assert [q for _, q in got] == GENERIC_QUESTIONS and all(i is None for i, _ in got)
    assert [q for _, q in suggestions([])] == GENERIC_QUESTIONS


def test_newest_dataset_comes_first_and_carries_its_id():
    old = _ds(SALES, id=1, at="2026-01-01")
    new = _ds([{"name": "city", "kind": "categorical", "unique_count": 5},
               {"name": "price", "kind": "numeric", "unique_count": 50}], id=2, at="2026-06-01")
    got = suggestions([old, new])
    assert got[0] == (2, "Which city has the highest price?")
    assert len(got) == 4 and len({q for _, q in got}) == 4


def test_a_running_analysis_is_detected():
    assert has_active_run([{"status": "verified"}, {"status": "running"}])
    assert not has_active_run([{"status": "verified"}, {"status": "failed"}])
    assert not has_active_run([])


def test_scene_loop_cannot_die_on_a_bad_frame_or_a_negative_radius():
    from hero_scene import SCENE_HTML

    assert "Math.max(0, waveR - 7)" in SCENE_HTML, "arc() throws on a negative radius at t=4 s, which froze the loop"
    assert SCENE_HTML.index("requestAnimationFrame(frame)") < SCENE_HTML.index("try { draw("), "the next frame is scheduled before drawing"
    assert "catch (e)" in SCENE_HTML and "__hero" not in SCENE_HTML, "errors are contained and no debug hooks ship"


def test_scene_is_a_transparent_page_backdrop_not_a_boxed_panel():
    from hero_scene import SCENE_HTML

    assert 'content="dark"' in SCENE_HTML, "the iframe must declare the page's colour scheme or the browser paints it opaque"
    assert "destination-out" in SCENE_HTML, "old frames fade to transparent so the page shows through"
    assert "bg(" not in SCENE_HTML and "border-radius:28px" not in SCENE_HTML, "no painted panel background or rounded box"


def test_landing_core_scene_is_an_honest_transparent_demo():
    from landing_scene import LANDING_SCENE_HTML as html

    assert 'content="dark"' in html and "clearRect" in html, "transparent canvas so the page's aurora shows through"
    assert "prefers-reduced-motion" in html and "requestAnimationFrame(frame)" in html
    f = html.index("function frame")
    assert html.index("requestAnimationFrame(frame)", f) < html.index("try {", f), "the next frame is scheduled before drawing, so one bad frame can never stop the loop"
    assert "__lp" not in html, "no debug hooks ship"
    for agent in ("Planner", "Analyst", "Critic"):
        assert agent in html or agent.lower() in html


def test_landing_core_is_a_shader_with_a_graceful_fallback_and_layered_orbits():
    from landing_scene import LANDING_SCENE_HTML as html

    assert "getContext('webgl'" in html and "gl_FragColor" in html, "the glossy form is rendered by a WebGL shader"
    assert "premultipliedAlpha: true" in html and "gl.clearColor(0, 0, 0, 0)" in html, "the form composites over the page"
    assert html.count('<canvas id=') == 3, "back orbits, the shader, then front orbits: beads pass behind and in front of the form"
    assert "if (glOK) drawGL(" in html and "else {" in html, "without WebGL a soft 2D orb is drawn instead"
    assert "glScale *= .85" in html, "the shader's resolution steps down if frames run long"


def test_signing_in_lands_on_home_whichever_page_the_tab_was_on(monkeypatch):
    from types import SimpleNamespace as NS

    import auth

    moves, state = [], {}
    monkeypatch.setattr(auth.st, "session_state", state, raising=False)
    monkeypatch.setattr(auth.st, "switch_page", lambda target: moves.append(target))

    def at(url):
        monkeypatch.setattr(auth.st, "context", NS(url=url), raising=False)

    at("http://x.test/Analyses")
    auth.go_home_if_asked()
    assert moves == [], "no explicit sign-in step, so a refresh or a deep link stays where it is"

    state["_go_home"] = True
    auth.go_home_if_asked()
    assert moves == ["Home.py"] and "_go_home" not in state, "moved once, and the flag is used up"

    auth.go_home_if_asked()
    assert moves == ["Home.py"], "not again on the next run"

    state["_go_home"] = True
    at("http://x.test/")
    auth.go_home_if_asked()
    assert moves == ["Home.py"], "a tab already on Home stays put"

    state["_go_home"] = True
    monkeypatch.setattr(auth.st, "context", None, raising=False)
    auth.go_home_if_asked()
    assert moves == ["Home.py"], "if the address cannot be read, nothing moves"
