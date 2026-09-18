"""Layer 2 — a language model's per-product profile, written onto the catalog row.

The model is mocked at the HTTP layer. What is under test is everything around it: the question
built from a row, the answer's validation, and the rules deciding what an answer may overwrite.
"""

from __future__ import annotations

import json
import tempfile

import httpx
import pytest
from bcd_enrich.profile import (
    PROFILE_TOOL,
    Claude,
    ProductProfile,
    apply_profile,
    describe,
    request_params,
    select_products,
    system_prompt,
    tool_input,
)
from bcd_ingest.store import MedallionStore
from bcd_schema import (
    SENSORY_AXES,
    Category,
    ExtractionMethod,
    Product,
    ProductSpec,
    Provenance,
    SensorySource,
    SensoryVector,
    Sourced,
)

FILED = Provenance(source_id="ttb", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0,
                   quote="Other Rum Gold Usb")
STATED = Provenance(source_id="maker", method=ExtractionMethod.STATED_BY_PRODUCER,
                    confidence=1.0, url="https://example.test/heady")


def _row(name="Heady Topper", style="Ale", prov=FILED, sensory_source=SensorySource.STYLE_PRIOR,
         abv=None, description=None, category=Category.BEER):
    return Product(
        id="p:1", brand_id="brand:1", producer_id="prod:1", category=category, name=name,
        style=Sourced[str](value=style, provenance=prov) if style else None,
        spec=ProductSpec(abv_pct=Sourced[float](value=abv, provenance=prov) if abv else None),
        sensory=(SensoryVector(source=sensory_source, confidence=0.35,
                               axes={"citrus": 0.65, "bitterness": 0.75})
                 if sensory_source else None),
        description=Sourced[str](value=description, provenance=STATED) if description else None,
    ).model_dump(mode="json")


def _answer(**over):
    axes = {a: 0.0 for a in SENSORY_AXES}
    axes.update({"tropical": 0.8, "citrus": 0.7, "piney_resinous": 0.6, "bitterness": 0.6,
                 "body_fullness": 0.55, "carbonation": 0.5, "alcohol_warmth": 0.45,
                 "dryness_finish": 0.4})
    base = {
        "recognition": "known", "confidence": 0.85, "style": "New England IPA", "abv_pct": 8.0,
        "summary": "Tropical and piney, soft and deceptively strong.",
        "descriptors": ["Tropical", "pine", "grapefruit", "tropical"],
        "axes": axes,
        "ingredients": [{"name": "Simcoe", "kind": "hop", "role": "dry_hop"},
                        {"name": "simcoe", "kind": "hop", "role": "aroma_hop"},
                        {"name": "Apollo", "kind": "hop", "role": "bittering_hop"},
                        {"name": "Conan yeast", "kind": "yeast", "role": "yeast"}],
        "basis": "the brewery's own description and its published hops",
    }
    base.update(over)
    return base


# ---- the answer ---------------------------------------------------------------------------------

def test_the_answer_is_tidied_not_trusted():
    p = ProductProfile.model_validate(_answer(axes={"citrus": 1.7, "sweet": -1, "bogus": 1},
                                              abv_pct=140, recognition="Known"))
    assert p.axes["citrus"] == 1.0 and p.axes["sweet"] == 0.0 and "bogus" not in p.axes
    assert set(p.axes) == set(SENSORY_AXES)
    assert p.abv_pct is None  # not a strength anything is bottled at
    assert p.recognition == "known"
    assert p.descriptors == ["tropical", "pine", "grapefruit"]  # lower-cased, no repeats
    assert [i.name for i in p.ingredients] == ["Simcoe", "Apollo", "Conan yeast"]


def test_an_unknown_kind_or_role_falls_back_to_other():
    p = ProductProfile.model_validate(_answer(
        ingredients=[{"name": "Juniper", "kind": "botanical", "role": "infusion"}]))
    assert p.ingredients[0].kind.value == "other" and p.ingredients[0].role.value == "other"


def test_style_only_confidence_is_held_to_a_style_priors_worth():
    known = ProductProfile.model_validate(_answer(confidence=0.9))
    guess = ProductProfile.model_validate(_answer(recognition="style_only", confidence=0.9))
    assert known.sensory_confidence == 0.9
    assert guess.sensory_confidence == 0.4
    assert ProductProfile.model_validate(_answer(recognition="unknown")).usable is False


# ---- the write ----------------------------------------------------------------------------------

def test_a_known_product_gets_its_own_profile_style_ingredients_and_strength():
    rec = _row()
    changed = apply_profile(rec, ProductProfile.model_validate(_answer()), model="m")
    assert set(changed) == {"sensory", "style", "description", "ingredients+3", "abv"}
    p = Product.model_validate(rec)
    assert p.sensory.source == SensorySource.LLM_PROFILE and p.sensory.confidence == 0.85
    assert p.sensory.axes["tropical"] == 0.8 and "sweet" not in p.sensory.axes
    assert p.style.value == "New England IPA"
    assert p.style.provenance.method == ExtractionMethod.LLM_RECALLED
    assert p.style.provenance.confidence == 0.6  # the ceiling, whatever the model thinks
    assert p.style.provenance.quote == "was: Ale"
    assert p.spec.abv_pct.value == 8.0
    assert [i.raw_name for i in p.recipe.ingredients] == ["Simcoe", "Apollo", "Conan yeast"]
    assert p.recipe.ingredients[0].role.value == "dry_hop"
    assert p.description.value.startswith("Tropical")
    assert p.description.provenance.quote == "tropical, pine, grapefruit"
    assert p.description.provenance.extractor_version == "m/" + \
        __import__("bcd_enrich.profile", fromlist=["PROMPT_VERSION"]).PROMPT_VERSION


def test_a_style_only_answer_states_no_ingredient_and_no_strength():
    rec = _row(style=None)
    changed = apply_profile(rec, ProductProfile.model_validate(
        _answer(recognition="style_only", confidence=0.4)), model="m")
    assert "abv" not in changed and not any(c.startswith("ingredients") for c in changed)
    p = Product.model_validate(rec)
    assert p.sensory.confidence == 0.4
    assert p.spec.abv_pct is None and p.recipe.ingredients == []
    # A style guessed from the name at 0.4 is not worth writing either.
    assert p.style is None
    rec = _row(style=None)
    apply_profile(rec, ProductProfile.model_validate(
        _answer(recognition="style_only", confidence=0.55)), model="m")
    assert Product.model_validate(rec).style.value == "New England IPA"


def test_a_specific_style_on_record_is_kept():
    rec = _row(style="Gold rum", category=Category.SPIRIT)
    changed = apply_profile(rec, ProductProfile.model_validate(_answer(style="Dark rum")),
                            model="m")
    assert "style" not in changed
    assert Product.model_validate(rec).style.value == "Gold rum"


def test_what_drinkers_said_is_never_overwritten():
    rec = _row(sensory_source=SensorySource.REVIEW_CONSENSUS)
    changed = apply_profile(rec, ProductProfile.model_validate(_answer()), model="m")
    assert "sensory" not in changed
    assert Product.model_validate(rec).sensory.source == SensorySource.REVIEW_CONSENSUS


def test_a_chemistry_prior_yields_only_to_a_known_product():
    rec = _row(sensory_source=SensorySource.CHEMISTRY_PRIOR)
    apply_profile(rec, ProductProfile.model_validate(_answer(recognition="style_only")),
                  model="m")
    assert Product.model_validate(rec).sensory.source == SensorySource.CHEMISTRY_PRIOR
    apply_profile(rec, ProductProfile.model_validate(_answer()), model="m")
    assert Product.model_validate(rec).sensory.source == SensorySource.LLM_PROFILE


def test_the_makers_own_words_and_strength_stay():
    rec = _row(abv=8.0, description="Drink from the can.", prov=STATED)
    rec["recipe"] = {"ingredients": [{
        "role": "dry_hop", "entity_kind": "hop", "entity_ref": None, "raw_name": "simcoe",
        "quantity": None, "unit": None, "percent_of_bill": None, "timing": None,
        "provenance": STATED.model_dump(mode="json")}], "process_steps": []}
    changed = apply_profile(rec, ProductProfile.model_validate(_answer(abv_pct=7.5)), model="m")
    assert "abv" not in changed and "description" not in changed
    p = Product.model_validate(rec)
    assert p.spec.abv_pct.value == 8.0 and p.description.value == "Drink from the can."
    # The hop the maker already lists is not listed twice, whatever its capitalisation.
    assert [i.raw_name for i in p.recipe.ingredients] == ["simcoe", "Apollo", "Conan yeast"]


def test_a_strength_needs_the_model_to_be_sure():
    rec = _row()
    changed = apply_profile(rec, ProductProfile.model_validate(_answer(confidence=0.6)),
                            model="m")
    assert "abv" not in changed and "sensory" in changed


def test_applying_twice_changes_nothing_the_second_time():
    rec = _row()
    profile = ProductProfile.model_validate(_answer())
    assert apply_profile(rec, profile, model="m")
    assert apply_profile(rec, profile, model="m") == []


# ---- the question -------------------------------------------------------------------------------

def test_the_question_carries_what_the_catalog_knows():
    rec = _row(name="Black Seal", style="Gold rum", category=Category.SPIRIT, abv=40.0)
    q = describe(rec, {"name": "Goslings", "city": "Hamilton", "country": "BM",
                       "kind": "distillery"}, {"name": "Goslings"})
    assert "Product: Black Seal" in q
    assert "Producer: Goslings (Hamilton, BM) -- distillery" in q
    assert "Filed style: Gold rum (registry class: Other Rum Gold Usb)" in q
    assert "ABV on record: 40.0%" in q
    assert "Brand:" not in q  # the same name as the producer says nothing


def test_the_request_forces_the_tool_and_caches_the_rubric():
    body = request_params("Product: X", "claude-test")
    assert body["tool_choice"] == {"type": "tool", "name": "record_profile"}
    assert body["tools"] == [PROFILE_TOOL]
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"] == [{"role": "user", "content": "Product: X"}]
    # The rubric names every axis, in the schema's order, and shows the worked examples.
    text = system_prompt()
    assert all(f"- {a}:" in text for a in SENSORY_AXES)
    assert '"recognition": "style_only"' in text
    required = PROFILE_TOOL["input_schema"]["properties"]["axes"]["required"]
    assert tuple(required) == SENSORY_AXES


def test_the_tool_call_is_read_out_of_the_message():
    msg = {"content": [{"type": "text", "text": "Sure."},
                       {"type": "tool_use", "name": "record_profile", "input": _answer()}]}
    assert tool_input(msg)["style"] == "New England IPA"
    with pytest.raises(ValueError, match="no record_profile"):
        tool_input({"content": [{"type": "text", "text": "I cannot."}], "stop_reason": "end_turn"})


def test_the_client_retries_an_overloaded_api_and_returns_the_profile(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        assert request.headers["x-api-key"] == "sk-test"
        if len(calls) == 1:
            return httpx.Response(529, json={"type": "error",
                                             "error": {"type": "overloaded_error"}})
        return httpx.Response(200, json={
            "content": [{"type": "tool_use", "name": "record_profile", "input": _answer()}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 900, "output_tokens": 300},
        })

    # No waiting between attempts in a test.
    monkeypatch.setattr(Claude._post.retry, "wait", lambda *a, **k: 0)
    client = Claude("sk-test", "claude-test", client=httpx.Client(
        transport=httpx.MockTransport(handler)))
    answer, usage = client.profile("Product: Heady Topper")
    assert answer["recognition"] == "known" and usage["input_tokens"] == 900
    assert len(calls) == 2 and calls[1]["model"] == "claude-test"


def test_a_batch_is_one_submission_with_a_custom_id_per_product():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "msgbatch_1", "processing_status": "in_progress"})

    client = Claude("sk-test", "claude-test",
                    client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = client.submit_batch([("p00000-a", "Product: A"), ("p00001-b", "Product: B")])
    assert out["id"] == "msgbatch_1"
    assert [r["custom_id"] for r in seen[0]["requests"]] == ["p00000-a", "p00001-b"]
    assert seen[0]["requests"][1]["params"]["messages"][0]["content"] == "Product: B"


# ---- the selection ------------------------------------------------------------------------------

def test_the_scan_logs_select_what_the_camera_drew(tmp_path):
    store = MedallionStore(root=tempfile.mkdtemp())
    store.put_gold("prod:1", "producer", {"id": "prod:1", "name": "The Alchemist"})
    store.put_gold("prod:2", "producer", {"id": "prod:2", "name": "Some Other Brewery"})
    for pid, maker in (("p:heady", "prod:1"), ("p:heady2", "prod:2")):
        rec = _row()
        rec.update({"id": pid, "producer_id": maker})
        store.put_gold(pid, "product", rec)
    store.put_gold("p:sib", "product", {**_row(name="Focal Banger"), "id": "p:sib",
                                        "producer_id": "prod:1"})
    log = tmp_path / "scans1.jsonl"
    log.write_text(json.dumps({"candidates": [
        {"name": "Heady Topper", "producer": "The Alchemist", "score": 1.0}]}) + "\n"
        + json.dumps({"candidates": [{"name": "Nope", "producer": "", "score": 0.5}]}) + "\n")
    rows = select_products(store, ids=[], searches=[], from_scans=True,
                           scan_glob=str(tmp_path / "scans*.jsonl"), lineup=0, limit=None)
    assert [r["id"] for r in rows] == ["p:heady"]  # the maker's row, not the same-named one
    rows = select_products(store, ids=[], searches=[], from_scans=True,
                           scan_glob=str(tmp_path / "scans*.jsonl"), lineup=8, limit=None)
    assert {r["id"] for r in rows} == {"p:heady", "p:sib"}
    store.close()


# ---- answers from a file ------------------------------------------------------------------------

def test_answers_from_a_file_land_like_the_apis(tmp_path, capsys, monkeypatch):
    from bcd_enrich.profile import main

    # The CLI reads the repo's `.env`, which points at the live catalog and would leak the
    # rest of that file into every later test; this test's store is SQLite.
    monkeypatch.setattr("bcd_enrich.profile.load_dotenv", lambda: None)
    monkeypatch.setenv("BCD_STORE_BACKEND", "sqlite")
    store_root = tempfile.mkdtemp()
    store = MedallionStore(root=store_root)
    store.put_gold("prod:1", "producer", {"id": "prod:1", "name": "The Alchemist"})
    store.put_gold("p:1", "product", _row())
    store.close()
    questions = tmp_path / "q.jsonl"
    answers = tmp_path / "a.jsonl"

    assert main(["--root", store_root, "--ids", "p:1", "--export", str(questions)]) == 0
    q = json.loads(questions.read_text().splitlines()[0])
    assert q["id"] == "p:1" and q["question"].startswith("Product: Heady Topper")

    answers.write_text(json.dumps({"id": "p:1", "answer": _answer()}) + "\n"
                       + json.dumps({"id": "p:missing", "answer": _answer()}) + "\n"
                       + "not json\n")
    assert main(["--root", store_root, "--answers", str(answers), "--export", str(questions),
                 "--model", "claude-opus-5", "--via", "session"]) == 0
    out = capsys.readouterr().out
    assert "[K 0.85] 'Heady Topper': New England IPA" in out
    assert "no product 'p:missing'" in out and "line 3" in out
    assert "wrote 1 products" in out

    store = MedallionStore(root=store_root)
    p = Product.model_validate(store.get_gold("p:1"))
    assert p.sensory.source == SensorySource.LLM_PROFILE
    assert p.style.provenance.extractor_version.startswith("claude-opus-5/")
    docs = list(store.iter_bronze("llm-profile"))
    assert len(docs) == 1 and docs[0].natural_key == "p:1"
    assert docs[0].payload["via"] == "session"
    assert docs[0].payload["question"].startswith("Product: Heady Topper")
    # Asked again, the product counts as answered and no question goes out.
    assert main(["--root", store_root, "--ids", "p:1", "--export", str(questions)]) == 0
    assert questions.read_text() == ""
    store.close()
