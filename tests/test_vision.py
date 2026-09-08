"""The vision scan path — naming a label from the picture instead of from garbled OCR.

The rule this file exists to hold: the model supplies a *name*, never a fact. Every answer
still has to be a catalog row that accounts for what the model read, so a model that
invents a beer produces no answer rather than a confident wrong one.
"""

from __future__ import annotations

import asyncio
import base64
import tempfile

import httpx
import pytest
import respx
from bcd_api.app import _state, app as fastapi_app, scan_vision
from bcd_api.resolver import Resolver
from bcd_api.vision import (
    MAX_IMAGE_BYTES,
    AnthropicVision,
    OllamaVision,
    Sighting,
    StubVision,
    parse_sightings,
    provider_from_env,
)
from bcd_ingest.store import MedallionStore
from bcd_schema import (
    Brand,
    Category,
    DetectedText,
    ExtractionMethod,
    Producer,
    Product,
    ProductSpec,
    Provenance,
    ScanVisionRequest,
    Sourced,
)

PIXEL = base64.b64encode(b"\xff\xd8\xff\xe0 not really a jpeg, but bytes are bytes").decode()


@pytest.fixture()
def store():
    d = tempfile.mkdtemp()
    s = MedallionStore(root=d)
    prov = Provenance(source_id="ttb", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)
    s.put_gold("prod:alc", "producer",
               Producer(id="prod:alc", name="The Alchemist").model_dump(mode="json"))
    s.put_gold("brand:alc", "brand",
               Brand(id="brand:alc", producer_id="prod:alc",
                     name="The Alchemist").model_dump(mode="json"))
    for pid, name in (("ttb:1", "Heady Topper"), ("ttb:2", "Banger")):
        s.put_gold(pid, "product", Product(
            id=pid, brand_id="brand:alc", producer_id="prod:alc", category=Category.BEER,
            name=name,
            spec=ProductSpec(abv_pct=Sourced[float](value=8.0, provenance=prov)),
        ).model_dump(mode="json"))
    yield s
    s.close()


@pytest.fixture()
def api(store, monkeypatch):
    """The endpoint's module state, without booting a server. `scan_vision` is a plain async
    function over `_state`, so the seam under test is the logic, not FastAPI's routing."""
    monkeypatch.setitem(_state, "store", store)
    monkeypatch.setitem(_state, "resolver", Resolver(store))
    monkeypatch.setitem(_state, "profiles", {})
    return _state


def call(req: ScanVisionRequest):
    return asyncio.run(scan_vision(req))


# --- reading the model's reply -------------------------------------------------------

def test_a_plain_array_parses():
    got = parse_sightings('[{"name": "Heady Topper", "box": [0.1, 0.2, 0.3, 0.4]}]')
    assert got == [Sighting(name="Heady Topper", box=(0.1, 0.2, 0.3, 0.4))]


def test_a_fenced_reply_with_preamble_parses():
    # Asking for "a JSON array and nothing else" gets one most of the time.
    got = parse_sightings('Here you go:\n```json\n[{"name": "Focal Banger"}]\n```')
    assert [s.name for s in got] == ["Focal Banger"]


def test_the_same_beer_twice_is_one_sighting():
    # A four-pack is one product photographed four times, not four answers.
    got = parse_sightings('[{"name": "Heady Topper"}, {"name": "heady  topper"}]')
    assert len(got) == 1


def test_no_label_read_parses_to_nothing():
    assert parse_sightings("[]") == []
    assert parse_sightings("I can't make out any labels.") == []
    assert parse_sightings("") == []


def test_a_box_outside_the_frame_is_dropped_but_the_name_is_kept():
    # The box is the model's estimate and a wrong one puts the overlay on the wrong can.
    # Losing it costs a position; trusting it costs an answer.
    got = parse_sightings('[{"name": "Heady Topper", "box": [0.5, 0.5, 0.9, 0.9]}]')
    assert got == [Sighting(name="Heady Topper", box=None)]
    assert parse_sightings('[{"name": "X Ale", "box": "middle"}]')[0].box is None
    assert parse_sightings('[{"name": "X Ale", "box": [0.1, 0.1, 0, 0.2]}]')[0].box is None


# --- the provider seam ---------------------------------------------------------------

def test_no_key_means_no_provider(monkeypatch):
    # Unconfigured is a state, not an error: the text path predates this and must keep working.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert provider_from_env() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert provider_from_env() is not None


@respx.mock
def test_the_image_is_sent_and_the_names_come_back():
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "content": [{"type": "text", "text": '[{"name": "Heady Topper"}]'}]}))
    got = asyncio.run(AnthropicVision("sk-test", model="m").identify(b"\xff\xd8jpegbytes"))
    assert [s.name for s in got] == ["Heady Topper"]
    sent = route.calls.last.request
    assert sent.headers["x-api-key"] == "sk-test"
    body = httpx.Response(200, content=sent.content).json()
    block = body["messages"][0]["content"][0]
    assert block["type"] == "image"
    assert base64.b64decode(block["source"]["data"]) == b"\xff\xd8jpegbytes"


@respx.mock
def test_a_local_model_answers_the_same_contract():
    # Same seam, no key and no per-call cost. Ollama wraps the reply in {"message": {...}} and
    # is asked for {"labels": [...]} — a schema keeps a small model from narrating around it.
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {
            "content": '{"labels": [{"name": "The Alchemist Heady Topper"}]}'}}))
    got = asyncio.run(OllamaVision(model="m").identify(b"\xff\xd8jpegbytes"))
    assert [s.name for s in got] == ["The Alchemist Heady Topper"]
    sent = httpx.Response(200, content=route.calls.last.request.content).json()
    assert base64.b64decode(sent["messages"][0]["images"][0]) == b"\xff\xd8jpegbytes"
    assert sent["stream"] is False and sent["options"]["temperature"] == 0


def test_the_provider_is_chosen_explicitly_not_guessed(monkeypatch):
    # Nothing probes localhost: a wrong guess fails as silence, and "connection refused" in
    # `detail` is worth more than a provider that quietly elected itself.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("BCD_VISION_PROVIDER", "ollama")
    assert provider_from_env().label.startswith("ollama:")
    monkeypatch.setenv("BCD_VISION_PROVIDER", "anthropic")
    assert provider_from_env() is None          # named, but no key to name it with
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert provider_from_env().label.startswith("anthropic:")


# --- the endpoint --------------------------------------------------------------------

def test_without_a_key_the_endpoint_says_so_rather_than_failing(api, monkeypatch):
    monkeypatch.setitem(_state, "vision", None)
    resp = call(ScanVisionRequest(image_b64=PIXEL))
    assert resp.candidates == []
    assert "ANTHROPIC_API_KEY" in resp.detail


def test_a_broken_upload_is_reported_not_raised(api, monkeypatch):
    monkeypatch.setitem(_state, "vision", StubVision([Sighting("Heady Topper")]))
    assert "base64" in call(ScanVisionRequest(image_b64="not base64!!")).detail
    assert "empty" in call(ScanVisionRequest(image_b64="")).detail


def test_a_provider_that_times_out_degrades_to_no_answers(api, monkeypatch):
    class Timeout(StubVision):
        async def identify(self, image, media_type="image/jpeg"):
            raise httpx.ReadTimeout("too slow")

    monkeypatch.setitem(_state, "vision", Timeout())
    resp = call(ScanVisionRequest(image_b64=PIXEL))
    # The camera's hot path must not surface a 500 because a network call was slow.
    assert resp.candidates == [] and "ReadTimeout" in resp.detail


def test_a_label_the_model_reads_becomes_a_catalog_answer(api, monkeypatch):
    monkeypatch.setitem(_state, "vision",
                        StubVision([Sighting("The Alchemist Heady Topper", box=(0.2, 0.3, 0.4, 0.5))]))
    resp = call(ScanVisionRequest(image_b64=PIXEL))
    assert [c.resolved.product.name for c in resp.candidates] == ["Heady Topper"]
    assert resp.corroborated
    assert resp.sightings == ["The Alchemist Heady Topper"]
    # The overlay anchors on the frame the *server* built, so it has to come back with it.
    assert resp.detections[0].x == 0.2 and resp.detections[0].w == 0.4


def test_a_beer_the_catalog_does_not_have_is_reported_read_but_unresolved(api, monkeypatch):
    # "The model saw nothing" and "the catalog has nothing" are different problems, and
    # candidates alone cannot tell them apart.
    monkeypatch.setitem(_state, "vision", StubVision([Sighting("Pliny The Elder")]))
    resp = call(ScanVisionRequest(image_b64=PIXEL))
    assert resp.candidates == []
    assert resp.sightings == ["Pliny The Elder"] and resp.unresolved_indices == [0]
    assert not resp.corroborated


def test_a_row_that_is_only_a_piece_of_the_name_is_not_the_answer(api, monkeypatch):
    # The catalog holds a product literally named "Banger". Trigram matching is happy to
    # return it for "Focal Banger" -- the same failure as a row named "Mist" answering
    # "ACHE MIST-VERM". The model read a clean name, so the row has to be that whole name.
    monkeypatch.setitem(_state, "vision", StubVision([Sighting("Focal Banger")]))
    resp = call(ScanVisionRequest(image_b64=PIXEL))
    assert resp.candidates == []
    assert resp.unresolved_indices == [0]


def test_the_camera_text_is_recorded_not_matched(api, monkeypatch):
    # OCR fragments ride along for the scan log — they are what the camera saw at the moment
    # the picture was taken. They are not matched: a clean reading needs no corroboration from
    # a garbled one, and answering them here would return /v1/scan/resolve's job twice.
    monkeypatch.setitem(_state, "vision", StubVision([Sighting("Heady Topper")]))
    resp = call(ScanVisionRequest(
        image_b64=PIXEL,
        detections=[DetectedText(text="THE ALCHEMIST"), DetectedText(text="FADY TOPPE")]))
    assert len(resp.candidates) == 1
    assert all(c.detection_index == 0 for c in resp.candidates)


def test_two_sightings_of_one_beer_are_one_overlay(api, monkeypatch):
    monkeypatch.setitem(_state, "vision",
                        StubVision([Sighting("Heady Topper", box=(0.0, 0.0, 0.3, 0.9)),
                                    Sighting("Heady Topper Ale", box=(0.4, 0.0, 0.3, 0.9))]))
    resp = call(ScanVisionRequest(image_b64=PIXEL))
    assert len(resp.candidates) == 1


def test_an_oversized_frame_is_refused_before_the_model_is_called(api, monkeypatch):
    stub = StubVision([Sighting("Heady Topper")])
    monkeypatch.setitem(_state, "vision", stub)
    huge = base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode()
    resp = call(ScanVisionRequest(image_b64=huge))
    assert "over the" in resp.detail and stub.calls == []


def test_the_route_is_registered():
    paths = {getattr(r, "path", "") for r in fastapi_app.routes}
    assert "/v1/scan/vision" in paths
