"""Promote strengths drinkers told us about drinks the catalog already has.

`POST /v1/contribute/fact` files a typed strength in bronze and stops there, which is right and
is also a dead end until something promotes it. These are the gates, and they are deliberately
the ones a contributed DRINK is promoted on (`promote_contributions.py`), because the question
is the same: has this been told to us by enough people, independently, to be written down?

  1. two different ACCOUNTS must have told us the same thing;
  2. they must actually agree -- two people reading one tin should read the same number;
  3. the number must be possible for what is in the bottle;
  4. the row's own name must not already state a different one;
  5. the slot must be one a 0.7-confidence fact may fill -- empty, or holding something the
     ladder ranks below `user_contributed`. A filing or an open database outranks a drinker and
     is never overwritten here.

Gates 3 and 4 are imported from `fill_abv_from_open_data`, not re-derived: each was earned from
a row that was actually wrong (a 0.0% wit being "corrected" to 3.5%, a Carlsberg whose own name
said 5.0 being moved to 4.2), and a second copy of them would drift from the first.

    .venv/bin/python scripts/promote_facts.py            # dry run: writes nothing
    .venv/bin/python scripts/promote_facts.py --verbose  # also every held group and why
    .venv/bin/python scripts/promote_facts.py --apply    # write the ones that earned it

Then restart the API: a written strength is read from gold on the next request, but the label
index is built at startup and a row's spec travels with it.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest", "services/enrich", "scripts"):
    sys.path.insert(0, str(ROOT / p))

from bcd_api.app import USER_FACT_SOURCE, _load_dotenv  # noqa: E402
from bcd_ingest.store import open_store  # noqa: E402
from bcd_schema import Product  # noqa: E402
from bcd_schema.provenance import (  # noqa: E402
    METHOD_CONFIDENCE_CEILING,
    ExtractionMethod,
    Provenance,
    Sourced,
)
from fill_abv_from_open_data import plausible, states_a_strength  # noqa: E402

VERSION = "promote_facts/1"

#: How many separate accounts have to say it. Two, the same bar a contributed drink clears:
#: one account is a claim, two independent accounts is the smallest thing that is evidence.
ACCOUNTS_NEEDED = 2

#: How far apart two readings of one label may be. Tight on purpose -- both are reading the
#: same printed number, so this is slack for a rounding difference (4.5 against 4.6), not for
#: a disagreement. Anything wider is two people talking about two different drinks, or one of
#: them guessing, and the group is held for a look rather than averaged into a number nobody
#: said.
AGREEMENT = 0.2

#: What a drinker's word may be written over. Derived from the ladder rather than listed, so
#: it cannot drift when a method is added: anything the schema ranks BELOW `user_contributed`.
#: Today that is the style-prior guess, an LLM recall, a community clone and review consensus;
#: an open database (0.75), a retailer listing, a filing and a label all outrank it.
_CEILING = METHOD_CONFIDENCE_CEILING[ExtractionMethod.USER_CONTRIBUTED]
_REPLACEABLE = {m.value for m, c in METHOD_CONFIDENCE_CEILING.items() if c < _CEILING}


@dataclass
class Claim:
    account: str
    value: float
    received_at: str


@dataclass
class Promotion:
    pid: str
    name: str
    category: str
    value: float
    accounts: int
    was: str


def _fillable(abv: dict | None) -> str | None:
    """What the row's strength is now, when a drinker's word may replace it; None when it may
    not. The same shape `fill_abv_from_open_data._fillable` has, against a different ceiling."""
    if not isinstance(abv, dict) or abv.get("value") is None:
        return "empty"
    method = str((abv.get("provenance") or {}).get("method"))
    return method if method in _REPLACEABLE else None


def claims_by_row(store) -> dict[tuple[str, str], list[Claim]]:
    """Every fact in bronze, grouped by the row and field it is about.

    One account's latest word only. The phone replaces its own answer rather than queueing
    beside it, but the server keys documents by `account::client_id`, so an account that sent
    two client ids has two documents -- and two numbers from one person is one person who
    changed their mind, not two people agreeing.
    """
    latest: dict[tuple[str, str, str], Claim] = {}
    for doc in store.iter_bronze(USER_FACT_SOURCE):
        p = doc.payload or {}
        pid, field = p.get("product_id"), p.get("field")
        account, value = p.get("account_id"), p.get("value")
        if not (pid and field and account) or not isinstance(value, (int, float)):
            continue
        key = (str(pid), str(field), str(account))
        claim = Claim(str(account), float(value), str(p.get("received_at") or ""))
        if key not in latest or claim.received_at > latest[key].received_at:
            latest[key] = claim

    out: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    for (pid, field, _), claim in latest.items():
        out[(pid, field)].append(claim)
    return out


def plan(store, groups: dict[tuple[str, str], list[Claim]],
         ) -> tuple[list[Promotion], dict[str, list[str]]]:
    promote: list[Promotion] = []
    held: dict[str, list[str]] = defaultdict(list)

    for (pid, field), claims in sorted(groups.items()):
        if field != "abv_pct":
            held["a field this pass does not write"].append(f"{pid} {field}")
            continue
        if len({c.account for c in claims}) < ACCOUNTS_NEEDED:
            held[f"fewer than {ACCOUNTS_NEEDED} accounts"].append(
                f"{pid} {claims[0].value}% from 1 account")
            continue

        values = sorted(c.value for c in claims)
        if values[-1] - values[0] > AGREEMENT:
            held["the accounts disagree"].append(
                f"{pid} {', '.join(f'{v}%' for v in values)}")
            continue
        value = round(sum(values) / len(values), 1)

        row = store.get_gold(pid)
        if row is None:
            held["no such row any more"].append(pid)
            continue
        product = Product.model_validate(row)
        name = product.name or ""
        category = product.category.value if hasattr(product.category, "value") \
            else str(product.category)

        if not plausible(value, category, name):
            held["impossible for the category"].append(f"{name} ({category}) {value}%")
            continue

        stated = states_a_strength(name)
        if stated and not any(abs(value - s) <= AGREEMENT for s in stated):
            held["the row's own name says otherwise"].append(
                f"{name} says {stated}, told {value}%")
            continue

        was = _fillable((row.get("spec") or {}).get("abv_pct"))
        if was is None:
            held["already holds something that outranks a drinker"].append(
                f"{name} {value}%")
            continue

        promote.append(Promotion(pid=pid, name=name, category=category, value=value,
                                 accounts=len({c.account for c in claims}), was=was))
    return promote, held


def write(store, promotions: list[Promotion]) -> int:
    now = datetime.now(UTC).isoformat()
    written = 0
    for p in promotions:
        row = store.get_gold(p.pid)
        if row is None:
            continue
        product = Product.model_validate(row)
        product.spec.abv_pct = Sourced[float](
            value=p.value,
            provenance=Provenance(
                source_id=USER_FACT_SOURCE,
                url=None,
                quote=f"{p.accounts} accounts told us {p.value}%",
                method=ExtractionMethod.USER_CONTRIBUTED,
                confidence=_CEILING,
                extracted_at=now,
                extractor_version=VERSION,
            ),
        )
        store.put_gold(p.pid, "product", product.model_dump(mode="json"))
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write them (default: dry run)")
    ap.add_argument("--verbose", action="store_true", help="print every held group too")
    args = ap.parse_args()

    _load_dotenv()
    store = open_store()
    groups = claims_by_row(store)
    print(f"{len(groups)} (row, field) groups from bronze")

    promotions, held = plan(store, groups)
    for reason, rows in sorted(held.items()):
        print(f"  held, {reason}: {len(rows)}")
        if args.verbose:
            for r in rows[:20]:
                print(f"      {r}")

    print(f"\n{len(promotions)} ready to write:")
    for p in promotions:
        print(f"  {p.value:>5.1f}%  {p.name[:48]:48}  ({p.category}, was {p.was}, "
              f"{p.accounts} accounts)")

    if not args.apply:
        print("\nDry run. Nothing written. Add --apply to write them.")
        return 0

    written = write(store, promotions)
    print(f"\nwrote {written} strengths. Restart the API to pick them up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
