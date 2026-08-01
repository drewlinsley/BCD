# 06 — Legal & compliance

Not legal advice — an engineering brief on the constraints that shaped the build. Get counsel before launch.

## Crawl posture (the chosen risk)

The decision on record: **crawl everything publicly reachable, robots.txt-gated** — richer data now, in exchange for real exposure on sites whose ToS forbid scraping (Untappd, BeerAdvocate, RateBeer, Vivino). We mitigate **in code**, not in a doc:

- **robots.txt parsed and honored per host** (Protego) — [policy.py](../packages/crawler/bcd_crawler/policy.py)
- **per-host rate caps** + honoring `Crawl-delay`
- **identified User-Agent** with a contact URL (no spoofing)
- **per-source kill switch** and a **global crawl-cost guardrail**
- an **append-only fetch evidence log** — URL, timestamp, robots decision, status ([evidence.py](../packages/crawler/bcd_crawler/evidence.py)) — the paper trail per URL

Two structural safeguards:

1. **The canonical catalog is buildable from Tier A/B (open/gov/academic) alone**, so a Tier-D takedown degrades quality without killing the product.
2. **ToS-restricted sites are catalogued as `status: blocked` and never crawled** — their historical review data is ingested *legally* via the SNAP academic release instead.

> ⚠️ **This repo is public.** A published target list for ToS-restricted sites is discoverable by those sites. Before crawling Tier-D at scale, consider moving Tier-D registry entries to a **private submodule**. Flagged in the roadmap.

Relevant law to review with counsel: CFAA (post-*hiQ v. LinkedIn*, scraping public data is not per se a violation, but the picture is unsettled and fact-specific), breach-of-contract/ToS claims, and copyright on review text (we store facts + short quotes with attribution, not corpora of expression).

## Alcohol-specific app rules

- **Age gate + 17+ rating.** App Store rejects apps that "encourage consumption… or encourage minors." We ship a 17+ rating and an age check. A 17+ rating hides the app under parental controls — an accepted cost.
- **No autonomous purchase.** Alcohol sales legally require a **licensed retailer with age verification at delivery**; an agent cannot be the buyer of record. Hence the staged agent design ([06 below](#agent-tiers)).
- **Delivery law varies by state** (control states, dry counties, shipping bans). The app surfaces retailers; it is not itself the seller.

## Agent tiers (staged by legal risk)

| Tier | Action | Status |
|---|---|---|
| **1** | sentinel fires → push → **deep link into the retailer's own checkout**, or add to wantlist | v1 — safe, ships now |
| **2** | reserve-by-form where the retailer offers an official mechanism | post-v1 |
| **3** | "call the store" — agent composes the call **script**, user taps to approve & dial | post-v1 |

Tier 3 never places the order autonomously. Two-party call-recording consent laws vary by state; the agent drafts, the human acts.

## Monetization constraints (designed for now)

- **Alcohol advertising** is bound by self-regulatory codes (Beer Institute / DISCUS ~**73.6% adult-audience** composition). Ad targeting must be able to **prove audience composition** — which requires the age/consent architecture.
- **Selling purchase-habit data** is regulated under CPRA and needs a **working opt-out** — the `data_sharing` consent tier and export/delete endpoints in [05-telemetry.md](05-telemetry.md).

Both revenue paths are consent-architecture requirements, which is why telemetry/privacy is built before revenue.

## Credentials hygiene

`.env` is gitignored; only `.env.example` (no values) is committed. GitHub secret-scanning watches public repos. **Any key that touches history must be rotated** — see the roadmap's launch checklist.
