#!/usr/bin/env python3
"""Seed data/registry/sources/*.yaml from the curated master list below.

This script IS the human-editable master inventory; the per-source YAML files are the
machine-readable artifact the pipeline and validator consume. Re-run after editing to
regenerate. Every entry must carry a license and a legal_basis — that is the whole point.

Fields default sensibly so each row stays readable; see data/registry/schema.json.
"""

from __future__ import annotations

import os

import yaml

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "registry", "sources")


def S(id, name, tier, categories, provides, method, auth, license, legal_basis,
      *, jurisdiction=None, home=None, data=None, docs=None, robots_allows=None,
      rate_limit_rps=None, refresh="static", est_records=None, priority=3,
      status="planned", sentinel=False, notes=None):
    entry = {
        "id": id, "name": name, "tier": tier, "categories": categories,
        "access": {"method": method, "auth": auth, "robots_allows": robots_allows,
                   "rate_limit_rps": rate_limit_rps},
        "license": license, "legal_basis": legal_basis, "provides": provides,
        "refresh": refresh, "est_records": est_records, "priority": priority,
        "status": status,
    }
    if jurisdiction:
        entry["jurisdiction"] = jurisdiction
    urls = {k: v for k, v in (("home", home), ("data", data), ("docs", docs)) if v}
    if urls:
        entry["urls"] = urls
    if sentinel:
        entry["sentinel"] = True
    if notes:
        entry["notes"] = notes
    return entry


SOURCES = [
    # ============ TIER A — open / government / public domain ============
    S("ttb-cola-registry", "TTB Public COLA Registry", "A", ["beer", "spirits", "wine"],
      ["product", "producer", "sku", "abv", "label_image", "style"], "scrape", "none",
      "us-gov-public-domain", "public_domain", jurisdiction="US",
      home="https://www.ttb.gov/regulated-commodities/labeling/cola-public-registry",
      data="https://www.ttbonline.gov/colasonline/publicSearchColasBasic.do",
      refresh="daily", est_records=2600000, priority=1, status="active",
      notes="The US SKU universe: every label approval since 1999. Form-driven; scrape "
            "or license COLA Cloud. Highest-authority provenance (regulatory_filing)."),
    S("colacloud", "COLA Cloud", "A", ["beer", "spirits", "wine"],
      ["product", "producer", "sku", "upc", "abv", "label_image"], "api", "contract",
      "commercial-redistribution-of-public-data", "contract", jurisdiction="US",
      home="https://colacloud.us", docs="https://docs.colacloud.us",
      refresh="daily", est_records=2600000, priority=1,
      notes="TTB COLA pre-parsed: barcodes decoded, ABV OCR'd, 100+ fields, REST/"
            "Snowflake/MCP/bulk. Buys back ~2 months of scraping. Quote in week 1."),
    S("ttb-open-data", "TTB Open Data", "A", ["beer", "spirits", "wine"],
      ["producer", "ownership"], "bulk", "none", "us-gov-public-domain", "public_domain",
      jurisdiction="US", home="https://www.ttb.gov/data", refresh="monthly", priority=2,
      notes="Permittee lists + monthly brewery/DSP statistical reports. Producer registry "
            "and market sizing."),
    S("openbrewerydb", "Open Brewery DB", "A", ["beer", "venues"],
      ["producer", "venue"], "api", "none", "open-database-license", "open_license",
      home="https://www.openbrewerydb.org", data="https://api.openbrewerydb.org/v1/breweries",
      docs="https://www.openbrewerydb.org/documentation", robots_allows=True,
      refresh="weekly", est_records=8000, priority=1, status="active",
      notes="No auth, no rate limit. Producer + venue spine, day one. Connector implemented."),
    S("openfoodfacts", "Open Food Facts", "A", ["beer", "spirits", "ingredients"],
      ["product", "upc", "abv", "ingredients", "label_image"], "bulk", "none",
      "open-database-license", "open_license", home="https://world.openfoodfacts.org",
      data="https://world.openfoodfacts.org/data",
      docs="https://openfoodfacts.github.io/openfoodfacts-server/api/", robots_allows=True,
      rate_limit_rps=1.0, refresh="weekly", est_records=100000, priority=1, status="active",
      notes="The only large OPEN barcode->ingredient corpus. Serves scan-a-can directly. "
            "Connector implemented (v2 API)."),
    S("vinmonopolet", "Vinmonopolet API", "A", ["wine", "spirits", "beer"],
      ["product", "abv", "price", "style"], "api", "key", "official-api-terms",
      "open_license", jurisdiction="NO", home="https://www.vinmonopolet.no",
      docs="https://api.vinmonopolet.no", refresh="daily", priority=2,
      notes="Norwegian monopoly catalog, keyed. Clean structured product data incl. spirits."),
    S("systembolaget-mirror", "Systembolaget (community mirror)", "A",
      ["beer", "wine", "spirits"], ["product", "abv", "price", "style"], "bulk", "none",
      "community-mirror-open", "open_license", jurisdiction="SE",
      home="https://github.com/AlexGustafsson/systembolaget-api-data",
      refresh="daily", priority=3,
      notes="Official product API withdrawn; AlexGustafsson + C4illin mirrors are current."),
    S("alko", "Alko Product API", "A", ["beer", "wine", "spirits"],
      ["product", "abv", "price"], "bulk", "none", "community-mirror-open", "open_license",
      jurisdiction="FI", home="https://github.com/villeristi/alko-product-api",
      refresh="weekly", priority=4, notes="Finnish catalog + official price-list XLSX."),
    S("lcbo", "LCBO Catalog", "A", ["beer", "wine", "spirits"],
      ["product", "abv", "price", "sku"], "scrape", "none", "public_web", "public_web",
      jurisdiction="CA", home="https://www.lcbo.com", robots_allows=None, priority=3,
      notes="lcboapi.com is dead — scrape LCBO directly. Provincial pricing."),
    S("bcliquor", "BC Liquor Stores", "A", ["beer", "wine", "spirits"],
      ["product", "price", "inventory"], "scrape", "none", "public_web", "public_web",
      jurisdiction="CA", home="https://www.bcliquorstores.com", priority=4),
    S("saq", "SAQ (Quebec)", "A", ["wine", "spirits", "beer"],
      ["product", "price"], "scrape", "none", "public_web", "public_web",
      jurisdiction="CA", home="https://www.saq.com", priority=4),
    S("eu-wine-elabels", "EU Wine E-Labels (U-label et al.)", "A", ["wine"],
      ["ingredients", "abv", "process"], "scrape", "none", "public_web", "public_web",
      jurisdiction="EU", home="https://new.u-label.com", refresh="event", priority=2,
      notes="Legally MANDATED ingredient lists behind per-bottle QR (Reg 2021/2117; all "
            "2024+ harvest). Ground-truth ingredient data; a QR our scanner already reads."),
    S("google-patents", "Google Patents / USPTO", "A", ["beer", "spirits", "ingredients"],
      ["process", "ingredients"], "scrape", "none", "us-gov-public-domain", "public_domain",
      home="https://patents.google.com", refresh="monthly", priority=3,
      notes="Brewing/distilling PROCESS patents; hop plant patents (Summit PP18039, "
            "Columbus PP10956) with varietal chemistry. Legally-published proprietary detail."),
    S("bjcp-styles", "BJCP Style Guidelines", "A", ["beer", "mead", "cider"],
      ["style", "sensory"], "bulk", "none", "bjcp-terms-noncommercial", "open_license",
      home="https://www.bjcp.org/style", refresh="static", priority=2,
      notes="Structured styles w/ OG/FG/IBU/SRM/ABV ranges + sensory. The prior "
            "distribution for every inference."),
    S("brewers-association-styles", "Brewers Association Style Guidelines", "A", ["beer"],
      ["style", "sensory"], "bulk", "none", "ba-terms", "open_license",
      home="https://www.brewersassociation.org", refresh="static", priority=3),
    S("usda-hop-germplasm", "USDA ARS Hop Germplasm + NASS", "A", ["ingredients"],
      ["ingredients"], "bulk", "none", "us-gov-public-domain", "public_domain",
      jurisdiction="US", home="https://www.ars-grin.gov", refresh="static", priority=3,
      notes="Varietal lineage + acreage/production volume."),
    S("wikidata", "Wikidata / DBpedia", "A", ["all"],
      ["producer", "ownership"], "bulk", "none", "cc0", "open_license",
      home="https://www.wikidata.org", refresh="monthly", priority=2,
      notes="Brand/producer entities, aliases, ownership graphs. Entity-resolution backbone "
            "(who owns whom: ABI, Constellation, Diageo)."),
    S("openstreetmap", "OpenStreetMap + Overture", "A", ["venues"],
      ["venue"], "bulk", "none", "odbl", "open_license", home="https://www.openstreetmap.org",
      refresh="weekly", priority=2,
      notes="amenity=bar|pub, craft=brewery, shop=alcohol. Free venue spine w/ geo, no "
            "POI vendor bill."),
    S("cascade-hop-coa", "Hop COA aggregators (public)", "A", ["ingredients"],
      ["ingredients"], "scrape", "none", "public_web", "public_web",
      home="https://hopsconnect.com/resources/interpreting-coas", priority=4,
      notes="Public certificate-of-analysis pages: alpha/beta/oil by lot."),

    # ============ TIER B — academic / research releases ============
    S("snap-beeradvocate", "SNAP BeerAdvocate reviews", "B", ["beer"],
      ["review", "rating", "sensory"], "bulk", "none", "academic-research-only",
      "academic_release", home="https://snap.stanford.edu/data/web-BeerAdvocate.html",
      est_records=1500000, priority=1, refresh="static",
      notes="1.5M reviews, 264K beers, 33K users, 5 aspect ratings (appearance/aroma/"
            "palate/taste/overall) to Nov 2011. THE labeled ingredient->sensory bridge."),
    S("snap-ratebeer", "SNAP RateBeer reviews", "B", ["beer"],
      ["review", "rating", "sensory"], "bulk", "none", "academic-research-only",
      "academic_release", home="https://snap.stanford.edu/data/web-RateBeer.html",
      est_records=3000000, priority=1, refresh="static",
      notes="~3M reviews, same 5-aspect structure to Nov 2011."),
    S("mcauley-datasets", "McAuley UCSD recommender datasets", "B", ["beer", "spirits"],
      ["review", "rating"], "bulk", "none", "academic-research-only", "academic_release",
      home="https://cseweb.ucsd.edu/~jmcauley/datasets.html", priority=2, refresh="static",
      notes="Canonical source for the SNAP beer corpora + more."),
    S("kaggle-brewers-friend", "Kaggle Brewer's Friend recipes", "B", ["beer"],
      ["recipe", "ingredients", "process"], "bulk", "account", "kaggle-dataset-terms",
      "academic_release", home="https://www.kaggle.com/datasets/jtrofe/beer-recipes",
      est_records=75000, priority=2, refresh="static",
      notes="~75k homebrew recipes; angeredsquid variant has 180k+."),
    S("kaggle-homebrew", "Kaggle homebrew recipe corpora", "B", ["beer"],
      ["recipe", "ingredients"], "bulk", "account", "kaggle-dataset-terms",
      "academic_release",
      home="https://www.kaggle.com/datasets/matiasmiche/homebrew-beer-recipes",
      priority=3, refresh="static"),
    S("beerxml-archives", "BeerXML recipe archives / Compubeer", "B", ["beer"],
      ["recipe", "ingredients", "process"], "bulk", "none", "mixed-open", "academic_release",
      home="https://compubeer.net", est_records=400000, priority=3, refresh="static",
      notes="Assembled from 400k+ BeerXML files."),
    S("hopdatabase", "HopDatabase (kasperg3)", "B", ["ingredients"],
      ["ingredients", "sensory"], "bulk", "none", "open-source-repo", "open_license",
      home="https://github.com/kasperg3/HopDatabase", priority=2, refresh="static",
      notes="Alpha/beta acids, oil fractions scraped from merchants."),
    S("hops-datasets-almet", "hops-datasets (almet)", "B", ["ingredients"],
      ["ingredients", "sensory"], "bulk", "none", "open-source-repo", "open_license",
      home="https://github.com/almet/hops-datasets", priority=3, refresh="static",
      notes="Oils, lineage, age per variety."),
    S("meilgaard-flavor", "Meilgaard beer flavor terminology", "B", ["beer"],
      ["sensory"], "manual", "none", "literature-citation", "academic_release",
      priority=2, refresh="static",
      notes="Flavor chemistry of beer Pt II: 239 aroma volatiles + thresholds; the beer "
            "aroma wheel. Grounds the chemistry->sensory priors."),
    S("beer-flavoromics", "Lager flavoromics + malt sensory lexicon", "B", ["beer", "ingredients"],
      ["sensory"], "manual", "none", "literature-citation", "academic_release",
      priority=3, refresh="static",
      notes="594 volatiles, 71 with OAV>=1; brewing-malt sensory lexicon."),
    S("flavordb", "FlavorDB / VCF", "B", ["ingredients"],
      ["sensory"], "bulk", "none", "check-per-use", "academic_release",
      home="https://cosylab.iiitd.edu.in/flavordb", priority=4, refresh="static",
      notes="Compound->descriptor mappings. Check licensing per use."),
    S("world-whisky-distilleries", "World Whisky Distilleries (Kaggle)", "B", ["spirits"],
      ["producer"], "bulk", "account", "kaggle-dataset-terms", "academic_release",
      home="https://www.kaggle.com/datasets/koki25ando/world-whisky-distilleries-brands-dataset",
      priority=3, refresh="static"),

    # ============ TIER C — commercial / licensable ============
    S("vip", "Vermont Information Processing (VIP)", "C", ["beer", "spirits", "wine"],
      ["inventory", "price", "sku", "producer"], "license", "contract",
      "commercial-license", "contract", jurisdiction="US",
      home="https://www.vtinfo.com", priority=2,
      notes="Dominant three-tier ERP; depletion + retail scan. Sells three-tier data "
            "directly — the answer to the distributor-inventory question."),
    S("provi-sevenfifty", "Provi / SevenFifty", "C", ["beer", "spirits", "wine"],
      ["product", "price", "inventory"], "license", "account", "commercial-account",
      "contract", jurisdiction="US", home="https://www.provi.com", priority=3,
      notes="B2B marketplace, distributor portfolios + pricing. Buyer-side; needs a "
            "licensed retail account."),
    S("circana-iri", "Circana (IRI) / NielsenIQ", "C", ["beer", "spirits", "wine"],
      ["inventory", "price"], "license", "contract", "commercial-license", "contract",
      jurisdiction="US", home="https://www.circana.com", priority=4,
      notes="Off-premise retail scan/panel. Expensive; market-sizing not per-SKU discovery."),
    S("untappd-business", "Untappd for Business API", "C", ["beer"],
      ["menu", "venue", "product"], "api", "contract", "untappd-business-terms", "contract",
      home="https://docs.business.untappd.com", priority=2, sentinel=False,
      notes="Venue menus by menu ID, premium tier; Toast POS integration since Mar 2026. "
            "Licensed path to live draft lists."),
    S("beermenus", "BeerMenus", "C", ["beer"],
      ["menu", "venue", "product", "price"], "license", "contract", "commercial-license",
      "contract", home="https://www.beermenus.com", priority=3,
      notes="On/off-premise lists + analytics. Discovery-oriented."),
    S("digitalpour", "DigitalPour", "C", ["beer"],
      ["menu", "inventory", "price"], "license", "contract", "commercial-license",
      "contract", home="https://digitalpour.com", priority=3,
      notes="Draft management + customer menus; keg levels, sizes, prices. Live feed."),
    S("taphunter", "TapHunter / Evergreen / Sippo / RaspberryPints", "C", ["beer"],
      ["menu", "inventory"], "license", "contract", "commercial-license", "contract",
      home="https://www.evergreenhq.com", priority=4,
      notes="Digital taplist platforms; each a live draft feed if licensed."),
    S("whisky-hunter-api", "Whisky Hunter API", "C", ["spirits"],
      ["product", "price", "rating"], "api", "none", "site-terms", "public_web",
      home="https://whiskyhunter.net/api", priority=3,
      notes="Distillery lists, Top-1000, auction price history."),
    S("whisky-edition-api", "WHISKY:EDITION dev API", "C", ["spirits"],
      ["product"], "api", "key", "commercial-api", "contract",
      home="https://thewhiskyedition.com/developer", priority=4),
    S("gs1-upc", "GS1 / UPCitemdb / Go-UPC / Barcode Lookup", "C", ["all"],
      ["upc", "product"], "api", "key", "commercial-api", "contract",
      home="https://www.gs1.org", priority=3,
      notes="Barcode->product fallback when OFF misses a SKU."),

    # ============ TIER D — consumer web & communities (crawl targets) ============
    # -- producer sites (highest value) --
    S("producer-sites-beer", "US brewery websites (aggregate crawl)", "D", ["beer"],
      ["ingredients", "recipe", "process", "product"], "scrape", "none", "public_web",
      "public_web", robots_allows=None, rate_limit_rps=0.5, refresh="weekly",
      est_records=9900, priority=1,
      notes="~9,900 US breweries publishing STATED hop bills, ingredients. First-party "
            "fact, not inference. Highest-value crawl. Discovered via Parallel FindAll."),
    S("producer-sites-spirits", "US distillery websites (aggregate crawl)", "D", ["spirits"],
      ["process", "ingredients", "product"], "scrape", "none", "public_web", "public_web",
      rate_limit_rps=0.5, refresh="weekly", est_records=3000, priority=1,
      notes="~3,000 US distilleries publishing mash bills, barrel programs, proofs."),
    # -- beer review / community --
    S("untappd", "Untappd (consumer)", "D", ["beer"],
      ["review", "rating", "product", "venue"], "scrape", "none", "public_web", "public_web",
      robots_allows=None, rate_limit_rps=0.2, priority=2, status="blocked",
      notes="ToS forbids scraping. BLOCKED in registry: catalogued, not crawled, pending "
            "a decision. Licensed path is Untappd for Business."),
    S("beeradvocate", "BeerAdvocate + forums", "D", ["beer"],
      ["review", "rating", "product"], "scrape", "none", "public_web", "public_web",
      rate_limit_rps=0.2, priority=2, status="blocked",
      notes="ToS forbids scraping. Historical corpus available via SNAP (Tier B) legally."),
    S("ratebeer", "RateBeer", "D", ["beer"],
      ["review", "rating", "product"], "scrape", "none", "public_web", "public_web",
      rate_limit_rps=0.2, priority=3, status="blocked",
      notes="ABI-owned; ToS forbids scraping. Historical corpus via SNAP (Tier B)."),
    S("beermenus-public", "BeerMenus public pages", "D", ["beer", "venues"],
      ["menu", "venue", "product"], "scrape", "none", "public_web", "public_web",
      rate_limit_rps=0.3, priority=3),
    S("brewbound", "Brewbound", "D", ["beer"], ["release"], "scrape", "none",
      "public_web", "public_web", home="https://www.brewbound.com", refresh="daily",
      priority=2, sentinel=True, notes="Industry news; release/collab sentinel target."),
    S("good-beer-hunting", "Good Beer Hunting", "D", ["beer"], ["release"], "scrape",
      "none", "public_web", "public_web", home="https://www.goodbeerhunting.com",
      refresh="daily", priority=3, sentinel=True),
    S("hop-culture", "Hop Culture", "D", ["beer"], ["release"], "scrape", "none",
      "public_web", "public_web", home="https://www.hopculture.com", refresh="daily",
      priority=3, sentinel=True),
    S("beer-street-journal", "Beer Street Journal", "D", ["beer"], ["release"], "scrape",
      "none", "public_web", "public_web", home="https://www.beerstreetjournal.com",
      refresh="daily", priority=3, sentinel=True),
    S("craft-brewing-business", "Craft Brewing Business", "D", ["beer"], ["release"],
      "scrape", "none", "public_web", "public_web",
      home="https://www.craftbrewingbusiness.com", refresh="daily", priority=4, sentinel=True),
    S("vinepair", "VinePair", "D", ["beer", "spirits", "wine"], ["release", "review"],
      "scrape", "none", "public_web", "public_web", home="https://vinepair.com",
      refresh="daily", priority=3, sentinel=True),
    S("reddit-beer", "Reddit r/beer, r/craftbeer, r/beertrade", "D", ["beer"],
      ["review", "release"], "api", "oauth", "reddit-api-terms", "public_web",
      home="https://www.reddit.com/r/craftbeer", rate_limit_rps=1.0, refresh="daily",
      priority=3, sentinel=True, notes="Official API (OAuth). Trade + hype signal."),
    S("reddit-homebrewing", "Reddit r/Homebrewing, r/TheBrewery", "D", ["beer"],
      ["recipe", "ingredients"], "api", "oauth", "reddit-api-terms", "public_web",
      home="https://www.reddit.com/r/Homebrewing", rate_limit_rps=1.0, priority=3),
    # -- homebrew clone recipes (shortcut to commercial recipes) --
    S("homebrewtalk", "HomebrewTalk (clone recipes)", "D", ["beer"],
      ["recipe", "ingredients", "process"], "scrape", "none", "public_web", "public_web",
      home="https://www.homebrewtalk.com", rate_limit_rps=0.3, priority=2,
      notes="Published clone grain/hop bills for commercial beers — a shortcut to "
            "'what's in Pliny'. community_clone provenance."),
    S("brewers-friend-web", "Brewer's Friend (public recipes)", "D", ["beer"],
      ["recipe", "ingredients"], "scrape", "none", "public_web", "public_web",
      home="https://www.brewersfriend.com/homebrew/recipes", rate_limit_rps=0.3, priority=3),
    S("brewfather-public", "Brewfather / BeerSmith cloud (public)", "D", ["beer"],
      ["recipe", "ingredients"], "scrape", "none", "public_web", "public_web",
      home="https://web.brewfather.app", priority=4),
    S("byo", "Brew Your Own (BYO)", "D", ["beer"], ["recipe", "process"], "scrape",
      "none", "public_web", "public_web", home="https://byo.com", priority=4),
    S("morebeer-kits", "MoreBeer / Northern Brewer clone kits", "D", ["beer"],
      ["recipe", "ingredients"], "scrape", "none", "public_web", "public_web",
      home="https://www.morebeer.com", priority=4,
      notes="Kit pages list grain bills for named commercial clones."),
    # -- beer retail --
    S("tavour", "Tavour", "D", ["beer"], ["product", "price", "release"], "scrape",
      "none", "public_web", "public_web", home="https://tavour.com", refresh="daily",
      priority=3, sentinel=True, notes="Curated craft drops — release sentinel."),
    S("craftshack", "CraftShack", "D", ["beer"], ["product", "price"], "scrape", "none",
      "public_web", "public_web", home="https://www.craftshack.com", priority=4),
    S("halftime", "Half Time Beverage", "D", ["beer"], ["product", "price"], "scrape",
      "none", "public_web", "public_web", home="https://www.halftimebeverage.com", priority=4),
    S("belmont-station", "Belmont Station", "D", ["beer"], ["product", "price"], "scrape",
      "none", "public_web", "public_web", home="https://www.belmont-station.com", priority=5),
    S("totalwine", "Total Wine & More", "D", ["beer", "spirits", "wine"],
      ["product", "price", "inventory"], "scrape", "none", "public_web", "public_web",
      home="https://www.totalwine.com", rate_limit_rps=0.2, priority=2,
      notes="Huge catalog + per-store inventory. Robots-gated crawl."),
    S("binnys", "Binny's Beverage Depot", "D", ["beer", "spirits", "wine"],
      ["product", "price", "inventory"], "scrape", "none", "public_web", "public_web",
      home="https://www.binnys.com", priority=3, sentinel=True,
      notes="Allocated bourbon inventory — drop sentinel."),
    S("bevmo", "BevMo", "D", ["beer", "spirits", "wine"], ["product", "price"], "scrape",
      "none", "public_web", "public_web", home="https://www.bevmo.com", priority=4),
    # -- spirits review / community --
    S("whiskybase", "Whiskybase", "D", ["spirits"],
      ["product", "rating", "review", "price"], "scrape", "none", "public_web", "public_web",
      home="https://www.whiskybase.com", rate_limit_rps=0.2, priority=2,
      notes="Largest whisky DB: bottlings, ratings, prices. Check ToS before crawl."),
    S("distiller", "Distiller", "D", ["spirits"],
      ["product", "rating", "sensory"], "scrape", "none", "public_web", "public_web",
      home="https://distiller.com", rate_limit_rps=0.2, priority=3,
      notes="Ratings, tasting notes, flavor profiles, proof, age. API is partner-only."),
    S("connosr", "Connosr", "D", ["spirits"], ["review", "rating"], "scrape", "none",
      "public_web", "public_web", home="https://www.connosr.com", priority=4),
    S("breaking-bourbon", "Breaking Bourbon", "D", ["spirits"],
      ["review", "release", "process"], "scrape", "none", "public_web", "public_web",
      home="https://www.breakingbourbon.com", refresh="daily", priority=3, sentinel=True,
      notes="Mash bills, reviews, and release calendar — allocated-drop sentinel."),
    S("malt-review", "Malt Review", "D", ["spirits"], ["review"], "scrape", "none",
      "public_web", "public_web", home="https://maltreview.com", priority=4),
    S("whiskyfun", "Whiskyfun", "D", ["spirits"], ["review", "rating"], "scrape", "none",
      "public_web", "public_web", home="https://www.whiskyfun.com", priority=4),
    S("whiskystats", "Whiskystats", "D", ["spirits"], ["price", "rating"], "scrape",
      "none", "public_web", "public_web", home="https://www.whiskystats.com", priority=4,
      notes="Auction price indices."),
    S("whisky-auctioneer", "Whisky Auctioneer", "D", ["spirits"], ["price", "release"],
      "scrape", "none", "public_web", "public_web",
      home="https://whiskyauctioneer.com", refresh="daily", priority=4, sentinel=True),
    S("master-of-malt", "Master of Malt", "D", ["spirits"],
      ["product", "price", "abv"], "scrape", "none", "public_web", "public_web",
      home="https://www.masterofmalt.com", rate_limit_rps=0.3, priority=3),
    S("the-whisky-exchange", "The Whisky Exchange", "D", ["spirits", "wine"],
      ["product", "price"], "scrape", "none", "public_web", "public_web",
      home="https://www.thewhiskyexchange.com", priority=3),
    S("kandl", "K&L Wine Merchants", "D", ["spirits", "wine"],
      ["product", "price", "release"], "scrape", "none", "public_web", "public_web",
      home="https://www.klwines.com", refresh="daily", priority=4, sentinel=True,
      notes="Barrel picks + allocations."),
    S("seelbachs", "Seelbach's", "D", ["spirits"], ["product", "price", "release"],
      "scrape", "none", "public_web", "public_web", home="https://seelbachs.com",
      refresh="daily", priority=4, sentinel=True),
    S("reddit-bourbon", "Reddit r/bourbon, r/Scotch, r/whiskey, r/worldwhisky", "D",
      ["spirits"], ["review", "release"], "api", "oauth", "reddit-api-terms", "public_web",
      home="https://www.reddit.com/r/bourbon", rate_limit_rps=1.0, refresh="daily",
      priority=3, sentinel=True),
    S("whisky-advocate", "Whisky Advocate", "D", ["spirits"], ["review", "release"],
      "scrape", "none", "public_web", "public_web", home="https://www.whiskyadvocate.com",
      refresh="weekly", priority=4, sentinel=True),
    S("adi", "American Distilling Institute", "D", ["spirits"], ["producer", "release"],
      "scrape", "none", "public_web", "public_web", home="https://distilling.com", priority=5),
    # -- ingredient suppliers (public spec sheets) --
    S("yakima-chief", "Yakima Chief Hops (public specs)", "D", ["ingredients"],
      ["ingredients", "sensory"], "scrape", "none", "public_web", "public_web",
      home="https://www.yakimachief.com/commercial/hop-varieties.html", priority=2,
      notes="Alpha/beta/oil per variety + descriptors."),
    S("barthhaas", "BarthHaas hop varieties", "D", ["ingredients"],
      ["ingredients", "sensory"], "scrape", "none", "public_web", "public_web",
      home="https://www.barthhaas.com", priority=3),
    S("hopsteiner", "Hopsteiner variety catalog", "D", ["ingredients"],
      ["ingredients"], "scrape", "none", "public_web", "public_web",
      home="https://www.hopsteiner.com", priority=3),
    S("brewing-malt-suppliers", "Malt supplier spec sheets (Weyermann, Briess, Crisp)", "D",
      ["ingredients"], ["ingredients", "sensory"], "scrape", "none", "public_web",
      "public_web", home="https://www.weyermann.de", priority=3,
      notes="Color, extract, diastatic power, descriptors per malt."),
    S("yeast-labs", "Yeast lab catalogs (White Labs, Wyeast, Lallemand)", "D",
      ["ingredients"], ["ingredients", "sensory"], "scrape", "none", "public_web",
      "public_web", home="https://www.whitelabs.com/yeast-bank", priority=2,
      notes="Attenuation/temp ranges, flocculation, POF, ester profile per strain."),

    # ============ Discovery / sentinel orchestration (Parallel.ai) ============
    S("parallel-findall", "Parallel FindAll (source discovery)", "A", ["all"],
      ["producer", "venue"], "api", "key", "parallel-terms", "contract",
      home="https://parallel.ai", docs="https://docs.parallel.ai", refresh="daily",
      priority=1, sentinel=True,
      notes="Nightly job (25/hr cap): 'sites publishing US tap lists', 'distilleries "
            "publishing mash bills'. base $0.25+$0.03/match, pro $10+$1/match."),
    S("parallel-monitor", "Parallel Monitor (release/menu sentinels)", "A", ["all"],
      ["release", "menu"], "api", "key", "parallel-terms", "contract",
      home="https://parallel.ai", docs="https://docs.parallel.ai", refresh="event",
      priority=1, sentinel=True,
      notes="event_stream for releases/collabs/allocated drops; snapshot for menu drift. "
            "Webhook -> /v1/hooks/parallel. lite $3/1k, base $10/1k."),
    S("parallel-task", "Parallel Task (recipe recovery)", "A", ["all"],
      ["recipe", "ingredients", "process"], "api", "key", "parallel-terms", "contract",
      home="https://parallel.ai", refresh="event", priority=2,
      notes="Structured extraction w/ PER-FIELD CITATIONS — the verifiability story. "
            "core $25/1k, pro $100/1k, ultra8x $2.4k/1k."),
]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for src in SOURCES:
        path = os.path.join(OUT, f"{src['id']}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(src, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"wrote {len(SOURCES)} source files to {os.path.normpath(OUT)}")
    # quick tier tally
    from collections import Counter
    tally = Counter(s["tier"] for s in SOURCES)
    print("by tier:", dict(sorted(tally.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
