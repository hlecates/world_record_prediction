#!/usr/bin/env python3
import os
import time
import logging
from pathlib import Path

import config
import utils
from bs4 import BeautifulSoup
import requests

# ----------------------------------------------------------------------
# 1) helper to fetch & parse
# ----------------------------------------------------------------------
def fetch_page(url: str, pause: float = 1.0) -> BeautifulSoup:
    resp = utils.http_get_with_retries(url, headers={"User-Agent": config.USER_AGENT})
    time.sleep(pause)
    return BeautifulSoup(resp.text, "html.parser")

# ----------------------------------------------------------------------
# 2) grab the “slugs” off the index page
# ----------------------------------------------------------------------
def fetch_all_meet_slugs() -> list[str]:
    index_url = f"{config.USA_SWIMMING_BASE}/times/data-hub/meet-results"
    soup = fetch_page(index_url)
    tabs = soup.select(".usas-content-leftrailnavigationoption2-tab a")
    slugs = []
    for a in tabs:
        href = a["data-usas-href"]  # e.g. "//www.usaswimming.org/.../tyr-pro-swim-series-results"
        slug = href.rstrip("/").split("/")[-1]
        slugs.append(slug)
    return slugs

# ----------------------------------------------------------------------
# 3) for each slug, download every PDF link on that page
# ----------------------------------------------------------------------
def download_meet_pdfs(slugs: list[str], out_base: Path):
    for slug in slugs:
        meet_url = f"{config.USA_SWIMMING_BASE}/times/data-hub/meet-results/{slug}"
        logging.info(f"Fetching meet page: {meet_url}")
        soup = fetch_page(meet_url)

        # create sub‐folder for this meet
        meet_dir = out_base / slug
        meet_dir.mkdir(parents=True, exist_ok=True)

        # find & download each PDF
        for link in soup.select("a[href$='.pdf']"):
            pdf_url = link["href"]
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            elif pdf_url.startswith("/"):
                pdf_url = config.USA_SWIMMING_BASE + pdf_url

            fname = pdf_url.split("/")[-1]
            out_path = meet_dir / fname
            if out_path.exists():
                logging.info(f"  ⏭ already have {fname}")
                continue

            logging.info(f"  ↓ downloading {fname}")
            resp = requests.get(pdf_url, headers={"User-Agent": config.USER_AGENT})
            resp.raise_for_status()
            out_path.write_bytes(resp.content)

# ----------------------------------------------------------------------
# 4) main entrypoint
# ----------------------------------------------------------------------
def main():
    # set up logging
    utils.setup_logging()

    # target directory: ../data/raw/meet_results
    here = Path(__file__).parent
    out_base = (here / ".." / "data" / "raw" / "meet_results").resolve()

    # run
    slugs = fetch_all_meet_slugs()
    logging.info(f"Discovered {len(slugs)} meets → saving into {out_base}")
    download_meet_pdfs(slugs, out_base)
    logging.info("Done.")

if __name__ == "__main__":
    main()
