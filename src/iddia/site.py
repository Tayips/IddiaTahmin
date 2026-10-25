"""Tahmin sayfasını (docs/index.html) üretir; veri sayfanın içine gömülür."""
import json
from pathlib import Path

TEMPLATE = Path(__file__).parent / "site_template.html"


def render(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
