"""Headless-Chrome smoke test + screenshots of the viewer. Usage: python -m tests.shots"""
import sys
from playwright.sync_api import sync_playwright

URL = "file:///home/claude/crashlab/dist/crashlab.html"
SHOTS = [
    ("s2_impact", "?scene=2&t=1.95", None),
    ("s2_recon", "?scene=2&t=5&tab=recon&ghost=1", None),
    ("s4_recon", "?scene=4&t=4.4&tab=recon", None),
    ("s5_fog", "?scene=5&t=6.8&tab=live", None),
    ("s6_compound", "?scene=6&t=2.7", None),
    ("s3_evid", "?scene=3&evid=1&tab=method", None),
]


def main():
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("console", lambda m: errors.append(("console." + m.type, m.text)) if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errors.append(("pageerror", str(e))))
        for name, q, _ in SHOTS:
            pg.goto(URL + q)
            pg.wait_for_timeout(700)
            pg.screenshot(path=f"/tmp/shot_{name}.png")
            print("shot", name)
        # interaction smoke test
        pg.goto(URL + "?scene=1")
        pg.wait_for_timeout(500)
        for sel in ["#bPlay", "#bFwd", "#bBack", "#bImpact", "#tgEvid", "#tgEvid", "#btnReset"]:
            pg.click(sel)
        for tab in ["live", "recon", "method", "report"]:
            pg.click(f'#tabs button[data-tab="{tab}"]')
        for i in range(6):
            pg.click(f".stab >> nth={i}")
            pg.wait_for_timeout(120)
        pg.keyboard.press("Space"); pg.keyboard.press("ArrowRight"); pg.keyboard.press("e"); pg.keyboard.press("r")
        pg.mouse.move(500, 300); pg.mouse.wheel(0, -300); pg.mouse.wheel(0, 300)
        b.close()
    print("errors:", errors if errors else "none")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
