# AR REALM (Graduation Project)

Django + Three.js **Desktop Realm** with private realm JSON, room templates, material protocols, training, missions/achievements, and **Reality Override** (dual pipeline: web simulation + depth scan import).

## Docs

- **[Reality Scan Schema](docs/REALITY_SCAN_SCHEMA.md)** — unified `last_reality_scan` fields, coordinate spaces, API examples, and Scan Ghosts behavior.

## Dev quickstart

```bash
pip install -r requirements.txt   # if present
python manage.py migrate
python manage.py seed_sprint2_progress
python manage.py runserver
```

## Depth import tester (staff / DEBUG only)

`/depth-scan/tester/` — paste metric scan JSON, import via `/api/depth-scan/import/`, then open `/realm/` to verify ghosts.

`POST /api/reality-scan/clear/` — removes `last_reality_scan` only (does not reset room template or lighting); see [Reality Scan Schema](docs/REALITY_SCAN_SCHEMA.md). Available from **我的位面** and the tester page.
