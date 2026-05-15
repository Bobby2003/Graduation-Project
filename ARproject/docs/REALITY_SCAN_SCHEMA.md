# Reality Scan Schema

`last_reality_scan` is the unified scan result object stored under `UserProfile.private_realm_json`. It is shared by:

1. **Web Simulation Pipeline** — browser Reality Override page (`source: web_simulator`, `coordinate_space: screen_normalized`).
2. **Depth Scan Pipeline** — depth camera / scanner backends (`POST /api/depth-scan/import/`, typically `source: depth_camera`, `coordinate_space: metric_room`).

Downstream consumers read the same object for:

- `my_realm.html` — LAST REALITY SCAN summary  
- `/realm/` — HUD scan line + **Scan Ghosts**  
- Progress — e.g. `reality_scan_completed` (payload may include `source`, `coordinate_space`)

The two pipelines **do not replace each other**; they **merge** into one schema for product demo and real sensor data alike.

---

## Common fields

| Field | Type | Description |
|--------|------|-------------|
| `id` | string | Unique scan id (server-generated if omitted). |
| `mode` | string | e.g. `simulated`, `sensor`. |
| `source` | string | e.g. `web_simulator`, `depth_camera`. |
| `pipeline` | string | Processing pipeline name. |
| `coordinate_space` | string | `screen_normalized` or `metric_room` (drives ghost rendering). |
| `captured_at` | string | ISO-8601 timestamp (set by server on save/import). |
| `material_pack` | string | Material protocol key (catalog). |
| `surfaces` | array | Detected surfaces (server **whitelist-truncated**, see below). |
| `mesh_summary` | object or null | Optional mesh stats (`vertices`, `faces`, `unit` / `scale_unit`). |
| `raw_ref` | string or null | Optional pointer to raw data (max **300** chars server-side). |
| `device` | object or null | **Depth import only**: sanitized `type`, `name`, `serial` strings. |

---

## Surface object (whitelist)

Server keeps only these keys per surface (nested junk is dropped):

- `id`, `type`, `confidence`
- `bounds` — object with numeric `x`, `y`, `w`, `h` (screen-normalized pipeline)
- `center`, `size`, `normal` — numeric arrays (`center`/`normal` length 3; `size` up to 4)
- `material_pack`

**Limits:** Web save: up to **12** surfaces. Depth import: up to **64** surfaces.

---

## Coordinate spaces

### `screen_normalized`

Used by the browser simulation. Overlay / stored bounds are **0–1 relative to the video frame**:

```json
{
  "type": "floor",
  "confidence": 0.94,
  "bounds": { "x": 0.08, "y": 0.66, "w": 0.84, "h": 0.26 }
}
```

### `metric_room`

Used by depth / reconstruction pipelines. Surfaces use **room-scale** coordinates (meters in app space):

```json
{
  "type": "floor",
  "confidence": 0.91,
  "center": [0, 0.02, 0.8],
  "size": [3.2, 2.8],
  "normal": [0, 1, 0],
  "material_pack": "hacker_matrix"
}
```

Walls may use any outward normal, e.g. `[1, 0, 0]` for a side wall. **Scan Ghosts** orient metric planes with `THREE.Quaternion.setFromUnitVectors` from the default plane normal `(0,0,1)` to the surface normal.

---

## APIs

### Web: `POST /api/reality-override/save/`

Body: `{ "material_pack": "<key>", "scan": { ... } }`.  
Server fills `id`, `captured_at`, defaults `source` / `pipeline` / `coordinate_space` if missing, and cleans `surfaces` / `mesh_summary` / `raw_ref`.

### Depth: `POST /api/depth-scan/import/`

Body: JSON with `surfaces` (array), optional `material_pack`, `mesh_summary`, `device`, `pipeline`, `coordinate_space`, `raw_ref`, etc.  
**Does not** modify `room_template`, `lighting`, or `decorations` — only `last_reality_scan` and `material_pack`.  
May trigger `reality_scan_completed` server-side; response includes `progress`.

### Clear: `POST /api/reality-scan/clear/`

Removes **`last_reality_scan`** only. Does **not** change `room_template`, `material_pack`, `lighting`, or `decorations`.  
Returns `{ "ok": true, "realm": <merged_private_realm> }`.  
Use from **我的位面** (CLEAR SCAN DATA) or **Depth Import Tester** for demo reset.

**Surface limits (server):** Web save uses up to **`MAX_WEB_SURFACES` (12)** surfaces; depth import up to **`MAX_DEPTH_SURFACES` (64)**. Constants live in `ARapp/realm_scan.py`.

---

## Web example (minimal)

```json
{
  "material_pack": "cyber_neon",
  "scan": {
    "mode": "simulated",
    "source": "web_simulator",
    "pipeline": "reality_override_web",
    "coordinate_space": "screen_normalized",
    "surfaces": [
      {
        "id": "floor_001",
        "type": "floor",
        "confidence": 0.94,
        "bounds": { "x": 0.08, "y": 0.66, "w": 0.84, "h": 0.26 }
      }
    ]
  }
}
```

---

## Depth example (minimal)

```json
{
  "material_pack": "hacker_matrix",
  "surfaces": [
    {
      "id": "floor_plane_001",
      "type": "floor",
      "confidence": 0.91,
      "center": [0, 0.02, 0.8],
      "size": [3.2, 2.8],
      "normal": [0, 1, 0]
    }
  ],
  "mesh_summary": { "vertices": 12840, "faces": 24112, "unit": "meter" },
  "pipeline": "depth_reconstruction_v1",
  "coordinate_space": "metric_room",
  "device": { "type": "depth_camera", "name": "Orbbec / RealSense" }
}
```

---

## Debugging without hardware

When `DEBUG=True` or the user is **staff**, open **`/depth-scan/tester/`** to paste JSON and call `POST /api/depth-scan/import/`, then use **ENTER REALM** to verify metric Scan Ghosts.

---

## Implementation reference

- Server cleaning: `ARapp/realm_scan.py` (`MAX_WEB_SURFACES`, `MAX_DEPTH_SURFACES`, surface whitelist, `raw_ref` cap)  
- Merge-safe realm core fields: `ARapp/views.py` — `_merge_private_realm_core`  
- Ghost rendering: `static/js/realm/realm-main.js` — `createScanGhosts`, `coordinate_space` branch
