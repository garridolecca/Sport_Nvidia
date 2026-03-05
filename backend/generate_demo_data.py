"""
Generate realistic demo tracking data for testing the ArcGIS visualization
without needing to download the Alfheim dataset or run NVIDIA inference.

Simulates 22 players (11 per team) moving on the pitch with realistic patterns.
"""

import json
import math
import random
from pathlib import Path
from config import OUTPUT_DIR, STADIUM_LAT, STADIUM_LON, METERS_PER_DEG_LAT, METERS_PER_DEG_LON

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0
DURATION_SECONDS = 120  # 2 minutes of simulated play
FPS = 10


def field_to_gps(fx, fy):
    lon = STADIUM_LON + fx / METERS_PER_DEG_LON
    lat = STADIUM_LAT - fy / METERS_PER_DEG_LAT
    return round(lat, 7), round(lon, 7)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


class SimPlayer:
    def __init__(self, pid, team, home_x, home_y):
        self.id = pid
        self.team = team
        self.home_x = home_x
        self.home_y = home_y
        self.x = home_x + random.uniform(-5, 5)
        self.y = home_y + random.uniform(-5, 5)
        self.vx = 0
        self.vy = 0

    def step(self, dt, ball_x, ball_y):
        # Players drift toward ball but stay near home position
        dx_ball = ball_x - self.x
        dy_ball = ball_y - self.y
        dx_home = self.home_x - self.x
        dy_home = self.home_y - self.y

        dist_to_ball = math.sqrt(dx_ball**2 + dy_ball**2 + 1)

        # Attraction to ball (stronger when closer)
        ball_pull = min(0.3, 15.0 / dist_to_ball)
        home_pull = 0.1

        ax = ball_pull * dx_ball / dist_to_ball + home_pull * dx_home + random.gauss(0, 1.5)
        ay = ball_pull * dy_ball / dist_to_ball + home_pull * dy_home + random.gauss(0, 1.5)

        # Speed limit ~8 m/s
        self.vx = clamp((self.vx + ax * dt) * 0.92, -8, 8)
        self.vy = clamp((self.vy + ay * dt) * 0.92, -8, 8)

        self.x = clamp(self.x + self.vx * dt, 1, FIELD_LENGTH - 1)
        self.y = clamp(self.y + self.vy * dt, 1, FIELD_WIDTH - 1)


def generate():
    random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Formation: 4-4-2 for both teams
    #   GK, 4 defenders, 4 midfielders, 2 forwards
    team_a_positions = [
        (5, 34),     # GK
        (20, 10), (20, 27), (20, 41), (20, 58),   # DEF
        (40, 10), (40, 27), (40, 41), (40, 58),   # MID
        (55, 27), (55, 41),                        # FWD
    ]
    team_b_positions = [
        (100, 34),   # GK
        (85, 10), (85, 27), (85, 41), (85, 58),
        (65, 10), (65, 27), (65, 41), (65, 58),
        (50, 27), (50, 41),
    ]

    players = []
    for i, (hx, hy) in enumerate(team_a_positions):
        players.append(SimPlayer(i + 1, "A", hx, hy))
    for i, (hx, hy) in enumerate(team_b_positions):
        players.append(SimPlayer(i + 12, "B", hx, hy))

    # Ball movement
    ball_x, ball_y = 52.5, 34.0
    ball_vx, ball_vy = random.uniform(-5, 5), random.uniform(-3, 3)

    total_frames = DURATION_SECONDS * FPS
    dt = 1.0 / FPS
    frames = []

    for fi in range(total_frames):
        timestamp = fi * dt

        # Move ball
        ball_x += ball_vx * dt
        ball_y += ball_vy * dt
        if ball_x < 2 or ball_x > 103:
            ball_vx *= -1
            ball_x = clamp(ball_x, 2, 103)
        if ball_y < 2 or ball_y > 66:
            ball_vy *= -1
            ball_y = clamp(ball_y, 2, 66)

        # Random ball direction changes
        if random.random() < 0.05:
            ball_vx = random.uniform(-8, 8)
            ball_vy = random.uniform(-5, 5)

        # Move players
        frame_players = []
        for p in players:
            p.step(dt, ball_x, ball_y)
            lat, lon = field_to_gps(p.x, p.y)
            frame_players.append({
                "id": p.id,
                "team": p.team,
                "lat": lat,
                "lon": lon,
                "field_x": round(p.x, 2),
                "field_y": round(p.y, 2),
                "confidence": 1.0,
            })

        frames.append({
            "frame": fi,
            "timestamp": round(timestamp, 3),
            "players": frame_players,
        })

    # Field corners in GPS
    tl = field_to_gps(0, 0)
    tr = field_to_gps(FIELD_LENGTH, 0)
    br = field_to_gps(FIELD_LENGTH, FIELD_WIDTH)
    bl = field_to_gps(0, FIELD_WIDTH)

    output = {
        "source": "demo_simulation",
        "total_frames_processed": len(frames),
        "field": {
            "length_m": FIELD_LENGTH,
            "width_m": FIELD_WIDTH,
            "anchor_lat": tl[0],
            "anchor_lon": tl[1],
            "corner_tl": list(tl),
            "corner_tr": list(tr),
            "corner_br": list(br),
            "corner_bl": list(bl),
        },
        "frames": frames,
    }

    out_path = OUTPUT_DIR / "tracking_data.json"
    with open(out_path, "w") as f:
        json.dump(output, f)

    print(f"Generated {len(frames)} frames with {len(players)} players")
    print(f"Duration: {DURATION_SECONDS}s at {FPS} FPS")
    print(f"Saved to: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    generate()
