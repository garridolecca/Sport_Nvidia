require([
  "esri/Map",
  "esri/views/MapView",
  "esri/Graphic",
  "esri/layers/GraphicsLayer",
  "esri/geometry/Point",
  "esri/geometry/Polyline",
  "esri/geometry/Polygon",
  "esri/symbols/SimpleMarkerSymbol",
  "esri/symbols/SimpleLineSymbol",
  "esri/symbols/SimpleFillSymbol",
  "esri/symbols/TextSymbol",
], function (
  Map, MapView, Graphic, GraphicsLayer, Point, Polyline, Polygon,
  SimpleMarkerSymbol, SimpleLineSymbol, SimpleFillSymbol, TextSymbol
) {
  // ============================================================
  // Configuration
  // ============================================================
  const API_BASE = window.location.origin + "/api";

  // Player colors — cycle through these for different player IDs
  const PLAYER_COLORS = [
    "#e94560", "#00d2ff", "#f5a623", "#7ed321", "#bd10e0",
    "#50e3c2", "#ff6b6b", "#4ecdc4", "#ffe66d", "#a8e6cf",
    "#ffd93d", "#6c5ce7", "#fd79a8", "#00b894", "#e17055",
    "#0984e3", "#fdcb6e", "#6ab04c", "#eb4d4b", "#7158e2",
    "#3ae374", "#ff3838",
  ];

  // ============================================================
  // State
  // ============================================================
  let trackingData = null;
  let currentFrameIdx = 0;
  let isPlaying = false;
  let playSpeed = 1;
  let playInterval = null;
  let showTrails = false;
  let showHeatmap = false;
  let selectedPlayerId = null;
  let playerColorMap = {};
  let playerTrailPoints = {}; // id → [{lat, lon}]

  // ============================================================
  // Layers
  // ============================================================
  const fieldLayer = new GraphicsLayer({ title: "Field" });
  const trailLayer = new GraphicsLayer({ title: "Trails" });
  const heatLayer = new GraphicsLayer({ title: "Heatmap" });
  const playerLayer = new GraphicsLayer({ title: "Players" });
  const labelLayer = new GraphicsLayer({ title: "Labels" });

  // ============================================================
  // Map & View
  // ============================================================
  const map = new Map({
    basemap: "satellite",
    layers: [fieldLayer, heatLayer, trailLayer, playerLayer, labelLayer],
  });

  const view = new MapView({
    container: "viewDiv",
    map: map,
    zoom: 18,
    center: [18.9553, 69.6496], // Alfheim Stadium default
    constraints: { minZoom: 16, maxZoom: 21 },
  });

  // ============================================================
  // Data Loading
  // ============================================================
  async function loadData() {
    try {
      const resp = await fetch(API_BASE + "/tracking");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      trackingData = await resp.json();

      // Assign colors to player IDs
      const allIds = new Set();
      trackingData.frames.forEach(f => f.players.forEach(p => allIds.add(p.id)));
      const sortedIds = [...allIds].sort((a, b) => a - b);
      sortedIds.forEach((id, i) => {
        playerColorMap[id] = PLAYER_COLORS[i % PLAYER_COLORS.length];
      });

      // Setup timeline
      const timeline = document.getElementById("timeline");
      timeline.max = trackingData.frames.length - 1;
      timeline.value = 0;

      // Center map on field
      const field = trackingData.field;
      if (field) {
        const centerLat = (field.corner_tl[0] + field.corner_br[0]) / 2;
        const centerLon = (field.corner_tl[1] + field.corner_br[1]) / 2;
        view.center = [centerLon, centerLat];
        view.zoom = 18;
        drawField(field);
      }

      // Build player list
      buildPlayerList(sortedIds);

      // Render first frame
      renderFrame(0);

      // Hide loading
      document.getElementById("loading").style.display = "none";
      document.getElementById("frame-info").textContent =
        `${trackingData.frames.length} frames | ${sortedIds.length} players`;

    } catch (err) {
      document.getElementById("loading").innerHTML =
        `<p style="color:#e94560">Failed to load data: ${err.message}</p>
         <p style="margin-top:8px;font-size:12px;color:#8899aa">
           Make sure the backend server is running:<br>
           <code>cd backend && python server.py</code><br><br>
           And tracking data exists:<br>
           <code>python pipeline.py --zxy</code> (ground truth) or<br>
           <code>python pipeline.py video.avi</code> (NVIDIA detection)
         </p>`;
    }
  }

  // ============================================================
  // Draw Soccer Field Overlay
  // ============================================================
  function drawField(field) {
    const tl = field.corner_tl;
    const tr = field.corner_tr;
    const br = field.corner_br;
    const bl = field.corner_bl;

    // Pitch outline
    const pitchRing = [
      [tl[1], tl[0]], [tr[1], tr[0]], [br[1], br[0]], [bl[1], bl[0]], [tl[1], tl[0]],
    ];

    fieldLayer.add(new Graphic({
      geometry: new Polygon({ rings: [pitchRing] }),
      symbol: new SimpleFillSymbol({
        color: [34, 139, 34, 80],
        outline: { color: [255, 255, 255, 200], width: 2 },
      }),
    }));

    // Helper: interpolate between two [lat,lon] points
    function lerp(a, b, t) {
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    }

    // Center line
    const midTop = lerp(tl, tr, 0.5);
    const midBot = lerp(bl, br, 0.5);
    fieldLayer.add(new Graphic({
      geometry: new Polyline({
        paths: [[[midTop[1], midTop[0]], [midBot[1], midBot[0]]]],
      }),
      symbol: new SimpleLineSymbol({ color: [255, 255, 255, 150], width: 1.5 }),
    }));

    // Center circle (approximate with polygon)
    const centerLat = (tl[0] + br[0]) / 2;
    const centerLon = (tl[1] + br[1]) / 2;
    const radiusM = 9.15;
    const circlePoints = [];
    for (let i = 0; i <= 36; i++) {
      const angle = (i / 36) * 2 * Math.PI;
      const dLat = (radiusM * Math.cos(angle)) / 111320;
      const dLon = (radiusM * Math.sin(angle)) / 38800;
      circlePoints.push([centerLon + dLon, centerLat + dLat]);
    }
    fieldLayer.add(new Graphic({
      geometry: new Polygon({ rings: [circlePoints] }),
      symbol: new SimpleFillSymbol({
        color: [0, 0, 0, 0],
        outline: { color: [255, 255, 255, 150], width: 1.5 },
      }),
    }));

    // Penalty areas (16.5m from goal line, 40.32m wide)
    const penaltyDepth = 16.5 / 105;
    const penaltyHalfWidth = 20.16 / 68;

    [0, 1].forEach(side => {
      const depthT = side === 0 ? penaltyDepth : 1 - penaltyDepth;
      const topLeft = lerp(
        lerp(tl, tr, side === 0 ? 0 : 1 - penaltyDepth),
        lerp(bl, br, side === 0 ? 0 : 1 - penaltyDepth),
        0.5 - penaltyHalfWidth,
      );
      const topRight = lerp(
        lerp(tl, tr, side === 0 ? 0 : 1 - penaltyDepth),
        lerp(bl, br, side === 0 ? 0 : 1 - penaltyDepth),
        0.5 + penaltyHalfWidth,
      );
      const botLeft = lerp(
        lerp(tl, tr, side === 0 ? penaltyDepth : 1),
        lerp(bl, br, side === 0 ? penaltyDepth : 1),
        0.5 - penaltyHalfWidth,
      );
      const botRight = lerp(
        lerp(tl, tr, side === 0 ? penaltyDepth : 1),
        lerp(bl, br, side === 0 ? penaltyDepth : 1),
        0.5 + penaltyHalfWidth,
      );

      fieldLayer.add(new Graphic({
        geometry: new Polygon({
          rings: [[
            [topLeft[1], topLeft[0]],
            [topRight[1], topRight[0]],
            [botRight[1], botRight[0]],
            [botLeft[1], botLeft[0]],
            [topLeft[1], topLeft[0]],
          ]],
        }),
        symbol: new SimpleFillSymbol({
          color: [0, 0, 0, 0],
          outline: { color: [255, 255, 255, 120], width: 1.2 },
        }),
      }));
    });
  }

  // ============================================================
  // Render a Single Frame
  // ============================================================
  function renderFrame(frameIdx) {
    if (!trackingData || frameIdx >= trackingData.frames.length) return;

    currentFrameIdx = frameIdx;
    const frameData = trackingData.frames[frameIdx];

    // Clear dynamic layers
    playerLayer.removeAll();
    labelLayer.removeAll();

    // Draw each player
    frameData.players.forEach(player => {
      const color = playerColorMap[player.id] || "#ffffff";
      const isSelected = player.id === selectedPlayerId;

      // Player dot
      playerLayer.add(new Graphic({
        geometry: new Point({ longitude: player.lon, latitude: player.lat }),
        symbol: new SimpleMarkerSymbol({
          style: "circle",
          color: color,
          size: isSelected ? 14 : 10,
          outline: {
            color: isSelected ? "#ffffff" : "rgba(0,0,0,0.5)",
            width: isSelected ? 2 : 1,
          },
        }),
        attributes: { playerId: player.id },
      }));

      // Player ID label
      labelLayer.add(new Graphic({
        geometry: new Point({ longitude: player.lon, latitude: player.lat }),
        symbol: new TextSymbol({
          text: String(player.id),
          color: "#ffffff",
          font: { size: 9, weight: "bold" },
          yoffset: -14,
          haloColor: "rgba(0,0,0,0.7)",
          haloSize: 1,
        }),
      }));

      // Accumulate trail points
      if (!playerTrailPoints[player.id]) playerTrailPoints[player.id] = [];
      playerTrailPoints[player.id].push({ lat: player.lat, lon: player.lon });
    });

    // Draw trails if enabled
    if (showTrails) {
      drawTrails();
    }

    // Update UI
    updateUI(frameData);
  }

  // ============================================================
  // Draw Player Movement Trails
  // ============================================================
  function drawTrails() {
    trailLayer.removeAll();

    const idsToShow = selectedPlayerId ? [selectedPlayerId] : Object.keys(playerTrailPoints).map(Number);

    idsToShow.forEach(id => {
      const points = playerTrailPoints[id];
      if (!points || points.length < 2) return;

      // Only show last 100 trail points
      const recent = points.slice(-100);
      const path = recent.map(p => [p.lon, p.lat]);

      trailLayer.add(new Graphic({
        geometry: new Polyline({ paths: [path] }),
        symbol: new SimpleLineSymbol({
          color: playerColorMap[id] || "#ffffff",
          width: 2,
          style: "solid",
        }),
      }));
    });
  }

  // ============================================================
  // Heatmap (simple density visualization)
  // ============================================================
  function drawHeatmap() {
    heatLayer.removeAll();
    if (!showHeatmap || !trackingData) return;

    // Aggregate all positions up to current frame
    const grid = {};
    const gridSize = 0.00003; // ~3m cells

    for (let i = 0; i <= currentFrameIdx; i++) {
      const frame = trackingData.frames[i];
      frame.players.forEach(p => {
        if (selectedPlayerId && p.id !== selectedPlayerId) return;
        const gx = Math.floor(p.lon / gridSize);
        const gy = Math.floor(p.lat / gridSize);
        const key = `${gx},${gy}`;
        grid[key] = (grid[key] || 0) + 1;
      });
    }

    const maxCount = Math.max(...Object.values(grid), 1);

    Object.entries(grid).forEach(([key, count]) => {
      const [gx, gy] = key.split(",").map(Number);
      const lon = gx * gridSize;
      const lat = gy * gridSize;
      const intensity = count / maxCount;

      // Color: blue → yellow → red
      let r, g, b;
      if (intensity < 0.5) {
        r = Math.floor(intensity * 2 * 255);
        g = Math.floor(intensity * 2 * 255);
        b = 255 - r;
      } else {
        r = 255;
        g = Math.floor((1 - intensity) * 2 * 255);
        b = 0;
      }

      heatLayer.add(new Graphic({
        geometry: new Polygon({
          rings: [[
            [lon, lat],
            [lon + gridSize, lat],
            [lon + gridSize, lat + gridSize],
            [lon, lat + gridSize],
            [lon, lat],
          ]],
        }),
        symbol: new SimpleFillSymbol({
          color: [r, g, b, Math.floor(intensity * 120 + 30)],
          outline: { width: 0 },
        }),
      }));
    });
  }

  // ============================================================
  // Player List Panel
  // ============================================================
  function buildPlayerList(ids) {
    const container = document.getElementById("player-list");
    container.innerHTML = "";

    ids.forEach(id => {
      const item = document.createElement("div");
      item.className = "player-item";
      item.dataset.id = id;
      item.innerHTML = `
        <span class="dot" style="background:${playerColorMap[id]}"></span>
        <span>Player ${id}</span>
      `;
      item.addEventListener("click", () => {
        if (selectedPlayerId === id) {
          selectedPlayerId = null;
          item.classList.remove("selected");
        } else {
          document.querySelectorAll(".player-item.selected").forEach(el => el.classList.remove("selected"));
          selectedPlayerId = id;
          item.classList.add("selected");
        }
        renderFrame(currentFrameIdx);
        if (showHeatmap) drawHeatmap();
      });
      container.appendChild(item);
    });
  }

  // ============================================================
  // UI Updates
  // ============================================================
  function updateUI(frameData) {
    const ts = frameData.timestamp;
    const mins = Math.floor(ts / 60);
    const secs = Math.floor(ts % 60);
    const timeStr = `${mins}:${String(secs).padStart(2, "0")}`;

    document.getElementById("time-display").textContent = timeStr;
    document.getElementById("timeline").value = currentFrameIdx;
    document.getElementById("stat-frame").textContent = currentFrameIdx;
    document.getElementById("stat-time").textContent = timeStr;
    document.getElementById("stat-count").textContent = frameData.players.length;
  }

  // ============================================================
  // Playback Controls
  // ============================================================
  function togglePlay() {
    isPlaying = !isPlaying;
    document.getElementById("btn-play").textContent = isPlaying ? "\u23F8" : "\u25B6";
    document.getElementById("btn-play").classList.toggle("active", isPlaying);

    if (isPlaying) {
      const intervalMs = Math.max(16, 100 / playSpeed);
      playInterval = setInterval(() => {
        if (currentFrameIdx < trackingData.frames.length - 1) {
          renderFrame(currentFrameIdx + 1);
          if (showHeatmap && currentFrameIdx % 10 === 0) drawHeatmap();
        } else {
          togglePlay(); // stop at end
        }
      }, intervalMs);
    } else {
      clearInterval(playInterval);
    }
  }

  function changeSpeed(delta) {
    const speeds = [0.25, 0.5, 1, 2, 4, 8];
    const idx = speeds.indexOf(playSpeed);
    const newIdx = Math.max(0, Math.min(speeds.length - 1, idx + delta));
    playSpeed = speeds[newIdx];
    document.getElementById("speed-display").textContent = playSpeed + "x";

    // Restart playback with new speed
    if (isPlaying) {
      clearInterval(playInterval);
      isPlaying = false;
      togglePlay();
    }
  }

  // ============================================================
  // Event Listeners
  // ============================================================
  document.getElementById("btn-play").addEventListener("click", togglePlay);

  document.getElementById("btn-prev").addEventListener("click", () => {
    if (currentFrameIdx > 0) renderFrame(currentFrameIdx - 1);
  });

  document.getElementById("btn-next").addEventListener("click", () => {
    if (currentFrameIdx < trackingData.frames.length - 1) renderFrame(currentFrameIdx + 1);
  });

  document.getElementById("timeline").addEventListener("input", (e) => {
    const idx = parseInt(e.target.value, 10);
    playerTrailPoints = {}; // reset trails on seek
    renderFrame(idx);
    if (showHeatmap) drawHeatmap();
  });

  document.getElementById("btn-speed-down").addEventListener("click", () => changeSpeed(-1));
  document.getElementById("btn-speed-up").addEventListener("click", () => changeSpeed(1));

  document.getElementById("btn-trails").addEventListener("click", () => {
    showTrails = !showTrails;
    document.getElementById("btn-trails").classList.toggle("active", showTrails);
    if (!showTrails) trailLayer.removeAll();
    else drawTrails();
  });

  document.getElementById("btn-heatmap").addEventListener("click", () => {
    showHeatmap = !showHeatmap;
    document.getElementById("btn-heatmap").classList.toggle("active", showHeatmap);
    if (!showHeatmap) heatLayer.removeAll();
    else drawHeatmap();
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.code === "ArrowLeft" && currentFrameIdx > 0) renderFrame(currentFrameIdx - 1);
    if (e.code === "ArrowRight" && currentFrameIdx < trackingData.frames.length - 1) renderFrame(currentFrameIdx + 1);
  });

  // Click on map to select nearest player
  view.on("click", (event) => {
    view.hitTest(event).then((response) => {
      const hit = response.results.find(r => r.graphic && r.graphic.attributes && r.graphic.attributes.playerId);
      if (hit) {
        const id = hit.graphic.attributes.playerId;
        selectedPlayerId = selectedPlayerId === id ? null : id;
        document.querySelectorAll(".player-item.selected").forEach(el => el.classList.remove("selected"));
        if (selectedPlayerId) {
          const el = document.querySelector(`.player-item[data-id="${id}"]`);
          if (el) el.classList.add("selected");
        }
        renderFrame(currentFrameIdx);
        if (showHeatmap) drawHeatmap();
      }
    });
  });

  // ============================================================
  // Initialize
  // ============================================================
  view.when(() => loadData());
});
