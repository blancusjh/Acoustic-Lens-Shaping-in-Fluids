/* Local, deterministic playback of the saved fluid trajectory. No network calls. */
(() => {
  "use strict";
  const D = window.LENS_DATA,
    cfg = D.config;
  const $ = (id) => document.getElementById(id);
  const finiteObject = cfg.object_distance_m != null;
  $("experiment-title").textContent = finiteObject
    ? "Shaping a Cartesian diopter with sound"
    : "A liquid asphere, shaped by sound";
  $("experiment-subtitle").textContent =
    `${2 * cfg.clear_radius_m * 1000} mm clear aperture · ` +
    (finiteObject
      ? `object ${cfg.object_distance_m * 1000} mm → image +${cfg.focal_distance_m * 1000} mm`
      : `collimated input → focus ${cfg.focal_distance_m * 1000} mm`);
  $("illumination-note").textContent = finiteObject
    ? `nₒ = ${cfg.refractive_index}, nᵢ = ${cfg.image_refractive_index}. Conjugates are measured from the target vertex. Incident rays show the prescribed spherical wavefront inside the resin; external illumination through the window is not designed.`
    : "Parallel incident rays inside the resin. Distances use the target surface vertex.";
  $("frequency-value").textContent = `${cfg.frequency_hz / 1e6} MHz`;
  function decode(record) {
    const binary = atob(record.data),
      bytes = new Uint8Array(binary.length);
    for (let j = 0; j < binary.length; ++j) bytes[j] = binary.charCodeAt(j);
    return new {
      float32: Float32Array,
      int32: Int32Array,
      uint32: Uint32Array,
    }[record.dtype](bytes.buffer);
  }
  const full = vtk.Rendering.Misc.vtkFullScreenRenderWindow.newInstance({
    rootContainer: $("scene"),
    background: [17 / 255, 30 / 255, 45 / 255],
    containerStyle: {
      height: "100%",
      width: "100%",
      position: "absolute",
      top: "0",
      left: "0",
    },
  });
  const renderer = full.getRenderer(),
    renderWindow = full.getRenderWindow();
  const camera = renderer.getActiveCamera();
  const actors = {},
    polys = {},
    mappers = {};
  function addMesh(key, geometry) {
    const poly = vtk.Common.DataModel.vtkPolyData.newInstance();
    poly.getPoints().setData(decode(geometry.points), 3);
    if (geometry.polys.shape[0])
      poly.getPolys().setData(decode(geometry.polys));
    if (geometry.lines.shape[0])
      poly.getLines().setData(decode(geometry.lines));
    const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
    mapper.setInputData(poly);
    mapper.setScalarVisibility(false);
    const actor = vtk.Rendering.Core.vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(...geometry.color);
    actor.getProperty().setOpacity(geometry.opacity);
    actor.getProperty().setAmbient(0.25);
    actor.getProperty().setDiffuse(0.75);
    actor.getProperty().setSpecular(0.5);
    actor.getProperty().setSpecularPower(35);
    renderer.addActor(actor);
    actors[key] = actor;
    polys[key] = poly;
    mappers[key] = mapper;
  }
  for (const [key, geometry] of Object.entries(D.geometry))
    addMesh(key, geometry);
  const pointIds = decode(D.radial_ids),
    surfaceIds = decode(D.surface_ids),
    rowIds = decode(D.array_rows);
  const pressure = decode(D.pressure_pa),
    fluid = decode(D.fluid_velocity);
  const count = D.times.length,
    nv = D.slice_vertex_count;
  let frame = count - 1,
    playing = false,
    clockTime = D.times[frame],
    lastTick = 0;
  let activeView = "apparatus";
  const empty = {
    points: { dtype: "float32", shape: [0], data: "" },
    polys: { dtype: "uint32", shape: [0], data: "" },
    lines: { dtype: "uint32", shape: [0], data: "" },
    color: [0.96, 0.78, 0.43],
    opacity: 1,
  };
  addMesh("rays", empty);
  actors.rays.getProperty().setLineWidth(1.5);
  addMesh("flow", { ...empty, color: [0.8, 0.95, 1] });
  actors.flow.getProperty().setLineWidth(1.8);
  addMesh("aperture", { ...empty, color: [0.96, 0.78, 0.43] });
  actors.aperture.getProperty().setLineWidth(2.5);
  const pressureMax = Math.max(
    ...D.report.history.map((x) => x.peak_acoustic_pressure_pa),
  );
  const lut = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();
  [
    [0, 0.07, 0.14, 0.25],
    [0.25, 0.07, 0.42, 0.53],
    [0.5, 0.25, 0.75, 0.67],
    [0.75, 0.95, 0.8, 0.4],
    [1, 0.98, 0.54, 0.36],
  ].forEach((x) => lut.addRGBPoint(x[0] * pressureMax, ...x.slice(1)));
  mappers.pressure.setLookupTable(lut);
  mappers.pressure.setScalarRange(0, pressureMax);
  mappers.pressure.setScalarVisibility(true);
  actors.pressure.getProperty().setLighting(false);
  $("pressure-max").textContent = (pressureMax / 1e6).toFixed(2) + " MPa";
  const clip = vtk.Common.DataModel.vtkPlane.newInstance({
    normal: [0, 1, 0],
    origin: [0, 0, 0],
  });

  function interpolation(array, normalized) {
    const position = Math.min(
      array.length - 1,
      Math.max(0, normalized * (array.length - 1)),
    );
    const j = Math.min(array.length - 2, Math.floor(position));
    return array[j] * (j + 1 - position) + array[j + 1] * (position - j);
  }
  function phaseRGB(real, imag) {
    let h = (Math.atan2(imag, real) / (2 * Math.PI) + 1) % 1;
    const value =
      0.28 +
      0.72 * Math.min(1, Math.hypot(real, imag) / cfg.max_wall_speed_m_s);
    const hsv = [h * 6, 0.5, value],
      i = Math.floor(hsv[0]),
      f = hsv[0] - i;
    const p = value * 0.5,
      q = value * (1 - 0.5 * f),
      t = value * (0.5 + 0.5 * f);
    return [
      [value, t, p],
      [q, value, p],
      [p, value, t],
      [p, q, value],
      [t, p, value],
      [value, p, q],
    ][i % 6].map((x) => Math.round(255 * x));
  }
  function lineGeometry(key, segments) {
    const points = [],
      lines = [];
    segments.forEach((segment) => {
      const start = points.length / 3;
      lines.push(segment.length);
      segment.forEach((point, j) => {
        points.push(...point);
        lines.push(start + j);
      });
    });
    polys[key].getPoints().setData(new Float32Array(points), 3);
    polys[key].getLines().setData(new Uint32Array(lines));
    polys[key].modified();
  }
  function updateGeometry(k) {
    const points = polys.liquid.getPoints().getData(),
      normals = new Float32Array(points.length);
    for (let j = 0; j < pointIds.length; ++j) {
      const index = pointIds[j],
        x = points[3 * j],
        y = points[3 * j + 1],
        r = Math.hypot(x, y);
      if (index >= 0) {
        points[3 * j + 2] = 1000 * D.height_m[k][index];
        const slope = D.slope[k][index],
          length = Math.hypot(1, slope);
        normals[3 * j] = r ? (-slope * x) / r / length : 0;
        normals[3 * j + 1] = r ? (-slope * y) / r / length : 0;
        normals[3 * j + 2] = 1 / length;
      } else {
        normals[3 * j] = r ? x / r : 0;
        normals[3 * j + 1] = r ? y / r : 0;
        normals[3 * j + 2] = r ? 0 : -1;
      }
    }
    polys.liquid.getPoints().modified();
    polys.liquid.modified();
    const top = polys.surface.getPoints().getData(),
      topNormals = new Float32Array(top.length);
    surfaceIds.forEach((index, j) => {
      const x = top[3 * j],
        y = top[3 * j + 1],
        r = Math.hypot(x, y),
        slope = D.slope[k][index],
        length = Math.hypot(1, slope);
      top[3 * j + 2] = 1000 * D.height_m[k][index];
      topNormals[3 * j] = r ? (-slope * x) / r / length : 0;
      topNormals[3 * j + 1] = r ? (-slope * y) / r / length : 0;
      topNormals[3 * j + 2] = 1 / length;
    });
    polys.surface.getPoints().modified();
    polys.surface.modified();
    polys.surface.getPointData().setNormals(
      vtk.Common.Core.vtkDataArray.newInstance({
        name: "Interface normals",
        numberOfComponents: 3,
        values: topNormals,
      }),
    );
    actors.surface.getProperty().setInterpolationToPhong();
    actors.surface.getProperty().setSpecular(0.9);
    actors.surface.getProperty().setSpecularPower(75);
    // Let the renderer compute face normals on the shared side/top rim.
    const colors = new Uint8Array(rowIds.length * 3);
    rowIds.forEach((row, j) =>
      colors.set(phaseRGB(D.drive_real[k][row], D.drive_imag[k][row]), 3 * j),
    );
    polys.array.getCellData().setScalars(
      vtk.Common.Core.vtkDataArray.newInstance({
        name: "Array phase",
        numberOfComponents: 3,
        values: colors,
      }),
    );
    mappers.array.setScalarVisibility(true);
    mappers.array.setColorModeToDirectScalars();
    mappers.array.setScalarModeToUseCellData();
    const slice = polys.pressure.getPoints().getData(),
      scalars = new Float32Array(2 * nv);
    for (let side = 0; side < 2; ++side)
      for (let j = 0; j < nv; ++j) {
        const at = side * nv + j;
        slice[3 * at + 2] =
          1000 *
          (-cfg.depth_m * (1 - D.slice_eta[j]) +
            D.slice_eta[j] * interpolation(D.height_m[k], D.slice_r[j]));
        scalars[at] = pressure[k * nv + j];
      }
    polys.pressure.getPoints().modified();
    polys.pressure.modified();
    polys.pressure.getPointData().setScalars(
      vtk.Common.Core.vtkDataArray.newInstance({
        name: "Peak acoustic pressure (Pa)",
        values: scalars,
      }),
    );
    const raySegments = [],
      plane = D.report.target.target_plane_m * 1000;
    for (let j = 1; j < D.ray_r_m[k].length; j += 2)
      for (const sign of [-1, 1]) {
        const radius = sign * D.ray_r_m[k][j] * 1000,
          height = D.ray_z_m[k][j] * 1000,
          startZ = -cfg.depth_m * 1000 - (finiteObject ? 0 : 1.2),
          incomingSlope = D.ray_incident_slope?.[k]?.[j] || 0,
          startRadius = radius + sign * (startZ - height) * incomingSlope;
        raySegments.push([
          [startRadius, -0.018, startZ],
          [radius, -0.018, height],
          [sign * D.ray_spot_m[k][j] * 1000, -0.018, plane],
        ]);
      }
    lineGeometry("rays", raySegments);
    const ring = [];
    for (let j = 0; j <= 128; ++j) {
      const angle = (j * 2 * Math.PI) / 128;
      ring.push([
        cfg.clear_radius_m * 1000 * Math.cos(angle),
        cfg.clear_radius_m * 1000 * Math.sin(angle),
        1000 * interpolation(D.height_m[k], cfg.clear_radius_m / cfg.radius_m) +
          0.008,
      ]);
    }
    lineGeometry("aperture", [ring]);
    const flowSegments = [];
    for (let ir = 5; ir < cfg.mesh_radial; ir += 8)
      for (let iz = 8; iz < cfg.mesh_vertical; iz += 12) {
        const j = ir * (cfg.mesh_vertical + 1) + iz;
        const r = D.slice_r[j] * cfg.radius_m * 1000,
          z =
            1000 *
            (-cfg.depth_m * (1 - D.slice_eta[j]) +
              D.slice_eta[j] * interpolation(D.height_m[k], D.slice_r[j]));
        const ur = fluid[2 * k * nv + j] * 1000,
          uz = fluid[(2 * k + 1) * nv + j] * 1000;
        for (const sign of [-1, 1]) {
          const x = sign * r,
            dx = sign * ur,
            dz = uz,
            length = Math.hypot(dx, dz);
          if (length < 1e-5) continue;
          const end = [x + dx, -0.055, z + dz],
            wing = Math.min(0.06, length * 0.3);
          flowSegments.push([[x, -0.055, z], end]);
          flowSegments.push([
            [
              end[0] - (wing * (dx + 0.5 * dz)) / length,
              -0.055,
              end[2] - (wing * (dz - 0.5 * dx)) / length,
            ],
            end,
            [
              end[0] - (wing * (dx - 0.5 * dz)) / length,
              -0.055,
              end[2] - (wing * (dz + 0.5 * dx)) / length,
            ],
          ]);
        }
      }
    lineGeometry("flow", flowSegments);
  }
  function applyLayers() {
    document.querySelectorAll("[data-layer]").forEach((input) => {
      actors[input.dataset.layer].setVisibility(input.checked);
      if (input.dataset.layer === "cylinder")
        actors.rim.setVisibility(input.checked);
      if (input.dataset.layer === "liquid") {
        actors.aperture.setVisibility(input.checked);
        actors.surface.setVisibility(input.checked);
        if (activeView === "lens") actors.liquid.setVisibility(false);
      }
    });
    const cutaway = $("cutaway").checked;
    ["cylinder", "array", "rim"].forEach((key) => {
      mappers[key].removeAllClippingPlanes();
      if (cutaway) mappers[key].addClippingPlane(clip);
    });
    $("pressure-legend").style.display = actors.pressure.getVisibility()
      ? "block"
      : "none";
    $("field-note").textContent = actors.flow.getVisibility()
      ? "White flow vectors show one second of travel at the instantaneous computed velocity. They are slow liquid motion, not acoustic oscillation or streaming."
      : "Array color shows phase; brightness shows drive amplitude. The 16 sectors in each row share a drive.";
    $("view-name").textContent =
      activeView.toUpperCase() + (cutaway ? " / CUTAWAY" : " / FULL GEOMETRY");
    if (window.lensViewerState)
      window.lensViewerState.layers = Object.fromEntries(
        Object.entries(actors).map(([key, actor]) => [
          key,
          actor.getVisibility(),
        ]),
      );
    renderWindow.render();
  }
  function preset(name) {
    activeView = name;
    camera.setViewUp(0, 0, 1);
    camera.setParallelProjection(true);
    const layer = (key, value) =>
      (document.querySelector(`[data-layer="${key}"]`).checked = value);
    ["liquid", "base", "cylinder", "array"].forEach((key) => layer(key, true));
    layer("rays", name === "optics");
    layer("pressure", name === "section");
    layer("flow", false);
    $("cutaway").checked = name !== "lens";
    if (name === "apparatus") {
      camera.setPosition(13, -19, 12);
      camera.setFocalPoint(0, 0, -2.3);
      camera.setParallelScale(7.0);
    }
    if (name === "lens") {
      camera.setPosition(9, -13, 5);
      camera.setFocalPoint(0, 0, 0.2);
      camera.setParallelScale(4.7);
      layer("array", false);
      layer("cylinder", false);
      layer("base", false);
    }
    if (name === "optics") {
      camera.setPosition(15, -40, 15);
      camera.setFocalPoint(0, 0, 7);
      camera.setParallelScale(15.8);
    }
    if (name === "section") {
      camera.setPosition(0, -35, -1);
      camera.setFocalPoint(0, 0, -2.5);
      camera.setParallelScale(6.1);
      layer("liquid", false);
      layer("flow", true);
    }
    document
      .querySelectorAll("[data-view]")
      .forEach((b) => b.classList.toggle("active", b.dataset.view === name));
    renderer.resetCameraClippingRange();
    applyLayers();
  }
  function chart(id, curves, xRange, yRange, marker = null) {
    const canvas = $(id),
      ratio = window.devicePixelRatio || 1,
      width = canvas.clientWidth,
      height = canvas.clientHeight;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const c = canvas.getContext("2d");
    c.scale(ratio, ratio);
    const left = 39,
      right = 12,
      top = 9,
      bottom = 23,
      ww = width - left - right,
      hh = height - top - bottom;
    const px = (x) => left + ((x - xRange[0]) / (xRange[1] - xRange[0])) * ww;
    const py = (y) =>
      top + hh - ((y - yRange[0]) / (yRange[1] - yRange[0])) * hh;
    c.font = "9px -apple-system, sans-serif";
    c.lineWidth = 1;
    c.strokeStyle = "#26384c";
    c.fillStyle = "#8a9caf";
    for (let j = 0; j <= 3; ++j) {
      const y = yRange[0] + (j * (yRange[1] - yRange[0])) / 3;
      c.beginPath();
      c.moveTo(left, py(y));
      c.lineTo(width - right, py(y));
      c.stroke();
      c.textAlign = "right";
      c.fillText(
        id === "history-chart"
          ? Math.pow(10, y).toPrecision(1)
          : y.toFixed(Math.abs(yRange[1] - yRange[0]) < 2 ? 2 : 1),
        left - 7,
        py(y) + 3,
      );
    }
    for (let j = 0; j <= 4; ++j) {
      const x = xRange[0] + (j * (xRange[1] - xRange[0])) / 4;
      c.textAlign = "center";
      c.fillText(x.toFixed(1), px(x), height - 5);
    }
    c.save();
    c.beginPath();
    c.rect(left, top, ww, hh);
    c.clip();
    curves.forEach((curve) => {
      c.beginPath();
      c.strokeStyle = curve.color;
      c.lineWidth = curve.width || 1.8;
      c.setLineDash(curve.dash || []);
      curve.x.forEach((x, j) => {
        if (j === 0) c.moveTo(px(x), py(curve.y[j]));
        else c.lineTo(px(x), py(curve.y[j]));
      });
      c.stroke();
    });
    if (marker !== null) {
      c.setLineDash([3, 3]);
      c.strokeStyle = "#ecbd6e";
      c.beginPath();
      c.moveTo(px(marker), top);
      c.lineTo(px(marker), top + hh);
      c.stroke();
    }
    c.restore();
  }
  function drawCharts(k) {
    const radial = D.radial_m.map((x) => x * 1000),
      mm = (values) => values.map((x) => x * 1000);
    chart(
      "profile-chart",
      [
        { x: radial, y: mm(D.height_m[0]), color: "#71869f" },
        {
          x: radial,
          y: mm(D.target_height_m),
          color: "#edbb65",
          dash: [4, 3],
          width: 2.7,
        },
        { x: radial, y: mm(D.height_m[k]), color: "#4ad8cb" },
      ],
      [0, cfg.radius_m * 1000],
      [
        Math.min(0, ...D.height_m[k].map((v) => v * 1000)),
        1.08 * Math.max(...D.target_height_m, ...D.height_m[0]) * 1000,
      ],
    );
    const limit = Math.max(
      6,
      Math.min(70, Math.max(...D.departure_um[k].map(Math.abs)) * 1.1),
    );
    chart(
      "departure-chart",
      [
        {
          x: D.departure_r_m.map((x) => x * 1000),
          y: D.departure_um[k],
          color: "#4ad8cb",
        },
      ],
      [0, cfg.clear_radius_m * 1000],
      [-limit, limit],
    );
    chart(
      "history-chart",
      [
        {
          x: D.times,
          y: D.report.history.map((row) =>
            Math.log10(Math.max(0.01, row.rms_spot_at_target_m * 1e6)),
          ),
          color: "#4ad8cb",
        },
      ],
      [0, D.times[count - 1]],
      [-2, 3],
      D.times[k],
    );
  }
  function selectFrame(k) {
    frame = Math.max(0, Math.min(count - 1, k));
    const row = D.report.history[frame];
    $("time").value = frame;
    $("time-label").textContent = D.times[frame].toFixed(3) + " s";
    $("spot-value").textContent = (row.rms_spot_at_target_m * 1e6).toFixed(
      row.rms_spot_at_target_m > 1e-5 ? 1 : 2,
    );
    $("focus-value").textContent = (
      row.best_focus_from_vertex_m * 1000
    ).toFixed(2);
    $("sag-value").textContent = (D.height_m[frame][0] * 1000).toFixed(3);
    $("shape-value").textContent = (row.shape_rms_to_target_m * 1e6).toFixed(2);
    $("peak-pressure").textContent =
      (row.peak_acoustic_pressure_pa / 1e6).toFixed(2) + " MPa";
    $("power-value").textContent =
      (row.source_power_w * 1000).toFixed(1) + " mW";
    $("speed-value").textContent =
      (row.maximum_fluid_speed_m_s * 1000).toFixed(3) + " mm/s";
    $("state-name").textContent =
      frame === 0
        ? "Unforced equilibrium"
        : D.times[frame] < cfg.ramp_s
          ? "Array shaping the liquid"
          : "Computed driven state";
    updateGeometry(frame);
    drawCharts(frame);
    renderer.resetCameraClippingRange();
    renderWindow.render();
    // Read-only observable state for browser verification and reproducible inspection.
    window.lensViewerState = {
      frame,
      time_s: D.times[frame],
      vertex_mm: polys.liquid.getPoints().getData()[2],
      points_checksum: polys.liquid
        .getPoints()
        .getData()
        .reduce((a, b, j) => a + b * ((j % 7) + 1), 0),
      pressure_checksum: pressure
        .subarray(frame * nv, (frame + 1) * nv)
        .reduce((a, b) => a + b, 0),
      layers: Object.fromEntries(
        Object.entries(actors).map(([key, actor]) => [
          key,
          actor.getVisibility(),
        ]),
      ),
    };
  }
  function setPlaying(value) {
    playing = value;
    lastTick = 0;
    $("play").textContent = playing ? "Ⅱ Pause" : "▶ Play";
  }
  function animate(now) {
    if (playing) {
      if (lastTick)
        clockTime +=
          Math.min(0.1, (now - lastTick) / 1000) * Number($("speed").value);
      lastTick = now;
      let next = frame;
      while (next < count - 1 && D.times[next + 1] <= clockTime) next++;
      if (next !== frame) selectFrame(next);
      if (clockTime >= D.times[count - 1]) {
        selectFrame(count - 1);
        setPlaying(false);
      }
    }
    requestAnimationFrame(animate);
  }
  $("time").max = count - 1;
  $("end-label").textContent =
    D.times[count - 1].toFixed(2) + " s · final computed state";
  $("time").addEventListener("input", () => {
    setPlaying(false);
    clockTime = D.times[Number($("time").value)];
    selectFrame(Number($("time").value));
  });
  $("play").addEventListener("click", () => {
    if (!playing && frame === count - 1) {
      clockTime = 0;
      selectFrame(0);
    }
    setPlaying(!playing);
  });
  $("restart").addEventListener("click", () => {
    setPlaying(false);
    clockTime = 0;
    selectFrame(0);
  });
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => b.addEventListener("click", () => preset(b.dataset.view)));
  document.querySelectorAll("[data-layer],#cutaway").forEach((input) =>
    input.addEventListener("change", () => {
      applyLayers();
      selectFrame(frame);
    }),
  );
  window.addEventListener("resize", () => {
    full.resize();
    drawCharts(frame);
  });
  preset("apparatus");
  selectFrame(frame);
  $("loading").style.display = "none";
  requestAnimationFrame(animate);
})();
