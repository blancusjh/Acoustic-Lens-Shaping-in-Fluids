"""Display saved two-interface equilibria; no fabricated time evolution."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import DualConfig
from .surface import CartesianPatch, DualSurface


def load_setup(root):
    root = Path(root)
    base = root / "artifacts/dual-precision-2026-09-23"
    records, hashes = [], {}
    for folder, grid in (
        ("verify-mean-448-7p2mhz-retry1", "design"),
        ("verify-mean-448-surface-refinement", "design"),
        ("verify-mean-448-surface-refinement", "refined-1"),
    ):
        directory = base / folder
        paths = [
            directory / "results.json",
            directory / "config.json",
            directory / "asymmetric-pair-A" / f"{grid}-state.npz",
        ]
        for path in paths:
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = next(g for g in json.loads(paths[0].read_text())[0]["grids"] if g["grid"] == grid)
        cfg = DualConfig(**result["numerics"])
        space = DualSurface(cfg)
        inputs = json.loads(paths[1].read_text())["cases"][0]["faces"]
        targets = [
            CartesianPatch(cfg, j, f["optical"], f["vertex_displacement_m"])
            for j, f in enumerate(inputs)
        ]
        with np.load(paths[2], allow_pickle=False) as state:
            q, drive = state["coefficients_m"].copy(), state["source_velocity_m_s"].copy()
        r = np.linspace(0, cfg.radius_m, 1001)
        z = np.stack([cfg.levels_m[j + 1] + space.evaluate(q[j], r) for j in range(2)])
        target = np.stack([cfg.levels_m[j + 1] + targets[j].evaluate(r) for j in range(2)])
        record = {
            "cfg": cfg,
            "space": space,
            "q": q,
            "drive": drive,
            "r_m": r,
            "z_m": z,
            "target_z_m": target,
            "result": result,
        }
        assert np.all(z[0] < z[1]), "Liquid interfaces cross."
        np.testing.assert_allclose(z[:, -1], cfg.levels_m[1:3], atol=1e-12, rtol=0)
        if records:
            np.testing.assert_array_equal(drive, records[0]["drive"])
        records.append(record)
    return records, hashes


def setup_figures(records):
    rec = records[-1]
    cfg, space, q = rec["cfg"], rec["space"], rec["q"]
    fig = plt.figure(figsize=(10, 8), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    radius = cfg.radius_m * 1e3
    theta = np.linspace(0, 2 * np.pi, 81)
    rr, tt = np.meshgrid(np.linspace(0, radius, 45), theta)
    colors = ["#2166ac", "#d6604d"]
    for j, color in enumerate(colors):
        z = (cfg.levels_m[j + 1] + space.evaluate(q[j], rr * 1e-3)) * 1e3
        ax.plot_surface(
            rr * np.cos(tt), rr * np.sin(tt), z, color=color, alpha=0.55, linewidth=0, shade=True
        )
        ax.text(
            radius + 0.1,
            0,
            cfg.levels_m[j + 1] * 1e3,
            ["Front / bottom interface", "Back / top interface"][j],
            color=color,
            fontsize=9,
        )
    # Zero-thickness boundary outlines: this apparatus specifies no wall/plate thickness.
    for z in (cfg.levels_m[0] * 1e3, cfg.levels_m[-1] * 1e3):
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z), color=".4")
    for angle in np.linspace(0, np.pi, 9):
        ax.plot(
            [radius * np.cos(angle)] * 2,
            [radius * np.sin(angle)] * 2,
            np.array(cfg.levels_m)[[0, -1]] * 1e3,
            color=".7",
            lw=0.6,
        )
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        zlabel="Laboratory z (mm)",
        xlim=(-radius, radius),
        ylim=(-radius, radius),
        zlim=(-3, 3),
        title="Two-face setup · 104-element stationary solution\nBlue: bottom; orange: top · equal geometric scale",
    )
    ax.set_box_aspect((2 * radius, 2 * radius, (cfg.levels_m[-1] - cfg.levels_m[0]) * 1e3))
    ax.view_init(elev=15, azim=-60)

    profiles, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for j in range(2):
        axes[j, 0].plot(rec["r_m"] * 1e3, rec["target_z_m"][j] * 1e3, "k--", label="Target")
        for item, color in zip(records, ["#7570b3", "#1b9e77", "#d95f02"]):
            label = f"{item['cfg'].surface_elements} surface elements"
            r = item["r_m"]
            axes[j, 0].plot(r * 1e3, item["z_m"][j] * 1e3, color=color, label=label)
            pupil = r <= cfg.clear_radius_m
            axes[j, 1].plot(
                r[pupil] * 1e3,
                (item["z_m"][j, pupil] - item["target_z_m"][j, pupil]) * 1e9,
                color=color,
                label=label,
            )
        axes[j, 0].axvspan(cfg.clear_radius_m * 1e3, radius, color=".9")
        axes[j, 0].set(ylabel="Laboratory z (mm)", title=["Bottom / front", "Top / back"][j])
        axes[j, 1].axhspan(-10, 10, color="#b8e0d2", alpha=0.5)
        axes[j, 1].set_yscale("symlog", linthresh=10)
        axes[j, 1].set(ylabel="Signed height error (nm)", title="Pupil error · shaded ±10 nm")
        for a in axes[j]:
            a.set_xlabel("Radius (mm)")
            a.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    profiles.suptitle("Same source commands · surface-refinement comparison, NOT a timeline")
    return fig, profiles


def interactive_setup(records):
    """Canvas camera rotation and discrete equilibrium selection, without a time axis."""
    payload = []
    for rec in records:
        indices = np.arange(0, len(rec["r_m"]), 10)
        payload.append(
            {
                "elements": rec["cfg"].surface_elements,
                "r": (rec["r_m"][indices] * 1e3).tolist(),
                "z": (rec["z_m"][:, indices] * 1e3).tolist(),
                "errors_nm": [f["max_error_m"] * 1e9 for f in rec["result"]["faces"]],
            }
        )
    cfg = records[0]["cfg"]
    payload_json = json.dumps(
        {
            "states": payload,
            "radius_mm": cfg.radius_m * 1e3,
            "ends_mm": [cfg.levels_m[0] * 1e3, cfg.levels_m[-1] * 1e3],
        }
    )
    return """
<section id="dual-setup-panel" style="border:1px solid #aaa;padding:12px;background:white;color:#222">
<h2>Both interfaces in the two-face apparatus</h2>
<p>Stationary saved geometry. Rotation changes only the camera. No physical time,
formation trajectory, or two-face spot/audio evolution is available.</p>
<label>Saved equilibrium (surface elements):
<select id="dual-state"><option value="0">52</option><option value="1">80</option>
<option value="2" selected>104</option></select></label>
<button id="dual-rotate" type="button">Rotate camera</button>
<button id="dual-reset" type="button">Reset camera</button>
<p id="dual-status"></p>
<canvas id="dual-canvas" width="900" height="650" style="max-width:100%;height:auto"></canvas>
<p>Blue = front/bottom interface. Orange = back/top interface. Grey = cylinder and
rigid end-boundary outlines; no wall thickness or transducer hardware is invented.
Axes and geometry use mm with equal scale. Source partitions are omitted for clarity.</p>
</section>
<script>
(function(){
const data = PAYLOAD;
const canvas=document.getElementById('dual-canvas'), ctx=canvas.getContext('2d');
const select=document.getElementById('dual-state'), button=document.getElementById('dual-rotate');
let angle=-0.85, rotating=false, last=null;
function project(x,y,z){
 const u=Math.cos(angle)*x-Math.sin(angle)*y, v=Math.sin(angle)*x+Math.cos(angle)*y;
 const e=.30;
 return [450+65*u,310+65*(Math.sin(e)*v-Math.cos(e)*z)];
}
function path(points,color,width=1){
 ctx.beginPath();points.forEach((p,i)=>{const xy=project(...p);if(i)ctx.lineTo(...xy);else ctx.moveTo(...xy)});
 ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();
}
function ring(r,z,color,width){
 path(Array.from({length:97},(_,k)=>[r*Math.cos(k*Math.PI/48),r*Math.sin(k*Math.PI/48),z]),color,width);
}
function draw(){
 const s=data.states[Number(select.value)];ctx.clearRect(0,0,900,650);
 data.ends_mm.forEach(z=>ring(data.radius_mm,z,'#888',1));
 for(let k=0;k<12;k++){let t=k*Math.PI/6;
 path(data.ends_mm.map(z=>[data.radius_mm*Math.cos(t),data.radius_mm*Math.sin(t),z]),'#ddd');}
 ['#2166ac','#d6604d'].forEach((color,j)=>{
   for(let k=0;k<s.r.length;k+=5)ring(s.r[k],s.z[j][k],color,.8);
   for(let k=0;k<24;k++){let t=k*Math.PI/12;
     path(s.r.map((r,i)=>[r*Math.cos(t),r*Math.sin(t),s.z[j][i]]),color,.8);}
   const label=project(data.radius_mm,0,s.z[j][s.r.length-1]);
   ctx.font='15px sans-serif';ctx.fillStyle=color;
   ctx.fillText(j?'TOP / BACK':'BOTTOM / FRONT',label[0]+8,label[1]);
 });
 ctx.fillStyle='#333';ctx.font='16px sans-serif';
 ctx.fillText('Stationary geometry · camera rotation only · no physical timeline',100,35);
 ctx.fillText('Cylinder radius '+data.radius_mm+' mm; end planes '+data.ends_mm.join(', ')+' mm',180,610);
 document.getElementById('dual-status').textContent=s.elements+' surface elements | '+
 'mean max errors: bottom '+s.errors_nm[0].toFixed(2)+' nm; top '+s.errors_nm[1].toFixed(2)+
 ' nm | camera angle '+(angle*180/Math.PI).toFixed(1)+'°';
 window.dualSetupState={elements:s.elements,angle:angle,rotating:rotating,
  bottom_vertex_mm:s.z[0][0],top_vertex_mm:s.z[1][0],physical_time_s:null};
}
function tick(stamp){
 if(!rotating)return;if(last!==null)angle+=(stamp-last)*.00035;last=stamp;draw();requestAnimationFrame(tick);
}
button.onclick=()=>{rotating=!rotating;last=null;button.textContent=rotating?'Pause camera':'Rotate camera';draw();if(rotating)requestAnimationFrame(tick)};
document.getElementById('dual-reset').onclick=()=>{rotating=false;angle=-.85;last=null;button.textContent='Rotate camera';draw()};
select.onchange=draw;draw();
})();
</script>""".replace("PAYLOAD", payload_json)
