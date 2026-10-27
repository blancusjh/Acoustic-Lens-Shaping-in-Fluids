"""``afl design``: desired stigmatic singlet in, machine settings out."""

import json
import math
from pathlib import Path

from .machine import DEFAULT_MACHINE, Singlet
from .recipe import design

MILLI = 1e-3


def add_arguments(parser):
    lens = parser.add_argument_group("lens (distances from the front vertex, in --units, times --scale)")
    lens.add_argument("--lens", type=Path, help="JSON file with the Singlet fields instead of the options below")
    lens.add_argument("--object", help="object distance, negative below the lens; 'collimated' (or --object=-inf) for light from infinity")
    lens.add_argument("--image", type=float, help="image distance, positive above the lens")
    lens.add_argument("--thickness", type=float, help="vertex-to-vertex thickness")
    shape = lens.add_mutually_exclusive_group()
    shape.add_argument("--shape-factor", type=float, help="paper shape factor of the singlet")
    shape.add_argument("--intermediate", type=float, help="intermediate stigmatic point instead")
    lens.add_argument("--indices", type=float, nargs=3, default=(1.0, 1.7, 1.0), metavar=("N_OBJECT", "N_LENS", "N_IMAGE"))
    lens.add_argument("--scale", default="1", help="geometric scale applied to the distances, e.g. 1/14")
    lens.add_argument("--units", choices=("mm", "m"), default="mm")
    lens.add_argument("--name", default="")
    parser.add_argument("--machine", type=Path, default=DEFAULT_MACHINE, help="machine file (default: v1)")
    parser.add_argument("--no-startup", action="store_true", help="skip the start-up ramp (faster; status at best command-only)")
    parser.add_argument("--out", type=Path, help="write recipe.json and the design files to this new directory")


def _scale(text):
    if "/" in text:
        numerator, denominator = text.split("/")
        return float(numerator) / float(denominator)
    return float(text)


def lens_from_args(args, parser):
    if args.lens:
        return Singlet.from_dict(json.loads(args.lens.read_text()))
    if args.object is None or args.image is None or args.thickness is None:
        parser.error("give --lens FILE, or --object, --image and --thickness")
    if args.shape_factor is None and args.intermediate is None:
        parser.error("give --shape-factor or --intermediate")
    unit = MILLI if args.units == "mm" else 1.0
    text = args.object.strip().lower()
    # An object at infinity is always below the lens in this convention.
    obj = -math.inf if text in ("collimated", "inf", "+inf", "-inf", "infinity") else float(text)
    return Singlet(
        object_m=obj if math.isinf(obj) else obj * unit,
        image_m=args.image * unit,
        thickness_m=args.thickness * unit,
        shape_factor=args.shape_factor,
        intermediate_m=None if args.intermediate is None else args.intermediate * unit,
        indices=tuple(args.indices),
        scale=_scale(args.scale),
        name=args.name,
    )


def run(args, parser):
    lens = lens_from_args(args, parser)
    if args.out is not None and args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    recipe = design(lens, args.machine, startup=not args.no_startup)
    if args.out is not None:
        recipe.save(args.out)
    print(summary(recipe))
    return recipe


def summary(recipe):
    d = recipe.as_dict()
    lines = [f"status: {d['status']} — {d['meaning']}"]
    if d["message"] and d["message"] != d["meaning"]:
        lines.append(f"  {d['message']}")
    s, p = d["settings"], d["prediction"]
    if s:
        pins = s["pin_heights_above_bottom_window_m"]
        lines.append(f"pins above the bottom window: {pins[0] * 1e3:.4f} mm, {pins[1] * 1e3:.4f} mm")
        lines.append("fill volumes: " + ", ".join(f"{k} {v * 1e9:.4f} µL" for k, v in s["fill_volumes_m3"].items()))
        drive = s["drive"]
        lines.append(f"drive: {' + '.join(f'{f / 1e6:g} MHz' for f in drive['bands_hz'])} on {drive['channels']} "
                     f"channels, {drive['source_power_w']:.2f} W, band peaks "
                     + ", ".join(f"{x / 1e6:.2f} MPa" for x in drive["band_peak_pressure_pa"]))
    if p:
        if p["formed_pupil_error_m"] is not None:
            lines.append("formed face error (front, back): "
                         + ", ".join(f"{x * 1e9:.3g} nm" for x in p["formed_pupil_error_m"]))
        lines.append(f"frozen error {p['frozen_pupil_error_m'] * 1e9:.3g} nm; convex bound {p['convex_bound_m'] * 1e9:.3g} nm")
        if "spot_max_radius_m" in p:
            lines.append(f"spot max {p['spot_max_radius_m'] * 1e6:.3g} µm, rms {p['spot_rms_radius_m'] * 1e6:.3g} µm; "
                         f"Airy radius {p['airy_radius_m_at_587_6_nm'] * 1e6:.3g} µm")
    lines.append(f"({d['seconds']:.0f} s; {d['verification']})")
    return "\n".join(lines)
