"""Deterministic, bounded 2D explanatory light scattering.

Only explicitly supplied surface segments participate in intersections. Object
boxes are annotations, never reconstructed geometry. Reflectance and specular
fractions are assumptions, not quantities measured from the input photograph.
This is a six-direction illustration, not a physically converged path tracer.
"""
from __future__ import annotations

import math

ASSUMPTIONS = (
    "2D assumed segments; user supplied/LLM assumed material coefficients. "
    "Detection boxes are annotations, not collision geometry. Five fixed diffuse "
    "directions use cosine-normalized weights; this is an explanatory 2D model, "
    "not reconstructed 3D or a calibrated radiometric simulation."
)
_DIFFUSE_ANGLES = (-70.0, -35.0, 0.0, 35.0, 70.0)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _point(value, name, normalized=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two coordinates")
    point = (_number(value[0], name + "[0]"), _number(value[1], name + "[1]"))
    if normalized and not all(0 <= coordinate <= 1 for coordinate in point):
        raise ValueError(f"{name} coordinates must be in [0, 1]")
    return point


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def _unit(vector, name):
    length = math.hypot(*vector)
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError(f"{name} must have nonzero finite length")
    return vector[0] / length, vector[1] / length


def _validate_objects(objects):
    if not isinstance(objects, (list, tuple)) or not objects or len(objects) > 256:
        raise ValueError("objects must contain 1 to 256 entries")
    result = {}
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"objects[{index}] must be an object")
        identifier = _identifier(item.get("id"), f"objects[{index}].id")
        if identifier in result:
            raise ValueError(f"Duplicate object id: {identifier}")
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Object {identifier}: bbox must contain four normalized coordinates")
        bbox = tuple(_number(value, "bbox") for value in bbox)
        if not all(0 <= value <= 1 for value in bbox) or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            raise ValueError(f"Object {identifier}: bbox must be bounded and have positive area")
        reflectance = item.get("reflectance")
        if not isinstance(reflectance, (list, tuple)) or len(reflectance) != 2:
            raise ValueError(f"Object {identifier}: reflectance must be [min, max]")
        low, high = (_number(value, "reflectance") for value in reflectance)
        if not 0 <= low <= high <= 1:
            raise ValueError(f"Object {identifier}: reflectance must satisfy 0 <= min <= max <= 1")
        specular = _number(item.get("specular_fraction", 0.05), "specular_fraction")
        if not 0 <= specular <= 1:
            raise ValueError(f"Object {identifier}: specular_fraction must be in [0, 1]")
        result[identifier] = {"reflectance": (low + high) / 2, "specular_fraction": specular}
    return result


def _validate_surfaces(surfaces, objects, width, height, epsilon):
    if not isinstance(surfaces, (list, tuple)) or not surfaces or len(surfaces) > 512:
        raise ValueError("surfaces must contain 1 to 512 explicit segments")
    result = {}
    for index, item in enumerate(surfaces):
        if not isinstance(item, dict):
            raise ValueError(f"surfaces[{index}] must be an object")
        identifier = _identifier(item.get("id"), f"surfaces[{index}].id")
        if identifier in result:
            raise ValueError(f"Duplicate surface id: {identifier}")
        owner = _identifier(item.get("object_id"), "surface.object_id")
        if owner not in objects:
            raise ValueError(f"Surface {identifier} refers to unknown object {owner}")
        a = _point(item.get("a"), "surface.a", normalized=True)
        b = _point(item.get("b"), "surface.b", normalized=True)
        a, b = (a[0] * width, a[1] * height), (b[0] * width, b[1] * height)
        if math.dist(a, b) <= epsilon:
            raise ValueError(f"Surface {identifier} has zero or negligible length")
        result[identifier] = {"id": identifier, "object_id": owner, "a": a, "b": b}
    return result


def _project_to_segment(point, surface):
    delta = _sub(surface["b"], surface["a"])
    parameter = max(0.0, min(1.0, _dot(_sub(point, surface["a"]), delta) / _dot(delta, delta)))
    return surface["a"][0] + parameter * delta[0], surface["a"][1] + parameter * delta[1]


def _ray_segment_distance(point, direction, surface, epsilon):
    """Nearest positive isolated intersection, including collinear endpoints.

    An overlap starting at the ray origin is grazing contact, not a new hit.
    No object/surface is globally excluded: other faces of the same object and
    every positive crossing beyond the numerical epsilon remain eligible.
    """
    edge = _sub(surface["b"], surface["a"])
    delta = _sub(surface["a"], point)
    denominator = _cross(direction, edge)
    length = math.hypot(*edge)
    if abs(denominator) <= 1e-12 * length:
        if abs(_cross(delta, direction)) > epsilon:
            return None
        start, end = sorted((_dot(delta, direction), _dot(_sub(surface["b"], point), direction)))
        if end <= epsilon or start <= epsilon:
            return None
        return start
    distance = _cross(delta, edge) / denominator
    fraction = _cross(delta, direction) / denominator
    fraction_epsilon = epsilon / length
    if distance > epsilon and -fraction_epsilon <= fraction <= 1 + fraction_epsilon:
        return distance
    return None


def _frame_distance(point, direction, width, height):
    distances = []
    for coordinate, component, extent in zip(point, direction, (width, height)):
        if component > 1e-15:
            distances.append((extent - coordinate) / component)
        elif component < -1e-15:
            distances.append(-coordinate / component)
    return max(0.0, min(distances))


def _normal(surface, incoming):
    tangent = _unit(_sub(surface["b"], surface["a"]), "surface tangent")
    normal = (-tangent[1], tangent[0])
    if _dot(normal, incoming) > 0:
        normal = (-normal[0], -normal[1])
    return normal


def _outgoing(incoming, normal, reflected_energy, specular_fraction):
    rays = []
    specular_energy = reflected_energy * specular_fraction
    if specular_energy > 0:
        cosine = _dot(incoming, normal)
        reflected = _unit((incoming[0] - 2 * cosine * normal[0],
                           incoming[1] - 2 * cosine * normal[1]), "reflected direction")
        rays.append((reflected, specular_energy, "specular"))
    diffuse_energy = reflected_energy - specular_energy
    if diffuse_energy > 0:
        weights = [math.cos(math.radians(angle)) for angle in _DIFFUSE_ANGLES]
        total = math.fsum(weights)
        spent = 0.0
        for index, (angle, weight) in enumerate(zip(_DIFFUSE_ANGLES, weights)):
            radians = math.radians(angle)
            sine, cosine = math.sin(radians), math.cos(radians)
            direction = (normal[0] * cosine - normal[1] * sine,
                         normal[0] * sine + normal[1] * cosine)
            # The final ray takes the remaining budget, avoiding branch drift.
            energy = diffuse_energy - spent if index == len(weights) - 1 else diffuse_energy * weight / total
            spent += energy
            if _dot(direction, normal) <= 0:
                raise ArithmeticError("Diffuse direction left the incident-facing hemisphere")
            rays.append((direction, energy, "diffuse"))
    return rays


def trace_scattering(objects, surfaces, origin, incoming, width, height,
                     initial_surface_id, max_bounces=3):
    """Trace at most seven hits and 42 ray segments, without any randomness.

    Object bboxes/surface endpoints are normalized; origin and output endpoints
    are in pixels. ``incoming`` is a nonzero pixel-space direction (y down), not
    a source position. The first supplied impact has normalized energy 1.0.
    Bounce numbers start at 1. At most two strongest hit rays per node continue;
    all other hit rays retain their energy in ``untraced_energy`` accounting.
    ``max_bounces`` must be an integer from 1 through 3.
    """
    width, height = _number(width, "width"), _number(height, "height")
    if not (0 < width <= 1_000_000 and 0 < height <= 1_000_000):
        raise ValueError("width and height must be positive and at most 1,000,000 pixels")
    if isinstance(max_bounces, bool) or not isinstance(max_bounces, int) or not 1 <= max_bounces <= 3:
        raise ValueError("max_bounces must be an integer from 1 through 3")
    epsilon = max(1e-7, math.hypot(width, height) * 1e-9)
    tolerance = max(1e-5, math.hypot(width, height) * 1e-7)
    object_map = _validate_objects(objects)
    surface_map = _validate_surfaces(surfaces, object_map, width, height, epsilon)
    initial_surface_id = _identifier(initial_surface_id, "initial_surface_id")
    if initial_surface_id not in surface_map:
        raise ValueError("initial_surface_id does not identify a supplied segment")
    origin = _point(origin, "origin")
    if not (0 <= origin[0] <= width and 0 <= origin[1] <= height):
        raise ValueError("origin must lie inside the image frame")
    incoming = _unit(_point(incoming, "incoming"), "incoming")
    initial_surface = surface_map[initial_surface_id]
    projected_origin = _project_to_segment(origin, initial_surface)
    if math.dist(origin, projected_origin) > tolerance:
        raise ValueError("The supplied first impact must lie on initial_surface_id; bbox geometry is not inferred")
    # Remove only subpixel numerical drift from the explicitly supplied impact.
    origin = projected_origin
    segments, hits = [], []
    totals = {"absorbed_energy": 0.0, "escaped_energy": 0.0,
              "branch_limited_energy": 0.0, "depth_limited_energy": 0.0}

    def visit(point, direction, surface, energy_in, bounce):
        material = object_map[surface["object_id"]]
        normal = _normal(surface, direction)
        reflected = energy_in * material["reflectance"]
        absorbed = energy_in - reflected
        totals["absorbed_energy"] += absorbed
        rays = _outgoing(direction, normal, reflected, material["specular_fraction"])
        hits.append({"point": list(point), "object_id": surface["object_id"], "surface_id": surface["id"],
                     "bounce": bounce, "energy_in": energy_in, "reflected_energy": reflected,
                     "absorbed_energy": absorbed, "drawn_energy": math.fsum(ray[1] for ray in rays),
                     "normal": list(normal), "incoming_direction": list(direction)})
        continuations = []
        for ray_index, (outgoing, energy, kind) in enumerate(rays):
            boundary = _frame_distance(point, outgoing, width, height)
            nearest_surface, nearest_distance = None, boundary
            for candidate in surface_map.values():
                distance = _ray_segment_distance(point, outgoing, candidate, epsilon)
                if distance is None or distance > boundary + epsilon:
                    continue
                if nearest_surface is None or distance < nearest_distance - epsilon or (
                    abs(distance - nearest_distance) <= epsilon and candidate["id"] < nearest_surface["id"]
                ):
                    nearest_surface, nearest_distance = candidate, distance
            endpoint = (min(width, max(0.0, point[0] + nearest_distance * outgoing[0])),
                        min(height, max(0.0, point[1] + nearest_distance * outgoing[1])))
            segment = {"a": list(point), "b": list(endpoint), "energy": energy, "bounce": bounce,
                       "kind": kind, "from_object": surface["object_id"],
                       "to_object": nearest_surface["object_id"] if nearest_surface else None,
                       "hit": nearest_surface is not None,
                       "termination": "pending" if nearest_surface else "frame_boundary"}
            segments.append(segment)
            if nearest_surface is None:
                totals["escaped_energy"] += energy
            else:
                continuations.append((energy, ray_index, endpoint, outgoing, nearest_surface, segment))
        continuations.sort(key=lambda item: (-item[0], item[1]))
        for index, (energy, _, endpoint, outgoing, next_surface, segment) in enumerate(continuations):
            if bounce >= max_bounces:
                segment["termination"] = "depth_limit"
                totals["depth_limited_energy"] += energy
            elif index >= 2:
                segment["termination"] = "branch_limit"
                totals["branch_limited_energy"] += energy
            else:
                segment["termination"] = "continued"
                visit(endpoint, outgoing, next_surface, energy, bounce + 1)

    visit(origin, incoming, initial_surface, 1.0, 1)
    accounted = math.fsum(totals.values())
    if not math.isclose(accounted, 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ArithmeticError(f"Scattering energy bookkeeping failed: {accounted}")
    return {"segments": segments, "hits": hits, "assumptions": ASSUMPTIONS,
            "max_bounces": max_bounces, "initial_surface_id": initial_surface_id,
            "energy_accounting": {"input_energy": 1.0, **totals,
                                  "untraced_energy": totals["branch_limited_energy"] + totals["depth_limited_energy"],
                                  "accounted_energy": accounted},
            "geometry_epsilon_px": epsilon, "initial_point_tolerance_px": tolerance}
