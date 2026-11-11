"""Focused geometry/energy checks; no models, inference, or external services."""
import copy
import json
import math
import random
import unittest

from scattering import trace_scattering


def material(identifier, reflectance=(0.6, 0.8), specular=0.3):
    return {"id": identifier, "label": identifier, "material": "assumed",
            "bbox": [0, 0, 1, 1], "reflectance": list(reflectance), "specular_fraction": specular}


def surface(identifier, owner, a, b):
    return {"id": identifier, "object_id": owner, "a": a, "b": b}


class ScatteringTests(unittest.TestCase):
    def test_omitted_specular_fraction_matches_documented_default(self):
        default=material('floor',specular=.05)
        omitted={key:value for key,value in default.items() if key!='specular_fraction'}
        geometry=[surface('floor','floor',[0,.5],[1,.5])]
        expected=trace_scattering([default],geometry,[50,50],[0,1],100,100,'floor')
        actual=trace_scattering([omitted],geometry,[50,50],[0,1],100,100,'floor')
        self.assertEqual(actual,expected)

    def test_specular_reflection_angle_and_absorption(self):
        result = trace_scattering([material("floor", (0.5, 0.5), 1)],
            [surface("floor-line", "floor", [0.1, 0.5], [0.9, 0.5])],
            [50, 50], [1, 1], 100, 100, "floor-line")
        self.assertEqual(len(result["segments"]), 1)
        ray = result["segments"][0]
        self.assertEqual(ray["kind"], "specular")
        self.assertAlmostEqual(ray["b"][0], 100)
        self.assertAlmostEqual(ray["b"][1], 0)
        self.assertAlmostEqual(ray["energy"], 0.5)
        self.assertAlmostEqual(result["energy_accounting"]["absorbed_energy"], 0.5)
        self.assertAlmostEqual(result["energy_accounting"]["escaped_energy"], 0.5)

    def test_nearest_surface_stops_ray_before_farther_surface(self):
        objects = [material(name, (1, 1), 1) for name in ("floor", "near", "far")]
        surfaces = [surface("floor", "floor", [0, 0.8], [1, 0.8]),
                    surface("far", "far", [0, 0.2], [1, 0.2]),
                    surface("near", "near", [0, 0.5], [1, 0.5])]
        result = trace_scattering(objects, surfaces, [50, 80], [0, 1], 100, 100, "floor", 1)
        ray = result["segments"][0]
        self.assertEqual(ray["to_object"], "near")
        self.assertEqual(ray["b"], [50, 50])
        self.assertEqual(ray["termination"], "depth_limit")

    def test_another_surface_of_same_object_is_not_skipped(self):
        result = trace_scattering([material("object", (1, 1), 1)],
            [surface("a", "object", [0, 0.8], [1, 0.8]),
             surface("b", "object", [0, 0.5], [1, 0.5])],
            [50, 80], [0, 1], 100, 100, "a", 1)
        self.assertEqual(result["segments"][0]["to_object"], "object")
        self.assertEqual(result["segments"][0]["b"], [50, 50])

    def test_branches_conserve_energy_and_remain_bounded(self):
        objects = [material("room")]
        surfaces = [surface("bottom", "room", [0.1, 0.8], [0.9, 0.8]),
                    surface("top", "room", [0.1, 0.1], [0.9, 0.1]),
                    surface("left", "room", [0.1, 0.1], [0.1, 0.8]),
                    surface("right", "room", [0.9, 0.1], [0.9, 0.8])]
        result = trace_scattering(objects, surfaces, [50, 80], [0.4, 1], 100, 100, "bottom")
        self.assertLessEqual(len(result["hits"]), 7)
        self.assertLessEqual(len(result["segments"]), 42)
        self.assertEqual(max(hit["bounce"] for hit in result["hits"]), 3)
        for hit in result["hits"]:
            self.assertAlmostEqual(hit["reflected_energy"] + hit["absorbed_energy"], hit["energy_in"])
            self.assertAlmostEqual(hit["drawn_energy"], hit["reflected_energy"])
            self.assertLessEqual(hit["reflected_energy"], hit["energy_in"])
            self.assertLessEqual(sum(a * b for a, b in zip(hit["normal"], hit["incoming_direction"])), 1e-12)
        accounting = result["energy_accounting"]
        self.assertGreater(accounting["branch_limited_energy"], 0)
        self.assertGreater(accounting["depth_limited_energy"], 0)
        self.assertAlmostEqual(accounting["absorbed_energy"] + accounting["escaped_energy"] + accounting["untraced_energy"], 1)
        for ray in result["segments"]:
            self.assertTrue(all(0 <= coordinate <= 100 for coordinate in ray["b"]))
        json.dumps(result, allow_nan=False)

    def test_diffuse_rays_face_incident_hemisphere_and_cosine_weights(self):
        # Reversing the segment endpoints must not flip the illuminated side.
        result = trace_scattering([material("floor", (1, 1), 0)],
            [surface("s", "floor", [0.9, 0.5], [0.1, 0.5])],
            [50, 50], [0, 1], 100, 100, "s")
        rays = result["segments"]
        self.assertEqual(len(rays), 5)
        normal = result["hits"][0]["normal"]
        for ray in rays:
            delta = [b - a for a, b in zip(ray["a"], ray["b"])]
            cosine = sum(d * n for d, n in zip(delta, normal)) / math.hypot(*delta)
            self.assertGreaterEqual(cosine, math.cos(math.radians(70)) - 1e-12)
            self.assertLess(ray["b"][1], ray["a"][1])
        self.assertAlmostEqual(rays[0]["energy"], rays[4]["energy"])
        self.assertAlmostEqual(rays[1]["energy"], rays[3]["energy"])
        self.assertGreater(rays[2]["energy"], rays[1]["energy"])
        self.assertAlmostEqual(sum(ray["energy"] for ray in rays), 1)

    def test_boxes_do_not_create_geometry_or_randomness(self):
        objects = [material("floor", (1, 1), 1), material("annotation-only")]
        objects[1]["bbox"] = [0.4, 0.2, 0.6, 0.4]
        surfaces = [surface("s", "floor", [0.1, 0.8], [0.9, 0.8])]
        before = random.getstate()
        result = trace_scattering(objects, surfaces, [50, 80], [0, 1], 100, 100, "s")
        self.assertEqual(result, trace_scattering(objects, surfaces, [50, 80], [0, 1], 100, 100, "s"))
        self.assertEqual(before, random.getstate())
        self.assertIsNone(result["segments"][0]["to_object"])
        self.assertEqual(result["segments"][0]["b"], [50, 0])

    def test_invalid_coefficients_geometry_and_depth_fail(self):
        objects = [material("floor")]
        surfaces = [surface("s", "floor", [0.1, 0.5], [0.9, 0.5])]
        for key, bad in (("reflectance", [math.nan, 1]), ("reflectance", [-0.1, 0.5]),
                         ("reflectance", [0.8, 0.2]), ("specular_fraction", 1.1),
                         ("specular_fraction", math.inf), ("bbox", [0, 0, 1.1, 1])):
            changed = copy.deepcopy(objects)
            changed[0][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                trace_scattering(changed, surfaces, [50, 50], [0, 1], 100, 100, "s")
        for origin, incoming in (([50, 49], [0, 1]), ([math.nan, 50], [0, 1]), ([50, 50], [0, 0])):
            with self.assertRaises(ValueError):
                trace_scattering(objects, surfaces, origin, incoming, 100, 100, "s")
        for depth in (0, 4, 2.5, True):
            with self.assertRaises(ValueError):
                trace_scattering(objects, surfaces, [50, 50], [0, 1], 100, 100, "s", depth)
        bad_surfaces = copy.deepcopy(surfaces)
        bad_surfaces[0]["a"] = [math.nan, 0.5]
        with self.assertRaises(ValueError):
            trace_scattering(objects, bad_surfaces, [50, 50], [0, 1], 100, 100, "s")


if __name__ == "__main__":
    unittest.main()
