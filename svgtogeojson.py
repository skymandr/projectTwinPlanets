from dataclasses import dataclass
from typing import List, Optional, AnyStr
import math
import os

import numpy as np
import svg.path as spath


@dataclass
class Coord:
    x: float
    y: float


class AzimuthalCoord(Coord):
    @property
    def rho(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    @property
    def theta(self) -> float:
        return math.atan2(self.y, self.x)


class EqrecCoord(Coord):
    @property
    def lat(self) -> float:
        return self.y

    @property
    def lon(self) -> float:
        return self.x


class Extent(Coord):
    pass


def orthographic_to_eqrec(
    coord: Coord, extent: Extent, offset: Coord, border: float,
) ->  EqrecCoord:
    radius = (extent.y - border * 2) * 0.5
    if coord.x - offset.x <= extent.x * 0.5:
        origin = Coord(
            x=radius - offset.x + border,
            y=radius - offset.y + border,
        )
        eta = 0
    else:
        origin = Coord(
            x=extent.x - (radius + border) - offset.x,
            y=radius - offset.y + border,
        )
        eta = 1

    new_coord = AzimuthalCoord(x=coord.x - origin.x, y=coord.y - origin.y)

    rho = new_coord.rho / radius

    if rho > 1.0:
        return None

    lat = np.rad2deg(math.asin(new_coord.y / radius))
    lon = 90 + np.rad2deg(
        math.asin(new_coord.x / (radius * math.cos(np.deg2rad(lat))))
    ) + eta * 180

    return EqrecCoord(x=lon, y=lat)


def azimuthal_fakequidistant_to_eqrec(
    coord: Coord, extent: Extent, offset: Coord, border: float,
) ->  EqrecCoord:
    radius = (extent.y - border * 2) * 0.5
    if coord.x - offset.x <= extent.x * 0.5:
        origin = Coord(
            x=radius - offset.x + border,
            y=radius - offset.y + border,
        )
        eta = 0
    else:
        origin = Coord(
            x=extent.x - (radius + border) - offset.x,
            y=radius - offset.y + border,
        )
        eta = 1

    new_coord = AzimuthalCoord(x=coord.x - origin.x, y=coord.y - origin.y)

    rho = new_coord.rho / radius

    if rho > 1.0:
        return None

    x = math.cos(math.pi * 0.5 * rho)
    y = math.sin(math.pi * 0.5 * rho) * math.cos(new_coord.theta)
    z = math.sin(math.pi * 0.5 * rho) * math.sin(new_coord.theta)
    lat = np.rad2deg(math.atan2(z, math.sqrt(x * x + y * y)))
    lon = np.rad2deg(math.pi - math.atan2(x, y) + math.pi * eta)

    return EqrecCoord(x=lon, y=lat)


def points_to_eqrec(
    points: np.ndarray,
    extent: Extent,
    offset: Coord,
    border: float,
    orthographic: bool=False,
) -> List[EqrecCoord]:
    if orthographic:
        return [
            orthographic_to_eqrec(
                Coord(x=np.real(p), y=-np.imag(p)),
                extent,
                offset,
                border,
            ) for p in points
        ]
    return [
        azimuthal_fakequidistant_to_eqrec(
            Coord(x=np.real(p), y=-np.imag(p)),
            extent,
            offset,
            border,
        ) for p in points
    ]


def path_to_points(
    path: spath.Path, n: int=10
) -> List[complex]:
    n = int(math.ceil(path.length() * n))
    points = [path.point(t) for t in np.linspace(0, 1, n)]
    return points


def parse_d(d: AnyStr) -> List[spath.Path]:
    if 'm' in d or 'z' in d:
        raise ValueError("Path must use absolute path operators!")

    return [spath.parse_path(p) for p in d.split("Z")]


def process_template(
    template: dict,
    path_dir: str="paths",
    n_points: int=10,
    invert: bool=False,
    orthographic: bool=False,
) -> dict:
    extent = Extent(
        x=template.get("properties", dict()).get("extent", dict()).get("x", 0),
        y=template.get("properties", dict()).get("extent", dict()).get("y", 0),
    )
    offset = Coord(
        x=template.get("properties", dict()).get("offset", dict()).get("x", 0),
        y=template.get("properties", dict()).get("offset", dict()).get("y", 0),
    )
    border = template.get("properties", dict()).get("border", 0)

    for feature in template.get("features", []):
        print(feature["properties"]["name"])
        try:
            with open(
                os.path.join(path_dir, feature["properties"]["paths"]), 'r'
            ) as fp:
                path_string = fp.read()
        except FileNotFoundError as err:
            print(err)
            continue
        geometries = []
        for path in parse_d(path_string):
            coordinates = [
                [coord.lon, coord.lat]
                for coord in points_to_eqrec(
                    path_to_points(path, n=n_points),
                    extent,
                    offset,
                    border,
                    orthographic,
                ) if coord is not None
            ]
            if invert:
                coordinates = coordinates[::-1]
            if feature["geometry"]["type"] in ["Polygon", "MultiPolygon"]:
                if len(coordinates) > 2:
                    coordinates.append(coordinates[0])
                    geometries.append(coordinates)
            elif len(coordinates) > 0:
                geometries.append(coordinates)
        feature["geometry"]["coordinates"] = [geometries, ]

    return template


def export_geojson(
    template: dict,
    filename: str,
    export_dir: str="geojson",
    indent: Optional[int]=0,
    centre_zero: bool=False,
) -> None:
    import datetime as dt
    import json

    for popit in ("extent", "offset", "border"):
        try:
            template["properties"].pop(popit)
        except KeyError:
            continue

    template["properties"]["updated"] = dt.datetime.utcnow().isoformat()

    for feature in template.get("features", []):
        for popit in ("paths", ):
            try:
                feature["properties"].pop(popit)
            except KeyError:
                continue

        if centre_zero:
            geometry = feature["geometry"]
            for i, _ in enumerate(geometry["coordinates"]):
                for j, _ in enumerate(geometry["coordinates"][i]):
                    for k, c in enumerate(geometry["coordinates"][i][j]):
                        # This is wrong, but...
                        geometry["coordinates"][i][j][k][0] -= 180
                        # ... not ready for this:
                        # if c[0] != 180:
                        #     geometry["coordinates"][i][j][k][0] = \
                        #         (c[0] + 180) % 360 - 180

    with open(os.path.join(export_dir, filename), 'w') as fp:
        json.dump(template, fp, indent=indent)


def plot_template(template: dict) -> None:
    from matplotlib import pyplot as plt
    for feature in template.get("features", []):
        subfeatures = feature["geometry"].get("coordinates", [])
        for coords in subfeatures:
            for c in coords:
                if len(c) > 0:
                    c = np.array(c)
                    print(f"{feature['properties']['name']}: {c.shape}")
                    plt.plot(c[:, 0], c[:, 1])
