"""Geospatial helpers: great-circle distance and a named-location world map."""
import math


def haversine_km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


class World:
    def __init__(self, locations):
        self.by_id = {l["id"]: l for l in locations}
        self.by_name = {l["title"].casefold(): l for l in locations}

    @classmethod
    def from_docs(cls, docs):
        return cls([d for d in docs if d.get("kind") == "location" and d.get("coords")])

    def names(self):
        return [l["title"] for l in self.by_id.values()]

    def resolve(self, name):
        """Map a free-text place name to a canonical location title, or None."""
        if not name:
            return None
        n = name.casefold().strip()
        if n in self.by_name:
            return self.by_name[n]["title"]
        for key, loc in self.by_name.items():
            if n in key or key in n:
                return loc["title"]
        return None

    def coords(self, name):
        r = self.resolve(name)
        return tuple(self.by_name[r.casefold()]["coords"]) if r else None

    def coords_by_id(self, loc_id):
        l = self.by_id.get(loc_id)
        return tuple(l["coords"]) if l else None

    def distance_km(self, a, b) -> float:
        ca, cb = self.coords(a), self.coords(b)
        return haversine_km(ca, cb) if ca and cb else 0.0
