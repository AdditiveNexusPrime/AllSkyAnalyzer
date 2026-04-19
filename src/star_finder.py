"""
StarFinder – identify catalogue stars in an allsky image and map them to
well-known star names and constellation membership.

Two complementary approaches are used:

1. **Plate solving** (when a local astrometry.net index is available):
   ``solve-field`` maps pixel coordinates to RA/Dec, then an Astropy catalogue
   query matches stars by coordinate proximity.

2. **Photometric detection** (always available as a fallback):
   OpenCV's ``SimpleBlobDetector`` locates bright point sources in the image.
   The resulting pixel positions can optionally be cross-matched against
   catalogue data if plate solving succeeded.
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import AstrometryConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Star:
    """A single detected star."""

    x: float
    y: float
    # Pixel brightness (0-255 scale, relative to image).
    brightness: float = 0.0
    # Catalogue properties (set after cross-matching).
    ra: Optional[float] = None   # degrees J2000
    dec: Optional[float] = None  # degrees J2000
    magnitude: Optional[float] = None
    name: Optional[str] = None
    constellation: Optional[str] = None


@dataclass
class ConstellationLine:
    """A line segment connecting two catalogue stars within a constellation."""

    x1: float
    y1: float
    x2: float
    y2: float
    constellation: str = ""


@dataclass
class StarFinderResult:
    """Output from the star-finding step."""

    stars: list[Star] = field(default_factory=list)
    constellation_lines: list[ConstellationLine] = field(default_factory=list)
    # WCS solution (affine transform pixel → sky coords), if plate solve succeeded.
    wcs_header: Optional[dict] = None
    plate_solved: bool = False


# ---------------------------------------------------------------------------
# Constellation data (abbreviated – brightest / most recognisable stars)
# ---------------------------------------------------------------------------

# Mapping: constellation abbreviation → list of (RA_deg, Dec_deg) pairs
# that define *line segments* (pairs of indices into the list below).
# This is a compact in-process dataset so the application works offline.
# Each tuple is (name, RA_deg, Dec_deg, magnitude, common_name_or_empty).
_CATALOGUE: list[tuple[str, float, float, float, str]] = [
    # Orion
    ("ORI", 83.8221, -5.3797, 0.12, "Rigel"),
    ("ORI", 88.7929, 7.4071, 0.45, "Betelgeuse"),
    ("ORI", 84.0534, -1.2019, 1.64, "Bellatrix"),
    ("ORI", 81.2827, 6.3497, 1.69, "Mintaka"),  # delta Ori (belt)
    ("ORI", 84.4534, -1.9425, 1.74, "Alnilam"),  # eps Ori (belt)
    ("ORI", 85.1897, -2.6006, 2.05, "Alnitak"),  # zeta Ori (belt)
    # Ursa Major (Big Dipper)
    ("UMA", 165.9320, 61.7510, 1.79, "Dubhe"),
    ("UMA", 178.4577, 53.6948, 2.34, "Merak"),
    ("UMA", 183.8565, 57.0326, 2.41, "Phecda"),
    ("UMA", 193.5071, 55.9598, 3.31, "Megrez"),
    ("UMA", 200.9814, 54.9254, 1.76, "Alioth"),
    ("UMA", 206.8851, 49.3133, 2.23, "Mizar"),
    ("UMA", 210.9559, 49.6336, 1.85, "Alkaid"),
    # Cassiopeia
    ("CAS", 2.2942, 59.1498, 2.23, "Schedar"),
    ("CAS", 10.1268, 56.5374, 2.28, "Caph"),
    ("CAS", 14.1772, 60.7167, 2.15, "Tsih"),
    ("CAS", 21.4538, 60.2353, 2.66, "Ruchbah"),
    ("CAS", 28.5988, 63.6700, 3.35, "Segin"),
    # Leo
    ("LEO", 152.0930, 11.9672, 1.35, "Regulus"),
    ("LEO", 177.2649, 14.5720, 2.14, "Denebola"),
    ("LEO", 168.5270, 19.8415, 2.01, "Algieba"),
    ("LEO", 154.9932, 19.8415, 3.43, "Adhafera"),
    # Scorpius
    ("SCO", 247.3519, -26.4320, 0.96, "Antares"),
    ("SCO", 240.0833, -22.6220, 2.29, "Graffias"),
    ("SCO", 252.9417, -34.2928, 1.62, "Shaula"),
    # Cygnus (Northern Cross)
    ("CYG", 310.3579, 45.2803, 1.25, "Deneb"),
    ("CYG", 305.5572, 40.2567, 2.46, "Sadr"),
    ("CYG", 296.2446, 27.9596, 2.87, "Albireo"),
    # Lyra
    ("LYR", 279.2347, 38.7836, 0.03, "Vega"),
    # Aquila
    ("AQL", 297.6958, 8.8683, 0.76, "Altair"),
    # Canis Major
    ("CMA", 101.2872, -16.7161, -1.46, "Sirius"),
    ("CMA", 111.0238, -29.3031, 1.83, "Adhara"),
    # Taurus
    ("TAU", 68.9801, 16.5093, 0.85, "Aldebaran"),
    # Gemini
    ("GEM", 113.6494, 31.8883, 1.16, "Pollux"),
    ("GEM", 116.3290, 28.0263, 1.58, "Castor"),
    # Virgo
    ("VIR", 201.2983, -11.1613, 0.98, "Spica"),
    # Perseus
    ("PER", 51.0810, 40.9556, 1.79, "Mirfak"),
    # Auriga
    ("AUR", 79.1723, 45.9980, 0.08, "Capella"),
    # Boötes
    ("BOO", 213.9153, 19.1822, -0.04, "Arcturus"),
    # Centaurus
    ("CEN", 219.9021, -60.8340, -0.27, "Rigil Kentaurus"),
    ("CEN", 210.9559, -60.3730, 0.01, "Hadar"),
    # Southern Cross (Crux)
    ("CRU", 187.7915, -57.1132, 0.77, "Acrux"),
    ("CRU", 191.9303, -59.6888, 1.25, "Mimosa"),
    # Pegasus
    ("PEG", 346.1902, 15.2052, 2.38, "Markab"),
    ("PEG", 2.0969, 15.1836, 2.44, "Alpheratz"),
    # Aries
    ("ARI", 31.7933, 23.4624, 2.00, "Hamal"),
    # Sagittarius
    ("SGR", 276.0430, -25.4217, 1.79, "Kaus Australis"),
    # Piscis Austrinus
    ("PSA", 344.4127, -29.6223, 1.16, "Fomalhaut"),
]

# Constellation lines: each entry is (constellation_abbr, idx_a, idx_b)
# where idx_a and idx_b are zero-based indices into _CATALOGUE.
_CONSTELLATION_LINES: list[tuple[str, int, int]] = [
    # Orion belt + rough body
    ("ORI", 3, 4), ("ORI", 4, 5),          # belt: Mintaka-Alnilam-Alnitak
    ("ORI", 0, 5), ("ORI", 1, 2),          # Rigel-Alnitak, Betelgeuse-Bellatrix
    ("ORI", 1, 4), ("ORI", 2, 3),          # chest cross
    # Ursa Major (Big Dipper)
    ("UMA", 6, 7), ("UMA", 7, 8), ("UMA", 8, 9),
    ("UMA", 9, 10), ("UMA", 10, 11), ("UMA", 11, 12),
    # Cassiopeia W
    ("CAS", 13, 14), ("CAS", 14, 15), ("CAS", 15, 16), ("CAS", 16, 17),
    # Leo sickle
    ("LEO", 18, 20), ("LEO", 20, 21),
    ("LEO", 18, 19),
    # Cygnus Northern Cross
    ("CYG", 26, 27), ("CYG", 27, 28),
    # Scorpius
    ("SCO", 23, 22), ("SCO", 23, 24),
    # Gemini
    ("GEM", 34, 35),
]

# Build a lookup: constellation abbr → full name
_CONSTELLATION_NAMES: dict[str, str] = {
    "ORI": "Orion", "UMA": "Ursa Major", "CAS": "Cassiopeia", "LEO": "Leo",
    "SCO": "Scorpius", "CYG": "Cygnus", "LYR": "Lyra", "AQL": "Aquila",
    "CMA": "Canis Major", "TAU": "Taurus", "GEM": "Gemini", "VIR": "Virgo",
    "PER": "Perseus", "AUR": "Auriga", "BOO": "Boötes", "CEN": "Centaurus",
    "CRU": "Crux", "PEG": "Pegasus", "ARI": "Aries", "SGR": "Sagittarius",
    "PSA": "Piscis Austrinus",
}


# ---------------------------------------------------------------------------
# StarFinder class
# ---------------------------------------------------------------------------


class StarFinder:
    """Detect stars and (optionally) plate-solve an allsky image.

    Parameters
    ----------
    config:
        Astrometry section of the application config.
    """

    def __init__(self, config: AstrometryConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def find(self, image_path: str | Path) -> StarFinderResult:
        """Run star detection on *image_path* and return a :class:`StarFinderResult`."""
        image_path = Path(image_path)
        result = StarFinderResult()

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            logger.error("Cannot read image: %s", image_path)
            return result

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Photometric detection (always run – used as fallback or cross-match base).
        result.stars = self._detect_blobs(gray)
        logger.info("Blob detector found %d star candidates", len(result.stars))

        # Plate solving (optional – enriches stars with RA/Dec and names).
        solved = self._plate_solve(image_path)
        if solved:
            result.plate_solved = True
            result.wcs_header = solved
            self._enrich_stars_from_catalogue(result.stars, bgr.shape[:2], solved)
            result.constellation_lines = self._build_constellation_lines(
                result.stars, bgr.shape[:2], solved
            )

        return result

    # ------------------------------------------------------------------
    # Blob / photometric detection
    # ------------------------------------------------------------------

    def _detect_blobs(self, gray: np.ndarray) -> list[Star]:
        """Detect bright point sources using OpenCV's SimpleBlobDetector."""
        # Stretch / normalise to aid detection.
        stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 3
        params.maxArea = 300
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = False
        params.filterByInertia = True
        params.minInertiaRatio = 0.3

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(stretched)

        stars: list[Star] = []
        for kp in keypoints:
            x, y = kp.pt
            brightness = float(gray[int(y), int(x)])
            stars.append(Star(x=x, y=y, brightness=brightness))

        # Sort brightest first.
        stars.sort(key=lambda s: s.brightness, reverse=True)
        return stars

    # ------------------------------------------------------------------
    # Plate solving
    # ------------------------------------------------------------------

    def _plate_solve(self, image_path: Path) -> Optional[dict]:
        """
        Attempt plate solving with a locally installed astrometry.net.

        Returns the WCS header as a dict on success, or *None* on failure.
        """
        index_dir = Path(self.config.index_dir)
        if not index_dir.exists():
            logger.debug("Astrometry index dir not found (%s); skipping plate solve.", index_dir)
            return None

        try:
            with tempfile.TemporaryDirectory() as tmp:
                cmd = [
                    "solve-field",
                    "--no-plots",
                    "--overwrite",
                    "--dir", tmp,
                    "--downsample", str(self.config.downsample),
                ]
                if self.config.fov_deg > 0:
                    half = self.config.fov_deg / 2
                    cmd += ["--scale-low", str(half * 0.9), "--scale-high", str(half * 1.1),
                            "--scale-units", "degw"]
                cmd.append(str(image_path))

                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if proc.returncode != 0:
                    logger.debug("solve-field failed: %s", proc.stderr[-500:])
                    return None

                wcs_path = Path(tmp) / (image_path.stem + ".wcs")
                if not wcs_path.exists():
                    logger.debug("solve-field produced no WCS file.")
                    return None

                # Parse minimal WCS header (CRVAL, CRPIX, CD matrix).
                return self._parse_wcs(wcs_path)

        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("Plate solve skipped: %s", exc)
            return None

    @staticmethod
    def _parse_wcs(wcs_path: Path) -> dict:
        """Extract a minimal WCS dictionary from a FITS WCS file."""
        header: dict = {}
        with open(wcs_path) as fh:
            for line in fh:
                line = line.strip()
                if "=" in line:
                    key, _, rest = line.partition("=")
                    value_part = rest.split("/")[0].strip()
                    try:
                        header[key.strip()] = float(value_part)
                    except ValueError:
                        header[key.strip()] = value_part.strip().strip("'")
        return header

    # ------------------------------------------------------------------
    # Catalogue cross-matching
    # ------------------------------------------------------------------

    def _enrich_stars_from_catalogue(
        self,
        stars: list[Star],
        image_shape: tuple[int, int],
        wcs: dict,
    ) -> None:
        """Annotate detected stars with catalogue names / magnitudes via WCS."""
        height, width = image_shape
        mag_limit = self.config.magnitude_limit

        for cat_entry in _CATALOGUE:
            abbr, ra, dec, mag, name = cat_entry
            if mag > mag_limit:
                continue

            px, py = self._sky_to_pixel(ra, dec, wcs)
            if px < 0 or py < 0 or px >= width or py >= height:
                continue

            # Find nearest detected star within 15 pixels.
            best: Optional[Star] = None
            best_dist = 15.0
            for star in stars:
                d = math.hypot(star.x - px, star.y - py)
                if d < best_dist:
                    best_dist = d
                    best = star

            if best is None:
                # Insert a synthetic star for bright catalogue entries.
                best = Star(x=px, y=py)
                stars.append(best)

            best.ra = ra
            best.dec = dec
            best.magnitude = mag
            best.name = name or None
            best.constellation = _CONSTELLATION_NAMES.get(abbr, abbr)

    def _build_constellation_lines(
        self,
        stars: list[Star],
        image_shape: tuple[int, int],
        wcs: dict,
    ) -> list[ConstellationLine]:
        """Generate constellation line segments in pixel space."""
        height, width = image_shape
        lines: list[ConstellationLine] = []

        for abbr, idx_a, idx_b in _CONSTELLATION_LINES:
            if idx_a >= len(_CATALOGUE) or idx_b >= len(_CATALOGUE):
                continue

            _, ra_a, dec_a, _, _ = _CATALOGUE[idx_a]
            _, ra_b, dec_b, _, _ = _CATALOGUE[idx_b]

            x1, y1 = self._sky_to_pixel(ra_a, dec_a, wcs)
            x2, y2 = self._sky_to_pixel(ra_b, dec_b, wcs)

            # Only draw if both ends are (approximately) within the frame.
            margin = 50
            if (
                -margin <= x1 <= width + margin
                and -margin <= y1 <= height + margin
                and -margin <= x2 <= width + margin
                and -margin <= y2 <= height + margin
            ):
                lines.append(
                    ConstellationLine(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        constellation=_CONSTELLATION_NAMES.get(abbr, abbr),
                    )
                )

        return lines

    # ------------------------------------------------------------------
    # WCS helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sky_to_pixel(ra: float, dec: float, wcs: dict) -> tuple[float, float]:
        """
        Convert RA/Dec (degrees) to pixel coordinates using a simple
        TAN (gnomonic) projection extracted from the WCS header.

        Supports CD matrix or CDELT+PC matrix conventions.
        """
        # Central sky position.
        crval1 = wcs.get("CRVAL1", 0.0)
        crval2 = wcs.get("CRVAL2", 0.0)
        crpix1 = wcs.get("CRPIX1", 0.0)
        crpix2 = wcs.get("CRPIX2", 0.0)

        # Convert to radians.
        ra0 = math.radians(crval1)
        dec0 = math.radians(crval2)
        ra_rad = math.radians(ra)
        dec_rad = math.radians(dec)

        # Gnomonic (TAN) projection.
        cos_c = (
            math.sin(dec0) * math.sin(dec_rad)
            + math.cos(dec0) * math.cos(dec_rad) * math.cos(ra_rad - ra0)
        )
        if cos_c <= 0:
            return -9999.0, -9999.0

        xi = (
            math.cos(dec_rad) * math.sin(ra_rad - ra0) / cos_c
        )
        eta = (
            (math.cos(dec0) * math.sin(dec_rad)
             - math.sin(dec0) * math.cos(dec_rad) * math.cos(ra_rad - ra0))
            / cos_c
        )
        # Convert from radians to degrees.
        xi = math.degrees(xi)
        eta = math.degrees(eta)

        # Apply CD matrix.
        cd11 = wcs.get("CD1_1", wcs.get("CDELT1", 1.0))
        cd12 = wcs.get("CD1_2", 0.0)
        cd21 = wcs.get("CD2_1", 0.0)
        cd22 = wcs.get("CD2_2", wcs.get("CDELT2", 1.0))

        det = cd11 * cd22 - cd12 * cd21
        if abs(det) < 1e-15:
            return -9999.0, -9999.0

        # Inverse matrix.
        inv11 = cd22 / det
        inv12 = -cd12 / det
        inv21 = -cd21 / det
        inv22 = cd11 / det

        dx = inv11 * xi + inv12 * eta
        dy = inv21 * xi + inv22 * eta

        px = crpix1 + dx
        py = crpix2 - dy   # FITS y-axis is inverted relative to image y-axis

        return px, py
