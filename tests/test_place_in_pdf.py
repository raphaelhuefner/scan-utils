from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw


SCRIPT_PATH = Path(__file__).parents[1] / "place-in-pdf.py"
SPEC = importlib.util.spec_from_file_location("place_in_pdf", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
place_in_pdf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = place_in_pdf
SPEC.loader.exec_module(place_in_pdf)


class PlaceInPdfTests(unittest.TestCase):
    def test_places_images_at_physical_size_and_starts_new_pages_as_needed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.png"
            second = root / "second.jpg"
            third = root / "third.png"
            output = root / "output.pdf"
            create_image(first, (400, 400), dpi=(100, 100))
            create_image(second, (400, 400), dpi=(100, 100))
            create_image(third, (400, 400), dpi=(100, 100))

            self.assertEqual(run_main(["--output", str(output), str(root)]), 0)

            with pymupdf.open(output) as document:
                self.assertEqual(document.page_count, 2)
                first_page_images = document[0].get_image_info()
                second_page_images = document[1].get_image_info()
                self.assertEqual(len(first_page_images), 2)
                self.assertEqual(len(second_page_images), 1)
                first_box = pymupdf.Rect(first_page_images[0]["bbox"])
                second_box = pymupdf.Rect(first_page_images[1]["bbox"])
                third_box = pymupdf.Rect(second_page_images[0]["bbox"])
                self.assertAlmostEqual(first_box.width, 288, delta=0.01)
                self.assertAlmostEqual(first_box.height, 288, delta=0.01)
                self.assertAlmostEqual(
                    first_box.x0, (document[0].rect.width - 288) / 2, delta=0.01
                )
                self.assertAlmostEqual(
                    second_box.y0 - first_box.y1,
                    place_in_pdf.mm_to_points(5),
                    delta=0.01,
                )
                self.assertAlmostEqual(
                    third_box.y0, place_in_pdf.mm_to_points(10), delta=0.01
                )

    def test_letter_page_size_is_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            output = root / "output.pdf"
            create_image(source, (100, 100), dpi=(100, 100))

            self.assertEqual(
                run_main(
                    ["--output", str(output), "--page-size", "letter", str(source)]
                ),
                0,
            )

            with pymupdf.open(output) as document:
                self.assertAlmostEqual(document[0].rect.width, 612, delta=0.01)
                self.assertAlmostEqual(document[0].rect.height, 792, delta=0.01)

    def test_non_square_dpi_is_used_per_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            output = root / "output.pdf"
            create_image(source, (200, 300), dpi=(100, 150))

            self.assertEqual(run_main(["--output", str(output), str(source)]), 0)

            with pymupdf.open(output) as document:
                box = pymupdf.Rect(document[0].get_image_info()[0]["bbox"])
                self.assertAlmostEqual(box.width, 144, delta=0.01)
                self.assertAlmostEqual(box.height, 144, delta=0.02)

    def test_exif_rotation_is_honored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            output = root / "output.pdf"
            image = Image.new("RGB", (300, 600), "red")
            ImageDraw.Draw(image).rectangle((0, 300, 299, 599), fill="blue")
            exif = image.getexif()
            exif[place_in_pdf.EXIF_ORIENTATION_TAG] = 6
            image.save(source, dpi=(300, 300), exif=exif, quality=95)

            self.assertEqual(run_main(["--output", str(output), str(source)]), 0)

            with pymupdf.open(output) as document:
                page = document[0]
                box = pymupdf.Rect(page.get_image_info()[0]["bbox"])
                self.assertAlmostEqual(box.width, 144, delta=0.01)
                self.assertAlmostEqual(box.height, 72, delta=0.01)
                pixels = page.get_pixmap()
                middle_y = int((box.y0 + box.y1) / 2)
                left = pixels.pixel(int(box.x0 + box.width * 0.2), middle_y)
                right = pixels.pixel(int(box.x0 + box.width * 0.8), middle_y)
                self.assertGreater(left[2], left[0])
                self.assertGreater(right[0], right[2])

    def test_duplicate_source_is_rejected_unless_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            rejected_output = root / "rejected.pdf"
            allowed_output = root / "allowed.pdf"
            create_image(source, (100, 100), dpi=(100, 100))

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = run_main(
                    ["--output", str(rejected_output), str(source), str(source)]
                )
            self.assertEqual(result, 1)
            self.assertIn("duplicate source image", stderr.getvalue())
            self.assertFalse(rejected_output.exists())

            self.assertEqual(
                run_main(
                    [
                        "--output",
                        str(allowed_output),
                        "--allow-duplicates",
                        str(source),
                        str(source),
                    ]
                ),
                0,
            )
            with pymupdf.open(allowed_output) as document:
                self.assertEqual(len(document[0].get_image_info()), 2)

    def test_symlink_alias_is_rejected_as_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            alias = root / "alias.png"
            output = root / "output.pdf"
            create_image(source, (100, 100), dpi=(100, 100))
            alias.symlink_to(source)

            with redirect_stderr(io.StringIO()):
                result = run_main(
                    ["--output", str(output), str(source), str(alias)]
                )

            self.assertEqual(result, 1)
            self.assertFalse(output.exists())

    def test_mirrored_exif_orientation_is_honored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.jpg"
            output = root / "output.pdf"
            image = Image.new("RGB", (600, 300), "red")
            ImageDraw.Draw(image).rectangle((300, 0, 599, 299), fill="blue")
            exif = image.getexif()
            exif[place_in_pdf.EXIF_ORIENTATION_TAG] = 2
            image.save(source, dpi=(300, 300), exif=exif, quality=95)

            self.assertEqual(run_main(["--output", str(output), str(source)]), 0)

            with pymupdf.open(output) as document:
                page = document[0]
                box = pymupdf.Rect(page.get_image_info()[0]["bbox"])
                pixels = page.get_pixmap()
                middle_y = int((box.y0 + box.y1) / 2)
                left = pixels.pixel(int(box.x0 + box.width * 0.2), middle_y)
                right = pixels.pixel(int(box.x0 + box.width * 0.8), middle_y)
                self.assertGreater(left[2], left[0])
                self.assertGreater(right[0], right[2])

    def test_missing_dpi_is_atomic_and_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            valid = root / "valid.png"
            invalid = root / "invalid.png"
            rejected_output = root / "rejected.pdf"
            allowed_output = root / "allowed.pdf"
            create_image(valid, (100, 100), dpi=(100, 100))
            create_image(invalid, (100, 100))

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = run_main(
                    ["--output", str(rejected_output), str(valid), str(invalid)]
                )
            self.assertEqual(result, 1)
            self.assertIn("missing DPI metadata", stderr.getvalue())
            self.assertFalse(rejected_output.exists())

            self.assertEqual(
                run_main(
                    [
                        "--output",
                        str(allowed_output),
                        "--force-source-dpi",
                        "200",
                        str(invalid),
                    ]
                ),
                0,
            )
            self.assertTrue(allowed_output.exists())

    def test_oversized_image_is_rejected_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            output = root / "output.pdf"
            create_image(source, (1000, 1000), dpi=(100, 100))

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = run_main(["--output", str(output), str(source)])

            self.assertEqual(result, 1)
            self.assertIn("does not fit", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            output = root / "output.pdf"
            create_image(source, (100, 100), dpi=(100, 100))
            output.write_bytes(b"existing")

            with redirect_stderr(io.StringIO()):
                result = run_main(["--output", str(output), str(source)])

            self.assertEqual(result, 1)
            self.assertEqual(output.read_bytes(), b"existing")


def run_main(arguments: list[str]) -> int:
    with redirect_stdout(io.StringIO()):
        return place_in_pdf.main(arguments)


def create_image(
    path: Path,
    size: tuple[int, int],
    dpi: tuple[int, int] | None = None,
    orientation: int | None = None,
) -> None:
    image = Image.new("RGB", size, "white")
    options: dict[str, object] = {}
    if dpi is not None:
        options["dpi"] = dpi
    if orientation is not None:
        exif = image.getexif()
        exif[place_in_pdf.EXIF_ORIENTATION_TAG] = orientation
        options["exif"] = exif
    image.save(path, **options)


if __name__ == "__main__":
    unittest.main()
