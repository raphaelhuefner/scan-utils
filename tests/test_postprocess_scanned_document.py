from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_PATH = Path(__file__).parents[1] / "postprocess-scanned-document.py"
SPEC = importlib.util.spec_from_file_location("postprocess_scanned_document", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
postprocess = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = postprocess
SPEC.loader.exec_module(postprocess)


class PostprocessScannedDocumentTests(unittest.TestCase):
    def test_single_input_directory_defaults_output_directories_to_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scans = root / "scans"
            scans.mkdir()

            args = postprocess.parse_args([str(scans)])

            self.assertEqual(args.archive_dir, root / "aligned-original")
            self.assertEqual(args.email_dir, root / "email-images")
            self.assertEqual(args.email_jpeg_quality, 40)

    def test_explicit_output_directory_overrides_single_directory_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scans = root / "scans"
            archive = root / "custom-archive"
            scans.mkdir()

            args = postprocess.parse_args(
                ["--archive-dir", str(archive), str(scans)]
            )

            self.assertEqual(args.archive_dir, archive)
            self.assertEqual(args.email_dir, root / "email-images")

    def test_output_directories_are_required_for_non_directory_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "scan.jpg"
            source.touch()

            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                postprocess.parse_args([str(source)])

    def test_email_jpeg_quality_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scans = root / "scans"
            scans.mkdir()

            args = postprocess.parse_args(
                ["--email-jpeg-quality", "55", str(scans)]
            )

            self.assertEqual(args.email_jpeg_quality, 55)

    def test_directory_expansion_is_sorted_in_place_and_respects_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scans = root / "scans"
            skipped_directory = scans / "skip-me"
            skipped_directory.mkdir(parents=True)
            (scans / "b.PNG").touch()
            (scans / "a.jpg").touch()
            (skipped_directory / "c.jpeg").touch()
            explicit = root / "last.jpg"
            explicit.touch()

            sources = postprocess.discover_sources(
                [str(scans), str(explicit)],
                [postprocess.absolute_path(skipped_directory)],
            )

            self.assertEqual(
                [source.path for source in sources],
                [scans / "a.jpg", scans / "b.PNG", explicit],
            )

    def test_pipeline_deskews_crops_and_downscales(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "scan.jpg"
            archive = root / "archive"
            email = root / "email"
            create_scan(source, dpi=600)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = postprocess.main(
                    [
                        "--archive-dir",
                        str(archive),
                        "--email-dir",
                        str(email),
                        str(source),
                    ]
                )

            self.assertEqual(result, 0, stdout.getvalue())
            with Image.open(archive / "scan.jpg") as archive_image:
                archive_size = archive_image.size
                self.assertAlmostEqual(archive_image.info["dpi"][0], 600, delta=1)
                _, archive_angle = postprocess.detect_document_rectangle(
                    archive_image.convert("RGB")
                )
            with Image.open(email / "scan.jpg") as email_image:
                self.assertAlmostEqual(email_image.info["dpi"][0], 300, delta=1)
                self.assertEqual(email_image.size[0], round(archive_size[0] / 2))
                self.assertEqual(email_image.size[1], round(archive_size[1] / 2))
            self.assertAlmostEqual(archive_angle, 0, delta=0.2)
            self.assertLess(archive_size[0], 1000)
            self.assertLess(archive_size[1], 800)

    def test_missing_dpi_is_reported_without_creating_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "scan.png"
            archive = root / "archive"
            email = root / "email"
            create_scan(source)

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = postprocess.main(
                    [
                        "--archive-dir",
                        str(archive),
                        "--email-dir",
                        str(email),
                        str(source),
                    ]
                )

            self.assertEqual(result, 1)
            self.assertIn("missing DPI metadata", stderr.getvalue())
            self.assertEqual(list(archive.iterdir()), [])
            self.assertEqual(list(email.iterdir()), [])

    def test_detects_outer_document_instead_of_dark_center_fold(self) -> None:
        scanner_bed = np.zeros((800, 1000, 3), dtype=np.uint8)
        for row in range(scanner_bed.shape[0]):
            scanner_bed[row, :, :] = 185 + round(row / scanner_bed.shape[0] * 30)
        document = Image.new("RGB", (700, 500), "#eeeeee")
        draw = ImageDraw.Draw(document)
        draw.line((350, 0, 350, 499), fill="#333333", width=15)
        draw.line((80, 110, 300, 110), fill="#999999", width=4)
        draw.line((400, 160, 620, 160), fill="#999999", width=4)
        rotated = document.rotate(3, expand=True, fillcolor="#cccccc")
        scanner_bed_image = Image.fromarray(scanner_bed)
        scanner_bed_image.paste(
            rotated,
            (
                (scanner_bed_image.width - rotated.width) // 2,
                (scanner_bed_image.height - rotated.height) // 2,
            ),
        )

        rectangle, angle = postprocess.detect_document_rectangle(scanner_bed_image)
        edges = np.roll(rectangle, -1, axis=0) - rectangle
        lengths = sorted(np.linalg.norm(edges, axis=1))

        self.assertAlmostEqual(angle, -3, delta=0.3)
        self.assertAlmostEqual(lengths[-1], 700, delta=15)
        self.assertAlmostEqual(lengths[0], 500, delta=15)

    def test_rerun_reports_existing_archive_instead_of_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "scan.jpg"
            archive = root / "archive"
            email = root / "email"
            create_scan(source, dpi=600)
            arguments = [
                "--archive-dir",
                str(archive),
                "--email-dir",
                str(email),
                str(source),
            ]
            self.assertEqual(postprocess.main(arguments), 0)
            archived_bytes = (archive / "scan.jpg").read_bytes()

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = postprocess.main(arguments)

            self.assertEqual(result, 1)
            self.assertIn("archive: output already exists", stderr.getvalue())
            self.assertEqual((archive / "scan.jpg").read_bytes(), archived_bytes)


def create_scan(path: Path, dpi: int | None = None) -> None:
    scanner_bed = Image.new("RGB", (1000, 800), "#555555")
    document = Image.new("RGB", (680, 480), "white")
    draw = ImageDraw.Draw(document)
    draw.rectangle((0, 0, 679, 479), outline="#111111", width=5)
    draw.line((80, 100, 590, 100), fill="#777777", width=5)
    draw.line((80, 150, 540, 150), fill="#777777", width=5)
    rotated = document.rotate(4, expand=True, fillcolor="#555555")
    scanner_bed.paste(
        rotated,
        ((scanner_bed.width - rotated.width) // 2, (scanner_bed.height - rotated.height) // 2),
    )
    options = {"dpi": (dpi, dpi)} if dpi is not None else {}
    scanner_bed.save(path, **options)


if __name__ == "__main__":
    unittest.main()
