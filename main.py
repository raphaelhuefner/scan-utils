from pathlib import Path
from PIL import Image


def read_image_geometry(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        width_px, height_px = image.size
        dpi = image.info.get("dpi")

    if not dpi:
        raise ValueError(f"{path}: missing physical resolution metadata")

    dpi_x, dpi_y = map(float, dpi)
    if dpi_x <= 0 or dpi_y <= 0:
        raise ValueError(f"{path}: invalid DPI metadata: {dpi!r}")

    return {
        "width_px": width_px,
        "height_px": height_px,
        "dpi_x": dpi_x,
        "dpi_y": dpi_y,
        "width_pt": width_px / dpi_x * 72,
        "height_pt": height_px / dpi_y * 72,
    }


def main():
    print("Hello from pdf-with-multiple-img!")


if __name__ == "__main__":
    main()
