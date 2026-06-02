# Multi-image PDFs

Allows multiple JPEG or PNG images to be placed per PDF page.

(In contrast, `img2pdf` only allows 1 raster image per PDF page.)

One main goal is that the resulting PDF file size is very close to the total file sizes of all
input JPG or PNG files.

Typically these images would be scanned documents.

## Place images in a PDF

`place-in-pdf.py` places JPEG and PNG files at the physical size specified by
their DPI metadata. It uses A4 portrait pages by default, keeps a margin around
each page, centers images horizontally, and stacks as many images vertically as
fit without scaling them.

Run it through `uv`:

```bash
uv run python place-in-pdf.py \
  --output scans.pdf \
  scans/
```

Directory arguments are expanded recursively in alphabetical order. Existing
output files are never overwritten. Use `--page-size letter` for Letter pages,
`--margin-mm` and `--gap-mm` to adjust the layout, or `--force-source-dpi` when
image metadata is missing or unreliable. Duplicate source files are rejected
unless `--allow-duplicates` is given.

## Post-process scanned documents

`postprocess-scanned-document.py` deskews and crops scanned JPEG or PNG files.
It writes one full-resolution archive image and one downscaled 300 DPI JPEG for
each successfully processed source image. It does not create PDFs.

Run it through `uv`:

```bash
uv run python postprocess-scanned-document.py \
  --archive-dir output/aligned-original \
  --email-dir output/email-images \
  --jpeg-quality 75 \
  scans/
```

Directory arguments are expanded recursively in alphabetical order. Use
`--skip IMG_OR_DIR` one or more times to exclude source files or directory
trees during a rerun. Existing output files are reported as errors and are
never overwritten.

If scanner DPI metadata is absent or unreliable, pass an explicit value:

```bash
uv run python postprocess-scanned-document.py \
  --archive-dir output/aligned-original \
  --email-dir output/email-images \
  --force-source-dpi 600 \
  scans/
```
