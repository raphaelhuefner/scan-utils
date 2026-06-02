# Multi-image PDFs

Allows multiple JPEG or PNG images to be placed per PDF page.

(In contrast, `img2pdf` only allows 1 raster image per PDF page.)

One main goal is that the resulting PDF file size is very close to the total file sizes of all
input JPG or PNG files.

Typically these images would be scanned documents.

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
