# Scan Utils

A loose collection of helper scripts for batch-processing scanned documents.

## Post-process scanned documents

`postprocess-scanned-document.py` deskews and crops scanned JPEG or PNG files.
It writes one full-resolution archive image and one downscaled 300 DPI JPEG for
each successfully processed source image. It does not create PDFs.

Run it through `uv`:

```bash
uv run python postprocess-scanned-document.py \
  --archive-dir output/aligned-original \
  --email-dir output/email-images \
  --email-jpeg-quality 40 \
  scans/
```

Directory arguments are expanded recursively in alphabetical order. Use
`--skip IMG_OR_DIR` one or more times to exclude source files or directory
trees during a rerun. Existing output files are reported as errors and are
never overwritten.

When the only input is one directory, `--archive-dir` and `--email-dir` default
to sibling directories named `aligned-original` and `email-images`.

If scanner DPI metadata is absent or unreliable, pass an explicit value with
`--force-source-dpi`.

## Place images in a PDF

`place-in-pdf.py` places JPEG and PNG files at the physical size specified by
their DPI metadata. It uses A4 portrait pages by default, keeps a margin around
each page, centers images horizontally, and stacks as many images vertically as
fit without scaling them.

- Allows multiple JPEG or PNG images to be placed per PDF page. (In contrast to
  `img2pdf`, which only allows 1 raster image per PDF page.)
- Does not re-compress the image data, so the resulting PDF file size is very
  close to the total file sizes of all input JPG or PNG files.

Run it through `uv`:

```bash
uv run python place-in-pdf.py \
  --output scanned-document.pdf \
  scans/
```

Directory arguments are expanded recursively in alphabetical order. Existing
output files are never overwritten. Use `--page-size letter` for Letter pages,
`--margin-mm` and `--gap-mm` to adjust the layout, or `--force-source-dpi` when
image metadata is missing or unreliable. Duplicate source files are rejected
unless `--allow-duplicates` is given.
