# Outputs

Benchmark outputs are written under `outputs/`.

New run directories should be versioned and non-overwriting, for example:

```text
outputs/2026-07-07_ByteRCNN_fft75_512/
outputs/2026-07-08_ByteNet_fft75_512/
```

The output-versioning helper also maintains:

- `outputs/latest`
- `outputs/best`
- `outputs/history.json`

Publication assets can be exported to `paper_assets/` with subdirectories for
figures, tables, CSV files, and JSON files.
