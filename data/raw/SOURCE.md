# Dataset Source

This reproduction uses the 8 WAV files published alongside Siddagangaiah et
al. (2020) on Mendeley Data:

- DOI: 10.17632/zv6ygrptr4.1
- URL: https://data.mendeley.com/datasets/zv6ygrptr4/1

Files (download and place directly in this directory, `data/raw/`):

- Whistle1.wav
- Whistle2.wav
- Whistle3.wav
- Whistle4.wav
- Click1.wav
- Click2.wav
- Click3.wav
- Click4.wav

**These files are NOT bundled with this repository** (binary audio, and
licensing/distribution of the third-party dataset is the depositors'
prerogative, not ours). Download them from the DOI above, then verify
integrity with:

```bash
sha256sum -c CHECKSUMS.sha256
```

(Populate `CHECKSUMS.sha256` after your own download, using
`sha256sum *.wav > CHECKSUMS.sha256` — a template is provided as
`CHECKSUMS.sha256.template` since we do not have local copies of the
authors' original files to hash in this environment.)

Verified properties of these 8 files (from direct inspection reported in
the build spec — treat as ground truth, do not re-derive differently) are
tabulated in `docs/paper_parameters.md`.

## Long-duration data (NOT available)

The 4-hour ETS and 4-hour HB recordings used for Table 1's confusion
matrices, and their manual annotation vectors, are **not** part of this
public dataset. They are available only from the corresponding authors on
request — see `docs/author_correspondence.md`.
