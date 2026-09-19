# Data sheet: `ReefCheck974.rdata`

| Property | Value |
|---|---|
| File | `ReefCheck974.rdata` (R workspace holding one data frame, `ReefCheck`) |
| SHA-256 | `24db10e06fe10a32133ab389374eebd1592628b1e45ad724c76bf3dc14fccd9c` |
| Rows × columns | 12,392 × 12 |
| Years | 1997 to 2017 (year only, no survey date) |
| Response | `Bleaching`: `Yes` in 361 rows (2.9%) |
| After the 2018 cleaning | 9,111 rows, 255 `Yes` (2.8%) |

The records come from Reef Check reef surveys. Reef Check teams and trained volunteers use one standard protocol worldwide. This file is the extract that the 2018 project used. Cite Reef Check when you use the data. Current data is available from Reef Check at <https://www.reefcheck.org>.

## Columns

| Column | Type | Levels in the raw file (blank shown as `""`) |
|---|---|---|
| `Bleaching` | factor | `No`, `Yes` |
| `Ocean` | factor | `Pacific`, `Atlantic`, `Indian`, `Red Sea`, `Arabian Gulf`, `East Pacific`, `""` |
| `Year` | integer | 1997 to 2017 |
| `Depth` | numeric, metres | 0.1 to 23 |
| `Storms` | factor | `yes`, `no`, `unknown`, `y`, `""` |
| `HumanImpact` | factor | `none`, `low`, `moderate`, `high`, `unknown`, `""` |
| `Siltation` | factor | `never`, `occasionally`, `Occasionally`, `often`, `always`, `""` |
| `Dynamite` | factor | `none`, `low`, `moderate`, `high`, `prior`, `""` |
| `Poison` | factor | `none`, `low`, `moderate`, `high`, `unknown`, `""` |
| `Sewage` | factor | `none`, `low`, `moderate`, `high`, `k`, `""` |
| `Industrial` | factor | `none`, `low`, `moderate`, `high`, `""` |
| `Commercial` | factor | `none`, `low`, `moderate`, `high`, `""` |

The cleaning rules are in `coralforest/data.py` and `R/reproduce_2018.R`. Both follow the 2018 notebook. Blank cells become `unknown` where the column already has an `unknown` level, and otherwise become the lowest level. Rows with any `unknown` are removed. The two typo levels (`y`, `Occasionally`) are merged, and the rows with `k` or `prior` are removed.

## Fields the file does not have

- A site identifier, so repeat surveys of one reef cannot be grouped.
- Coordinates: `Ocean` is the only location field.
- A survey date finer than the year.
- Sea-surface temperature or degree heating weeks.
