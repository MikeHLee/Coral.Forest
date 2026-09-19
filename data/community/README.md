# Community tables

What the reefs of a country are worth to the people near them. Three published
tables, each read from the paper that printed it, with the extraction and the
row checks kept beside them in `source_tables/`.

| File | Column | Unit | Rows | Source |
|---|---|---|---|---|
| `reef_fishers.csv` | `fishery` | people who fish reefs, 2010 | 98 | Teh, Teh & Sumaila 2013, Table S3 |
| `reef_tourism.csv` | `tourism` | US dollars a year, 2013 dollars | 80 | Spalding et al. 2017, Table A1 |
| `reef_flood_protection.csv` | `shoreline` | people whose annual flood exposure reefs lower | 30 | Beck et al. 2018, Table 1 |
| `reef_flood_protection.csv` | `annual_averted_damages_musd` | millions of 2011 US dollars a year | 15 | Beck et al. 2018, Table 1 |

Every source is open access under CC BY 4.0. `source_tables/*.meta.json` names
the place each licence is stated, the exact table and caption, the units of
every column, and the download URL. `source_tables/*.verification.txt` quotes
the source text next to eight parsed values per table.

## Citations

- Teh, L.S.L., Teh, L.C.L. and Sumaila, U.R. (2013). A global estimate of the
  number of coral reef fishers. PLoS ONE 8(6): e65397.
  doi:10.1371/journal.pone.0065397
- Spalding, M., Burke, L., Wood, S.A., Ashpole, J., Hutchison, J. and
  zu Ermgassen, P. (2017). Mapping the global value and distribution of coral
  reef tourism. Marine Policy 82: 104-113. doi:10.1016/j.marpol.2017.05.014
- Beck, M.W., Losada, I.J., Menendez, P., Reguero, B.G., Diaz-Simal, P. and
  Fernandez, F. (2018). The global flood protection savings provided by coral
  reefs. Nature Communications 9: 2186. doi:10.1038/s41467-018-04568-z

## What these tables do not cover

- **Beck 2018 prints rankings, not a full table.** Only the top 15 countries by
  averted damages and the top 30 by people protected are published, so 30
  countries carry a `shoreline` value and 15 carry an averted-damages value.
  The 30 published values sum to 197,797 people; the paper states more than
  200,000 worldwide. The country table covers the 1-in-100-year event; the
  other return periods appear only as global figures.
- **Teh 2013 is one country short.** The 98 published rows sum to 6,122,041
  fishers against a printed total of 6,145,200. Two independent conversions of
  the source file return the same 98 rows, and the regional sums place the
  23,159-fisher gap in the Eastern Pacific and Atlantic region. No row was
  invented to close it.
- **The United States does not join across the three.** Beck prints one row for
  the country; Spalding prints Florida and Hawaii separately; Teh prints
  "Florida & US Gulf of Mexico" and "Hawaii" separately. They are left as the
  sources printed them.
- **Dollar years differ.** Beck is in 2011 dollars, Spalding in 2013 dollars
  (except World Travel and Tourism Council figures, which keep their original
  year). Nothing here is deflated to a common year.
- One Beck column, `annual_averted_damages_over_gdp`, has no unit in the
  source. It is kept in `source_tables/` and is not used by the atlas.
