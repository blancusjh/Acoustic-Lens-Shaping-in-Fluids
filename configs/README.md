# Configuration archive

The dated JSON files are immutable inputs to preserved experiments. Similar
files intentionally retain their actual resolved values so older results can
be reproduced without reconstructing a chain of overrides. New campaigns
should keep shared machine/material/prescription choices explicit in their
input record and write the resolved SI configuration beside each output.

`relocations.json` maps old `artifacts/<campaign>/...` paths to current
`artifacts/studies/<study>/...` paths or to the separate research archive.
`acoustic_freeform.core.paths.data_path` reads that map for historical inputs.
New configurations should use current paths directly.

One historical follow-up,
`dual/precision/stigmatic-conic-coupled-448.json`, names a missing
`coupled-observations-448/observation-operator.npz`. Its upstream coupled
observation output must be produced and verified before that configuration can
run. An unrelated observation operator is not an interchangeable input.
