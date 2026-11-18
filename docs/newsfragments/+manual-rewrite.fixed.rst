The manual's Python examples run: they import ``Bounds`` and ``PyObjective`` from ``eindir._core`` instead of names that do not exist, and CI runs every Python block of the manual.
The GLE and surrogate tutorials are in Rust, which is where those types are, with the code included from ``examples/`` so ``cargo test`` builds it.
The Gradients page and the FAQ no longer promise a finite-difference fallback in Python, the FAQ cites the Zenodo record instead of a missing ``CITATION.cff``, and the explanation pages cite their sources with checked DOIs.
The README's compatibility contract lists the DLPack major version among the fields that must match and says that the consumer's minor versions must not exceed the library's.
