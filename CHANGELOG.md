## UNRELEASED - YYYY-MM-DD

### Added
* setup.py
* New methods `Comparison.html` and `Comparison.save_html` 
### Fixed
### Changed
* Breaking change: `Evaluator`'s field `figures` is now `lazy_figures: List[Callable[[], Figure]]`. Most notably,
this avoids issues when generating multiple HTML documents from the same `Evaluation` object.

## 0.1.0 - 2019-10-09

### Added
* Initial release.
