# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0](https://github.com/BYUCamachoLab/SiPANN/releases/tag/v2.0.0) - <small>2022-07-06</small>

Maintenance update and drop support for Python 3.6.

### Added
- None

### Changed
- Bug fix for `SimphonyWrapper` class in `scee_int.py`. Changed tuple of `str` to `PinList`. [Check this PR for more details.](https://github.com/BYUCamachoLab/SiPANN/pull/24)
- Bug fix in `nn.py` [See here for more info.](https://github.com/BYUCamachoLab/SiPANN/pull/29)
- Code cleanup [See here for more info.](https://github.com/BYUCamachoLab/SiPANN/pull/27)

### Removed
- Support for Python 3.6


---

## [1.4.0](https://github.com/BYUCamachoLab/SiPANN/releases/tag/v1.4.0) - <small>2022-06-13</small>

Maintenance update.

### Added
- None

### Changed
- Correct "invalid ports" errors (fixed some conditional statements).

### Removed
- None

---

## [1.3.2](https://github.com/BYUCamachoLab/SiPANN/releases/tag/v1.3.2) - <small>2022-05-16</small>

Maintenance update.

### Added
- None

### Changed
- Code cleanup (changed some syntax to be more Pythonic).
- Modified some code to work with the latest version of 
  [Simphony](https://github.com/BYUCamachoLab/simphony).

### Removed
- None

---

## [1.3.1](https://github.com/BYUCamachoLab/SiPANN/releases/tag/v1.3.1) - <small></small>

This version is mostly a maintenance update. We have added

* Continuous Integration via github actions
* precommit hooks to keep code maintained
* Small bug fix in `SiPANN.scee.Waveguide`. It should now work properly with `SiPANN.scee_int.SimphonyWrapper`.
