"""Legacy integration tests that require an external local dataset mirror."""

import pytest

pytestmark = pytest.mark.skip(reason="legacy manual integration tests require external local SAR data")
