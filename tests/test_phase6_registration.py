from src.phase6_registration import (
    AMENDED_DESIGN_SHA256, ORIGINAL_DESIGN_SHA256, registered_source_matches,
)


def test_only_exact_authorized_deviation_is_compatible():
    assert registered_source_matches('DESIGN.md', ORIGINAL_DESIGN_SHA256, ORIGINAL_DESIGN_SHA256)
    assert registered_source_matches('DESIGN.md', AMENDED_DESIGN_SHA256, ORIGINAL_DESIGN_SHA256)
    assert not registered_source_matches('DESIGN.md', '0' * 64, ORIGINAL_DESIGN_SHA256)
    assert not registered_source_matches('prompts/templates_4.json', AMENDED_DESIGN_SHA256, ORIGINAL_DESIGN_SHA256)
    assert not registered_source_matches('DESIGN.md', AMENDED_DESIGN_SHA256, '0' * 64)
