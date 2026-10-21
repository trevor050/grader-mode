"""Accept only the original registration or its explicitly authorized deviation.

Raw generation provenance retains the full DESIGN hash at registration 8af0156.
After rebasing, the document additionally contains the lead's 8219501 protocol
deviation. No other document or prompt drift is accepted by this compatibility
check; the deviation changes scoring, not generation.
"""

ORIGINAL_DESIGN_SHA256 = 'f1d2d05185c1593deba90a7b58601691d18b99ba047e02866d06abca10c90c84'
AMENDED_DESIGN_SHA256 = '8b1339e103c787704080a129019b1ba181775ed81edd968f91662d9cbfdcfaa5'
DEVIATION_COMMIT = '8219501182ce380c45115efd256938a2e372f088'


def registered_source_matches(filename, actual, expected):
    if actual == expected:
        return True
    return (filename == 'DESIGN.md' and expected == ORIGINAL_DESIGN_SHA256
            and actual == AMENDED_DESIGN_SHA256)
