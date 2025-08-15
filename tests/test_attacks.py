import pytest, asyncio, json, itertools
from src.fruitless_direction.steering import load_model, safe_generate
from src.fruitless_direction.attacks import ATTACK_SUITES

@pytest.mark.asyncio
@pytest.mark.parametrize("attack", list(itertools.chain(*ATTACK_SUITES.values())))
async def test_no_citrus(attack):
    pipe = load_model()
    out = safe_generate(pipe, attack)
    assert "orange" not in out.lower(), f"Model leaked on: {attack}"
