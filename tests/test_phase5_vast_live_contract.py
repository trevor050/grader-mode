"""Provider response shapes observed before the authorized Phase 5 rental."""
import pytest
from src.phase5_vast_main import public_account, offer_costs


def test_vast_signed_account_total_is_not_a_nonnegative_balance():
    result=public_account(dict(credit=2.8247990723,total_spend=-7.1752009277,email='private'))
    assert result==dict(credit=2.8247990723,total_spend=-7.1752009277)
    with pytest.raises(ValueError):
        public_account(dict(credit=-1,total_spend=-7))


def test_vast_offer_verification_string_and_eighty_gb_rate():
    offer=dict(gpu_name='RTX 5090',num_gpus=1,gpu_ram=32607,reliability=.9907895,
               verification='verified',rentable=True,dph_base=.6,storage_cost=.2,
               inet_down_cost=.0026041666666666665,inet_up_cost=.00390625)
    costs=offer_costs(offer)
    assert costs['hourly_usd']==pytest.approx(.6+80*.2/720)
    with pytest.raises(Exception):
        offer_costs(offer|{'verification':'unverified'})


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),True])
def test_total_spend_is_signed_but_finite_and_not_boolean(value):
    with pytest.raises(ValueError):public_account(dict(credit=2.8,total_spend=value))


def test_fresh_scan_keeps_real_verification_field_and_drops_private_fields(tmp_path,monkeypatch):
    from src.phase5_vast_main import Provider
    backend=Provider(tmp_path);calls=[]
    actual=dict(id=123,gpu_name='RTX 5090',num_gpus=1,gpu_ram=32607,reliability=.9907895,
                verification='verified',rentable=True,dph_base=.6,storage_cost=.2,
                inet_down_cost=.0026041666666666665,inet_up_cost=.00390625,
                private_note='exclude',cuda_max_good=13.0)
    monkeypatch.setattr(backend,'cli',lambda *args:calls.append(args) or [actual])
    scanned=backend.scan(123)
    assert scanned['verification']=='verified' and 'private_note' not in scanned
    assert offer_costs(scanned)['hourly_usd']==pytest.approx(.6+80*.2/720)
    assert calls[0][0:2]==('search','offers') and '--storage' in calls[0]
    for patch in ({'verification':'unverified'},{'verification':None},
                  {'verification':'verified','verified':False}):
        with pytest.raises(Exception):offer_costs(scanned|patch)


def test_read_only_cli_retries_empty_success_without_retrying_create(tmp_path, monkeypatch):
    from src.phase5_vast_main import Provider
    backend=Provider(tmp_path); replies=iter(['', '[]']); calls=[]
    monkeypatch.setattr(backend,'run',lambda argv: calls.append(argv) or next(replies))
    monkeypatch.setattr('src.phase5_vast_main.time.sleep',lambda _:None)
    assert backend.cli('show','instances') == []
    assert len(calls)==2
    calls.clear()
    monkeypatch.setattr(backend,'run',lambda argv: calls.append(argv) or '')
    with pytest.raises(Exception):backend.cli('create','instance',123)
    assert len(calls)==1


def test_destroy_is_noninteractive_and_accepts_cli_empty_ack(tmp_path,monkeypatch):
    from src.phase5_vast_main import Provider
    backend=Provider(tmp_path,label='owned'); calls=[]
    monkeypatch.setattr(backend,'instances',lambda:[dict(id=123,label='owned')])
    monkeypatch.setattr(backend,'run',lambda argv:calls.append(argv) or '')
    backend.destroy_owned()
    assert calls==[['vastai','destroy','instance','123','--yes','--raw']]
