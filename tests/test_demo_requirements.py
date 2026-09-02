from pathlib import Path
R=(Path(__file__).resolve().parents[1]/'requirements-demo.txt').read_text().lower()
ACTIVE=[x.strip() for x in R.splitlines() if x.strip() and not x.lstrip().startswith('#')]
def test_demo_dependency_scope():
    assert 'qdrant-client==1.19.0' in ACTIVE
    assert not any(x.startswith(('torch','triton','nvidia-','cuda-')) for x in ACTIVE)
