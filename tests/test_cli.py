import numpy as np
import pytest

from layerlens.cli import main


def test_cli_writes_diffusion_profile(tmp_path):
    out = tmp_path / "profile.csv"
    main(["solve", "--cells", "8", "--peclet", "0", "--output", str(out)])
    data = np.loadtxt(out, delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, 0], data[:, 1], atol=1e-14)


@pytest.mark.parametrize("args", [["--cells", "1"], ["--peclet", "nan"]])
def test_cli_rejects_bad_parameters(args):
    with pytest.raises(SystemExit) as exc:
        main(["solve", *args])
    assert exc.value.code == 2
