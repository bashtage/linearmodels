from typing import NamedTuple

import numpy as np
import pytest

from linearmodels.panel.covariance import (
    ACCovariance,
    ClusteredCovariance,
    CovarianceManager,
    DriscollKraay,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
)
import linearmodels.typing.data


class PanelData(NamedTuple):
    n: int
    t: int
    k: int
    x: linearmodels.typing.data.Float64Array
    epsilon: linearmodels.typing.data.Float64Array
    params: linearmodels.typing.data.Float64Array
    df_resid: int
    y: linearmodels.typing.data.Float64Array
    time_ids: linearmodels.typing.data.Int64Array
    cluster1: linearmodels.typing.data.Int64Array
    entity_ids: linearmodels.typing.data.Int64Array
    cluster2: linearmodels.typing.data.Int64Array
    cluster3: linearmodels.typing.data.Int64Array
    cluster4: linearmodels.typing.data.Int64Array
    cluster5: linearmodels.typing.data.Int64Array


@pytest.fixture
def panel_data() -> PanelData:
    n, t, k = 100, 50, 10
    x = np.random.random_sample((n * t, k))
    epsilon = np.random.random_sample((n * t, 1))
    params = np.arange(1, k + 1)[:, None] / k
    df_resid = n * t
    y = x @ params + epsilon
    time_ids = np.tile(np.arange(t, dtype=np.int64)[:, None], (n, 1))
    cluster1 = np.tile(np.arange(n, dtype=np.int64)[:, None], (t, 1))
    entity_ids = cluster1
    cluster2 = np.tile(np.arange(t)[:, None], (n, 1))
    cluster3 = np.random.randint(0, 10, (n * t, 1), dtype=np.int64)
    cluster4 = np.random.randint(0, 10, (n * t, 2), dtype=np.int64)
    cluster5 = np.random.randint(0, 10, (n * t, 3), dtype=np.int64)
    return PanelData(
        n=n,
        t=t,
        k=k,
        x=x,
        epsilon=epsilon,
        params=params,
        df_resid=df_resid,
        y=y,
        time_ids=time_ids,
        entity_ids=entity_ids,
        cluster1=cluster1,
        cluster2=cluster2,
        cluster3=cluster3,
        cluster4=cluster4,
        cluster5=cluster5,
    )


@pytest.mark.smoke
def test_heteroskedastic_smoke(panel_data) -> None:
    cov = HeteroskedasticCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)
    cov = HeteroskedasticCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)


@pytest.mark.smoke
def test_homoskedastic_smoke(panel_data) -> None:
    cov = HomoskedasticCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)
    cov = HomoskedasticCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)


@pytest.mark.smoke
def test_clustered_covariance_smoke(panel_data) -> None:
    cov = ClusteredCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)

    cov = ClusteredCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
        clusters=panel_data.cluster1,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)

    cov = ClusteredCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
        clusters=panel_data.cluster2,
        group_debias=True,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)

    cov = ClusteredCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
        clusters=panel_data.cluster3,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)
    cov = ClusteredCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
        clusters=panel_data.cluster3,
        group_debias=True,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)

    cov = ClusteredCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
        clusters=panel_data.cluster4,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)

    cov = ClusteredCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        extra_df=0,
        clusters=panel_data.cluster4,
        group_debias=True,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)


def test_clustered_covariance_error(panel_data) -> None:
    with pytest.raises(ValueError):
        ClusteredCovariance(
            panel_data.y,
            panel_data.x,
            panel_data.params,
            panel_data.entity_ids,
            panel_data.time_ids,
            extra_df=0,
            clusters=panel_data.cluster5,
        )

    with pytest.raises(ValueError):
        ClusteredCovariance(
            panel_data.y,
            panel_data.x,
            panel_data.params,
            panel_data.entity_ids,
            panel_data.time_ids,
            extra_df=0,
            clusters=panel_data.cluster4[::2],
        )


@pytest.mark.smoke
def test_driscoll_kraay_smoke(panel_data) -> None:
    cov = DriscollKraay(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)
    cov = DriscollKraay(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        kernel="parzen",
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)
    cov = DriscollKraay(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        bandwidth=12,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)


@pytest.mark.smoke
def test_ac_covariance_smoke(panel_data) -> None:
    cov = ACCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)
    cov = ACCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        kernel="parzen",
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)
    cov = ACCovariance(
        panel_data.y,
        panel_data.x,
        panel_data.params,
        panel_data.entity_ids,
        panel_data.time_ids,
        bandwidth=12,
    ).cov
    assert cov.shape == (panel_data.k, panel_data.k)


def test_covariance_manager() -> None:
    cm = CovarianceManager(
        "made-up-class", HomoskedasticCovariance, HeteroskedasticCovariance
    )
    with pytest.raises(ValueError):
        assert cm["clustered"] is not None

    with pytest.raises(KeyError):
        assert cm["unknown"] is not None

    assert cm["unadjusted"] is HomoskedasticCovariance
    assert cm["homoskedastic"] is HomoskedasticCovariance
    assert cm["robust"] is HeteroskedasticCovariance
    assert cm["heteroskedastic"] is HeteroskedasticCovariance
