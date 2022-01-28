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
from linearmodels.typing import Float64Array, Int64Array


class TestData(NamedTuple):
    n: int
    t: int
    k: int
    x: Float64Array
    epsilon: Float64Array
    params: Float64Array
    df_resid: int
    y: Float64Array
    time_ids: Int64Array
    cluster1: Int64Array
    entity_ids: Int64Array
    cluster2: Int64Array
    cluster3: Int64Array
    cluster4: Int64Array
    cluster5: Int64Array


@pytest.fixture
def test_data() -> TestData:
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
    return TestData(
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
def test_heteroskedastic_smoke(test_data) -> None:
    cov = HeteroskedasticCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)
    cov = HeteroskedasticCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)


@pytest.mark.smoke
def test_homoskedastic_smoke(test_data) -> None:
    cov = HomoskedasticCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)
    cov = HomoskedasticCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)


@pytest.mark.smoke
def test_clustered_covariance_smoke(test_data) -> None:
    cov = ClusteredCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)

    cov = ClusteredCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
        clusters=test_data.cluster1,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)

    cov = ClusteredCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
        clusters=test_data.cluster2,
        group_debias=True,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)

    cov = ClusteredCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
        clusters=test_data.cluster3,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)
    cov = ClusteredCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
        clusters=test_data.cluster3,
        group_debias=True,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)

    cov = ClusteredCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
        clusters=test_data.cluster4,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)

    cov = ClusteredCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        extra_df=0,
        clusters=test_data.cluster4,
        group_debias=True,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)


def test_clustered_covariance_error(test_data) -> None:
    with pytest.raises(ValueError):
        ClusteredCovariance(
            test_data.y,
            test_data.x,
            test_data.params,
            test_data.entity_ids,
            test_data.time_ids,
            extra_df=0,
            clusters=test_data.cluster5,
        )

    with pytest.raises(ValueError):
        ClusteredCovariance(
            test_data.y,
            test_data.x,
            test_data.params,
            test_data.entity_ids,
            test_data.time_ids,
            extra_df=0,
            clusters=test_data.cluster4[::2],
        )


@pytest.mark.smoke
def test_driscoll_kraay_smoke(test_data) -> None:
    cov = DriscollKraay(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)
    cov = DriscollKraay(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        kernel="parzen",
    ).cov
    assert cov.shape == (test_data.k, test_data.k)
    cov = DriscollKraay(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        bandwidth=12,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)


@pytest.mark.smoke
def test_ac_covariance_smoke(test_data) -> None:
    cov = ACCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)
    cov = ACCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        kernel="parzen",
    ).cov
    assert cov.shape == (test_data.k, test_data.k)
    cov = ACCovariance(
        test_data.y,
        test_data.x,
        test_data.params,
        test_data.entity_ids,
        test_data.time_ids,
        bandwidth=12,
    ).cov
    assert cov.shape == (test_data.k, test_data.k)


def test_covariance_manager() -> None:
    cm = CovarianceManager(
        "made-up-class", HomoskedasticCovariance, HeteroskedasticCovariance
    )
    with pytest.raises(ValueError):
        cm["clustered"]

    with pytest.raises(KeyError):
        cm["unknown"]

    assert cm["unadjusted"] is HomoskedasticCovariance
    assert cm["homoskedastic"] is HomoskedasticCovariance
    assert cm["robust"] is HeteroskedasticCovariance
    assert cm["heteroskedastic"] is HeteroskedasticCovariance
