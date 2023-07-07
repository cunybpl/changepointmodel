from changepointmodel.core.calc import tstat
import numpy as np


def test_tstat_bad(dummy_t_test_bad):
    data = dummy_t_test_bad

    twop_data = data[0]
    threepc_data = data[1]
    threeph_data = data[2]
    fourp_data = data[3]
    fivep_data = data[4]

    # 2P
    result = tstat.twop(
        X=np.array(twop_data.result.input_data.X),
        y=np.array(twop_data.result.input_data.y),
        pred_y=np.array(twop_data.result.pred_y),
        slope=twop_data.result.coeffs.slopes[0],
    )
    assert result == (None, np.inf)

    # 3PC
    result = tstat.threepc(
        X=np.array(threepc_data.result.input_data.X),
        y=np.array(threepc_data.result.input_data.y),
        pred_y=np.array(threepc_data.result.pred_y),
        slope=threepc_data.result.coeffs.slopes[0],
        changepoint=threepc_data.result.coeffs.changepoints[0],
    )
    assert result == (None, 1.1224972160321822)

    # 3PH
    result = tstat.threeph(
        X=np.array(threeph_data.result.input_data.X),
        y=np.array(threeph_data.result.input_data.y),
        pred_y=np.array(threeph_data.result.pred_y),
        slope=threeph_data.result.coeffs.slopes[0],
        changepoint=threeph_data.result.coeffs.changepoints[0],
    )
    assert result == (-0.4898979485566357, None)

    # 4P
    result = tstat.fourp(
        X=np.array(fourp_data.result.input_data.X),
        y=np.array(fourp_data.result.input_data.y),
        pred_y=np.array(fourp_data.result.pred_y),
        ls=fourp_data.result.coeffs.slopes[0],
        rs=fourp_data.result.coeffs.slopes[1],
        changepoint=fourp_data.result.coeffs.changepoints[0],
    )
    assert result == (-0.7348469228349536, 1.1224972160321822)

    # 5P
    result = tstat.fivep(
        X=np.array(fivep_data.result.input_data.X),
        y=np.array(fivep_data.result.input_data.y),
        pred_y=np.array(fivep_data.result.pred_y),
        ls=fivep_data.result.coeffs.slopes[0],
        rs=fivep_data.result.coeffs.slopes[1],
        lcp=fivep_data.result.coeffs.changepoints[0],
        rcp=fivep_data.result.coeffs.changepoints[1],
    )
    assert result == (-0.24494897427831785, 0.31622776601683794)


def test_tstat_good(dummy_t_test_good):
    data = dummy_t_test_good

    twop_data = data[0]
    threepc_data = data[1]
    threeph_data = data[2]
    fourp_data = data[3]
    fivep_data = data[4]

    # 2P
    result = tstat.twop(
        X=np.array(twop_data.result.input_data.X),
        y=np.array(twop_data.result.input_data.y),
        pred_y=np.array(twop_data.result.pred_y),
        slope=twop_data.result.coeffs.slopes[0],
    )
    assert result == (None, np.inf)

    # 3PC
    result = tstat.threepc(
        X=np.array(threepc_data.result.input_data.X),
        y=np.array(threepc_data.result.input_data.y),
        pred_y=np.array(threepc_data.result.pred_y),
        slope=threepc_data.result.coeffs.slopes[0],
        changepoint=threepc_data.result.coeffs.changepoints[0],
    )
    assert result == (None, np.inf)

    # 3PH
    result = tstat.threeph(
        X=np.array(threeph_data.result.input_data.X),
        y=np.array(threeph_data.result.input_data.y),
        pred_y=np.array(threeph_data.result.pred_y),
        slope=threeph_data.result.coeffs.slopes[0],
        changepoint=threeph_data.result.coeffs.changepoints[0],
    )
    assert result == (-np.inf, None)

    # 4P
    result = tstat.fourp(
        X=np.array(fourp_data.result.input_data.X),
        y=np.array(fourp_data.result.input_data.y),
        pred_y=np.array(fourp_data.result.pred_y),
        ls=fourp_data.result.coeffs.slopes[0],
        rs=fourp_data.result.coeffs.slopes[1],
        changepoint=fourp_data.result.coeffs.changepoints[0],
    )
    assert result == (-np.inf, np.inf)

    # 5P
    result = tstat.fivep(
        X=np.array(fivep_data.result.input_data.X),
        y=np.array(fivep_data.result.input_data.y),
        pred_y=np.array(fivep_data.result.pred_y),
        ls=fivep_data.result.coeffs.slopes[0],
        rs=fivep_data.result.coeffs.slopes[1],
        lcp=fivep_data.result.coeffs.changepoints[0],
        rcp=fivep_data.result.coeffs.changepoints[1],
    )
    assert result == (-np.inf, np.inf)
