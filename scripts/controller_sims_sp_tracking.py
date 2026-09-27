import numpy as np

from pid_tuneopt.controls import sims


def _discrete_plant(A, B, thetap=0.0):
    """Model dict accepted by ``set_point_tracking`` / ``disturbance_rejection``."""
    Kp = np.sum(np.array(B)) / (1 + np.sum(-1*np.array(A)))
    return {
        "A": np.asarray(A, dtype=float),
        "B": np.asarray(B, dtype=float),
        "thetap": float(thetap),
        "Kp": float(Kp),
    }


if __name__ == "__main__":
    Ts = 1
    model = _discrete_plant([0.5], [0.5])
    u1, y1, status1 = sims.set_point_tracking(
        model, 0.3, 1.0, 0.0, Ts=Ts, endtime=50
    )
    t1 = np.arange(len(y1)) * Ts

    Ts = 1
    model = _discrete_plant([0.5], [0.5], thetap=2)
    u2, y2, status2 = sims.set_point_tracking(
        model, 0.3, 1.0, 0.0, Ts=Ts, endtime=50
    )
    t2 = np.arange(len(y2)) * Ts

    Ts = 2
    model = _discrete_plant([0.5], [0.5], thetap=3)
    u3, y3, status3 = sims.set_point_tracking(
        model, 0.3, 1.0, 0.0, Ts=Ts, endtime=50
    )
    t3 = np.arange(len(y3)) * Ts
